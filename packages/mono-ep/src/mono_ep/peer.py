# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Experimental peer-memory chunk production with system-scope publication."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_publish(SRC, DST, QUEUE, TASK: tl.constexpr, N: tl.constexpr):
    lanes = tl.arange(0, 128)
    for start in range(0, N, 128):
        offsets = start + lanes
        values = tl.load(SRC + offsets, offsets < N, other=0)
        tl.store(DST + offsets, values, offsets < N)
    # Every writer fences its stores before lane zero publishes for the whole CTA.
    tl.inline_asm_elementwise(
        "fence.acq_rel.sys; mov.u32 $0, $1;",
        constraints="=r,r",
        args=[lanes],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()
    tl.atomic_xchg(QUEUE + TASK * 4 + 3, 1, sem="release", scope="sys")


def _check(source, destination, queue, task):
    if source.dtype != torch.bfloat16 or destination.dtype != source.dtype:
        raise ValueError("Peer chunk producer currently requires BF16")
    if source.shape != destination.shape:
        raise ValueError("Peer copy shapes must match")
    if queue.dtype != torch.int32 or queue.ndim != 2 or queue.shape[1] != 4:
        raise ValueError("Queue must be int32[num_tasks, 4]")
    if task < 0 or task >= queue.shape[0]:
        raise ValueError("Invalid queue task")
    if any(
        t.device.type != "cuda" or not t.is_contiguous()
        for t in (source, destination, queue)
    ):
        raise ValueError("Peer tensors must be contiguous CUDA views")
    if any(t.device != source.device for t in (destination, queue)):
        raise ValueError("Peer mappings must belong to the current source device")


@torch.library.custom_op(
    "mono_ep::peer_copy_publish", mutates_args=("destination", "queue")
)
def peer_copy_publish(
    source: torch.Tensor, destination: torch.Tensor, queue: torch.Tensor, task: int
) -> None:
    """Copy one contiguous chunk into peer memory, then publish its ready flag.

    destination and queue must be peer mappings established before execution.
    Queue metadata must already be initialized and immutable. The consumer
    must finish before either the input or ready flag is reused. The caller
    owns stream ordering and must reserve capacity for this producer CTA.
    This single-CTA reference producer is not bandwidth tuned.
    """
    _check(source, destination, queue, task)
    _copy_publish[(1,)](source, destination, queue, task, source.numel(), num_warps=4)


@peer_copy_publish.register_fake
def _peer_copy_publish_fake(source, destination, queue, task):
    _check(source, destination, queue, task)


@triton.jit
def _copy_fp8_publish(
    SRC, DST, SF, DF, QUEUE, TASK: tl.constexpr, N: tl.constexpr, S: tl.constexpr
):
    lanes = tl.arange(0, 128)
    for start in range(0, N, 128):
        offsets = start + lanes
        tl.store(
            DST + offsets, tl.load(SRC + offsets, offsets < N, other=0), offsets < N
        )
    for start in range(0, S, 128):
        offsets = start + lanes
        tl.store(DF + offsets, tl.load(SF + offsets, offsets < S, other=0), offsets < S)
    tl.inline_asm_elementwise(
        "fence.acq_rel.sys; mov.u32 $0, $1;",
        constraints="=r,r",
        args=[lanes],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()
    tl.atomic_xchg(QUEUE + TASK * 4 + 3, 1, sem="release", scope="sys")


def _check_fp8(source, destination, source_scales, destination_scales, queue, task):
    if source.dtype != torch.float8_e4m3fn or destination.dtype != source.dtype:
        raise ValueError("FP8 peer values must use E4M3")
    if source_scales.dtype != torch.int32 or destination_scales.dtype != torch.int32:
        raise ValueError("FP8 peer scales must use packed INT32 UE8M0 storage")
    if (
        source.shape != destination.shape
        or source_scales.shape != destination_scales.shape
    ):
        raise ValueError("Peer value and scale shapes must match their sources")
    if queue.dtype != torch.int32 or queue.ndim != 2 or queue.shape[1] != 4:
        raise ValueError("Queue must be int32[num_tasks, 4]")
    if task < 0 or task >= queue.shape[0]:
        raise ValueError("Invalid queue task")
    if source.device.type != "cuda" or any(
        t.device != source.device or not t.is_contiguous()
        for t in (source, destination, source_scales, destination_scales, queue)
    ):
        raise ValueError(
            "Peer value and scale storage must be contiguous CUDA mappings"
        )


@torch.library.custom_op(
    "mono_ep::peer_copy_fp8_publish",
    mutates_args=("destination", "destination_scales", "queue"),
)
def peer_copy_fp8_publish(
    source: torch.Tensor,
    destination: torch.Tensor,
    source_scales: torch.Tensor,
    destination_scales: torch.Tensor,
    queue: torch.Tensor,
    task: int,
) -> None:
    """Copy E4M3 values and packed scales in one CTA before publishing readiness.

    Scale arguments are contiguous physical storage, e.g. [packed_K, M],
    whose transpose supplies the consumer's MN-major scale view. Mapping,
    lifetime and SM reservation requirements match peer_copy_publish.
    """
    _check_fp8(source, destination, source_scales, destination_scales, queue, task)
    _copy_fp8_publish[(1,)](
        source.view(torch.uint8),
        destination.view(torch.uint8),
        source_scales,
        destination_scales,
        queue,
        task,
        source.numel(),
        source_scales.numel(),
        num_warps=4,
    )


@peer_copy_fp8_publish.register_fake
def _peer_copy_fp8_publish_fake(
    source, destination, source_scales, destination_scales, queue, task
):
    _check_fp8(source, destination, source_scales, destination_scales, queue, task)


_PACKED_FENCE_ASM = tl.constexpr(
    "fence.acq_rel.sys; " + " ".join(f"mov.u32 ${i}, ${i + 16};" for i in range(16))
)
_PACKED_FENCE_CONSTRAINTS = tl.constexpr(",".join(["=r"] * 16 + ["r"] * 16))


@triton.jit
def _quantize_publish(
    X,
    Q,
    SF,
    QUEUE,
    COUNTER,
    TASK: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    GRAN: tl.constexpr,
    INDICES=None,
    ROUTED: tl.constexpr = False,
    SOURCE_ROWS: tl.constexpr = 0,
    PUBLISH_TASKS: tl.constexpr = 1,
    METADATA: tl.constexpr = False,
    SLOTS=None,
    IDS=None,
    STARTS=None,
    SIZES=None,
    ENDS=None,
    SCORES=None,
    OUT_IDS=None,
    OUT_TOKENS=None,
    OUT_SLOTS=None,
    OUT_ENDS=None,
    OUT_SCORES=None,
    TOPK: tl.constexpr = 1,
    ROW_TILE: tl.constexpr = 16,
    FENCE_PACK: tl.constexpr = 16,
    SKIP_PADDING: tl.constexpr = False,
    BF16: tl.constexpr = False,
):
    row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    group = tl.program_id(1)
    cols = group * (4 * GRAN) + tl.arange(0, 4 * GRAN)
    source_row = row
    if METADATA:
        slot = tl.load(SLOTS + row, row < ROWS, -1)
        source_row = tl.where(slot >= 0, slot // TOPK, -1)
        if group == 0:
            tl.store(OUT_IDS + row, tl.load(IDS + row, row < ROWS, -1), row < ROWS)
            tl.store(OUT_TOKENS + row, source_row, row < ROWS)
            tl.store(OUT_SLOTS + row, slot, row < ROWS)
            score = tl.load(SCORES + slot, (row < ROWS) & (slot >= 0), 0)
            tl.store(OUT_SCORES + row, score, row < ROWS)
            expert_mask = row < PUBLISH_TASKS
            tl.store(OUT_ENDS + row, tl.load(ENDS + row, expert_mask, 0), expert_mask)
            tl.store(QUEUE + row * 4, row, expert_mask)
            tl.store(
                QUEUE + row * 4 + 1, tl.load(STARTS + row, expert_mask, 0), expert_mask
            )
            tl.store(
                QUEUE + row * 4 + 2, tl.load(SIZES + row, expert_mask, 0), expert_mask
            )
            tl.inline_asm_elementwise(
                "fence.acq_rel.sys; mov.u32 $0, $1;",
                constraints="=r,r",
                args=[row],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
    elif ROUTED:
        source_row = tl.load(INDICES + row, row < ROWS, -1)
        tl.device_assert(
            (source_row >= -1) & (source_row < SOURCE_ROWS), "Invalid routed source row"
        )
    active_tile = True
    if SKIP_PADDING:
        active_tile = tl.sum(((row < ROWS) & (source_row >= 0)).to(tl.int32), 0) > 0
    if active_tile:
        values = (
            tl.load(
                X + source_row[:, None] * COLS + cols[None, :],
                (row[:, None] < ROWS) & (source_row[:, None] >= 0),
                0,
            )
            .to(tl.float32)
            .reshape(ROW_TILE, 4, GRAN)
        )
        if BF16:
            quantized = values
        else:
            amax = tl.maximum(tl.max(tl.abs(values), 2), 1.0e-4)
            bits = (amax / 448.0).to(tl.int32, bitcast=True)
            exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
            exponent = tl.minimum(tl.maximum(exponent, 1), 254)
            scale = (exponent << 23).to(tl.float32, bitcast=True)
            quantized = tl.minimum(
                tl.maximum(values / scale[:, :, None], -448.0), 448.0
            )
        payload_mask = row < ROWS
        if SKIP_PADDING:
            # Receiver must initialize zero payloads/scales before any peer dispatch.
            tl.static_assert(ROUTED or METADATA)
            payload_mask = payload_mask & (source_row >= 0)
        tl.store(
            Q + row[:, None] * COLS + cols[None, :],
            quantized.reshape(ROW_TILE, 4 * GRAN),
            payload_mask[:, None],
        )
        if not BF16:
            packed = tl.sum(exponent.to(tl.uint32) << (tl.arange(0, 4)[None, :] * 8), 1)
            tl.store(SF + group * ROWS + row, packed, payload_mask)
        if FENCE_PACK == 16:
            # Group registers to avoid a system fence for every payload element.
            tl.inline_asm_elementwise(
                _PACKED_FENCE_ASM,
                constraints=_PACKED_FENCE_CONSTRAINTS,
                args=[values.to(tl.int32, bitcast=True)],
                dtype=tl.int32,
                is_pure=False,
                pack=16,
            )
        else:
            tl.static_assert(FENCE_PACK == 1)
            tl.inline_asm_elementwise(
                "fence.acq_rel.sys; mov.u32 $0, $1;",
                constraints="=r,r",
                args=[values.to(tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        tl.debug_barrier()
    completed = tl.atomic_add(COUNTER, 1, sem="acq_rel", scope="sys")
    if completed == tl.num_programs(0) * tl.num_programs(1) - 1:
        for task in range(PUBLISH_TASKS):
            tl.atomic_xchg(QUEUE + (TASK + task) * 4 + 3, 1, sem="release", scope="sys")


def _check_quantize(
    source, destination, destination_scales, queue, task, gran_k, indices=None
):
    if source.dtype not in (torch.bfloat16, torch.float32) or source.ndim != 2:
        raise ValueError("Peer quantization requires a BF16/FP32 matrix")
    rows, cols = source.shape
    if indices is not None:
        if indices.ndim != 1 or indices.dtype != torch.int32:
            raise ValueError("Routing indices must be an INT32 vector")
        if indices.device != source.device or not indices.is_contiguous():
            raise ValueError("Routing indices must be contiguous on the source device")
        if rows <= 0:
            raise ValueError("Routed source must have at least one row")
        rows = indices.shape[0]
    if gran_k not in (32, 128) or rows <= 0 or cols <= 0 or cols % (4 * gran_k):
        raise ValueError(
            "Peer quantization requires nonempty rows and K divisible by 4*gran_k"
        )
    if destination.dtype != torch.float8_e4m3fn or destination.shape != (rows, cols):
        raise ValueError("Peer values must be a matching E4M3 matrix")
    if destination_scales.dtype != torch.int32 or destination_scales.shape != (
        cols // (4 * gran_k),
        rows,
    ):
        raise ValueError("Peer scales must be INT32[packed_K,M] storage")
    if rows % 4:
        raise ValueError("Peer scale rows must meet the TMA 16-byte stride alignment")
    if queue.dtype != torch.int32 or queue.ndim != 2 or queue.shape[1] != 4:
        raise ValueError("Queue must be int32[num_tasks, 4]")
    if task < 0 or task >= queue.shape[0]:
        raise ValueError("Invalid queue task")
    if source.device.type != "cuda" or any(
        t.device != source.device or not t.is_contiguous()
        for t in (source, destination, destination_scales, queue)
    ):
        raise ValueError("Peer quantization requires contiguous CUDA mappings")


@torch.library.custom_op(
    "mono_ep::peer_quantize_publish",
    mutates_args=("destination", "destination_scales", "queue"),
)
def peer_quantize_publish(
    source: torch.Tensor,
    destination: torch.Tensor,
    destination_scales: torch.Tensor,
    queue: torch.Tensor,
    task: int,
    gran_k: int = 32,
) -> None:
    """Quantize BF16/FP32 directly into peer E4M3 values and packed scales.

    Row/scale tiles write both payloads; the last CTA publishes readiness.
    An invocation-local counter preserves the consumer queue metadata. This
    avoids local quantized intermediates. Mapping, lifetime and SM reservations
    follow peer_copy_fp8_publish; K must be divisible by four scale groups.
    """
    _check_quantize(source, destination, destination_scales, queue, task, gran_k)
    counter = torch.zeros((), device=source.device, dtype=torch.int32)
    _quantize_publish[
        (triton.cdiv(destination.shape[0], 16), source.shape[1] // (4 * gran_k))
    ](
        source,
        destination,
        destination_scales,
        queue,
        counter,
        task,
        source.shape[0],
        source.shape[1],
        gran_k,
        num_warps=4,
        enable_fp_fusion=False,
    )


@peer_quantize_publish.register_fake
def _peer_quantize_publish_fake(
    source, destination, destination_scales, queue, task, gran_k=32
):
    _check_quantize(source, destination, destination_scales, queue, task, gran_k)


@torch.library.custom_op(
    "mono_ep::peer_gather_quantize_publish",
    mutates_args=("destination", "destination_scales", "queue"),
)
def peer_gather_quantize_publish(
    source: torch.Tensor,
    indices: torch.Tensor,
    destination: torch.Tensor,
    destination_scales: torch.Tensor,
    queue: torch.Tensor,
    task: int,
    gran_k: int = 32,
    num_tasks: int = 1,
    skip_padding: bool = False,
) -> None:
    """Gather routed rows, quantize and publish directly into peer memory.

    indices maps output rows to source rows; -1 creates a zero padding row.
    Repeated indices duplicate tokens for different expert assignments.
    The caller constructs routing metadata and reserves the destination.
    Other mapping/lifetime constraints match peer_quantize_publish.
    num_tasks consecutive queue descriptors are released after the full packed
    payload is visible; descriptors must be initialized before this launch.
    skip_padding requires receiver-initialized zero payload and zero-row scales
    globally visible before dispatch; the caller owns that ordering.
    """
    _check_quantize(
        source, destination, destination_scales, queue, task, gran_k, indices
    )
    if num_tasks < 1 or task + num_tasks > queue.shape[0]:
        raise ValueError("Invalid packed dispatch publication range")
    counter = torch.zeros((), device=source.device, dtype=torch.int32)
    _quantize_publish[
        (triton.cdiv(destination.shape[0], 16), source.shape[1] // (4 * gran_k))
    ](
        source,
        destination,
        destination_scales,
        queue,
        counter,
        task,
        indices.shape[0],
        source.shape[1],
        gran_k,
        indices,
        True,
        source.shape[0],
        PUBLISH_TASKS=num_tasks,
        SKIP_PADDING=skip_padding,
        num_warps=4,
        enable_fp_fusion=False,
        debug=True,
    )


@peer_gather_quantize_publish.register_fake
def _peer_gather_quantize_publish_fake(
    source,
    indices,
    destination,
    destination_scales,
    queue,
    task,
    gran_k=32,
    num_tasks=1,
    skip_padding=False,
):
    _check_quantize(
        source, destination, destination_scales, queue, task, gran_k, indices
    )
    if num_tasks < 1 or task + num_tasks > queue.shape[0]:
        raise ValueError("Invalid packed dispatch publication range")


def _check_packed_dispatch(
    source,
    slots,
    ids,
    starts,
    sizes,
    ends,
    scores,
    destination,
    destination_scales,
    queue,
    out_ids,
    out_tokens,
    out_slots,
    out_ends,
    out_scores,
    bf16=False,
):
    if bf16:
        if (
            source.ndim != 2
            or source.shape[0] <= 0
            or source.shape[1] <= 0
            or source.shape[1] % 128
            or source.dtype != torch.bfloat16
            or destination.shape != (ids.numel(), source.shape[1])
            or destination.dtype != torch.bfloat16
            or queue.ndim != 2
            or queue.shape[1] != 4
            or queue.shape[0] <= 0
            or queue.dtype != torch.int32
            or ids.numel() <= 0
            or source.device.type != "cuda"
            or any(
                t.device != source.device or not t.is_contiguous()
                for t in (source, destination, queue)
            )
        ):
            raise ValueError("Invalid packed BF16 payload or readiness queue")
    else:
        _check_quantize(source, destination, destination_scales, queue, 0, 32, ids)
    rows, experts = destination.shape[0], queue.shape[0]
    if scores.ndim != 2 or scores.shape[0] != source.shape[0] or scores.shape[1] < 1:
        raise ValueError("Scores must match source tokens and top-k slots")
    if scores.numel() >= 2**31 or experts > rows:
        raise ValueError("Packed dispatch metadata exceeds supported capacity")
    specs = (
        (slots, (rows,), torch.int64),
        (ids, (rows,), torch.int32),
        (starts, (experts,), torch.int32),
        (sizes, (experts,), torch.int32),
        (ends, (experts,), torch.int32),
        (scores, scores.shape, torch.float32),
        (out_ids, (rows,), torch.int32),
        (out_tokens, (rows,), torch.int32),
        (out_slots, (rows,), torch.int32),
        (out_ends, (experts,), torch.int32),
        (out_scores, (rows,), torch.float32),
    )
    for tensor, shape, dtype in specs:
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != source.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("Invalid packed dispatch metadata shape, dtype or storage")


@torch.library.custom_op(
    "mono_ep::peer_packed_quantize_publish",
    mutates_args=(
        "destination",
        "destination_scales",
        "queue",
        "out_ids",
        "out_tokens",
        "out_slots",
        "out_ends",
        "out_scores",
    ),
)
def peer_packed_quantize_publish(
    source: torch.Tensor,
    slots: torch.Tensor,
    ids: torch.Tensor,
    starts: torch.Tensor,
    sizes: torch.Tensor,
    ends: torch.Tensor,
    scores: torch.Tensor,
    destination: torch.Tensor,
    destination_scales: torch.Tensor,
    queue: torch.Tensor,
    out_ids: torch.Tensor,
    out_tokens: torch.Tensor,
    out_slots: torch.Tensor,
    out_ends: torch.Tensor,
    out_scores: torch.Tensor,
    skip_padding: bool = False,
) -> None:
    """Publish compact MXFP8 values, scores and routing metadata in one kernel.

    Layout inputs come from DeviceExpertLayout with 128-aligned expert ranges.
    All destinations are peer mappings. Queue flags must be zero and previous
    consumers finished before launch. The final CTA releases every expert only
    after all payload and metadata writers have completed system-scope fences.
    skip_padding has the receiver initialization contract of routed dispatch.
    """
    _check_packed_dispatch(
        source,
        slots,
        ids,
        starts,
        sizes,
        ends,
        scores,
        destination,
        destination_scales,
        queue,
        out_ids,
        out_tokens,
        out_slots,
        out_ends,
        out_scores,
    )
    _launch_packed_dispatch(
        source,
        slots,
        ids,
        starts,
        sizes,
        ends,
        scores,
        destination,
        destination_scales,
        queue,
        out_ids,
        out_tokens,
        out_slots,
        out_ends,
        out_scores,
        16,
        skip_padding=skip_padding,
    )


def _launch_packed_dispatch(
    source,
    slots,
    ids,
    starts,
    sizes,
    ends,
    scores,
    destination,
    destination_scales,
    queue,
    out_ids,
    out_tokens,
    out_slots,
    out_ends,
    out_scores,
    row_tile,
    fence_pack=16,
    skip_padding=False,
    bf16=False,
):
    counter = torch.zeros((), device=source.device, dtype=torch.int32)
    _quantize_publish[
        (triton.cdiv(destination.shape[0], row_tile), source.shape[1] // 128)
    ](
        source,
        destination,
        destination_scales,
        queue,
        counter,
        0,
        destination.shape[0],
        source.shape[1],
        32,
        SOURCE_ROWS=source.shape[0],
        PUBLISH_TASKS=queue.shape[0],
        METADATA=True,
        SLOTS=slots,
        IDS=ids,
        STARTS=starts,
        SIZES=sizes,
        ENDS=ends,
        SCORES=scores,
        OUT_IDS=out_ids,
        OUT_TOKENS=out_tokens,
        OUT_SLOTS=out_slots,
        OUT_ENDS=out_ends,
        OUT_SCORES=out_scores,
        TOPK=scores.shape[1],
        ROW_TILE=row_tile,
        FENCE_PACK=fence_pack,
        SKIP_PADDING=skip_padding,
        BF16=bf16,
        num_warps=4,
        enable_fp_fusion=False,
    )


@peer_packed_quantize_publish.register_fake
def _peer_packed_quantize_publish_fake(
    source,
    slots,
    ids,
    starts,
    sizes,
    ends,
    scores,
    destination,
    destination_scales,
    queue,
    out_ids,
    out_tokens,
    out_slots,
    out_ends,
    out_scores,
    skip_padding=False,
):
    _check_packed_dispatch(
        source,
        slots,
        ids,
        starts,
        sizes,
        ends,
        scores,
        destination,
        destination_scales,
        queue,
        out_ids,
        out_tokens,
        out_slots,
        out_ends,
        out_scores,
    )


@torch.library.custom_op(
    "mono_ep::peer_packed_bf16_publish",
    mutates_args=(
        "destination",
        "queue",
        "out_ids",
        "out_tokens",
        "out_slots",
        "out_ends",
        "out_scores",
    ),
)
def peer_packed_bf16_publish(
    source: torch.Tensor,
    slots: torch.Tensor,
    ids: torch.Tensor,
    starts: torch.Tensor,
    sizes: torch.Tensor,
    ends: torch.Tensor,
    scores: torch.Tensor,
    destination: torch.Tensor,
    queue: torch.Tensor,
    out_ids: torch.Tensor,
    out_tokens: torch.Tensor,
    out_slots: torch.Tensor,
    out_ends: torch.Tensor,
    out_scores: torch.Tensor,
    skip_padding: bool = False,
) -> None:
    """Gather BF16 token rows and publish packed payload plus routing metadata.

    Inputs use a 128-aligned DeviceExpertLayout. Queue flags must be cleared
    after prior consumers finish. With skip_padding, the receiver must zero the
    destination before any producer starts. Every CTA contributes to completion,
    including empty tiles, before the last CTA releases all expert flags.
    """
    _peer_packed_bf16_fake(
        source,
        slots,
        ids,
        starts,
        sizes,
        ends,
        scores,
        destination,
        queue,
        out_ids,
        out_tokens,
        out_slots,
        out_ends,
        out_scores,
        skip_padding,
    )
    _launch_packed_dispatch(
        source,
        slots,
        ids,
        starts,
        sizes,
        ends,
        scores,
        destination,
        destination,
        queue,
        out_ids,
        out_tokens,
        out_slots,
        out_ends,
        out_scores,
        16,
        skip_padding=skip_padding,
        bf16=True,
    )


@peer_packed_bf16_publish.register_fake
def _peer_packed_bf16_fake(
    source,
    slots,
    ids,
    starts,
    sizes,
    ends,
    scores,
    destination,
    queue,
    out_ids,
    out_tokens,
    out_slots,
    out_ends,
    out_scores,
    skip_padding=False,
):
    _check_packed_dispatch(
        source,
        slots,
        ids,
        starts,
        sizes,
        ends,
        scores,
        destination,
        destination,
        queue,
        out_ids,
        out_tokens,
        out_slots,
        out_ends,
        out_scores,
        bf16=True,
    )


@torch.library.custom_op(
    "mono_ep::peer_gather_bf16_publish", mutates_args=("destination", "queue")
)
def peer_gather_bf16_publish(
    source: torch.Tensor,
    indices: torch.Tensor,
    destination: torch.Tensor,
    queue: torch.Tensor,
    num_tasks: int,
    skip_padding: bool = False,
) -> None:
    """Gather BF16 rows and release preinitialized descriptors after publication."""
    _peer_gather_bf16_fake(source, indices, destination, queue, num_tasks, skip_padding)
    counter = torch.zeros((), device=source.device, dtype=torch.int32)
    _quantize_publish[(triton.cdiv(indices.numel(), 16), source.shape[1] // 128)](
        source,
        destination,
        destination,
        queue,
        counter,
        0,
        indices.numel(),
        source.shape[1],
        32,
        indices,
        True,
        source.shape[0],
        PUBLISH_TASKS=num_tasks,
        SKIP_PADDING=skip_padding,
        BF16=True,
        num_warps=4,
    )


@peer_gather_bf16_publish.register_fake
def _peer_gather_bf16_fake(
    source, indices, destination, queue, num_tasks, skip_padding=False
):
    if (
        source.ndim != 2
        or source.dtype != torch.bfloat16
        or source.shape[0] <= 0
        or source.shape[1] <= 0
        or source.shape[1] % 128
        or indices.ndim != 1
        or indices.numel() <= 0
        or indices.dtype != torch.int32
        or destination.shape != (indices.numel(), source.shape[1])
        or destination.dtype != torch.bfloat16
        or queue.ndim != 2
        or queue.shape[1] != 4
        or queue.dtype != torch.int32
        or not 0 < num_tasks <= queue.shape[0]
        or source.device.type != "cuda"
        or any(
            t.device != source.device or not t.is_contiguous()
            for t in (source, indices, destination, queue)
        )
    ):
        raise ValueError("Invalid BF16 gather payload, indices or publication queue")
