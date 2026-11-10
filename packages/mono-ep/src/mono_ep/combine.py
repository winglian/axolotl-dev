# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Experimental peer scatter-add combine with device-side completion."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_publish(
    X,
    IDS,
    Y,
    QUEUE,
    TASK: tl.constexpr,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SCORES=None,
    SCORE_IDS=None,
    SCORE_DEST=None,
    SCORE_QUEUE=None,
    SCORE_SLOTS: tl.constexpr = 0,
    WITH_SCORES: tl.constexpr = False,
    ROW_TILE: tl.constexpr = 16,
    VECTORIZE: tl.constexpr = True,
    ASSIGN: tl.constexpr = False,
    REDUCE_ROWS: tl.constexpr = False,
    REDUCE_WIDTH: tl.constexpr = 1,
):
    lanes = tl.arange(0, 128)
    if REDUCE_ROWS:
        row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
        token = tl.load(IDS + row, row < ROWS, -1)
        tl.device_assert((token >= -1) & (token < TOKENS), "Invalid score slot")
        col = tl.arange(0, REDUCE_WIDTH)
        partial = tl.load(
            X + row[:, None] * HIDDEN + col[None, :],
            (row[:, None] < ROWS) & (token[:, None] >= 0) & (col[None, :] < HIDDEN),
            0,
        ).to(tl.float32)
        value = tl.sum(partial, 1)
        tl.atomic_add(Y + token, value, token >= 0, sem="relaxed", scope="sys")
        fence_lanes = value.to(tl.int32)
    elif VECTORIZE:
        row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
        token = tl.load(IDS + row, row < ROWS, -1)
        tl.device_assert(
            (token >= -1) & (token < TOKENS), "Invalid combine token index"
        )
        value = tl.full((ROW_TILE, 128), 0, tl.float32)
        for start in range(0, HIDDEN, 128):
            col = start + lanes
            value = tl.load(
                X + row[:, None] * HIDDEN + col[None, :],
                (row[:, None] < ROWS) & (token[:, None] >= 0) & (col[None, :] < HIDDEN),
                0,
            ).to(tl.float32)
            if ASSIGN:
                tl.store(
                    Y + token[:, None] * HIDDEN + col[None, :],
                    value,
                    (token[:, None] >= 0) & (col[None, :] < HIDDEN),
                )
            else:
                tl.atomic_add(
                    Y + token[:, None] * HIDDEN + col[None, :],
                    value,
                    (token[:, None] >= 0) & (col[None, :] < HIDDEN),
                    sem="relaxed",
                    scope="sys",
                )
        if WITH_SCORES:
            slot = tl.load(SCORE_IDS + row, row < ROWS, -1)
            tl.device_assert(
                (slot >= -1) & (slot < SCORE_SLOTS), "Invalid combine score index"
            )
            score = tl.load(SCORES + row, row < ROWS, 0)
            tl.atomic_add(
                SCORE_DEST + slot, score, slot >= 0, sem="relaxed", scope="sys"
            )
        fence_lanes = value.to(tl.int32)
    else:
        for offset in range(ROW_TILE):
            row = tl.program_id(0) * ROW_TILE + offset
            token = tl.load(IDS + row, row < ROWS, -1)
            tl.device_assert(
                (token >= -1) & (token < TOKENS), "Invalid combine token index"
            )
            for start in range(0, HIDDEN, 128):
                col = start + lanes
                value = tl.load(
                    X + row * HIDDEN + col,
                    (row < ROWS) & (token >= 0) & (col < HIDDEN),
                    0,
                ).to(tl.float32)
                tl.atomic_add(
                    Y + token * HIDDEN + col,
                    value,
                    (token >= 0) & (col < HIDDEN),
                    sem="relaxed",
                    scope="sys",
                )
            if WITH_SCORES:
                slot = tl.load(SCORE_IDS + row, row < ROWS, -1)
                tl.device_assert(
                    (slot >= -1) & (slot < SCORE_SLOTS), "Invalid combine score index"
                )
                score = tl.load(SCORES + row, row < ROWS, 0)
                tl.atomic_add(
                    SCORE_DEST + slot, score, slot >= 0, sem="relaxed", scope="sys"
                )
        fence_lanes = lanes
    tl.inline_asm_elementwise(
        "fence.acq_rel.sys; mov.u32 $0, $1;",
        constraints="=r,r",
        args=[fence_lanes],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()
    # The last CTA acquires the other CTAs' releases before publishing readiness.
    completed = tl.atomic_add(QUEUE + TASK * 4 + 2, 1, sem="acq_rel", scope="sys")
    if completed == tl.num_programs(0) - 1:
        tl.atomic_xchg(QUEUE + TASK * 4 + 3, 1, sem="release", scope="sys")
        if WITH_SCORES:
            tl.atomic_xchg(SCORE_QUEUE + TASK * 4 + 3, 1, sem="release", scope="sys")


@triton.jit
def _wait_copy(X, Y, QUEUE, TASKS: tl.constexpr, N: tl.constexpr):
    lanes = tl.arange(0, 128)
    for task in range(TASKS):
        pointers = QUEUE + task * 4 + 3 + lanes * 0
        ready = tl.inline_asm_elementwise(
            "ld.acquire.sys.global.u32 $0, [$1];",
            constraints="=r,l",
            args=[pointers],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        spins = 0
        while tl.min(ready, 0) == 0:
            spins += 1
            tl.device_assert(spins < 10000000, "Combine completion timeout")
            ready = tl.inline_asm_elementwise(
                "ld.acquire.sys.global.u32 $0, [$1];",
                constraints="=r,l",
                args=[pointers],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
    tl.debug_barrier()
    if N > 0:
        for start in range(0, N, 128):
            offset = start + lanes
            tl.store(Y + offset, tl.load(X + offset, offset < N, 0.0), offset < N)


def _check_queue(queue, device):
    if (
        queue.dtype != torch.int32
        or queue.ndim != 2
        or queue.shape[1] != 4
        or queue.shape[0] == 0
    ):
        raise ValueError("Combine queue must be INT32[num_producers,4]")
    if queue.device != device or not queue.is_contiguous():
        raise ValueError("Combine queue must be contiguous on the operand device")


def _check_scatter(
    source, indices, destination, queue, task, allow_bf16=False, reduce_rows=False
):
    if (
        source.ndim != 2
        or destination.ndim != 2
        or source.shape[1] == 0
        or destination.shape[1] != (1 if reduce_rows else source.shape[1])
    ):
        raise ValueError("Combine requires matching hidden dimensions")
    if source.dtype not in (torch.bfloat16, torch.float32) or (
        destination.dtype != torch.float32
        and not (allow_bf16 and destination.dtype == torch.bfloat16)
    ):
        raise ValueError("Combine accumulates BF16/FP32 contributions into FP32")
    if destination.shape[0] == 0 or destination.shape[1] == 0:
        raise ValueError("Combine destination must be nonempty")
    if indices.dtype != torch.int32 or indices.shape != (source.shape[0],):
        raise ValueError("Combine indices must be INT32[source_rows]")
    if source.device.type != "cuda" or any(
        t.device != source.device or not t.is_contiguous()
        for t in (source, indices, destination)
    ):
        raise ValueError("Combine requires contiguous CUDA tensors and peer mappings")
    _check_queue(queue, source.device)
    if task < 0 or task >= queue.shape[0]:
        raise ValueError("Invalid combine producer index")


@torch.library.custom_op(
    "mono_ep::peer_scatter_add_publish", mutates_args=("destination", "queue")
)
def peer_scatter_add_publish(
    source: torch.Tensor,
    indices: torch.Tensor,
    destination: torch.Tensor,
    queue: torch.Tensor,
    task: int,
) -> None:
    """Add already-weighted expert rows into peer token rows and signal completion.

    -1 indices are padding. Each producer owns a distinct queue row. The
    caller zeros accumulation and entire queue rows before any producer or consumer
    starts, and preserves all buffers until every consumer finishes. FP32
    atomic accumulation is not bitwise deterministic across producers or CTAs.
    Queue column 2 counts completed CTAs; column 3 is the readiness flag.
    """
    _check_scatter(source, indices, destination, queue, task)
    _scatter_publish[(max(1, triton.cdiv(source.shape[0], 16)),)](
        source,
        indices,
        destination,
        queue,
        task,
        source.shape[0],
        destination.shape[0],
        source.shape[1],
        VECTORIZE=source.shape[1] > 256,
        num_warps=4,
        debug=True,
    )


@peer_scatter_add_publish.register_fake
def _scatter_fake(source, indices, destination, queue, task):
    _check_scatter(source, indices, destination, queue, task)


@torch.library.custom_op(
    "mono_ep::peer_reduce_scatter_publish", mutates_args=("destination", "queue")
)
def peer_reduce_scatter_publish(
    partial: torch.Tensor,
    indices: torch.Tensor,
    destination: torch.Tensor,
    queue: torch.Tensor,
    task: int,
) -> None:
    """Reduce score partials, scatter into peer slots, and publish completion.

    Inactive (-1) rows are never read and may be uninitialized. Destination and
    queue ownership/reset rules match peer_scatter_add_publish.
    """
    _reduce_scatter_fake(partial, indices, destination, queue, task)
    _scatter_publish[(max(1, triton.cdiv(partial.shape[0], 16)),)](
        partial,
        indices,
        destination,
        queue,
        task,
        partial.shape[0],
        destination.shape[0],
        partial.shape[1],
        REDUCE_ROWS=True,
        REDUCE_WIDTH=triton.next_power_of_2(partial.shape[1]),
        num_warps=4,
        debug=True,
    )


@peer_reduce_scatter_publish.register_fake
def _reduce_scatter_fake(partial, indices, destination, queue, task):
    _check_scatter(partial, indices, destination, queue, task, reduce_rows=True)
    if partial.dtype != torch.float32:
        raise ValueError("Routing-score partials must be FP32")


@torch.library.custom_op(
    "mono_ep::peer_scatter_store_publish", mutates_args=("destination", "queue")
)
def peer_scatter_store_publish(
    source: torch.Tensor,
    indices: torch.Tensor,
    destination: torch.Tensor,
    queue: torch.Tensor,
    task: int,
) -> None:
    """Store unique rows into BF16/FP32 peer slots and publish completion.

    Nonnegative indices must be unique across all producers. Padding uses -1;
    callers clear unwritten slots and queue entries before producers start.
    """
    _check_scatter(source, indices, destination, queue, task, allow_bf16=True)
    _scatter_publish[(max(1, triton.cdiv(source.shape[0], 16)),)](
        source,
        indices,
        destination,
        queue,
        task,
        source.shape[0],
        destination.shape[0],
        source.shape[1],
        VECTORIZE=True,
        ASSIGN=True,
        num_warps=4,
        debug=True,
    )


@peer_scatter_store_publish.register_fake
def _scatter_store_fake(source, indices, destination, queue, task):
    _check_scatter(source, indices, destination, queue, task, allow_bf16=True)


def _check_backward_scatter(
    dx,
    token_ids,
    ds,
    score_ids,
    dx_dest,
    ds_dest,
    dx_queue,
    ds_queue,
    task,
    assign=False,
):
    if ds.shape != (dx.shape[0],) or ds.dtype != torch.float32:
        raise ValueError("Routing gradients must be FP32 with one value per expert row")
    _check_scatter(dx, token_ids, dx_dest, dx_queue, task, allow_bf16=assign)
    _check_scatter(ds[:, None], score_ids, ds_dest, ds_queue, task)
    if ds.device != dx.device:
        raise ValueError("Input and routing gradients must share the producer device")


@torch.library.custom_op(
    "mono_ep::peer_backward_scatter_publish",
    mutates_args=("dx_dest", "ds_dest", "dx_queue", "ds_queue"),
)
def peer_backward_scatter_publish(
    dx: torch.Tensor,
    token_ids: torch.Tensor,
    ds: torch.Tensor,
    score_ids: torch.Tensor,
    dx_dest: torch.Tensor,
    ds_dest: torch.Tensor,
    dx_queue: torch.Tensor,
    ds_queue: torch.Tensor,
    task: int,
) -> None:
    """Combine input and score gradients in one peer kernel and publish both.

    Score IDs address flattened routing slots independently of token IDs;
    either map may use -1 padding. Both payloads complete before either flag
    is published. Buffer ownership and reset rules match peer_scatter_add_publish.
    """
    _check_backward_scatter(
        dx, token_ids, ds, score_ids, dx_dest, ds_dest, dx_queue, ds_queue, task
    )
    _scatter_publish[(max(1, triton.cdiv(dx.shape[0], 16)),)](
        dx,
        token_ids,
        dx_dest,
        dx_queue,
        task,
        dx.shape[0],
        dx_dest.shape[0],
        dx.shape[1],
        ds,
        score_ids,
        ds_dest,
        ds_queue,
        ds_dest.shape[0],
        True,
        VECTORIZE=dx.shape[1] > 256,
        num_warps=4,
        debug=True,
    )


@peer_backward_scatter_publish.register_fake
def _backward_scatter_fake(
    dx, token_ids, ds, score_ids, dx_dest, ds_dest, dx_queue, ds_queue, task
):
    _check_backward_scatter(
        dx, token_ids, ds, score_ids, dx_dest, ds_dest, dx_queue, ds_queue, task
    )


def _check_wait(source, destination, queue):
    if (
        source.dtype != torch.float32
        or destination.dtype not in (torch.float32, torch.bfloat16)
        or source.shape != destination.shape
    ):
        raise ValueError(
            "Combine copy requires FP32 input and matching FP32/BF16 output"
        )
    if source.device.type != "cuda" or any(
        t.device != source.device or not t.is_contiguous()
        for t in (source, destination)
    ):
        raise ValueError("Combine copy requires contiguous CUDA buffers")
    _check_queue(queue, source.device)


@torch.library.custom_op("mono_ep::combine_wait_copy", mutates_args=("destination",))
def combine_wait_copy(
    source: torch.Tensor, destination: torch.Tensor, queue: torch.Tensor
) -> None:
    """Wait for all producers on the GPU, then copy/cast accumulated token rows.

    The caller must reserve SM capacity for producers and ensure each queue
    row will be published exactly once. No producer may modify the source
    after publishing; buffer/flag reuse requires completion of this consumer.
    Large copies run after the single waiting CTA on the same stream, avoiding
    grids of waiting CTAs that could starve the producers.
    """
    _check_wait(source, destination, queue)
    _wait_copy[(1,)](
        source,
        destination,
        queue,
        queue.shape[0],
        source.numel() if source.numel() <= 4096 else 0,
        num_warps=4,
        debug=True,
    )
    if source.numel() > 4096:
        destination.copy_(source)


@combine_wait_copy.register_fake
def _wait_fake(source, destination, queue):
    _check_wait(source, destination, queue)


@torch.library.custom_op(
    "mono_ep::grouped_fp8_scatter", mutates_args=("destination", "queue")
)
def grouped_fp8_scatter(
    x: torch.Tensor,
    sx: torch.Tensor,
    w: torch.Tensor,
    sw: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    destination: torch.Tensor,
    queue: torch.Tensor,
    task: int,
    assign: bool = False,
    row_end: torch.Tensor | None = None,
) -> None:
    """Scatter BF16-rounded grouped GEMM results and publish peer completion.

    The caller zeros destination and queue before launch and preserves them until
    consumption completes. Scratch storage supplies descriptor metadata only.
    With assign=True, every nonnegative destination ID must be unique across
    all producers; untouched slots must be initialized by the caller.
    row_end optionally bounds the active prefix on the GPU; rows beyond it must
    have no valid assignments. Every CTA still publishes completion.
    """
    _grouped_fp8_scatter_fake(
        x, sx, w, sw, expert_ids, token_ids, destination, queue, task, assign, row_end
    )
    import deep_gemm

    scratch = torch.empty(
        (x.shape[0], w.shape[1]), device=x.device, dtype=torch.bfloat16
    )
    sx = deep_gemm.transform_sf_into_required_layout(
        sx, x.shape[0], x.shape[1], (1, 32)
    )
    sw = deep_gemm.transform_sf_into_required_layout(
        sw, w.shape[1], w.shape[2], (1, 32), num_groups=w.shape[0]
    )
    kwargs = {"assign": True} if assign else {}
    if row_end is not None:
        kwargs["row_end"] = row_end
    deep_gemm.fp8_grouped_scatter_gemm_nt(
        (x, sx),
        (w, sw),
        scratch,
        expert_ids,
        token_ids,
        destination,
        queue,
        task,
        **kwargs,
    )


@grouped_fp8_scatter.register_fake
def _grouped_fp8_scatter_fake(
    x,
    sx,
    w,
    sw,
    expert_ids,
    token_ids,
    destination,
    queue,
    task,
    assign=False,
    row_end=None,
):
    if row_end is not None and (
        row_end.device != x.device
        or row_end.dtype != torch.int32
        or row_end.numel() != 1
        or not row_end.is_contiguous()
    ):
        raise ValueError("Scatter row_end must be one contiguous CUDA INT32 value")
    if (
        x.ndim != 2
        or w.ndim != 3
        or x.shape[1] != w.shape[2]
        or x.dtype != torch.float8_e4m3fn
        or w.dtype != torch.float8_e4m3fn
        or sx.dtype not in (torch.int32, torch.float32)
        or sw.dtype not in (torch.int32, torch.float32)
    ):
        raise ValueError("Expected E4M3 x[M,K], w[E,N,K] and FP32/UE8M0 scales")
    if x.device.type != "cuda" or any(
        t.device != x.device
        for t in (sx, w, sw, expert_ids, token_ids, destination, queue)
    ):
        raise ValueError("Fused scatter operands must share a CUDA device")
    if (
        x.shape[0] == 0
        or expert_ids.shape != (x.shape[0],)
        or token_ids.shape != (x.shape[0],)
        or expert_ids.dtype != torch.int32
        or token_ids.dtype != torch.int32
        or not expert_ids.is_contiguous()
        or not token_ids.is_contiguous()
        or destination.ndim != 2
        or destination.shape[0] == 0
        or destination.shape[1] != w.shape[1]
        or (
            destination.dtype != torch.float32
            and not (assign and destination.dtype == torch.bfloat16)
        )
        or not destination.is_contiguous()
    ):
        raise ValueError("Invalid packed expert IDs, token IDs or scatter destination")
    _check_queue(queue, x.device)
    if task < 0 or task >= queue.shape[0]:
        raise ValueError("Invalid combine producer index")


@triton.jit
def _reduce_slots(X, Y, N: tl.constexpr, H: tl.constexpr, TOPK: tl.constexpr):
    offset = tl.program_id(0) * 256 + tl.arange(0, 256)
    row, col = offset // H, offset % H
    value = tl.full((256,), 0, tl.float32)
    for slot in range(TOPK):
        value += tl.load(X + (row * TOPK + slot) * H + col, offset < N, 0.0).to(
            tl.float32
        )
    tl.store(Y + offset, value, offset < N)


def _check_slot_reduce(source, destination, queue, topk):
    if (
        topk < 1
        or source.ndim != 2
        or destination.ndim != 2
        or source.shape != (destination.shape[0] * topk, destination.shape[1])
        or destination.numel() == 0
        or source.dtype not in (torch.float32, torch.bfloat16)
        or destination.dtype not in (torch.float32, torch.bfloat16)
        or source.device.type != "cuda"
        or destination.device != source.device
        or not source.is_contiguous()
        or not destination.is_contiguous()
    ):
        raise ValueError(
            "Expected contiguous CUDA BF16/FP32 [tokens*topk,hidden] slots"
        )
    _check_queue(queue, source.device)


@torch.library.custom_op("mono_ep::combine_wait_reduce", mutates_args=("destination",))
def combine_wait_reduce(
    source: torch.Tensor, destination: torch.Tensor, queue: torch.Tensor, topk: int
) -> None:
    """Acquire every producer's completion before reducing exclusive route slots."""
    _check_slot_reduce(source, destination, queue, topk)
    _wait_copy[(1,)](
        source, destination, queue, queue.shape[0], 0, num_warps=4, debug=True
    )
    _reduce_slots[(triton.cdiv(destination.numel(), 256),)](
        source,
        destination,
        destination.numel(),
        destination.shape[1],
        topk,
        num_warps=4,
    )


@combine_wait_reduce.register_fake
def _slot_reduce_fake(source, destination, queue, topk):
    _check_slot_reduce(source, destination, queue, topk)


@torch.library.custom_op(
    "mono_ep::grouped_bf16_scatter", mutates_args=("destination", "queue")
)
def grouped_bf16_scatter(
    x: torch.Tensor,
    w: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    destination: torch.Tensor,
    queue: torch.Tensor,
    task: int,
    assign: bool = False,
    row_end: torch.Tensor | None = None,
) -> None:
    """Fuse BF16 grouped GEMM with peer scatter and completion publication.

    Buffer initialization, unique-ID requirements for assign, and optional GPU
    row bounds follow grouped_fp8_scatter. Scratch is descriptor storage only.
    """
    _grouped_bf16_scatter_fake(
        x, w, expert_ids, token_ids, destination, queue, task, assign, row_end
    )
    import deep_gemm

    scratch = x.new_empty((x.shape[0], w.shape[1]))
    deep_gemm.bf16_grouped_scatter_gemm_nt(
        x,
        w,
        scratch,
        expert_ids,
        token_ids,
        destination,
        queue,
        task,
        assign=assign,
        row_end=row_end,
    )


@grouped_bf16_scatter.register_fake
def _grouped_bf16_scatter_fake(
    x,
    w,
    expert_ids,
    token_ids,
    destination,
    queue,
    task,
    assign=False,
    row_end=None,
):
    if row_end is not None and (
        row_end.device != x.device
        or row_end.dtype != torch.int32
        or row_end.numel() != 1
        or not row_end.is_contiguous()
    ):
        raise ValueError("Scatter row_end must be one contiguous CUDA INT32 value")
    if (
        x.ndim != 2
        or w.ndim != 3
        or x.shape[1] != w.shape[2]
        or x.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
    ):
        raise ValueError("Expected BF16 x[M,K] and w[E,N,K]")
    if x.device.type != "cuda" or any(
        t.device != x.device for t in (w, expert_ids, token_ids, destination, queue)
    ):
        raise ValueError("Fused scatter operands must share a CUDA device")
    if (
        x.shape[0] == 0
        or expert_ids.shape != (x.shape[0],)
        or token_ids.shape != (x.shape[0],)
        or expert_ids.dtype != torch.int32
        or token_ids.dtype != torch.int32
        or not expert_ids.is_contiguous()
        or not token_ids.is_contiguous()
        or destination.ndim != 2
        or destination.shape[0] == 0
        or destination.shape[1] != w.shape[1]
        or (
            destination.dtype != torch.float32
            and not (assign and destination.dtype == torch.bfloat16)
        )
        or not destination.is_contiguous()
    ):
        raise ValueError("Invalid packed expert IDs, token IDs or scatter destination")
    _check_queue(queue, x.device)
    if task < 0 or task >= queue.shape[0]:
        raise ValueError("Invalid combine producer index")
