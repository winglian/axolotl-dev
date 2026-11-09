# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Receive-side packing from symmetric peer mappings using GPU routing metadata."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_peer_rows(
    SOURCES,
    ASSIGNMENTS,
    OUTPUT,
    ROW_END,
    ROWS: tl.constexpr,
    BOUNDED: tl.constexpr,
    CAPACITY: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tiles = tl.cdiv(WIDTH, BLOCK)
    end = tl.load(ROW_END) if BOUNDED else ROWS
    for task in range(tl.program_id(0), end * tiles, tl.num_programs(0)):
        row = (task // tiles).to(tl.int64)
        cols = (task % tiles) * BLOCK + tl.arange(0, BLOCK)
        assignment = tl.load(ASSIGNMENTS + row).to(tl.int64)
        token = assignment // TOPK
        source_rank = token // CAPACITY
        source_token = token % CAPACITY
        values = tl.full((BLOCK,), 0, tl.float32)
        for rank in tl.static_range(len(SOURCES)):
            if (assignment >= 0) & (source_rank == rank):
                values = tl.load(
                    SOURCES[rank] + source_token * WIDTH + cols,
                    cols < WIDTH,
                    other=0,
                ).to(tl.float32)
        tl.store(OUTPUT + row * WIDTH + cols, values, cols < WIDTH)


def _validate(sources, row_assignments, topk, row_end=None):
    if not sources or topk < 1:
        raise ValueError("Expected peer sources and positive top-k")
    first = sources[0]
    if (
        first.ndim != 2
        or min(first.shape) < 1
        or first.dtype not in (torch.bfloat16, torch.float32)
        or first.device.type != "cuda"
    ):
        raise ValueError("Expected BF16/FP32 CUDA [capacity, width] peer mappings")
    if any(
        t.shape != first.shape
        or t.dtype != first.dtype
        or t.device != first.device
        or not t.is_contiguous()
        for t in sources
    ):
        raise ValueError("Peer mappings must share shape, dtype and local device")
    if (
        row_assignments.ndim != 1
        or row_assignments.dtype != torch.int64
        or row_assignments.device != first.device
        or not row_assignments.is_contiguous()
    ):
        raise ValueError("Expected contiguous CUDA INT64 packed-row assignments")
    if row_end is not None and (
        row_end.numel() != 1
        or row_end.dtype != torch.int32
        or row_end.device != first.device
        or not row_end.is_contiguous()
    ):
        raise ValueError("Expected one contiguous CUDA INT32 row bound")


@torch.library.custom_op("mono_ep::pack_peer_rows", mutates_args=())
def pack_peer_rows(
    sources: list[torch.Tensor],
    row_assignments: torch.Tensor,
    topk: int,
    row_end: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather global token assignments directly into packed expert rows.

    Sources are equal-capacity mappings in process-group rank order. Assignment
    IDs encode ((source_rank * capacity + token) * topk + slot), or -1 for padding.
    The caller must publish inputs before this operation and protect all source
    mappings from reuse until it finishes. This operation performs no barrier.
    Optional GPU row_end must be in [0, packed capacity]; rows beyond it stay zero.
    """
    _validate(sources, row_assignments, topk, row_end)
    shape = (row_assignments.numel(), sources[0].shape[1])
    out = (
        sources[0].new_empty(shape) if row_end is None else sources[0].new_zeros(shape)
    )
    if out.shape[0]:
        _pack_peer_rows[(min(1024, out.shape[0] * triton.cdiv(out.shape[1], 256)),)](
            tuple(sources),
            row_assignments,
            out,
            row_assignments if row_end is None else row_end,
            out.shape[0],
            row_end is not None,
            sources[0].shape[0],
            topk,
            out.shape[1],
            256,
        )
    return out


@pack_peer_rows.register_fake
def _pack_peer_rows_fake(sources, row_assignments, topk, row_end=None):
    _validate(sources, row_assignments, topk, row_end)
    return sources[0].new_empty((row_assignments.numel(), sources[0].shape[1]))


@triton.jit
def _return_peer_rows(
    DX,
    DS,
    ASSIGNMENTS,
    DESTINATIONS,
    SCORE_DESTINATIONS,
    ROW_END,
    ROWS: tl.constexpr,
    BOUNDED: tl.constexpr,
    SLOTS: tl.constexpr,
    WIDTH: tl.constexpr,
    WITH_DX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tiles = tl.cdiv(WIDTH, BLOCK)
    end = tl.load(ROW_END) if BOUNDED else ROWS
    for task in range(tl.program_id(0), end * tiles, tl.num_programs(0)):
        row = (task // tiles).to(tl.int64)
        tile = task % tiles
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        assignment = tl.load(ASSIGNMENTS + row).to(tl.int64)
        rank = assignment // SLOTS
        slot = assignment % SLOTS
        for peer in tl.static_range(len(DESTINATIONS)):
            if (assignment >= 0) & (rank == peer):
                if WITH_DX:
                    values = tl.load(DX + row * WIDTH + cols, cols < WIDTH, 0)
                    tl.store(
                        DESTINATIONS[peer] + slot * WIDTH + cols, values, cols < WIDTH
                    )
                if tile == 0:
                    tl.store(SCORE_DESTINATIONS[peer] + slot, tl.load(DS + row))


def return_peer_rows(
    dx, ds, row_assignments, destinations, score_destinations, row_end=None
):
    """Write unique routed slots; caller owns zeroing and system-scope barriers."""
    rows = ds.shape[0]
    width = dx.shape[1] if dx is not None else 1
    if rows:
        _return_peer_rows[(min(1024, rows * triton.cdiv(width, 256)),)](
            dx if dx is not None else ds,
            ds,
            row_assignments,
            tuple(destinations),
            tuple(score_destinations),
            row_assignments if row_end is None else row_end,
            rows,
            row_end is not None,
            destinations[0].shape[0],
            width,
            dx is not None,
            256,
        )


@triton.jit
def _pack_peer_ready(
    SOURCES,
    ASSIGNMENTS,
    OUTPUT,
    QUEUE,
    COUNTERS,
    CAPACITY: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    WORKERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(1)
    worker = tl.program_id(0)
    begin = tl.load(QUEUE + expert * 4 + 1)
    count = tl.load(QUEUE + expert * 4 + 2)
    tiles = tl.cdiv(WIDTH, BLOCK)
    for task in range(worker, count * tiles, WORKERS):
        row = (begin + task // tiles).to(tl.int64)
        cols = (task % tiles) * BLOCK + tl.arange(0, BLOCK)
        assignment = tl.load(ASSIGNMENTS + row).to(tl.int64)
        token = assignment // TOPK
        source_rank = token // CAPACITY
        source_token = token % CAPACITY
        values = tl.full((BLOCK,), 0, tl.float32)
        for rank in tl.static_range(len(SOURCES)):
            if (assignment >= 0) & (source_rank == rank):
                values = tl.load(
                    SOURCES[rank] + source_token * WIDTH + cols,
                    cols < WIDTH,
                    other=0,
                ).to(tl.float32)
        tl.store(OUTPUT + row * WIDTH + cols, values, cols < WIDTH)
    tl.debug_barrier()
    completed = tl.atomic_add(COUNTERS + expert, 1, sem="acq_rel", scope="sys")
    if completed == WORKERS - 1:
        tl.atomic_xchg(QUEUE + expert * 4 + 3, 1, sem="release", scope="sys")


@torch.library.custom_op(
    "mono_ep::pack_peer_rows_ready_out", mutates_args=("out", "queue", "counters")
)
def pack_peer_rows_ready_out(
    sources: list[torch.Tensor],
    row_assignments: torch.Tensor,
    topk: int,
    out: torch.Tensor,
    queue: torch.Tensor,
    counters: torch.Tensor,
    workers: int = 32,
) -> None:
    """Gather and publish each expert independently to a readiness-aware consumer.

    Queue rows contain expert/start/padded-count/ready. Caller initializes output,
    ready flags and counters to zero and publishes source data before launch.
    Queue/counters are invocation-owned until producer and consumer both finish.
    Concurrent consumers must reserve SM capacity for producer progress.
    """
    _pack_peer_rows_ready_fake(
        sources, row_assignments, topk, out, queue, counters, workers
    )
    _pack_peer_ready[(workers, queue.shape[0])](
        tuple(sources),
        row_assignments,
        out,
        queue,
        counters,
        sources[0].shape[0],
        topk,
        out.shape[1],
        workers,
        256,
    )


@pack_peer_rows_ready_out.register_fake
def _pack_peer_rows_ready_fake(
    sources, row_assignments, topk, out, queue, counters, workers=32
):
    _validate(sources, row_assignments, topk)
    if workers < 1 or workers > 128:
        raise ValueError("Expected 1–128 producer workers per task")
    if (
        out.shape != (row_assignments.numel(), sources[0].shape[1])
        or out.dtype != sources[0].dtype
        or out.device != sources[0].device
        or not out.is_contiguous()
        or queue.ndim != 2
        or queue.shape[0] < 1
        or queue.shape[1] != 4
        or queue.dtype != torch.int32
        or queue.device != out.device
        or not queue.is_contiguous()
        or counters.shape != (queue.shape[0],)
        or counters.dtype != torch.int32
        or counters.device != out.device
        or not counters.is_contiguous()
    ):
        raise ValueError("Invalid peer gather output or readiness metadata")
