# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Bandwidth-shaped peer row streams for the fixed-slot dispatch layout.

The layout (``peer_dispatch.py``) gives every receive row a fixed source: row ``s * cap + t``
is token ``t`` of rank ``s``, so a program that owns a contiguous range of receive rows
streams from exactly one peer. These kernels move whole rows as 16-byte vectors with
``BLOCK_ROWS`` rows in flight per program, size the grid to the SM count, and spread the
programs over the peers so every link carries traffic at once. The fused runtime's
per-expert ``pack_peer_rows`` / ``return_peer_rows`` (``peer_packing.py``) are untouched.

All launchers are plain functions: they run inside the dispatch custom ops, which are
already opaque to the compiler, and perform no barrier of their own.
"""

import torch
import triton
import triton.language as tl

TILE_BYTES = 32768
MAX_BLOCK_COLS = 512
_PEER_DELTAS: dict[tuple[int, ...], torch.Tensor] = {}
_SM_COUNTS: dict[torch.device, int] = {}


def _wide_view(tensor):
    """Rows viewed as the widest integer the row length allows, so 16 B vectors are
    whole elements; zero rows stay zero under any view."""
    row_bytes = tensor.shape[1] * tensor.element_size()
    for dtype in (torch.int64, torch.int32, torch.int16):
        if row_bytes % dtype.itemsize == 0:
            return tensor.view(dtype)
    return tensor.view(torch.int8)


def _tile(width, element_size):
    block_cols = min(triton.next_power_of_2(width), MAX_BLOCK_COLS)
    block_rows = max(8, min(1024, TILE_BYTES // (block_cols * element_size)))
    return block_rows, triton.next_power_of_2(block_rows), block_cols


def peer_deltas(tensors, element_size):
    """``[len(tensors)]`` int64 offsets of every buffer from ``tensors[0]`` in units of
    ``element_size`` (the kernels' row view), cached per address tuple: the views of a
    symmetric workspace never move, and building the tensor per call would be a pageable
    host-to-device copy (a stream sync). The kernels add a delta to the first buffer's
    pointer instead of casting a loaded address to a pointer, which Triton's alignment
    analysis cannot see through (the loads would fall back to 8-byte scalars)."""
    key = (element_size, *(t.data_ptr() for t in tensors))
    deltas = _PEER_DELTAS.get(key)
    if deltas is None or deltas.device != tensors[0].device:
        if any((ptr - key[1]) % 16 for ptr in key[1:]):
            raise ValueError(
                "peer buffers must be 16-byte aligned relative to each other"
            )
        deltas = torch.tensor(
            [(ptr - key[1]) // element_size for ptr in key[1:]],
            dtype=torch.int64,
            device=tensors[0].device,
        )
        _PEER_DELTAS[key] = deltas
    return deltas


def forget_peer_deltas(tensors):
    addresses = tuple(t.data_ptr() for t in tensors)
    for key in [k for k in _PEER_DELTAS if k[1:] == addresses]:
        del _PEER_DELTAS[key]


def _programs_per_source(device, world, programs_per_sm):
    sms = _SM_COUNTS.get(device)
    if sms is None:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        _SM_COUNTS[device] = sms
    return max(1, (sms * programs_per_sm + world - 1) // world)


def _check_rows(rows, name, dtype, numel):
    if (
        rows.dtype != dtype
        or rows.ndim != 1
        or rows.numel() != numel
        or not rows.is_contiguous()
    ):
        raise ValueError(f"expected a contiguous {dtype} [{numel}] {name}")


@triton.jit(do_not_specialize=["rank", "programs_per_source", "limit"])
def _pull_rows(
    BASE,
    DELTAS,
    ROUTED,
    PREFIX,
    OUT,
    SLOTS,
    rank,
    programs_per_source,
    limit,
    CAPACITY: tl.constexpr,
    WORLD: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_ROWS_POW2: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    VEC: tl.constexpr,
    COMPACT: tl.constexpr,
):
    pid = tl.program_id(0)
    source = (pid % WORLD + rank) % WORLD
    lane = pid // WORLD
    delta = tl.multiple_of(tl.load(DELTAS + source), VEC)
    src = BASE + delta
    first = source.to(tl.int64) * CAPACITY
    row_ids = tl.arange(0, BLOCK_ROWS_POW2)
    tiles = tl.cdiv(CAPACITY, BLOCK_ROWS)
    t = lane * BLOCK_ROWS + row_ids
    valid = (t < CAPACITY) & (row_ids < BLOCK_ROWS)
    flags_next = tl.load(ROUTED + first + t, mask=valid, other=0)
    if COMPACT:
        prefix_next = tl.load(PREFIX + first + t, mask=valid, other=0)
    for tile in range(lane, tiles, programs_per_source):
        t = tile * BLOCK_ROWS + row_ids
        valid = (t < CAPACITY) & (row_ids < BLOCK_ROWS)
        flags = flags_next
        # flags of this program's next tile, issued ahead of this tile's remote loads
        t_next = (tile + programs_per_source) * BLOCK_ROWS + row_ids
        valid_next = (t_next < CAPACITY) & (row_ids < BLOCK_ROWS)
        flags_next = tl.load(ROUTED + first + t_next, mask=valid_next, other=0)
        routed = (flags != 0) & valid
        src_rows = t.to(tl.int64) * WIDTH
        if COMPACT:
            prefix = prefix_next
            prefix_next = tl.load(PREFIX + first + t_next, mask=valid_next, other=0)
            # a bounded receive smaller than the routed count drops the rows past it
            routed = routed & (prefix - 1 < limit)
            dst_rows = (prefix - 1) * WIDTH
            tl.store(SLOTS + prefix - 1, first + t, mask=routed)
        else:
            dst_rows = (first + t) * WIDTH
        for c in tl.static_range(tl.cdiv(WIDTH, BLOCK_COLS)):
            cols = c * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
            col_ok = cols < WIDTH
            values = tl.load(
                src + src_rows[:, None] + cols[None, :],
                mask=routed[:, None] & col_ok[None, :],
                other=0,
            )
            if COMPACT:
                store_mask = routed[:, None] & col_ok[None, :]
            else:
                store_mask = valid[:, None] & col_ok[None, :]
            tl.store(OUT + dst_rows[:, None] + cols[None, :], values, mask=store_mask)


def pull_rows(
    sources,
    routed,
    rank,
    *,
    out=None,
    prefix=None,
    count=None,
    programs_per_sm=4,
    slots=None,
):
    """Receive rows from the peers' ``[capacity, width]`` ``sources`` (process-group rank
    order), this rank's own copy included.

    Fixed layout (``prefix`` is ``None``): returns ``[world * capacity, width]`` with row
    ``s * capacity + t`` = ``sources[s][t]`` where ``routed`` is set and a zero row
    elsewhere. Compact (``prefix`` = inclusive cumsum of ``routed``, ``count`` = its last
    value as a host int): returns ``([count, width], slots [count] int64)``, the routed rows
    in receive-row order and the receive row each one came from. A ``count`` below the
    routed count drops the rows past it; above it, the rows past the routed count stay
    unwritten (``zero_tail``).
    """
    first = sources[0]
    world, capacity, width = len(sources), first.shape[0], first.shape[1]
    if any(
        s.shape != first.shape or s.dtype != first.dtype or not s.is_contiguous()
        for s in sources
    ):
        raise ValueError("peer sources must be contiguous and share shape and dtype")
    _check_rows(routed, "routed mask", torch.bool, world * capacity)
    compact = prefix is not None
    if compact:
        _check_rows(prefix, "prefix", torch.int64, world * capacity)
        if count is None:
            raise ValueError("compact pulls need the routed count on the host")
        out = first.new_empty((count, width)) if out is None else out
        if slots is None:
            slots = torch.empty(count, dtype=torch.int64, device=first.device)
    else:
        out = first.new_empty((world * capacity, width)) if out is None else out
        slots = routed  # unused placeholder pointer
    if out.shape[1] != width or out.dtype != first.dtype or not out.is_contiguous():
        raise ValueError(
            "out must be a contiguous [rows, width] tensor like the sources"
        )
    if (compact and count == 0) or world * capacity == 0:
        return (out, slots) if compact else out
    wide = _wide_view(first)
    block_rows, block_rows_pow2, block_cols = _tile(wide.shape[1], wide.element_size())
    programs_per_source = _programs_per_source(first.device, world, programs_per_sm)
    _pull_rows[(world * programs_per_source,)](
        wide,
        peer_deltas(sources, wide.element_size()),
        routed.view(torch.int8),
        prefix if compact else routed,
        _wide_view(out),
        slots,
        rank,
        programs_per_source,
        out.shape[0],
        capacity,
        world,
        wide.shape[1],
        block_rows,
        block_rows_pow2,
        block_cols,
        max(1, 16 // wide.element_size()),
        compact,
        num_warps=8,
    )
    return (out, slots) if compact else out


@triton.jit(do_not_specialize=["limit"])
def _zero_tail(
    OUT,
    PREFIX,
    limit,
    TOTAL: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    count = tl.load(PREFIX + TOTAL - 1)
    rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    tail = (rows >= count) & (rows < limit)
    if tl.max(tail.to(tl.int32), 0) != 0:
        for c in tl.static_range(tl.cdiv(WIDTH, BLOCK_COLS)):
            cols = c * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
            mask = tail[:, None] & (cols < WIDTH)[None, :]
            zeros = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=OUT.dtype.element_ty)
            tl.store(OUT + rows[:, None] * WIDTH + cols[None, :], zeros, mask=mask)


def zero_tail(out, prefix):
    """Zero the rows of a compact ``out`` at or past the routed count ``prefix[-1]`` (read
    on device), so a receive bounded above the count needs no host count."""
    if out.shape[0] == 0:
        return out
    wide = _wide_view(out)
    _, block_rows, block_cols = _tile(wide.shape[1], wide.element_size())
    _zero_tail[(triton.cdiv(out.shape[0], block_rows),)](
        wide,
        prefix,
        out.shape[0],
        prefix.numel(),
        wide.shape[1],
        block_rows,
        block_cols,
        num_warps=8,
    )
    return out


@triton.jit(do_not_specialize=["rank", "programs_per_source"])
def _push_rows(
    PAYLOAD,
    BASE,
    DELTAS,
    ROUTED,
    BOUNDS,
    SLOTS,
    rank,
    programs_per_source,
    CAPACITY: tl.constexpr,
    WORLD: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_ROWS_POW2: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    VEC: tl.constexpr,
    COMPACT: tl.constexpr,
):
    pid = tl.program_id(0)
    dest = (pid % WORLD + rank) % WORLD
    lane = pid // WORLD
    delta = tl.multiple_of(tl.load(DELTAS + dest), VEC)
    dst = BASE + delta
    first = dest.to(tl.int64) * CAPACITY
    slot_base = rank.to(tl.int64) * CAPACITY
    row_ids = tl.arange(0, BLOCK_ROWS_POW2)
    if COMPACT:
        begin = tl.load(BOUNDS + dest)
        end = tl.load(BOUNDS + dest + 1)
    else:
        begin = first
        end = first + CAPACITY
    tiles = (end - begin + BLOCK_ROWS - 1).to(tl.int32) // BLOCK_ROWS
    for tile in range(lane, tiles, programs_per_source):
        i = begin + tile * BLOCK_ROWS + row_ids
        valid = (i < end) & (row_ids < BLOCK_ROWS)
        if COMPACT:
            slot = tl.load(SLOTS + i, mask=valid, other=0)
            active = valid
            t = slot - first
        else:
            flags = tl.load(ROUTED + i, mask=valid, other=0)
            active = (flags != 0) & valid
            t = i - first
        src_rows = i * WIDTH
        dst_rows = (slot_base + t) * WIDTH
        for c in tl.static_range(tl.cdiv(WIDTH, BLOCK_COLS)):
            cols = c * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
            mask = active[:, None] & (cols < WIDTH)[None, :]
            values = tl.load(PAYLOAD + src_rows[:, None] + cols[None, :], mask=mask)
            tl.store(dst + dst_rows[:, None] + cols[None, :], values, mask=mask)


def push_rows(payload, destinations, rows, rank, *, capacity=None, programs_per_sm=4):
    """Write payload rows into unique slots of the peers' ``[world * capacity, width]``
    ``destinations``: receive row ``s * capacity + t`` goes to
    ``destinations[s][rank * capacity + t]``. Fixed layout: ``payload`` covers every
    receive row and ``rows`` is the routed mask. Compact: ``payload[i]`` is receive row
    ``rows[i]`` (the ascending ``slots`` from ``pull_rows``); each program then streams
    one destination's run of compact rows, found with a ``searchsorted`` on the slots.
    """
    first = destinations[0]
    world, width = len(destinations), first.shape[1]
    capacity = first.shape[0] // world if capacity is None else capacity
    if first.shape[0] < (rank + 1) * capacity:
        raise ValueError("destinations do not hold this rank's slot range")
    if any(
        d.shape != first.shape or d.dtype != first.dtype or not d.is_contiguous()
        for d in destinations
    ):
        raise ValueError("destinations must be contiguous and share shape and dtype")
    if (
        payload.dtype != first.dtype
        or payload.ndim != 2
        or payload.shape[1] != width
        or not payload.is_contiguous()
    ):
        raise ValueError(
            "payload must be a contiguous [rows, width] tensor like the slots"
        )
    compact = rows.dtype == torch.int64
    if compact:
        _check_rows(rows, "slots", torch.int64, payload.shape[0])
        if payload.shape[0] == 0:
            return
        edges = torch.arange(world + 1, device=rows.device, dtype=torch.int64)
        bounds = torch.searchsorted(rows, edges * capacity)
        routed = rows
    else:
        _check_rows(rows, "routed mask", torch.bool, world * capacity)
        if payload.shape[0] != world * capacity:
            raise ValueError("fixed-layout payload must cover every receive row")
        routed = rows.view(torch.int8)
        bounds = rows
    if world * capacity == 0:
        return
    wide = _wide_view(payload)
    block_rows, block_rows_pow2, block_cols = _tile(wide.shape[1], wide.element_size())
    programs_per_source = _programs_per_source(first.device, world, programs_per_sm)
    _push_rows[(world * programs_per_source,)](
        wide,
        _wide_view(first),
        peer_deltas(destinations, wide.element_size()),
        routed,
        bounds,
        rows,
        rank,
        programs_per_source,
        capacity,
        world,
        wide.shape[1],
        block_rows,
        block_rows_pow2,
        block_cols,
        max(1, 16 // wide.element_size()),
        compact,
        num_warps=8,
    )


@triton.jit(do_not_specialize=["tokens"])
def _reduce_slots(
    BUF,
    MASK,
    OUT,
    tokens,
    CAPACITY: tl.constexpr,
    WORLD: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    valid = t < tokens
    col_ok = cols < WIDTH
    acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)
    for p in range(WORLD):
        hit = tl.load(MASK + p * tokens + t, mask=valid, other=0) != 0
        rows = (p * CAPACITY + t).to(tl.int64) * WIDTH
        values = tl.load(
            BUF + rows[:, None] + cols[None, :],
            mask=hit[:, None] & col_ok[None, :],
            other=0.0,
        )
        acc += values.to(tl.float32)
    tl.store(
        OUT + t.to(tl.int64)[:, None] * WIDTH + cols[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=valid[:, None] & col_ok[None, :],
    )


def reduce_slots(buffer, mask, world, out_dtype):
    """``out[t] = sum_p mask[p, t] ? buffer[p * capacity + t] : 0`` over the ``world``
    producer slots of this rank's ``[world * capacity, width]`` return buffer, accumulated
    in FP32 in producer order and rounded once to ``out_dtype``: bitwise what the masked
    per-producer ``total += slot.float()`` loop computes."""
    capacity, width = buffer.shape[0] // world, buffer.shape[1]
    tokens = mask.shape[1]
    if mask.dtype != torch.bool or mask.shape[0] != world or not mask.is_contiguous():
        raise ValueError("expected a contiguous bool [world, tokens] producer mask")
    if not buffer.is_contiguous() or tokens > capacity:
        raise ValueError("expected a contiguous return buffer covering the tokens")
    out = torch.empty((tokens, width), dtype=out_dtype, device=buffer.device)
    if tokens == 0 or width == 0:
        return out
    block_cols = min(triton.next_power_of_2(width), 2048)
    block_rows = max(1, min(256, 8192 // block_cols))
    grid = (triton.cdiv(tokens, block_rows), triton.cdiv(width, block_cols))
    _reduce_slots[grid](
        buffer,
        mask.view(torch.int8),
        out,
        tokens,
        capacity,
        world,
        width,
        block_rows,
        block_cols,
        num_warps=8,
    )
    return out
