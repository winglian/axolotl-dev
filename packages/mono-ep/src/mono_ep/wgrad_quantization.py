# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Fused transpose and blockwise FP8 packing for Hopper expert weight gradients."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(
    X,
    ENDS,
    VALID_ENDS,
    Q,
    S,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
    SCALE_MN_MAJOR: tl.constexpr,
):
    base = tl.program_id(0).to(tl.int64) * 128
    col = tl.program_id(1) * COL_BLOCK + tl.arange(0, COL_BLOCK)
    g = tl.arange(0, GROUP_BLOCK)
    ends = tl.load(ENDS + g, g < GROUPS, 2147483647)
    group = tl.sum((base >= ends).to(tl.int32), 0)
    valid = group < GROUPS
    start = tl.load(ENDS + group - 1, valid & (group > 0), 0).to(tl.int64)
    end = tl.load(ENDS + group, valid, 0).to(tl.int64)
    row = base + tl.arange(0, 128)
    mask = valid & (row[:, None] < end) & (col[None, :] < N)
    live_end = tl.load(VALID_ENDS + group, valid, 0)
    value = tl.load(
        X + row[:, None] * N + col[None, :], mask & (row[:, None] < live_end), 0
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(value), 0), 1.0e-4)
    raw = amax / 448.0
    bits = raw.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (tl.minimum(tl.maximum(exponent, 1), 254) << 23).to(
        tl.float32, bitcast=True
    )
    quantized = tl.minimum(tl.maximum(value / scale[None, :], -448.0), 448.0)
    offset = start * N + col[None, :] * (end - start) + row[:, None] - start
    tl.store(Q + offset, quantized, mask)
    tl.store(Q + row[:, None] * N + col[None, :], 0.0, ~valid & (col[None, :] < N))
    sf_offset = (
        (base // 128 * N + col)
        if SCALE_MN_MAJOR
        else (col.to(tl.int64) * (ROWS // 128) + base // 128)
    )
    tl.store(
        S + sf_offset,
        tl.where(valid, scale, 0),
        col < N,
    )


def quantize_transposed_groups(
    x, sizes, *, use_psum_layout=False, scales_mn_major=False, valid_ends=None
):
    """Pack zero-padded [sum(K), N] into concatenated [N, K_group] matrices.

    Sizes are multiples of 128; prefix ends may leave an unused capacity tail.
    Prefix ends must be nondecreasing and bounded by the input row capacity.
    Returned scales are [N, capacity/128] FP32 powers of two, indexed by physical
    K block. Unused capacity is zeroed without reading the input tail.
    """
    if x.ndim != 2 or not x.is_contiguous() or x.shape[0] % 128:
        raise ValueError("expected contiguous input with 128-aligned group rows")
    if (
        sizes.ndim != 1
        or not sizes.is_contiguous()
        or sizes.dtype != torch.int32
        or sizes.device != x.device
        or not sizes.numel()
    ):
        raise ValueError(
            "expected nonempty contiguous int32 group metadata on input device"
        )
    if valid_ends is not None and (
        valid_ends.shape != sizes.shape
        or valid_ends.dtype != torch.int32
        or valid_ends.device != x.device
        or not valid_ends.is_contiguous()
    ):
        raise ValueError("valid ends must match padded group metadata")
    rows, cols = x.shape
    values = torch.empty(rows * cols, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty_strided(
        (cols, rows // 128),
        (1, cols) if scales_mn_major else (rows // 128, 1),
        device=x.device,
        dtype=torch.float32,
    )
    if rows:
        ends = sizes if use_psum_layout else sizes.cumsum(0, dtype=torch.int32)
        _pack[(rows // 128, triton.cdiv(cols, 32))](
            x,
            ends,
            ends if valid_ends is None else valid_ends,
            values,
            scales,
            cols,
            rows,
            sizes.numel(),
            triton.next_power_of_2(sizes.numel()),
            32,
            scales_mn_major,
        )
    return values, scales
