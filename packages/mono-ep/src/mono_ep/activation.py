# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""BF16 SwiGLU with FP32 routing weights and routing-weight gradients."""

import torch
import triton
import triton.language as tl


@triton.jit
def _activation_fp8(
    Z,
    S,
    Q,
    SCALE,
    I: tl.constexpr,
    GROUPS: tl.constexpr,
    ROWS: tl.constexpr,
    ROW_TILE: tl.constexpr,
    GROUP_TILE: tl.constexpr,
    ROW_END=None,
    BOUNDED: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64) * ROW_TILE + tl.arange(0, ROW_TILE)
    group = tl.program_id(1) * GROUP_TILE + tl.arange(0, GROUP_TILE)
    col = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = (row[:, None, None] < ROWS) & (col[None, :, :] < I)
    end = ROWS
    if BOUNDED:
        end = tl.load(ROW_END)
    scale = tl.full((ROW_TILE, GROUP_TILE), 2.0**-22, tl.float32)
    quantized = tl.full((ROW_TILE, GROUP_TILE, 32), 0, tl.float32)
    if tl.program_id(0) * ROW_TILE < end:
        load_mask = mask & (row[:, None, None] < end)
        offset = row[:, None, None] * (2 * I) + col[None, :, :]
        gate = tl.load(Z + offset, load_mask, 0).to(tl.float32)
        up = tl.load(Z + offset + I, load_mask, 0).to(tl.float32)
        score = tl.load(S + row, (row < ROWS) & (row < end), 0)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        # Preserve the BF16 intermediate rounding of the separate activation path.
        value = (
            (gate * sigmoid * up * score[:, None, None]).to(tl.bfloat16).to(tl.float32)
        )
        amax = tl.maximum(tl.max(tl.abs(value), 2), 1.0e-4)
        bits = (amax / 448.0).to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (tl.minimum(tl.maximum(exponent, 1), 254) << 23).to(
            tl.float32, bitcast=True
        )
        quantized = tl.minimum(tl.maximum(value / scale[:, :, None], -448.0), 448.0)
    tl.store(Q + row[:, None, None] * I + col[None, :, :], quantized, mask)
    tl.store(
        SCALE + row[:, None] * GROUPS + group[None, :],
        scale,
        (row[:, None] < ROWS) & (group[None, :] < GROUPS),
    )


def swiglu_fp8(z, scores, row_end=None):
    """Fuse weighted SwiGLU and FP8; optional GPU row_end in [0, rows] zeroes tail."""
    if (
        z.ndim != 2
        or z.shape[1] == 0
        or z.shape[1] % 2
        or z.dtype != torch.bfloat16
        or not z.is_cuda
        or not z.is_contiguous()
        or scores.shape != (z.shape[0],)
        or scores.dtype != torch.float32
        or scores.device != z.device
        or not scores.is_contiguous()
    ):
        raise ValueError("Expected contiguous CUDA BF16 gate/up and FP32 row scores")
    if row_end is not None and (
        row_end.device != z.device
        or row_end.dtype != torch.int32
        or row_end.numel() != 1
        or not row_end.is_contiguous()
    ):
        raise ValueError("row_end must be one contiguous CUDA INT32 value")
    rows, inner = z.shape[0], z.shape[1] // 2
    groups = triton.cdiv(inner, 32)
    values = torch.empty((rows, inner), device=z.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, groups), device=z.device, dtype=torch.float32)
    if rows:
        _activation_fp8[(triton.cdiv(rows, 4), triton.cdiv(groups, 4))](
            z,
            scores,
            values,
            scales,
            inner,
            groups,
            rows,
            4,
            4,
            z if row_end is None else row_end,
            row_end is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return values, scales


@triton.jit
def _activation(
    Z,
    S,
    H,
    DH,
    DZ,
    DS,
    I: tl.constexpr,
    BACKWARD: tl.constexpr,
    SAVE_H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    gate = tl.load(Z + row * (2 * I) + col, col < I, 0).to(tl.float32)
    up = tl.load(Z + row * (2 * I) + I + col, col < I, 0).to(tl.float32)
    score = tl.load(S + row)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    silu = gate * sigmoid
    if SAVE_H:
        tl.store(H + row * I + col, silu * up * score, col < I)
    if BACKWARD:
        dh = tl.load(DH + row * I + col, col < I, 0).to(tl.float32)
        dg = dh * score * up * (sigmoid + silu * (1.0 - sigmoid))
        du = dh * score * silu
        tl.store(DZ + row * (2 * I) + col, dg, col < I)
        tl.store(DZ + row * (2 * I) + I + col, du, col < I)
        tl.store(DS + row, tl.sum(dh * silu * up, 0))


def swiglu(z, scores, dh=None, *, cached_h=None):
    """Return weighted activation, optionally its input and score gradients."""
    rows, width = z.shape
    inner = width // 2
    if width % 2 or not z.is_contiguous() or not scores.is_contiguous():
        raise ValueError("SwiGLU needs contiguous gate/up pairs and scores")
    if scores.shape != (rows,) or scores.dtype != torch.float32:
        raise ValueError("routing weights must be an FP32 vector")
    h = z.new_empty((rows, inner)) if cached_h is None else cached_h
    dz = torch.empty_like(z) if dh is not None else None
    ds = torch.empty_like(scores) if dh is not None else None
    if dh is not None and (dh.shape != h.shape or not dh.is_contiguous()):
        raise ValueError("activation gradient has the wrong shape or stride")
    if rows:
        _activation[(rows,)](
            z,
            scores,
            h,
            dh,
            dz,
            ds,
            inner,
            dh is not None,
            cached_h is None,
            triton.next_power_of_2(inner),
            enable_fp_fusion=False,
        )
    return (h, dz, ds) if dh is not None else h
