# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Direct columnwise MXFP8 preparation for grouped weight-gradient GEMMs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(
    X,
    Q,
    S,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    INPUT_SCALES=None,
    S0: tl.constexpr = 0,
    S1: tl.constexpr = 0,
    PACKED: tl.constexpr = False,
    ROW_END=None,
    BOUNDED: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (row[:, None] < ROWS) & (col[None, :] < COLS)
    end = ROWS
    if BOUNDED:
        end = tl.load(ROW_END)
    scale = tl.full((BLOCK,), 2.0**-22, tl.float32)
    quantized = tl.full((32, BLOCK), 0, tl.float32)
    if tl.program_id(0) * 32 < end:
        value = tl.load(
            X + row[:, None] * COLS + col[None, :], mask & (row[:, None] < end), 0.0
        ).to(tl.float32)
        if PACKED:
            packed = tl.load(
                INPUT_SCALES + row[:, None] * S0 + (col[None, :] // 128) * S1,
                mask & (row[:, None] < end),
                0,
            )
            exponent = (packed >> (((col[None, :] // 32) % 4) * 8)) & 255
            input_scale = (exponent << 23).to(tl.float32, bitcast=True)
            value = (value * input_scale).to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(value), 0), 1.0e-4)
        bits = (amax / 448.0).to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (tl.minimum(tl.maximum(exponent, 1), 254) << 23).to(
            tl.float32, bitcast=True
        )
        quantized = tl.minimum(tl.maximum(value / scale[None, :], -448.0), 448.0)
    tl.store(Q + row[:, None] * COLS + col[None, :], quantized, mask)
    tl.store(S + tl.program_id(0) * COLS + col, scale, col < COLS)


def _check_row_end(x, row_end):
    if row_end is not None and (
        row_end.dtype != torch.int32
        or row_end.device != x.device
        or row_end.numel() != 1
        or not row_end.is_contiguous()
    ):
        raise ValueError("row_end must be one contiguous CUDA INT32 value")


def quantize_columns(x, row_end=None):
    """Return columnwise FP8; optional GPU row_end in [0,K] zeroes the tail."""
    if (
        x.ndim != 2
        or not x.is_cuda
        or not x.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError(
            "column quantization requires a contiguous CUDA BF16/FP32 matrix"
        )
    _check_row_end(x, row_end)
    rows, cols = x.shape
    values = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (triton.cdiv(rows, 32), cols), device=x.device, dtype=torch.float32
    )
    if rows and cols:
        _quantize[(triton.cdiv(rows, 32), triton.cdiv(cols, 32))](
            x,
            values,
            scales,
            rows,
            cols,
            32,
            x,
            S0=0,
            S1=0,
            PACKED=False,
            ROW_END=x if row_end is None else row_end,
            BOUNDED=row_end is not None,
            enable_fp_fusion=False,
        )
    return values, scales


def quantize_packed_columns(x, packed_scales, row_end=None):
    """Requantize with BF16 rounding; optional GPU row_end in [0,K] zeroes the tail."""
    if (
        x.ndim != 2
        or not x.is_cuda
        or not x.is_contiguous()
        or x.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or x.shape[1] == 0
        or x.shape[1] % 128
        or packed_scales.dtype != torch.int32
        or packed_scales.device != x.device
        or packed_scales.shape != (x.shape[0], x.shape[1] // 128)
    ):
        raise ValueError("Expected contiguous FP8 values and INT32[row,col/128] scales")
    _check_row_end(x, row_end)
    rows, cols = x.shape
    values = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        ((rows + 31) // 32, cols), device=x.device, dtype=torch.float32
    )
    if rows:
        _quantize[(triton.cdiv(rows, 32), triton.cdiv(cols, 32))](
            x.view(torch.float8_e4m3fn),
            values,
            scales,
            rows,
            cols,
            32,
            packed_scales,
            *packed_scales.stride(),
            PACKED=True,
            ROW_END=x if row_end is None else row_end,
            BOUNDED=row_end is not None,
            enable_fp_fusion=False,
        )
    return values, scales
