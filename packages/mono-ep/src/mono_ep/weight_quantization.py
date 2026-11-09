# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Fused transpose and 128x128 E4M3 quantization for Hopper weights."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_transposed_128x128(
    X,
    Q,
    S,
    SOURCE_ROWS: tl.constexpr,
    SOURCE_COLS: tl.constexpr,
    OUT_ROW_BLOCKS: tl.constexpr,
    OUT_COL_BLOCKS: tl.constexpr,
    EXPERT_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
):
    expert = tl.program_id(0).to(tl.int64)
    out_row_block = tl.program_id(1)
    out_col_block = tl.program_id(2)
    out_row = out_row_block * 128 + tl.arange(0, 128)[:, None]
    out_col = out_col_block * 128 + tl.arange(0, 128)[None, :]
    valid = (out_row < SOURCE_COLS) & (out_col < SOURCE_ROWS)
    input_offset = (
        expert * EXPERT_STRIDE
        + out_col.to(tl.int64) * ROW_STRIDE
        + out_row.to(tl.int64) * COL_STRIDE
    )
    value = tl.load(X + input_offset, valid, 0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.max(tl.abs(value), axis=1), axis=0), 1.0e-4)
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, 1), 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    quantized = tl.minimum(tl.maximum(value / scale, -448.0), 448.0)
    output_offset = (
        expert * SOURCE_COLS * SOURCE_ROWS
        + out_row.to(tl.int64) * SOURCE_ROWS
        + out_col.to(tl.int64)
    )
    tl.store(Q + output_offset, quantized, valid)
    scale_offset = (
        expert * OUT_ROW_BLOCKS * OUT_COL_BLOCKS
        + out_row_block.to(tl.int64) * OUT_COL_BLOCKS
        + out_col_block.to(tl.int64)
    )
    tl.store(S + scale_offset, scale)


def quantize_transposed_weights_128x128(x):
    """Transpose expert weights while quantizing 128x128 output tiles."""
    if (
        x.ndim != 3
        or not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float32)
        or x.shape[1] == 0
        or x.shape[2] == 0
        or any(stride <= 0 for stride in x.stride())
    ):
        raise ValueError("Expected dense-strided CUDA expert weights")
    experts, source_rows, source_cols = x.shape
    out_row_blocks = triton.cdiv(source_cols, 128)
    out_col_blocks = triton.cdiv(source_rows, 128)
    values = torch.empty(
        (experts, source_cols, source_rows),
        device=x.device,
        dtype=torch.float8_e4m3fn,
    )
    scales = torch.empty(
        (experts, out_row_blocks, out_col_blocks),
        device=x.device,
        dtype=torch.float32,
    )
    if experts:
        _quantize_transposed_128x128[(experts, out_row_blocks, out_col_blocks)](
            x,
            values,
            scales,
            source_rows,
            source_cols,
            out_row_blocks,
            out_col_blocks,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            enable_fp_fusion=False,
        )
    return values, scales
