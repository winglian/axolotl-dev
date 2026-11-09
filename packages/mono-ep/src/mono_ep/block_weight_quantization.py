# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-kernel 128x128 blockwise FP8 weight quantization.

Bitwise equal to ``quantize_fp8(w, block_rows=128, block_cols=128)`` and to
``quantize_fp8(_interleave_weights(w1), block_rows=128, block_cols=128)``, which run as about
ten eager kernels (pad copy, abs, amax, scale, divide, clamp, convert, interleave copy).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_blocks(
    X,
    Q,
    S,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    INNER: tl.constexpr,
    ROW_BLOCKS: tl.constexpr,
    COL_BLOCKS: tl.constexpr,
    ES: tl.constexpr,
    RS: tl.constexpr,
    CS: tl.constexpr,
    INTERLEAVE: tl.constexpr,
):
    expert = tl.program_id(0).to(tl.int64)
    rb = tl.program_id(1)
    cb = tl.program_id(2)
    r = tl.arange(0, 128)[:, None]
    out_row = rb * 128 + r
    if INTERLEAVE:
        # deep_gemm.mega._interleave_weights (gran 8): [gate 0..7, up 0..7, gate 8..15, ...]
        src_row = (out_row // 16) * 8 + out_row % 8 + ((out_row % 16) >= 8) * INNER
    else:
        src_row = out_row
    col = cb * 128 + tl.arange(0, 128)[None, :]
    mask = (out_row < ROWS) & (col < COLS)
    value = tl.load(
        X + expert * ES + src_row.to(tl.int64) * RS + col.to(tl.int64) * CS, mask, 0
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.max(tl.abs(value), 1), 0), 1.0e-4)
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, 1), 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    quantized = tl.minimum(tl.maximum(value / scale, -448.0), 448.0)
    offset = expert * ROWS * COLS + out_row.to(tl.int64) * COLS + col.to(tl.int64)
    tl.store(Q + offset, quantized, mask)
    tl.store(S + expert * ROW_BLOCKS * COL_BLOCKS + rb * COL_BLOCKS + cb, scale)


def quantize_weights_128x128(x, *, interleave_gate_up=False):
    """Return contiguous E4M3 [experts, rows, cols] values and FP32 [experts, rows/128, cols/128] scales.

    With interleave_gate_up the rows are first reordered like deep_gemm.mega._interleave_weights.
    """
    if (
        x.ndim != 3
        or not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float32)
        or x.shape[1] == 0
        or x.shape[2] == 0
        or any(s <= 0 for s in x.stride())
    ):
        raise ValueError("Expected dense-strided CUDA BF16/FP32 expert weights")
    experts, rows, cols = x.shape
    if interleave_gate_up and rows % 16:
        raise ValueError("Interleaved gate/up weights need rows divisible by 16")
    row_blocks, col_blocks = triton.cdiv(rows, 128), triton.cdiv(cols, 128)
    values = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (experts, row_blocks, col_blocks), device=x.device, dtype=torch.float32
    )
    if experts:
        _quantize_blocks[(experts, row_blocks, col_blocks)](
            x,
            values,
            scales,
            rows,
            cols,
            rows // 2,
            row_blocks,
            col_blocks,
            *x.stride(),
            interleave_gate_up,
            enable_fp_fusion=False,
        )
    return values, scales
