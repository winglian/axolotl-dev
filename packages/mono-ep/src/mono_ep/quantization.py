# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compile-friendly E4M3 quantization with DeepGEMM's untransformed scale layout."""

import torch
import torch.nn.functional as F


def quantize_fp8(x, *, block_rows=1, block_cols=32):
    """Return E4M3 values and FP32 power-of-two dequantization scales.

    Leading dimensions are independent batches (or experts). Use 1x32 for
    MXFP8, 1x128 for Hopper activations, and 128x128 for Hopper weights.
    Scales are deliberately untransformed; GEMM adapters must apply the
    architecture's required scale layout before invoking a kernel.
    """
    if x.ndim < 2 or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("FP8 quantization expects BF16/FP32 matrices")
    if (block_rows, block_cols) not in ((1, 32), (1, 128), (128, 128)):
        raise ValueError("unsupported FP8 block geometry")
    rows, cols = x.shape[-2:]
    if cols == 0:
        raise ValueError("quantized matrices need a nonzero reduction dimension")
    if x.is_cuda and x.is_contiguous() and block_rows == 1:
        from .token_quantization import quantize_tokens

        return quantize_tokens(x, block_cols)
    if (
        x.is_cuda
        and x.ndim == 3
        and block_rows == 1
        and x.transpose(-1, -2).is_contiguous()
    ):
        from .token_quantization import quantize_transposed_weights

        return quantize_transposed_weights(x, block_cols)
    pad_rows, pad_cols = -rows % block_rows, -cols % block_cols
    padded = F.pad(x.float(), (0, pad_cols, 0, pad_rows))
    nr, nc = (rows + pad_rows) // block_rows, (cols + pad_cols) // block_cols
    blocks = padded.reshape(*x.shape[:-2], nr, block_rows, nc, block_cols)
    amax = blocks.abs().amax(dim=(-3, -1), keepdim=True).clamp_min(1e-4)
    raw_scale = amax / 448.0
    bits = raw_scale.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scales = (exponent.clamp(1, 254) << 23).view(torch.float32)
    quantized = (blocks / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
    values = quantized.reshape(padded.shape)[..., :rows, :cols].contiguous()
    return values, scales.squeeze(-1).squeeze(-2)
