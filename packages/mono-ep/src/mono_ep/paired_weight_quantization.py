# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Original-layout gate/up FP8 weights with shared 64+64-row scales."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_pairs(
    X,
    Q,
    S,
    INNER: tl.constexpr,
    HIDDEN: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    ES: tl.constexpr,
    RS: tl.constexpr,
    CS: tl.constexpr,
):
    expert = tl.program_id(0).to(tl.int64)
    pair = tl.program_id(1)
    kb = tl.program_id(2)
    r = tl.arange(0, 128)[:, None]
    row = pair * 64 + r % 64 + (r >= 64) * INNER
    col = kb * 128 + tl.arange(0, 128)[None, :]
    value = tl.load(
        X + expert * ES + row.to(tl.int64) * RS + col.to(tl.int64) * CS, col < HIDDEN, 0
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.max(tl.abs(value), 1), 0), 1.0e-4)
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, 1), 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    quantized = tl.minimum(tl.maximum(value / scale, -448.0), 448.0)
    offset = (
        expert * (2 * INNER) * HIDDEN + row.to(tl.int64) * HIDDEN + col.to(tl.int64)
    )
    tl.store(Q + offset, quantized, col < HIDDEN)
    tl.store(
        S + expert * (INNER // 64) * K_BLOCKS + pair.to(tl.int64) * K_BLOCKS + kb, scale
    )


def quantize_paired_gate_up_weights(x):
    """Quantize original [expert, gate/up, hidden] weights without interleaving."""
    if (
        x.ndim != 3
        or not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float32)
        or x.shape[1] == 0
        or x.shape[1] % 128
        or x.shape[2] == 0
        or any(s <= 0 for s in x.stride())
    ):
        raise ValueError("Expected CUDA BF16/FP32 gate/up weights with 64-row halves")
    experts, twice_inner, hidden = x.shape
    inner = twice_inner // 2
    k_blocks = triton.cdiv(hidden, 128)
    values = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (experts, inner // 64, k_blocks), device=x.device, dtype=torch.float32
    )
    if experts:
        _quantize_pairs[(experts, inner // 64, k_blocks)](
            x,
            values,
            scales,
            inner,
            hidden,
            k_blocks,
            *x.stride(),
            enable_fp_fusion=False,
        )
    return values, scales
