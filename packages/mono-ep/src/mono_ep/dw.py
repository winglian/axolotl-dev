# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""SM100 expert weight gradients from the persistent backward's pools.

The pools hold each local expert's rows from ``starts[e]`` (a multiple of 128) to
``ends[e]``, with the tail of its last 128-row block zeroed. The k-grouped GEMM
writes ``out[e] = a_pool[rows of e]^T b_pool[rows of e]`` straight into an
uninitialized FP32 output; only the slabs of experts with no rows, which it leaves
unwritten, are zeroed afterwards. ``MONO_EP_SM100_DW_PRECISION=mxfp8`` (opt-in) quantizes
the pools to E4M3 with one UE8M0 scale per 32 pool rows and column and runs DeepGEMM's
k-grouped ``recipe=(1, 1, 32)`` GEMM instead.
"""

import os

import torch
import triton
import triton.language as tl


def direct_dw():
    """``MONO_EP_SM100_DW_DIRECT``: ``1`` (default) writes D directly, ``0`` accumulates into zeros."""
    value = os.getenv("MONO_EP_SM100_DW_DIRECT", "1")
    if value not in ("0", "1"):
        raise ValueError("MONO_EP_SM100_DW_DIRECT must be 0 or 1")
    return value == "1"


def dw_precision():
    value = os.getenv("MONO_EP_SM100_DW_PRECISION", "bf16")
    if value not in ("bf16", "mxfp8"):
        raise ValueError("MONO_EP_SM100_DW_PRECISION must be bf16 or mxfp8")
    return value


@triton.jit
def _quantize_k_blocks(
    X, Q, S, ENDS, NUM_EXPERTS: tl.constexpr, COLS: tl.constexpr, BLOCK_C: tl.constexpr
):
    # One 128-row block and BLOCK_C columns: E4M3 values and, per column, one packed int32
    # of four UE8M0 exponents (rows 32 j .. 32 j + 31 in byte j)
    block = tl.program_id(0)
    if block * 128 >= tl.load(ENDS + NUM_EXPERTS - 1):
        return
    row = block.to(tl.int64) * 128 + tl.arange(0, 128)
    col = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    value = tl.load(X + row[:, None] * COLS + col[None, :]).to(tl.float32)
    grouped = tl.reshape(value, (4, 32, BLOCK_C))
    amax = tl.maximum(tl.max(tl.abs(grouped), 1), 1.0e-4)
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, 1), 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    quantized = tl.minimum(tl.maximum(grouped / scale[:, None, :], -448.0), 448.0)
    tl.store(
        Q + row[:, None] * COLS + col[None, :],
        tl.reshape(quantized, (128, BLOCK_C)).to(Q.dtype.element_ty),
    )
    packed = tl.sum(exponent << (tl.arange(0, 4)[:, None] * 8), 0)
    tl.store(S + block.to(tl.int64) * COLS + col, packed)


def quantize_k_blocks(pool, ends):
    """E4M3 copy of a ``[rows, C]`` BF16 pool and ``[rows / 128, C]`` packed UE8M0 scales along rows."""
    rows, cols = pool.shape
    if rows % 128 or cols % 128 or not pool.is_contiguous():
        raise ValueError(
            "pools must be contiguous with rows and columns multiples of 128"
        )
    q = torch.empty(rows, cols, device=pool.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(rows // 128, cols, device=pool.device, dtype=torch.int32)
    _quantize_k_blocks[(rows // 128, cols // 128)](
        pool, q, s, ends, ends.numel(), cols, 128
    )
    return q, s


@triton.jit
def _zero_empty_experts(OUT, COUNTS, NUMEL: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    if tl.load(COUNTS + expert) != 0:
        return
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + expert.to(tl.int64) * NUMEL + offset, 0.0, offset < NUMEL)


def zero_empty_experts(out, counts):
    """Zero the ``[M, N]`` slabs of experts with no rows."""
    experts, numel = out.shape[0], out[0].numel()
    block = 16384
    _zero_empty_experts[(experts, triton.cdiv(numel, block))](out, counts, numel, block)
    return out


def weight_grad(a_pool, b_pool, ends, counts, experts):
    """FP32 ``[experts, A cols, B cols]`` weight gradients, written directly."""
    import deep_gemm

    out = torch.empty(
        (experts, a_pool.shape[1], b_pool.shape[1]),
        device=a_pool.device,
        dtype=torch.float32,
    )
    if dw_precision() == "mxfp8":
        deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
            quantize_k_blocks(a_pool, ends),
            quantize_k_blocks(b_pool, ends),
            out,
            None,
            ends,
            None,
            (1, 1, 32),
            "mn",
            True,
        )
    else:
        deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
            a_pool, b_pool, out, None, ends, None, use_psum_layout=True
        )
    return zero_empty_experts(out, counts)
