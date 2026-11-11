# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""SM100 expert weight gradients from the persistent backward's pools.

The pools hold each local expert's rows from ``starts[e]`` (a multiple of 128) to
``ends[e]``, with the tail of its last 128-row block zeroed. The k-grouped GEMM
writes ``out[e] = a_pool[rows of e]^T b_pool[rows of e]`` straight into an
uninitialized FP32 output; only the slabs of experts with no rows, which it leaves
unwritten, are zeroed afterwards.
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
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        a_pool, b_pool, out, None, ends, None, use_psum_layout=True
    )
    return zero_empty_experts(out, counts)
