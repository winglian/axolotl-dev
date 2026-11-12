# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The LoRA merge ``bf16(w + scaling * B_e @ A_e)`` per expert in one pass: each 128x128 tile reads
``w`` once, adds the rank-r product (BF16 factors, FP32 accumulation) and rounds once."""

import torch
import triton
import triton.language as tl


@triton.jit
def _merge(
    W,
    B,
    A,
    OUT,
    scaling,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    WE,
    WR,
    WC,
    RP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = tl.program_id(2).to(tl.int64)
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    k = tl.arange(0, RP)
    b = tl.load(B + e * ROWS * RP + rows[:, None] * RP + k[None, :])
    a = tl.load(A + e * RP * COLS + k[:, None] * COLS + cols[None, :])
    w = tl.load(W + e * WE + rows[:, None].to(tl.int64) * WR + cols[None, :] * WC)
    merged = w.to(tl.float32) + scaling * tl.dot(b, a)
    tl.store(
        OUT + e * ROWS * COLS + rows[:, None].to(tl.int64) * COLS + cols[None, :],
        merged.to(tl.bfloat16),
    )


def merge_triton(w, lora_b, lora_a, scaling):
    """``[E, out, in]`` BF16 merge of ``w`` with ``lora_b [E, out, r]`` and ``lora_a [E, r, in]``;
    ``out`` and ``in`` must be multiples of 128."""
    experts, rows, cols = w.shape
    rank = lora_a.shape[1]
    if rows % 128 or cols % 128:
        raise ValueError("the Triton merge needs 128-aligned expert weights")
    rp = max(16, triton.next_power_of_2(rank))
    b = torch.zeros(experts, rows, rp, device=w.device, dtype=torch.bfloat16)
    b[..., :rank] = lora_b
    a = torch.zeros(experts, rp, cols, device=w.device, dtype=torch.bfloat16)
    a[:, :rank] = lora_a
    out = torch.empty(experts, rows, cols, device=w.device, dtype=torch.bfloat16)
    if experts:
        _merge[(rows // 128, cols // 128, experts)](
            w.detach(),
            b,
            a,
            out,
            float(scaling),
            rows,
            cols,
            *w.stride(),
            rp,
            128,
            num_warps=8,
        )
    return out
