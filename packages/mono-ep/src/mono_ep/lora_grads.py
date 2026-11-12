# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Rank-r LoRA adapter gradients from the persistent backward's pools in two Triton passes.

Pass one projects the pools onto the factors per 128-row block, ``P1 = dZ B1``, ``Q1 = X A1ᵀ``,
``P2 = dY B2``, ``Q2 = H A2ᵀ`` (FP32 accumulation, stored BF16 like the grouped-GEMM route);
pass two contracts over each expert's rows, ``dA1 = s P1ᵀ X``, ``dB1 = s dZᵀ Q1``,
``dA2 = s P2ᵀ H``, ``dB2 = s dYᵀ Q2`` (FP32 out, where the grouped-GEMM route stores BF16).
"""

import torch
import triton
import triton.language as tl

BLOCK_ROWS = tl.constexpr(128)


@triton.jit
def _project(
    X,
    W,
    OUT,
    STARTS,
    COUNTS,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One 128-row pool block: OUT[rows, R] = X[rows, :K] @ W[owner] with W [E, K, R]
    block = tl.program_id(0)
    experts = tl.arange(0, EXPERTS_POW2)
    starts = tl.load(STARTS + experts, experts < NUM_EXPERTS, other=2147483647)
    owner = tl.max(tl.where(starts <= block * BLOCK_ROWS, experts, -1), 0)
    start = tl.load(STARTS + tl.maximum(owner, 0))
    end = start + tl.load(COUNTS + tl.maximum(owner, 0))
    if (owner >= 0) & (block * BLOCK_ROWS < end):
        rows = block.to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid = rows < end
        r = tl.arange(0, R)
        acc = tl.zeros([BLOCK_ROWS, R], dtype=tl.float32)
        w_base = W + owner.to(tl.int64) * K * R
        for k0 in range(0, K, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(
                X + rows[:, None] * K + cols[None, :], valid[:, None], other=0.0
            )
            w = tl.load(w_base + cols[:, None] * R + r[None, :])
            acc = tl.dot(x, w, acc)
        tl.store(
            OUT + rows[:, None] * R + r[None, :], acc.to(tl.bfloat16), valid[:, None]
        )


@triton.jit
def _contract(
    X,
    Y,
    OUT,
    STARTS,
    COUNTS,
    scaling,
    K: tl.constexpr,
    R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # OUT[e, K-block, R] = scaling * X[rows of e, K-block]ᵀ @ Y[rows of e, R], FP32
    e = tl.program_id(0)
    kb = tl.program_id(1)
    start = tl.load(STARTS + e)
    count = tl.load(COUNTS + e)
    r = tl.arange(0, R)
    cols = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_K, R], dtype=tl.float32)
    for m0 in range(0, count, BLOCK_M):
        rows = start.to(tl.int64) + m0 + tl.arange(0, BLOCK_M)
        valid = m0 + tl.arange(0, BLOCK_M) < count
        x = tl.load(X + rows[:, None] * K + cols[None, :], valid[:, None], other=0.0)
        y = tl.load(Y + rows[:, None] * R + r[None, :], valid[:, None], other=0.0)
        acc = tl.dot(tl.trans(x), y, acc)
    out = OUT + e.to(tl.int64) * K * R
    tl.store(out + cols[:, None] * R + r[None, :], acc * scaling)


def _padded_rank(rank):
    return max(16, triton.next_power_of_2(rank))


def _factor(t, transpose, rp):
    """``[E, K, Rp]`` BF16 copy of a factor, the rank zero-padded to ``rp``."""
    t = t.detach().to(torch.bfloat16)
    if transpose:
        t = t.transpose(1, 2)
    out = torch.zeros(*t.shape[:2], rp, device=t.device, dtype=torch.bfloat16)
    out[..., : t.shape[2]] = t
    return out


def lora_weight_grads_triton(pools, starts, counts, a1, b1, a2, b2, scaling):
    """``dA1, dB1, dA2, dB2`` (FP32, the factors' shapes) from the BF16 ``(x, dy, h, dz)`` pools."""
    x_pool, dy_pool, h_pool, dz_pool = pools
    experts, rank = a1.shape[:2]
    hidden, inter = a1.shape[2], h_pool.shape[1]
    rp = _padded_rank(rank)
    rows = h_pool.shape[0]
    device = h_pool.device
    pow2 = triton.next_power_of_2(experts)
    w = dict(
        a1=_factor(a1, True, rp),
        b1=_factor(b1, False, rp),
        a2=_factor(a2, True, rp),
        b2=_factor(b2, False, rp),
    )
    source = dict(x=x_pool, dy=dy_pool, h=h_pool, dz=dz_pool)

    def project(name, weight, k):
        out = torch.zeros(rows, rp, device=device, dtype=torch.bfloat16)
        _project[(rows // BLOCK_ROWS,)](
            source[name],
            w[weight],
            out,
            starts,
            counts,
            experts,
            pow2,
            k,
            rp,
            64,
            num_warps=4,
        )
        return out

    def contract(name, y, k):
        out = torch.empty(experts, k, rp, device=device, dtype=torch.float32)
        _contract[(experts, k // 128)](
            source[name],
            y,
            out,
            starts,
            counts,
            float(scaling),
            k,
            rp,
            128,
            64,
            num_warps=4,
        )
        return out[..., :rank]

    p1 = project("dz", "b1", 2 * inter)
    q1 = project("x", "a1", hidden)
    p2 = project("dy", "b2", hidden)
    q2 = project("h", "a2", inter)
    da1 = contract("x", p1, hidden).transpose(1, 2)
    db1 = contract("dz", q1, 2 * inter)
    da2 = contract("h", p2, inter).transpose(1, 2)
    db2 = contract("dy", q2, hidden)
    return da1, db1, da2, db2
