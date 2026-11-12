# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Rank-r LoRA adapter gradients from the persistent backward's pools in two Triton passes.

Pass one projects the pools onto the factors per 128-row block, ``P1 = dZ B1``, ``Q1 = X A1ᵀ``,
``P2 = dY B2``, ``Q2 = H A2ᵀ`` (FP32 accumulation, stored BF16 like the grouped-GEMM route);
pass two contracts over each expert's rows, ``dA1 = s P1ᵀ X``, ``dB1 = s dZᵀ Q1``,
``dA2 = s P2ᵀ H``, ``dB2 = s dYᵀ Q2`` (FP32 out). The pools are either the BF16 x / dy / h / dz
pools or, for the SM100 MXFP8 backward launched without its BF16 copies, the row-wise E4M3 x / dy
/ dz pools (one UE8M0 exponent per 32 columns, four packed per int32 word per 128 columns, rows
in UTCCP order inside each 128-row block) next to the BF16 h pool. E4M3 bytes are read as int32
words and decoded with integer ops and one multiply (no 8-bit tensors in the kernel, no exp2):
each of the four byte lanes of a word is its own K slice of the dot product, so the K order is
permuted the same way on both operands and nothing is reshaped.
"""

import torch
import triton
import triton.language as tl

BLOCK_ROWS = tl.constexpr(128)


@triton.jit
def _fp8_tiles(X, SW, SF_ROWS, rows, words_col, valid, kblock, K: tl.constexpr):
    """The four BF16 ``[rows, words]`` tiles of pool columns ``4 * words_col + lane`` (E4M3
    dequantized), all columns inside 128-column block ``kblock``. An E4M3 byte placed as
    ``sign << 31 | (byte & 0x7F) << 20`` is an FP32 (subnormal for exponent 0) equal to its value
    times 2^-120; two exact power-of-two multiplies (2^120, then the block scale) decode it."""
    words = tl.load(
        X + rows[:, None] * (K // 4) + words_col[None, :], valid[:, None], other=0
    )
    r = rows % 128
    sf_row = rows - r + (r % 32) * 4 + r // 32
    word = tl.load(SW + kblock * SF_ROWS + sf_row, valid, other=0)
    shift = ((4 * words_col) % 128) // 32 * 8
    # stale words of masked rows may hold 255: keep the scale finite so 0 * scale stays 0
    exponent = tl.minimum((word[:, None] >> shift[None, :]) & 255, 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True) * 1.329227995784916e36
    b0 = words
    b1 = words >> 8
    b2 = words >> 16
    b3 = words >> 24
    x0 = (
        (((b0 & 0x7F) << 20) | ((b0 & 0x80) << 24)).to(tl.float32, bitcast=True) * scale
    ).to(tl.bfloat16)
    x1 = (
        (((b1 & 0x7F) << 20) | ((b1 & 0x80) << 24)).to(tl.float32, bitcast=True) * scale
    ).to(tl.bfloat16)
    x2 = (
        (((b2 & 0x7F) << 20) | ((b2 & 0x80) << 24)).to(tl.float32, bitcast=True) * scale
    ).to(tl.bfloat16)
    x3 = (
        (((b3 & 0x7F) << 20) | ((b3 & 0x80) << 24)).to(tl.float32, bitcast=True) * scale
    ).to(tl.bfloat16)
    return x0, x1, x2, x3


@triton.jit
def _project(
    X,
    SW,
    SF_ROWS,
    W,
    OUT,
    STARTS,
    COUNTS,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    FP8: tl.constexpr,
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
            if FP8:
                words_col = k0 // 4 + tl.arange(0, BLOCK_K // 4)
                x0, x1, x2, x3 = _fp8_tiles(
                    X, SW, SF_ROWS, rows, words_col, valid, k0 // 128, K
                )
                w_rows = w_base + 4 * words_col[:, None] * R + r[None, :]
                acc = tl.dot(x0, tl.load(w_rows), acc)
                acc = tl.dot(x1, tl.load(w_rows + R), acc)
                acc = tl.dot(x2, tl.load(w_rows + 2 * R), acc)
                acc = tl.dot(x3, tl.load(w_rows + 3 * R), acc)
            else:
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
    SW,
    SF_ROWS,
    Y,
    OUT,
    STARTS,
    COUNTS,
    scaling,
    K: tl.constexpr,
    R: tl.constexpr,
    FP8: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # OUT[e, K-block, R] = scaling * X[rows of e, K-block]ᵀ @ Y[rows of e, R], FP32
    e = tl.program_id(0)
    kb = tl.program_id(1)
    start = tl.load(STARTS + e)
    count = tl.load(COUNTS + e)
    r = tl.arange(0, R)
    out = OUT + e.to(tl.int64) * K * R
    if FP8:
        words_col = kb * (BLOCK_K // 4) + tl.arange(0, BLOCK_K // 4)
        acc0 = tl.zeros([BLOCK_K // 4, R], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_K // 4, R], dtype=tl.float32)
        acc2 = tl.zeros([BLOCK_K // 4, R], dtype=tl.float32)
        acc3 = tl.zeros([BLOCK_K // 4, R], dtype=tl.float32)
        for m0 in range(0, count, BLOCK_M):
            rows = start.to(tl.int64) + m0 + tl.arange(0, BLOCK_M)
            valid = m0 + tl.arange(0, BLOCK_M) < count
            y = tl.load(Y + rows[:, None] * R + r[None, :], valid[:, None], other=0.0)
            x0, x1, x2, x3 = _fp8_tiles(X, SW, SF_ROWS, rows, words_col, valid, kb, K)
            acc0 = tl.dot(tl.trans(x0), y, acc0)
            acc1 = tl.dot(tl.trans(x1), y, acc1)
            acc2 = tl.dot(tl.trans(x2), y, acc2)
            acc3 = tl.dot(tl.trans(x3), y, acc3)
        col = 4 * words_col[:, None]
        tl.store(out + (col + 0) * R + r[None, :], acc0 * scaling)
        tl.store(out + (col + 1) * R + r[None, :], acc1 * scaling)
        tl.store(out + (col + 2) * R + r[None, :], acc2 * scaling)
        tl.store(out + (col + 3) * R + r[None, :], acc3 * scaling)
    else:
        cols = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        acc = tl.zeros([BLOCK_K, R], dtype=tl.float32)
        for m0 in range(0, count, BLOCK_M):
            rows = start.to(tl.int64) + m0 + tl.arange(0, BLOCK_M)
            valid = m0 + tl.arange(0, BLOCK_M) < count
            x = tl.load(
                X + rows[:, None] * K + cols[None, :], valid[:, None], other=0.0
            )
            y = tl.load(Y + rows[:, None] * R + r[None, :], valid[:, None], other=0.0)
            acc = tl.dot(tl.trans(x), y, acc)
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


def lora_weight_grads_triton(
    pools, starts, counts, a1, b1, a2, b2, scaling, fp8_pools=None
):
    """``dA1, dB1, dA2, dB2`` (FP32, the factors' shapes). ``pools`` are the BF16 ``(x, dy, h, dz)``
    pools; with ``fp8_pools = (qx, qdy, qdz, sx, sdy, sdz)`` x, dy and dz come from the MXFP8
    backward's E4M3 pools instead and only ``pools[2]`` (h) is read."""
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
    if fp8_pools is not None:
        qx, qdy, qdz, sx, sdy, sdz = fp8_pools
        source = dict(
            x=(qx.view(torch.int32), sx, True),
            dy=(qdy.view(torch.int32), sdy, True),
            dz=(qdz.view(torch.int32), sdz, True),
        )
    else:
        source = dict(
            x=(x_pool, None, False),
            dy=(dy_pool, None, False),
            dz=(dz_pool, None, False),
        )
    source["h"] = (h_pool, None, False)

    def project(name, weight, k):
        x, sw, fp8 = source[name]
        out = torch.zeros(rows, rp, device=device, dtype=torch.bfloat16)
        _project[(rows // BLOCK_ROWS,)](
            x,
            sw if fp8 else x,
            sw.shape[1] if fp8 else 0,
            w[weight],
            out,
            starts,
            counts,
            experts,
            pow2,
            k,
            rp,
            fp8,
            64,
            num_warps=4,
        )
        return out

    def contract(name, y, k):
        x, sw, fp8 = source[name]
        out = torch.empty(experts, k, rp, device=device, dtype=torch.float32)
        _contract[(experts, k // 128)](
            x,
            sw if fp8 else x,
            sw.shape[1] if fp8 else 0,
            y,
            out,
            starts,
            counts,
            float(scaling),
            k,
            rp,
            fp8,
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
