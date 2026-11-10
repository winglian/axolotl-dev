# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Triton packed-output requant of merged NVFP4 LoRA experts.

Inputs are the frozen packed base ``u8 [E, N, K/2]`` + e4m3 block scales + the
checkpoint's fixed per-``(expert, projection)`` per-tensor scales, plus the
rank-major PEFT factors ``A [E*r, K]`` / ``B [N, r*E]`` and ``scaling``. The
tile is dequantized, rounded to ``round_dtype``, the fp32 delta is added and
rounded again (the merge writer's op order), then quantized on the same
per-tensor scales; nothing here computes a scale from an amax.

Outputs per expert: packed codes ``u8 [N, K/2]``, block scales e4m3 ``[N, K/16]``
row-major, and the same scales in the 128x4 swizzled SF-atom layout.

Numerics are those of :mod:`nvfp4_requant` (torchao by default, ModelOpt in
``modelopt_compat``). Every division is ``div_rn`` and the kernel is launched
with ``enable_fp_fusion=False``. Nibbles are packed along K within a row. The
kernel never materializes an 8-bit tensor: codes and block scales are loaded
and stored as int32 words (four bytes each, which needs ``K % 64 == 0``) and
unpacked / repacked with shifts, so every layout conversion the compiler
inserts is on an int32 or fp32 tensor. ptxas 12.9 (Triton 3.8 on sm_90)
miscompiles 8-bit ``convert_layout`` ops fed by fresh e4m3 casts, and the GPU
tests assert that none appears in any compiled instance's TTGIR.
"""

from __future__ import annotations

import functools

import torch

from .nvfp4_requant import (
    DELTA_KERNELS,
    SF_VEC_SIZE,
    RequantBuffers,
    RequantNumerics,
    RequantResult,
    atom_grid,
)
from .triton_nvfp4 import _cdiv, triton_available

__all__ = [
    "RequantBuffers",
    "requant_lora_packed",
    "triton_available",
]

# ``bf16x3`` on bf16 LoRA factors issues one tensor-core dot instead of three
# (the split's low parts are exactly zero); bytes are identical either way.
BF16_EXACT_FASTPATH = True


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _e2m1_value(m):
        # m in [0, 8): (0, 0.5, 1, 1.5, 2, 3, 4, 6). Selects only: an exp2-based
        # form came back negated for a fraction of low nibbles in some warp
        # configurations (same bytes under ptxas 12.9 and 13.x, so the PTX
        # itself was wrong), with no change to the TTGIR of the byte path.
        val = tl.where(m == 1, 0.5, 0.0)
        val = tl.where(m == 2, 1.0, val)
        val = tl.where(m == 3, 1.5, val)
        val = tl.where(m == 4, 2.0, val)
        val = tl.where(m == 5, 3.0, val)
        val = tl.where(m == 6, 4.0, val)
        return tl.where(m == 7, 6.0, val)

    @triton.jit
    def _e4m3_rne_normal(s):
        # RNE of a positive fp32 in [2**-6, 448] onto e4m3, returned as fp32.
        # Bit-identical to torch's `.to(float8_e4m3fn)` on every arch (no fp8 cvt).
        b = s.to(tl.int32, bitcast=True)
        mant_odd = (b >> 20) & 1
        b = b + 524287 + mant_odd  # 2**19 - 1 magic adder -> round to nearest even
        b = (b >> 20) << 20  # keep sign + exponent + 3 mantissa bits
        return tl.minimum(b.to(tl.float32, bitcast=True), 448.0)

    @triton.jit
    def _e4m3_rne_full(s):
        # As above, plus the subnormal range [2**-9, 2**-6): fixed spacing 2**-9,
        # so round s * 512 to the nearest even integer with the 2**23 magic add.
        normal = _e4m3_rne_normal(tl.maximum(s, 0.015625))
        t = s * 512.0
        n = (t + 8388608.0) - 8388608.0
        return tl.where(s < 0.015625, n * 0.001953125, normal)

    @triton.jit
    def _e4m3_from_bits(b):
        # e4m3 bit pattern (int32, sign clear) -> exact fp32 value
        exp4 = (b >> 3) & 0xF
        mant = b & 7
        normal = (((exp4 + 120) << 23) | (mant << 20)).to(tl.float32, bitcast=True)
        return tl.where(exp4 == 0, mant.to(tl.float32) * 0.001953125, normal)

    @triton.jit
    def _e4m3_bits(sdec):
        # fp32 holding an exact e4m3 value in [0, 448] -> its e4m3 bit pattern
        b = sdec.to(tl.int32, bitcast=True)
        exp32 = (b >> 23) & 0xFF
        mant3 = (b >> 20) & 7
        normal = ((exp32 - 120) << 3) | mant3
        sub = (sdec * 512.0).to(tl.int32)
        bits = tl.where(sdec >= 0.015625, normal, sub)
        # int32 on purpose: the 8-bit narrowing happens after the last reshape
        return tl.where(sdec == 0.0, 0, bits)

    @triton.jit
    def _e2m1_index(a):
        # RNE onto (0, .5, 1, 1.5, 2, 3, 4, 6): ties to the even code
        return (
            (a > 0.25).to(tl.int32)
            + (a >= 0.75).to(tl.int32)
            + (a > 1.25).to(tl.int32)
            + (a >= 1.75).to(tl.int32)
            + (a > 2.5).to(tl.int32)
            + (a >= 3.5).to(tl.int32)
            + (a > 5.0).to(tl.int32)
        )

    @triton.jit
    def _requant_kernel(
        q_ptr,  # u8 [E, N, K/2] packed base, as int32 words
        s_ptr,  # u8 [E, N, K/16] e4m3 bits of the base, as int32 words
        dq_pts_ptr,  # fp32 [E, P] base per-tensor scales
        a_ptr,  # [E*R, K] rank-major LoRA A
        b_ptr,  # [N, R*E] rank-major LoRA B
        scaling,
        fq_pts_ptr,  # fp32 [E, P] output per-tensor scales
        oq_ptr,  # u8 [E_local, N, K/2], as int32 words
        os_ptr,  # u8 [E_local, N, K/16], as int32 words
        osfa_ptr,  # u8 [E_local, RM, RK, 512], as int32 words
        e0,
        num_experts,
        N,
        rows_per_proj,
        P,
        RM,
        RK,
        half_n,
        K: tl.constexpr,
        R: tl.constexpr,
        R_PAD: tl.constexpr,
        ROUND_BF16: tl.constexpr,
        HAS_DQ_PTS: tl.constexpr,
        HAS_FQ_PTS: tl.constexpr,
        MODELOPT_COMPAT: tl.constexpr,
        REUSE_SCALE: tl.constexpr,
        DELTA_MODE: tl.constexpr,
        WRITE_SFA: tl.constexpr,
        INTERLEAVE_ROWS: tl.constexpr,
        BM: tl.constexpr,
        BK: tl.constexpr,
    ):
        local_e = tl.program_id(2)
        e = e0 + local_e
        tl.static_assert(K % 64 == 0, "K must be a multiple of 64")
        tl.static_assert(BK % 64 == 0, "BK must be a multiple of 64")
        NB: tl.constexpr = BK // 16  # blocks per tile
        NW: tl.constexpr = BK // 64  # scale words per tile (4 blocks each)
        # No 8-bit tensor exists in this kernel: codes and scales are loaded and
        # stored as int32 words and unpacked / repacked with shifts, so every
        # layout conversion the compiler inserts is on an int32 or fp32 tensor.
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        mask_r = rows < N
        blk = tl.program_id(1) * NB + tl.arange(0, NB)  # scale-factor column
        mask_b = blk < K // 16
        grp = tl.program_id(1) * NW + tl.arange(0, NW)  # 4-block group = 1 word
        mask_g = grp < K // 64
        jw = tl.arange(0, 2)  # code words per block
        sh4 = tl.arange(0, 4) * 8  # bit offset of each byte in a word
        mask_w = tl.broadcast_to(
            mask_r[:, None, None] & mask_b[None, :, None], (BM, NB, 2)
        )
        mask_s = mask_r[:, None] & mask_g[None, :]
        src_row = e.to(tl.int64) * N + rows
        if INTERLEAVE_ROWS:
            # concat [gate; up] row n -> the gated engine's interleaved row
            # (2n for gate, 2(n - N/2) + 1 for up); the output is then the
            # engine's B operand and SF atoms as they are, no repack
            drow = tl.where(rows < half_n, 2 * rows, 2 * (rows - half_n) + 1)
        else:
            drow = rows
        dst_row = local_e.to(tl.int64) * N + drow
        proj = tl.minimum(rows // rows_per_proj, P - 1)

        cols = tl.program_id(1) * BK + tl.arange(0, BK)
        mask_c = cols < K
        rr = tl.arange(0, R_PAD)
        mask_rr = rr < R
        bt = tl.load(
            b_ptr + rows[:, None] * (R * num_experts) + rr[None, :] * num_experts + e,
            mask=mask_r[:, None] & mask_rr[None, :],
            other=0.0,
        )
        at = tl.load(
            a_ptr + (e * R + rr)[:, None].to(tl.int64) * K + cols[None, :],
            mask=mask_rr[:, None] & mask_c[None, :],
            other=0.0,
        )
        if DELTA_MODE == 0:
            # IEEE fp32: the merge writer's fp32 matmul grid, no tensor-core rounding
            delta = tl.dot(bt.to(tl.float32), at.to(tl.float32), input_precision="ieee")
        elif DELTA_MODE == 2:
            # bf16x3 with bf16 operands: both low parts are exactly zero, and
            # accumulating two all-zero dots into a zero accumulator leaves it
            # bit-identical, so only the hi*hi dot is issued
            delta = tl.dot(bt.to(tl.bfloat16), at.to(tl.bfloat16))
        else:
            # split-bf16x3: hi*hi + hi*lo + lo*hi on tensor cores, small terms first
            b32 = bt.to(tl.float32)
            a32 = at.to(tl.float32)
            b_hi = b32.to(tl.bfloat16)
            b_lo = (b32 - b_hi.to(tl.float32)).to(tl.bfloat16)
            a_hi = a32.to(tl.bfloat16)
            a_lo = (a32 - a_hi.to(tl.float32)).to(tl.bfloat16)
            delta = tl.dot(b_lo, a_hi)
            delta = tl.dot(b_hi, a_lo, acc=delta)
            delta = tl.dot(b_hi, a_hi, acc=delta)
        # K index 2p (low nibble) / 2p + 1 (high nibble)
        d_lo, d_hi = tl.split(tl.reshape(delta * scaling, (BM, NB, 8, 2)))

        # packed codes: 2 words per block per row, byte j = word j//4 >> 8*(j%4)
        wq = tl.load(
            q_ptr
            + src_row[:, None, None] * (K // 8)
            + (blk * 2)[None, :, None]
            + jw[None, None, :],
            mask=mask_w,
            other=0,
        )
        b = tl.reshape(
            (wq[:, :, :, None] >> sh4[None, None, None, :]) & 0xFF, (BM, NB, 8)
        )
        lo = b & 0x0F
        hi = (b >> 4) & 0x0F
        v_lo = _e2m1_value(lo & 7) * tl.where(lo >= 8, -1.0, 1.0)
        v_hi = _e2m1_value(hi & 7) * tl.where(hi >= 8, -1.0, 1.0)
        # block scales: 1 word per 4 blocks per row
        sw = tl.load(
            s_ptr + src_row[:, None] * (K // 64) + grp[None, :], mask=mask_s, other=0
        )
        s_base = _e4m3_from_bits(
            tl.reshape((sw[:, :, None] >> sh4[None, None, :]) & 0xFF, (BM, NB))
        )  # [BM, NB] e4m3 values
        sc = s_base
        if HAS_DQ_PTS:
            dqp = tl.load(dq_pts_ptr + e * P + proj, mask=mask_r, other=1.0)
            sc = sc * dqp[:, None]
        sc3 = tl.broadcast_to(sc[:, :, None], (BM, NB, 8))
        if ROUND_BF16:
            w_lo = (v_lo * sc3).to(tl.bfloat16).to(tl.float32)
            w_hi = (v_hi * sc3).to(tl.bfloat16).to(tl.float32)
        else:
            w_lo = v_lo * sc3
            w_hi = v_hi * sc3

        w_lo = w_lo + d_lo
        w_hi = w_hi + d_hi
        if ROUND_BF16:
            w_lo = w_lo.to(tl.bfloat16).to(tl.float32)
            w_hi = w_hi.to(tl.bfloat16).to(tl.float32)

        amax = tl.max(tl.maximum(tl.abs(w_lo), tl.abs(w_hi)), axis=2)  # [BM, NB]
        if HAS_FQ_PTS:
            prow = tl.load(fq_pts_ptr + e * P + proj, mask=mask_r, other=1.0)
            pblk = tl.broadcast_to(prow[:, None], (BM, NB))
        if REUSE_SCALE:
            # the shipped block scales, kept unless the merged block outgrows them
            if HAS_FQ_PTS:
                hp_base = pblk * s_base
            else:
                hp_base = s_base
            outgrown = amax > 6.0 * hp_base
        if MODELOPT_COMPAT:
            # ModelOpt: amax / (6 * ws2), zero blocks -> 1.0, clamp [2**-9, 448],
            # true division by fp32(ws) * ws2, sign from x < 0
            if HAS_FQ_PTS:
                bs = tl.math.div_rn(amax, 6.0 * pblk)
            else:
                bs = tl.math.div_rn(amax, tl.full(amax.shape, 6.0, tl.float32))
            bs = tl.where(bs == 0.0, 1.0, bs)
            s = tl.minimum(tl.maximum(bs, 0.001953125), 448.0)
            sdec = _e4m3_rne_full(s)
            if REUSE_SCALE:
                sdec = tl.where(outgrown, sdec, s_base)
            if HAS_FQ_PTS:
                total = sdec * pblk
            else:
                total = sdec
            t3 = tl.broadcast_to(total[:, :, None], (BM, NB, 8))
            q_lo = tl.math.div_rn(w_lo, t3)
            q_hi = tl.math.div_rn(w_hi, t3)
            sign_lo = (q_lo < 0).to(tl.int32)
            sign_hi = (q_hi < 0).to(tl.int32)
        else:
            # torchao: amax * (1/6) (torch divides by a python scalar as a reciprocal
            # multiply), / pts, clamp [2**-6, 448], encode with (1/pts) / scale,
            # sign from the fp32 sign bit (-0.0 keeps code 8)
            bs = amax * (1.0 / 6.0)
            if HAS_FQ_PTS:
                bs = tl.math.div_rn(bs, pblk)
            s = tl.minimum(tl.maximum(bs, 0.015625), 448.0)
            sdec = _e4m3_rne_normal(s)
            if REUSE_SCALE:
                sdec = tl.where(outgrown, sdec, s_base)
            if HAS_FQ_PTS:
                r = tl.math.div_rn(tl.math.div_rn(1.0, pblk), sdec)
            else:
                r = tl.math.div_rn(1.0, sdec)
            r3 = tl.broadcast_to(r[:, :, None], (BM, NB, 8))
            q_lo = tl.minimum(tl.maximum(w_lo * r3, -6.0), 6.0)
            q_hi = tl.minimum(tl.maximum(w_hi * r3, -6.0), 6.0)
            sign_lo = (q_lo.to(tl.int32, bitcast=True) >> 31) & 1
            sign_hi = (q_hi.to(tl.int32, bitcast=True) >> 31) & 1
        code_lo = _e2m1_index(tl.abs(q_lo)) | (sign_lo << 3)
        code_hi = _e2m1_index(tl.abs(q_hi)) | (sign_hi << 3)
        # low nibble first along K; 4 bytes are packed into one word with shifts
        packed = code_lo | (code_hi << 4)
        words = tl.sum(
            tl.reshape(packed, (BM, NB, 2, 4)) << sh4[None, None, None, :], axis=3
        )
        tl.store(
            oq_ptr
            + dst_row[:, None, None] * (K // 8)
            + (blk * 2)[None, :, None]
            + jw[None, None, :],
            words,
            mask=mask_w,
        )

        swords = tl.sum(
            tl.reshape(_e4m3_bits(sdec), (BM, NW, 4)) << sh4[None, None, :], axis=2
        )  # [BM, NW]
        tl.store(
            os_ptr + dst_row[:, None] * (K // 64) + grp[None, :], swords, mask=mask_s
        )
        if WRITE_SFA:
            # 128x4 atom: row r, sf col c -> atom (r//128, c//4), byte
            # (r%32)*16 + ((r//32)%4)*4 + c%4; a group's 4 blocks are one word
            atom = (local_e.to(tl.int64) * RM + drow // 128)[:, None] * RK + grp[
                None, :
            ]
            inner = ((drow % 32) * 16 + ((drow // 32) % 4) * 4)[:, None]
            tl.store(osfa_ptr + atom * 128 + inner // 4, swords, mask=mask_s)

    return _requant_kernel


@functools.lru_cache(maxsize=1)
def _get_kernel():
    return _kernels()


def _pts_table(pts: torch.Tensor | None, num_experts: int, like: torch.Tensor):
    """``None`` | ``[E]`` | ``[E, P]`` | ``[E, 1, 1]`` -> (fp32 ``[E, P]`` contiguous, P, has)."""
    if pts is None:
        return like.new_zeros(1, dtype=torch.float32), 1, False
    t = pts.detach().to(torch.float32).reshape(num_experts, -1).contiguous()
    return t, int(t.shape[1]), True


def _launch(
    *,
    numerics: RequantNumerics,
    e0: int,
    e_local: int,
    num_experts: int,
    n_local: int,
    k: int,
    rows_per_proj: int,
    out_pts,
    write_atoms: bool,
    device,
    lora,
    block=(64, 64),
    num_warps: int = 4,
    out: RequantBuffers | None = None,
    interleave_rows: bool = False,
) -> RequantResult:
    kernel = _get_kernel()
    assert k % 64 == 0, f"K={k} must be a multiple of 64 (4 scale bytes per word)"
    sf_k = k // SF_VEC_SIZE
    rm, rk = atom_grid(n_local, sf_k)
    like = torch.empty(0, device=device)
    fq_pts, p_out, has_fq = _pts_table(out_pts, num_experts, like)
    if has_fq:
        assert rows_per_proj * p_out >= n_local, (
            f"{p_out} projections x {rows_per_proj} rows do not cover {n_local} rows"
        )
    if interleave_rows:
        assert n_local % 2 == 0, "interleaved rows need an even [gate; up] row count"
    if out is None:
        out = RequantBuffers.allocate(
            e_local, n_local, k, device, with_atoms=write_atoms
        )
    else:
        out.check(e_local, n_local, k)
    qdata, scale, atoms = out.qdata, out.scale, out.atoms
    if not write_atoms:
        atoms = None
    bm, bk = block
    assert bk % 64 == 0, f"block K {bk} must be a multiple of 64"
    zero = like.new_zeros(1, dtype=torch.int32)
    delta_mode = DELTA_KERNELS.index(numerics.delta_kernel)
    q, s, dq_pts, has_dq, a, b, scaling, r = lora
    r_pad = max(16, 1 << (r - 1).bit_length())
    if (
        delta_mode == 1
        and BF16_EXACT_FASTPATH
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
    ):
        delta_mode = 2
    grid = (_cdiv(n_local, bm), _cdiv(k, bk), e_local)
    # all byte tensors are passed as int32 words (K % 64 == 0 keeps rows aligned)
    kernel[grid](
        q.view(torch.int32),
        s.view(torch.int32),
        dq_pts,
        a,
        b,
        float(scaling),
        fq_pts,
        qdata.view(torch.int32),
        scale.view(torch.int32),
        atoms.view(torch.int32) if atoms is not None else zero,
        e0,
        num_experts,
        n_local,
        rows_per_proj,
        p_out,
        rm,
        rk,
        n_local // 2,
        K=k,
        R=r,
        R_PAD=r_pad,
        ROUND_BF16=numerics.round_dtype == torch.bfloat16,
        HAS_DQ_PTS=has_dq,
        HAS_FQ_PTS=has_fq,
        MODELOPT_COMPAT=numerics.modelopt_compat,
        REUSE_SCALE=numerics.scale_mode == "reuse",
        DELTA_MODE=delta_mode,
        WRITE_SFA=atoms is not None,
        INTERLEAVE_ROWS=interleave_rows,
        BM=bm,
        BK=bk,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    pts_local = fq_pts[e0 : e0 + e_local].clone() if has_fq else None
    return RequantResult(
        qdata,
        scale.view(torch.float8_e4m3fn),
        atoms,
        pts_local,
        numerics,
        e0,
        row_order="interleaved" if interleave_rows else "concat",
    )


def requant_lora_packed(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    base_pts: torch.Tensor | None,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
    out_pts: torch.Tensor | None,
    e0: int = 0,
    e1: int | None = None,
    *,
    numerics: RequantNumerics | None = None,
    write_atoms: bool = True,
    block: tuple[int, int] = (64, 64),
    num_warps: int = 4,
    out: RequantBuffers | None = None,
    interleave_rows: bool = False,
) -> RequantResult:
    """Packed ``Q(round(DQ(base) + scaling * B_e @ A_e))`` for experts ``e0:e1``.

    ``qdata u8 [E, N, K/2]``, ``scale`` e4m3 ``[E, N, K/16]`` (both contiguous,
    ``K % 64 == 0``),
    ``base_pts`` None | ``[E]`` | ``[E, P]`` | ``[E, 1, 1]`` (the base's own
    per-tensor scales, per projection if the loader kept them apart),
    ``lora_A [E*r, K]`` / ``lora_B [N, r*E]`` rank-major, ``out_pts`` the
    output per-(expert, projection) scales ``[E, P]`` (or None, single level).
    Same numerics as ``merge_aware_snap_triton`` followed by a packed quantize,
    so ``dequant(result)`` equals the snapped BF16 weight the trainer uses.

    ``out`` writes into caller-owned buffers (``RequantBuffers`` of ``e1 - e0``
    experts; the atom buffer's padding must already be zero) instead of
    allocating. ``interleave_rows`` stores concat ``[gate; up]`` row ``n`` at
    interleaved row ``2n`` / ``2(n - N/2) + 1`` in all three outputs, the row
    order of the gated SM100 engine (``GroupedNvfp4Gemm(concat_b=True)``), so
    the codes and atoms are the engine's operands without a repack; the
    per-projection scales still follow the concat rows.
    """
    numerics = numerics or RequantNumerics()
    E, N, k2 = qdata.shape
    K = 2 * k2
    e1 = E if e1 is None else e1
    assert scale.shape == (E, N, K // SF_VEC_SIZE), tuple(scale.shape)
    assert lora_A.shape[0] % E == 0 and lora_A.shape[1] == K, tuple(lora_A.shape)
    r = lora_A.shape[0] // E
    assert lora_B.shape == (N, r * E), tuple(lora_B.shape)
    q = qdata.view(torch.uint8).contiguous()
    s = scale.view(torch.uint8).contiguous()
    dq_pts, p_dq, has_dq = _pts_table(base_pts, E, q)
    p_out = 1 if out_pts is None else out_pts.reshape(E, -1).shape[1]
    if has_dq and p_dq != p_out and out_pts is not None:
        raise ValueError(
            f"base per-tensor scales have {p_dq} projections, output has {p_out}"
        )
    p = max(p_dq, p_out)
    assert N % p == 0, f"N={N} not divisible by {p} projections"
    if numerics.scale_mode == "reuse":
        same = (out_pts is None and base_pts is None) or (
            out_pts is not None
            and base_pts is not None
            and (
                out_pts is base_pts or torch.equal(_pts_table(out_pts, E, q)[0], dq_pts)
            )
        )
        if not same:
            raise ValueError(
                "scale_mode='reuse' keeps the base block scales, so the output "
                "per-tensor scales must equal the base per-tensor scales"
            )
    return _launch(
        numerics=numerics,
        e0=e0,
        e_local=e1 - e0,
        num_experts=E,
        n_local=N,
        k=K,
        rows_per_proj=N // p,
        out_pts=out_pts,
        write_atoms=write_atoms,
        device=q.device,
        lora=(
            q,
            s,
            dq_pts,
            has_dq,
            lora_A.contiguous(),
            lora_B.contiguous(),
            scaling,
            r,
        ),
        block=block,
        num_warps=num_warps,
        out=out,
        interleave_rows=interleave_rows,
    )
