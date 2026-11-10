# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The five 128x128 blockwise FP8 weight forms of the SM90 persistent kernels, optionally cached.

Forward consumes ``w1_paired`` (original gate/up layout, 64+64-row scale blocks) and ``w2``;
backward consumes ``w1_interleaved`` (DeepGEMM gran-8 gate/up interleave), ``w2t`` and ``w1t``.
A paired block and the interleaved block with the same index hold the same 128x128 elements,
and a ``w2`` block and its ``w2t`` block are transposes of each other, so one pass over each
master tensor produces every form with one amax per block. All forms are bitwise equal to the
per-call quantizers.

``MONO_EP_SM90_FP8_WEIGHT_CACHE``:

* ``0``: quantize per call, as before.
* ``1`` (default): cache the forms per weight version (:mod:`mono_ep.weight_cache`); the first call
  of a step builds all five in one launch, the checkpoint recompute and the backward reuse them.
* ``forms``: cache, but build only the requested forms with the per-form kernels.

The cache follows the invalidation contract of :mod:`mono_ep.weight_cache`.

The transposed forms are quantized from a second, transposed read of the master block rather
than from ``tl.trans`` of a tile: ptxas 12.9 (the assembler Triton 3.8 uses for pre-Blackwell
targets) miscompiles the 8-bit layout conversion that a transpose of an e4m3 tile lowers to
(see ``tests/test_triton_fp8_transpose.py``). The strides of those reads are hidden from
Triton's axis analysis so the reads are not layout anchors and take the stores' layout; the
cast then happens in the layout it is stored in and no 8-bit conversion remains.
"""

import os

import torch
import triton
import triton.language as tl

FORWARD_FORMS = ("w1_paired", "w2")
BACKWARD_FORMS = ("w1_interleaved", "w2t", "w1t")
_PREFIX = "sm90_fp8:"


@triton.jit
def _e4m3_scale(amax):
    amax = tl.maximum(amax, 1.0e-4)
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, 1), 254)
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def _opaque(stride):
    # hides the stride from axis analysis: a load through it is not a layout anchor
    return tl.inline_asm_elementwise(
        "mov.b32 $0, $1;", "=r,r", [stride], dtype=tl.int32, is_pure=True, pack=1
    )


@triton.jit
def _quantize_forms(
    W1,
    W2,
    Q1P,
    Q1I,
    Q1T,
    S1P,
    S1T,
    Q2,
    Q2T,
    S2,
    S2T,
    INNER: tl.constexpr,
    HIDDEN: tl.constexpr,
    E1S: tl.constexpr,
    R1S: tl.constexpr,
    C1S: tl.constexpr,
    E2S: tl.constexpr,
    R2S: tl.constexpr,
    C2S: tl.constexpr,
):
    IB: tl.constexpr = INNER // 128
    HB: tl.constexpr = HIDDEN // 128
    block = tl.program_id(0)
    expert = tl.program_id(1).to(tl.int64)
    r = tl.arange(0, 128)
    c = tl.arange(0, 128)
    if tl.program_id(2) == 0:
        # One 128-row gate block and its up block: w1t blocks (gate, up), paired/interleaved
        # blocks 2*cb (first 64 rows of each) and 2*cb+1 (last 64 rows of each).
        cb = block // HB
        kb = block % HB
        rows = (cb * 128 + r).to(tl.int64)
        cols = (kb * 128 + c).to(tl.int64)
        base = W1 + expert * E1S + cols[None, :] * C1S
        g = tl.load(base + rows[:, None] * R1S).to(tl.float32)
        u = tl.load(base + (rows[:, None] + INNER) * R1S).to(tl.float32)
        g_rows = tl.max(tl.abs(g), 1)
        u_rows = tl.max(tl.abs(u), 1)
        first = r < 64
        s0 = _e4m3_scale(
            tl.maximum(
                tl.max(tl.where(first, g_rows, 0.0), 0),
                tl.max(tl.where(first, u_rows, 0.0), 0),
            )
        )
        s1 = _e4m3_scale(
            tl.maximum(
                tl.max(tl.where(first, 0.0, g_rows), 0),
                tl.max(tl.where(first, 0.0, u_rows), 0),
            )
        )
        s_rows = tl.where(first, s0, s1)[:, None]
        gp = tl.minimum(tl.maximum(g / s_rows, -448.0), 448.0).to(Q1P.dtype.element_ty)
        up = tl.minimum(tl.maximum(u / s_rows, -448.0), 448.0).to(Q1P.dtype.element_ty)
        out = expert * (2 * INNER) * HIDDEN + cols[None, :]
        tl.store(Q1P + out + rows[:, None] * HIDDEN, gp)
        tl.store(Q1P + out + (rows[:, None] + INNER) * HIDDEN, up)
        interleaved = (cb * 256 + (r // 8) * 16 + r % 8).to(tl.int64)[:, None]
        tl.store(Q1I + out + interleaved * HIDDEN, gp)
        tl.store(Q1I + out + (interleaved + 8) * HIDDEN, up)
        paired_scales = S1P + expert * (2 * IB) * HB + (2 * cb) * HB + kb
        tl.store(paired_scales, s0)
        tl.store(paired_scales + HB, s1)

        sg = _e4m3_scale(tl.max(g_rows, 0))
        su = _e4m3_scale(tl.max(u_rows, 0))
        base_t = (
            W1
            + expert * E1S
            + cols[:, None] * _opaque(C1S)
            + rows[None, :] * _opaque(R1S)
        )
        gT = tl.load(base_t).to(tl.float32)
        uT = tl.load(base_t + INNER * R1S).to(tl.float32)
        gt = tl.minimum(tl.maximum(gT / sg, -448.0), 448.0).to(Q1T.dtype.element_ty)
        ut = tl.minimum(tl.maximum(uT / su, -448.0), 448.0).to(Q1T.dtype.element_ty)
        t_rows = (kb * 128 + r).to(tl.int64)[:, None]
        t_cols = (cb * 128 + c).to(tl.int64)[None, :]
        t_out = Q1T + expert * HIDDEN * (2 * INNER) + t_rows * (2 * INNER) + t_cols
        tl.store(t_out, gt)
        tl.store(t_out + INNER, ut)
        t_scales = S1T + expert * HB * (2 * IB) + kb * (2 * IB) + cb
        tl.store(t_scales, sg)
        tl.store(t_scales + IB, su)
    else:
        rb = block // IB
        cb = block % IB
        rows = (rb * 128 + r).to(tl.int64)
        cols = (cb * 128 + c).to(tl.int64)
        w = tl.load(W2 + expert * E2S + rows[:, None] * R2S + cols[None, :] * C2S).to(
            tl.float32
        )
        s = _e4m3_scale(tl.max(tl.max(tl.abs(w), 1), 0))
        q = tl.minimum(tl.maximum(w / s, -448.0), 448.0).to(Q2.dtype.element_ty)
        tl.store(
            Q2 + expert * HIDDEN * INNER + rows[:, None] * INNER + cols[None, :], q
        )
        wT = tl.load(
            W2
            + expert * E2S
            + cols[:, None] * _opaque(C2S)
            + rows[None, :] * _opaque(R2S)
        ).to(tl.float32)
        qt = tl.minimum(tl.maximum(wT / s, -448.0), 448.0).to(Q2T.dtype.element_ty)
        t_rows = (cb * 128 + r).to(tl.int64)[:, None]
        t_cols = (rb * 128 + c).to(tl.int64)[None, :]
        tl.store(Q2T + expert * INNER * HIDDEN + t_rows * HIDDEN + t_cols, qt)
        tl.store(S2 + expert * HB * IB + rb * IB + cb, s)
        tl.store(S2T + expert * IB * HB + cb * HB + rb, s)


def _fusable(w1, w2):
    if w1.ndim != 3 or w2.ndim != 3 or not (w1.is_cuda and w2.is_cuda):
        return False
    experts, twice_inner, hidden = w1.shape
    return (
        twice_inner % 256 == 0
        and hidden % 128 == 0
        and w2.shape == (experts, hidden, twice_inner // 2)
        and w1.dtype in (torch.bfloat16, torch.float32)
        and w2.dtype in (torch.bfloat16, torch.float32)
        and all(s > 0 for s in (*w1.stride(), *w2.stride()))
        # the row/column strides pass through a 32-bit inline-asm move in the kernel
        and all(s < 2**31 for s in (*w1.stride()[1:], *w2.stride()[1:]))
    )


def quantize_all_forms(w1, w2):
    """All five forms in one launch; requires hidden and intermediate divisible by 128."""
    if not _fusable(w1, w2):
        raise ValueError("Fused SM90 weight forms need 128-aligned dense CUDA weights")
    experts, twice_inner, hidden = w1.shape
    inner = twice_inner // 2
    device, fp8 = w1.device, torch.float8_e4m3fn

    def scales(rows, cols):
        return torch.empty(
            (experts, rows // 128, cols // 128), device=device, dtype=torch.float32
        )

    q1p = torch.empty((experts, twice_inner, hidden), device=device, dtype=fp8)
    q1i = torch.empty_like(q1p)
    q1t = torch.empty((experts, hidden, twice_inner), device=device, dtype=fp8)
    q2 = torch.empty((experts, hidden, inner), device=device, dtype=fp8)
    q2t = torch.empty((experts, inner, hidden), device=device, dtype=fp8)
    s1p, s1t = scales(twice_inner, hidden), scales(hidden, twice_inner)
    s2, s2t = scales(hidden, inner), scales(inner, hidden)
    if experts:
        _quantize_forms[((inner // 128) * (hidden // 128), experts, 2)](
            w1,
            w2,
            q1p,
            q1i,
            q1t,
            s1p,
            s1t,
            q2,
            q2t,
            s2,
            s2t,
            inner,
            hidden,
            *w1.stride(),
            *w2.stride(),
            num_warps=8,
            enable_fp_fusion=False,
        )
    return dict(
        w1_paired=(q1p, s1p),
        w1_interleaved=(q1i, s1p),
        w1t=(q1t, s1t),
        w2=(q2, s2),
        w2t=(q2t, s2t),
    )


def quantize_form(w1, w2, name):
    """One form with the per-call quantizers (the uncached path's kernels)."""
    from .block_weight_quantization import quantize_weights_128x128
    from .paired_weight_quantization import quantize_paired_gate_up_weights
    from .weight_quantization import quantize_transposed_weights_128x128

    if name == "w1_paired":
        return quantize_paired_gate_up_weights(w1)
    if name == "w1_interleaved":
        return quantize_weights_128x128(w1, interleave_gate_up=True)
    if name == "w1t":
        return quantize_transposed_weights_128x128(w1)
    if name == "w2":
        return quantize_weights_128x128(w2)
    if name == "w2t":
        return quantize_transposed_weights_128x128(w2)
    raise ValueError(f"unknown SM90 weight form {name}")


def cache_mode():
    value = os.getenv("MONO_EP_SM90_FP8_WEIGHT_CACHE", "1")
    if value not in ("0", "1", "forms"):
        raise ValueError("MONO_EP_SM90_FP8_WEIGHT_CACHE must be 0, 1 or forms")
    return value


def cached_forms(w1, w2, names, mode):
    """``names`` forms of ``(w1, w2)`` from the weight cache (mode ``1`` or ``forms``)."""
    from .weight_cache import cached_weight_forms

    def build(missing):
        if mode == "1" and _fusable(w1, w2):
            return {_PREFIX + k: v for k, v in quantize_all_forms(w1, w2).items()}
        return {
            _PREFIX + name: quantize_form(w1, w2, name)
            for name in (n[len(_PREFIX) :] for n in missing)
        }

    return cached_weight_forms(w1, w2, [_PREFIX + n for n in names], build)
