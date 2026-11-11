# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""One-read builder of the SM100 MXFP8 backward weight forms.

Each 128 x 128 tile of W1 is read once and written as the interleaved row-wise form (scales along
H) and the transposed form (scales along 2I); each tile of W2 as the transposed form. Values and
packed UE8M0 scale words are bitwise identical to :func:`mono_ep.token_quantization.prepare_persistent_mxfp8`.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _ue8m0_exponent(amax):
    bits = (tl.maximum(amax, 1.0e-4) / 448.0).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return tl.minimum(tl.maximum(exponent, 1), 254)


@triton.jit
def _mx_weight_tiles(
    W,
    QR,
    SR,
    QT,
    ST,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ROW_FORM: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    T_FORM: tl.constexpr,
    PASS_ROWS: tl.constexpr,
    RECIPROCAL: tl.constexpr = False,
):
    # One [128 rows, 128 cols] tile of W[e] ([E, ROWS, COLS]), in four 32-row passes.
    # Row form: QR [E, ROWS, COLS] with one scale per row and 32 columns, words [E][COLS / 128][ROWS].
    # Transposed form: QT [E, COLS, ROWS] with one scale per column and 32 rows, words
    # [E][ROWS / 128][COLS]. Word rows follow the UTCCP order the kernel's scale loaders read.
    e = tl.program_id(0).to(tl.int64)
    rb = tl.program_id(1)
    cb = tl.program_id(2)
    col = cb * 128 + tl.arange(0, 128)
    base = e * ROWS * COLS
    if T_FORM:
        packed_t = tl.zeros([128], dtype=tl.int32)
    for j in tl.static_range(128 // PASS_ROWS):
        r = rb * 128 + j * PASS_ROWS + tl.arange(0, PASS_ROWS)
        value = tl.load(W + base + r[:, None] * COLS + col[None, :]).to(tl.float32)
        if ROW_FORM:
            groups = tl.reshape(value, [PASS_ROWS, 4, 32])
            exponent = _ue8m0_exponent(tl.max(tl.abs(groups), 2))
            if RECIPROCAL:
                # Power-of-two scales: multiplying by the exact inverse rounds like the division
                inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
                scaled = groups * inverse[:, :, None]
            else:
                scale = (exponent << 23).to(tl.float32, bitcast=True)
                scaled = groups / scale[:, :, None]
            quantized = tl.reshape(
                tl.minimum(tl.maximum(scaled, -448.0), 448.0), [PASS_ROWS, 128]
            )
            if INTERLEAVE:
                out_row = (r % (ROWS // 2) // 8) * 16 + (r // (ROWS // 2)) * 8 + r % 8
            else:
                out_row = r
            tl.store(
                QR + base + out_row[:, None] * COLS + col[None, :],
                quantized.to(QR.dtype.element_ty),
            )
            word = tl.sum(exponent << (tl.arange(0, 4) * 8)[None, :], 1)
            sf_row = (out_row // 128) * 128 + (out_row % 32) * 4 + (out_row % 128) // 32
            tl.store(SR + (e * (COLS // 128) + cb) * ROWS + sf_row, word)
        if T_FORM:
            row_groups = tl.reshape(value, [PASS_ROWS // 32, 32, 128])
            exponent_t = _ue8m0_exponent(tl.max(tl.abs(row_groups), 1))
            if RECIPROCAL:
                inverse_t = ((254 - exponent_t) << 23).to(tl.float32, bitcast=True)
                scaled_t = row_groups * inverse_t[:, None, :]
            else:
                scale_t = (exponent_t << 23).to(tl.float32, bitcast=True)
                scaled_t = row_groups / scale_t[:, None, :]
            quantized_t = tl.reshape(
                tl.minimum(tl.maximum(scaled_t, -448.0), 448.0), [PASS_ROWS, 128]
            )
            tl.store(
                QT + e * COLS * ROWS + col[:, None] * ROWS + r[None, :],
                tl.trans(quantized_t).to(QT.dtype.element_ty),
            )
            packed_t += tl.sum(
                exponent_t
                << (j * (PASS_ROWS // 32) + tl.arange(0, PASS_ROWS // 32))[:, None] * 8,
                0,
            )
    if T_FORM:
        sf_col = (col // 128) * 128 + (col % 32) * 4 + (col % 128) // 32
        tl.store(ST + (e * (ROWS // 128) + rb) * COLS + sf_col, packed_t)


def _launch_config():
    """``MONO_EP_SM100_MXFP8_FORMS_CONFIG`` = ``num_warps,pass_rows,reciprocal`` (``/`` also separates); default
    ``4,32,0``, the original launch."""
    value = os.getenv("MONO_EP_SM100_MXFP8_FORMS_CONFIG", "4,32,0")
    num_warps, pass_rows, reciprocal = (
        int(v) for v in value.replace("/", ",").split(",")
    )
    return num_warps, pass_rows, bool(reciprocal)


def _forms(w, *, row_form, interleave, t_form, num_warps=None, pass_rows=None):
    experts, rows, cols = w.shape
    if (
        not w.is_cuda
        or not w.is_contiguous()
        or w.dtype not in (torch.bfloat16, torch.float32)
        or rows % (256 if interleave else 128)
        or cols % 128
    ):
        raise ValueError(
            "expected contiguous [E, N, K] CUDA weights with N, K multiples of 128"
        )
    default_warps, default_pass_rows, reciprocal = _launch_config()
    num_warps = default_warps if num_warps is None else num_warps
    pass_rows = default_pass_rows if pass_rows is None else pass_rows
    device = w.device
    qr = sr = qt = st = w
    if row_form:
        qr = torch.empty(experts, rows, cols, device=device, dtype=torch.float8_e4m3fn)
        sr = torch.empty(experts, cols // 128, rows, device=device, dtype=torch.int32)
    if t_form:
        qt = torch.empty(experts, cols, rows, device=device, dtype=torch.float8_e4m3fn)
        st = torch.empty(experts, rows // 128, cols, device=device, dtype=torch.int32)
    if experts:
        _mx_weight_tiles[(experts, rows // 128, cols // 128)](
            w,
            qr,
            sr,
            qt,
            st,
            rows,
            cols,
            row_form,
            interleave,
            t_form,
            pass_rows,
            reciprocal,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    out = []
    if row_form:
        out.append((qr, sr.transpose(1, 2)))
    if t_form:
        out.append((qt, st.transpose(1, 2)))
    return out


def fused_mx_backward_forms(w1, w2, **launch):
    """``(sm100_mx_w1, sm100_mx_w2t, sm100_mx_w1t)`` as ``(values, scale words)`` pairs."""
    w1_form, w1t_form = _forms(
        w1, row_form=True, interleave=True, t_form=True, **launch
    )
    (w2t_form,) = _forms(w2, row_form=False, interleave=False, t_form=True, **launch)
    return w1_form, w2t_form, w1t_form
