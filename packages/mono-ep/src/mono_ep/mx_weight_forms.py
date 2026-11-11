# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""One-read builders of the SM100 MXFP8 backward weight forms.

Each 128 x 128 tile of W1 is read once and written as the interleaved row-wise form (scales along
H) and the transposed form (scales along 2I); each tile of W2 as the transposed form. Values and
packed UE8M0 scale words are bitwise identical to :func:`mono_ep.token_quantization.prepare_persistent_mxfp8`.
Two builders share the contract: the Triton kernel (``fused_mx_backward_forms``) and the CUDA
kernel in ``include/mxfp8_forms_launch.cu`` (``cuda_mx_backward_forms``), which stages each tile
in shared memory with a swizzle so both passes read it conflict-free and runs at the DRAM
bound on B200 (see docs/configuration.md, ``MONO_EP_SM100_MXFP8_WEIGHT_FORMS``).
"""

import functools
import os
from pathlib import Path

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
    ``2,32,1`` (bitwise equal to the original ``4,32,0`` and ~18 % faster on B200)."""
    value = os.getenv("MONO_EP_SM100_MXFP8_FORMS_CONFIG", "2,32,1")
    num_warps, pass_rows, reciprocal = (
        int(v) for v in value.replace("/", ",").split(",")
    )
    return num_warps, pass_rows, bool(reciprocal)


def _check(w, interleave):
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
    return experts, rows, cols


def _outputs(w, *, row_form, t_form):
    experts, rows, cols = w.shape
    device = w.device
    qr = sr = qt = st = w
    if row_form:
        qr = torch.empty(experts, rows, cols, device=device, dtype=torch.float8_e4m3fn)
        sr = torch.empty(experts, cols // 128, rows, device=device, dtype=torch.int32)
    if t_form:
        qt = torch.empty(experts, cols, rows, device=device, dtype=torch.float8_e4m3fn)
        st = torch.empty(experts, rows // 128, cols, device=device, dtype=torch.int32)
    return qr, sr, qt, st


def _pack(row_form, t_form, qr, sr, qt, st):
    out = []
    if row_form:
        out.append((qr, sr.transpose(1, 2)))
    if t_form:
        out.append((qt, st.transpose(1, 2)))
    return out


def _forms(w, *, row_form, interleave, t_form, num_warps=None, pass_rows=None):
    experts, rows, cols = _check(w, interleave)
    default_warps, default_pass_rows, reciprocal = _launch_config()
    num_warps = default_warps if num_warps is None else num_warps
    pass_rows = default_pass_rows if pass_rows is None else pass_rows
    qr, sr, qt, st = _outputs(w, row_form=row_form, t_form=t_form)
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
    return _pack(row_form, t_form, qr, sr, qt, st)


def fused_mx_backward_forms(w1, w2, **launch):
    """``(sm100_mx_w1, sm100_mx_w2t, sm100_mx_w1t)`` as ``(values, scale words)`` pairs."""
    w1_form, w1t_form = _forms(
        w1, row_form=True, interleave=True, t_form=True, **launch
    )
    (w2t_form,) = _forms(w2, row_form=False, interleave=False, t_form=True, **launch)
    return w1_form, w2t_form, w1t_form


@functools.lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    include = Path(__file__).resolve().parent / "include"
    verbose = os.environ.get("MONO_EP_MXFP8_FORMS_VERBOSE") == "1"
    return load(
        name="mono_ep_mxfp8_forms",
        sources=[
            str(include / "mxfp8_forms_bindings.cpp"),
            str(include / "mxfp8_forms_launch.cu"),
        ],
        extra_include_paths=[str(include)],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--fmad=false",
            "-gencode=arch=compute_100a,code=sm_100a",
            *(["-Xptxas=-v"] if verbose else []),
        ],
        verbose=verbose,
    )


def cuda_forms_available(device=None):
    """The CUDA builder is compiled for SM100 only."""
    return (
        torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] == 10
    )


def _cuda_debug():
    """``MONO_EP_SM100_MXFP8_FORMS_DEBUG``: phase-timing bits for the CUDA builder (skip the load = 1, the
    quantization = 2, the stores = 4); the output is garbage when set."""
    return int(os.getenv("MONO_EP_SM100_MXFP8_FORMS_DEBUG", "0"))


def _cuda_forms(w, *, row_form, interleave, t_form):
    experts, rows, cols = _check(w, interleave)
    if w.data_ptr() % 16:
        raise ValueError("expected 16-byte aligned weights")
    qr, sr, qt, st = _outputs(w, row_form=row_form, t_form=t_form)
    if experts:
        err = _extension().launch(
            w.data_ptr(),
            0 if w.dtype == torch.bfloat16 else 1,
            experts,
            rows,
            cols,
            int(row_form),
            int(interleave),
            int(t_form),
            qr.data_ptr(),
            sr.data_ptr(),
            qt.data_ptr(),
            st.data_ptr(),
            _cuda_debug(),
            torch.cuda.current_stream(w.device).cuda_stream,
        )
        if err:
            raise RuntimeError(f"mxfp8 forms launch failed with code {err}")
    return _pack(row_form, t_form, qr, sr, qt, st)


def cuda_mx_backward_forms(w1, w2):
    """:func:`fused_mx_backward_forms` from the CUDA kernel (same tuple, same bytes)."""
    w1_form, w1t_form = _cuda_forms(w1, row_form=True, interleave=True, t_form=True)
    (w2t_form,) = _cuda_forms(w2, row_form=False, interleave=False, t_form=True)
    return w1_form, w2t_form, w1t_form
