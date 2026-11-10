# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""ptxas 12.9 (Triton 3.8's assembler for pre-Blackwell targets) miscompiles the 8-bit layout
conversion that a transpose of a freshly cast e4m3 tile lowers to on sm_90; the SM90 weight
forms must not contain such a conversion."""

import re

import pytest
import torch

EIGHT_BIT_CONVERT = re.compile(
    r"convert_layout .*tensor<[^>]*x(i8|f8E4M3FN|f8E5M2|f8E4M3FNUZ|f8E5M2FNUZ)\b"
)
BAD, FP32_TRANS, RELOAD = 0, 1, 2


def _cuda_fp8():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires SM89 or newer")
    pytest.importorskip("triton")


def _ptxas_major():
    from triton.backends.nvidia.compiler import get_ptxas

    major, minor = torch.cuda.get_device_capability()
    return int(get_ptxas(major * 10 + minor).version.split(".")[0])


def eight_bit_layout_conversions(ttgir):
    return [
        line.strip() for line in ttgir.splitlines() if EIGHT_BIT_CONVERT.search(line)
    ]


def compiled_kernels(jit_fn):
    found = []
    for entry in getattr(jit_fn, "device_caches", {}).values():
        caches = entry[0] if isinstance(entry, (tuple, list)) else entry
        if isinstance(caches, dict):
            found.extend(k for k in caches.values() if hasattr(k, "asm"))
    return found


def _kernel():
    import triton
    import triton.language as tl
    from mono_ep.sm90_weight_forms import _e4m3_scale, _opaque

    @triton.jit
    def transposed_block(W, Q, QT, N: tl.constexpr, PATH: tl.constexpr):
        # the w2 / w2t pattern of the fused forms kernel: one 128x128 block of a (128, N) tensor
        cb = tl.program_id(0)
        r = tl.arange(0, 128).to(tl.int64)
        c = (cb * 128 + tl.arange(0, 128)).to(tl.int64)
        w = tl.load(W + r[:, None] * N + c[None, :])
        s = _e4m3_scale(tl.max(tl.max(tl.abs(w), 1), 0))
        q = tl.minimum(tl.maximum(w / s, -448.0), 448.0).to(Q.dtype.element_ty)
        tl.store(Q + r[:, None] * N + c[None, :], q)
        if PATH == 0:
            qt = tl.trans(q)
        elif PATH == 1:
            qt = tl.minimum(tl.maximum(tl.trans(w) / s, -448.0), 448.0).to(
                Q.dtype.element_ty
            )
        else:
            wT = tl.load(W + c[:, None] * _opaque(1) + r[None, :] * _opaque(N))
            qt = tl.minimum(tl.maximum(wT / s, -448.0), 448.0).to(Q.dtype.element_ty)
        tl.store(QT + c[:, None] * 128 + r[None, :], qt)

    return transposed_block


@pytest.mark.gpu
@pytest.mark.parametrize("path", [BAD, FP32_TRANS, RELOAD])
def test_transposed_e4m3_block(path):
    _cuda_fp8()
    n = 1024
    torch.manual_seed(1)
    w = torch.randn(128, n, device="cuda")
    q = torch.empty(128, n, device="cuda", dtype=torch.float8_e4m3fn)
    qt = torch.empty(n, 128, device="cuda", dtype=torch.float8_e4m3fn)
    kernel = _kernel()[(n // 128,)](
        w, q, qt, n, path, num_warps=8, enable_fp_fusion=False
    )
    torch.cuda.synchronize()
    bytes_equal = torch.equal(qt.view(torch.uint8), q.view(torch.uint8).t())
    conversions = eight_bit_layout_conversions(kernel.asm["ttgir"])
    if path == RELOAD:
        assert not conversions, conversions
        assert bytes_equal
        return
    assert conversions, "expected the transpose to lower to an 8-bit layout conversion"
    if path == FP32_TRANS:
        # the 92c804aad6 workaround: correct for this (stored tile, 128x128, 8 warps) schedule
        assert bytes_equal
        return
    pre_blackwell = torch.cuda.get_device_capability() < (10, 0)
    if pre_blackwell and _ptxas_major() < 13:
        if torch.cuda.get_device_capability() == (9, 0):
            assert not bytes_equal, (
                "ptxas 12.x no longer miscompiles the fp8 transpose on sm_90; the transposed "
                "reloads in sm90_weight_forms.py can be reconsidered"
            )
        elif not bytes_equal:
            pytest.xfail("ptxas 12.x miscompiles the fp8 tile transpose")
    else:
        assert bytes_equal


@pytest.mark.gpu
def test_fused_forms_have_no_8bit_layout_conversion():
    _cuda_fp8()
    from mono_ep import sm90_weight_forms

    w1 = torch.randn(1, 512, 256, device="cuda")
    w2 = torch.randn(1, 256, 256, device="cuda")
    sm90_weight_forms.quantize_all_forms(w1, w2)
    kernels = compiled_kernels(sm90_weight_forms._quantize_forms)
    assert kernels, "no compiled _quantize_forms kernel found in the JIT cache"
    for kernel in kernels:
        assert not eight_bit_layout_conversions(kernel.asm["ttgir"])
