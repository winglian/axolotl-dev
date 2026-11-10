# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("the CUDA NVFP4 requant needs SM100", allow_module_level=True)

from mono_ep.nvfp4_requant import (  # noqa: E402
    cuda_nvfp4_requant as cr,
    nvfp4_requant as ref,
    triton_nvfp4_requant as tr,
)

E, N, K = 3, 128, 320  # 128-wide K tiles do not divide K


def _same(x, y):
    return torch.equal(x.view(torch.uint8), y.view(torch.uint8))


def _base(gen, rows=N, cols=K, projections=1):
    w = torch.randn(E, rows, cols, device="cuda", generator=gen) * 0.02
    w[:, :, :16] = 0.0
    pts = ref.per_tensor_scale_from_amax(
        w.reshape(E, projections, -1).abs().amax(dim=2)
    )
    q, s = ref.quantize_nvfp4_packed_ref(
        w.reshape(E * rows, cols), ref.expand_pts_rows(pts, rows)
    )
    return q.view(E, rows, cols // 2), s.view(E, rows, cols // 16), pts


def test_conversions_match_torch_encoders():
    gen = torch.Generator(device="cuda").manual_seed(0)
    # E2M1 ties and their neighbours, plus random values in range
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda")
    x = torch.cat(
        [
            ties,
            torch.nextafter(ties, ties + 1),
            torch.nextafter(ties, ties - 1),
            torch.rand(1 << 20, device="cuda", generator=gen) * 7,
        ]
    )
    x = torch.cat([x, -x, torch.tensor([0.0, -0.0], device="cuda")])
    got = cr.convert_probe("e2m1_encode", x.view(torch.int32)) & 0xF
    exp = ref.encode_e2m1_rne(x.clamp(-6.0, 6.0), sign_from_bit=True).to(torch.int32)
    assert torch.equal(got, exp)
    s = torch.rand(1 << 20, device="cuda", generator=gen) * 448
    got = cr.convert_probe("e4m3_encode", s.view(torch.int32)) & 0xFF
    exp = s.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
    assert torch.equal(got, exp)


@pytest.mark.parametrize("r", [3, 16])
@pytest.mark.parametrize("lora_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("modelopt", [False, True])
@pytest.mark.parametrize("scale_mode", ["fresh", "reuse"])
@pytest.mark.parametrize("projections", [1, 2])
@pytest.mark.parametrize("interleave_rows", [False, True])
def test_bytes_match_triton(
    r, lora_dtype, modelopt, scale_mode, projections, interleave_rows
):
    gen = torch.Generator(device="cuda").manual_seed(r)
    q, s, pts = _base(gen, projections=projections)
    A = (torch.randn(r * E, K, device="cuda", generator=gen) * 0.05).to(lora_dtype)
    B = (torch.randn(N, r * E, device="cuda", generator=gen) * 0.05).to(lora_dtype)
    numerics = ref.RequantNumerics(modelopt_compat=modelopt, scale_mode=scale_mode)
    kw = dict(numerics=numerics, interleave_rows=interleave_rows)
    for e0, e1 in ((0, E), (1, 2)):
        t = tr.requant_lora_packed(q, s, pts, A, B, 1.75, pts, e0, e1, **kw)
        c = cr.requant_lora_packed_cuda(q, s, pts, A, B, 1.75, pts, e0, e1, **kw)
        assert _same(t.qdata, c.qdata)
        assert _same(t.scale, c.scale)
        assert _same(t.scale_atoms, c.scale_atoms)


def test_paired_equals_two_requants():
    gen = torch.Generator(device="cuda").manual_seed(7)
    r = 16
    q, s, pts = _base(gen)
    qt, st, _ = _base(gen, rows=K, cols=N)
    a = (torch.randn(E, r, K, device="cuda", generator=gen) * 0.05).bfloat16()
    b = (torch.randn(E, N, r, device="cuda", generator=gen) * 0.05).bfloat16()
    normal, trans = cr.requant_paired(q, s, qt, st, pts, a, b, 2.0, interleave=8)
    exp_n = cr.requant_natural(q, s, pts, a, b, 2.0, pts, interleave=8)
    exp_t = cr.requant_natural(
        qt, st, pts, b.transpose(1, 2), a.transpose(1, 2), 2.0, pts
    )
    for x, y in ((normal, exp_n), (trans, exp_t)):
        assert _same(x.qdata, y.qdata)
        assert _same(x.scale, y.scale)
        assert _same(x.scale_atoms, y.scale_atoms)


def test_fused_transposed_quantizes_the_merged_tile():
    gen = torch.Generator(device="cuda").manual_seed(9)
    r = 16
    q, s, pts = _base(gen)
    a = (torch.randn(E, r, K, device="cuda", generator=gen) * 0.05).bfloat16()
    b = (torch.randn(E, N, r, device="cuda", generator=gen) * 0.05).bfloat16()
    normal, trans = cr.requant_fused(q, s, pts, a, b, 2.0)
    assert _same(normal.qdata, cr.requant_natural(q, s, pts, a, b, 2.0, pts).qdata)
    # merged bf16 weight with the kernels' sequential rank-order delta (exact for bf16 factors)
    w = (
        ref.dequantize_nvfp4_packed_ref(
            q.reshape(E * N, K // 2),
            s.reshape(E * N, K // 16),
            pts.repeat_interleave(N),
            torch.bfloat16,
        )
        .float()
        .view(E, N, K)
    )
    acc = torch.zeros_like(w)
    for i in range(r):
        acc = acc + b[:, :, i : i + 1].float() * a[:, i : i + 1, :].float()
    merged = (w + acc * 2.0).bfloat16().float()
    exp_q, exp_s = ref.quantize_nvfp4_packed_ref(
        merged.transpose(1, 2).reshape(E * K, N), pts.repeat_interleave(K)
    )
    assert _same(trans.qdata.reshape(E * K, N // 2), exp_q)
    assert _same(trans.scale.reshape(E * K, N // 16), exp_s)
    assert _same(trans.scale_atoms, ref.scale_atoms_from_rowmajor(trans.scale))
