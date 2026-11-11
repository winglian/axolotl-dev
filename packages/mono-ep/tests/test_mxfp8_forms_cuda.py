# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The CUDA MXFP8 weight-form builder writes the Triton builder's bytes (values and scale words)."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("the CUDA MXFP8 weight forms need SM100", allow_module_level=True)

from mono_ep import mx_weight_forms as forms  # noqa: E402


def _same(a, b):
    return all(
        torch.equal(x[0].view(torch.uint8), y[0].view(torch.uint8))
        and x[1].shape == y[1].shape
        and x[1].stride() == y[1].stride()
        and torch.equal(x[1], y[1])
        for x, y in zip(a, b, strict=True)
    )


def _weights(experts, inner, hidden, dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w1 = torch.randn(experts, 2 * inner, hidden, device="cuda", generator=gen) * 0.02
    w2 = torch.randn(experts, hidden, inner, device="cuda", generator=gen) * 0.02
    # exact zeros, whole all-zero blocks (amax floor), tiny and saturating magnitudes
    w1.view(-1)[::7919] = 0.0
    w1[:, :8, :64] = 0.0
    w2[:, 32:64, :] = 0.0
    w1[:, 16, :32] = 1.0e-7
    w2[:, 0, :32] = 3.0e4
    w1[:, 200, 100] = -448.0 * 2.0**-120
    return w1.to(dtype), w2.to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [(3, 128, 256), (2, 384, 640), (1, 768, 2048)],
    ids=["tiny", "odd-tiles", "30b-ish"],
)
def test_cuda_forms_match_triton(dtype, shape):
    experts, inner, hidden = shape
    w1, w2 = _weights(experts, inner, hidden, dtype)
    assert _same(
        forms.cuda_mx_backward_forms(w1, w2), forms.fused_mx_backward_forms(w1, w2)
    )


def test_cuda_forms_match_eager_reference():
    from mono_ep.weight_cache import _mx_forms

    w1, w2 = _weights(2, 256, 512, torch.bfloat16, seed=1)
    assert _same(forms.cuda_mx_backward_forms(w1, w2), _mx_forms(w1, w2))


def test_cuda_forms_all_zero_experts():
    w1 = torch.zeros(2, 256, 256, device="cuda", dtype=torch.bfloat16)
    w2 = torch.zeros(2, 256, 128, device="cuda", dtype=torch.float32)
    out = forms.cuda_mx_backward_forms(w1, w2)
    assert _same(out, forms.fused_mx_backward_forms(w1, w2))
    # amax floor 1e-4: every value 0, every word the floor exponent (1e-4 / 448 rounded up to 2^-22 -> 105)
    for values, words in out:
        assert not values.view(torch.uint8).any()
        assert (words == 0x69696969).all()


def test_selector_and_cache_dispatch(monkeypatch):
    from mono_ep import token_quantization, weight_cache

    monkeypatch.delenv("MONO_EP_SM100_MXFP8_WEIGHT_FORMS", raising=False)
    monkeypatch.delenv("MONO_EP_SM100_MXFP8_COMPILED_QUANT", raising=False)
    assert token_quantization.mxfp8_weight_form_builder() == "cuda"
    monkeypatch.setenv("MONO_EP_SM100_MXFP8_WEIGHT_FORMS", "fused")
    assert token_quantization.mxfp8_weight_form_builder() == "fused"
    monkeypatch.setenv("MONO_EP_SM100_MXFP8_WEIGHT_FORMS", "cuda")
    w1, w2 = _weights(1, 128, 256, torch.bfloat16, seed=2)
    monkeypatch.setenv("MONO_EP_WEIGHT_CACHE", "0")
    got = weight_cache.mxfp8_backward_weight_forms(w1, w2)
    assert _same(got, forms.fused_mx_backward_forms(w1, w2))
