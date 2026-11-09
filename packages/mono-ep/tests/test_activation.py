# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Weighted SwiGLU derivatives, including zero routing weights and partial tiles."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("rows,inner", [(1, 128), (137, 257), (0, 256)])
@pytest.mark.parametrize("compiled", [False, True])
def test_weighted_swiglu(rows, inner, compiled):
    from mono_ep.activation import swiglu

    torch.manual_seed(81)
    z = torch.randn(
        rows, 2 * inner, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    scores = torch.randn(rows, device="cuda", dtype=torch.float32)
    scores[::3] = 0
    scores.requires_grad_()
    dh = torch.randn(rows, inner, device="cuda", dtype=torch.bfloat16)
    gate, up = z.float().chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate) * up * scores[:, None]).bfloat16()
    dz, ds = torch.autograd.grad(expected, (z, scores), dh)
    kernel = torch.compile(swiglu, fullgraph=True) if compiled else swiglu
    actual, actual_dz, actual_ds = kernel(z.detach(), scores.detach(), dh)
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.008)
    torch.testing.assert_close(actual_dz, dz, atol=0.016, rtol=0.008)
    torch.testing.assert_close(actual_ds, ds, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("rows,inner", [(1, 128), (137, 257), (0, 256), (256, 1536)])
@pytest.mark.parametrize("compiled", [False, True])
def test_swiglu_fp8_matches_separate(rows, inner, compiled):
    from mono_ep.activation import swiglu, swiglu_fp8
    from mono_ep.quantization import quantize_fp8

    torch.manual_seed(81)
    z = torch.randn(rows, 2 * inner, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(rows, device="cuda", dtype=torch.float32)
    scores[::3] = 0
    expected, expected_scales = quantize_fp8(swiglu(z, scores))
    operation = (
        torch.compile(swiglu_fp8, fullgraph=True, dynamic=False)
        if compiled
        else swiglu_fp8
    )
    values, scales = operation(z, scores)
    torch.testing.assert_close(values.float(), expected.float(), atol=0, rtol=0)
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
