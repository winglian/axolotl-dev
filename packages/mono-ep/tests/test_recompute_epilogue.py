# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Recompute epilogue preserves BF16 intermediates and explicit derivatives."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]


@pytest.mark.parametrize("psum", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rows,inner", [(0, 512), (384, 512), (384, 768)])
def test_recompute_epilogue(compiled, rows, inner, psum):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Recompute epilogue requires SM100")
    pytest.importorskip("deep_gemm")
    from mono_ep.activation import swiglu
    from mono_ep.ops import grouped_bf16
    from mono_ep.recompute import grouped_recompute_swiglu

    torch.manual_seed(134)
    x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3, 2 * inner, 512, device="cuda") / 512**0.5
    dh = torch.randn(rows, inner, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(rows, device="cuda")
    ids = torch.arange(rows, device="cuda", dtype=torch.int32) // 128
    ids[ids == 2] = -1
    z = grouped_bf16(x, w.bfloat16(), ids)
    expected = swiglu(z, scores, dh)
    if compiled:
        torch.compiler.reset()
    operation = (
        torch.compile(grouped_recompute_swiglu, fullgraph=True)
        if compiled
        else grouped_recompute_swiglu
    )
    ends = torch.tensor(
        [128, 256, 256] if rows else [0, 0, 0], device="cuda", dtype=torch.int32
    )
    actual = operation(x, w, ids, dh, scores, ends if psum else None)
    valid = ids >= 0
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            result[~valid], torch.zeros_like(result[~valid]), atol=0, rtol=0
        )
        if rows:
            error = (
                result[valid].float() - reference[valid].float()
            ).norm() / reference[valid].float().norm().clamp_min(1e-8)
            assert error < 0.005, error.item()
        assert result.shape == reference.shape
