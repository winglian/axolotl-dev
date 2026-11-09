# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Inactive expert tiles must not evaluate poisoned SwiGLU inputs."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]


@pytest.mark.parametrize("pattern", ["empty", "holes", "tail"])
@pytest.mark.parametrize("compiled", [False, True])
def test_inactive_swiglu_tiles(pattern, compiled):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("DeepGEMM epilogue requires SM100")
    pytest.importorskip("deep_gemm")
    from mono_ep.activation import swiglu
    from mono_ep.ops import grouped_bf16, grouped_swiglu_backward

    torch.manual_seed(877)
    rows, hidden, inner = 512, 256, 256
    ids = torch.zeros(rows, device="cuda", dtype=torch.int32)
    if pattern == "empty":
        ids.fill_(-1)
    elif pattern == "holes":
        ids[128:256] = -1
    else:
        ids[256:] = -1
    invalid = ids < 0
    dy = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weights = (
        torch.randn(1, hidden, inner, device="cuda", dtype=torch.bfloat16) / hidden**0.5
    )
    z = torch.randn(rows, 2 * inner, device="cuda", dtype=torch.bfloat16)
    scores = torch.rand(rows, device="cuda")
    z[invalid] = 0
    scores[invalid] = 0
    dh = grouped_bf16(dy, weights.transpose(1, 2).contiguous(), ids)
    expected = swiglu(z, scores, dh)
    z[invalid] = float("nan")
    scores[invalid] = float("nan")
    operation = grouped_swiglu_backward
    if compiled:
        torch.compiler.reset()
        operation = torch.compile(operation, fullgraph=True)
    actual = operation(dy, weights, ids, z, scores)
    for got, want in zip(actual, expected, strict=True):
        assert torch.isfinite(got).all()
        assert not got[invalid].count_nonzero().item()
        error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(
            1e-8
        )
        assert error.item() < 1e-3


@pytest.mark.parametrize("end", [0, 1, 17, 128, 137])
@pytest.mark.parametrize("compiled", [False, True])
def test_bounded_swiglu_quantization(end, compiled):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("FP8 activation regression requires SM100")
    pytest.importorskip("triton")
    from mono_ep.activation import swiglu_fp8

    torch.compiler.reset()
    z = torch.randn(137, 514, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(137, device="cuda")
    z[end:] = 0
    scores[end:] = 0
    expected = swiglu_fp8(z, scores)
    z[end:] = float("nan")
    scores[end:] = float("nan")
    row_end = torch.tensor(end, device="cuda", dtype=torch.int32)
    operation = torch.compile(swiglu_fp8, fullgraph=True) if compiled else swiglu_fp8
    actual = operation(z, scores, row_end)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.float(), want.float(), atol=0, rtol=0)
