# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""MXFP8 fused derivatives agree with the explicit quantized GEMM path."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rows", [0, 384])
def test_mxfp8_fused_derivatives(compiled, rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("MXFP8 epilogue requires SM100")
    pytest.importorskip("deep_gemm")
    from mono_ep.activation import swiglu
    from mono_ep.fp8 import expert_gemm_fp8
    from mono_ep.ops import grouped_mxfp8_swiglu_backward

    torch.manual_seed(172)
    hidden, inner = 512, 256
    dy = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(2, hidden, inner, device="cuda") / hidden**0.5
    ids = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    ids[:128] = 0
    ids[128:256] = 1
    z = torch.randn(rows, 2 * inner, device="cuda", dtype=torch.bfloat16)
    scores = torch.rand(rows, device="cuda")
    z[ids < 0] = 0
    scores[ids < 0] = 0
    dh = expert_gemm_fp8(dy, w2.transpose(1, 2).contiguous(), ids, format="mxfp8")
    expected = swiglu(z, scores, dh)
    operation = (
        torch.compile(grouped_mxfp8_swiglu_backward, fullgraph=True)
        if compiled
        else grouped_mxfp8_swiglu_backward
    )
    actual = operation(dy, w2, ids, z, scores)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=0.02, rtol=0.02)
        assert torch.count_nonzero(value[ids < 0]) == 0
