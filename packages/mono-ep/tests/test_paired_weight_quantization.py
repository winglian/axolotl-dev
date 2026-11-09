# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Paired gate/up scales preserve the existing interleaved weight contract."""

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("shape", [(2, 1536, 2048), (2, 256, 257)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sliced", [False, True])
def test_paired_weights(shape, dtype, compiled, sliced):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires SM89 or newer")
    pytest.importorskip("triton")
    from mono_ep.paired_weight_quantization import quantize_paired_gate_up_weights
    from mono_ep.quantization import quantize_fp8

    if compiled:
        torch.compiler.reset()
    torch.manual_seed(192)
    e, n, h = shape
    x = torch.randn(e, n, h * (2 if sliced else 1), device="cuda", dtype=dtype)
    if sliced:
        x = x[:, :, ::2]
    inner = n // 2
    x[0, :inner] *= 32
    x[1, inner:] *= 32
    x[0, :64, :128] = 0
    x[0, inner : inner + 64, :128] = 0
    interleaved = (
        x.reshape(e, 2, inner // 8, 8, h)
        .permute(0, 2, 1, 3, 4)
        .reshape(e, n, h)
        .contiguous()
    )
    expected_q, expected_sf = quantize_fp8(interleaved, block_rows=128, block_cols=128)
    expected_q = (
        expected_q.reshape(e, inner // 8, 2, 8, h)
        .permute(0, 2, 1, 3, 4)
        .reshape(e, n, h)
        .contiguous()
    )
    operation = (
        torch.compile(quantize_paired_gate_up_weights, fullgraph=True)
        if compiled
        else quantize_paired_gate_up_weights
    )
    q, sf = operation(x)
    assert q.is_contiguous() and sf.is_contiguous()
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    torch.testing.assert_close(sf, expected_sf, rtol=0, atol=0)
