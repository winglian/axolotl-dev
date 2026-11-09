# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Independent FP8 block-format checks on CPU and CUDA."""

import math

import pytest
import torch
from mono_ep.quantization import quantize_fp8


@pytest.mark.parametrize("block", [(1, 32), (1, 128), (128, 128)])
@pytest.mark.parametrize("shape", [(0, 256), (3, 137), (2, 131, 257)])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_fp8_blocks(shape, block, compiled, device):
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
            pytest.skip("FP8 conversion requires CUDA SM89 or newer")
        pytest.importorskip("triton")
    torch.manual_seed(41)
    x = torch.randn(shape, dtype=torch.bfloat16, device=device)
    # Adjacent experts must not share amax, even with partial tiles.
    if len(shape) == 3:
        x[1] *= 64
    operation = (
        torch.compile(
            quantize_fp8,
            backend="inductor" if device == "cuda" else "aot_eager",
            fullgraph=True,
        )
        if compiled
        else quantize_fp8
    )
    values, scales = operation(x, block_rows=block[0], block_cols=block[1])
    assert values.dtype == torch.float8_e4m3fn
    assert values.shape == x.shape
    assert scales.dtype == torch.float32
    assert scales.shape == (
        *shape[:-2],
        math.ceil(shape[-2] / block[0]),
        math.ceil(shape[-1] / block[1]),
    )
    batches = math.prod(shape[:-2])
    original = x.reshape(batches, *shape[-2:]).float()
    actual = values.float().reshape_as(original)
    scale_batches = scales.reshape(batches, *scales.shape[-2:])
    for batch in range(batches):
        for r in range(0, shape[-2], block[0]):
            for c in range(0, shape[-1], block[1]):
                tile = original[batch, r : r + block[0], c : c + block[1]]
                scale = 2.0 ** math.ceil(
                    math.log2(max(tile.abs().max().item(), 1e-4) / 448.0)
                )
                assert (
                    scale_batches[batch, r // block[0], c // block[1]].item() == scale
                )
                expected = (tile / scale).to(torch.float8_e4m3fn).float()
                torch.testing.assert_close(
                    actual[batch, r : r + block[0], c : c + block[1]],
                    expected,
                    atol=0,
                    rtol=0,
                )


def test_zero_and_exact_scale_boundaries():
    x = torch.zeros(5, 32)
    x[1:] = torch.tensor([448.0, 449.0, 224.0, 225.0])[:, None]
    values, scales = quantize_fp8(x)
    torch.testing.assert_close(scales[1:, 0], torch.tensor([1.0, 2.0, 0.5, 1.0]))
    assert torch.isfinite(values.float()).all()
    assert torch.count_nonzero(values[0].float()) == 0
    assert scales[0, 0] > 0


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
def test_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="BF16/FP32"):
        quantize_fp8(torch.ones(2, 32, dtype=dtype))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 256, 384), (2, 257, 385)])
@pytest.mark.parametrize("stride_kind", ["contiguous", "transpose", "sliced"])
def test_quantize_transposed_weights_128x128(compiled, dtype, shape, stride_kind):
    if compiled:
        torch.compiler.reset()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires CUDA SM89 or newer")
    pytest.importorskip("triton")
    from mono_ep.weight_quantization import quantize_transposed_weights_128x128

    torch.manual_seed(79)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if stride_kind == "transpose":
        x = torch.randn(
            (shape[0], shape[2], shape[1]), device="cuda", dtype=dtype
        ).transpose(1, 2)
    elif stride_kind == "sliced":
        x = torch.randn((shape[0], shape[1], shape[2] * 2), device="cuda", dtype=dtype)[
            :, :, ::2
        ]
    x[0, :128, :128] = 0
    x[1, 0, :4] = torch.tensor([448.0, 449.0, 224.0, 225.0], device="cuda", dtype=dtype)
    expected_q, expected_s = quantize_fp8(
        x.transpose(1, 2).contiguous(), block_rows=128, block_cols=128
    )
    operation = (
        torch.compile(quantize_transposed_weights_128x128, fullgraph=True)
        if compiled
        else quantize_transposed_weights_128x128
    )
    actual_q, actual_s = operation(x)
    assert actual_q.shape == x.transpose(1, 2).shape
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    torch.testing.assert_close(actual_s, expected_s, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 1536, 2048), (2, 2048, 768)])
def test_quantize_transposed_weights_128x128_qwen(compiled, dtype, shape):
    if compiled:
        torch.compiler.reset()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires CUDA SM89 or newer")
    pytest.importorskip("triton")
    from mono_ep.weight_quantization import quantize_transposed_weights_128x128

    torch.manual_seed(83)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    expected_q, expected_s = quantize_fp8(
        x.transpose(1, 2).contiguous(), block_rows=128, block_cols=128
    )
    operation = (
        torch.compile(quantize_transposed_weights_128x128, fullgraph=True)
        if compiled
        else quantize_transposed_weights_128x128
    )
    actual_q, actual_s = operation(x)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    torch.testing.assert_close(actual_s, expected_s, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_quantize_transposed_weights_128x128_scale_boundaries(dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires CUDA SM89 or newer")
    pytest.importorskip("triton")
    from mono_ep.weight_quantization import quantize_transposed_weights_128x128

    x = torch.zeros((1, 256, 256), device="cuda", dtype=dtype)
    x[0, 0, 0] = 448
    x[0, 128, 0] = 449
    x[0, 0, 128] = 224
    x[0, 128, 128] = 225
    actual_q, actual_s = quantize_transposed_weights_128x128(x)
    expected_q, expected_s = quantize_fp8(
        x.transpose(1, 2).contiguous(), block_rows=128, block_cols=128
    )
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    torch.testing.assert_close(actual_s, expected_s, atol=0, rtol=0)
    expected_scales = (
        torch.tensor([[1.0, 1.0], [0.5, 1.0]], device="cuda")
        if dtype == torch.bfloat16
        else torch.tensor([[1.0, 2.0], [0.5, 1.0]], device="cuda")
    )
    torch.testing.assert_close(actual_s[0], expected_scales, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("compiled", [False, True])
def test_hopper_activation_cache_finite_bf16_range(compiled):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires CUDA SM89 or newer")
    pytest.importorskip("triton")
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16).float()
    values = values[torch.isfinite(values)]
    x = (values[:, None] * torch.linspace(0, 1, 128)).bfloat16()
    amax = x.double().abs().amax(-1).clamp_min(1e-4)
    scales = torch.exp2(torch.ceil(torch.log2(amax / 448.0))).float()[:, None]
    expected = (x.float() / scales).to(torch.float8_e4m3fn)
    operation = (
        torch.compile(quantize_fp8, fullgraph=True) if compiled else quantize_fp8
    )
    actual, actual_scales = operation(x.cuda(), block_rows=1, block_cols=128)
    torch.testing.assert_close(actual_scales.cpu(), scales, atol=0, rtol=0)
    assert torch.equal(actual.view(torch.uint8).cpu(), expected.view(torch.uint8))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("block_cols", [32, 128])
def test_quantize_tokens_into_strided_scales(compiled, block_cols):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires CUDA SM89 or newer")
    pytest.importorskip("triton")
    from mono_ep.token_quantization import quantize_tokens, quantize_tokens_into

    torch.manual_seed(73)
    rows, cols = 19, 256
    groups = cols // block_cols
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    expected_q, expected_s = quantize_tokens(x, block_cols, None)
    q_out = torch.empty_like(expected_q)
    sentinel = -12345.0
    scale_storage = torch.full(
        (rows, groups + 3), sentinel, device="cuda", dtype=torch.float32
    )
    s_out = scale_storage[:, :groups]
    operation = (
        torch.compile(quantize_tokens_into, fullgraph=True)
        if compiled
        else quantize_tokens_into
    )
    actual_q, actual_s = operation(x, q_out, s_out, block_cols)
    assert actual_q.data_ptr() == q_out.data_ptr()
    assert actual_s.data_ptr() == s_out.data_ptr()
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    torch.testing.assert_close(actual_s, expected_s, atol=0, rtol=0)
    torch.testing.assert_close(
        scale_storage[:, groups:],
        torch.full_like(scale_storage[:, groups:], sentinel),
        atol=0,
        rtol=0,
    )
