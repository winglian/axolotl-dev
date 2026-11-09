# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Check columnwise FP8 values and scales against the independent Torch formula."""

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(0, 256), (137, 257), (128, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("compiled", [False, True])
def test_column_quantization(shape, dtype, compiled):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires CUDA SM89 or newer")
    pytest.importorskip("triton")
    from mono_ep.column_quantization import quantize_columns

    x = torch.randn(shape, device="cuda", dtype=dtype)
    if shape[0]:
        x[:32, 0] = 0
    operation = (
        torch.compile(quantize_columns, fullgraph=True)
        if compiled
        else quantize_columns
    )
    actual, scales = operation(x)
    assert actual.shape == x.shape
    assert actual.dtype == torch.float8_e4m3fn
    assert scales.shape == ((shape[0] + 31) // 32, shape[1])
    for start in range(0, shape[0], 32):
        block = x[start : start + 32].float()
        expected_scale = torch.exp2(
            torch.ceil(torch.log2(block.abs().amax(0).clamp_min(1e-4) / 448))
        )
        expected = (block / expected_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        torch.testing.assert_close(scales[start // 32], expected_scale, atol=0, rtol=0)
        torch.testing.assert_close(
            actual[start : start + 32].float(), expected.float(), atol=0, rtol=0
        )


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [0, 128, 137])
@pytest.mark.parametrize("compiled", [False, True])
def test_packed_column_quantization(rows, compiled):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires CUDA SM89 or newer")
    pytest.importorskip("triton")
    from mono_ep.column_quantization import (
        quantize_columns,
        quantize_packed_columns,
    )

    values = torch.randn(rows, 256, device="cuda").to(torch.float8_e4m3fn)
    exponents = torch.randint(115, 140, (rows, 8), device="cuda", dtype=torch.int64)
    packed = sum(exponents[:, i::4] << (i * 8) for i in range(4)).int()
    packed = packed.T.contiguous().T
    decoded = (
        values.float()
        * torch.exp2(exponents.float() - 127).repeat_interleave(32, dim=1)
    ).bfloat16()
    expected, expected_scales = quantize_columns(decoded)
    operation = (
        torch.compile(quantize_packed_columns, fullgraph=True)
        if compiled
        else quantize_packed_columns
    )
    actual, actual_scales = operation(values.view(torch.uint8), packed)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("end", [0, 17, 128, 137, 256])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_bounded_column_quantization(end, packed, compiled):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires CUDA SM89 or newer")
    pytest.importorskip("triton")
    from mono_ep.column_quantization import (
        quantize_columns,
        quantize_packed_columns,
    )

    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    row_end = torch.tensor(end, device="cuda", dtype=torch.int32)
    if packed:
        x = x.to(torch.float8_e4m3fn)
        scales = (
            torch.full((256, 2), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
            .T.contiguous()
            .T
        )
        operation = quantize_packed_columns
        args = (x, scales, row_end)
    else:
        operation = quantize_columns
        args = (x, row_end)
    expected_input = x.bfloat16()
    expected_input[end:] = 0
    expected = quantize_columns(expected_input)
    x[end:] = float("nan")
    if packed:
        scales[end:] = -1
    if compiled:
        torch.compiler.reset()
        operation = torch.compile(operation, fullgraph=True)
    actual = operation(*args)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.float(), want.float(), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("granularity", [32, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("compiled", [False, True])
def test_masked_token_quantization(granularity, dtype, compiled):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires CUDA SM89 or newer")
    pytest.importorskip("triton")
    from mono_ep.token_quantization import quantize_tokens

    torch.compiler.reset()
    x = torch.randn(137, 257, device="cuda", dtype=dtype)
    ids = torch.arange(137, device="cuda", dtype=torch.int32) % 4
    ids[::3] = -1
    x[ids < 0] = float("nan")
    operation = (
        torch.compile(quantize_tokens, fullgraph=True) if compiled else quantize_tokens
    )
    actual, scales = operation(x, granularity, ids)
    clean = x.masked_fill((ids < 0).unsqueeze(-1), 0).float()
    for start in range(0, x.shape[1], granularity):
        block = clean[:, start : start + granularity]
        scale = torch.exp2(
            torch.ceil(torch.log2(block.abs().amax(1).clamp_min(1e-4) / 448))
        )
        expected = (block / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        torch.testing.assert_close(
            scales[:, start // granularity], scale, atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual[:, start : start + granularity].float(),
            expected.float(),
            atol=0,
            rtol=0,
        )


@pytest.mark.gpu
@pytest.mark.gpu_blackwell
def test_wide_row_offsets():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Large-offset regression runs on SM100")
    if torch.cuda.mem_get_info()[0] < 24 * 1024**3:
        pytest.skip("Large-offset regression requires 24 GiB free GPU memory")
    pytest.importorskip("triton")
    from mono_ep.activation import swiglu
    from mono_ep.column_quantization import quantize_columns
    from mono_ep.packing import combine_rows, pack_rows
    from mono_ep.token_quantization import quantize_tokens

    rows, cols = 2**20 + 128, 2048
    x = torch.ones(rows, cols, device="cuda", dtype=torch.bfloat16)
    x[-1].fill_(2)
    indices = torch.tensor([0, rows - 1], device="cuda")
    ids = torch.zeros(rows, device="cuda", dtype=torch.int32)
    for operation in (quantize_tokens, quantize_columns):
        args = (32, ids) if operation is quantize_tokens else ()
        q, scales = operation(x, *args)
        assert torch.isfinite(scales).all()
        expected_scale = torch.tensor([2**-8, 2**-7], device="cuda")[:, None]
        torch.testing.assert_close(
            q[indices].float(), x[indices].float() / expected_scale, atol=0, rtol=0
        )
        del q, scales

    assignments = torch.arange(rows, device="cuda", dtype=torch.int32)
    packed = pack_rows(x, assignments, 1)
    torch.testing.assert_close(packed[indices], x[indices], atol=0, rtol=0)
    restored = combine_rows(packed, assignments[indices], 2, 1)
    torch.testing.assert_close(restored, x[indices], atol=0, rtol=0)
    del packed, restored

    scores = torch.ones(rows, device="cuda")
    actual = swiglu(x, scores)
    sample = x[indices].float()
    gate, up = sample.chunk(2, dim=1)
    expected = (gate * torch.sigmoid(gate) * up).bfloat16()
    torch.testing.assert_close(actual[indices], expected, atol=0, rtol=0)
