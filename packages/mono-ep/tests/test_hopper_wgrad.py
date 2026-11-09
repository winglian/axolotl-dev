# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Independent quantized reference for GPU-size Hopper grouped dW."""

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.gpu_blackwell
@pytest.mark.parametrize("widths", [(256, 128), (1536, 2048), (2048, 768)])
@pytest.mark.parametrize("sizes", [(0, 128, 0, 256), (0, 0, 0, 0), (128, 256, 128, 0)])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mask_padding", [False, True])
@pytest.mark.parametrize("prepacked", [False, True])
def test_hopper_fp8_weight_gradients(widths, sizes, compiled, mask_padding, prepacked):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Grouped FP8 NT implementation requires Hopper SM90")
    deep_gemm = pytest.importorskip("deep_gemm")
    if not hasattr(deep_gemm._C, "k_grouped_fp8_gemm_nt_gpu_sizes"):
        pytest.skip("Rebuild vendored DeepGEMM GPU-size API")
    from mono_ep.hopper_wgrad import weight_grad_fp8_hopper

    torch.backends.cuda.matmul.allow_tf32 = False
    capacity = 640
    tensors = [
        torch.randn(capacity, n, device="cuda", dtype=torch.bfloat16) for n in widths
    ]
    for tensor in tensors:
        tensor[sum(sizes) :] = float("nan")
    ends = torch.tensor(sizes, device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    logical = []
    start = 0
    for size in sizes:
        logical.append(start + max(size - 7, 0) if mask_padding else start + size)
        if mask_padding:
            for tensor in tensors:
                tensor[logical[-1] : start + size] = float("nan")
        start += size
    valid_ends = (
        torch.tensor(logical, device="cuda", dtype=torch.int32)
        if mask_padding
        else None
    )
    operation = weight_grad_fp8_hopper
    arguments = (*tensors, ends, valid_ends)
    if prepacked:
        from mono_ep.hopper_wgrad import weight_grad_fp8_hopper_packed
        from mono_ep.wgrad_quantization import quantize_transposed_groups

        operands = [
            quantize_transposed_groups(
                tensor,
                ends,
                use_psum_layout=True,
                scales_mn_major=True,
                valid_ends=valid_ends,
            )
            for tensor in tensors
        ]
        for (values, scales), width in zip(operands, widths, strict=True):
            values[sum(sizes) * width :] = float("nan")
            scales[:, sum(sizes) // 128 :] = float("nan")
        operation = weight_grad_fp8_hopper_packed
        arguments = (
            *operands[0],
            *operands[1],
            torch.tensor(sizes, device="cuda", dtype=torch.int32),
            capacity,
        )
    if compiled:
        torch.compiler.reset()
        operation = torch.compile(operation, fullgraph=True)
    actual = operation(*arguments)
    if sum(sizes):
        from mono_ep.wgrad_quantization import quantize_transposed_groups

        packed = [
            quantize_transposed_groups(
                t, ends, use_psum_layout=True, valid_ends=valid_ends
            )
            for t in tensors
        ]
        control = torch.zeros_like(actual)
        live = sum(sizes)
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            (packed[0][0][: live * widths[0]], packed[0][1][:, : live // 128]),
            (packed[1][0][: live * widths[1]], packed[1][1][:, : live // 128]),
            control,
            list(sizes),
            torch.tensor(sizes, device="cuda", dtype=torch.int32),
            control,
            recipe=(1, 1, 128),
        )
        torch.testing.assert_close(actual, control, rtol=1e-5, atol=1e-5)
    start = 0
    for group, size in enumerate(sizes):
        decoded = []
        for tensor in tensors:
            source = tensor[start : start + size].float().clone()
            source[logical[group] - start :] = 0
            source = source.reshape(-1, 128, tensor.shape[1])
            scale = torch.exp2(
                torch.ceil(torch.log2(source.abs().amax(1).clamp_min(1e-4) / 448))
            )
            quantized = (
                (source / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            )
            decoded.append(
                (quantized.float() * scale[:, None]).reshape(size, tensor.shape[1])
            )
        expected = decoded[0].T @ decoded[1]
        error = (actual[group] - expected).norm() / expected.norm().clamp_min(1e-8)
        assert error < 0.002, (group, error.item())
        start += size
