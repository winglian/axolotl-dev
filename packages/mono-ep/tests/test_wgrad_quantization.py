# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Check GPU-offset Hopper dW packing, including poisoned capacity tails."""

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize("sizes", [(0, 128, 0, 256), (0, 0, 0, 0), (128, 256, 128, 0)])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("mn_major", [False, True])
def test_grouped_transpose_capacity(sizes, compiled, prefix, mn_major):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires CUDA SM89 or newer")
    pytest.importorskip("triton")
    from mono_ep.wgrad_quantization import quantize_transposed_groups

    capacity, cols = 640, 257
    live = sum(sizes)
    x = torch.randn(capacity, cols, device="cuda", dtype=torch.bfloat16)
    x[live:] = float("nan")
    metadata = torch.tensor(sizes, device="cuda", dtype=torch.int32)
    if prefix:
        metadata = metadata.cumsum(0, dtype=torch.int32)
    operation = quantize_transposed_groups
    if compiled:
        torch.compiler.reset()
        operation = torch.compile(operation, fullgraph=True)
    q, scales = operation(x, metadata, use_psum_layout=prefix, scales_mn_major=mn_major)
    expected = torch.zeros(capacity * cols, device="cuda")
    expected_scales = torch.zeros(cols, capacity // 128, device="cuda")
    start = 0
    for size in sizes:
        for offset in range(0, size, 128):
            block = x[start + offset : start + offset + 128].float()
            scale = torch.exp2(
                torch.ceil(torch.log2(block.abs().amax(0).clamp_min(1e-4) / 448))
            )
            quantized = (block / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            expected[start * cols : (start + size) * cols].view(cols, size)[
                :, offset : offset + 128
            ] = quantized.float().T
            expected_scales[:, (start + offset) // 128] = scale
        start += size
    torch.testing.assert_close(q.float(), expected, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
