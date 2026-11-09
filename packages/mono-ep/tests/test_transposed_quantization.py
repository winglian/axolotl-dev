# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Direct transpose quantization preserves the existing FP8 rounding recipe."""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 137, 65), (2, 512, 768), (0, 128, 64)])
def test_transposed_quantization(compiled, dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("Transpose quantization requires CUDA")
    pytest.importorskip("triton")
    from mono_ep.quantization import quantize_fp8

    torch.manual_seed(31)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    view = x.transpose(1, 2)
    expected = quantize_fp8(view.contiguous())
    operation = (
        torch.compile(quantize_fp8, fullgraph=True) if compiled else quantize_fp8
    )
    actual = operation(view)
    torch.testing.assert_close(actual[0].float(), expected[0].float(), atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
