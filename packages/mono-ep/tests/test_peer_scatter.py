# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Direct epilogue scatter preserves unique assignments and untouched slots."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("hidden", [512, 2048])
def test_peer_scatter(compiled, hidden):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Peer scatter requires SM100")
    pytest.importorskip("deep_gemm")
    from mono_ep.ops import grouped_bf16
    from mono_ep.peer_scatter import peer_gemm_out

    torch.manual_seed(923)
    x = torch.randn(384, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.randn(3, hidden, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    ends = torch.tensor([128, 256, 256], device="cuda", dtype=torch.int32)
    ids = torch.arange(384, device="cuda", dtype=torch.int32) // 128
    ids[256:] = -1
    assignments = torch.randperm(512, device="cuda")[:384].int()
    assignments[::7] = -1
    assignments[256:] = -1
    expected_gemm = grouped_bf16(x, weights, ids)
    outputs = [
        torch.full((256, hidden), -19, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    expected = torch.full((512, hidden), -19, device="cuda", dtype=torch.bfloat16)
    valid = assignments >= 0
    expected[assignments[valid].long()] = expected_gemm[valid]
    pointers = torch.tensor(
        [t.data_ptr() for t in outputs], device="cuda", dtype=torch.int64
    )
    if compiled:
        torch.compiler.reset()
    operation = (
        torch.compile(peer_gemm_out, fullgraph=True) if compiled else peer_gemm_out
    )
    queue = operation(x, weights, ends, assignments, outputs, pointers)
    torch.testing.assert_close(torch.cat(outputs), expected, atol=0.005, rtol=0.005)
    assert queue[0, 3].item() == 1
