# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Direct retained projection packing preserves routes and masks unused slots."""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rows", [0, 137])
def test_retained_packing(compiled, rows):
    if not torch.cuda.is_available():
        pytest.skip("Retained packing requires CUDA")
    pytest.importorskip("triton")
    from mono_ep.packing import pack_retained_projections

    torch.manual_seed(52)
    saved = torch.full(
        (3, 19, 4, 513), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    ranks = torch.tensor([2, 0, 1, 2, 0], device="cuda", dtype=torch.int64)
    tokens = torch.tensor([1, 18, 4, 1, 9], device="cuda", dtype=torch.int64)
    assignments = torch.randint(0, 20, (rows,), device="cuda")
    assignments[::3] = -1
    valid = assignments >= 0
    receiver = assignments[valid] // 4
    slot = assignments[valid] % 4
    payload = torch.randn(3, 19, 4, 513, device="cuda", dtype=torch.bfloat16)
    saved[ranks[receiver], tokens[receiver], slot] = payload[
        ranks[receiver], tokens[receiver], slot
    ]
    expected = torch.zeros(rows, 513, device="cuda", dtype=torch.bfloat16)
    expected[valid] = saved[ranks[receiver], tokens[receiver], slot]
    operation = (
        torch.compile(pack_retained_projections, fullgraph=True)
        if compiled
        else pack_retained_projections
    )
    actual = operation(saved, ranks, tokens, assignments)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
