# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Peer gather indexing and padding, using local mappings for isolated validation."""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [0, 137])
@pytest.mark.parametrize("bounded", [False, True])
def test_peer_packing(compiled, dtype, rows, bounded):
    if not torch.cuda.is_available():
        pytest.skip("Peer packing requires CUDA")
    pytest.importorskip("triton")
    from mono_ep.peer_packing import pack_peer_rows

    torch.manual_seed(83)
    capacity, topk, width = 19, 4, 513
    sources = [
        torch.randn(capacity, width, dtype=dtype, device="cuda") for _ in range(3)
    ]
    assignments = torch.randint(0, 3 * capacity * topk, (rows,), device="cuda")
    assignments[::3] = -1
    expected = torch.zeros(rows, width, dtype=dtype, device="cuda")
    valid = assignments >= 0
    expected[valid] = torch.cat(sources)[assignments[valid] // topk]
    row_end = (
        torch.tensor(rows // 2, device="cuda", dtype=torch.int32) if bounded else None
    )
    if bounded:
        expected[rows // 2 :] = 0
    if compiled:
        torch.compiler.reset()
    operation = (
        torch.compile(pack_peer_rows, fullgraph=True) if compiled else pack_peer_rows
    )
    torch.testing.assert_close(
        operation(sources, assignments, topk, row_end), expected, rtol=0, atol=0
    )
    for source in sources:
        source.add_(1)
    expected[valid] = torch.cat(sources)[assignments[valid] // topk]
    if bounded:
        row_end.fill_(rows)
    torch.testing.assert_close(
        operation(sources, assignments, topk, row_end), expected, rtol=0, atol=0
    )
