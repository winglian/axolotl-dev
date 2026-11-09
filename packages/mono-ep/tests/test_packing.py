# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""GPU-resident ragged packing and deterministic restoration."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("tokens", [0, 1, 137])
@pytest.mark.parametrize("compiled", [False, True])
def test_device_layout(tokens, compiled):
    from mono_ep.packing import DeviceExpertLayout

    torch.manual_seed(82)
    experts, topk, hidden = 8, 4, 257
    ids = torch.randint(0, experts - 1, (tokens, topk), device="cuda")
    ids[::3, 1:] = -1
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(tokens, topk, device="cuda")

    def run(x, scores, ids):
        layout = DeviceExpertLayout.build(ids, experts, 128)
        packed = layout.pack(x)
        return (
            packed,
            layout.scores(scores),
            layout.combine(packed, tokens),
            layout.restore_scores(layout.scores(scores), scores),
            layout.row_assignments,
            layout.expert_ids,
            layout.ends,
            layout.chunk_descriptors(),
            layout.token_ids(),
        )

    run_fn = torch.compile(run, fullgraph=True) if compiled else run
    (
        packed,
        packed_scores,
        restored,
        restored_scores,
        row_assignments,
        expert_ids,
        ends,
        descriptors,
        token_ids,
    ) = run_fn(x, scores, ids)
    valid = row_assignments >= 0
    assignments = row_assignments[valid]
    torch.testing.assert_close(packed[valid], x[assignments // topk], atol=0, rtol=0)
    torch.testing.assert_close(
        packed_scores[valid], scores.flatten()[assignments], atol=0, rtol=0
    )
    assert torch.all(packed[~valid] == 0)
    assert torch.all(packed_scores[~valid] == 0)
    assert torch.all(expert_ids[~valid] == -1)
    assert torch.equal(assignments.sort().values, torch.where(ids.flatten() >= 0)[0])
    torch.testing.assert_close(
        token_ids, torch.where(row_assignments >= 0, row_assignments // topk, -1).int()
    )
    assert (descriptors[:, 3] == 0).all()
    torch.testing.assert_close(
        restored, (x.float() * (ids >= 0).sum(-1)[:, None]).bfloat16(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        restored_scores, scores.masked_fill(ids < 0, 0), atol=0, rtol=0
    )
    start = 0
    for expert, end in enumerate(ends.tolist()):
        count = int((ids == expert).sum().item())
        assert descriptors[expert].tolist() == [
            expert,
            start,
            ((count + 127) // 128) * 128,
            0,
        ]
        assert end == start + count
        assert torch.all(expert_ids[start:end] == expert)
        start = (end + 127) // 128 * 128


@pytest.mark.parametrize("experts,tokens", [(1, 0), (1, 137), (7, 137), (128, 4096)])
@pytest.mark.parametrize("all_masked", [False, True])
def test_layout_metadata_counts(experts, tokens, all_masked):
    from mono_ep.packing import DeviceExpertLayout

    torch.manual_seed(93)
    ids = torch.randint(-1, experts, (tokens, 4), device="cuda", dtype=torch.int32)
    if all_masked:
        ids.fill_(-1)
    layout = DeviceExpertLayout.build(ids, experts, 256)
    flat = ids.cpu().flatten()
    start = 0
    for expert, (base, end, size) in enumerate(
        zip(
            layout.starts.tolist(),
            layout.ends.tolist(),
            layout.padded_sizes.tolist(),
            strict=True,
        )
    ):
        assignments = torch.where(flat == expert)[0]
        count = assignments.numel()
        padded = ((count + 255) // 256) * 256
        assert (base, end, size) == (start, start + count, padded)
        torch.testing.assert_close(layout.row_assignments[base:end].cpu(), assignments)
        start += padded
