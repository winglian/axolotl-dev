# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Exact peer-quantization payloads and immutable consumer descriptors."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gran", [32, 128])
@pytest.mark.parametrize(
    "mode", ["dense", "routed", "receiver_clear", "receiver_clear_empty"]
)
@pytest.mark.parametrize("compiled", [False, True])
def test_peer_quantization(dtype, gran, mode, compiled):
    from mono_ep.peer import peer_gather_quantize_publish, peer_quantize_publish
    from mono_ep.quantization import quantize_fp8

    routed = mode != "dense"
    skip_padding = mode.startswith("receiver_clear")
    torch.manual_seed(271)
    rows, hidden = 132, 1024
    source = torch.randn(137 if routed else rows, hidden, dtype=dtype)
    indices = torch.randint(source.shape[0], (rows,), dtype=torch.int32)
    indices[::7] = -1
    indices[16:48] = -1
    if mode == "receiver_clear_empty":
        indices.fill_(-1)
    gathered = (
        source[indices.clamp_min(0).long()].masked_fill((indices < 0)[:, None], 0)
        if routed
        else source
    )
    expected, scales = quantize_fp8(gathered, block_cols=gran)
    exponents = (scales.view(torch.int32) >> 23) & 255
    packed = (exponents.reshape(rows, -1, 4).long() << (torch.arange(4) * 8)).sum(-1)
    packed = packed.int().T.contiguous()
    source, indices = source.cuda(), indices.cuda()
    destination = torch.empty((rows, hidden), device="cuda", dtype=torch.float8_e4m3fn)
    destination_scales = torch.empty(packed.shape, device="cuda", dtype=torch.int32)
    descriptor = torch.tensor([[3, 128, 256, 0]], dtype=torch.int32)
    queue = descriptor.cuda()
    operation = peer_gather_quantize_publish if routed else peer_quantize_publish
    if compiled:
        torch.compiler.reset()
        operation = torch.compile(operation, fullgraph=True, dynamic=False)
    args = (source, indices) if routed else (source,)
    for _ in range(3):
        destination.view(torch.uint8).fill_(127)
        destination_scales.fill_(-1)
        queue.copy_(descriptor)
        if skip_padding:
            destination.zero_()
            destination_scales.fill_(0x69696969)
        kwargs = {"skip_padding": skip_padding} if routed else {}
        operation(*args, destination, destination_scales, queue, 0, gran, **kwargs)
        torch.testing.assert_close(
            destination.cpu().float(), expected.float(), atol=0, rtol=0
        )
        torch.testing.assert_close(destination_scales.cpu(), packed, atol=0, rtol=0)
        want = descriptor.clone()
        want[0, 3] = 1
        torch.testing.assert_close(queue.cpu(), want, atol=0, rtol=0)


@pytest.mark.parametrize("rows,hidden", [(1, 128), (137, 2048)])
@pytest.mark.parametrize("transposed_storage", [False, True])
def test_packed_decode(rows, hidden, transposed_storage):
    from mono_ep.scheduled import _dequantize_packed

    torch.manual_seed(617)
    x = torch.randn(rows, hidden, device="cuda").to(torch.float8_e4m3fn)
    exponents = torch.randint(
        80, 160, (rows, hidden // 32), device="cuda", dtype=torch.int32
    )
    exponents[:, 0] = 0
    shifts = torch.arange(4, device="cuda", dtype=torch.int64) * 8
    packed = (exponents.reshape(rows, -1, 4).long() << shifts).sum(-1).int()
    if transposed_storage:
        packed = packed.T.contiguous().T
    expected = (
        x.float() * (exponents << 23).view(torch.float32).repeat_interleave(32, -1)
    ).bfloat16()
    torch.testing.assert_close(_dequantize_packed(x, packed), expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("pattern", ["mixed", "skewed", "empty"])
@pytest.mark.parametrize("skip_padding", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_packed_bf16_publication(pattern, skip_padding, compiled):
    if not torch.cuda.is_available():
        pytest.skip("Peer publication requires CUDA")
    pytest.importorskip("triton")
    from mono_ep.packing import DeviceExpertLayout
    from mono_ep.peer import peer_gather_bf16_publish, peer_packed_bf16_publish

    torch.compiler.reset()
    torch.manual_seed(623)
    tokens, topk, experts, hidden = 137, 2, 3, 256
    source = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(tokens, topk, device="cuda")
    assignments = torch.randint(
        experts, (tokens, topk), device="cuda", dtype=torch.int32
    )
    assignments[::5] = -1
    if pattern == "skewed":
        assignments[assignments >= 0] = 0
    elif pattern == "empty":
        assignments.fill_(-1)
    layout = DeviceExpertLayout.build(assignments, experts, 128)
    slots = layout.row_assignments
    rows = slots.numel()
    expected_tokens = torch.where(slots >= 0, slots // topk, -1).int()
    expected = source[expected_tokens.clamp_min(0)].masked_fill((slots < 0)[:, None], 0)
    expected_scores = scores.flatten()[slots.clamp_min(0)].masked_fill(slots < 0, 0)
    destination = torch.empty(rows, hidden, device="cuda", dtype=torch.bfloat16)
    queue = torch.empty(experts, 4, device="cuda", dtype=torch.int32)
    out_ids, out_tokens, out_slots = [
        torch.empty(rows, device="cuda", dtype=torch.int32) for _ in range(3)
    ]
    out_ends = torch.empty_like(layout.ends)
    out_scores = torch.empty(rows, device="cuda")
    operation = (
        torch.compile(peer_packed_bf16_publish, fullgraph=True)
        if compiled
        else peer_packed_bf16_publish
    )
    gather = (
        torch.compile(peer_gather_bf16_publish, fullgraph=True)
        if compiled
        else peer_gather_bf16_publish
    )
    for _ in range(3):
        destination.fill_(float("nan"))
        if skip_padding:
            destination.zero_()
        queue.zero_()
        for value in (out_ids, out_tokens, out_slots, out_ends):
            value.fill_(-123)
        out_scores.fill_(float("nan"))
        operation(
            source,
            slots,
            layout.expert_ids,
            layout.starts,
            layout.padded_sizes,
            layout.ends,
            scores,
            destination,
            queue,
            out_ids,
            out_tokens,
            out_slots,
            out_ends,
            out_scores,
            skip_padding,
        )
        for actual, want in zip(
            (destination, out_ids, out_tokens, out_slots, out_ends, out_scores),
            (
                expected,
                layout.expert_ids,
                expected_tokens,
                slots.int(),
                layout.ends,
                expected_scores,
            ),
            strict=True,
        ):
            torch.testing.assert_close(actual, want, atol=0, rtol=0)
        expected_queue = torch.stack(
            (
                torch.arange(experts, device="cuda", dtype=torch.int32),
                layout.starts,
                layout.padded_sizes,
                torch.ones(experts, device="cuda", dtype=torch.int32),
            ),
            dim=1,
        )
        torch.testing.assert_close(queue, expected_queue, atol=0, rtol=0)
        queue[:, 3] = 0
        destination.fill_(float("nan"))
        if skip_padding:
            destination.zero_()
        gather(source, expected_tokens, destination, queue, experts, skip_padding)
        torch.testing.assert_close(destination, expected, atol=0, rtol=0)
        torch.testing.assert_close(queue, expected_queue, atol=0, rtol=0)
