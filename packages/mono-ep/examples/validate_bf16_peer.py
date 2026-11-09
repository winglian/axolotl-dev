# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Two-rank packed BF16 forward/backward peer composition validation."""

import argparse
import json
import os

import deep_gemm
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mono_ep.bf16 import (
    bf16_packed_backward,
    bf16_packed_backward_combine,
    bf16_packed_forward,
    bf16_packed_forward_combine,
)
from mono_ep.combine import combine_wait_copy
from mono_ep.packing import DeviceExpertLayout
from mono_ep.peer import peer_packed_bf16_publish


def run(compiled=False, skip_padding=False):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 2
    peer = 1 - rank
    torch.manual_seed(994 + rank)
    token_count, topk, hidden, inner = 137, 2, 256, 128
    rows = ((token_count * topk + 2 * 127 + 127) // 128) * 128
    handles = []

    def shared(shape, dtype):
        local = symm_mem.empty(*shape, device="cuda", dtype=dtype)
        handle = symm_mem.rendezvous(local, group=dist.group.WORLD)
        handles.append(handle)
        return local, handle.get_buffer(peer, shape, dtype)

    x, peer_x = shared((rows, hidden), torch.bfloat16)
    dy, peer_dy = shared((rows, hidden), torch.bfloat16)
    ready, peer_ready = shared((2, 4), torch.int32)
    backward_ready, peer_backward_ready = shared((2, 4), torch.int32)
    output, peer_output = shared((token_count, hidden), torch.float32)
    dx, peer_dx = shared((token_count, hidden), torch.float32)
    ds, peer_ds = shared((token_count * topk, 1), torch.float32)
    complete, peer_complete = shared((1, 4), torch.int32)
    dx_complete, peer_dx_complete = shared((1, 4), torch.int32)
    ds_complete, peer_ds_complete = shared((1, 4), torch.int32)
    ids, peer_ids = shared((rows,), torch.int32)
    tokens, peer_tokens = shared((rows,), torch.int32)
    slots, peer_slots = shared((rows,), torch.int32)
    ends, peer_ends = shared((2,), torch.int32)
    scores, peer_scores = shared((rows,), torch.float32)
    source = torch.randn(token_count, hidden, device="cuda", dtype=torch.bfloat16)
    gradient = torch.randn_like(source)
    source_scores = torch.rand(token_count, topk, device="cuda")
    w1 = (torch.randn(2, 2 * inner, hidden, device="cuda") / hidden**0.5).bfloat16()
    w2 = (torch.randn(2, hidden, inner, device="cuda") / inner**0.5).bfloat16()

    masters = [torch.nn.Parameter(w.float()) for w in (w1, w2)]
    reference_masters = [torch.nn.Parameter(w.detach().clone()) for w in masters]
    optimizer = torch.optim.AdamW(masters, lr=1e-3)
    reference_optimizer = torch.optim.AdamW(reference_masters, lr=1e-3)
    consumed = [torch.empty_like(t) for t in (output, dx, ds)]

    def consume():
        for source_value, destination, flag in zip(
            (output, dx, ds),
            consumed,
            (complete, dx_complete, ds_complete),
            strict=True,
        ):
            combine_wait_copy(source_value, destination, flag)

    def gather(value):
        result = [torch.empty_like(value) for _ in range(2)]
        dist.all_gather(result, value)
        return result[peer]

    incoming, incoming_dy, incoming_scores = map(
        gather, (source, gradient, source_scores)
    )
    forward = bf16_packed_forward_combine
    backward = bf16_packed_backward_combine
    dispatch = peer_packed_bf16_publish
    if compiled:
        dispatch = torch.compile(dispatch, fullgraph=True)
        forward = torch.compile(forward, fullgraph=True)
        backward = torch.compile(backward, fullgraph=True)
    deep_gemm.set_num_sms(32)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda._sleep(1)
    for trial, pattern in enumerate(("mixed", "skewed", "empty")):
        w1, w2 = [w.detach().bfloat16() for w in masters]
        assignments = torch.randint(
            2, (token_count, topk), device="cuda", dtype=torch.int32
        )
        assignments[::7] = -1
        if pattern == "skewed":
            assignments[assignments >= 0] = 0
        elif pattern == "empty":
            assignments.fill_(-1)
        send_layout = DeviceExpertLayout.build(assignments, 2, 128)
        received_layout = DeviceExpertLayout.build(gather(assignments), 2, 128)
        valid_x, valid_dy = (
            received_layout.pack(incoming),
            received_layout.pack(incoming_dy),
        )
        reference_scores = received_layout.scores(incoming_scores)
        queue = received_layout.chunk_descriptors()
        queue[:, 3] = 1
        expected_y, expected_z = bf16_packed_forward(
            valid_x, w1, w2, queue, reference_scores, received_layout.expert_ids
        )
        expected_dx, expected_ds, expected_dw1, expected_dw2 = bf16_packed_backward(
            valid_dy,
            valid_x,
            w1,
            w2,
            queue,
            reference_scores,
            expected_z,
            received_layout.expert_ids,
            received_layout.ends,
        )
        owned_y, owned_dx, owned_ds = map(
            gather,
            (
                received_layout.combine(expected_y.float(), token_count),
                received_layout.combine(expected_dx.float(), token_count),
                received_layout.restore_scores(expected_ds, incoming_scores).reshape(
                    -1, 1
                ),
            ),
        )

        def send(value, destination, publication, send_layout=send_layout):
            dispatch(
                value,
                send_layout.row_assignments,
                send_layout.expert_ids,
                send_layout.starts,
                send_layout.padded_sizes,
                send_layout.ends,
                source_scores,
                destination,
                publication,
                peer_ids,
                peer_tokens,
                peer_slots,
                peer_ends,
                peer_scores,
                skip_padding,
            )

        # Cold compilation must finish before consumers can wait on producers.
        ready.zero_()
        backward_ready.zero_()
        x.zero_()
        dy.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        send(source, peer_x, peer_ready)
        send(gradient, peer_dy, peer_backward_ready)
        torch.cuda.synchronize()
        dist.barrier()
        for value in (output, dx, ds, complete, dx_complete, ds_complete):
            value.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        z = forward(
            x, w1, w2, ready, scores, ids, tokens, peer_output, peer_complete, 0
        )
        backward(
            dy,
            x,
            w1,
            w2,
            backward_ready,
            scores,
            z,
            ids,
            ends,
            tokens,
            slots,
            peer_dx,
            peer_ds,
            peer_dx_complete,
            peer_ds_complete,
            0,
        )
        torch.cuda.synchronize()
        dist.barrier()
        consume()
        torch.cuda.synchronize()
        for value in (output, dx, ds, complete, dx_complete, ds_complete):
            value.zero_()
        ready.zero_()
        backward_ready.zero_()
        for value in (ids, tokens, slots, ends):
            value.fill_(-123)
        scores.fill_(float("nan"))
        x.fill_(float("nan"))
        dy.fill_(float("nan"))
        if skip_padding:
            x.zero_()
            dy.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        with torch.cuda.stream(consumer):
            z = forward(
                x, w1, w2, ready, scores, ids, tokens, peer_output, peer_complete, 0
            )
        with torch.cuda.stream(producer):
            torch.cuda._sleep(10_000_000)
            send(source, peer_x, peer_ready)
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.stream(consumer):
            dw1, dw2 = backward(
                dy,
                x,
                w1,
                w2,
                backward_ready,
                scores,
                z,
                ids,
                ends,
                tokens,
                slots,
                peer_dx,
                peer_ds,
                peer_dx_complete,
                peer_ds_complete,
                0,
            )
        with torch.cuda.stream(producer):
            torch.cuda._sleep(10_000_000)
            send(gradient, peer_dy, peer_backward_ready)
        torch.cuda.synchronize()
        dist.barrier()
        consume()
        torch.cuda.synchronize()
        errors = []
        for actual, expected in zip(
            (*consumed, dw1, dw2),
            (owned_y, owned_dx, owned_ds, expected_dw1, expected_dw2),
            strict=True,
        ):
            assert torch.isfinite(actual).all()
            error = (actual - expected).norm() / expected.norm().clamp_min(1e-8)
            assert error.item() < 1e-5
            errors.append(error.item())
        for flag in (complete, dx_complete, ds_complete):
            assert flag[0, 3].item() == 1
        for parameter, grad in zip(masters, (dw1, dw2), strict=True):
            parameter.grad = grad
        for parameter, grad in zip(
            reference_masters, (expected_dw1, expected_dw2), strict=True
        ):
            parameter.grad = grad
        optimizer.step()
        reference_optimizer.step()
        for parameter, expected in zip(masters, reference_masters, strict=True):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        print(
            json.dumps(
                dict(
                    rank=rank,
                    trial=trial,
                    routing=pattern,
                    routed_bf16_producer=True,
                    compiled=compiled,
                    skip_padding=skip_padding,
                    relative_l2=errors,
                    fp32_master_adamw_exact=True,
                    consumer_acquire=True,
                )
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--skip-padding", action="store_true")
    args = parser.parse_args()
    run(args.compiled, args.skip_padding)
