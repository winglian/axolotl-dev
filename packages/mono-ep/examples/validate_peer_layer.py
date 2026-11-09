# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compose routed dispatch, FP8 SwiGLU experts and peer combine on two GPUs."""

import argparse
import json
import os

import deep_gemm
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mono_ep.combine import combine_wait_copy
from mono_ep.peer import peer_gather_quantize_publish
from mono_ep.quantization import quantize_fp8
from mono_ep.scheduled import fp8_expert_chunk_combine


def dispatch(*args):
    peer_gather_quantize_publish(*args)


def finish(*args):
    combine_wait_copy(*args)


def reference_quantize(x):
    blocks = x.float().unflatten(-1, (-1, 32))
    scale = torch.pow(
        2.0,
        torch.ceil(
            torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)
        ),
    )
    return (
        (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    ).flatten(-2)


def check_backward(
    rank,
    raw_all,
    scores_all,
    w1,
    w2,
    values,
    scales,
    received_scores,
    z_buffers,
    token_ids,
    compiled,
    upstream=None,
    reference_upstream=None,
    reference_weights=None,
    communication_state=None,
):
    from mono_ep.scheduled import fp8_expert_chunk_backward_combine

    rows, hidden = values.shape[1:]
    tokens, inner = raw_all[rank].shape[0], w2.shape[-1]
    dy = (
        torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        if upstream is None
        else upstream
    )
    reference_dy = dy if reference_upstream is None else reference_upstream
    rw1, rw2 = (w1, w2) if reference_weights is None else reference_weights
    communication_state = {} if communication_state is None else communication_state
    dy_all = [torch.empty_like(dy) for _ in range(2)]
    dist.all_gather(dy_all, reference_dy)
    w1t, s1t = quantize_fp8(w1.transpose(1, 2).contiguous())
    w2t, s2t = quantize_fp8(w2.transpose(1, 2).contiguous())
    s2t = deep_gemm.transform_sf_into_required_layout(
        s2t, inner, hidden, (1, 32), num_groups=1
    )
    reference_dx, reference_ds = [], []
    reference_dw1, reference_dw2 = (
        torch.zeros_like(w1, dtype=torch.float32),
        torch.zeros_like(w2, dtype=torch.float32),
    )
    for sender in range(2):
        px = torch.nn.functional.pad(raw_all[sender], (0, 0, 0, rows - tokens))
        pdy = torch.nn.functional.pad(dy_all[sender], (0, 0, 0, rows - tokens))
        ps = torch.nn.functional.pad(scores_all[sender][:, rank], (0, rows - tokens))
        decoded_x, decoded_dy = (
            reference_quantize(px).bfloat16(),
            reference_quantize(pdy).bfloat16(),
        )
        zr = (
            (decoded_x.float() @ reference_quantize(rw1[0]).T)
            .bfloat16()
            .float()
            .requires_grad_()
        )
        sr = ps.detach().clone().requires_grad_()
        gate, up = zr.chunk(2, -1)
        h = torch.nn.functional.silu(gate) * up * sr[:, None]
        dh = (
            decoded_dy.float() @ reference_quantize(rw2[0].T.contiguous()).T
        ).bfloat16()
        dz, ds = torch.autograd.grad(h, (zr, sr), dh.float())
        dz = dz.bfloat16()
        dx = (
            reference_quantize(dz) @ reference_quantize(rw1[0].T.contiguous()).T
        ).bfloat16()
        reference_dx.append(dx[:tokens].float())
        score_grad = torch.zeros(tokens, 2, device="cuda")
        score_grad[:, rank] = ds[:tokens]
        reference_ds.append(score_grad)
        reference_dw1 += (
            reference_quantize(dz.T.contiguous())
            @ reference_quantize(decoded_x.T.contiguous()).T
        )[None]
        reference_dw2 += (
            reference_quantize(decoded_dy.T.contiguous())
            @ reference_quantize(h.bfloat16().T.contiguous()).T
        )[None]
    reference_dx, reference_ds = torch.stack(reference_dx), torch.stack(reference_ds)
    dist.all_reduce(reference_dx)
    dist.all_reduce(reference_ds)
    expected = (
        reference_dx[rank],
        reference_ds[rank].reshape(-1, 1),
        reference_dw1,
        reference_dw2,
    )

    if "buffers" not in communication_state:
        buffers = [
            symm_mem.empty(2, rows, hidden, device="cuda", dtype=torch.uint8),
            symm_mem.empty(2, hidden // 128, rows, device="cuda", dtype=torch.int32),
            symm_mem.empty(2, 4, device="cuda", dtype=torch.int32),
            symm_mem.empty(tokens, hidden, device="cuda", dtype=torch.float32),
            symm_mem.empty(tokens * 2, 1, device="cuda", dtype=torch.float32),
            symm_mem.empty(2, 4, device="cuda", dtype=torch.int32),
            symm_mem.empty(2, 4, device="cuda", dtype=torch.int32),
        ]
        handles = [symm_mem.rendezvous(t, group=dist.group.WORLD) for t in buffers]
        peers = [
            h.get_buffer(1 - rank, tuple(t.shape), t.dtype)
            for h, t in zip(handles, buffers, strict=True)
        ]
        communication_state.update(buffers=buffers, handles=handles, peers=peers)
    buffers, peers = communication_state["buffers"], communication_state["peers"]
    incoming, incoming_scales, ready, dx_acc, ds_acc, dx_flags, ds_flags = buffers
    (
        peer_incoming,
        peer_scales,
        peer_ready,
        peer_dx,
        peer_ds,
        peer_dx_flags,
        peer_ds_flags,
    ) = peers
    incoming, peer_incoming = (
        incoming.view(torch.float8_e4m3fn),
        peer_incoming.view(torch.float8_e4m3fn),
    )
    dx_out, ds_out = torch.empty_like(dx_acc), torch.empty_like(ds_acc)
    score_ids = torch.where(token_ids >= 0, token_ids * 2 + rank, -1)
    producer = (
        torch.compile(dispatch, fullgraph=True, dynamic=False) if compiled else dispatch
    )
    compute = (
        torch.compile(fp8_expert_chunk_backward_combine, fullgraph=True, dynamic=False)
        if compiled
        else fp8_expert_chunk_backward_combine
    )
    consumer = (
        torch.compile(finish, fullgraph=True, dynamic=False) if compiled else finish
    )
    if "streams" not in communication_state:
        communication_state["streams"] = (
            torch.cuda.Stream(),
            [torch.cuda.Stream(), torch.cuda.Stream()],
            [torch.cuda.Stream(), torch.cuda.Stream()],
        )
    sender_stream, compute_streams, finish_streams = communication_state["streams"]

    def reset_backward():
        incoming.zero_()
        incoming_scales.zero_()
        ready.copy_(
            torch.tensor([[0, 0, rows, 0]] * 2, device="cuda", dtype=torch.int32)
        )
        for tensor in (dx_acc, ds_acc, dx_flags, ds_flags):
            tensor.zero_()
        dx_out.fill_(7)
        ds_out.fill_(7)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

    def send_backward():
        producer(
            dy,
            token_ids,
            incoming[rank],
            incoming_scales[rank],
            ready[rank : rank + 1],
            0,
        )
        producer(
            dy,
            token_ids,
            peer_incoming[rank],
            peer_scales[rank],
            peer_ready[rank : rank + 1],
            0,
        )

    def calculate_backward(sender):
        dx_target, ds_target, dx_queue, ds_queue = (
            (dx_acc, ds_acc, dx_flags, ds_flags)
            if sender == rank
            else (peer_dx, peer_ds, peer_dx_flags, peer_ds_flags)
        )
        return compute(
            incoming[sender],
            incoming_scales[sender].T,
            values[sender],
            scales[sender].T,
            w1t,
            s1t,
            w2t,
            s2t,
            ready[sender : sender + 1],
            received_scores[sender],
            z_buffers[sender],
            token_ids,
            score_ids,
            dx_target,
            dx_queue,
            ds_target,
            ds_queue,
            rank,
        )

    reset_backward()
    send_backward()
    torch.cuda.synchronize()
    dist.barrier()
    warm = [calculate_backward(sender) for sender in range(2)]
    torch.cuda.synchronize()
    dist.barrier()
    consumer(dx_acc, dx_out, dx_flags)
    consumer(ds_acc, ds_out, ds_flags)
    torch.cuda.synchronize()
    dist.barrier()
    del warm
    errors = []
    for _trial in range(3):
        reset_backward()
        with torch.cuda.stream(finish_streams[0]):
            consumer(dx_acc, dx_out, dx_flags)
        with torch.cuda.stream(finish_streams[1]):
            consumer(ds_acc, ds_out, ds_flags)
        weight_grads = []
        for sender, stream in enumerate(compute_streams):
            with torch.cuda.stream(stream):
                weight_grads.append(calculate_backward(sender))
        with torch.cuda.stream(sender_stream):
            torch.cuda._sleep(10_000_000 * (rank + 1))
            send_backward()
        torch.cuda.synchronize()
        dw1 = sum(gradient[0] for gradient in weight_grads)
        dw2 = sum(gradient[1] for gradient in weight_grads)
        trial_errors = []
        for got, want in zip((dx_out, ds_out, dw1, dw2), expected, strict=True):
            error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(
                1e-8
            )
            assert torch.isfinite(error) and error < 0.01, error
            trial_errors.append(error.item())
        errors.append(trial_errors)
        dist.barrier()
    print(
        json.dumps(
            dict(
                rank=rank,
                compiled=compiled,
                order=["dx", "ds", "dw1", "dw2"],
                backward_relative_l2=errors,
            )
        ),
        flush=True,
    )

    return (dw1, dw2), (reference_dw1, reference_dw2)


def run(compiled, backward=False, training=False):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1031 + rank)
    tokens, rows, hidden, inner = 137, 256, 256, 128
    raw = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(tokens, 2, device="cuda").softmax(-1)
    payload_scores = torch.nn.functional.pad(scores.T.contiguous(), (0, rows - tokens))
    token_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    token_ids[tokens:] = -1
    expert_ids = torch.zeros(rows, device="cuda", dtype=torch.int32)
    w1 = (
        torch.randn(1, 2 * inner, hidden, device="cuda", dtype=torch.bfloat16)
        / hidden**0.5
    )
    w2 = torch.randn(1, hidden, inner, device="cuda", dtype=torch.bfloat16) / inner**0.5
    if training:
        w1, w2 = w1.float(), w2.float()
    reference_weights = [w1.clone(), w2.clone()]
    optimizer = (
        torch.optim.AdamW([w1, w2], lr=1e-3, foreach=False) if training else None
    )
    reference_optimizer = (
        torch.optim.AdamW(reference_weights, lr=1e-3, foreach=False)
        if training
        else None
    )
    target = torch.randn(tokens, hidden, device="cuda")
    backward_state = {}
    raw_all, scores_all = (
        [torch.empty_like(raw) for _ in range(2)],
        [torch.empty_like(scores) for _ in range(2)],
    )
    dist.all_gather(raw_all, raw)
    dist.all_gather(scores_all, scores)

    buffers = [
        symm_mem.empty(2, rows, hidden, device="cuda", dtype=torch.uint8),
        symm_mem.empty(2, hidden // 128, rows, device="cuda", dtype=torch.int32),
        symm_mem.empty(2, rows, device="cuda", dtype=torch.float32),
        symm_mem.empty(2, 4, device="cuda", dtype=torch.int32),
        symm_mem.empty(tokens, hidden, device="cuda", dtype=torch.float32),
        symm_mem.empty(2, 4, device="cuda", dtype=torch.int32),
    ]
    handles = [symm_mem.rendezvous(t, group=dist.group.WORLD) for t in buffers]
    peers = [
        h.get_buffer(1 - rank, tuple(t.shape), t.dtype)
        for h, t in zip(handles, buffers, strict=True)
    ]
    values, scales, received_scores, ready, accumulation, complete = buffers
    (
        peer_values,
        peer_scales,
        peer_scores,
        peer_ready,
        peer_accumulation,
        peer_complete,
    ) = peers
    values = values.view(torch.float8_e4m3fn)
    peer_values = peer_values.view(torch.float8_e4m3fn)
    z_buffers = torch.empty(2, rows, 2 * inner, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(accumulation)
    producer = (
        torch.compile(dispatch, fullgraph=True, dynamic=False) if compiled else dispatch
    )
    compute = (
        torch.compile(fp8_expert_chunk_combine, fullgraph=True, dynamic=False)
        if compiled
        else fp8_expert_chunk_combine
    )
    consumer = (
        torch.compile(finish, fullgraph=True, dynamic=False) if compiled else finish
    )
    send_stream, finish_stream = torch.cuda.Stream(), torch.cuda.Stream()
    compute_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    deep_gemm.set_num_sms(32)

    def reset():
        values.zero_()
        scales.zero_()
        received_scores.zero_()
        ready.copy_(
            torch.tensor([[0, 0, rows, 0]] * 2, device="cuda", dtype=torch.int32)
        )
        accumulation.zero_()
        complete.zero_()
        output.fill_(7)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

    def send():
        received_scores[rank].copy_(payload_scores[rank])
        producer(raw, token_ids, values[rank], scales[rank], ready[rank : rank + 1], 0)
        peer_scores[rank].copy_(payload_scores[1 - rank])
        producer(
            raw,
            token_ids,
            peer_values[rank],
            peer_scales[rank],
            peer_ready[rank : rank + 1],
            0,
        )

    def calculate(sender):
        target, flags = (
            (accumulation, complete)
            if sender == rank
            else (peer_accumulation, peer_complete)
        )
        compute(
            values[sender],
            scales[sender].T,
            qw1,
            sw1,
            qw2,
            sw2,
            ready[sender : sender + 1],
            received_scores[sender],
            z_buffers[sender],
            expert_ids,
            token_ids,
            target,
            flags,
            rank,
        )

    for step in range(3 if training else 1):
        qw1, sw1 = quantize_fp8(w1)
        sw1 = deep_gemm.transform_sf_into_required_layout(
            sw1, 2 * inner, hidden, (1, 32), num_groups=1
        )
        qw2, sw2 = quantize_fp8(w2)
        references = []
        for sender in range(2):
            z = (
                reference_quantize(raw_all[sender])
                @ reference_quantize(reference_weights[0][0]).T
            ).bfloat16()
            gate, up = z.float().chunk(2, -1)
            h = (
                torch.nn.functional.silu(gate) * up * scores_all[sender][:, rank, None]
            ).bfloat16()
            references.append(
                (reference_quantize(h) @ reference_quantize(reference_weights[1][0]).T)
                .bfloat16()
                .float()
            )
        expected = torch.stack(references)
        dist.all_reduce(expected)
        expected = expected[rank]

        reset()
        send()
        torch.cuda.synchronize()
        dist.barrier()
        for sender in range(2):
            calculate(sender)
        torch.cuda.synchronize()
        dist.barrier()
        consumer(accumulation, output, complete)
        torch.cuda._sleep(1)
        torch.cuda.synchronize()
        dist.barrier()
        errors = []
        for _trial in range(3):
            reset()
            with torch.cuda.stream(finish_stream):
                consumer(accumulation, output, complete)
            for sender, stream in enumerate(compute_streams):
                with torch.cuda.stream(stream):
                    calculate(sender)
            with torch.cuda.stream(send_stream):
                torch.cuda._sleep(10_000_000 * (rank + 1))
                send()
            torch.cuda.synchronize()
            error = (output - expected).norm() / expected.norm()
            assert torch.isfinite(error) and error < 0.01, error
            errors.append(error.item())
            dist.barrier()
        print(
            json.dumps(
                dict(
                    rank=rank, step=step, compiled=compiled, forward_relative_l2=errors
                )
            ),
            flush=True,
        )
        torch.cuda.synchronize()
        dist.barrier()
        if backward or training:
            gradients, reference_gradients = check_backward(
                rank,
                raw_all,
                scores_all,
                w1,
                w2,
                values,
                scales,
                received_scores,
                z_buffers,
                token_ids,
                compiled,
                upstream=(2 * (output - target) / output.numel()).bfloat16()
                if training
                else None,
                reference_upstream=(
                    2 * (expected - target) / expected.numel()
                ).bfloat16()
                if training
                else None,
                reference_weights=reference_weights if training else None,
                communication_state=backward_state,
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            before = [w.clone() for w in (w1, w2)]
            for weight, grad, reference_weight, reference_grad in zip(
                (w1, w2), gradients, reference_weights, reference_gradients, strict=True
            ):
                weight.grad = grad
                reference_weight.grad = reference_grad
            optimizer.step()
            reference_optimizer.step()
            weight_errors = []
            for weight, reference_weight, previous in zip(
                (w1, w2), reference_weights, before, strict=True
            ):
                error = (weight - reference_weight).norm() / reference_weight.norm()
                assert torch.isfinite(error) and error < 0.001, error
                assert (weight != previous).any()
                weight_errors.append(error.item())
            for state in optimizer.state.values():
                assert all(torch.isfinite(value).all() for value in state.values())
            print(
                json.dumps(
                    dict(
                        rank=rank,
                        step=step,
                        loss=(output - target).square().mean().item(),
                        reference_loss=(expected - target).square().mean().item(),
                        weight_relative_l2=weight_errors,
                    )
                ),
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    run(args.compiled, args.backward, args.training)
