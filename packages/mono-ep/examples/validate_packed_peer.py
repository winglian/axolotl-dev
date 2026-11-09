# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compact multi-expert peer dispatch consumed by independent ready-queue GEMMs."""

import argparse
import json
import os

import deep_gemm
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mono_ep.chunk import chunk_fp8_out
from mono_ep.packing import DeviceExpertLayout
from mono_ep.peer import peer_packed_quantize_publish
from mono_ep.quantization import quantize_fp8
from mono_ep.scheduled import fp8_packed_backward


def layout(indices, experts):
    packed = DeviceExpertLayout.build(indices, experts, 128)
    return (
        packed.token_ids(),
        packed.chunk_descriptors(),
        packed.expert_ids,
        packed.ends,
        packed.row_assignments,
    )


def gather(t):
    parts = [torch.empty_like(t) for _ in range(2)]
    dist.all_gather(parts, t)
    return parts[1 - dist.get_rank()]


def run(compiled, tokens=137, hidden=256):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 2
    torch.manual_seed(351 + rank)
    topk, experts, width = 3, 4, 256
    capacity = ((tokens * topk + experts * 127 + 127) // 128) * 128
    values = symm_mem.empty(capacity, hidden, device="cuda", dtype=torch.uint8)
    scales = symm_mem.empty(hidden // 128, capacity, device="cuda", dtype=torch.int32)
    queue = symm_mem.empty(experts, 4, device="cuda", dtype=torch.int32)
    metadata = [
        symm_mem.empty(capacity, device="cuda", dtype=torch.int32) for _ in range(3)
    ] + [
        symm_mem.empty(experts, device="cuda", dtype=torch.int32),
        symm_mem.empty(capacity, device="cuda", dtype=torch.float32),
    ]
    buffers = (values, scales, queue, *metadata)
    handles = [symm_mem.rendezvous(t, group=dist.group.WORLD) for t in buffers]
    peer_values, peer_scales, peer_queue, *peer_metadata = [
        h.get_buffer(1 - rank, tuple(t.shape), t.dtype)
        for h, t in zip(handles, buffers, strict=True)
    ]
    source = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn(experts, width, hidden, device="cuda", dtype=torch.bfloat16)
        / hidden**0.5
    )
    qw, sw = quantize_fp8(weight)
    packed_sw = deep_gemm.transform_sf_into_required_layout(
        sw, width, hidden, (1, 32), num_groups=experts
    )
    decoded_w = qw.float() * sw.repeat_interleave(32, -1)
    output = torch.empty(capacity, width, device="cuda", dtype=torch.bfloat16)
    deep_gemm.set_num_sms(16)
    prepare = (
        torch.compile(layout, fullgraph=True, dynamic=False) if compiled else layout
    )
    producer = (
        torch.compile(peer_packed_quantize_publish, fullgraph=True, dynamic=False)
        if compiled
        else peer_packed_quantize_publish
    )
    consumer = (
        torch.compile(chunk_fp8_out, fullgraph=True, dynamic=False)
        if compiled
        else chunk_fp8_out
    )
    sender = torch.cuda.Stream()
    consumers = [torch.cuda.Stream() for _ in range(experts)]
    for case in ("mixed", "skewed", "masked"):
        ids = torch.randint(
            experts - 1, (tokens, topk), device="cuda", dtype=torch.int32
        )
        ids[::7] = -1
        if case == "skewed":
            ids.fill_(0)
            ids[:17, 0] = 2
        if case == "masked":
            ids.fill_(-1)
        token_ids, descriptors, expert_ids, ends, slots = prepare(ids, experts)
        starts, sizes = descriptors[:, 1].contiguous(), descriptors[:, 2].contiguous()
        source_scores = torch.rand(tokens, topk, device="cuda")
        packed_scores = source_scores.flatten()[slots.clamp_min(0)].masked_fill(
            slots < 0, 0
        )
        expected_metadata = [
            gather(t) for t in (expert_ids, token_ids, slots.int(), ends, packed_scores)
        ]
        peer_ids, peer_ends = gather(expert_ids), gather(ends)
        assert token_ids.numel() == capacity
        px = source[token_ids.clamp_min(0).long()].masked_fill(
            (token_ids < 0)[:, None], 0
        )
        qx, sx = quantize_fp8(px)
        expected_values = gather(qx.view(torch.uint8))
        expected_scales = gather(
            deep_gemm.transform_sf_into_required_layout(
                sx, capacity, hidden, (1, 32)
            ).T.contiguous()
        )
        decoded = gather(qx.float() * sx.repeat_interleave(32, -1))
        peer_descriptors = gather(descriptors)
        expected = torch.full_like(output, 7)
        active = torch.zeros(capacity, device="cuda", dtype=torch.bool)
        for expert, start, count, _ in peer_descriptors.tolist():
            active[start : start + count] = True
            expected[start : start + count] = (
                decoded[start : start + count] @ decoded_w[expert].T
            ).bfloat16()

        def reset():
            values.zero_()
            scales.zero_()
            queue.zero_()
            for tensor in metadata:
                tensor.fill_(-17)
            output.fill_(7)
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()

        def send(
            payload,
            slots=slots,
            expert_ids=expert_ids,
            starts=starts,
            sizes=sizes,
            ends=ends,
            source_scores=source_scores,
        ):
            producer(
                payload,
                slots,
                expert_ids,
                starts,
                sizes,
                ends,
                source_scores,
                peer_values.view(torch.float8_e4m3fn),
                peer_scales,
                peer_queue,
                *peer_metadata,
            )

        reset()
        send(source)
        torch.cuda.synchronize()
        dist.barrier()
        for expert in range(experts):
            consumer(
                values.view(torch.float8_e4m3fn),
                scales.T,
                qw,
                packed_sw,
                output,
                queue,
                expert,
            )
        torch.cuda._sleep(1)
        torch.cuda.synchronize()
        dist.barrier()
        errors = []
        for _trial in range(3):
            reset()
            for expert, stream in enumerate(consumers):
                with torch.cuda.stream(stream):
                    consumer(
                        values.view(torch.float8_e4m3fn),
                        scales.T,
                        qw,
                        packed_sw,
                        output,
                        queue,
                        expert,
                    )
            with torch.cuda.stream(sender):
                torch.cuda._sleep(10_000_000 * (rank + 1))
                send(source)
            torch.cuda.synchronize()
            torch.testing.assert_close(values, expected_values, atol=0, rtol=0)
            torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
            torch.testing.assert_close(
                queue[:, :3], peer_descriptors[:, :3], atol=0, rtol=0
            )
            assert (queue[:, 3] == 1).all()
            for got, want in zip(metadata, expected_metadata, strict=True):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
            relative = (
                (output[active].float() - expected[active].float()).norm()
                / expected[active].float().norm().clamp_min(1e-8)
            ).item()
            assert relative < 0.01, relative
            torch.testing.assert_close(
                output[~active], expected[~active], atol=0, rtol=0
            )
            errors.append(relative)
            dist.barrier()
        saved_x, saved_sx = values.view(torch.float8_e4m3fn).clone(), scales.T.clone()
        z = output.clone().masked_fill(~active[:, None], 0)
        scores = torch.rand(capacity, device="cuda").masked_fill(peer_ids < 0, 0)
        w2 = (
            torch.randn(
                experts, hidden, width // 2, device="cuda", dtype=torch.bfloat16
            )
            / (width // 2) ** 0.5
        )
        w1t, s1t = quantize_fp8(weight.transpose(1, 2).contiguous())
        w2t, s2t = quantize_fp8(w2.transpose(1, 2).contiguous())
        packed_s2t = deep_gemm.transform_sf_into_required_layout(
            s2t, width // 2, hidden, (1, 32), num_groups=experts
        )
        source = torch.randn_like(source)
        packed_dy = source[token_ids.clamp_min(0).long()].masked_fill(
            (token_ids < 0)[:, None], 0
        )
        qdy, sdy = quantize_fp8(packed_dy)
        decoded_dy = gather(qdy.float() * sdy.repeat_interleave(32, -1)).bfloat16()

        def ref_quant(t):
            blocks = t.float().reshape(t.shape[0], -1, 32)
            scale = torch.pow(
                2.0,
                torch.ceil(
                    torch.log2(
                        blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
                    )
                ),
            )
            return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape(
                t.shape
            )

        refs = [
            torch.zeros_like(saved_x, dtype=torch.bfloat16),
            torch.zeros_like(scores),
            torch.zeros_like(weight, dtype=torch.float32),
            torch.zeros_like(w2, dtype=torch.float32),
        ]
        for expert, start, count, _ in peer_descriptors.tolist():
            if not count:
                continue
            sl = slice(start, start + count)
            zi = z[sl].float().detach().requires_grad_()
            si = scores[sl].detach().requires_grad_()
            gate, up = zi.chunk(2, -1)
            hi = torch.nn.functional.silu(gate) * up * si[:, None]
            dh = (
                decoded_dy[sl].float() @ ref_quant(w2[expert].T.contiguous()).T
            ).bfloat16()
            dz, ds = torch.autograd.grad(hi, (zi, si), dh.float())
            dz = dz.bfloat16()
            refs[0][sl] = (
                ref_quant(dz) @ ref_quant(weight[expert].T.contiguous()).T
            ).bfloat16()
            refs[1][sl] = ds
            refs[2][expert] = (
                ref_quant(dz.T.contiguous())
                @ ref_quant(decoded[sl].bfloat16().T.contiguous()).T
            )
            refs[3][expert] = (
                ref_quant(decoded_dy[sl].T.contiguous())
                @ ref_quant(hi.detach().bfloat16().T.contiguous()).T
            )
        backward = (
            torch.compile(fp8_packed_backward, fullgraph=True, dynamic=False)
            if compiled
            else fp8_packed_backward
        )

        def consume_backward(
            backward=backward,
            saved_x=saved_x,
            saved_sx=saved_sx,
            w1t=w1t,
            s1t=s1t,
            w2t=w2t,
            packed_s2t=packed_s2t,
            scores=scores,
            z=z,
            peer_ids=peer_ids,
            peer_ends=peer_ends,
        ):
            return backward(
                values.view(torch.float8_e4m3fn),
                scales.T,
                saved_x,
                saved_sx,
                w1t,
                s1t,
                w2t,
                packed_s2t,
                queue,
                scores,
                z,
                peer_ids,
                peer_ends,
            )

        reset()
        send(source)
        torch.cuda.synchronize()
        dist.barrier()
        consume_backward()
        torch.cuda.synchronize()
        dist.barrier()
        backward_errors = []
        for _trial in range(3):
            reset()
            with torch.cuda.stream(consumers[0]):
                grads = consume_backward()
            with torch.cuda.stream(sender):
                torch.cuda._sleep(10_000_000 * (rank + 1))
                send(source)
            torch.cuda.synchronize()
            trial_errors = []
            for got, ref in zip(grads, refs, strict=True):
                error = (
                    (got.float() - ref.float()).norm()
                    / ref.float().norm().clamp_min(1e-8)
                ).item()
                assert error < 0.01, (case, trial_errors, error)
                trial_errors.append(error)
            torch.testing.assert_close(
                grads[0][peer_ids < 0], refs[0][peer_ids < 0], atol=0, rtol=0
            )
            backward_errors.append(trial_errors)
            dist.barrier()
        print(
            json.dumps(
                dict(
                    rank=rank,
                    compiled=compiled,
                    case=case,
                    experts=experts,
                    capacity=capacity,
                    relative_l2=errors,
                    backward_relative_l2=backward_errors,
                )
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--tokens", type=int, default=137)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()
    run(args.compiled, args.tokens, args.hidden)
