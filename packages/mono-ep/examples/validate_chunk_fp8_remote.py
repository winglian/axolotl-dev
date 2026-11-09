# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate one-kernel peer FP8 value/scale publication into queued GEMM."""

import argparse
import json
import os

import deep_gemm
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mono_ep.chunk import chunk_fp8_out, chunk_fp8_swiglu_backward_out
from mono_ep.peer import (
    peer_copy_fp8_publish,
    peer_gather_quantize_publish,
    peer_quantize_publish,
)
from mono_ep.quantization import quantize_fp8


def run(compiled, quantized_producer=False, routed_producer=False):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    peer = 1 - rank
    torch.manual_seed(531 + rank)
    rows, hidden, inner = 512, 512, 256
    raw = torch.randn(
        137 if routed_producer else rows, hidden, dtype=torch.bfloat16, device="cuda"
    )
    indices = torch.randint(0, raw.shape[0], (rows,), device="cuda", dtype=torch.int32)
    indices[::7] = -1
    gathered = (
        raw[indices.clamp_min(0).long()].masked_fill((indices < 0)[:, None], 0)
        if routed_producer
        else raw
    )
    source, scales = quantize_fp8(gathered)
    packed = deep_gemm.transform_sf_into_required_layout(
        scales, rows, hidden, (1, 32)
    ).T.contiguous()
    decoded = source.float() * scales.repeat_interleave(32, -1)
    references = [torch.empty_like(decoded) for _ in range(2)]
    dist.all_gather(references, decoded)
    w, ws = quantize_fp8(
        torch.randn(3, inner, hidden, dtype=torch.bfloat16, device="cuda")
    )
    sw = deep_gemm.transform_sf_into_required_layout(
        ws, inner, hidden, (1, 32), num_groups=3
    )
    expected = (
        references[peer][128:384] @ (w[2].float() * ws[2].repeat_interleave(32, -1)).T
    )
    storage = symm_mem.empty(rows, hidden, dtype=torch.uint8, device="cuda")
    scale_storage = symm_mem.empty(*packed.shape, dtype=torch.int32, device="cuda")
    queue = symm_mem.empty(1, 4, dtype=torch.int32, device="cuda")
    handles = [
        symm_mem.rendezvous(t, group=dist.group.WORLD)
        for t in (storage, scale_storage, queue)
    ]
    peer_storage, peer_scales, peer_queue = [
        h.get_buffer(peer, tuple(t.shape), t.dtype)
        for h, t in zip(handles, (storage, scale_storage, queue), strict=True)
    ]
    x, peer_x = (
        storage.view(torch.float8_e4m3fn),
        peer_storage.view(torch.float8_e4m3fn),
    )
    sx = scale_storage.T
    out = torch.full((rows, inner), 7, device="cuda", dtype=torch.bfloat16)
    producer_op = peer_quantize_publish if quantized_producer else peer_copy_fp8_publish
    producer_args = (
        (raw, peer_x, peer_scales, peer_queue, 0)
        if quantized_producer
        else (source, peer_x, packed, peer_scales, peer_queue, 0)
    )
    if routed_producer:
        producer_op = peer_gather_quantize_publish
        producer_args = (raw, indices, peer_x, peer_scales, peer_queue, 0)
    source_values = [torch.empty_like(source.view(torch.uint8)) for _ in range(2)]
    source_scales = [torch.empty_like(packed) for _ in range(2)]
    dist.all_gather(source_values, source.view(torch.uint8))
    dist.all_gather(source_scales, packed)
    consumer_op = chunk_fp8_out
    if compiled:
        producer_op = torch.compile(producer_op, fullgraph=True, dynamic=False)
        consumer_op = torch.compile(consumer_op, fullgraph=True, dynamic=False)
    queue.copy_(torch.tensor([[2, 128, 256, 1]], device="cuda", dtype=torch.int32))
    x.copy_(source)
    scale_storage.copy_(packed)
    deep_gemm.set_num_sms(32)
    consumer_op(x, sx, w, sw, out, queue, 0)
    producer_op(*producer_args)
    torch.cuda._sleep(1)
    torch.cuda.synchronize()
    dist.barrier()
    consumer, producer = torch.cuda.Stream(), torch.cuda.Stream()
    errors = []
    for _ in range(3):
        storage.zero_()
        scale_storage.zero_()
        queue[0, 3] = 0
        out.fill_(7)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        with torch.cuda.stream(consumer):
            consumer_op(x, sx, w, sw, out, queue, 0)
        with torch.cuda.stream(producer):
            torch.cuda._sleep(10_000_000)
            producer_op(*producer_args)
        torch.cuda.synchronize()
        torch.testing.assert_close(storage, source_values[peer], atol=0, rtol=0)
        torch.testing.assert_close(scale_storage, source_scales[peer], atol=0, rtol=0)
        error = (out[128:384].float() - expected).norm() / expected.norm()
        assert torch.isfinite(error) and error < 0.01, error
        assert (out[:128] == 7).all() and (out[384:] == 7).all()
        errors.append(error.item())
        dist.barrier()
    print(
        json.dumps(
            dict(
                rank=rank,
                compiled=compiled,
                quantized_producer=quantized_producer,
                routed_producer=routed_producer,
                peer_relative_l2=errors,
            )
        ),
        flush=True,
    )
    z = torch.randn(rows, inner * 2, device="cuda", dtype=torch.bfloat16)
    scores = torch.rand(rows, device="cuda")
    dz = torch.full_like(z, 7)
    partial = torch.full((rows, inner // 8), 7, device="cuda", dtype=torch.float32)
    zr = z[128:384].float().requires_grad_()
    sr = scores[128:384].detach().clone().requires_grad_()
    gate, up = zr.chunk(2, dim=-1)
    expected_h = torch.nn.functional.silu(gate) * up * sr[:, None]
    expected_dz, expected_ds = torch.autograd.grad(
        expected_h, (zr, sr), expected.bfloat16().float()
    )
    backward_op = chunk_fp8_swiglu_backward_out
    if compiled:
        backward_op = torch.compile(backward_op, fullgraph=True, dynamic=False)
    backward_op(x, sx, w, sw, out, queue, 0, z, scores, dz, partial)
    torch.cuda.synchronize()
    dist.barrier()
    backward_errors = []
    for _ in range(3):
        storage.zero_()
        scale_storage.zero_()
        queue[0, 3] = 0
        for output in (out, dz, partial):
            output.fill_(7)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        with torch.cuda.stream(consumer):
            backward_op(x, sx, w, sw, out, queue, 0, z, scores, dz, partial)
        with torch.cuda.stream(producer):
            torch.cuda._sleep(10_000_000)
            producer_op(*producer_args)
        torch.cuda.synchronize()
        trial_errors = []
        for actual, want in (
            (out[128:384], expected_h),
            (dz[128:384], expected_dz),
            (partial[128:384].sum(-1), expected_ds),
        ):
            error = (actual.float() - want.float()).norm() / want.float().norm()
            assert torch.isfinite(error) and error < 0.01, error
            trial_errors.append(error.item())
        for output in (out, dz, partial):
            assert (output[:128] == 7).all() and (output[384:] == 7).all()
        backward_errors.append(trial_errors)
        dist.barrier()
    print(
        json.dumps(
            dict(
                rank=rank, compiled=compiled, peer_backward_relative_l2=backward_errors
            )
        ),
        flush=True,
    )
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--quantized-producer", action="store_true")
    parser.add_argument("--routed-producer", action="store_true")
    args = parser.parse_args()
    run(args.compiled, args.quantized_producer, args.routed_producer)
