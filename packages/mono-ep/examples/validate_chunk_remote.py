# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate peer-memory production into a queued BF16 GEMM on two GPUs."""

import argparse
import json
import os
from pathlib import Path

import deep_gemm
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from validate_chunk_gemm import publish


def run(fused=False, compiled=False):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    peer = 1 - rank
    torch.manual_seed(431 + rank)
    rows, hidden, inner = 512, 256, 256
    x = symm_mem.empty(rows, hidden, dtype=torch.bfloat16, device="cuda")
    queue = symm_mem.empty(1, 4, dtype=torch.int32, device="cuda")
    x_handle = symm_mem.rendezvous(x, group=dist.group.WORLD)
    q_handle = symm_mem.rendezvous(queue, group=dist.group.WORLD)
    peer_x = x_handle.get_buffer(peer, (rows, hidden), torch.bfloat16)
    peer_q = q_handle.get_buffer(peer, (1, 4), torch.int32)
    copy_operation = None
    if fused:
        from mono_ep.peer import peer_copy_publish

        copy_operation = (
            torch.compile(peer_copy_publish, fullgraph=True, dynamic=False)
            if compiled
            else peer_copy_publish
        )
    source = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    sources = [torch.empty_like(source) for _ in range(2)]
    dist.all_gather(sources, source)
    w = torch.randn(3, inner, hidden, dtype=torch.bfloat16, device="cuda") / 16
    out = torch.full((rows, inner), 7, dtype=torch.bfloat16, device="cuda")
    expected = sources[peer][128:384] @ w[2].T
    queue.copy_(torch.tensor([[2, 128, 256, 1]], device="cuda", dtype=torch.int32))
    x.copy_(source)
    deep_gemm.set_num_sms(32)
    deep_gemm.bf16_chunk_gemm_nt(x, w, out, queue, 0)
    publish[(1,)](peer_q, 0)
    peer_x.copy_(source)
    if copy_operation is not None:
        copy_operation(source[128:384], peer_x[128:384], peer_q, 0)
        from mono_ep.peer import _copy_publish

        kernel = _copy_publish.warmup(
            source[128:384],
            peer_x[128:384],
            peer_q,
            0,
            source[128:384].numel(),
            num_warps=4,
            grid=(1,),
        )
        ptx = kernel.asm["ptx"]
        sync_lines = [
            line.strip()
            for line in ptx.splitlines()
            if any(
                token in line
                for token in ("fence.", "bar.sync", "release.sys", "atom.", "st.")
            )
        ]
        assert any("fence.acq_rel.sys" in line for line in sync_lines), sync_lines
        assert any("atom.global.sys.release.exch" in line for line in sync_lines), (
            sync_lines
        )
        print(json.dumps({"rank": rank, "producer_sync_ptx": sync_lines}), flush=True)
        if compiled:
            wrappers = []
            for filename in Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).rglob("*.py"):
                code = filename.read_text()
                if "mono_ep.peer_copy_publish" in code:
                    wrappers.append(code)
            assert wrappers, "No generated producer wrapper found"
            print(json.dumps({"rank": rank, "producer_wrappers": wrappers}), flush=True)
    torch.cuda._sleep(1)
    torch.cuda.synchronize()
    dist.barrier()
    consumer, producer = torch.cuda.Stream(), torch.cuda.Stream()
    errors = []
    for _trial in range(3):
        x.zero_()
        queue[0, 3] = 0
        out.fill_(7)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        with torch.cuda.stream(consumer):
            deep_gemm.bf16_chunk_gemm_nt(x, w, out, queue, 0)
        with torch.cuda.stream(producer):
            torch.cuda._sleep(10_000_000)
            if copy_operation is None:
                peer_x.copy_(source)
                publish[(1,)](peer_q, 0)
            else:
                copy_operation(source[128:384], peer_x[128:384], peer_q, 0)
        torch.cuda.synchronize()
        error = (
            out[128:384].float() - expected.float()
        ).norm() / expected.float().norm()
        assert torch.isfinite(error) and error < 0.03, error
        assert (out[:128] == 7).all() and (out[384:] == 7).all()
        errors.append(error.item())
        dist.barrier()
    print(
        json.dumps(
            {
                "rank": rank,
                "fused": fused,
                "compiled": compiled,
                "peer_relative_l2": errors,
            }
        ),
        flush=True,
    )
    torch.cuda.synchronize()
    dist.barrier()
    del peer_x, peer_q, x_handle, q_handle, x, queue
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()
    run(args.fused, args.compiled)
