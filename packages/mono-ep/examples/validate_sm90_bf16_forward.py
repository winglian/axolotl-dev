# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Check the SM90 BF16 persistent forward at one layer shape on two Hopper GPUs.

Runs the runtime forward on balanced routing and on skewed routing (every token to rank 0's
experts, so rank 0's ring wraps), compares each output against an FP32 reference and saves the
outputs so two builds of the kernel can be compared bitwise across processes.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def reference(x, ids, scores, w1, w2, hidden):
    out = torch.zeros(x.shape[0], hidden, device=x.device, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        z = (x[rows].float() @ w1[expert].float().T).bfloat16().float()
        gate, up = z.chunk(2, -1)
        h = (torch.nn.functional.silu(gate) * up * scores[rows, slots, None]).bfloat16()
        out.index_add_(0, rows, (h.float() @ w2[expert].float().T).bfloat16().float())
    return out


def main(args):
    from mono_ep.runtime import PersistentMoE

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    name = torch.cuda.get_device_name()
    assert torch.cuda.get_device_capability()[0] == 9, name
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    assert world == 2, world
    hidden, inner, experts, topk = (
        args.hidden,
        args.intermediate,
        args.experts,
        args.topk,
    )
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        args.tokens,
        topk,
        hidden,
        inner,
        compute_dtype="bf16",
        backward_backend="nccl",
    )
    torch.manual_seed(77)
    w1_all = (
        torch.randn(experts, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    w2_all = (
        torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
    ).bfloat16()
    local = experts // world
    w1 = w1_all[rank * local : (rank + 1) * local].contiguous()
    w2 = w2_all[rank * local : (rank + 1) * local].contiguous()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for routing in ("balanced", "skewed"):
        torch.manual_seed(100 + rank)
        count = args.tokens - 37 * rank
        x = torch.randn(count, hidden, device="cuda", dtype=torch.bfloat16)
        ids = torch.rand(count, experts, device="cuda").argsort(-1)[:, :topk]
        if routing == "skewed":
            ids = torch.rand(count, local, device="cuda").argsort(-1)[:, :topk]
        ids = ids.contiguous()
        ids[::19, -1] = -1
        scores = torch.rand(count, topk, device="cuda").softmax(-1)
        dist.barrier()
        y = runtime(x, ids, scores, w1, w2)
        torch.cuda.synchronize()
        y = y.clone()
        expected = reference(
            x, ids.clamp_min(0), scores * (ids >= 0), w1_all, w2_all, hidden
        )
        histogram = torch.bincount(ids[ids >= 0], minlength=experts)
        dist.all_reduce(histogram)
        mine = histogram[rank * local : (rank + 1) * local]
        used_pool_rows = int((((mine + 127) // 128) * 128).sum().item())
        ring_tokens = int(runtime.buffer.l1_acts.shape[0])
        rel = ((y.float() - expected).norm() / expected.norm().clamp_min(1e-12)).item()
        max_abs = (y.float() - expected).abs().max().item()
        torch.save(y.cpu(), out_dir / f"{args.shape_name}_{routing}_rank{rank}.pt")
        record = dict(
            shape=args.shape_name,
            rank=rank,
            device=name,
            routing=routing,
            tokens=count,
            hidden=hidden,
            intermediate=inner,
            experts=experts,
            topk=topk,
            rel_error_vs_fp32_reference=rel,
            max_abs_error_vs_fp32_reference=max_abs,
            output_abs_max=y.float().abs().max().item(),
            finite=bool(torch.isfinite(y.float()).all()),
            ring_tokens=ring_tokens,
            used_pool_rows=used_pool_rows,
            ring_wrapped=used_pool_rows > ring_tokens,
        )
        print("SM90_BF16_FORWARD " + json.dumps(record), flush=True)
        assert record["finite"] and rel < 0.02, record
        if routing == "skewed" and rank == 0:
            assert record["ring_wrapped"], record
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-name", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--experts", type=int, default=64, help="total experts")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=4096, help="max tokens per rank")
    parser.add_argument("--out-dir", required=True)
    main(parser.parse_args())
