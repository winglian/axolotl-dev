# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Per-rank forward / backward time of the fused mono-ep layer: each rank times its own calls (CUDA events, a barrier
before every iteration so skew does not carry over), and rank 0 reports every rank's median plus max - mean, the slowest
rank and the per-rank receive counts. Separates load skew (the slowest rank receives the most rows) from a slow device.

Prints ``SPREAD_JSON <payload>`` on rank 0.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from mok_vs_mono_ep import SHAPES, _skewed_routing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--routing", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    shape = SHAPES[args.shape]
    tokens, hidden, inner, experts = (
        shape[k] for k in ("tokens", "hidden", "intermediate", "experts")
    )
    topk, local = 8, experts // world

    import deep_gemm
    from mono_ep.runtime import PersistentMoE

    deep_gemm.set_num_sms(
        torch.cuda.get_device_properties(local_rank).multi_processor_count
    )
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inner,
        fuse_activation=True,
        backward_backend="native",
    )
    torch.manual_seed(812 + rank)
    x = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    if args.routing == "skewed":
        ids = _skewed_routing(tokens, experts, topk, rank)
    else:
        ids = (
            torch.rand(tokens, experts, device="cuda")
            .argsort(-1)[:, :topk]
            .contiguous()
        )
    scores = torch.rand(tokens, topk, device="cuda")
    scores = (scores / scores.sum(-1, keepdim=True)).requires_grad_()
    w1 = (
        torch.randn(local, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(local, hidden, inner, device="cuda") / inner**0.5
    ).requires_grad_()
    dy = torch.randn_like(x)
    counts = torch.bincount(ids.flatten(), minlength=experts).view(world, local).sum(-1)
    dist.all_reduce(counts)
    fwd, bwd = [], []
    for it in range(args.iters + 3):
        for t in (x, scores, w1, w2):
            t.grad = None
        dist.barrier()
        torch.cuda.synchronize()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        events[0].record()
        y = runtime(x, ids, scores, w1, w2)
        events[1].record()
        y.backward(dy)
        events[2].record()
        events[2].synchronize()
        if it >= 3:
            fwd.append(events[0].elapsed_time(events[1]))
            bwd.append(events[1].elapsed_time(events[2]))
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    mine = torch.tensor([med(fwd), med(bwd)], device="cuda", dtype=torch.float64)
    every = [torch.empty_like(mine) for _ in range(world)]
    dist.all_gather(every, mine)
    every = torch.stack(every).tolist()
    if rank == 0:
        f = [e[0] for e in every]
        b = [e[1] for e in every]
        payload = dict(
            shape=args.shape,
            routing=args.routing,
            env={k: v for k, v in os.environ.items() if k.startswith("MONO_EP_")},
            forward_ms=f,
            backward_ms=b,
            forward_max_minus_mean=max(f) - sum(f) / world,
            backward_max_minus_mean=max(b) - sum(b) / world,
            slowest_backward_rank=b.index(max(b)),
            received_rows_per_rank=counts.tolist(),
        )
        print("SPREAD_JSON " + json.dumps(payload), flush=True)
    dist.barrier()
    runtime.close()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
