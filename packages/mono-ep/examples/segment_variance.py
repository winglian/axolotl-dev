# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Where the SM100 MXFP8 backward's iteration-to-iteration spread comes from: 40+ fused forward + backward iterations
with MONO_EP_BACKWARD_SEGMENT_TIMING=1, recording CUDA events and host timestamps at the backward's segment boundaries
(token quant, weight forms, persistent kernel, requant, dW1, dW2). Per segment, device time (event to event, launch gaps
included) and host time (Python wall between the same marks), median and spread over iterations, on every rank; plus the
weight-form cache counters with and without weight version bumps.

Prints ``SEGVAR_JSON <payload>`` on rank 0.
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
from mok_vs_mono_ep import SHAPES


def _summary(values):
    ordered = sorted(values)
    n = len(ordered)
    return dict(
        median=statistics.median(ordered),
        p25=ordered[n // 4],
        p75=ordered[(3 * n) // 4],
        min=ordered[0],
        max=ordered[-1],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument(
        "--bump", action="store_true", help="bump weight versions each iteration"
    )
    args = parser.parse_args()
    os.environ["MONO_EP_BACKWARD_SEGMENT_TIMING"] = "1"
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
    from mono_ep import persistent_backward, weight_cache
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
    ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk].contiguous()
    scores = torch.rand(tokens, topk, device="cuda")
    scores = (scores / scores.sum(-1, keepdim=True)).requires_grad_()
    w1 = (
        torch.randn(local, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(local, hidden, inner, device="cuda") / inner**0.5
    ).requires_grad_()
    dy = torch.randn_like(x)
    segments, totals = {}, []
    stats_before = dict(weight_cache.cache_stats())
    for it in range(args.iters + 3):
        for t in (x, scores, w1, w2):
            t.grad = None
        if args.bump:
            for t in (w1, w2):
                torch.autograd.graph.increment_version(t)
        persistent_backward.SEGMENT_MARKS.clear()
        dist.barrier()
        torch.cuda.synchronize()
        runtime(x, ids, scores, w1, w2).backward(dy)
        torch.cuda.synchronize()
        marks = list(persistent_backward.SEGMENT_MARKS)
        if it < 3 or len(marks) < 2:
            continue
        for (_, h0, e0), (name, h1, e1) in zip(marks, marks[1:], strict=False):
            entry = segments.setdefault(name, dict(device=[], host=[]))
            entry["device"].append(e0.elapsed_time(e1))
            entry["host"].append((h1 - h0) * 1e3)
        totals.append(marks[0][2].elapsed_time(marks[-1][2]))
    stats_after = dict(weight_cache.cache_stats())
    local_payload = dict(
        total=_summary(totals),
        segments={
            n: dict(device=_summary(v["device"]), host=_summary(v["host"]))
            for n, v in segments.items()
        },
        cache_delta={
            k: stats_after.get(k, 0) - stats_before.get(k, 0) for k in stats_after
        },
    )
    every = [None] * world
    dist.all_gather_object(every, local_payload)
    if rank == 0:
        print(
            "SEGVAR_JSON "
            + json.dumps(
                dict(
                    shape=args.shape,
                    bump=args.bump,
                    iters=args.iters,
                    env={
                        k: v for k, v in os.environ.items() if k.startswith("MONO_EP_")
                    },
                    ranks=every,
                )
            ),
            flush=True,
        )
    dist.barrier()
    runtime.close()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
