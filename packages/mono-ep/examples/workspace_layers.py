# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Bounded workspace under a realistic schedule: L decoder-like layers, each a non-reentrant checkpoint
region that computes the MoE input (a norm) and the routing (router linear + top-k) inside the region,
so the checkpoint recompute produces new x / index tensors; M micro-batches with gradient accumulation.
Runs the stack with a worst-case (max) and a bounded (auto) backward on one runtime and reports dx and
parameter-gradient parity plus the bound's counters (records found / missing, grows, pool rows).

Prints ``WSLAYERS_JSON <payload>`` on rank 0.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from mok_vs_mono_ep import SHAPES
from torch.utils.checkpoint import checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), default="30b")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--microbatches", type=int, default=2)
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
    from mono_ep.persistent_backward import NativePersistentBackward
    from mono_ep.runtime import PersistentMoE

    deep_gemm.set_num_sms(
        torch.cuda.get_device_properties(local_rank).multi_processor_count
    )
    runtime = PersistentMoE(
        dist.group.WORLD, experts, tokens, topk, hidden, inner, fuse_activation=True
    )
    os.environ["MONO_EP_WORKSPACE_BOUND"] = "max"
    worst = NativePersistentBackward(runtime)
    os.environ["MONO_EP_WORKSPACE_BOUND"] = "auto"
    bounded = NativePersistentBackward(runtime)
    torch.manual_seed(812 + rank)
    params = []
    for _ in range(args.layers):
        norm = torch.ones(hidden, device="cuda", requires_grad=True)
        router = (
            torch.randn(experts, hidden, device="cuda") / hidden**0.5
        ).requires_grad_()
        w1 = (
            torch.randn(local, 2 * inner, hidden, device="cuda") / hidden**0.5
        ).requires_grad_()
        w2 = (
            torch.randn(local, hidden, inner, device="cuda") / inner**0.5
        ).requires_grad_()
        params.append((norm, router, w1, w2))
    batches = [
        (
            torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16),
            torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16),
        )
        for _ in range(args.microbatches)
    ]

    def layer(h, norm, router, w1, w2):
        x = (
            h.float()
            * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6)
            * norm
        ).to(torch.bfloat16)
        values, ids = (x.float() @ router.t()).topk(topk, dim=-1)
        scores = torch.softmax(values, dim=-1)
        return h + runtime(
            x.contiguous(), ids.contiguous(), scores.contiguous(), w1, w2
        )

    def run(backend):
        runtime._peer_backward = backend
        for group in params:
            for p in group:
                p.grad = None
        dxs = []
        for h0, dy in batches:
            h = h0.clone().requires_grad_()
            out = h
            for group in params:
                out = checkpoint(
                    layer, out, *group, use_reentrant=False, preserve_rng_state=False
                )
            out.backward(dy)
            dxs.append(h.grad.clone())
        return dxs, [p.grad.clone() for group in params for p in group]

    dx_max, g_max = run(worst)
    dx_ctl, g_ctl = run(worst)
    dx_auto, g_auto = run(bounded)

    def compare(a, b):
        stats = torch.tensor(
            [
                float(not torch.equal(a, b)),
                float(
                    (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
                ),
            ],
            device="cuda",
        )
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        return dict(bitwise=stats[0].item() == 0.0, rel=stats[1].item())

    bound = bounded.bound
    counters = torch.tensor(
        [
            float(bound.unrecorded),
            float(bound.grows),
            float(bounded.pools[0].shape[0]),
            float(bound.found),
        ],
        device="cuda",
    )
    dist.all_reduce(counters, op=dist.ReduceOp.MAX)
    payload = dict(
        shape=args.shape,
        layers=args.layers,
        microbatches=args.microbatches,
        dx=[compare(a, b) for a, b in zip(dx_auto, dx_max, strict=True)],
        control_dx_max_vs_max=[
            compare(a, b) for a, b in zip(dx_ctl, dx_max, strict=True)
        ],
        control_param_grads_max_vs_max=max(
            (compare(a, b) for a, b in zip(g_ctl, g_max, strict=True)),
            key=lambda e: e["rel"],
        ),
        param_grads_worst=max(
            (compare(a, b) for a, b in zip(g_auto, g_max, strict=True)),
            key=lambda e: e["rel"],
        ),
        backwards=args.layers * args.microbatches,
        records_found_max_rank=int(counters[3].item()),
        unrecorded_max_rank=int(counters[0].item()),
        grows_max_rank=int(counters[1].item()),
        pool_rows_max_rank=int(counters[2].item()),
        worst_rows=worst.worst_rows,
    )
    if rank == 0:
        print("WSLAYERS_JSON " + json.dumps(payload), flush=True)
    dist.barrier()
    runtime.close()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
