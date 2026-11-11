# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Bounded workspace parity: the same inputs through a worst-case (``max``) and a bounded (``auto``)
native backward on one runtime, per routing; dx / dW must be bitwise, dscore within its
shared-memory-atomic order noise. Cases: balanced, skewed, adversarial (every token to rank 0's
experts, so rank 0's pools must grow to the worst case), and ``unrecorded`` (the bounded backward
runs without its forward's record and must fall back to the worst case).

Prints ``WSPAR_JSON <payload>`` on rank 0.
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
    w1 = (
        torch.randn(local, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(local, hidden, inner, device="cuda") / inner**0.5
    ).requires_grad_()

    def routing(case):
        if case == "skewed":
            return _skewed_routing(tokens, experts, topk, rank)
        if case == "adversarial":
            return (
                torch.rand(tokens, local, device="cuda")
                .argsort(-1)[:, :topk]
                .contiguous()
            )
        return (
            torch.rand(tokens, experts, device="cuda")
            .argsort(-1)[:, :topk]
            .contiguous()
        )

    results = {}
    for case in ("balanced", "skewed", "unrecorded", "adversarial"):
        x = torch.randn(
            tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        ids = routing(case)
        scores = torch.rand(tokens, topk, device="cuda")
        scores = (scores / scores.sum(-1, keepdim=True)).requires_grad_()
        dy = torch.randn_like(x)
        grads = {}
        for name, backend in (("max", worst), ("auto", bounded), ("max_again", worst)):
            runtime._peer_backward = backend
            for t in (x, scores, w1, w2):
                t.grad = None
            y = runtime(x, ids, scores, w1, w2)
            if case == "unrecorded" and backend.bound is not None:
                backend.bound.records.clear()
            y.backward(dy)
            grads[name] = [t.grad.clone() for t in (x, scores, w1, w2)]
        entry = {}
        for label, other in (
            ("auto_vs_max", "auto"),
            ("max_vs_max_control", "max_again"),
        ):
            sub = {}
            for key, a, b in zip(
                ("dx", "dscore", "dw1", "dw2"), grads[other], grads["max"], strict=True
            ):
                stats = torch.tensor(
                    [
                        float(not torch.equal(a, b)),
                        float(
                            (a.float() - b.float()).norm()
                            / b.float().norm().clamp_min(1e-12)
                        ),
                    ],
                    device="cuda",
                )
                dist.all_reduce(stats, op=dist.ReduceOp.MAX)
                sub[key] = dict(bitwise=stats[0].item() == 0.0, rel=stats[1].item())
            entry[label] = sub
        rows = torch.tensor([float(bounded.pools[0].shape[0])], device="cuda")
        dist.all_reduce(rows, op=dist.ReduceOp.MAX)
        entry["bounded_pool_rows_max_rank"] = int(rows.item())
        entry["worst_rows"] = worst.worst_rows
        results[case] = entry
    payload = dict(
        shape=args.shape,
        results=results,
        grows_rank0=bounded.bound.grows,
        unrecorded_rank0=bounded.bound.unrecorded,
    )
    if rank == 0:
        print("WSPAR_JSON " + json.dumps(payload), flush=True)
    dist.barrier()
    runtime.close()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
