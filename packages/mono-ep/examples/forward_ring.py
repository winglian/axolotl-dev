# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Bounded SM100 BF16 forward ring: one runtime with the worst-case ring (``max``) and one with the
routed-row budget ring (``auto``, optionally ``MONO_EP_FORWARD_RING_ROWS`` to force a small ring and
exercise the ring gate), the same inputs through both. y / dx bitwise per routing, forward and
forward+backward times interleaved in one process (median, max over ranks), ring tokens and symmetric
buffer bytes. Adversarial routing sends every token to rank 0's experts, so rank 0 runs gated.

Prints ``FWRING_JSON <payload>`` on rank 0.
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
    runtimes = {}
    for mode in ("max", "auto"):
        os.environ["MONO_EP_WORKSPACE_BOUND"] = mode
        runtimes[mode] = PersistentMoE(
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

    def reduce_max(values):
        t = torch.tensor(values, device="cuda", dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.tolist()

    results = {}
    for case in ("balanced", "skewed", "adversarial"):
        x = torch.randn(
            tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        ids = routing(case)
        scores = torch.rand(tokens, topk, device="cuda")
        scores = (scores / scores.sum(-1, keepdim=True)).requires_grad_()
        dy = torch.randn_like(x)
        outs = {}
        for mode, runtime in runtimes.items():
            for t in (x, scores, w1, w2):
                t.grad = None
            y = runtime(x, ids, scores, w1, w2)
            y.backward(dy)
            outs[mode] = (y.detach().clone(), x.grad.clone(), w1.grad.clone())
        diffs = reduce_max(
            [float(not torch.equal(outs["auto"][i], outs["max"][i])) for i in range(3)]
        )
        fwd = {m: [] for m in runtimes}
        step = {m: [] for m in runtimes}
        for it in range(args.iters + 3):
            for mode, runtime in runtimes.items():
                for t in (x, scores, w1, w2):
                    t.grad = None
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                torch.cuda.synchronize()
                events[0].record()
                y = runtime(x, ids, scores, w1, w2)
                events[1].record()
                y.backward(dy)
                events[2].record()
                events[2].synchronize()
                if it >= 3:
                    fwd[mode].append(events[0].elapsed_time(events[1]))
                    step[mode].append(events[0].elapsed_time(events[2]))
        med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
        times = reduce_max(
            [med(fwd["max"]), med(fwd["auto"]), med(step["max"]), med(step["auto"])]
        )
        bound = runtimes["auto"]._peer_backward.bound
        results[case] = dict(
            y_bitwise=diffs[0] == 0.0,
            dx_bitwise=diffs[1] == 0.0,
            dw1_bitwise=diffs[2] == 0.0,
            forward_ms_max=times[0],
            forward_ms_auto=times[1],
            fwd_bwd_ms_max=times[2],
            fwd_bwd_ms_auto=times[3],
            high_water_rows_max_rank=reduce_max([float(bound.high_water)])[0],
            over_budget_forwards_max_rank=reduce_max(
                [float(bound.forward_over_budget)]
            )[0],
        )
    payload = dict(
        shape=args.shape,
        world=world,
        forward_ring_rows_env=os.getenv("MONO_EP_FORWARD_RING_ROWS"),
        forward_ring_budget=runtimes["auto"].forward_ring_budget,
        ring_tokens={m: r.buffer.l1_acts.shape[0] for m, r in runtimes.items()},
        forward_symmetric_gib={
            m: r.buffer.buffer.nbytes / 2**30 for m, r in runtimes.items()
        },
        backward_symmetric_gib={
            m: r._peer_backward.buffer.buffer.nbytes / 2**30
            for m, r in runtimes.items()
        },
        results=results,
    )
    if rank == 0:
        print("FWRING_JSON " + json.dumps(payload), flush=True)
    dist.barrier()
    for runtime in runtimes.values():
        runtime.close()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
