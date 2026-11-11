# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Bounded workspace of the NVFP4 LoRA kernels: one runtime with the worst-case pools and
rings (``MONO_EP_WORKSPACE_BOUND=max``) and one bounded (``auto``; ``MONO_EP_FORWARD_RING_ROWS``
forces a small forward ring so the ring gate binds), the same inputs through both, per routing
(balanced, skewed, adversarial: every token to rank 0's experts). ``y`` / ``dx`` / ``dscore`` /
``dA1`` are compared bitwise and by relative norm, next to a max-vs-max control of the same
pair (the NVFP4 backward's grouped GEMMs are not bitwise run to run). Forward and
forward+backward times interleaved in one process (median, max over ranks); ring tokens,
symmetric-buffer and pool bytes; bound counters.

Prints ``NVWS_JSON <payload>`` on rank 0.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from nvfp4_lora_shapes import random_layer, reduce_max, rel, routing, shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="30b")
    parser.add_argument("--tokens", type=int, default=0)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cases", default="balanced,skewed,adversarial")
    parser.add_argument("--reserved-sms", type=int, default=16)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    geo = shape(args.shape, args.tokens)
    tokens, hidden, inner, experts, topk = (
        geo[k] for k in ("tokens", "hidden", "intermediate", "experts", "topk")
    )
    local = experts // world

    import deep_gemm
    from mono_ep.api import check_nccl_env
    from mono_ep.nvfp4_lora import RequantCache, backward_for, nvfp4_lora_moe
    from mono_ep.runtime import PersistentMoE

    check_nccl_env(reserved_sms=args.reserved_sms)
    physical = torch.cuda.get_device_properties(device).multi_processor_count
    deep_gemm.set_num_sms(physical - args.reserved_sms)
    free0 = torch.cuda.mem_get_info(device)[0]
    runtimes, backwards, symm = {}, {}, {}
    for mode in ("max", "auto"):
        os.environ["MONO_EP_WORKSPACE_BOUND"] = mode
        before = torch.cuda.mem_get_info(device)[0]
        alloc = torch.cuda.memory_allocated(device)
        runtimes[mode] = PersistentMoE(
            dist.group.WORLD,
            experts,
            tokens,
            topk,
            hidden,
            inner,
            compute_dtype="nvfp4_lora",
            reserved_sms=args.reserved_sms,
        )
        backwards[mode] = backward_for(runtimes[mode])
        torch.cuda.synchronize()
        symm[mode] = dict(
            device_delta_gib=(before - torch.cuda.mem_get_info(device)[0]) / 2**30,
            allocator_delta_gib=(torch.cuda.memory_allocated(device) - alloc) / 2**30,
            forward_ring_tokens=runtimes[mode].buffer.l1_acts.shape[0],
            forward_symm_gib=runtimes[mode].buffer.buffer.nbytes / 2**30,
            backward_ring_tokens=backwards[mode].buffer.l1_acts.shape[0],
            backward_symm_gib=backwards[mode].buffer.buffer.nbytes / 2**30,
            pool_rows=backwards[mode].rows,
            pool_gib=backwards[mode].pool_bytes() / 2**30,
        )
    layer, adapters = random_layer(local, hidden, inner, device, 812 + rank)
    cache = RequantCache()

    def step(mode, x, ids, scores, dy):
        a = [t.clone().requires_grad_() for t in adapters]
        x = x.clone().requires_grad_()
        s = scores.clone().requires_grad_()
        y = nvfp4_lora_moe(x, ids, s, *a, runtimes[mode], layer, cache)
        y.backward(dy)
        return [y.detach(), x.grad, s.grad, a[0].grad]

    names = ("y", "dx", "dscore", "dA1")
    results = {}
    for case in args.cases.split(","):
        torch.manual_seed(4321 + rank)
        x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
        ids = routing(case, tokens, experts, topk, rank)
        scores = torch.rand(tokens, topk, device=device)
        scores = scores / scores.sum(-1, keepdim=True)
        dy = torch.randn_like(x) * 0.1
        ref = step("max", x, ids, scores, dy)
        control = step("max", x, ids, scores, dy)
        auto = step("auto", x, ids, scores, dy)
        torch.cuda.synchronize()
        flags = reduce_max(
            [float(not torch.equal(a, r)) for a, r in zip(auto, ref, strict=True)]
            + [float(not torch.equal(c, r)) for c, r in zip(control, ref, strict=True)]
        )
        rels = reduce_max(
            [rel(a, r) for a, r in zip(auto, ref, strict=True)]
            + [rel(c, r) for c, r in zip(control, ref, strict=True)]
        )
        fwd = {m: [] for m in runtimes}
        both = {m: [] for m in runtimes}
        for it in range(args.iters + 2):
            for mode in runtimes:
                a = [t.clone().requires_grad_() for t in adapters]
                xg = x.clone().requires_grad_()
                sg = scores.clone().requires_grad_()
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                torch.cuda.synchronize()
                events[0].record()
                y = nvfp4_lora_moe(xg, ids, sg, *a, runtimes[mode], layer, cache)
                events[1].record()
                y.backward(dy)
                events[2].record()
                events[2].synchronize()
                if it >= 2:
                    fwd[mode].append(events[0].elapsed_time(events[1]))
                    both[mode].append(events[0].elapsed_time(events[2]))
        med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
        times = reduce_max(
            [med(fwd["max"]), med(fwd["auto"]), med(both["max"]), med(both["auto"])]
        )
        bound = backwards["auto"].bound
        results[case] = dict(
            auto_vs_max_bitwise={n: flags[i] == 0.0 for i, n in enumerate(names)},
            max_vs_max_bitwise={n: flags[4 + i] == 0.0 for i, n in enumerate(names)},
            auto_vs_max_rel={n: rels[i] for i, n in enumerate(names)},
            max_vs_max_rel={n: rels[4 + i] for i, n in enumerate(names)},
            forward_ms_max=times[0],
            forward_ms_auto=times[1],
            fwd_bwd_ms_max=times[2],
            fwd_bwd_ms_auto=times[3],
            pool_rows_auto_max_rank=reduce_max([float(backwards["auto"].rows)])[0],
            high_water_rows_max_rank=reduce_max([float(bound.high_water)])[0],
            grows_max_rank=reduce_max([float(bound.grows)])[0],
            found_max_rank=reduce_max([float(bound.found)])[0],
            unrecorded_max_rank=reduce_max([float(bound.unrecorded)])[0],
            over_budget_forwards_max_rank=reduce_max(
                [float(bound.forward_over_budget)]
            )[0],
        )
    payload = dict(
        shape=args.shape,
        geometry=geo,
        world=world,
        requant_backend=layer.backend,
        merged_cache=cache.mode,
        forward_ring_rows_env=os.getenv("MONO_EP_FORWARD_RING_ROWS"),
        forward_ring_tokens_env=os.getenv("MONO_EP_FORWARD_RING_TOKENS"),
        forward_ring_budget=runtimes["auto"].forward_ring_budget,
        workspace=symm,
        device_used_gib_after_construction=(free0 - torch.cuda.mem_get_info(device)[0])
        / 2**30,
        results=results,
    )
    if rank == 0:
        print("NVWS_JSON " + json.dumps(payload), flush=True)
        out = os.environ.get("NVFP4_OUT")
        if out:
            with open(out, "w") as f:
                json.dump(payload, f, indent=1)
    dist.barrier()
    for runtime in runtimes.values():
        runtime.close()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
