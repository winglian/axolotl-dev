# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Persistent memory by source of the NVFP4 LoRA path at a training geometry: a stack of
``--layers`` MoE blocks (router + frozen packed NVFP4 experts + fp32 LoRA adapters under FSDP2
``fully_shard`` on the ``dp_shard`` mesh, EP across the ``ep`` mesh), non-reentrant activation
checkpointing per block, AdamW steps. Reports per rank: the packed base, the stored transposed
base (if any), the requant cache's resident bytes and build counts, the backward pools, the
symmetric buffers, allocator peak / reserved / resident between steps, step time and the
bounded-workspace counters, plus the per-layer figures scaled to ``--model-layers``.
``--compare-modes`` runs the same stack under ``MONO_EP_NVFP4_MERGED_CACHE=all`` and ``layer``
(and ``all`` twice as the run-to-run control) and compares losses and adapter gradients.

Run under torchrun; prints ``NVMEM_JSON <payload>`` on rank 0.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from nvfp4_lora_shapes import RANK, SCALING, random_layer, reduce_max, rel, shape
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
    def __init__(self, layer, adapters, hidden, experts, topk, runtime, cache):
        super().__init__()
        self.a1, self.b1, self.a2, self.b2 = (nn.Parameter(t.clone()) for t in adapters)
        self.router = nn.Parameter(torch.randn(experts, hidden) / hidden**0.5)
        self.layer, self.runtime, self.cache, self.topk = layer, runtime, cache, topk

    def forward(self, x):
        from mono_ep.nvfp4_lora import nvfp4_lora_moe

        h = F.rms_norm(x.float(), (x.shape[-1],)).to(torch.bfloat16)
        logits = h.float() @ self.router.float().t()
        scores, ids = logits.softmax(-1).topk(self.topk, dim=-1)
        scores = scores / scores.sum(-1, keepdim=True)
        y = nvfp4_lora_moe(
            h,
            ids.int(),
            scores,
            self.a1,
            self.b1,
            self.a2,
            self.b2,
            self.runtime,
            self.layer,
            self.cache,
        )
        return x + y


def run(args, mesh, geo, mode, record):
    from mono_ep.nvfp4_lora import RequantCache, backward_for
    from mono_ep.runtime import PersistentMoE

    os.environ["MONO_EP_NVFP4_MERGED_CACHE"] = mode
    device = torch.device("cuda", torch.cuda.current_device())
    ep_group = mesh["ep"].get_group()
    ep_rank = mesh["ep"].get_local_rank()
    tokens, hidden, inner, experts, topk = (
        geo[k] for k in ("tokens", "hidden", "intermediate", "experts", "topk")
    )
    local = experts // mesh["ep"].size()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free0, alloc0 = (
        torch.cuda.mem_get_info(device)[0],
        torch.cuda.memory_allocated(device),
    )
    runtime = PersistentMoE(
        ep_group,
        experts,
        tokens,
        topk,
        hidden,
        inner,
        compute_dtype="nvfp4_lora",
        reserved_sms=args.reserved_sms,
    )
    torch.cuda.synchronize()
    free1, alloc1 = (
        torch.cuda.mem_get_info(device)[0],
        torch.cuda.memory_allocated(device),
    )
    backward = backward_for(runtime)
    torch.cuda.synchronize()
    free2, alloc2 = (
        torch.cuda.mem_get_info(device)[0],
        torch.cuda.memory_allocated(device),
    )
    mem = dict(
        forward_symm_gib=(free0 - free1 - (alloc1 - alloc0)) / 2**30,
        backward_symm_gib=(free1 - free2 - (alloc2 - alloc1)) / 2**30,
        backward_pools_gib=backward.pool_bytes() / 2**30,
        backward_pool_rows=backward.rows,
        forward_ring_tokens=runtime.buffer.l1_acts.shape[0],
        backward_ring_tokens=backward.buffer.l1_acts.shape[0],
    )
    cache = RequantCache()
    blocks = []
    base_bytes = transposed_bytes = 0
    for i in range(args.layers):
        layer, adapters = random_layer(
            local, hidden, inner, device, 100 * i + ep_rank, backend=args.backend
        )
        base_bytes += sum(
            t.numel() * t.element_size()
            for t in (layer.q1, layer.s1, layer.q2, layer.s2)
        )
        transposed_bytes += layer.transposed_base_bytes()
        torch.manual_seed(7 + i)
        blocks.append(
            Block(layer, adapters, hidden, experts, topk, runtime, cache).to(device)
        )
        del adapters
    model = nn.Sequential(*blocks)
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    for block in blocks:
        fully_shard(
            block, mesh=mesh["dp_shard"], mp_policy=mp, reshard_after_forward=True
        )
    fully_shard(model, mesh=mesh["dp_shard"], mp_policy=mp, reshard_after_forward=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0, fused=True
    )
    torch.cuda.synchronize()
    mem["adapters_and_router_gib"] = (
        sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    )
    mem["packed_base_gib"] = base_bytes / 2**30
    mem["transposed_base_gib"] = transposed_bytes / 2**30
    mem["per_layer_packed_base_mib"] = base_bytes / args.layers / 2**20
    mem["per_layer_transposed_base_mib"] = transposed_bytes / args.layers / 2**20

    gen = torch.Generator(device=device).manual_seed(
        4321 + dist.get_rank(mesh["dp_shard"].get_group())
    )
    steps = []
    losses, grads = [], None
    for step in range(args.steps):
        torch.cuda.synchronize()
        resident_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        total = 0.0
        for _micro in range(args.micro):
            x = torch.randn(
                tokens, hidden, device=device, dtype=torch.bfloat16, generator=gen
            )
            for block in blocks:
                x = (
                    checkpoint(block, x, use_reentrant=False)
                    if args.checkpoint
                    else block(x)
                )
            loss = (x.float() ** 2).mean() / args.micro
            loss.backward()
            total += loss.item()
        if step == args.steps - 1:
            grads = [
                p.grad.to_local().detach().float().clone() for p in model.parameters()
            ]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        end.record()
        end.synchronize()
        losses.append(total)
        steps.append(
            dict(
                step=step,
                loss=total,
                step_ms=start.elapsed_time(end),
                peak_active_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                reserved_gib=torch.cuda.memory_reserved(device) / 2**30,
                resident_before_gib=resident_before / 2**30,
                resident_after_gib=torch.cuda.memory_allocated(device) / 2**30,
                cache=cache.stats(),
                bound=None
                if backward.bound is None
                else dict(
                    rows=backward.rows,
                    pool_gib=backward.pool_bytes() / 2**30,
                    found=backward.bound.found,
                    unrecorded=backward.bound.unrecorded,
                    grows=backward.bound.grows,
                    high_water=backward.bound.high_water,
                    over_budget=backward.bound.forward_over_budget,
                ),
            )
        )
    torch.cuda.synchronize()
    device_used = (free0 - torch.cuda.mem_get_info(device)[0]) / 2**30
    result = dict(
        mode=mode,
        requant_backend=blocks[0].layer.backend,
        memory=mem,
        steps=steps,
        device_used_gib=device_used,
    )
    record(result)
    out = (losses, grads)
    del optimizer, model, blocks, cache
    runtime.close()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="30b")
    parser.add_argument("--tokens", type=int, default=0)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--model-layers", type=int, default=48)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--micro", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--reserved-sms", type=int, default=16)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    parser.add_argument(
        "--modes", default=os.environ.get("MONO_EP_NVFP4_MERGED_CACHE", "layer")
    )
    parser.add_argument("--compare-modes", action="store_true")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh(
        "cuda", (world // args.ep, args.ep), mesh_dim_names=("dp_shard", "ep")
    )
    geo = shape(args.shape, args.tokens)

    import deep_gemm
    from mono_ep.api import check_nccl_env

    check_nccl_env(reserved_sms=args.reserved_sms)
    physical = torch.cuda.get_device_properties(device).multi_processor_count
    deep_gemm.set_num_sms(physical - args.reserved_sms)

    runs = []
    modes = ["all", "layer", "all"] if args.compare_modes else args.modes.split(",")
    outs = []
    for mode in modes:
        t0 = time.time()
        outs.append(run(args, mesh, geo, mode, runs.append))
        runs[-1]["wall_s"] = time.time() - t0
    comparison = None
    if args.compare_modes:
        (l_all, g_all), (l_layer, g_layer), (l_ctl, g_ctl) = outs
        comparison = dict(
            losses=dict(all=l_all, layer=l_layer, control=l_ctl),
            loss_rel_layer_vs_all=max(
                abs(a - b) / max(abs(a), 1e-12)
                for a, b in zip(l_all, l_layer, strict=True)
            ),
            loss_rel_control_vs_all=max(
                abs(a - b) / max(abs(a), 1e-12)
                for a, b in zip(l_all, l_ctl, strict=True)
            ),
            grads_bitwise_layer_vs_all=reduce_max(
                [
                    float(
                        not all(
                            torch.equal(a, b)
                            for a, b in zip(g_all, g_layer, strict=True)
                        )
                    )
                ]
            )[0]
            == 0.0,
            grads_bitwise_control_vs_all=reduce_max(
                [
                    float(
                        not all(
                            torch.equal(a, b) for a, b in zip(g_all, g_ctl, strict=True)
                        )
                    )
                ]
            )[0]
            == 0.0,
            grads_rel_layer_vs_all=reduce_max(
                [max(rel(a, b) for a, b in zip(g_layer, g_all, strict=True))]
            )[0],
            grads_rel_control_vs_all=reduce_max(
                [max(rel(a, b) for a, b in zip(g_ctl, g_all, strict=True))]
            )[0],
        )
    # Max over ranks of the memory figures (ranks differ in routing, not in layout)
    for r in runs:
        r["memory"] = {k: reduce_max([float(v)])[0] for k, v in r["memory"].items()}
        for s in r["steps"]:
            for k in (
                "peak_active_gib",
                "reserved_gib",
                "resident_before_gib",
                "resident_after_gib",
                "step_ms",
            ):
                s[k] = reduce_max([s[k]])[0]
            s["cache"]["resident_bytes"] = reduce_max(
                [float(s["cache"]["resident_bytes"])]
            )[0]
            s["cache"]["max_resident_bytes"] = reduce_max(
                [float(s["cache"]["max_resident_bytes"])]
            )[0]
        r["device_used_gib"] = reduce_max([r["device_used_gib"]])[0]
        per_layer_cache_mib = (
            max(s["cache"]["max_resident_bytes"] for s in r["steps"]) / 2**20
        )
        if r["mode"] == "all":
            per_layer_cache_mib /= args.layers
        r["scaled_to_model"] = dict(
            model_layers=args.model_layers,
            packed_base_gib=r["memory"]["per_layer_packed_base_mib"]
            * args.model_layers
            / 2**10,
            transposed_base_gib=r["memory"]["per_layer_transposed_base_mib"]
            * args.model_layers
            / 2**10,
            merged_cache_gib=(
                per_layer_cache_mib * args.model_layers
                if r["mode"] == "all"
                else per_layer_cache_mib
            )
            / 2**10,
            backward_pools_gib=r["memory"]["backward_pools_gib"],
            symm_gib=r["memory"]["forward_symm_gib"] + r["memory"]["backward_symm_gib"],
        )
    payload = dict(
        shape=args.shape,
        geometry=geo,
        world=world,
        ep=args.ep,
        layers=args.layers,
        steps=args.steps,
        micro=args.micro,
        checkpoint=args.checkpoint,
        workspace_bound=os.environ.get("MONO_EP_WORKSPACE_BOUND", "auto"),
        lora_rank=RANK,
        scaling=SCALING,
        runs=runs,
        comparison=comparison,
    )
    if rank == 0:
        print("NVMEM_JSON " + json.dumps(payload), flush=True)
        out = os.environ.get("NVFP4_OUT")
        if out:
            with open(out, "w") as f:
                json.dump(payload, f, indent=1)
    dist.barrier()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
