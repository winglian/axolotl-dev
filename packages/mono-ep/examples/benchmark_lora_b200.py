# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""8xB200 (or 8xH100) EP8 timing of one MoE layer through Axolotl's experts seam: fused LoRA
(``bf16_lora`` / ``mxfp8_lora``; ``fp8_lora`` on Hopper) against the dispatch mode around SonicMoE with the same
LoRA adapter, and against the fused full fine-tune of the same precision.

One variant per process (``--variant``). Per iteration the two contracts run in an order
reversed every other iteration: (c) checkpointed 2F+B and (d) F+B under selective
activation checkpointing that saves the mono-ep ops (the fused forward ops, or the
dispatch / combine ops of the dispatch mode, as the Axolotl seam registers them). Every
iteration is one optimizer step: the fused paths rebuild their weight forms (full FT:
``invalidate_weight_caches``; LoRA: the merged-weight cache too). 5 warmups, then
``--iters`` measured; host-wall per iteration, max over ranks, median and IQR. Peak GiB per
contract is the max over ranks of the allocator peak within the contract. For the fused LoRA
variant the per-step merge and the weight-form build are timed separately.
Run under ``torch.distributed.run --nproc-per-node=8``."""

import argparse
import functools
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SHAPES = {
    # tokens per GPU, hidden, intermediate, global experts, top-k
    "30b": (16384, 2048, 768, 128, 8),
    "235b": (16384, 4096, 1536, 128, 8),
    "beam": (16384, 5120, 2560, 256, 8),
}
RANK, ALPHA = 16, 32
WARMUP = 5


def _routing(kind, tokens, experts, topk, rank):
    from dispatch_sonicmoe_contract import _skewed_routing

    if kind == "skewed":
        return _skewed_routing(tokens, experts, topk, rank)
    gen = torch.Generator(device="cuda").manual_seed(812 + rank)
    return (
        torch.rand(tokens, experts, device="cuda", generator=gen)
        .argsort(-1)[:, :topk]
        .contiguous()
    )


def _build_experts(group, hidden, inner, experts, topk, frozen):
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    from axolotl.integrations.expert_parallel.shard import shard_expert_weights

    config = Qwen3MoeConfig(
        hidden_size=hidden,
        moe_intermediate_size=inner,
        num_experts=experts,
        num_experts_per_tok=topk,
        norm_topk_prob=True,
        hidden_act="silu",
    )
    config._experts_implementation = "expert_parallel"
    container = nn.Module()
    container.experts = Qwen3MoeExperts(config)
    torch.manual_seed(812)
    with torch.no_grad():
        container.experts.gate_up_proj.copy_(
            torch.randn(experts, 2 * inner, hidden) / hidden**0.5
        )
        container.experts.down_proj.copy_(
            torch.randn(experts, hidden, inner) / inner**0.5
        )
    container.cuda().to(torch.bfloat16)
    assert shard_expert_weights(container, group) == 1
    module = container.experts
    for parameter in module.parameters():
        parameter.requires_grad_(not frozen)
    return module


def _attach_lora(module, hidden, inner, rank):
    """PEFT ``target_parameters`` factors of this rank's experts, handed over as the SonicMoE
    ParamWrapper fast path does (``lora_A [E*r, in]``, ``lora_B [out, r*E]``)."""
    e = module.num_local_experts
    gen = torch.Generator(device="cuda").manual_seed(4242 + rank)

    def factor(*shape, scale):
        return (
            (torch.randn(*shape, device="cuda", generator=gen) * scale)
            .bfloat16()
            .requires_grad_()
        )

    factors = {
        "gate_up_proj": (
            factor(e * RANK, hidden, scale=hidden**-0.5),
            factor(2 * inner, RANK * e, scale=0.02),
        ),
        "down_proj": (
            factor(e * RANK, inner, scale=inner**-0.5),
            factor(hidden, RANK * e, scale=0.02),
        ),
    }
    scaling = ALPHA / RANK
    wrapper = SimpleNamespace(
        r={"default": RANK}, active_adapters=["default"], _ep_lora_sharded=True
    )
    module._sonicmoe_lora = {k: (a, b, scaling) for k, (a, b) in factors.items()}
    module._sonicmoe_lora_wrappers = dict.fromkeys(factors, wrapper)
    return [t for pair in factors.values() for t in pair]


def _median_iqr(values):
    q = statistics.quantiles(values, n=4)
    return statistics.median(values), q[0], q[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--routing", choices=("balanced", "skewed"), required=True)
    parser.add_argument(
        "--variant",
        choices=("fused_lora", "dispatch_lora", "fused_full"),
        required=True,
    )
    parser.add_argument(
        "--precision", choices=("bf16", "mxfp8", "fp8"), default="mxfp8"
    )
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--profile",
        type=int,
        default=0,
        help="after timing, profile this many (d) steps and report GPU time by kernel (rank 0)",
    )
    args = parser.parse_args()
    tokens, hidden, inner, num_experts, topk = SHAPES[args.shape]

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    group = dist.group.WORLD

    from mono_ep import MonoEPDispatchRuntime, MonoEPRuntime
    from mono_ep.weight_cache import invalidate_weight_caches

    from axolotl.integrations.expert_parallel import (
        experts_fn,
        fused_lora,
        torch_dispatch,
    )

    torch_dispatch.set_ep_group(group)
    experts_fn.register_all()
    experts_fn.set_backend("mono_ep")
    lora = args.variant != "fused_full"
    module = _build_experts(group, hidden, inner, num_experts, topk, frozen=lora)
    adapters = _attach_lora(module, hidden, inner, rank) if lora else []
    state = None
    info = dict(variant=args.variant)
    if args.variant == "dispatch_lora":
        from dispatch_sonicmoe_contract import _prepare_sonicmoe

        info["sonicmoe"] = _prepare_sonicmoe(
            "H100" if torch.cuda.get_device_capability()[0] == 9 else "B200"
        )
        runtime = MonoEPDispatchRuntime(
            group=group, max_tokens_per_rank=tokens, top_k=topk, hidden=hidden
        )
        experts_fn.set_mono_ep_mode("dispatch")
        experts_fn.set_local_implementation("sonicmoe")
        save_ops = set(runtime.save_ops)
    else:
        alignment = MonoEPRuntime.token_alignment()
        precision = args.precision + ("_lora" if lora else "")
        runtime = MonoEPRuntime(
            group=group,
            num_experts=num_experts,
            max_tokens_per_rank=-(-tokens // alignment) * alignment,
            top_k=topk,
            hidden=hidden,
            intermediate=inner,
            precision=precision,
        )
        experts_fn.set_mono_ep_mode("fused")
        info["precision"] = runtime.precision
        save_ops = {"mono_ep::persistent_forward", "mono_ep::lora_forward"}
        if lora:
            state = fused_lora.FusedLoraState()
            fused_lora.set_state(state)
    experts_fn.set_mono_ep_runtime(runtime)
    import mono_ep.lora  # noqa: F401  (registers mono_ep::lora_forward)

    save_ops = {
        getattr(getattr(torch.ops, ns), name).default
        for ns, name in (op.split("::") for op in save_ops)
    }

    def policy(ctx, op, *a, **k):
        if op in save_ops:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    torch.manual_seed(812 + rank)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    x.requires_grad_()
    ids = _routing(args.routing, tokens, num_experts, topk, rank)
    scores = torch.rand(tokens, topk, device="cuda")
    scores = (scores / scores.sum(-1, keepdim=True)).requires_grad_()
    dy = torch.randn_like(x)
    leaves = [x, scores, *adapters] + (
        [] if lora else [module.gate_up_proj, module.down_proj]
    )

    def step():
        """One optimizer step's worth of invalidation, then nothing else."""
        invalidate_weight_caches()
        if state is not None:
            state.cache.invalidate()
        for t in leaves:
            t.grad = None

    def contract_c():
        step()
        checkpoint(
            module, x, ids, scores, use_reentrant=False, preserve_rng_state=False
        ).backward(dy)

    def contract_d():
        step()
        checkpoint(
            module,
            x,
            ids,
            scores,
            use_reentrant=False,
            preserve_rng_state=False,
            context_fn=functools.partial(create_selective_checkpoint_contexts, policy),
        ).backward(dy)

    contracts = {"c": contract_c, "d": contract_d}
    times = {k: [] for k in contracts}
    peaks = dict.fromkeys(contracts, 0.0)
    elapsed = torch.empty(world, device="cuda")
    for iteration in range(WARMUP + args.iters):
        order = list(contracts) if iteration % 2 == 0 else list(contracts)[::-1]
        for name in order:
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            contracts[name]()
            torch.cuda.synchronize()
            ms = torch.tensor([(time.perf_counter() - start) * 1e3], device="cuda")
            dist.all_gather_into_tensor(elapsed, ms)
            peak = torch.tensor(
                [torch.cuda.max_memory_allocated() / 2**30], device="cuda"
            )
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if iteration >= WARMUP:
                times[name].append(max(elapsed.tolist()))
                peaks[name] = max(peaks[name], float(peak))
    profile = None
    if args.profile:
        from torch.profiler import ProfilerActivity, profile as torch_profile

        dist.barrier()
        torch.cuda.synchronize()
        with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile):
                contract_d()
            torch.cuda.synchronize()
        events = [
            e
            for e in prof.key_averages()
            if getattr(e, "device_type", None) is not None
            and str(e.device_type).endswith("CUDA")
            and e.self_device_time_total > 0
        ]
        total = sum(e.self_device_time_total for e in events)
        profile = dict(
            steps=args.profile,
            gpu_ms_per_step=total / 1e3 / args.profile,
            kernels=[
                dict(
                    name=e.key[:160],
                    ms_per_step=e.self_device_time_total / 1e3 / args.profile,
                    calls_per_step=e.count / args.profile,
                )
                for e in sorted(events, key=lambda e: -e.self_device_time_total)[:40]
            ],
        )
    finite = all(
        torch.isfinite(t.grad.float()).all().item()
        for t in leaves
        if t.grad is not None
    )

    rebuild = None
    if state is not None:
        # the per-step merge and the weight forms the fused kernels read, timed apart
        from mono_ep.lora import LoraExperts
        from mono_ep.weight_cache import (
            mxfp8_backward_weight_forms,
            mxfp8_forward_weight_forms,
            weight_layouts,
        )

        from axolotl.integrations.expert_parallel.nvfp4_lora import (
            local_expert_lora,
            map_adapter,
        )

        a1, b1, s = module._sonicmoe_lora["gate_up_proj"]
        a2, b2, _ = module._sonicmoe_lora["down_proj"]
        wr = module._sonicmoe_lora_wrappers["gate_up_proj"]
        a1, b1, e, r = local_expert_lora(a1, b1, module, wr)
        a2, b2, _, _ = local_expert_lora(a2, b2, module, wr)
        a1, b1 = map_adapter(a1.detach(), b1.detach(), e, r, hidden)
        a2, b2 = map_adapter(a2.detach(), b2.detach(), e, r, inner)
        w1, w2 = experts_fn._local_expert_weights(module)
        layer = LoraExperts(scaling=s).bind(w1, w2)

        def forms(m):
            invalidate_weight_caches()
            if args.precision == "fp8":
                from mono_ep.sm90_weight_forms import quantize_all_forms

                quantize_all_forms(m.w1, m.w2)
            elif args.precision == "mxfp8":
                mxfp8_forward_weight_forms(m.w1, m.w2)
                mxfp8_backward_weight_forms(m.w1, m.w2)
            else:
                weight_layouts(m.w1, m.w2, "w1_interleaved", "w2", "w2t", "w1t")

        def timed(fn, reps=20):
            fn()
            torch.cuda.synchronize()
            samples = []
            for _ in range(reps):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                fn()
                b.record()
                b.synchronize()
                samples.append(a.elapsed_time(b))
            return statistics.median(samples)

        from mono_ep.lora import _fused_mxfp8_forms

        merged = layer.merged(a1, b1, a2, b2)
        fused_forms = _fused_mxfp8_forms(runtime._moe, w1)
        rebuild = dict(
            merge_ms=timed(lambda: layer.merged(a1, b1, a2, b2)),
            forms_ms=timed(lambda: forms(merged)),
            merged_gib=merged.nbytes() / 2**30,
            fused_forms=fused_forms,
            fused_forms_ms=timed(
                lambda: layer.bind(w1, w2, True).merged(a1, b1, a2, b2)
            )
            if fused_forms
            else None,
            cache=state.stats(),
        )
    result = dict(
        shape=args.shape,
        routing=args.routing,
        world=world,
        tokens_per_gpu=tokens,
        lora_rank=RANK if lora else None,
        **info,
        iters=args.iters,
        finite=finite,
        contracts={
            name: dict(
                zip(("median_ms", "q1_ms", "q3_ms"), _median_iqr(v), strict=True),
                peak_gib=peaks[name],
                samples=v,
            )
            for name, v in times.items()
        },
        rebuild=rebuild,
        profile=profile,
        env={
            k: os.environ.get(k)
            for k in (
                "NCCL_MAX_CTAS",
                "MONO_EP_WORKSPACE_BOUND",
                "MONO_EP_STORED_CONTEXT",
                "MONO_EP_LORA_MERGED_CACHE",
                "MONO_EP_SM100_DW_PRECISION",
                "MONO_EP_LORA_MXFP8_FORMS",
                "MONO_EP_LORA_WGRAD",
                "MONO_EP_LORA_MXFP8_POOLS",
            )
        },
    )
    if rank == 0:
        print("LORA_BENCH " + json.dumps({k: v for k, v in result.items()}))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=1)
    dist.barrier()
    if hasattr(runtime, "destroy"):
        runtime.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
