# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""One Qwen3-30B-A3B MoE layer at EP8 under the fused mono-ep benchmark contract, but with
the dispatch mode around SonicMoE / grouped_mm against torch all-to-all.

Same shapes, routing generators, scopes (forward / backward / checkpointed 2F+B), warmups,
sample counts and rank aggregation as ``benchmark_persistent_training.py``; expert weights
are BF16 (SonicMoE's native dtype), where the fused line keeps FP32 masters quantized per call.
Run under ``torch.distributed.run --nproc-per-node=8`` with ``--worker``."""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

TOKENS, HIDDEN, INNER, EXPERTS, TOPK = 16384, 2048, 768, 128, 8
WARMUP, MEASURED = 5, 15
SCOPES = ("forward", "backward", "checkpointed_layer")
VARIANTS = (
    ("torch+sonicmoe", "torch", "sonicmoe"),
    ("mono_ep+sonicmoe", "mono_ep", "sonicmoe"),
    ("mono_ep+grouped_mm", "mono_ep", "grouped_mm"),
    ("torch+grouped_mm", "torch", "grouped_mm"),
)
PARITY_PAIRS = (
    ("mono_ep+sonicmoe", "torch+sonicmoe"),
    ("mono_ep+grouped_mm", "torch+grouped_mm"),
)


def _skewed_routing(tokens, experts, topk, rank):
    """Exponentially decaying expert popularity in expert order (benchmark_ep8_trace.py)."""
    generator = torch.Generator(device="cuda").manual_seed(812 + rank)
    logits = -torch.arange(experts, device="cuda", dtype=torch.float32) / (
        1.5 * experts
    )
    uniform = torch.rand(tokens, experts, device="cuda", generator=generator)
    gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-20)))
    return (logits + gumbel).topk(topk, dim=-1).indices.contiguous()


def _routing(routing, rank):
    torch.manual_seed(812 + rank)
    if routing == "skewed":
        return _skewed_routing(TOKENS, EXPERTS, TOPK, rank)
    return torch.rand(TOKENS, EXPERTS, device="cuda").argsort(-1)[:, :TOPK].contiguous()


def _prepare_sonicmoe(profile_set):
    """Register axolotl's SonicMoE forward, seed the measured quack profiles for
    ``profile_set`` and apply the kernels plugin's quack policy (config defaults: pow2 key
    buckets, sweep on), in the plugin's order. The profiles bound every grouped key to its
    measured shortlist; the policy compiles in-process (the hub-vendored quack's forkserver
    pool cannot import it) and buckets keys."""
    import copy

    from sonicmoe_contract_profiles import PROFILE_SETS, PROFILE_SOURCES
    from transformers.integrations.sonicmoe import load_sonicmoe_kernel

    from axolotl.integrations.kernels.autotune_profiles import apply_autotune_profiles
    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        register_sonicmoe_experts,
    )
    from axolotl.integrations.kernels.plugin import _check_sonicmoe_gpu_compat
    from axolotl.integrations.kernels.quack_autotune_policy import (
        install_quack_key_policy,
    )

    # Same device setup the kernels plugin runs before loading a model: USE_QUACK_GEMM on
    # Blackwell and the cutlass-dsl 4.8 `nvvm.mbarrier_arrive` shim the hub-vendored quack
    # 0.6.1 needs on its SM100 2-CTA release path.
    _check_sonicmoe_gpu_compat()
    register_sonicmoe_experts()
    load_sonicmoe_kernel()
    profiles = copy.deepcopy(PROFILE_SETS[profile_set])
    gpu_name = torch.cuda.get_device_name()
    # the H100 entries also serve an H200 allocation: drop the gpu filter, record the device
    gpu_filter_dropped = any(p["gpu"].lower() not in gpu_name.lower() for p in profiles)
    if gpu_filter_dropped:
        for profile in profiles:
            profile.pop("gpu")
    seeded = apply_autotune_profiles(profiles)
    assert seeded > 0, "apply_autotune_profiles attached to no quack autotuners"
    covered = install_quack_key_policy()
    assert covered > 0, "install_quack_key_policy covered no quack autotuners"
    return {
        "profile_set": profile_set,
        "source": PROFILE_SOURCES[profile_set],
        "profiles": len(profiles),
        "configs": sum(len(p["configs"]) for p in profiles),
        "gpu_name": gpu_name,
        "gpu_filter_dropped": gpu_filter_dropped,
        "autotuners_seeded": seeded,
        "policy_autotuners_covered": covered,
    }


def _autotune_outcome():
    """Per quack GEMM key on this rank: candidates offered (after the profiles), compiled,
    failed, the winner and how it was chosen; plus per-kernel totals."""
    from axolotl.integrations.kernels.sonicmoe_autotune import (
        collect_sonicmoe_autotune_configs,
    )

    keys = []
    per_kernel = {}
    for record in collect_sonicmoe_autotune_configs(with_timings=True):
        benchmarked = record.get("configs_benchmarked")
        failed = record.get("failed_configs", 0)
        keys.append(
            {
                "kernel": record["kernel"],
                "key": record.get("key"),
                "call_args": record.get("call_args"),
                "candidates_offered": record.get("candidate_config_count"),
                "candidates_compiled": (
                    None if benchmarked is None else benchmarked - failed
                ),
                "candidates_failed": failed,
                "winner": record.get("config"),
                "source": record.get("source", "unknown"),
                "sweep_seconds": record.get("sweep_seconds"),
                "timings_ms": record.get("timings_ms"),
            }
        )
        entry = per_kernel.setdefault(
            record["kernel"],
            {
                "keys": 0,
                "candidates_offered_max": 0,
                "candidates_benchmarked": 0,
                "candidates_failed": 0,
                "sources": {},
            },
        )
        entry["keys"] += 1
        entry["candidates_offered_max"] = max(
            entry["candidates_offered_max"], record.get("candidate_config_count") or 0
        )
        entry["candidates_benchmarked"] += benchmarked or 0
        entry["candidates_failed"] += failed
        source = record.get("source", "unknown")
        entry["sources"][source] = entry["sources"].get(source, 0) + 1
    return {"per_kernel": per_kernel, "keys": keys}


def _build_experts(group):
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    from axolotl.integrations.expert_parallel.shard import shard_expert_weights

    config = Qwen3MoeConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=INNER,
        num_experts=EXPERTS,
        num_experts_per_tok=TOPK,
        norm_topk_prob=True,
        hidden_act="silu",
    )
    config._experts_implementation = "expert_parallel"
    container = nn.Module()
    container.experts = Qwen3MoeExperts(config)
    torch.manual_seed(812)
    with torch.no_grad():
        container.experts.gate_up_proj.copy_(
            torch.randn(EXPERTS, 2 * INNER, HIDDEN) / HIDDEN**0.5
        )
        container.experts.down_proj.copy_(
            torch.randn(EXPERTS, HIDDEN, INNER) / INNER**0.5
        )
    container.cuda().to(torch.bfloat16)
    assert shard_expert_weights(container, group) == 1
    experts = container.experts
    assert experts.num_local_experts == EXPERTS // dist.get_world_size(group)
    for parameter in experts.parameters():
        parameter.requires_grad_(True)
    return experts


def run(profile_set):
    from mono_ep import MonoEPDispatchRuntime

    from axolotl.integrations.expert_parallel import experts_fn, torch_dispatch

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    assert world == 8
    group = dist.group.WORLD
    torch_dispatch.set_ep_group(group)
    experts_fn.register_all()
    quack_setup = _prepare_sonicmoe(profile_set)
    experts = _build_experts(group)
    runtime = MonoEPDispatchRuntime(
        group=group, max_tokens_per_rank=TOKENS, top_k=TOPK, hidden=HIDDEN
    )
    experts_fn.set_mono_ep_runtime(runtime)
    experts_fn.set_mono_ep_mode("dispatch")
    workspace = runtime._dispatch.workspace
    receive_buffers = {
        "receive_rows": runtime.receive_rows,
        "recv_x_bf16_bytes": runtime.receive_rows * HIDDEN * 2,
        "symmetric_local_bytes": {
            name: views[workspace.rank].nbytes
            for name, views in workspace.views.items()
        },
    }

    torch.manual_seed(812 + rank)
    x = torch.randn(
        TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    scores = torch.rand(TOKENS, TOPK, device="cuda")
    scores = (scores / scores.sum(-1, keepdim=True)).requires_grad_()
    dy = torch.randn_like(x)
    grads_of = (x, scores, experts.gate_up_proj, experts.down_proj)

    def operation(x, ids, scores):
        return experts(x, ids, scores)

    def configure(backend, local):
        experts_fn.set_backend(backend)
        experts_fn.set_local_implementation(local)

    rank_elapsed = torch.empty(world, device="cuda", dtype=torch.float32)
    results = {}
    autotune_warm_seconds = {}
    for routing in ("balanced", "skewed"):
        ids = _routing(routing, rank)
        # warm SonicMoE's quack sweeps (serial in-process compile) outside the timed loop
        started = time.perf_counter()
        for _label, backend, local in VARIANTS:
            configure(backend, local)
            for tensor in grads_of:
                tensor.grad = None
            operation(x, ids, scores).backward(dy)
        torch.cuda.synchronize()
        dist.barrier()
        autotune_warm_seconds[routing] = time.perf_counter() - started

        cases = [
            (scope, label, backend, local)
            for scope in SCOPES
            for label, backend, local in VARIANTS
        ]
        labels = [f"{scope}/{label}" for scope, label, _, _ in cases]
        times = {label: [] for label in labels}
        mean_rank_times = {label: [] for label in labels}
        rank_times = {label: [] for label in labels}
        peak_memory = {label: [0, 0] for label in labels}
        captured = {}
        for iteration in range(WARMUP + MEASURED):
            ordered = cases if iteration % 2 == 0 else cases[::-1]
            for scope, variant, backend, local in ordered:
                label = f"{scope}/{variant}"
                configure(backend, local)
                for tensor in grads_of:
                    tensor.grad = None
                y = operation(x, ids, scores) if scope == "backward" else None
                dist.barrier()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                if scope == "checkpointed_layer":
                    y = checkpoint(
                        operation,
                        x,
                        ids,
                        scores,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                elif scope != "backward":
                    y = operation(x, ids, scores)
                if scope != "forward":
                    y.backward(dy)
                torch.cuda.synchronize()
                elapsed = torch.tensor(
                    (time.perf_counter() - start) * 1000, device="cuda"
                )
                dist.all_gather_into_tensor(rank_elapsed, elapsed.reshape(1))
                rank_values = rank_elapsed.tolist()
                if iteration >= WARMUP:
                    times[label].append(max(rank_values))
                    mean_rank_times[label].append(statistics.mean(rank_values))
                    rank_times[label].append(rank_values)
                    peak_memory[label][0] = max(
                        peak_memory[label][0], torch.cuda.max_memory_allocated()
                    )
                    peak_memory[label][1] = max(
                        peak_memory[label][1], torch.cuda.max_memory_reserved()
                    )
                if iteration == WARMUP and scope in ("forward", "backward"):
                    captured[label] = (
                        {"y": y.detach().clone()}
                        if scope == "forward"
                        else {
                            "dx": x.grad.clone(),
                            "dscore": scores.grad.clone(),
                            "dgate_up": experts.gate_up_proj.grad.clone(),
                            "ddown": experts.down_proj.grad.clone(),
                        }
                    )
                assert torch.isfinite(y).all()
                if scope != "forward":
                    for tensor in grads_of:
                        assert (
                            tensor.grad is not None
                            and torch.isfinite(tensor.grad).all()
                        )
                del y
        parity = {}
        for candidate, reference in PARITY_PAIRS:
            for scope in ("forward", "backward"):
                entry = {}
                for key, a in captured[f"{scope}/{candidate}"].items():
                    b = captured[f"{scope}/{reference}"][key]
                    diff = (a.float() - b.float()).abs()
                    stats = torch.tensor(
                        [
                            float(not torch.equal(a, b)),
                            float(diff.max()),
                            float(diff.norm() / b.float().norm().clamp_min(1e-12)),
                        ],
                        device="cuda",
                    )
                    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
                    entry[key] = {
                        "bitwise_equal_all_ranks": stats[0].item() == 0.0,
                        "max_abs_diff_max_rank": stats[1].item(),
                        "rel_norm_max_rank": stats[2].item(),
                    }
                parity[f"{candidate}_vs_{reference}/{scope}"] = entry
        captured.clear()
        memory_tensor = torch.tensor(
            list(peak_memory.values()), device="cuda", dtype=torch.int64
        )
        dist.all_reduce(memory_tensor, op=dist.ReduceOp.MAX)
        peak_memory = dict(zip(peak_memory, memory_tensor.cpu().tolist(), strict=True))
        results[routing] = {
            "median_ms": {k: statistics.median(v) for k, v in times.items()},
            "sample_counts": {k: len(v) for k, v in times.items()},
            "mean_rank_median_ms": {
                k: statistics.median(v) for k, v in mean_rank_times.items()
            },
            "mean_rank_mean_ms": {
                k: statistics.mean(v) for k, v in mean_rank_times.items()
            },
            "samples_ms": times,
            "per_rank_samples_ms": rank_times,
            "peak_memory_bytes": {
                label: {"allocated": value[0], "reserved": value[1]}
                for label, value in peak_memory.items()
            },
            "parity_first_measured_iteration": parity,
            "autotune_warm_seconds": autotune_warm_seconds[routing],
        }

    local_outcome = _autotune_outcome()
    outcomes = [None] * world
    dist.all_gather_object(outcomes, local_outcome)
    failed = sum(
        entry["candidates_failed"]
        for outcome in outcomes
        for entry in outcome["per_kernel"].values()
    )
    offered = [
        key["candidates_offered"] or 0
        for outcome in outcomes
        for key in outcome["keys"]
    ]
    runtime.destroy()
    if rank == 0:
        print(
            "RESULT "
            + json.dumps(
                {
                    "benchmark": "qwen3_30b_a3b_dispatch_mode_vs_torch_all_to_all",
                    "device": torch.cuda.get_device_name(),
                    "torch_version": torch.__version__,
                    "world_size": world,
                    "expert_parallel_size": world,
                    "tokens_per_gpu": TOKENS,
                    "hidden": HIDDEN,
                    "intermediate": INNER,
                    "experts": EXPERTS,
                    "topk": TOPK,
                    "local_experts_per_rank": EXPERTS // world,
                    "expert_weight_dtype": "bfloat16",
                    "master_dtype": "bfloat16 (SonicMoE's native dtype; the fused mono-ep line uses FP32 masters quantized per call)",
                    "transport_dtype": {
                        "mono_ep": "bf16 rows, fp32 combine accumulation",
                        "torch": "bf16 all_to_all_single",
                    },
                    "variants": {
                        label: {"backend": backend, "local_kernel": local}
                        for label, backend, local in VARIANTS
                    },
                    "scopes": SCOPES,
                    "checkpoint_scope": "torch_nonreentrant_2F_plus_B",
                    "router_in_timing": False,
                    "optimizer_in_timing": False,
                    "warmup_iterations": WARMUP,
                    "measured_iterations": MEASURED,
                    "timing": "host_wall_synchronized; max and arithmetic mean across ranks per iteration, median across measured iterations",
                    "memory_scope": "max across measured iterations and ranks; all variants share one process, the peer runtime is resident throughout",
                    "receive_buffers": receive_buffers,
                    "quack_profiles": quack_setup,
                    "quack_autotune_per_kernel_rank0": local_outcome["per_kernel"],
                    "quack_autotune_keys_rank0": local_outcome["keys"],
                    "quack_candidates_failed_all_ranks": failed,
                    "quack_candidates_offered_all_ranks": {
                        "keys": len(offered),
                        "min": min(offered, default=None),
                        "max": max(offered, default=None),
                    },
                    "routings": results,
                }
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    assert failed == 0, f"{failed} quack candidates failed across ranks"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profiles", choices=("H100", "B200"), required=True)
    args = parser.parse_args()
    if not args.worker:
        parser.error("run under torch.distributed.run with --worker")
    run(args.profiles)
