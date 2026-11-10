# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""One Qwen3-30B-A3B MoE layer at EP8 under the fused mono-ep benchmark contract, but with
the dispatch mode around SonicMoE / grouped_mm against torch all-to-all.

Same shapes, routing generators, scopes (forward / backward / checkpointed 2F+B), warmups,
sample counts and rank aggregation as ``benchmark_persistent_training.py``; expert weights
are BF16 (SonicMoE's native dtype), where the fused line keeps FP32 masters quantized per call.
``--deep-ep`` adds axolotl's DeepEP backend as a baseline (same scopes, routings and
samples; rows moved per transport, a ``num_sms`` sweep and a profiler breakdown).
Run under ``torch.distributed.run --nproc-per-node=8`` with ``--worker``."""

import argparse
import functools
import importlib.metadata
import inspect
import json
import os
import re
import statistics
import subprocess  # nosec B404
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

TOKENS, HIDDEN, INNER, EXPERTS, TOPK = 16384, 2048, 768, 128, 8
WARMUP, MEASURED = 5, 15
SCOPES = ("forward", "backward", "checkpointed_layer")
# (label, backend, local kernel, mono_ep runtime): the original four variants, the legacy
# per-expert row kernels now named as such, plus the streamed transfers (the dispatch
# mode's default), the compact receive rows, and the fused mono_ep runtime (mode=fused,
# no local kernel) re-measured in the same allocation
VARIANTS = (
    ("torch+sonicmoe", "torch", "sonicmoe", None),
    ("mono_ep_legacy+sonicmoe", "mono_ep", "sonicmoe", "legacy"),
    ("mono_ep_legacy+grouped_mm", "mono_ep", "grouped_mm", "legacy"),
    ("torch+grouped_mm", "torch", "grouped_mm", None),
    ("mono_ep+sonicmoe", "mono_ep", "sonicmoe", "streamed"),
    ("mono_ep+grouped_mm", "mono_ep", "grouped_mm", "streamed"),
    ("mono_ep_compact+sonicmoe", "mono_ep", "sonicmoe", "compact"),
    ("mono_ep_compact+grouped_mm", "mono_ep", "grouped_mm", "compact"),
    ("mono_ep_fused", "mono_ep", None, "fused"),
)
RUNTIMES = {
    "legacy": {"transfer": "legacy", "compact": False},
    "streamed": {"transfer": "streamed", "compact": False},
    "compact": {"transfer": "streamed", "compact": True},
}
# the fused reference lines: FP8 (blockwise) on H100 with FP8 transport + combine, BF16 on B200
FUSED_PRECISION = {"H100": "fp8", "B200": "bf16"}
FUSED_PROVENANCE = (
    "fused kernels of the mono-ep base (features/mono-ep @ 9ae994be97: SM90/SM100 persistent "
    "kernels from before B200 rounds 1-3, no two-row dispatch pulls, forward lookahead or "
    "weight-layout cache, no MONO_EP_SM100_* selectors); the current fused reference for this "
    "contract is the B200 round-3 build (f73dad341d): 7.66 ms checkpointed 2F+B on 8xB200, "
    "13.48 ms on 8xH100 (FP8); this row is a same-allocation sanity point, not the fused-vs-dispatch gap"
)
FUSED_ENV = (
    "MONO_EP_SM90_FP8_TRANSPORT",
    "MONO_EP_SM90_FP8_COMBINE",
    "MONO_EP_SM90_FP8_TRAP_TIMEOUTS",
    "NCCL_MAX_CTAS",
    "DG_JIT_CACHE_DIR",
)
PARITY_PAIRS = (
    ("mono_ep_fused", "torch+sonicmoe"),
    ("mono_ep_legacy+sonicmoe", "torch+sonicmoe"),
    ("mono_ep_legacy+grouped_mm", "torch+grouped_mm"),
    ("mono_ep+sonicmoe", "torch+sonicmoe"),
    ("mono_ep+grouped_mm", "torch+grouped_mm"),
    ("mono_ep_compact+sonicmoe", "torch+sonicmoe"),
    ("mono_ep_compact+grouped_mm", "torch+grouped_mm"),
    ("mono_ep+sonicmoe", "mono_ep_legacy+sonicmoe"),
    ("mono_ep_compact+sonicmoe", "mono_ep_legacy+sonicmoe"),
)
# DeepEP rows (``--deep-ep``): axolotl's ``_DeepEPBackend`` (normal-mode intranode kernels,
# K-wide rows, the plugin's buffer sizing and preset configs) around the same local kernels.
# ``deep_ep`` runs the kernels alone (``AXOLOTL_EP_NO_BARRIER=1``); ``deep_ep_barrier`` is
# the backend as shipped, with ``barrier_ep`` (device sync + NCCL barrier) before every
# combine. ``num_sms`` None = ``deep_ep.Buffer.num_sms`` as the backend leaves it.
DEEP_EP_VARIANTS = (
    ("deep_ep+sonicmoe", "deep_ep", "sonicmoe", "deep_ep"),
    ("deep_ep+grouped_mm", "deep_ep", "grouped_mm", "deep_ep"),
    ("deep_ep_barrier+sonicmoe", "deep_ep", "sonicmoe", "deep_ep_barrier"),
)
DEEP_EP_SETTINGS = {
    "deep_ep": {"ep_barrier": False, "num_sms": None},
    "deep_ep_barrier": {"ep_barrier": True, "num_sms": None},
}
DEEP_EP_PARITY_PAIRS = (
    ("deep_ep+sonicmoe", "torch+sonicmoe"),
    ("deep_ep+grouped_mm", "torch+grouped_mm"),
    ("deep_ep_barrier+sonicmoe", "deep_ep+sonicmoe"),
    ("mono_ep_compact+sonicmoe", "deep_ep+sonicmoe"),
)
# rel-L2 bound for the DeepEP rows against torch + the same local kernel: the torch backend
# combines in BF16 (``index_add_``), DeepEP in FP32 with one rounding, which the mono_ep rows
# showed at 3.4e-3; a BF16 ulp is 7.8e-3
DEEP_EP_PARITY_REL_L2 = 1e-2
# the normal-kernel SM budget (even, channels = num_sms / 2). Values up to 40 fit the
# plugin's 256 MiB NVL buffer; ``--deep-ep-num-sms-max`` above that sizes the buffer to the
# largest hint instead (the kernels address the buffer per channel, so the default rows are
# unaffected) and records both sizes
DEEP_EP_NUM_SMS_SWEEP = (20, 24, 32, 40, 48, 64)
DEEP_EP_SWEEP_WARMUP, DEEP_EP_SWEEP_MEASURED = 3, 10
DEEP_EP_KNOB_THRESHOLD = 0.05
PROFILE_VARIANTS = (
    "torch+sonicmoe",
    "mono_ep+sonicmoe",
    "mono_ep_compact+sonicmoe",
    "deep_ep+sonicmoe",
    "deep_ep_barrier+sonicmoe",
)
# kernel-name buckets for the profiler breakdown (lower-cased substrings, first match wins)
PROFILE_BUCKETS = (
    ("deep_ep", ("deep_ep", "intranode", "notify_dispatch", "dispatch_layout")),
    ("mono_ep", ("_pull_rows", "_push_rows", "_reduce_slots", "symm", "barrier")),
    ("nccl", ("nccl",)),
    ("gemm", ("gemm", "quack", "cutlass", "cute", "grouped", "sonic")),
    ("copy", ("memcpy", "memset", "copy")),
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


def _build_fused_runtime(group, profile_set):
    """The production fused runtime (``MonoEPRuntime``) on this allocation, or ``None`` with
    the reason when the image has no vendored DeepGEMM build."""
    import deep_gemm
    from mono_ep import MonoEPRuntime

    # the symmetric buffer rounds the capacity up to DeepGEMM's token alignment, as the
    # plugin does (1920 on the current vendored build: 16384 tokens -> 17280 slots)
    alignment = MonoEPRuntime.token_alignment()
    capacity = -(-TOKENS // alignment) * alignment
    runtime = MonoEPRuntime(
        group=group,
        num_experts=EXPERTS,
        max_tokens_per_rank=capacity,
        top_k=TOPK,
        hidden=HIDDEN,
        intermediate=INNER,
        precision=FUSED_PRECISION[profile_set],
    )
    assert runtime.max_tokens_per_rank >= TOKENS
    return runtime, {
        "mode": "fused",
        "max_tokens_per_rank": runtime.max_tokens_per_rank,
        "token_alignment": alignment,
        "class": "mono_ep.MonoEPRuntime",
        "precision": runtime.precision,
        "backward_backend": runtime.backward_backend,
        "reserved_sms": runtime.reserved_sms,
        "deep_gemm_num_sms": deep_gemm.get_num_sms(),
        "masters": "bfloat16 parameters quantized per call (the r3 reference runs kept FP32 masters)",
        "compiled": False,
        "provenance": FUSED_PROVENANCE,
        "env": {name: os.environ.get(name) for name in FUSED_ENV},
    }


def _preset_line(function, num_ranks):
    """The ``Config(...)`` literal DeepEP's preset table holds for ``num_ranks``."""
    for line in inspect.getsource(function).splitlines():
        match = re.match(rf"\s*{num_ranks}:\s*(Config\(.*\)),?\s*$", line)
        if match:
            return match.group(1)
    return None


def _setup_deep_ep(group, num_sms_max):
    """Build axolotl's DeepEP buffer the way the EP plugin does (``configure_buffer`` with
    the ``ExpertParallelArgs`` defaults, one normal-mode ``Buffer`` per process, preset
    dispatch / combine configs for the group size) and record the build and the knobs."""
    import deep_ep
    import deep_ep_cpp

    from axolotl.integrations.expert_parallel import buffer as ep_buffer
    from axolotl.integrations.expert_parallel.args import ExpertParallelArgs

    fields = ExpertParallelArgs.model_fields
    num_nvl_bytes_default = fields["expert_parallel_num_nvl_bytes"].default
    num_rdma_bytes = fields["expert_parallel_num_rdma_bytes"].default
    world = dist.get_world_size(group)
    sweep = tuple(n for n in DEEP_EP_NUM_SMS_SWEEP if n <= num_sms_max)
    hints = {}
    for num_sms in DEEP_EP_NUM_SMS_SWEEP:
        # the dispatch preset for 8 ranks (recv 256 tokens per channel) bounds the buffer
        config = deep_ep.Config(num_sms, 6, 256, 6, 128)
        hints[num_sms] = config.get_nvl_buffer_size_hint(HIDDEN * 2, world)
    num_nvl_bytes = max(num_nvl_bytes_default, *(hints[n] for n in sweep))
    ep_buffer.configure_buffer(
        group, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=num_rdma_bytes
    )
    buffer = ep_buffer.get_buffer()
    default_num_sms = deep_ep.Buffer.num_sms
    ep_buffer.get_dispatch_config()
    ep_buffer.get_combine_config()
    try:
        nvshmem_wheel = importlib.metadata.version(
            f"nvidia-nvshmem-cu{torch.version.cuda.split('.')[0]}"
        )
    except importlib.metadata.PackageNotFoundError:
        nvshmem_wheel = None
    ldd = subprocess.run(  # nosec B603 B607
        ["ldd", deep_ep_cpp.__file__], capture_output=True, text=True, check=False
    ).stdout
    facts = {
        "package_version": importlib.metadata.version("deep_ep"),
        "module": deep_ep.__file__,
        "nvshmem_wheel": nvshmem_wheel,
        "cpp_links_nvshmem": [
            line.split()[0] for line in ldd.splitlines() if "nvshmem" in line
        ],
        "buffer": {
            "class": "deep_ep.Buffer",
            "low_latency_mode": False,
            "num_nvl_bytes": num_nvl_bytes,
            "num_nvl_bytes_default": num_nvl_bytes_default,
            "num_rdma_bytes": num_rdma_bytes,
            "group": "WORLD (the EP group)",
            "source": "axolotl.integrations.expert_parallel.buffer.configure_buffer with the ExpertParallelArgs defaults",
        },
        "num_sms_default": default_num_sms,
        "dispatch_config_preset": _preset_line(
            deep_ep.Buffer.get_dispatch_config, world
        ),
        "combine_config_preset": _preset_line(deep_ep.Buffer.get_combine_config, world),
        "nvl_buffer_size_hint_bytes_by_num_sms": hints,
        "num_sms_sweep": sweep,
        "dispatch_kwargs": {
            "async_finish": False,
            "allocate_on_comm_stream": False,
            "expert_alignment": 1,
            "num_worst_tokens": 0,
            "previous_event": None,
        },
        "layout": "K-wide: one row per (token, destination rank), top-k ids / weights per row; get_dispatch_layout per forward",
        "barrier_ep": "torch.cuda.synchronize + dist.barrier before every combine and backward-of-dispatch unless AXOLOTL_EP_NO_BARRIER is set",
        "device_capability": list(torch.cuda.get_device_capability()),
    }
    return buffer, facts


def _configure_deep_ep(setting, default_num_sms, world):
    """Apply a DeepEP row's knobs: the ``barrier_ep`` env switch and ``Buffer.num_sms``
    (the plugin never sets it, so the preset configs cached by axolotl's buffer module are
    rebuilt only when a row asks for a different SM count)."""
    import deep_ep

    from axolotl.integrations.expert_parallel import buffer as ep_buffer

    if setting["ep_barrier"]:
        os.environ.pop("AXOLOTL_EP_NO_BARRIER", None)
    else:
        os.environ["AXOLOTL_EP_NO_BARRIER"] = "1"
    num_sms = setting["num_sms"] or default_num_sms
    if deep_ep.Buffer.num_sms != num_sms:
        deep_ep.Buffer.set_num_sms(num_sms)
        ep_buffer._DISPATCH_CONFIG = deep_ep.Buffer.get_dispatch_config(world)
        ep_buffer._COMBINE_CONFIG = deep_ep.Buffer.get_combine_config(world)


def _transfer_rows(buffer, ids, world):
    """Rows each transport moves for this routing, from DeepEP's dispatch layout: one
    K-wide row per (token, destination rank) for DeepEP and for the streamed / compact
    dispatch modes (both mask unrouted rows at the load, ``peer_stream._pull_rows``); the
    fused runtime's expert-packed layout duplicates a token per top-k slot; the fixed
    receive layout the streamed compute sees is ``world * capacity`` rows."""
    with torch.no_grad():
        num_per_rank, _, _, _, _ = buffer.get_dispatch_layout(ids, EXPERTS)
    sent = num_per_rank.to(torch.int64)
    matrix = torch.empty(world, world, device="cuda", dtype=torch.int64)
    dist.all_gather_into_tensor(matrix, sent.reshape(1, world))
    remote = matrix.clone()
    remote.fill_diagonal_(0)
    received = matrix.sum(0)
    received_remote = remote.sum(0)
    row_bytes = HIDDEN * 2
    per_rank_remote = received_remote.float()

    def bytes_per_direction(rows_per_rank):
        return {
            "dispatch_x_bf16": rows_per_rank * row_bytes,
            "combine_y_bf16": rows_per_rank * row_bytes,
            "backward_combine_dy_bf16": rows_per_rank * row_bytes,
            "backward_dispatch_dx_bf16": rows_per_rank * row_bytes,
        }

    k_wide_remote_mean = per_rank_remote.mean().item()
    k_wide_remote_max = per_rank_remote.max().item()
    return {
        "rows_per_rank": {
            "k_wide_received_total_mean": received.float().mean().item(),
            "k_wide_received_remote_mean": k_wide_remote_mean,
            "k_wide_received_remote_max": k_wide_remote_max,
            "k_wide_received_remote_min": per_rank_remote.min().item(),
            "k_duplicated_slots": TOKENS * TOPK,
            "fixed_layout_rows": world * TOKENS,
            "tokens": TOKENS,
        },
        "send_matrix_rows": matrix.tolist(),
        "nvlink_bytes_per_rank_per_direction": {
            "deep_ep": bytes_per_direction(k_wide_remote_mean),
            "mono_ep_streamed": bytes_per_direction(k_wide_remote_mean),
            "mono_ep_compact": bytes_per_direction(k_wide_remote_mean),
            "max_rank_factor": k_wide_remote_max / max(k_wide_remote_mean, 1.0),
        },
        "metadata_bytes_per_rank": {
            "deep_ep_per_dispatch": k_wide_remote_mean * TOPK * (8 + 4)
            + k_wide_remote_mean * 4,
            "mono_ep_per_dispatch": (world - 1) * TOKENS * TOPK * (8 + 4),
            "mono_ep_per_backward_dispatch_gw_fp32": k_wide_remote_mean * TOPK * 4,
            "note": "DeepEP ships top-k ids (int64) + weights (fp32) + source index per routed row; mono_ep's collect_routing reads every peer's full idx / w slabs (capacity rows) and pushes fp32 top-k weight grads per routed row",
        },
        "note": "payload rows are identical across the three transports; streamed differs from compact only in the rows the local kernel sees (fixed_layout_rows vs k_wide_received_total)",
    }


def _profile_breakdown(label, step):
    """Self device time of one fwd+bwd step on this rank, bucketed by kernel family."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    buckets = {name: 0.0 for name, _ in PROFILE_BUCKETS}
    buckets["other"] = 0.0
    kernels = []
    for event in prof.key_averages():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = event.key
        ms = event.self_device_time_total / 1000.0
        if ms <= 0:
            continue
        lowered = name.lower()
        bucket = next(
            (b for b, needles in PROFILE_BUCKETS if any(n in lowered for n in needles)),
            "other",
        )
        buckets[bucket] += ms
        kernels.append((ms, event.count, name[:160], bucket))
    kernels.sort(reverse=True)
    return {
        "label": label,
        "device_ms_by_bucket": buckets,
        "device_ms_total": sum(buckets.values()),
        "top_kernels": [
            {"ms": ms, "count": count, "name": name, "bucket": bucket}
            for ms, count, name, bucket in kernels[:14]
        ],
    }


def run(profile_set, deep_ep=False, deep_ep_num_sms_max=40):
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
    runtimes = {
        name: MonoEPDispatchRuntime(
            group=group, max_tokens_per_rank=TOKENS, top_k=TOPK, hidden=HIDDEN, **kw
        )
        for name, kw in RUNTIMES.items()
    }
    try:
        fused, fused_info = _build_fused_runtime(group, profile_set)
    except Exception as error:  # the stock images carry no vendored DeepGEMM build
        fused, fused_info = None, {"mode": "fused", "unavailable": repr(error)}
    # every rank must run the same variant list (the ops are collective)
    fused_everywhere = torch.tensor(float(fused is not None), device="cuda")
    dist.all_reduce(fused_everywhere, op=dist.ReduceOp.MIN)
    if fused is not None and fused_everywhere.item() == 0.0:
        fused.destroy()
        fused, fused_info = (
            None,
            {"mode": "fused", "unavailable": "failed on another rank"},
        )
    variants = [v for v in VARIANTS if v[3] != "fused" or fused is not None]
    pairs = list(PARITY_PAIRS)
    deep_ep_buffer, deep_ep_info, deep_ep_default_num_sms = None, None, None
    if deep_ep:
        deep_ep_buffer, deep_ep_info = _setup_deep_ep(group, deep_ep_num_sms_max)
        deep_ep_default_num_sms = deep_ep_info["num_sms_default"]
        variants.extend(DEEP_EP_VARIANTS)
        pairs.extend(DEEP_EP_PARITY_PAIRS)

    def parity_pairs_present():
        labels_present = {label for label, _, _, _ in variants}
        return tuple(
            pair
            for pair in pairs
            if pair[0] in labels_present and pair[1] in labels_present
        )

    workspace = runtimes["streamed"]._dispatch.workspace
    receive_buffers = {
        "receive_rows": runtimes["streamed"].receive_rows,
        "recv_x_bf16_bytes": runtimes["streamed"].receive_rows * HIDDEN * 2,
        "symmetric_local_bytes_per_runtime": {
            name: views[workspace.rank].nbytes
            for name, views in workspace.views.items()
        },
        "runtimes_resident": list(RUNTIMES),
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

    def configure(backend, local, runtime):
        experts_fn.set_backend(backend)
        if backend == "deep_ep":
            experts_fn.set_local_implementation(local)
            _configure_deep_ep(
                DEEP_EP_SETTINGS[runtime], deep_ep_default_num_sms, world
            )
            return
        if runtime == "fused":
            experts_fn.set_mono_ep_mode("fused")
            experts_fn.set_mono_ep_runtime(fused)
            return
        experts_fn.set_mono_ep_mode("dispatch")
        experts_fn.set_local_implementation(local)
        if runtime is not None:
            experts_fn.set_mono_ep_runtime(runtimes[runtime])

    rank_elapsed = torch.empty(world, device="cuda", dtype=torch.float32)

    def fwd_bwd(ids):
        for tensor in grads_of:
            tensor.grad = None
        operation(x, ids, scores).backward(dy)

    def timed_fwd_bwd(ids, warmup, measured):
        """Median over iterations of the max-rank wall ms of one fwd+bwd."""
        samples = []
        for iteration in range(warmup + measured):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            fwd_bwd(ids)
            torch.cuda.synchronize()
            elapsed = torch.tensor((time.perf_counter() - start) * 1000, device="cuda")
            dist.all_gather_into_tensor(rank_elapsed, elapsed.reshape(1))
            if iteration >= warmup:
                samples.append(max(rank_elapsed.tolist()))
        return statistics.median(samples)

    def num_sms_sweep(ids):
        """``deep_ep+sonicmoe`` fwd+bwd per SM budget; adds a full row for the best budget
        when it moves the default by more than ``DEEP_EP_KNOB_THRESHOLD``."""
        num_nvl_bytes = deep_ep_info["buffer"]["num_nvl_bytes"]
        sweep = {}
        for num_sms in deep_ep_info["num_sms_sweep"]:
            hint = deep_ep_info["nvl_buffer_size_hint_bytes_by_num_sms"][num_sms]
            if hint > num_nvl_bytes:
                sweep[num_sms] = {"skipped": f"buffer hint {hint} > {num_nvl_bytes}"}
                continue
            experts_fn.set_backend("deep_ep")
            experts_fn.set_local_implementation("sonicmoe")
            _configure_deep_ep(
                {"ep_barrier": False, "num_sms": num_sms},
                deep_ep_default_num_sms,
                world,
            )
            sweep[num_sms] = {
                "fwd_bwd_ms": timed_fwd_bwd(
                    ids, DEEP_EP_SWEEP_WARMUP, DEEP_EP_SWEEP_MEASURED
                )
            }
        default_ms = sweep[deep_ep_default_num_sms]["fwd_bwd_ms"]
        best = min((v["fwd_bwd_ms"], k) for k, v in sweep.items() if "fwd_bwd_ms" in v)
        added = None
        if best[1] != deep_ep_default_num_sms and (
            abs(best[0] - default_ms) / default_ms > DEEP_EP_KNOB_THRESHOLD
        ):
            added = f"deep_ep_sms{best[1]}"
            DEEP_EP_SETTINGS[added] = {"ep_barrier": False, "num_sms": best[1]}
            variants.append((f"{added}+sonicmoe", "deep_ep", "sonicmoe", added))
            pairs.append((f"{added}+sonicmoe", "deep_ep+sonicmoe"))
        return {
            "scope": "fwd+bwd, balanced routing, median max-rank ms",
            "warmup_iterations": DEEP_EP_SWEEP_WARMUP,
            "measured_iterations": DEEP_EP_SWEEP_MEASURED,
            "by_num_sms": sweep,
            "default_num_sms": deep_ep_default_num_sms,
            "best_num_sms": best[1],
            "best_vs_default": best[0] / default_ms,
            "threshold": DEEP_EP_KNOB_THRESHOLD,
            "row_added": added,
        }

    results = {}
    autotune_warm_seconds = {}
    tolerance_checks = {}
    sweep_outcome = None
    transfer_rows = {}
    breakdowns = {}
    for routing in ("balanced", "skewed"):
        ids = _routing(routing, rank)
        # warm SonicMoE's quack sweeps (serial in-process compile) outside the timed loop
        started = time.perf_counter()
        for _label, backend, local, runtime_name in variants:
            configure(backend, local, runtime_name)
            fwd_bwd(ids)
        torch.cuda.synchronize()
        dist.barrier()
        autotune_warm_seconds[routing] = time.perf_counter() - started
        if deep_ep:
            if routing == "balanced":
                sweep_outcome = num_sms_sweep(ids)
            transfer_rows[routing] = _transfer_rows(deep_ep_buffer, ids, world)
            breakdowns[routing] = {}
            for label, backend, local, runtime_name in variants:
                if label not in PROFILE_VARIANTS:
                    continue
                configure(backend, local, runtime_name)
                fwd_bwd(ids)  # one unprofiled step after the last config switch
                torch.cuda.synchronize()
                dist.barrier()
                breakdowns[routing][label] = _profile_breakdown(
                    label, functools.partial(fwd_bwd, ids)
                )

        cases = [
            (scope, label, backend, local, runtime_name)
            for scope in SCOPES
            for label, backend, local, runtime_name in variants
        ]
        parity_pairs = parity_pairs_present()
        labels = [f"{scope}/{label}" for scope, label, _, _, _ in cases]
        times = {label: [] for label in labels}
        mean_rank_times = {label: [] for label in labels}
        rank_times = {label: [] for label in labels}
        peak_memory = {label: [0, 0] for label in labels}
        captured = {}
        for iteration in range(WARMUP + MEASURED):
            ordered = cases if iteration % 2 == 0 else cases[::-1]
            for scope, variant, backend, local, runtime_name in ordered:
                label = f"{scope}/{variant}"
                configure(backend, local, runtime_name)
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
        for candidate, reference in parity_pairs:
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
                if candidate.startswith("deep_ep") or reference.startswith("deep_ep"):
                    worst = max(v["rel_norm_max_rank"] for v in entry.values())
                    tolerance_checks[
                        f"{routing}/{candidate}_vs_{reference}/{scope}"
                    ] = {
                        "worst_rel_l2": worst,
                        "within_bf16_tolerance": worst <= DEEP_EP_PARITY_REL_L2,
                    }
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
    barriers = {
        name: runtime._dispatch.extra_barriers for name, runtime in runtimes.items()
    }
    for runtime in runtimes.values():
        runtime.destroy()
    if fused is not None:
        fused.destroy()
    parity_failures = [
        k for k, v in tolerance_checks.items() if not v["within_bf16_tolerance"]
    ]

    def variant_info(backend, local, runtime_name):
        info = {"backend": backend, "local_kernel": local, "mono_ep_runtime": None}
        if backend == "mono_ep":
            info["mono_ep_runtime"] = RUNTIMES.get(runtime_name, fused_info)
        elif backend == "deep_ep":
            info["deep_ep"] = DEEP_EP_SETTINGS[runtime_name]
        return info

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
                    "master_dtype": "bfloat16 (SonicMoE's native dtype; the fused variant here quantizes the same BF16 parameters per call, the r3 reference runs kept FP32 masters)",
                    "transport_dtype": {
                        "mono_ep": "bf16 rows, fp32 combine accumulation",
                        "torch": "bf16 all_to_all_single",
                        "deep_ep": "bf16 rows, fp32 combine accumulation (normal-mode intranode kernels)",
                    },
                    "mono_ep_barriers": "one symmetric-memory barrier per op (dispatch, combine, each backward), plus extra_barriers when two same-family ops ran back to back",
                    "variants": {
                        label: variant_info(backend, local, runtime_name)
                        for label, backend, local, runtime_name in variants
                    },
                    "fused_runtime": fused_info,
                    "deep_ep": deep_ep_info,
                    "deep_ep_num_sms_sweep": sweep_outcome,
                    "deep_ep_parity_tolerance": {
                        "rel_l2_bound": DEEP_EP_PARITY_REL_L2,
                        "checks": tolerance_checks,
                        "failures": parity_failures,
                    },
                    "transfer_rows": transfer_rows,
                    "profiler_breakdown_rank0": {
                        "scope": "one fwd+bwd per variant under torch.profiler after the warm; self device ms by kernel family on rank 0",
                        "routings": breakdowns,
                    },
                    "mono_ep_extra_barriers_per_runtime": barriers,
                    "scopes": SCOPES,
                    "checkpoint_scope": "torch_nonreentrant_2F_plus_B",
                    "router_in_timing": False,
                    "optimizer_in_timing": False,
                    "warmup_iterations": WARMUP,
                    "measured_iterations": MEASURED,
                    "timing": "host_wall_synchronized; max and arithmetic mean across ranks per iteration, median across measured iterations",
                    "memory_scope": "max across measured iterations and ranks; all variants share one process, the three dispatch runtimes, the fused runtime and (with --deep-ep) the DeepEP buffer are resident throughout",
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
    assert not parity_failures, f"DeepEP rows outside bf16 tolerance: {parity_failures}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profiles", choices=("H100", "B200"), required=True)
    parser.add_argument(
        "--deep-ep",
        action="store_true",
        help="add the DeepEP rows (needs the deep_ep package in the image)",
    )
    parser.add_argument(
        "--deep-ep-num-sms-max",
        type=int,
        default=40,
        help="largest num_sms in the DeepEP sweep (above 40 the buffer grows past the plugin default)",
    )
    args = parser.parse_args()
    if not args.worker:
        parser.error("run under torch.distributed.run with --worker")
    run(
        args.profiles,
        deep_ep=args.deep_ep,
        deep_ep_num_sms_max=args.deep_ep_num_sms_max,
    )
