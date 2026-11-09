# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Measure persistent MoE training and separately profile its GPU work."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from mono_ep.runtime import PersistentMoE

_COMPACT_FORWARD_ENV = "MONO_EP_SM90_FP8_COMPACT_FORWARD"
_COMPACT_BACKWARD_ENV = "MONO_EP_SM90_FP8_COMPACT_BACKWARD"


def _select_kernel_layouts(forward_layout, backward_layout):
    for name, layout in (
        (_COMPACT_FORWARD_ENV, forward_layout),
        (_COMPACT_BACKWARD_ENV, backward_layout),
    ):
        if layout == "compact384":
            os.environ[name] = "1"
        else:
            os.environ.pop(name, None)


def run(
    profile=False,
    compute_dtype="bf16",
    compare_fusion=False,
    recompute=False,
    peer_backward=False,
    compare_ready=False,
    compare_peer_return=False,
    compare_owned=False,
    compare_nccl=False,
    compare_native=False,
    compiled=False,
    compare_forward_layouts=False,
    compare_kernel_layouts=False,
):
    if compare_forward_layouts and compare_kernel_layouts:
        raise ValueError(
            "Forward-layout and kernel-layout comparisons are separate runs"
        )
    if (compare_forward_layouts or compare_kernel_layouts) and (
        compute_dtype != "blockwise" or not compare_native or not compiled or profile
    ):
        raise ValueError(
            "Kernel-layout comparison requires compiled H200 FP8 native/NCCL timing"
        )
    if compute_dtype == "blockwise" and (
        not compare_native
        or not recompute
        or not compiled
        or peer_backward
        or compare_fusion
        or compare_ready
        or compare_peer_return
        or compare_owned
    ):
        raise ValueError(
            "Hopper FP8 timing requires compiled native/NCCL recomputation only"
        )
    compare_nccl = compare_nccl or compare_native
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    assert world == 8
    tokens, hidden, inner, experts, topk = 16384, 2048, 768, 128, 8
    torch.manual_seed(812 + rank)
    x = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk].contiguous()
    scores = torch.rand(tokens, topk, device="cuda")
    scores = (scores / scores.sum(-1, keepdim=True)).requires_grad_()
    w1 = (
        torch.randn(experts // world, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(experts // world, hidden, inner, device="cuda") / inner**0.5
    ).requires_grad_()
    dy = torch.randn_like(x)
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inner,
        compute_dtype=compute_dtype,
        fuse_activation=compute_dtype == "bf16",
        retain_activations=not recompute,
    )
    if peer_backward:
        from mono_ep.peer_backward import PeerBackward

        runtime._peer_backward = PeerBackward(
            runtime, fused_peer_return=compare_peer_return, owned_blocks=compare_owned
        )
    peer_backend = runtime._peer_backward
    native_backend = None
    if compare_native:
        from mono_ep.persistent_backward import NativePersistentBackward

        native_backend = NativePersistentBackward(runtime)
    if compiled:
        operation = (
            torch.compile(
                runtime,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )
            if compare_forward_layouts or compare_kernel_layouts
            else torch.compile(runtime, fullgraph=True)
        )
    else:
        operation = runtime
    scopes = (
        "forward",
        "backward",
        "recomputed_layer" if recompute else "retained_layer",
        "checkpointed_layer",
    )
    kernel_layouts = (
        (
            ("baseline512", "baseline512"),
            ("compact384", "baseline512"),
            ("compact384", "compact384"),
        )
        if compare_kernel_layouts
        else (
            (("baseline512", "baseline512"), ("compact384", "baseline512"))
            if compare_forward_layouts
            else (("baseline512", "baseline512"),)
        )
    )
    cases = [
        (
            scope,
            fused,
            ready,
            peer_return,
            owned,
            use_peer,
            native,
            forward_layout,
            backward_layout,
            (
                f"{scope}/{'native' if native else 'peer' if use_peer else 'nccl'}"
                if compare_native
                else f"{scope}/{'peer' if use_peer else 'nccl'}"
                if compare_nccl
                else f"{scope}/{'owned_blocks' if owned else 'grouped'}"
                if compare_owned
                else f"{scope}/{'peer_epilogue' if peer_return else 'separate_return'}"
                if compare_peer_return
                else f"{scope}/{'ready' if ready else 'staged'}"
                if compare_ready
                else f"{scope}/{'fused' if fused else 'unfused'}"
                if compare_fusion
                else scope
            )
            + (f"/{forward_layout}" if compare_forward_layouts else "")
            + (
                f"/forward_layout={forward_layout}/backward_layout={backward_layout}"
                f"/native_backward={'selected' if native else 'not_applicable'}"
                if compare_kernel_layouts
                else ""
            ),
        )
        for scope in scopes
        for fused in ((False, True) if compare_fusion else (runtime.fuse_activation,))
        for ready in (
            (False, True)
            if compare_ready
            else (bool(peer_backward and peer_backend.ready_gather),)
        )
        for peer_return in ((False, True) if compare_peer_return else (False,))
        for owned in ((False, True) if compare_owned else (False,))
        for use_peer in ((False, True) if compare_nccl else (peer_backward,))
        for native in ((False, True) if compare_native and use_peer else (False,))
        for forward_layout, backward_layout in kernel_layouts
        if not (compute_dtype == "blockwise" and use_peer and not native)
    ]
    times = {label: [] for _, _, _, _, _, _, _, _, _, label in cases}
    mean_rank_times = {label: [] for label in times}
    rank_times = {label: [] for label in times}
    rank_elapsed = torch.empty(world, device="cuda", dtype=torch.float32)
    peak_memory = {label: [0, 0] for label in times}
    workspace = {"forward_symmetric_local": runtime.buffer.buffer.nbytes}
    if native_backend is not None:
        workspace.update(
            native_backward_symmetric_local=native_backend.buffer.buffer.nbytes,
            native_backward_source_dy=native_backend.dy.nbytes,
            native_backward_source_dy_scales=(
                native_backend.dy_sf.nbytes
                if getattr(native_backend, "dy_sf", None) is not None
                else 0
            ),
            native_backward_score_partials=native_backend.ds_partials.nbytes,
            native_backward_bf16_pools=sum(t.nbytes for t in native_backend.pools[:4]),
            native_backward_fp8_cache=sum(t.nbytes for t in native_backend.pools[4:10]),
            native_backward_packed_wgrad=sum(
                t.nbytes for t in native_backend.pools[10:]
            ),
            native_backward_metadata=native_backend.starts.nbytes
            + native_backend.counts.nbytes
            + (
                0
                if native_backend.padded_counts is None
                else native_backend.padded_counts.nbytes
            ),
        )
    if compare_forward_layouts or compare_kernel_layouts:
        runtime._peer_backward = native_backend
        for forward_layout, backward_layout in kernel_layouts:
            _select_kernel_layouts(forward_layout, backward_layout)
            warmup_y = operation(x, ids, scores, w1, w2)
            if compare_kernel_layouts:
                warmup_y.backward(dy)
            del warmup_y
            for tensor in (x, scores, w1, w2):
                tensor.grad = None
        torch.cuda.synchronize()
    for iteration in range(5 if profile else 20):
        for (
            scope,
            fused,
            ready,
            peer_return,
            owned,
            use_peer,
            native,
            forward_layout,
            backward_layout,
            label,
        ) in cases if iteration % 2 == 0 else cases[::-1]:
            _select_kernel_layouts(forward_layout, backward_layout)
            runtime.fuse_activation = fused
            runtime._peer_backward = (
                native_backend if native else peer_backend if use_peer else None
            )
            if use_peer and not native:
                runtime._peer_backward.ready_gather = ready
                runtime._peer_backward.fused_peer_return = peer_return
                runtime._peer_backward.owned_blocks = owned
            for tensor in (x, scores, w1, w2):
                tensor.grad = None
            y = operation(x, ids, scores, w1, w2) if scope == "backward" else None
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
                    w1,
                    w2,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            elif scope != "backward":
                y = operation(x, ids, scores, w1, w2)
            if scope != "forward":
                y.backward(dy)
            torch.cuda.synchronize()
            elapsed = torch.tensor((time.perf_counter() - start) * 1000, device="cuda")
            dist.all_gather_into_tensor(rank_elapsed, elapsed.reshape(1))
            rank_values = rank_elapsed.tolist()
            if iteration >= 5:
                times[label].append(max(rank_values))
                mean_rank_times[label].append(statistics.mean(rank_values))
                rank_times[label].append(rank_values)
                peak_memory[label][0] = max(
                    peak_memory[label][0], torch.cuda.max_memory_allocated()
                )
                peak_memory[label][1] = max(
                    peak_memory[label][1], torch.cuda.max_memory_reserved()
                )
            assert torch.isfinite(y).all()
            if scope != "forward":
                for tensor in (x, scores, w1, w2):
                    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
            del y
    if not profile:
        memory_tensor = torch.tensor(
            list(peak_memory.values()), device="cuda", dtype=torch.int64
        )
        dist.all_reduce(memory_tensor, op=dist.ReduceOp.MAX)
        peak_memory = dict(zip(peak_memory, memory_tensor.cpu().tolist(), strict=True))
    if rank == 0 and not profile:
        print(
            json.dumps(
                dict(
                    benchmark="qwen3_30b_a3b_persistent_training",
                    device=torch.cuda.get_device_name(),
                    torch_version=torch.__version__,
                    world_size=world,
                    timeout_handler=(
                        "trap_only"
                        if os.getenv("MONO_EP_SM90_FP8_TRAP_TIMEOUTS") == "1"
                        else "diagnostic_calls"
                    ),
                    active_sms=runtime.num_sms,
                    physical_sms=torch.cuda.get_device_properties(
                        x.device
                    ).multi_processor_count,
                    native_backward_bf16_scratch_rows=(
                        native_backend.pools[0].shape[0]
                        if native_backend is not None
                        else None
                    ),
                    native_backward_operand_capacity_rows=(
                        native_backend.pools[4].shape[0]
                        if native_backend is not None and native_backend.blockwise
                        else None
                    ),
                    tokens_per_gpu=tokens,
                    hidden=hidden,
                    intermediate=inner,
                    experts=experts,
                    topk=topk,
                    compute_dtype=compute_dtype,
                    master_dtype="float32",
                    transport_dtype=(
                        "forward_fp8_backward_input_fp8_combine_fp8"
                        if getattr(native_backend, "fp8_combine", False)
                        else "forward_fp8_backward_input_fp8_combine_bf16"
                        if getattr(native_backend, "fp8_transport", False)
                        else "forward_fp8_backward_bf16"
                        if compute_dtype == "blockwise"
                        else "fp8"
                        if compute_dtype == "mxfp8"
                        else "bf16"
                    ),
                    transport_precision=(
                        "FP8 backward input and combine; BF16 final outputs"
                        if getattr(native_backend, "fp8_combine", False)
                        else "mixed transport"
                        if compute_dtype == "blockwise"
                        else compute_dtype
                    ),
                    prime_reference_ms={
                        "forward": 3.9,
                        "backward": 8.2,
                        "checkpointed_layer": 15.9,
                    },
                    prime_precision_matched=False,
                    native_backward_input_transport=(
                        "fp8"
                        if getattr(native_backend, "fp8_transport", False)
                        else "bf16"
                    ),
                    native_backward_combine_transport=(
                        "fp8"
                        if getattr(native_backend, "fp8_combine", False)
                        else "bf16"
                    ),
                    control_scope="same persistent forward; fixed-capacity NCCL backward",
                    native_fused_activation_backward=compare_native,
                    compare_ready=compare_ready,
                    compare_peer_return=compare_peer_return,
                    compare_owned=compare_owned,
                    compare_nccl=compare_nccl,
                    compare_native=compare_native,
                    compare_forward_layouts=compare_forward_layouts,
                    compare_kernel_layouts=compare_kernel_layouts,
                    kernel_layout_scope=(
                        "forward selector affects native and NCCL controls; backward selector affects native persistent backward only"
                        if compare_kernel_layouts
                        else None
                    ),
                    forward_layout_selector=(
                        "call_time_host_environment"
                        if compare_forward_layouts or compare_kernel_layouts
                        else "production_default"
                    ),
                    retained_activations=not recompute,
                    backward_recomputes_w1=recompute,
                    fused_activation_backward="paired_comparison"
                    if compare_fusion
                    else runtime.fuse_activation,
                    ready_gather=bool(peer_backward and peer_backend.ready_gather),
                    backward_transport="paired_native_nccl"
                    if compare_native and compute_dtype == "blockwise"
                    else "paired_native_peer_nccl"
                    if compare_native
                    else "paired_peer_nccl"
                    if compare_nccl
                    else "direct_peer"
                    if peer_backward
                    else "nccl_fixed_capacity",
                    router_in_timing=False,
                    optimizer_in_timing=False,
                    whole_layer_fullgraph=compiled,
                    triton_cudagraphs_disabled_for_forward_layout_comparison=(
                        compare_forward_layouts
                    ),
                    triton_cudagraphs_disabled_for_kernel_layout_comparison=(
                        compare_kernel_layouts
                    ),
                    checkpoint_scope="torch_nonreentrant_2F_plus_B_with_saved_routes",
                    warmup_iterations=5,
                    measured_iterations=15,
                    timing="host_wall_synchronized_maximum_rank",
                    workspace_bytes=workspace,
                    peak_memory_bytes={
                        label: dict(allocated=value[0], reserved=value[1])
                        for label, value in peak_memory.items()
                    },
                    memory_scope="maximum across measured iterations and ranks; all paired backends resident; PyTorch allocator excludes some external allocations; explicit workspace bytes are a separate inventory, not additive",
                    samples_ms=times,
                    median_ms={k: statistics.median(v) for k, v in times.items()},
                    mean_rank_samples_ms=mean_rank_times,
                    per_rank_samples_ms=rank_times,
                    mean_rank_median_ms={
                        k: statistics.median(v) for k, v in mean_rank_times.items()
                    },
                    mean_rank_mean_ms={
                        k: statistics.mean(v) for k, v in mean_rank_times.items()
                    },
                    rank_aggregation="max or arithmetic mean across ranks per iteration, then median across measured iterations",
                )
            ),
            flush=True,
        )
    if profile:
        local_records = []
        profile_cases = (
            (("nccl_fixed_capacity", None), ("native", native_backend))
            if compare_native
            else (
                (
                    "direct_peer" if peer_backward else "nccl_fixed_capacity",
                    runtime._peer_backward,
                ),
            )
        )
        for transport, backend in profile_cases:
            runtime._peer_backward = backend
            for profile_scope in ("forward", "backward"):
                for tensor in (x, scores, w1, w2):
                    tensor.grad = None
                y = (
                    operation(x, ids, scores, w1, w2)
                    if profile_scope == "backward"
                    else None
                )
                dist.barrier()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as trace:
                    if profile_scope == "forward":
                        y = operation(x, ids, scores, w1, w2)
                    else:
                        y.backward(dy)
                    torch.cuda.synchronize()
                trace_path = Path(
                    f"/tmp/ultra-{profile_scope}-{transport}-rank-{rank}.json"  # nosec B108
                )
                trace.export_chrome_trace(str(trace_path))
                raw_events = json.loads(trace_path.read_text())["traceEvents"]
                gpu_timeline = [
                    dict(
                        name=e["name"],
                        start_us=e["ts"],
                        duration_us=e["dur"],
                        stream=e.get("args", {}).get("stream"),
                    )
                    for e in raw_events
                    if e.get("cat") in ("kernel", "gpu_memcpy") and "dur" in e
                ]
                host_sync_events = [
                    dict(name=e["name"], start_us=e["ts"], duration_us=e["dur"])
                    for e in raw_events
                    if "dur" in e
                    and (
                        "Synchronize" in e.get("name", "")
                        or "_local_scalar_dense" in e.get("name", "")
                        or "DtoH" in e.get("name", "")
                    )
                ]
                events = trace.key_averages()

                def summarize(event):
                    return dict(
                        name=event.key,
                        calls=event.count,
                        self_cpu_us=event.self_cpu_time_total,
                        self_device_us=event.self_device_time_total,
                    )

                record = dict(
                    profile=profile_scope,
                    rank=rank,
                    compute_dtype=compute_dtype,
                    whole_layer_fullgraph=compiled,
                    backward_transport=transport,
                    gpu_timeline=gpu_timeline,
                    host_sync_events=host_sync_events,
                    final_device_synchronize_is_explicit=True,
                    device_events=[
                        summarize(e)
                        for e in sorted(
                            events, key=lambda e: e.self_device_time_total, reverse=True
                        )[:25]
                    ],
                    cpu_events=[
                        summarize(e)
                        for e in sorted(
                            events, key=lambda e: e.self_cpu_time_total, reverse=True
                        )[:15]
                    ],
                    warning="Aggregated event service times are not wall latency; overlap and profiler overhead apply",
                )
                local_records.append(record)
                del y
        records = [None] * world
        dist.all_gather_object(records, local_records)
        if rank == 0:
            print(json.dumps({"profiles": records}), flush=True)
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--compute-dtype", choices=("bf16", "mxfp8", "blockwise"), default="bf16"
    )
    parser.add_argument("--compare-fusion", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--peer-backward", action="store_true")
    parser.add_argument("--compare-ready", action="store_true")
    parser.add_argument("--compare-peer-return", action="store_true")
    parser.add_argument("--compare-owned", action="store_true")
    parser.add_argument("--compare-nccl", action="store_true")
    parser.add_argument("--compare-native", action="store_true")
    parser.add_argument("--compare-forward-layouts", action="store_true")
    parser.add_argument("--compare-kernel-layouts", action="store_true")
    args = parser.parse_args()
    if args.compiled and not args.recompute:
        parser.error("Compiled timing requires recomputation")
    if (
        args.compare_native
        and args.compute_dtype != "blockwise"
        and (
            not args.peer_backward
            or args.compare_ready
            or args.compare_peer_return
            or args.compare_fusion
            or args.compare_owned
            or args.profile
            or args.compute_dtype != "bf16"
        )
    ):
        parser.error("Native comparison requires isolated BF16 recomputing peer timing")
    if args.compare_nccl and (
        not args.peer_backward
        or args.compare_ready
        or args.compare_peer_return
        or args.compare_fusion
        or args.compare_owned
        or args.profile
    ):
        parser.error("NCCL comparison requires isolated recomputing peer timing")
    if args.compare_owned and (
        not args.peer_backward
        or args.compare_ready
        or args.compare_peer_return
        or args.compare_fusion
        or args.profile
        or args.compute_dtype != "bf16"
    ):
        parser.error("Owned-block comparison requires isolated BF16 peer timing")
    if args.compare_peer_return and (
        not args.peer_backward
        or args.compare_ready
        or args.compare_fusion
        or args.profile
        or args.compute_dtype != "bf16"
    ):
        parser.error("Peer-return comparison requires isolated BF16 peer timing")
    if args.compare_fusion and args.profile:
        parser.error("Fusion A/B timing and profiling are separate runs")
    if args.peer_backward and (not args.recompute):
        parser.error("Peer backward requires recomputing training")
    if args.compare_ready and (
        not args.peer_backward
        or args.compare_fusion
        or args.profile
        or args.compute_dtype != "bf16"
    ):
        parser.error(
            "Readiness comparison requires BF16 peer timing without fusion comparison"
        )
    run(
        args.profile,
        args.compute_dtype,
        args.compare_fusion,
        args.recompute,
        args.peer_backward,
        args.compare_ready,
        args.compare_peer_return,
        args.compare_owned,
        args.compare_nccl,
        args.compare_native,
        args.compiled,
        args.compare_forward_layouts,
        args.compare_kernel_layouts,
    )
