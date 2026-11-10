# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Multi-rank phase trace and timing decomposition of the SM90 FP8 persistent training step.

Runs the same compiled layer and contract as benchmark_persistent_training.py (native backward,
FP8 transport and combine) at any world size (16 local experts per rank), and records per rank:
host wall per scope, GPU span of the scope, the native kernel's CUDA-event duration and its
offset inside the scope, a torch.profiler kernel breakdown, and (trace iterations) the kernels'
opt-in phase trace with the multi-rank extension (globaltimer stamps comparable across the ranks
of one node, per-peer pull statistics, combine-readiness waits). Everything is gathered to rank 0,
which writes a gzip JSON (--out) and prints an EP_TRACE_JSON summary line.
"""

import argparse
import gzip
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

TRACE_HEADER, TRACE_SLOTS, TRACE_FIELDS, DIST_STRIDE = 64, 64, 16, 64
TRACE_STRIDE = TRACE_HEADER + TRACE_SLOTS * TRACE_FIELDS
ENVS = {
    "backward": (
        "MONO_EP_SM90_FP8_PHASE_TRACE",
        "MONO_EP_SM90_FP8_DIST_TRACE",
        "fp8_mono_ep_backward_sm90",
    ),
    "forward": (
        "MONO_EP_SM90_FP8_FORWARD_PHASE_TRACE",
        "MONO_EP_SM90_FP8_FORWARD_DIST_TRACE",
        "fp8_mono_ep_forward_sm90",
    ),
}


def _median(values):
    return statistics.median(values) if values else None


def _routing(kind, tokens, experts, topk, rank):
    generator = torch.Generator(device="cuda").manual_seed(812 + rank)
    if kind == "balanced":
        return (
            torch.rand(tokens, experts, device="cuda", generator=generator)
            .argsort(-1)[:, :topk]
            .contiguous()
        )
    # Exponentially decaying expert popularity in expert order: low ranks own the hot experts.
    logits = -torch.arange(experts, device="cuda", dtype=torch.float32) / (
        1.5 * experts
    )
    gumbel = -torch.log(
        -torch.log(
            torch.rand(tokens, experts, device="cuda", generator=generator).clamp_min(
                1e-20
            )
        )
    )
    return (logits + gumbel).topk(topk, dim=-1).indices.contiguous()


def _cache_stats():
    import sys

    module = sys.modules.get("mono_ep.weight_cache")
    return None if module is None else module.cache_stats()


def analyze_rank(trace, world, sms):
    """Per-rank summary of one iteration's raw trace (int64 [sms*TRACE_STRIDE + sms*64]).

    Globaltimer stamps stay int64 until a node-wide base is subtracted (float64 would round them).
    """
    cta = trace[: sms * TRACE_STRIDE].view(sms, TRACE_STRIDE)
    ext = trace[sms * TRACE_STRIDE :].view(sms, DIST_STRIDE)
    h = cta[:, :TRACE_HEADER]
    start_gt = h[:, 0]
    end_gt = torch.maximum(h[:, 18], ext[:, 5])
    base = int(start_gt.min())

    def rel(gt):
        return (gt - base).double() / 1e6

    nspc = (h[:, 18] - h[:, 0]).double() / (h[:, 5] - h[:, 1]).double()

    def cyc_ms(c):
        return c.double() * nspc / 1e6

    out = dict(
        kernel_start_gt=base,
        kernel_end_gt=int(end_gt.max()),
        kernel_wall_ms=float(rel(end_gt.max())),
        cta_start_spread_ms=float(rel(start_gt.max())),
        epilogue_end_spread_ms=float(rel(h[:, 18].max()) - rel(h[:, 18].min())),
        sm_clock_ghz=float((1 / nspc).median()),
        tasks=int(h[:, 17].sum()),
    )
    marks = {
        "dispatch_count_done": ext[:, 0],
        "dispatch_grid_sync_done": ext[:, 1],
        "pull_barrier_done": ext[:, 2],
        "pull_start": ext[:, 3],
        "pull_end": ext[:, 4],
        "first_task_claim": ext[:, 48],
        "task_loop_end": ext[:, 8],
        "epilogue_grid_sync_done": ext[:, 9],
        "epilogue_end": h[:, 18],
        "dispatch_cleanup_barrier_enter": ext[:, 6],
        "dispatch_cleanup_barrier_exit": ext[:, 5],
    }
    out["marks_ms_after_kernel_start"] = {
        k: dict(
            min=float(rel(v).min()),
            median=float(rel(v).median()),
            max=float(rel(v).max()),
        )
        for k, v in marks.items()
    }

    def d(a, b):
        return (a - b).double() / 1e6

    epilogue_warps = 8
    per_cta = dict(
        count_and_offsets=d(ext[:, 0], h[:, 0]),
        dispatch_grid_sync=d(ext[:, 1], ext[:, 0]),
        pull_nvlink_barrier=d(ext[:, 2], ext[:, 1]),
        recv_count_and_metadata=d(ext[:, 3], ext[:, 2]),
        pull_loop=d(ext[:, 4], ext[:, 3]),
        first_claim_after_start=d(ext[:, 48], h[:, 0]),
        task_loop_end_after_start=d(ext[:, 8], h[:, 0]),
        epilogue_grid_sync_wait=d(ext[:, 9], ext[:, 8]),
        combine_tail_after_grid_sync=d(h[:, 18], ext[:, 9]),
        combine_ready_wait_mean_warp=cyc_ms(ext[:, 10]) / epilogue_warps,
        combine_ready_first_token_wait_max_warp=cyc_ms(ext[:, 11]),
        dispatch_cleanup_barrier=d(ext[:, 5], ext[:, 6]),
        epilogue_end_after_start=d(h[:, 18], h[:, 0]),
        sched_wait=cyc_ms(h[:, 6]),
        mainloop_total=cyc_ms(h[:, 9:13].sum(1)),
        epilogue_total=cyc_ms(h[:, 13:17].sum(1)),
        mainloop_full_wait=cyc_ms(h[:, 7] + h[:, 8]),
        dispatch_ring_wait_mean_warp=cyc_ms(h[:, [41, 45, 49, 53]].double().mean(1)),
        dispatch_pull_load_wait_mean_warp=cyc_ms(
            h[:, [42, 46, 50, 54]].double().mean(1)
        ),
    )
    out["phases_median_cta_ms"] = {k: float(v.median()) for k, v in per_cta.items()}
    out["phases_max_cta_ms"] = {k: float(v.max()) for k, v in per_cta.items()}
    pull_start = ext[:, 3].min()
    rows = ext[:, 24 : 24 + world]
    out["per_peer"] = []
    for p in range(world):
        mask = rows[:, p] > 0
        entry = dict(
            peer=p,
            rows=int(rows[:, p].sum()),
            pull_wait_ms_sum_over_warps=float(cyc_ms(ext[:, 16 + p]).sum()),
            pull_wait_us_per_row=float(
                cyc_ms(ext[:, 16 + p]).sum() * 1e3 / max(int(rows[:, p].sum()), 1)
            ),
        )
        if mask.any():
            # Stored as ~globaltimer through atomicMax, i.e. -(gt + 1) as int64.
            first = (-ext[mask, 32 + p] - 1).min()
            entry["first_arrival_ms_after_pull_start"] = float(
                (first - pull_start).double() / 1e6
            )
            entry["last_arrival_ms_after_pull_start"] = float(
                (ext[mask, 40 + p].max() - pull_start).double() / 1e6
            )
        out["per_peer"].append(entry)
    return out


def run(args):
    import deep_gemm
    from mono_ep.persistent_backward import NativePersistentBackward
    from mono_ep.runtime import PersistentMoE

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device_name = torch.cuda.get_device_name()
    if args.expect_gpu and args.expect_gpu not in device_name:
        raise RuntimeError(f"Expected {args.expect_gpu}, got {device_name}")
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    tokens, hidden, inner, topk = 16384, 2048, 768, 8
    experts = 16 * world
    torch.manual_seed(812 + rank)
    x = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    ids = _routing(args.routing, tokens, experts, topk, rank)
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
        compute_dtype="blockwise",
        fuse_activation=False,
        retain_activations=False,
    )
    backend = NativePersistentBackward(runtime)
    runtime._peer_backward = backend
    operation = torch.compile(runtime, fullgraph=True)
    sms = deep_gemm.get_num_sms()
    trace_numel = sms * TRACE_STRIDE + sms * DIST_STRIDE
    trace_buffers = {
        d: torch.zeros(trace_numel, device="cuda", dtype=torch.int64) for d in ENVS
    }

    variants = {}
    for spec in args.variants.split(","):
        name, _, env = spec.partition(":")
        variants[name] = dict(
            item.split("=", 1) for item in filter(None, env.split("+"))
        )
    managed = {k for v in variants.values() for k in v} | {
        e for d in ENVS.values() for e in d[:2]
    }

    def set_env(values, tracing):
        for key in managed:
            os.environ.pop(key, None)
        os.environ.update(values)
        if tracing:
            for direction, (ptr_env, dist_env, _) in ENVS.items():
                os.environ[ptr_env] = str(trace_buffers[direction].data_ptr())
                os.environ[dist_env] = "1"

    state = dict(events=[], traces=[], host=[])
    import mono_ep.runtime as runtime_module

    def stamp_entry(function):
        def stamped(*a, **k):
            state["host"].append(("entry", time.perf_counter(), None))
            return function(*a, **k)

        return stamped

    runtime_module._forward_impl = stamp_entry(runtime_module._forward_impl)
    backend.backward = stamp_entry(backend.backward)
    originals = {d: getattr(deep_gemm._C, e[2]) for d, e in ENVS.items()}

    def wrap(direction):
        original = originals[direction]

        def wrapped(*call_args, **call_kwargs):
            tracing = ENVS[direction][0] in os.environ
            if tracing:
                trace_buffers[direction].zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            host_start = time.perf_counter()
            result = original(*call_args, **call_kwargs)
            state["host"].append(("native", host_start, time.perf_counter()))
            end.record()
            state["events"].append((direction, start, end))
            if tracing:
                state["traces"].append((direction, trace_buffers[direction].clone()))
            return result

        return wrapped

    for direction, entry in ENVS.items():
        setattr(deep_gemm._C, entry[2], wrap(direction))

    scopes = ("forward", "backward", "checkpointed")
    results = {
        name: {
            scope: dict(
                host_ms=[],
                host_to_first_op_ms=[],
                host_op_prep_ms=[],
                host_native_call_ms=[],
                gpu_span_ms=[],
                native_ms=[],
                native_offset_ms=[],
                post_native_ms=[],
            )
            for scope in scopes
        }
        for name in variants
    }
    traces_out = {}
    gathered = torch.empty(world, device="cuda", dtype=torch.float64)

    def gather(value):
        dist.all_gather_into_tensor(
            gathered, torch.tensor([value], device="cuda", dtype=torch.float64)
        )
        return gathered.tolist()

    def bump_versions():
        # Stands in for the optimizer step: new weight versions, values unchanged.
        if args.bump_versions:
            for tensor in (w1, w2):
                torch.autograd.graph.increment_version(tensor)

    def step(scope, tracing):
        for tensor in (x, scores, w1, w2):
            tensor.grad = None
        bump_versions()
        y = operation(x, ids, scores, w1, w2) if scope == "backward" else None
        dist.barrier()
        torch.cuda.synchronize()
        state["events"].clear()
        state["traces"].clear()
        state["host"].clear()
        span0, span1 = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start = time.perf_counter()
        span0.record()
        if scope == "forward":
            y = operation(x, ids, scores, w1, w2)
        elif scope == "checkpointed":
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
            y.backward(dy)
        else:
            y.backward(dy)
        span1.record()
        torch.cuda.synchronize()
        host = (time.perf_counter() - start) * 1000
        native = [
            (s, e)
            for d, s, e in state["events"]
            if d == scope or scope == "checkpointed"
        ]
        assert len(native) == (3 if scope == "checkpointed" else 1), (
            scope,
            len(native),
        )
        entries = [t for kind, t, _ in state["host"] if kind == "entry"]
        calls = [(t0, t1) for kind, t0, t1 in state["host"] if kind == "native"]
        record = dict(
            host_ms=host,
            host_to_first_op_ms=(entries[0] - start) * 1000,
            host_op_prep_ms=sum(c[0] - e for e, c in zip(entries, calls, strict=True))
            * 1000,
            host_native_call_ms=sum(t1 - t0 for t0, t1 in calls) * 1000,
            gpu_span_ms=span0.elapsed_time(span1),
            native_ms=sum(s.elapsed_time(e) for s, e in native),
            native_offset_ms=span0.elapsed_time(native[0][0]),
            post_native_ms=native[-1][1].elapsed_time(span1),
        )
        trace = [t for d, t in state["traces"] if d == scope]
        state["last_outputs"] = (
            dict(y=y.detach().clone())
            if scope == "forward"
            else dict(
                dx=x.grad.clone(),
                dscore=scores.grad.clone(),
                dw1=w1.grad.clone(),
                dw2=w2.grad.clone(),
            )
        )
        del y
        return record, (trace[0] if tracing else None)

    parity = {}
    captured = {}

    def check_parity(name, scope):
        ref_name = names[0]
        outputs = captured[(name, scope)]
        ref = captured[(ref_name, scope)]
        entry = {}
        for label, a in outputs.items():
            b = ref[label]
            diff = (a.float() - b.float()).abs()
            entry[label] = dict(
                bitwise_equal=gather(float(torch.equal(a, b))),
                max_abs_diff=gather(float(diff.max())),
                rel_norm=gather(float(diff.norm() / b.float().norm().clamp_min(1e-12))),
            )
        parity[f"{name}_vs_{ref_name}/{scope}"] = entry

    forms_check = None
    if args.check_weight_forms:
        from mono_ep.sm90_weight_forms import (
            cached_forms,
            quantize_all_forms,
            quantize_form,
        )

        fused = quantize_all_forms(w1.detach(), w2.detach())
        bump_versions()
        cached = dict(
            zip(
                fused,
                cached_forms(w1.detach(), w2.detach(), list(fused), "1"),
                strict=True,
            )
        )
        forms_check = {}
        for label, (values, scales) in fused.items():
            reference = quantize_form(w1.detach(), w2.detach(), label)
            forms_check[label] = dict(
                fused_values_bitwise=gather(
                    float(
                        torch.equal(
                            values.view(torch.uint8), reference[0].view(torch.uint8)
                        )
                    )
                ),
                fused_scales_bitwise=gather(float(torch.equal(scales, reference[1]))),
                cached_values_bitwise=gather(
                    float(
                        torch.equal(
                            cached[label][0].view(torch.uint8),
                            reference[0].view(torch.uint8),
                        )
                    )
                ),
                cached_scales_bitwise=gather(
                    float(torch.equal(cached[label][1], reference[1]))
                ),
                shapes=[list(values.shape), list(reference[0].shape)],
            )
        del fused, cached

    quant_check = None
    if args.check_fused_weight_quant:
        from deep_gemm.mega import _interleave_weights
        from mono_ep.block_weight_quantization import quantize_weights_128x128
        from mono_ep.quantization import quantize_fp8

        pairs = (
            (
                quantize_weights_128x128(w1.detach(), interleave_gate_up=True),
                quantize_fp8(
                    _interleave_weights(w1.detach()), block_rows=128, block_cols=128
                ),
            ),
            (
                quantize_weights_128x128(w2.detach()),
                quantize_fp8(w2.detach(), block_rows=128, block_cols=128),
            ),
        )
        quant_check = {
            label: dict(
                values_bitwise=gather(
                    float(torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8)))
                ),
                scales_bitwise=gather(float(torch.equal(a[1], b[1]))),
                shapes=[
                    list(a[0].shape),
                    list(b[0].shape),
                    list(a[1].shape),
                    list(b[1].shape),
                ],
            )
            for label, (a, b) in zip(("w1_interleaved", "w2"), pairs, strict=True)
        }

    names = list(variants)
    for iteration in range(args.warmup + args.iters):
        order = names if iteration % 2 == 0 else names[::-1]
        for name in order:
            set_env(variants[name], False)
            for scope in scopes if iteration % 2 == 0 else scopes[::-1]:
                record, _ = step(scope, False)
                outputs = state.pop("last_outputs")
                if iteration == args.warmup:
                    captured[(name, scope)] = outputs
                del outputs
                values = {k: gather(v) for k, v in record.items()}
                if iteration >= args.warmup:
                    for key, per_rank in values.items():
                        results[name][scope][key].append(per_rank)
        if iteration == args.warmup:
            for name in names[1:]:
                for scope in scopes:
                    check_parity(name, scope)
            captured.clear()
    # Trace iterations (separate: the trace adds overhead)
    traced = (
        names
        if args.trace_variants == "all"
        else names[:1]
        if args.trace_variants == "first"
        else args.trace_variants.split(",")
    )
    for name in [] if args.skip_phase_trace else traced:
        set_env(variants[name], True)
        for scope in ("forward", "backward"):
            per_iter = []
            for _ in range(args.trace_warmup):
                step(scope, True)
            for _ in range(args.trace_iters):
                record, trace = step(scope, True)
                all_traces = torch.empty(
                    world, trace.numel(), device="cuda", dtype=torch.int64
                )
                dist.all_gather_into_tensor(all_traces, trace)
                per_iter.append(
                    (dict((k, gather(v)) for k, v in record.items()), all_traces.cpu())
                )
            traces_out[f"{name}/{scope}"] = per_iter
        set_env({}, False)
    counts = torch.empty(
        world, backend.counts.numel(), device="cuda", dtype=torch.int32
    )
    dist.all_gather_into_tensor(counts, backend.counts.contiguous())

    # Profiler breakdown of one step per scope for each traced variant (key 'variant/scope')
    profile = {}
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CUDA]
        if args.profile_cpu:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        profile_scopes = ("forward", "forward_again", "backward")
        for profile_name in traced:
            set_env(variants[profile_name], False)
            for scope in profile_scopes:
                for tensor in (x, scores, w1, w2):
                    tensor.grad = None
                if scope != "forward_again":
                    bump_versions()
                y = operation(x, ids, scores, w1, w2) if scope != "forward" else None
                dist.barrier()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=activities) as prof:
                    with torch.profiler.record_function("mono_ep_scope"):
                        if scope.startswith("forward"):
                            y = operation(x, ids, scores, w1, w2)
                        else:
                            y.backward(dy)
                    torch.cuda.synchronize()
                path = f"/tmp/ep-trace-{profile_name}-{scope}-rank-{rank}.json"  # nosec B108
                prof.export_chrome_trace(path)
                with open(path) as f:
                    events = json.load(f)["traceEvents"]
                kernels = [
                    dict(name=e["name"][:120], start_us=e["ts"], duration_us=e["dur"])
                    for e in events
                    if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                    and "dur" in e
                ]
                kernels.sort(key=lambda k: k["start_us"])
                if args.profile_cpu:
                    host = [
                        dict(
                            name=e["name"][:120],
                            cat=e.get("cat"),
                            start_us=e["ts"],
                            duration_us=e["dur"],
                        )
                        for e in events
                        if e.get("cat") in ("cuda_runtime", "cuda_driver")
                        or (
                            e.get("cat") == "user_annotation"
                            and e.get("name") == "mono_ep_scope"
                        )
                    ]
                    host.sort(key=lambda k: k["start_us"])
                    kernels = dict(kernels=kernels, host=host)
                objs = [None] * world
                dist.all_gather_object(objs, kernels)
                profile[f"{profile_name}/{scope}"] = objs
                del y
        set_env({}, False)

    if rank == 0:
        summary = dict(
            benchmark="ep_trace_persistent_training",
            device=device_name,
            torch_version=torch.__version__,
            world_size=world,
            routing=args.routing,
            sms=sms,
            tokens_per_gpu=tokens,
            experts=experts,
            topk=topk,
            warmup=args.warmup,
            iters=args.iters,
            trace_iters=args.trace_iters,
            trace_warmup=args.trace_warmup,
            timeout_handler=os.getenv("MONO_EP_SM90_FP8_TRAP_TIMEOUTS"),
            transport="fp8 input, fp8 combine",
            routed_rows_per_rank=counts.sum(1).tolist(),
            routed_rows_per_rank_expert=counts.tolist(),
            row_imbalance_max_over_mean=float(
                counts.sum(1).max() / counts.sum(1).double().mean()
            ),
            variants=variants,
            parity_first_measured_iteration=parity,
            fused_weight_quant_check=quant_check,
            weight_forms_check=forms_check,
            bump_versions=args.bump_versions,
            weight_cache_stats=_cache_stats(),
            timing={},
        )
        for name in names:
            for scope in scopes:
                r = results[name][scope]
                entry = dict(samples=len(r["host_ms"]))
                for key, rows in r.items():
                    entry[f"{key}_max_rank_median"] = _median([max(v) for v in rows])
                    entry[f"{key}_mean_rank_median"] = _median(
                        [statistics.mean(v) for v in rows]
                    )
                    entry[f"{key}_per_rank_median"] = [
                        _median([v[i] for v in rows]) for i in range(world)
                    ]
                entry["raw"] = r
                summary["timing"][f"{name}/{scope}"] = entry
        analysis = {}
        raw_traces = {}
        for key, per_iter in traces_out.items():
            walls = []
            analyses = []
            for record, all_traces in per_iter:
                ranks = [analyze_rank(all_traces[r], world, sms) for r in range(world)]

                # Globaltimers of different GPUs are not synchronized (offsets of tens of ms were
                # observed), so ranks are aligned at the exit of the pre-pull NVLink barrier, which
                # every rank leaves within NVLink latency of the last arrival.
                def mark(a, name, stat="max"):
                    return a["marks_ms_after_kernel_start"][name][stat]

                anchor = [mark(a, "pull_barrier_done") for a in ranks]
                ends = [
                    a["kernel_wall_ms"] - p for a, p in zip(ranks, anchor, strict=True)
                ]
                cross = dict(
                    alignment="pull_barrier_done (max over CTAs) per rank = 0",
                    kernel_start_skew_ms=max(anchor) - min(anchor),
                    kernel_end_skew_ms=max(ends) - min(ends),
                    union_wall_ms=max(ends) + max(anchor),
                    per_rank_start_ms=[-p for p in anchor],
                    per_rank_end_ms=ends,
                    per_rank_task_loop_end_max_ms=[
                        mark(a, "task_loop_end") - p
                        for a, p in zip(ranks, anchor, strict=True)
                    ],
                    per_rank_epilogue_grid_sync_done_max_ms=[
                        mark(a, "epilogue_grid_sync_done") - p
                        for a, p in zip(ranks, anchor, strict=True)
                    ],
                    raw_globaltimer_start_offsets_ms=[
                        (
                            a["kernel_start_gt"]
                            - min(r["kernel_start_gt"] for r in ranks)
                        )
                        / 1e6
                        for a in ranks
                    ],
                )
                walls.append(max(a["kernel_wall_ms"] for a in ranks))
                analyses.append(dict(record=record, cross_rank=cross, per_rank=ranks))
            mid = sorted(range(len(walls)), key=walls.__getitem__)[len(walls) // 2]
            analysis[key] = dict(
                iterations=len(analyses),
                median_iteration=mid,
                max_rank_native_wall_per_iteration=walls,
                median=analyses[mid],
                all=analyses,
            )
            raw_traces[key] = per_iter[mid][1].tolist()
        summary["trace_analysis"] = {
            k: dict(
                iterations=v["iterations"],
                median_iteration=v["median_iteration"],
                walls=v["max_rank_native_wall_per_iteration"],
                cross_rank=v["median"]["cross_rank"],
                record=v["median"]["record"],
                per_rank=v["median"]["per_rank"],
            )
            for k, v in analysis.items()
        }
        full = dict(
            summary,
            trace_analysis_all=analysis,
            raw_traces_median_iteration=raw_traces,
            profile=profile,
        )
        if args.out:
            with gzip.open(args.out, "wt") as f:
                json.dump(full, f)
        brief = {
            k: summary[k]
            for k in (
                "device",
                "world_size",
                "routing",
                "routed_rows_per_rank",
                "row_imbalance_max_over_mean",
                "parity_first_measured_iteration",
                "fused_weight_quant_check",
                "weight_forms_check",
                "bump_versions",
                "weight_cache_stats",
            )
        }
        brief["timing"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "raw"}
            for k, v in summary["timing"].items()
        }
        print("EP_TRACE_JSON " + json.dumps(brief), flush=True)
    for direction, entry in ENVS.items():
        setattr(deep_gemm._C, entry[2], originals[direction])
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--variants", default="default:")
    parser.add_argument(
        "--trace-variants", default="first", help="first, all or a comma list"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--trace-iters", type=int, default=5)
    parser.add_argument("--trace-warmup", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--check-fused-weight-quant", action="store_true")
    parser.add_argument("--check-weight-forms", action="store_true")
    parser.add_argument(
        "--bump-versions",
        action="store_true",
        help="increment the weight version counters before every step (optimizer stand-in)",
    )
    parser.add_argument("--profile-cpu", action="store_true")
    parser.add_argument("--skip-phase-trace", action="store_true")
    parser.add_argument("--expect-gpu", default="H100")
    parser.add_argument("--out", default="")
    run(parser.parse_args())
