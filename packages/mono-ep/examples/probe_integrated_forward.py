# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-rank SM90 FP8 persistent forward A/B with parity against a reference saved by another source tree."""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

TRACE_HEADER, TRACE_TASKS, TRACE_FIELDS = 64, 64, 16
TRACE_STRIDE = TRACE_HEADER + TRACE_TASKS * TRACE_FIELDS
TRACE_ENV = "MONO_EP_SM90_FP8_FORWARD_PHASE_TRACE"


def _median(values):
    return statistics.median(values) if values else None


def cta_tail(h, ns_per_cycle):
    """Per-CTA task-loop end (clock64 mark 4 mapped onto each CTA's globaltimer start), ms after kernel start."""
    start = h[:, 0].min()
    end = (h[:, 0] - start + (h[:, 4] - h[:, 1]) * ns_per_cycle) / 1e6
    kernel_end = float((h[:, 18].max() - start) / 1e6)
    q = end.sort().values
    median, mean, last = float(q.median()), float(q.mean()), float(q[-1])
    return dict(
        first=float(q[0]),
        p10=float(q[int(0.1 * (len(q) - 1))]),
        median=median,
        mean=mean,
        p90=float(q[int(0.9 * (len(q) - 1))]),
        last=last,
        kernel_end=kernel_end,
        tail_median_to_last=last - median,
        tail_mean_to_last=last - mean,
        spread_first_to_last=last - float(q[0]),
    )


def summarize_trace(trace):
    """Convert one iteration's raw per-CTA trace into milliseconds."""
    t = trace.double()
    h = t[:, :TRACE_HEADER]
    kernel_ns = h[:, 18] - h[:, 0]
    kernel_cycles = h[:, 5] - h[:, 1]
    ns_per_cycle = kernel_ns / kernel_cycles
    ms = lambda cycles: cycles * ns_per_cycle / 1e6  # noqa: E731

    mainloop = h[:, 9] + h[:, 10]
    full_wait = h[:, 7] + h[:, 8]
    epilogue = h[:, 13] + h[:, 14]
    per_cta = {
        "kernel_wall": kernel_ns / 1e6,
        "prologue_before_task_loop": ms(h[:, 3] - h[:, 1]),
        "task_loop": ms(h[:, 4] - h[:, 3]),
        "tail_grid_sync_after_task_loop": ms(h[:, 35] - h[:, 4]),
        "tail_combine_reduce": ms(h[:, 36]),
        "tail_combine_peer_wait_mean_warp": ms(h[:, 37] / 8),
        "sched_wait": ms(h[:, 6]),
        "mainloop_total": ms(mainloop),
        "mainloop_full_wait_first_kblock": ms(h[:, 7]),
        "mainloop_full_wait_other_kblocks": ms(h[:, 8]),
        "mainloop_excl_full_waits": ms(mainloop - full_wait),
        "mainloop_l1_z": ms(h[:, 9]),
        "mainloop_l2_y": ms(h[:, 10]),
        "full_wait_first_l1": ms(h[:, 11]),
        "full_wait_first_l2": ms(h[:, 12]),
        "epilogue_total": ms(epilogue),
        "epilogue_l1": ms(h[:, 13]),
        "epilogue_l2": ms(h[:, 14]),
        "epi_accum_to_smem_sync": ms(h[:, 20]),
        "epi_l1_l2_empty_spin": ms(h[:, 21]),
        "epi_l1_swiglu_h_store": ms(h[:, 22]),
        "epi_l1_odd_even_partner_spin": ms(h[:, 23]),
        "epi_l1_odd_h_fp8_row_quant": ms(h[:, 24]),
        "epi_l1_tail_arrive": ms(h[:, 25]),
        "epi_l2_combine_scatter": ms(h[:, 26]),
        "wg1_mainloop_total": ms(h[:, 28]),
        "wg1_full_wait": ms(h[:, 29]),
        "a_loader_l1_full_spin": ms(h[:, 30]),
        "a_loader_l2_kblock_dependency_wait": ms(h[:, 31]),
        "a_loader_empty_barrier_wait": ms(h[:, 32]),
        "b_loader_empty_barrier_wait": ms(h[:, 34]),
        "dispatch_pull_loop_start_after_kernel_start": ms(h[:, 2] - h[:, 1]),
        "dispatch_pull_loop_end_after_kernel_start_max_warp": ms(
            h[:, 56:60].max(1).values - h[:, 1]
        ),
        "dispatch_cleanup_end_after_kernel_start": ms(h[:, 62] - h[:, 1]),
        "dispatch_busy_mean_warp": ms(h[:, [40, 44, 48, 52]].mean(1)),
        "dispatch_ring_wait_mean_warp": ms(h[:, [41, 45, 49, 53]].mean(1)),
        "dispatch_pull_wait_mean_warp": ms(h[:, [42, 46, 50, 54]].mean(1)),
        "dispatch_sf_meta_mean_warp": ms(h[:, [43, 47, 51, 55]].mean(1)),
        "fraction_no_math_wg_in_mainloop": 1 - (mainloop * ns_per_cycle / kernel_ns),
    }
    counts = {
        "tasks_per_cta": h[:, 17],
        "l1_tasks_per_cta": h[:, 15],
        "l2_tasks_per_cta": h[:, 16],
        "l1_odd_tasks_per_cta": h[:, 27],
        "dispatch_rows_per_cta": h[:, 60],
    }
    summary = {
        "cta_tail_ms": cta_tail(h, ns_per_cycle),
        "median_over_ctas_ms": {k: float(v.median()) for k, v in per_cta.items()},
        "mean_over_ctas_ms": {k: float(v.mean()) for k, v in per_cta.items()},
        "max_over_ctas_ms": {k: float(v.max()) for k, v in per_cta.items()},
        "counts_median": {k: float(v.median()) for k, v in counts.items()},
        "counts_sum": {k: float(v.sum()) for k, v in counts.items()},
        "sm_clock_ghz_median": float((1 / ns_per_cycle).median()),
    }
    m = t[:, TRACE_HEADER:].reshape(t.shape[0], TRACE_TASKS, TRACE_FIELDS)
    valid = m[:, :, 0] > 0
    scale = ns_per_cycle[:, None].expand(-1, TRACE_TASKS) / 1e3
    kind = (m[:, :, 0].long() - 1) & 0xFF
    n_block = m[:, :, 0].long() >> 8
    names = {
        "duration_claim_to_end": m[:, :, 2] - m[:, :, 1],
        "mainloop": m[:, :, 4],
        "epilogue": m[:, :, 5],
        "sched_wait": m[:, :, 6],
        "full_barrier_wait_all": m[:, :, 7],
        "full_barrier_wait_first": m[:, :, 14],
        "accum_to_smem_sync": m[:, :, 8],
        "construct1_swiglu_or_scatter": m[:, :, 9],
        "construct2_partner_spin": m[:, :, 10],
        "construct3_h_quant": m[:, :, 11],
        "l2_empty_spin": m[:, :, 12],
    }
    per_task = {}
    for label, sel in (
        ("l1_even", valid & (kind == 0) & (n_block % 2 == 0)),
        ("l1_odd", valid & (kind == 0) & (n_block % 2 == 1)),
        ("l2", valid & (kind == 1)),
    ):
        medians = {}
        if sel.any():
            medians = {k: float((v * scale)[sel].median()) for k, v in names.items()}
        per_task[label] = {"count": int(sel.sum()), **medians}
    summary["per_task_median_us_first64_tasks_per_cta"] = per_task
    return summary


def run(args):
    import deep_gemm
    from mono_ep.runtime import PersistentMoE, _forward_impl

    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name()
    if "H100" not in device_name:
        raise RuntimeError(f"Expected H100, got {device_name}")
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    assert dist.get_world_size() == 1
    torch.manual_seed(881)
    tokens, hidden, inner, experts, topk = 16384, 2048, 768, 16, 8
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inner,
        compute_dtype="blockwise",
        retain_activations=False,
        backward_backend="native",
    )
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk].contiguous()
    if args.routing == "skewed":
        ids[:] = torch.arange(topk, device="cuda")
    scores = torch.rand(tokens, topk, device="cuda").softmax(-1)
    w1 = torch.randn(experts, 2 * inner, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
    counts = torch.bincount(ids.flatten(), minlength=experts)
    padded_rows = int(((counts + 127) // 128 * 128).sum())
    original = deep_gemm._C.fp8_mono_ep_forward_sm90
    trace = torch.zeros(
        deep_gemm.get_num_sms() * TRACE_STRIDE, device="cuda", dtype=torch.int64
    )
    variants = {}
    for spec in args.variants.split(","):
        name, _, env = spec.partition(":")
        values = {}
        for item in filter(None, env.split("+")):
            key, _, value = item.partition("=")
            values[key] = value
        variants[name] = values
    managed = {TRACE_ENV} | {k for v in variants.values() for k in v}

    def set_env(values):
        for key in managed:
            os.environ.pop(key, None)
        for key, value in values.items():
            os.environ[key] = str(trace.data_ptr()) if value == "TRACE" else value

    intervals, traces = [], []

    def compare_ref(name, y):
        if not args.ref_path:
            return None
        if not os.path.exists(args.ref_path):
            torch.save(dict(tag=args.ref_tag, variant=name, y=y.cpu()), args.ref_path)
            return dict(saved_reference=True, ref_tag=args.ref_tag)
        ref = torch.load(args.ref_path, weights_only=True)
        b = ref["y"].to(y.device)
        diff = (y.float() - b.float()).abs()
        return dict(
            ref_tag=ref["tag"],
            ref_variant=ref["variant"],
            bitwise_equal=bool(torch.equal(y, b)),
            max_abs_diff=float(diff.max()),
            rel_norm=float(diff.norm() / b.float().norm().clamp_min(1e-12)),
        )

    ring_tokens = []

    def timed_native(*call_args, **call_kwargs):
        tracing = TRACE_ENV in os.environ
        ring_tokens.append(call_args[8])
        if tracing:
            trace.zero_()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        result = original(*call_args, **call_kwargs)
        end.record()
        intervals.append((start, end))
        if tracing:
            traces.append(trace.view(-1, TRACE_STRIDE).clone())
        return result

    def forward():
        return _forward_impl(x, ids, scores, w1, w2, runtime)[0]

    results = {
        name: dict(samples_ms=[], call_samples_ms=[], env=values)
        for name, values in variants.items()
    }
    call_intervals = []

    def bump_versions():
        # PROBE_BUMP_VERSIONS=1 stands in for an optimizer step (new weight versions, same values),
        # so a weight cache misses on every call.
        if os.environ.get("PROBE_BUMP_VERSIONS") == "1":
            for tensor in (w1, w2):
                torch.autograd.graph.increment_version(tensor)

    def timed_call(fn):
        bump_versions()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        out = fn()
        end.record()
        call_intervals.append((start, end))
        return out

    reference = None
    deep_gemm._C.fp8_mono_ep_forward_sm90 = timed_native
    try:
        for round_idx in range(args.rounds):
            for name, values in variants.items():
                set_env(values)
                intervals.clear()
                for _ in range(args.warmup):
                    y = timed_call(forward)
                torch.cuda.synchronize()
                if round_idx == 0:
                    y = y.clone()
                    repeat = forward().clone()
                    if reference is None:
                        reference = (name, y)
                    diff = (y.float() - reference[1].float()).abs()
                    results[name]["parity_vs_" + reference[0]] = dict(
                        bitwise_equal=bool(torch.equal(y, reference[1])),
                        max_abs_diff=float(diff.max()),
                        rel_norm=float(
                            diff.norm() / reference[1].float().norm().clamp_min(1e-12)
                        ),
                        finite=bool(torch.isfinite(y.float()).all()),
                        self_repeat_bitwise_equal=bool(torch.equal(y, repeat)),
                    )
                    results[name]["parity_vs_ref"] = compare_ref(name, y)
                intervals.clear()
                traces.clear()
                call_intervals.clear()
                for _ in range(args.iters):
                    timed_call(forward)
                torch.cuda.synchronize()
                results[name]["call_samples_ms"].extend(
                    a.elapsed_time(b) for a, b in call_intervals
                )
                results[name]["samples_ms"].extend(
                    a.elapsed_time(b) for a, b in intervals
                )
                if traces and round_idx == 0:
                    summaries = [summarize_trace(tr.cpu()) for tr in traces]
                    results[name]["trace_iterations"] = len(summaries)
                    mid = sorted(
                        range(len(summaries)),
                        key=lambda i: summaries[i]["median_over_ctas_ms"][
                            "kernel_wall"
                        ],
                    )[len(summaries) // 2]
                    results[name]["trace_summary_median_iteration"] = summaries[mid]
                    results[name]["trace_kernel_wall_per_iteration"] = [
                        s["median_over_ctas_ms"]["kernel_wall"] for s in summaries
                    ]
                    results[name]["trace_raw_header_median_iteration"] = (
                        traces[mid][:, :TRACE_HEADER].cpu().tolist()
                    )
    finally:
        deep_gemm._C.fp8_mono_ep_forward_sm90 = original
        set_env({})
    for value in results.values():
        value["median_ms"] = _median(value["samples_ms"])
        value["samples"] = len(value["samples_ms"])
        value["call_median_ms"] = _median(value["call_samples_ms"])
    print(
        "DIAG_JSON "
        + json.dumps(
            dict(
                benchmark="diag_single_rank_persistent_fp8_forward",
                device=device_name,
                routing=args.routing,
                tokens=tokens,
                hidden=hidden,
                intermediate=inner,
                experts=experts,
                topk=topk,
                routed_rows=int(counts.sum()),
                padded_rows=padded_rows,
                ring_tokens=int(ring_tokens[0]) if ring_tokens else None,
                ring_wraps=padded_rows > int(ring_tokens[0]) if ring_tokens else None,
                timeout_handler=os.getenv("MONO_EP_SM90_FP8_TRAP_TIMEOUTS"),
                warmup_per_round=args.warmup,
                iters_per_round=args.iters,
                rounds=args.rounds,
                interleaving="variants alternate within each round, same process and GPU",
                results=results,
            )
        ),
        flush=True,
    )
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--variants", default="baseline:,trace:" + TRACE_ENV + "=TRACE")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--ref-path", default="")
    parser.add_argument("--ref-tag", default="")
    run(parser.parse_args())
