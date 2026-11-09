# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""A/B timing plus bitwise dZ-cache / packed h,dZ comparison of the single-rank SM90 backward."""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

TRACE_HEADER, TRACE_MBLOCKS, TRACE_FIELDS = 64, 64, 16
TRACE_STRIDE = TRACE_HEADER + TRACE_MBLOCKS * TRACE_FIELDS
TRACE_ENV = "MONO_EP_SM90_FP8_PHASE_TRACE"
POOLS = {
    8: "dz_row_cache_fp8",
    9: "dz_row_cache_scales",
    12: "h_col_packed_fp8",
    13: "dz_col_packed_fp8",
    16: "h_col_scales",
    17: "dz_col_scales",
}


def _median(values):
    return statistics.median(values) if values else None


def summarize_trace(trace):
    """Convert one iteration's raw per-CTA trace into milliseconds."""
    t = trace.double()
    h = t[:, :TRACE_HEADER]
    kernel_ns = h[:, 18] - h[:, 0]
    kernel_cycles = h[:, 5] - h[:, 1]
    ns_per_cycle = kernel_ns / kernel_cycles
    ms = lambda cycles: cycles * ns_per_cycle / 1e6  # noqa: E731

    mainloop = h[:, 9:13].sum(1)
    full_wait = h[:, 7] + h[:, 8]
    epilogue = h[:, 13:17].sum(1)
    per_cta = {
        "kernel_wall": kernel_ns / 1e6,
        "prologue_before_task_loop": ms(h[:, 3] - h[:, 1]),
        "task_loop": ms(h[:, 4] - h[:, 3]),
        "combine_tail_after_task_loop": ms(h[:, 5] - h[:, 4]),
        "sched_wait": ms(h[:, 6]),
        "mainloop_total": ms(mainloop),
        "mainloop_full_wait_first_kblock": ms(h[:, 7]),
        "mainloop_full_wait_other_kblocks": ms(h[:, 8]),
        "mainloop_excl_full_waits": ms(mainloop - full_wait),
        "mainloop_dh": ms(h[:, 9]),
        "mainloop_recompute_low": ms(h[:, 10]),
        "mainloop_recompute_high": ms(h[:, 11]),
        "mainloop_dx": ms(h[:, 12]),
        "epilogue_total": ms(epilogue),
        "epilogue_dh": ms(h[:, 13]),
        "epilogue_recompute_low": ms(h[:, 14]),
        "epilogue_recompute_high": ms(h[:, 15]),
        "epilogue_dx": ms(h[:, 16]),
        "epi_accum_to_smem_sync": ms(h[:, 20]),
        "epi_dh0_l2_empty_spin": ms(h[:, 21]),
        "epi_dh0_pack_cached_xdy_columns": ms(h[:, 22]),
        "epi_recompute_swiglu_elementwise": ms(h[:, 25]),
        "epi_recompute_h_dz_column_pack": ms(h[:, 26]),
        "epi_recompute_dz_row_cache_quant": ms(h[:, 27]),
        "epi_recompute_tail_arrive_dscore": ms(h[:, 34]),
        "epi_dx_fp8_combine_scatter": ms(h[:, 35]),
        "full_wait_first_dh_all": ms(h[:, 36]),
        "full_wait_first_dh_n0_staging": ms(h[:, 23]),
        "full_wait_first_recompute": ms(h[:, 38]),
        "full_wait_first_dx": ms(h[:, 37]),
        "wg1_mainloop_total": ms(h[:, 28]),
        "wg1_full_wait": ms(h[:, 29]),
        "a_loader_l1_full_spin_staging": ms(h[:, 30]),
        "a_loader_l2_full_spin": ms(h[:, 31]),
        "a_loader_empty_barrier_wait": ms(h[:, 32]),
        "dispatch_pull_loop_start_after_kernel_start": ms(h[:, 2] - h[:, 1]),
        "dispatch_pull_loop_end_after_kernel_start_max_warp": ms(
            h[:, 56:60].max(1).values - h[:, 1]
        ),
        "dispatch_busy_mean_warp": ms(h[:, [40, 44, 48, 52]].mean(1)),
        "dispatch_ring_wait_mean_warp": ms(h[:, [41, 45, 49, 53]].mean(1)),
        "dispatch_tma_load_wait_mean_warp": ms(h[:, [42, 46, 50, 54]].mean(1)),
        "dispatch_tma_store_wait_mean_warp": ms(h[:, [43, 47, 51, 55]].mean(1)),
        "fraction_no_math_wg_in_mainloop": 1 - (mainloop * ns_per_cycle / kernel_ns),
        "fraction_no_math_wg_issuing_mma_excl_waits": 1
        - ((mainloop - full_wait) * ns_per_cycle / kernel_ns),
    }
    counts = {
        "tasks_per_cta": h[:, 17],
        "mblocks_per_cta": h[:, 19],
        "dh0_tasks_per_cta": h[:, 24],
        "dispatch_rows_per_cta": h[:, 60],
    }
    summary = {
        "median_over_ctas_ms": {k: float(v.median()) for k, v in per_cta.items()},
        "mean_over_ctas_ms": {k: float(v.mean()) for k, v in per_cta.items()},
        "max_over_ctas_ms": {k: float(v.max()) for k, v in per_cta.items()},
        "counts_median": {k: float(v.median()) for k, v in counts.items()},
        "counts_sum": {k: float(v.sum()) for k, v in counts.items()},
        "sm_clock_ghz_median": float((1 / ns_per_cycle).median()),
    }
    # Per M-block records
    m = t[:, TRACE_HEADER:].reshape(t.shape[0], TRACE_MBLOCKS, TRACE_FIELDS)
    valid = m[:, :, 0] > 0
    scale = ns_per_cycle[:, None].expand(-1, TRACE_MBLOCKS)[valid] / 1e3
    rec = m[valid]
    fields = {
        "duration_claim_to_last_epilogue": rec[:, 2] - rec[:, 1],
        "staging_wait_first_full_barrier_dh_n0": rec[:, 3],
        "mainloop": rec[:, 4],
        "epilogue": rec[:, 5],
        "sched_wait": rec[:, 6],
        "full_barrier_wait_all": rec[:, 7],
        "dh_n0_l2spin_plus_xdy_pack": rec[:, 8],
        "mainloop_dh": rec[:, 10],
        "mainloop_recompute_low": rec[:, 11],
        "mainloop_recompute_high": rec[:, 12],
        "mainloop_dx": rec[:, 13],
        "epilogue_dx": rec[:, 14],
        "full_wait_first_kblock_all_tasks": rec[:, 15],
    }
    summary["per_mblock_median_us"] = {
        k: float((v * scale).median()) for k, v in fields.items()
    }
    summary["per_mblock_count"] = int(valid.sum())
    # Wave structure: claim time of first M-block relative to kernel start
    first = m[:, 0, 1] - h[:, 1]
    summary["first_mblock_claim_after_kernel_start_ms_median"] = float(
        (first * ns_per_cycle / 1e6).median()
    )
    return summary


def run(args):
    import deep_gemm
    from mono_ep.runtime import PersistentMoE

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
    backend = runtime._peer_backward
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk].contiguous()
    if args.routing == "skewed":
        ids[:] = torch.arange(topk, device="cuda")
    scores = torch.rand(tokens, topk, device="cuda").softmax(-1)
    w1 = torch.randn(experts, 2 * inner, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
    original = deep_gemm._C.fp8_mono_ep_backward_sm90
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

    def timed_native(*call_args, **call_kwargs):
        tracing = TRACE_ENV in os.environ
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

    results = {
        name: dict(samples_ms=[], env=values) for name, values in variants.items()
    }
    reference = None
    deep_gemm._C.fp8_mono_ep_backward_sm90 = timed_native
    try:
        for round_idx in range(args.rounds):
            for name, values in variants.items():
                set_env(values)
                intervals.clear()
                for _ in range(args.warmup):
                    outputs = backend.backward(dy, x, ids, scores, w1, w2)
                torch.cuda.synchronize()
                if round_idx == 0:
                    # Poison every compared cache so a skipped write cannot inherit a previous variant's bytes.
                    for index in POOLS:
                        pool = backend.pools[index]
                        pool.view(
                            torch.uint8 if pool.element_size() == 1 else torch.int32
                        ).fill_(0x7F if pool.element_size() == 1 else 0x7FC00001)
                    outputs = backend.backward(dy, x, ids, scores, w1, w2)
                    torch.cuda.synchronize()
                    outputs = [o.clone() for o in outputs]
                    pools = [backend.pools[i].clone() for i in POOLS]
                    if reference is None:
                        reference = (name, outputs, pools)
                    pool_parity = {}
                    for index, a, b in zip(POOLS, pools, reference[2], strict=True):
                        kind = torch.uint8 if a.element_size() == 1 else torch.int32
                        mismatch = int((a.view(kind) != b.view(kind)).sum())
                        pool_parity[POOLS[index]] = dict(
                            bitwise_equal=mismatch == 0, mismatched_elements=mismatch
                        )
                    results[name]["pool_parity_vs_" + reference[0]] = pool_parity
                    parity = {}
                    for label, a, b in zip(
                        ("dx", "dscore", "dw1", "dw2"),
                        outputs,
                        reference[1],
                        strict=True,
                    ):
                        diff = (a.float() - b.float()).abs()
                        parity[label] = dict(
                            bitwise_equal=bool(torch.equal(a, b)),
                            max_abs_diff=float(diff.max()),
                            rel_norm=float(
                                diff.norm() / b.float().norm().clamp_min(1e-12)
                            ),
                            finite=bool(torch.isfinite(a.float()).all()),
                        )
                    results[name]["parity_vs_" + reference[0]] = parity
                intervals.clear()
                traces.clear()
                for _ in range(args.iters):
                    backend.backward(dy, x, ids, scores, w1, w2)
                torch.cuda.synchronize()
                results[name]["samples_ms"].extend(
                    a.elapsed_time(b) for a, b in intervals
                )
                if traces:
                    results[name].setdefault("self_check", []).extend(
                        dict(
                            hdz_pack_mismatches=int(tr[:, 61].sum()),
                            dz_quant_mismatches=int(tr[:, 62].sum()),
                            checked=int(tr[:, 63].sum()),
                        )
                        for tr in traces
                    )
                if traces and round_idx == 0:
                    summaries = [summarize_trace(tr.cpu()) for tr in traces]
                    results[name]["trace_iterations"] = len(summaries)
                    results[name]["trace_summaries"] = summaries
                    mid = sorted(
                        range(len(summaries)),
                        key=lambda i: summaries[i]["median_over_ctas_ms"][
                            "kernel_wall"
                        ],
                    )[len(summaries) // 2]
                    results[name]["trace_raw_header_median_iteration"] = (
                        traces[mid][:, :TRACE_HEADER].cpu().tolist()
                    )
    finally:
        deep_gemm._C.fp8_mono_ep_backward_sm90 = original
        set_env({})
    for value in results.values():
        value["median_ms"] = _median(value["samples_ms"])
        value["samples"] = len(value["samples_ms"])
    print(
        "DIAG_JSON "
        + json.dumps(
            dict(
                benchmark="hdz_pack_single_rank_persistent_backward",
                device=device_name,
                routing=args.routing,
                tokens=tokens,
                hidden=hidden,
                intermediate=inner,
                experts=experts,
                topk=topk,
                routed_rows=int(backend.counts.sum()),
                max_tokens_per_rank=int(backend.buffer.num_max_tokens_per_rank),
                ring_tokens=int(backend.buffer.l1_acts.shape[0]),
                padded_rows=int(backend.padded_counts.sum()),
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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=2)
    run(parser.parse_args())
