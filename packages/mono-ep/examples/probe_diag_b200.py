# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-rank SM100 persistent forward/backward diagnosis on one B200.

Modes:
  floors   plain grouped BF16 GEMM floors on the operands the native backward captures
  entry    native-entry timing of variants (baseline / phase trace / ablation), interleaved
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

TOKENS, HIDDEN, INNER, EXPERTS, TOPK = (
    # ``tokens x hidden x inner x experts x topk`` override (default: the Qwen3-30B-A3B per-GPU shape)
    int(v)
    for v in os.environ.get("MONO_EP_PROBE_SHAPE", "16384x2048x768x16x8").split("x")
)
ENV = {
    "bwd": ("MONO_EP_SM100_PHASE_TRACE", "MONO_EP_SM100_DIAG_ABLATE"),
    "fwd": (
        "MONO_EP_SM100_FORWARD_PHASE_TRACE",
        "MONO_EP_SM100_FORWARD_DIAG_ABLATE",
    ),
}
STAGES_ENV = "MONO_EP_SM100_BACKWARD_STAGES"
# Variant keys mapped to environment selectors (besides trace/ablate)
VARIANT_ENV = {
    "stages": STAGES_ENV,
    "opt": "MONO_EP_SM100_BACKWARD_OPTIONS",
    "bk": "MONO_EP_SM100_BACKWARD_BLOCK_K",
    "cache": "MONO_EP_WEIGHT_CACHE",
    "pull": "MONO_EP_SM100_BACKWARD_PULL_BYTES",
    "dwd": "MONO_EP_SM100_DW_DIRECT",
    "wcc": "MONO_EP_WEIGHT_CACHE_COMPILED",
    "fpull": "MONO_EP_SM100_FORWARD_PULL_BYTES",
    "fopt": "MONO_EP_SM100_FORWARD_OPTIONS",
    "wcf": "MONO_EP_WEIGHT_CACHE_FUSED",
    "idw": "MONO_EP_SM100_INKERNEL_DW",
    "idwpb": "MONO_EP_SM100_INKERNEL_DW_PER_BLOCK",
    "idww": "MONO_EP_SM100_INKERNEL_DW_WIDE",
}
STRIDE = {"bwd": 96 + 64 * 8, "fwd": 96}
NATIVE = {"bwd": "bf16_mono_ep_backward", "fwd": "bf16_mono_ep"}
# The in-kernel dW entry (MONO_EP_SM100_INKERNEL_DW=1) is timed and traced as the backward too
NATIVE_DW = "bf16_mono_ep_backward_dw"


def _median(values):
    return statistics.median(values) if values else None


def _setup(routing, seed=881):
    import deep_gemm
    from mono_ep.runtime import PersistentMoE

    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name()
    if "B200" not in device_name:
        raise RuntimeError(f"Expected B200, got {device_name}")
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    assert dist.get_world_size() == 1
    torch.manual_seed(seed)
    runtime = PersistentMoE(
        dist.group.WORLD,
        EXPERTS,
        TOKENS,
        TOPK,
        HIDDEN,
        INNER,
        compute_dtype="bf16",
        retain_activations=False,
        backward_backend="native",
    )
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    ids = torch.rand(TOKENS, EXPERTS, device="cuda").argsort(-1)[:, :TOPK].contiguous()
    if routing == "skewed":
        ids[:] = torch.arange(TOPK, device="cuda")
    scores = torch.rand(TOKENS, TOPK, device="cuda").softmax(-1)
    w1 = torch.randn(EXPERTS, 2 * INNER, HIDDEN, device="cuda") / HIDDEN**0.5
    w2 = torch.randn(EXPERTS, HIDDEN, INNER, device="cuda") / INNER**0.5
    return deep_gemm, runtime, device_name, (x, dy, ids, scores, w1, w2)


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = []
    for _ in range(iters):
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        a.record()
        fn()
        b.record()
        events.append((a, b))
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in events]


def run_floors(args):
    deep_gemm, runtime, device_name, (x, dy, ids, scores, w1, w2) = _setup(args.routing)
    backend = runtime._peer_backward
    backend.backward(dy, x, ids, scores, w1, w2)
    torch.cuda.synchronize()
    x_pool, dy_pool, h_pool, dz_pool = [p.clone() for p in backend.pools]
    starts, counts = backend.starts.clone(), backend.counts.clone()
    ends = (starts + counts).int()
    rows = int(counts.sum())
    m_total = int(-(-int(ends[-1]) // 128) * 128)
    w1b, w2b = w1.bfloat16(), w2.bfloat16()
    w2t = w2b.transpose(1, 2).contiguous()
    w1t = w1b.transpose(1, 2).contiguous()

    def grouped(a, b):
        out = torch.empty(m_total, b.shape[1], device="cuda", dtype=torch.bfloat16)

        def call():
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                a[:m_total], b, out, ends, use_psum_layout=True
            )

        return out, call

    gemms = {
        "bwd_dh": (dy_pool, w2t),
        "bwd_z_recompute": (x_pool, w1b),
        "bwd_dx": (dz_pool, w1t),
        "fwd_z": (x_pool, w1b),
        "fwd_y": (h_pool, w2b),
    }
    results, checks = {}, {}
    for name, (a, b) in gemms.items():
        out, call = grouped(a, b)
        samples = _time(call, args.warmup, args.iters)
        n, k = b.shape[1], b.shape[2]
        flops = 2.0 * rows * n * k
        med = _median(samples)
        results[name] = dict(
            median_ms=med,
            samples=len(samples),
            samples_ms=samples,
            n=n,
            k=k,
            tflops=flops / (med * 1e-3) / 1e12,
        )
        errs = []
        for e in range(EXPERTS):
            s, c = int(starts[e]), int(counts[e])
            for r in {s, s + c - 1} if c else ():
                ref = a[r].float() @ b[e].float().T
                errs.append(
                    float((out[r].float() - ref).norm() / ref.norm().clamp_min(1e-8))
                )
        checks[name] = dict(max_rel_row_error=max(errs), rows_checked=len(errs))

    from mono_ep.ops import weight_grad_bf16

    for name, (a, b) in {"dw1": (dz_pool, x_pool), "dw2": (dy_pool, h_pool)}.items():
        holder = {}

        def call(a=a, b=b, holder=holder):
            holder["out"] = weight_grad_bf16(a, b, ends, None, EXPERTS, True)

        samples = _time(call, args.warmup, args.iters)
        med = _median(samples)
        flops = 2.0 * rows * a.shape[1] * b.shape[1]
        results[name] = dict(
            median_ms=med,
            samples=len(samples),
            samples_ms=samples,
            m=a.shape[1],
            n=b.shape[1],
            tflops=flops / (med * 1e-3) / 1e12,
            scope="production k-grouped BF16 dW (FP32 output), includes output zero-fill",
        )
        e = int(torch.argmax(counts))
        s, c = int(starts[e]), int(counts[e])
        ref = a[s : s + c].float().T @ b[s : s + c].float()
        got = holder["out"][e]
        checks[name] = dict(
            rel_error_largest_expert=float(
                (got - ref).norm() / ref.norm().clamp_min(1e-8)
            )
        )

    # Elementwise SwiGLU backward / forward as standalone fused kernels
    z = torch.randn(rows, 2 * INNER, device="cuda", dtype=torch.bfloat16)
    dh = torch.randn(rows, INNER, device="cuda", dtype=torch.bfloat16)
    s = torch.rand(rows, device="cuda")

    def swiglu_backward(z, dh, s):
        gate, up = z.float().chunk(2, -1)
        sig = torch.sigmoid(gate)
        silu = gate * sig
        g = dh.float()
        h = (silu * up * s[:, None]).bfloat16()
        dg = (g * s[:, None] * up * (sig + silu * (1 - sig))).bfloat16()
        du = (g * s[:, None] * silu).bfloat16()
        ds = (g * silu * up).sum(-1)
        return h, torch.cat((dg, du), -1), ds

    def swiglu_forward(z, s):
        gate, up = z.float().chunk(2, -1)
        return (torch.nn.functional.silu(gate) * up * s[:, None]).bfloat16()

    compiled = {}
    for name, fn, fargs in (
        ("swiglu_backward_elementwise", swiglu_backward, (z, dh, s)),
        ("swiglu_forward_elementwise", swiglu_forward, (z, s)),
    ):
        try:
            cfn = torch.compile(fn, fullgraph=True)
            cfn(*fargs)
            compiled[name] = True
        except Exception as error:  # noqa: BLE001
            cfn = fn
            compiled[name] = f"eager fallback: {type(error).__name__}"
        samples = _time(
            lambda cfn=cfn, fargs=fargs: cfn(*fargs), args.warmup, args.iters
        )
        results[name] = dict(
            median_ms=_median(samples), samples=len(samples), samples_ms=samples
        )
    sums = dict(
        backward_gemms_ms=sum(
            results[k]["median_ms"] for k in ("bwd_dh", "bwd_z_recompute", "bwd_dx")
        ),
        dw_ms=results["dw1"]["median_ms"] + results["dw2"]["median_ms"],
        forward_gemms_ms=results["fwd_z"]["median_ms"] + results["fwd_y"]["median_ms"],
    )
    sums["backward_plus_dw_ms"] = sums["backward_gemms_ms"] + sums["dw_ms"]
    print(
        "DIAG_JSON "
        + json.dumps(
            dict(
                benchmark="diag_b200_floors",
                device=device_name,
                routing=args.routing,
                dtype="bf16 operands, fp32 accumulate (DeepGEMM SM100 grouped kernels)",
                routed_rows=rows,
                padded_rows=m_total,
                counts=counts.tolist(),
                warmup=args.warmup,
                iters=args.iters,
                results=results,
                spot_checks=checks,
                compiled=compiled,
                sums=sums,
            )
        ),
        flush=True,
    )
    runtime.close()
    dist.destroy_process_group()


def summarize_trace(direction, trace):
    t = trace.double()
    h = t[:, :96]
    wall_ns = h[:, 2] - h[:, 0]
    cycles = h[:, 3] - h[:, 1]
    ok = (wall_ns > 0) & (cycles > 0)
    h, wall_ns, cycles = h[ok], wall_ns[ok], cycles[ok]
    npc = wall_ns / cycles

    def ms(v):
        return v * npc / 1e6

    leader = h[:, 32] > 0
    per_cta = {
        "kernel_wall": wall_ns / 1e6,
        "epi_prologue_before_task_loop": ms(h[:, 4] - h[:, 1]),
        "epi_task_loop": ms(h[:, 5] - h[:, 4]),
        "epi_tmem_full_wait": ms(h[:, 6]),
        "combine_grid_sync_and_barriers": ms(h[:, 21] - h[:, 5]),
        "combine_reduce_loop": ms(h[:, 22] - h[:, 21]),
        "tma_a_l1_full_spin": ms(h[:, 43]),
        "tma_a_l2_spin_or_dependency": ms(h[:, 44]),
        "tma_a_empty_wait": ms(h[:, 45]),
        "dispatch_pull_start_after_kernel_start": ms(h[:, 60] - h[:, 1]),
        "dispatch_pull_end_after_kernel_start_max_warp": ms(h[:, 59] - h[:, 1]),
        "dispatch_busy_mean_warp": ms(h[:, [46, 49, 52, 55]].mean(1)),
        "dispatch_ring_wait_mean_warp": ms(h[:, [47, 50, 53, 56]].mean(1)),
        "dispatch_copy_mean_warp": ms(h[:, [48, 51, 54, 57]].mean(1)),
        "dispatch_cleanup_end_after_kernel_start": ms(h[:, 62] - h[:, 1]),
    }
    mma = {
        "mma_tmem_empty_wait": ms(h[:, 30]),
        "mma_full_barrier_wait": ms(h[:, 31]),
        "mma_loop_span": ms(h[:, 33] - h[:, 32]),
        "mma_sched_wait": ms(h[:, 42]),
    }
    if direction == "bwd":
        names = ("dh", "recompute_low", "recompute_high", "dx")
        per_cta.update(
            {
                "epi_sched_wait": ms(h[:, 26]),
                "epi_dh0_l2_empty_spin_and_dscore_zero": ms(h[:, 15]),
                "epi_dh_retain_tmem_to_smem": ms(h[:, 16]),
                "epi_recompute_atoms_swiglu_bwd_h_dz_stores_dscore": ms(h[:, 17]),
                "epi_dh_recompute_tail_fence_sync_arrive": ms(h[:, 18]),
                "epi_dscore_partial_write": ms(h[:, 25]),
                "epi_dx_combine_scatter": ms(h[:, 19]),
                "epi_dx_tail_sync": ms(h[:, 20]),
            }
        )
        for i, n in enumerate(names):
            per_cta[f"epi_busy_{n}"] = ms(h[:, 7 + i])
            per_cta[f"epi_tmem_full_wait_{n}"] = ms(h[:, 11 + i])
            mma[f"mma_kloop_{n}"] = ms(h[:, 34 + i])
            mma[f"mma_tmem_empty_wait_{n}"] = ms(h[:, 38 + i])
            mma[f"mma_kblocks_{n}"] = h[:, 64 + i]
        # In-kernel dW (zero without it): tiles, epilogue busy/wait, MMA k-loop, full waits, k-blocks, and the
        # epilogue clock at the end of the last main task and of the last dW tile
        per_cta.update(
            {
                "epi_busy_dw": ms(h[:, 69]),
                "epi_tmem_full_wait_dw": ms(h[:, 70]),
                "epi_last_main_task_end_after_kernel_start": ms(h[:, 74] - h[:, 1]),
                "epi_last_dw_end_after_kernel_start": ms(h[:, 75] - h[:, 1]),
                "epi_dw_tail_after_last_main_task": ms(
                    (h[:, 75] - h[:, 74]).clamp_min(0)
                ),
            }
        )
        mma["mma_kloop_dw"] = ms(h[:, 71])
        mma["mma_full_wait_dw"] = ms(h[:, 72])
        mma["mma_kblocks_dw"] = h[:, 73]
        kblocks = h[:, 64:68].sum(1) + h[:, 73]
        kloop = h[:, 34:38].sum(1) + h[:, 71]
    else:
        names = ("l1", "l2")
        per_cta.update(
            {
                "epi_sched_wait": ms(h[:, 17]),
                "epi_l1_l2_empty_spin": ms(h[:, 11]),
                "epi_l1_swiglu_compute_incl_tmem_load": ms(h[:, 12]),
                "epi_l1_h_store_stsm_tma": ms(h[:, 13]),
                "epi_l1_tail_store_wait_sync_arrive": ms(h[:, 14]),
                "epi_l2_combine_scatter": ms(h[:, 16]),
                "epi_l2_tail_sync": ms(h[:, 15]),
            }
        )
        for i, n in enumerate(names):
            per_cta[f"epi_busy_{n}"] = ms(h[:, 7 + i])
            per_cta[f"epi_tmem_full_wait_{n}"] = ms(h[:, 9 + i])
            mma[f"mma_kloop_{n}"] = ms(h[:, 34 + i])
            mma[f"mma_tmem_empty_wait_{n}"] = ms(h[:, 38 + i])
            mma[f"mma_kblocks_{n}"] = h[:, 64 + i]
        kblocks = h[:, 64:66].sum(1)
        kloop = h[:, 34:36].sum(1)
    # 2-CTA UMMA 256x128x64 BF16 = 4.19 MFLOP over two SMs at 8192 dense FLOP/clk/SM
    ideal_cycles = kblocks * 256
    mma["mma_kloop_total"] = ms(kloop)
    mma["mma_ideal_tensor_time_full_n"] = ms(ideal_cycles)
    mma["fraction_wall_mma_outside_kloops"] = 1 - ms(kloop) / (wall_ns / 1e6)
    mma["fraction_wall_mma_waiting_tmem_empty"] = ms(h[:, 30]) / (wall_ns / 1e6)
    mma["fraction_wall_ideal_tensor_busy"] = ms(ideal_cycles) / (wall_ns / 1e6)
    counts = {"rows_dispatched_per_cta": h[:, 58]}
    if direction == "bwd":
        counts.update(
            tasks_per_cta=h[:, 23], mblocks_per_cta=h[:, 24], dw_tiles_per_cta=h[:, 68]
        )
    else:
        counts.update(l1_tasks_per_cta=h[:, 18], l2_tasks_per_cta=h[:, 19])
    summary = {
        "ctas": int(ok.sum()),
        "leader_ctas": int(leader.sum()),
        "sm_clock_ghz_median": float((1 / npc).median()),
        "median_over_ctas_ms": {k: float(v.median()) for k, v in per_cta.items()},
        "max_over_ctas_ms": {k: float(v.max()) for k, v in per_cta.items()},
        "median_over_leader_ctas": {
            k: float(v[leader].median()) for k, v in mma.items()
        },
        "counts_median": {k: float(v.median()) for k, v in counts.items()},
        "counts_sum": {k: float(v.sum()) for k, v in counts.items()},
    }
    if direction == "bwd":
        m = t[ok][:, 96:].reshape(-1, 64, 8)
        valid = m[:, :, 0] > 0
        scale = npc[:, None].expand(-1, 64)[valid] / 1e3
        rec = m[valid]
        fields = {
            "claim_to_last_task_end": rec[:, 2] - rec[:, 1],
            "epilogue_busy": rec[:, 3],
            "epilogue_tmem_full_wait": rec[:, 4],
            "dh0_l2_empty_spin": rec[:, 5],
            "recompute_atoms": rec[:, 6],
            "dx_scatter": rec[:, 7],
        }
        summary["per_mblock_median_us"] = {
            k: float((v * scale).median()) for k, v in fields.items()
        }
        summary["per_mblock_count"] = int(valid.sum())
    return summary


def run_entry(args):
    deep_gemm, runtime, device_name, (x, dy, ids, scores, w1, w2) = _setup(args.routing)
    backend = runtime._peer_backward
    traces = {
        d: torch.zeros(
            deep_gemm.get_num_sms() * STRIDE[d], device="cuda", dtype=torch.int64
        )
        for d in ("bwd", "fwd")
    }
    originals = {d: getattr(deep_gemm._C, NATIVE[d]) for d in ("bwd", "fwd")}
    originals["bwd_dw"] = getattr(deep_gemm._C, NATIVE_DW, None)
    managed = {e for pair in ENV.values() for e in pair} | set(VARIANT_ENV.values())
    variants = []
    for spec in args.variants.split(","):
        name, direction, env = spec.split(":")
        values = {}
        for item in filter(None, env.split("+")):
            key, _, value = item.partition("=")
            values[key] = value
        variants.append((name, direction, values))

    def set_env(direction, values):
        for key in managed:
            os.environ.pop(key, None)
        trace_env, ablate_env = ENV[direction]
        if values.get("trace") == "1":
            os.environ[trace_env] = str(traces[direction].data_ptr())
        if values.get("ablate"):
            os.environ[ablate_env] = values["ablate"]
        for key, env_name in VARIANT_ENV.items():
            if values.get(key):
                os.environ[env_name] = values[key]

    intervals, raw = [], []

    def wrap(direction, original=None):
        original = original or originals[direction]
        trace_env = ENV[direction][0]

        def timed(*call_args, **call_kwargs):
            tracing = trace_env in os.environ
            if tracing:
                traces[direction].zero_()
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            result = original(*call_args, **call_kwargs)
            b.record()
            intervals.append((a, b))
            if tracing:
                raw.append(traces[direction].view(-1, STRIDE[direction]).clone())
            return result

        return timed

    def bump_weights():
        # Simulate an optimizer step: an in-place update that bumps the weight versions
        with torch.no_grad():
            w1.mul_(1.0)
            w2.mul_(1.0)

    def call(direction):
        if direction == "bwd":
            return backend.backward(dy, x, ids, scores, w1, w2)
        return (runtime(x, ids, scores, w1, w2),)

    results = {name: dict(direction=d, env=v, samples_ms=[]) for name, d, v in variants}
    references = {}
    for d in ("bwd", "fwd"):
        setattr(deep_gemm._C, NATIVE[d], wrap(d))
    if originals["bwd_dw"] is not None:
        setattr(deep_gemm._C, NATIVE_DW, wrap("bwd", originals["bwd_dw"]))
    try:
        for round_idx in range(args.rounds):
            ordered = variants if round_idx % 2 == 0 else variants[::-1]
            for name, direction, values in ordered:
                set_env(direction, values)
                bump = values.get("bump") == "1"
                for _ in range(args.warmup):
                    if bump:
                        bump_weights()
                    outputs = call(direction)
                torch.cuda.synchronize()
                if round_idx == 0:
                    outputs = [o.clone() for o in outputs]
                    ref = references.setdefault(direction, (name, outputs))
                    labels = (
                        ("dx", "dscore", "dw1", "dw2") if direction == "bwd" else ("y",)
                    )
                    parity = {}
                    for label, a, b in zip(labels, outputs, ref[1], strict=True):
                        diff = (a.float() - b.float()).abs()
                        parity[label] = dict(
                            bitwise_equal=bool(torch.equal(a, b)),
                            max_abs_diff=float(diff.max()),
                            rel_norm=float(
                                diff.norm() / b.float().norm().clamp_min(1e-12)
                            ),
                            finite=bool(torch.isfinite(a.float()).all()),
                        )
                    results[name]["parity_vs_" + ref[0]] = parity
                intervals.clear()
                raw.clear()
                outer = []
                for _ in range(args.iters):
                    if bump:
                        bump_weights()
                    a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    a.record()
                    call(direction)
                    b.record()
                    outer.append((a, b))
                torch.cuda.synchronize()
                results[name]["samples_ms"].extend(
                    a.elapsed_time(b) for a, b in intervals
                )
                results[name].setdefault("whole_call_samples_ms", []).extend(
                    a.elapsed_time(b) for a, b in outer
                )
                if raw and round_idx == 0:
                    summaries = [summarize_trace(direction, r.cpu()) for r in raw]
                    order = sorted(
                        range(len(summaries)),
                        key=lambda i: summaries[i]["median_over_ctas_ms"][
                            "kernel_wall"
                        ],
                    )
                    mid = order[len(order) // 2]
                    results[name]["trace_iterations"] = len(summaries)
                    results[name]["trace_median_iteration"] = summaries[mid]
                    results[name]["trace_kernel_wall_ms_all_iterations"] = [
                        s["median_over_ctas_ms"]["kernel_wall"] for s in summaries
                    ]
    finally:
        for d in ("bwd", "fwd"):
            setattr(deep_gemm._C, NATIVE[d], originals[d])
        if originals["bwd_dw"] is not None:
            setattr(deep_gemm._C, NATIVE_DW, originals["bwd_dw"])
        for key in managed:
            os.environ.pop(key, None)
    for value in results.values():
        value["median_ms"] = _median(value["samples_ms"])
        value["samples"] = len(value["samples_ms"])
        value["whole_call_median_ms"] = _median(value["whole_call_samples_ms"])
    print(
        "DIAG_JSON "
        + json.dumps(
            dict(
                benchmark="diag_b200_single_rank_native_entry",
                device=device_name,
                routing=args.routing,
                tokens=TOKENS,
                hidden=HIDDEN,
                intermediate=INNER,
                experts=EXPERTS,
                topk=TOPK,
                num_sms=deep_gemm.get_num_sms(),
                routed_rows=int(backend.counts.sum()),
                warmup_per_round=args.warmup,
                iters_per_round=args.iters,
                rounds=args.rounds,
                scope="samples_ms: native host entry CUDA events; excludes weight "
                "preparation, dW and input staging copies; local-only dispatch/combine. "
                "whole_call: the full single-rank backward (prep + native + dW) or "
                "forward op",
                interleaving="variants alternate within each round (order reversed on odd rounds), same process and GPU",
                results=results,
            )
        ),
        flush=True,
    )
    runtime.close()
    dist.destroy_process_group()


def _reference_backward(x, dy, ids, scores, w1, w2):
    """FP32 reference with the kernel's BF16 rounding points (as in the 2-GPU validator)."""
    w1b, w2b = w1.bfloat16().float(), w2.bfloat16().float()
    dx = torch.zeros_like(x, dtype=torch.float32)
    ds = torch.zeros_like(scores)
    dw1 = torch.zeros_like(w1b)
    dw2 = torch.zeros_like(w2b)
    for expert in range(w1.shape[0]):
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        xr, dyr = x[rows].float(), dy[rows].float()
        z = (xr @ w1b[expert].T).bfloat16().float()
        gate, up = z.chunk(2, dim=-1)
        dh = (dyr @ w2b[expert]).bfloat16().float()
        sigmoid = gate.sigmoid()
        silu = gate * sigmoid
        s = scores[rows, slots, None]
        weighted = dh * s
        dz = torch.cat(
            (weighted * up * (sigmoid + silu * (1 - sigmoid)), weighted * silu), -1
        ).bfloat16()
        dx.index_add_(0, rows, (dz.float() @ w1b[expert]).bfloat16().float())
        ds[rows, slots] = (dh * silu * up).sum(-1)
        h = (silu * up * s).bfloat16().float()
        dw1[expert] = dz.float().T @ xr
        dw2[expert] = dyr.T @ h
    return dx, ds, dw1, dw2


def run_validate(args):
    deep_gemm, runtime, device_name, (x, dy, ids, scores, w1, w2) = _setup(args.routing)
    backend = runtime._peer_backward
    expected = _reference_backward(x, dy, ids, scores, w1, w2)
    results, first = {}, None
    for spec in args.variants.split(","):
        name, direction, env = spec.split(":")
        assert direction == "bwd"
        for env_name in VARIANT_ENV.values():
            os.environ.pop(env_name, None)
        for item in filter(None, env.split("+")):
            key, _, value = item.partition("=")
            os.environ[VARIANT_ENV[key]] = value
        repeats = []
        for _ in range(args.iters):
            outputs = [o.clone() for o in backend.backward(dy, x, ids, scores, w1, w2)]
            torch.cuda.synchronize()
            repeats.append(outputs)
        errors = {
            label: [
                float(
                    (r[i].float() - expected[i]).norm()
                    / expected[i].norm().clamp_min(1e-12)
                )
                for r in repeats
            ]
            for i, label in enumerate(("dx", "dscore", "dw1", "dw2"))
        }
        if first is None:
            first = repeats[0]
        results[name] = dict(
            env=env,
            repeats=len(repeats),
            rel_error_vs_fp32_reference=errors,
            max_rel_error=dict((k, max(v)) for k, v in errors.items()),
            dx_bitwise_equal_to_first_variant=all(
                torch.equal(r[0], first[0]) for r in repeats
            ),
            dx_bitwise_reproducible=all(
                torch.equal(r[0], repeats[0][0]) for r in repeats
            ),
            finite=all(
                bool(torch.isfinite(t.float()).all()) for r in repeats for t in r
            ),
        )
    for env_name in VARIANT_ENV.values():
        os.environ.pop(env_name, None)
    print(
        "DIAG_JSON "
        + json.dumps(
            dict(
                benchmark="diag_b200_backward_fp32_reference_validation",
                device=device_name,
                routing=args.routing,
                routed_rows=int(backend.counts.sum()),
                reference="FP32 matmuls with BF16 rounding of z, dh, dz, h and partial dx "
                "as in examples/validate_native_persistent_backward.py",
                results=results,
            )
        ),
        flush=True,
    )
    runtime.close()
    dist.destroy_process_group()


def run_host(args):
    """Host-side launch cost per call with the GPU kept busy (CPU wall, no GPU waits)."""
    import time

    from mono_ep import weight_cache

    deep_gemm, runtime, device_name, (x, dy, ids, scores, w1, w2) = _setup(args.routing)
    backend = runtime._peer_backward
    variants = []
    for spec in args.variants.split(","):
        name, direction, env = spec.split(":")
        values = dict(item.partition("=")[::2] for item in filter(None, env.split("+")))
        variants.append((name, direction, values))
    marks = {}
    originals = {d: getattr(deep_gemm._C, NATIVE[d]) for d in ("bwd", "fwd")}
    original_layouts = weight_cache.weight_layouts

    def stamp(label, fn):
        def wrapped(*a, **k):
            marks.setdefault(label + "_enter", time.perf_counter_ns())
            out = fn(*a, **k)
            marks[label + "_exit"] = time.perf_counter_ns()
            return out

        return wrapped

    results = {}
    try:
        for d in ("bwd", "fwd"):
            setattr(deep_gemm._C, NATIVE[d], stamp("native", originals[d]))
        weight_cache.weight_layouts = stamp("layouts", original_layouts)
        for round_idx in range(args.rounds):
            ordered = variants if round_idx % 2 == 0 else variants[::-1]
            for name, direction, values in ordered:
                for key in VARIANT_ENV.values():
                    os.environ.pop(key, None)
                for key, value in values.items():
                    if key in VARIANT_ENV:
                        os.environ[VARIANT_ENV[key]] = value
                bump = values.get("bump") == "1"
                rows = results.setdefault(
                    name, dict(direction=direction, env=values, rows=[])
                )["rows"]
                for i in range(args.warmup + args.iters):
                    torch.cuda.synchronize()
                    if bump:
                        with torch.no_grad():
                            w1.mul_(1.0)
                            w2.mul_(1.0)
                    torch.cuda._sleep(50_000_000)
                    marks.clear()
                    start = time.perf_counter_ns()
                    if direction == "bwd":
                        backend.backward(dy, x, ids, scores, w1, w2)
                    else:
                        runtime(x, ids, scores, w1, w2)
                    end = time.perf_counter_ns()
                    if i >= args.warmup:
                        rows.append(
                            dict(
                                total_us=(end - start) / 1e3,
                                to_native_launch_us=(marks["native_exit"] - start)
                                / 1e3,
                                native_launch_us=(
                                    marks["native_exit"] - marks["native_enter"]
                                )
                                / 1e3,
                                layouts_us=(
                                    marks["layouts_exit"] - marks["layouts_enter"]
                                )
                                / 1e3
                                if "layouts_enter" in marks
                                else None,
                                after_native_us=(end - marks["native_exit"]) / 1e3,
                            )
                        )
                torch.cuda.synchronize()
    finally:
        for d in ("bwd", "fwd"):
            setattr(deep_gemm._C, NATIVE[d], originals[d])
        weight_cache.weight_layouts = original_layouts
        for key in VARIANT_ENV.values():
            os.environ.pop(key, None)
    for value in results.values():
        rows = value["rows"]
        value["median_us"] = {
            k: _median([r[k] for r in rows if r[k] is not None])
            for k in rows[0]
            if any(r[k] is not None for r in rows)
        }
        value["samples"] = len(rows)
    print(
        "DIAG_JSON "
        + json.dumps(
            dict(
                benchmark="diag_b200_host_launch_cost",
                device=device_name,
                routing=args.routing,
                scope="CPU wall per call with a 50M-cycle sleep kernel queued first, so no host call "
                "waits on the GPU; to_native_launch = call entry to the native launcher's return",
                results=results,
            )
        ),
        flush=True,
    )
    runtime.close()
    dist.destroy_process_group()


def run_overlap(args):
    """Step-4 feasibility: a backward's dW GEMMs on a side stream with R SMs while the next persistent forward
    (the previous layer's checkpoint recompute in training) runs on 148 - R SMs, against the serial schedule."""
    from mono_ep.ops import weight_grad_bf16
    from mono_ep.runtime import PersistentMoE

    deep_gemm, runtime, device_name, (x, dy, ids, scores, w1, w2) = _setup(args.routing)
    backend = runtime._peer_backward
    _, _, ref_dw1, ref_dw2 = backend.backward(dy, x, ids, scores, w1, w2)
    torch.cuda.synchronize()
    ref_dw1, ref_dw2 = ref_dw1.clone(), ref_dw2.clone()
    x_pool, dy_pool, h_pool, dz_pool = backend.pools[:4]
    ends = backend.starts + backend.counts
    physical = torch.cuda.get_device_properties(0).multi_processor_count

    def dw(sms):
        deep_gemm.set_num_sms(sms)
        a = weight_grad_bf16(dz_pool, x_pool, ends, None, backend.experts, True)
        b = weight_grad_bf16(dy_pool, h_pool, ends, None, backend.experts, True)
        return a, b

    reserved = [int(r) for r in args.variants.split(",") if r.isdigit()] or [
        16,
        24,
        32,
        48,
    ]
    runtimes = {0: runtime}
    for r in reserved:
        deep_gemm.set_num_sms(physical - r)
        runtimes[r] = PersistentMoE(
            dist.group.WORLD,
            EXPERTS,
            TOKENS,
            TOPK,
            HIDDEN,
            INNER,
            compute_dtype="bf16",
            retain_activations=False,
            reserved_sms=r,
        )
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()

    def schedule(r):
        if r == 0:
            out = dw(physical)
            deep_gemm.set_num_sms(physical)
            runtime(x, ids, scores, w1, w2)
            return out
        ready = torch.cuda.Event()
        ready.record(main)
        side.wait_event(ready)
        with torch.cuda.stream(side):
            out = dw(r)
        deep_gemm.set_num_sms(physical - r)
        runtimes[r](x, ids, scores, w1, w2)
        done = torch.cuda.Event()
        done.record(side)
        main.wait_event(done)
        return out

    def fwd_only(r):
        deep_gemm.set_num_sms(physical - r)
        runtimes[r](x, ids, scores, w1, w2)

    def dw_only(r):
        dw(physical - 0 if r == 0 else r)

    results = {}
    labels = [("serial", 0)] + [(f"overlap_r{r}", r) for r in reserved]
    for round_idx in range(args.rounds):
        for label, r in labels if round_idx % 2 == 0 else labels[::-1]:
            for kind, fn in (
                ("schedule", lambda r=r: schedule(r)),
                ("forward_alone", lambda r=r: fwd_only(r)),
                ("dw_alone", lambda r=r: dw_only(r)),
            ):
                samples = _time(fn, args.warmup, args.iters)
                results.setdefault(label, {}).setdefault(kind, []).extend(samples)
    parity = {}
    for label, r in labels:
        a, b = schedule(r)
        torch.cuda.synchronize()
        parity[label] = dict(
            dw1_bitwise=bool(torch.equal(a, ref_dw1)),
            dw2_bitwise=bool(torch.equal(b, ref_dw2)),
        )
    deep_gemm.set_num_sms(physical)
    summary = {
        label: {
            kind: dict(median_ms=_median(v), samples=len(v))
            for kind, v in kinds.items()
        }
        for label, kinds in results.items()
    }
    print(
        "DIAG_JSON "
        + json.dumps(
            dict(
                benchmark="diag_b200_dw_overlap_feasibility",
                device=device_name,
                routing=args.routing,
                scope="schedule = dW1 + dW2 of one backward plus one persistent forward call; serial runs both on all "
                "SMs back to back, overlap_rR runs dW on a side stream with R SMs while the forward runs on 148 - R",
                results=summary,
                parity_vs_inline_dw=parity,
            )
        ),
        flush=True,
    )
    for rt in runtimes.values():
        rt.close()
    dist.destroy_process_group()


def run_dw(args):
    """dW1/dW2 variants on the captured pools: production, swapped-and-transposed, direct output."""
    os.environ["DG_PRINT_CONFIGS"] = "1"
    deep_gemm, runtime, device_name, (x, dy, ids, scores, w1, w2) = _setup(args.routing)
    backend = runtime._peer_backward
    backend.backward(dy, x, ids, scores, w1, w2)
    torch.cuda.synchronize()
    x_pool, dy_pool, h_pool, dz_pool = [p.clone() for p in backend.pools]
    starts, counts = backend.starts.clone(), backend.counts.clone()
    ends = (starts + counts).int()
    rows = int(counts.sum())

    def kgrouped(a, b, c_zero):
        shape = (EXPERTS, a.shape[1], b.shape[1])
        if c_zero:
            out = torch.zeros(shape, device="cuda", dtype=torch.float32)
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                a, b, out, None, ends, out, use_psum_layout=True
            )
        else:
            out = torch.empty(shape, device="cuda", dtype=torch.float32)
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                a, b, out, None, ends, None, use_psum_layout=True
            )
        return out

    variants = {
        "dw1_production": lambda: kgrouped(dz_pool, x_pool, True),
        "dw1_direct": lambda: kgrouped(dz_pool, x_pool, False),
        "dw1_swapped_transposed": lambda: (
            kgrouped(x_pool, dz_pool, False).transpose(1, 2).contiguous()
        ),
        "dw2_production": lambda: kgrouped(dy_pool, h_pool, True),
        "dw2_direct": lambda: kgrouped(dy_pool, h_pool, False),
        "dw2_swapped_transposed": lambda: (
            kgrouped(h_pool, dy_pool, False).transpose(1, 2).contiguous()
        ),
        "dw2_swapped_only": lambda: kgrouped(h_pool, dy_pool, False),
    }
    outputs = {name: fn() for name, fn in variants.items()}
    torch.cuda.synchronize()
    samples = {name: [] for name in variants}
    for _ in range(args.rounds):
        for name, fn in variants.items():
            samples[name].extend(_time(fn, args.warmup, args.iters))
    results = {}
    for name in variants:
        prefix = name[:3]
        ref = outputs[f"{prefix}_production"]
        out = outputs[name]
        flops = (
            2.0
            * rows
            * (2 * INNER if prefix == "dw1" else HIDDEN)
            * (HIDDEN if prefix == "dw1" else INNER)
        )
        med = _median(samples[name])
        results[name] = dict(
            median_ms=med,
            samples=len(samples[name]),
            tflops=flops / (med * 1e-3) / 1e12,
            bitwise_equal_production=bool(
                out.shape == ref.shape and torch.equal(out, ref)
            ),
            rel_diff_production=float(
                (out.reshape(ref.shape) - ref).norm() / ref.norm()
            )
            if out.numel() == ref.numel()
            else None,
        )
    # Does the direct-output kernel write zero for an empty group?
    empty_ends = ends.clone()
    empty_ends[3] = empty_ends[2]
    probe = torch.full(
        (EXPERTS, dy_pool.shape[1], h_pool.shape[1]),
        float("nan"),
        device="cuda",
        dtype=torch.float32,
    )
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        dy_pool, h_pool, probe, None, empty_ends, None, use_psum_layout=True
    )
    torch.cuda.synchronize()
    empty_group = dict(
        all_zero=bool((probe[3] == 0).all()),
        any_nan=bool(probe[3].isnan().any()),
    )
    print(
        "DIAG_JSON "
        + json.dumps(
            dict(
                benchmark="diag_b200_dw_variants",
                device=device_name,
                routing=args.routing,
                routed_rows=rows,
                warmup=args.warmup,
                iters=args.iters,
                rounds=args.rounds,
                results=results,
                direct_output_empty_group=empty_group,
            )
        ),
        flush=True,
    )
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("floors", "entry", "validate", "dw", "host", "overlap")
    )
    parser.add_argument("--routing", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--variants", default="bwd_base:bwd:,fwd_base:fwd:")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    {
        "floors": run_floors,
        "entry": run_entry,
        "validate": run_validate,
        "dw": run_dw,
        "host": run_host,
        "overlap": run_overlap,
    }[args.mode](args)
