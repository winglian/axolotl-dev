# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Multi-rank SM100 persistent forward/backward timing and phase trace.

Runs the Qwen3-30B-A3B per-GPU workload (16,384 tokens, H 2048, I 768, 16
local experts, top-8) at the launched world size (8 for EP8, 1 for the
same-allocation single-rank control) and reports:

* untraced per-iteration CUDA-event timings of the forward and backward calls
  and of their parts (native kernel, weight layouts, dW, everything else),
  with the weight cache hit (no bump) and rebuilt (weights bumped per call);
* the per-CTA phase trace plus the distributed extension (diag bit 1024):
  globaltimer stamps of the dispatch prologue, pull start/end, epilogue task
  loop, local grid sync and combine; per-peer pull rows, first/last arrival and
  copy cycles; combine-ready waits. Raw per-rank traces are reduced on each rank
  and gathered to rank 0 for cross-rank skew.
"""

import argparse
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_diag_b200 import STRIDE, summarize_trace  # noqa: E402

TOKENS, HIDDEN, INNER, LOCAL_EXPERTS, TOPK = 16384, 2048, 768, 16, 8
DIST_STRIDE = 64
ENV = {
    "bwd": ("MONO_EP_SM100_PHASE_TRACE", "MONO_EP_SM100_DIAG_ABLATE"),
    "fwd": (
        "MONO_EP_SM100_FORWARD_PHASE_TRACE",
        "MONO_EP_SM100_FORWARD_DIAG_ABLATE",
    ),
}
NATIVE = {"bwd": "bf16_mono_ep_backward", "fwd": "bf16_mono_ep"}


def _median(values):
    return statistics.median(values) if values else None


def _dist_summary(direction, trace, world, sms):
    """Reduce one rank's raw trace (CTA records + distributed extension) to ms values."""
    stride = STRIDE[direction]
    raw_head = trace[: sms * stride].view(sms, stride)[:, :96]
    raw_ext = trace[sms * stride :].view(sms, DIST_STRIDE)
    wall = raw_head[:, 2] - raw_head[:, 0]
    clocks = raw_head[:, 3] - raw_head[:, 1]
    ok = (wall > 0) & (clocks > 0)
    raw_head, raw_ext = raw_head[ok], raw_ext[ok]
    head, ext = raw_head.double(), raw_ext.double()
    npc = wall[ok].double() / clocks[ok].double()
    start = raw_head[:, 0].min()
    end = raw_head[:, 2].max()

    def rel(v):
        valid = v > 0
        return (
            ((v[valid] - start).double() / 1e6)
            if valid.any()
            else torch.zeros(1).double()
        )

    def span(a, b):
        valid = (a > 0) & (b > 0)
        return (b[valid] - a[valid]).double() / 1e6

    def cyc(v):
        return (
            v * npc[: v.shape[0]] / 1e6
            if v.shape[0] == npc.shape[0]
            else v * npc.median() / 1e6
        )

    def stats(v):
        if v.numel() == 0:
            return None
        return dict(median=float(v.median()), max=float(v.max()), min=float(v.min()))

    leader = raw_ext[:, 48] > 0
    out = dict(
        start_ns=int(start),
        end_ns=int(end),
        kernel_ms=float((end - start).double() / 1e6),
        cta_start_spread_ms=float((raw_head[:, 0].max() - start).double() / 1e6),
        dispatch_prologue_end=stats(rel(raw_ext[:, 0])),
        pull_start=stats(rel(raw_ext[:, 1])),
        prepull_barrier_wait_ms=stats(span(raw_ext[:, 0], raw_ext[:, 1])),
        pull_end=stats(rel(raw_ext[:, 2])),
        dispatch_cleanup_start=stats(rel(raw_ext[:, 3])),
        dispatch_end=stats(rel(raw_ext[:, 4])),
        epilogue_first_task=stats(rel(raw_ext[:, 40])),
        epilogue_loop_end=stats(rel(raw_ext[:, 41])),
        epilogue_grid_sync_exit=stats(rel(raw_ext[:, 42])),
        local_grid_sync_wait_ms=stats(span(raw_ext[:, 41], raw_ext[:, 42])),
        combine_start=stats(rel(raw_ext[:, 43])),
        combine_end=stats(rel(raw_head[:, 2])),
        combine_span_ms=stats(span(raw_ext[:, 43], raw_head[:, 2])),
        combine_first_ready_wait_ms=stats(cyc(ext[:, 45])),
        combine_ready_wait_mean_warp_ms=stats(cyc(ext[:, 44]) / 8),
        combine_max_first_wait_any_warp_ms=stats(cyc(ext[:, 46])),
        combine_tokens_per_cta=stats(ext[:, 47]),
        mma_first_task=stats(rel(raw_ext[leader, 48])),
        mma_last_end=stats(rel(raw_ext[leader, 49])),
        tma_a_operand_wait_ms=stats(cyc(head[:, 43] + head[:, 44])),
        mma_full_barrier_wait_ms=stats(head[leader, 31] * npc.median() / 1e6),
        epilogue_sched_wait_ms=stats(
            cyc(head[:, 26] if direction == "bwd" else head[:, 17])
        ),
        rows_dispatched=float(head[:, 58].sum()),
    )
    peers = []
    for p in range(world):
        rows = ext[:, 8 + p]
        has = rows > 0
        first = ~raw_ext[has, 16 + p] if has.any() else None
        peers.append(
            dict(
                peer=p,
                rows=float(rows.sum()),
                first_arrival_ms=float((first.min() - start).double() / 1e6)
                if first is not None
                else None,
                last_arrival_ms=float(
                    (raw_ext[has, 24 + p].max() - start).double() / 1e6
                )
                if has.any()
                else None,
                copy_us_per_row=float(
                    (cyc(ext[has, 32 + p]).sum() * 1e3) / rows.sum().clamp_min(1)
                )
                if has.any()
                else None,
            )
        )
    out["peers"] = peers
    return out


def run(args):
    import deep_gemm
    from mono_ep import weight_cache
    from mono_ep.runtime import PersistentMoE

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device_name = torch.cuda.get_device_name()
    if "B200" not in device_name:
        raise RuntimeError(f"Expected B200, got {device_name}")
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    experts = LOCAL_EXPERTS * world
    torch.manual_seed(812 + rank)
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        TOKENS,
        TOPK,
        HIDDEN,
        INNER,
        compute_dtype="bf16",
        retain_activations=False,
        backward_backend="native",
    )
    backend = runtime._peer_backward
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    ids = torch.rand(TOKENS, experts, device="cuda").argsort(-1)[:, :TOPK].contiguous()
    if args.routing == "skewed":
        ids[:] = torch.arange(TOPK, device="cuda")
    scores = torch.rand(TOKENS, TOPK, device="cuda").softmax(-1)
    w1 = torch.randn(LOCAL_EXPERTS, 2 * INNER, HIDDEN, device="cuda") / HIDDEN**0.5
    w2 = torch.randn(LOCAL_EXPERTS, HIDDEN, INNER, device="cuda") / INNER**0.5
    sms = deep_gemm.get_num_sms()

    records = []
    originals = {d: getattr(deep_gemm._C, NATIVE[d]) for d in NATIVE}
    original_wgrad = deep_gemm.k_grouped_bf16_gemm_tn_contiguous
    original_layouts = weight_cache.weight_layouts

    def wrap(label, fn):
        def timed(*a, **k):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            out = fn(*a, **k)
            end.record()
            records.append((label, start, end))
            return out

        return timed

    for d in NATIVE:
        setattr(deep_gemm._C, NATIVE[d], wrap("native", originals[d]))
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous = wrap("dw", original_wgrad)
    weight_cache.weight_layouts = wrap("layouts", original_layouts)
    # MXFP8 backward (MONO_EP_SM100_BACKWARD_PRECISION=mxfp8): its own launcher and weight forms
    original_mx = getattr(deep_gemm._C, "mxfp8_mono_ep_backward", None)
    original_mx_forms = getattr(weight_cache, "mxfp8_backward_weight_forms", None)
    if original_mx is not None:
        deep_gemm._C.mxfp8_mono_ep_backward = wrap("native", original_mx)
        weight_cache.mxfp8_backward_weight_forms = wrap("layouts", original_mx_forms)

    def bump():
        with torch.no_grad():
            w1.mul_(1.0)
            w2.mul_(1.0)

    def call(direction):
        if direction == "bwd":
            return backend.backward(dy, x, ids, scores, w1, w2)
        return runtime(x, ids, scores, w1, w2)

    def iteration(direction, do_bump):
        if do_bump:
            bump()
        dist.barrier()
        torch.cuda.synchronize()
        records.clear()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        call(direction)
        end.record()
        torch.cuda.synchronize()
        parts = {"whole": start.elapsed_time(end)}
        for label, a, b in records:
            parts[label] = parts.get(label, 0.0) + a.elapsed_time(b)
        parts["other"] = parts["whole"] - sum(
            parts.get(k, 0.0) for k in ("native", "dw", "layouts")
        )
        return parts

    # Kernel variants: name -> env selectors, applied to both directions (``name:K=V+K=V``)
    kernel_variants = []
    for spec in filter(None, args.variants.split(",")):
        name, _, env = spec.partition(":")
        kernel_variants.append(
            (name, dict(item.split("=", 1) for item in filter(None, env.split("+"))))
        )
    if not kernel_variants:
        kernel_variants = [("default", {})]
    managed = {key for _, env in kernel_variants for key in env}

    def apply(env):
        for key in managed:
            os.environ.pop(key, None)
        os.environ.update(env)

    def stream_native(direction, do_bump, count):
        # Back-to-back calls without the per-call barrier/synchronize (training-like queueing)
        records.clear()
        for _ in range(count):
            if do_bump:
                bump()
            call(direction)
        torch.cuda.synchronize()
        return [a.elapsed_time(b) for label, a, b in records if label == "native"]

    keys = [k for k in args.keys.split(",") if k]
    calls = [(k.split("/")[0], k.split("/")[1] == "bump") for k in keys]
    timing = {
        f"{name}|{d}/{'bump' if b else 'hit'}": []
        for name, _ in kernel_variants
        for d, b in calls
    }
    streamed = {key: [] for key in timing}
    for round_idx in range(args.rounds):
        ordered = kernel_variants if round_idx % 2 == 0 else kernel_variants[::-1]
        for name, env in ordered:
            apply(env)
            for direction, do_bump in calls if round_idx % 2 == 0 else calls[::-1]:
                key = f"{name}|{direction}/{'bump' if do_bump else 'hit'}"
                for _ in range(args.warmup):
                    iteration(direction, do_bump)
                for _ in range(args.iters):
                    timing[key].append(iteration(direction, do_bump))
                if args.stream_iters:
                    dist.barrier()
                    streamed[key].extend(
                        stream_native(direction, do_bump, args.stream_iters)
                    )
    apply({})
    # Gather per-rank part medians and the max-rank whole-call medians
    local = {
        key: {
            part: _median([s[part] for s in samples if part in s])
            for part in ("whole", "native", "dw", "layouts", "other")
        }
        for key, samples in timing.items()
    }
    whole = {
        key: torch.tensor([s["whole"] for s in samples], device="cuda")
        for key, samples in timing.items()
    }
    max_rank = {}
    for key, values in whole.items():
        gathered = [torch.empty_like(values) for _ in range(world)]
        dist.all_gather(gathered, values)
        stacked = torch.stack(gathered)
        max_rank[key] = dict(
            max_rank_median_ms=float(stacked.max(0).values.median()),
            mean_rank_median_ms=float(stacked.mean(0).median()),
            samples=int(values.numel()),
        )
    native_only = {
        key: torch.tensor([s.get("native", 0.0) for s in samples], device="cuda")
        for key, samples in timing.items()
    }
    for key, values in native_only.items():
        gathered = [torch.empty_like(values) for _ in range(world)]
        dist.all_gather(gathered, values)
        stacked = torch.stack(gathered)
        max_rank[key]["native_max_rank_median_ms"] = float(
            stacked.max(0).values.median()
        )
        max_rank[key]["native_mean_rank_median_ms"] = float(stacked.mean(0).median())
    for key, samples in streamed.items():
        if not samples:
            continue
        values = torch.tensor(samples, device="cuda")
        gathered = [torch.empty_like(values) for _ in range(world)]
        dist.all_gather(gathered, values)
        stacked = torch.stack(gathered)
        max_rank[key]["stream_native_max_rank_median_ms"] = float(
            stacked.max(0).values.median()
        )
        max_rank[key]["stream_native_mean_rank_median_ms"] = float(
            stacked.mean(0).median()
        )
        max_rank[key]["stream_samples"] = int(values.numel())

    # Traced runs (cache hit; outputs unaffected by the trace)
    traces = {}
    buffers = {
        d: torch.zeros(
            sms * (STRIDE[d] + DIST_STRIDE), device="cuda", dtype=torch.int64
        )
        for d in NATIVE
    }
    traced = [
        v
        for v in kernel_variants
        if not args.trace_variants or v[0] in args.trace_variants.split(",")
    ]
    for name, env in traced if args.trace_iters else []:
        apply(env)
        for direction in ("fwd", "bwd"):
            trace_env, ablate_env = ENV[direction]
            os.environ[trace_env] = str(buffers[direction].data_ptr())
            os.environ[ablate_env] = "1024"
            raw = []
            for i in range(args.warmup + args.trace_iters):
                buffers[direction].zero_()
                parts = iteration(direction, False)
                if i >= args.warmup:
                    raw.append((parts, buffers[direction].clone()))
            os.environ.pop(trace_env)
            os.environ.pop(ablate_env)
            per_iter = []
            for parts, buf in raw:
                cpu = buf.cpu()
                summary = summarize_trace(
                    direction, cpu[: sms * STRIDE[direction]].view(sms, -1)
                )
                dsum = _dist_summary(direction, cpu, world, sms)
                per_iter.append(dict(parts=parts, trace=summary, dist=dsum))
            traces[f"{name}|{direction}"] = per_iter
    apply({})
    rows = backend.counts.sum().reshape(1).to(torch.int64)
    all_rows = [torch.empty_like(rows) for _ in range(world)]
    dist.all_gather(all_rows, rows)
    payload = dict(
        rank=rank,
        local_timing=local,
        traces=traces,
        routed_rows=int(rows),
    )
    gathered = [None] * world
    dist.all_gather_object(gathered, payload)
    for d in NATIVE:
        setattr(deep_gemm._C, NATIVE[d], originals[d])
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous = original_wgrad
    weight_cache.weight_layouts = original_layouts
    if original_mx is not None:
        deep_gemm._C.mxfp8_mono_ep_backward = original_mx
        weight_cache.mxfp8_backward_weight_forms = original_mx_forms
    if rank == 0:
        print(
            "EPTRACE_JSON "
            + json.dumps(
                dict(
                    benchmark="b200_ep_trace",
                    device=device_name,
                    torch=torch.__version__,
                    world=world,
                    routing=args.routing,
                    tokens_per_rank=TOKENS,
                    hidden=HIDDEN,
                    intermediate=INNER,
                    experts=experts,
                    topk=TOPK,
                    num_sms=sms,
                    warmup=args.warmup,
                    iters_per_round=args.iters,
                    rounds=args.rounds,
                    trace_iters=args.trace_iters,
                    stream_iters=args.stream_iters,
                    kernel_variants=dict(kernel_variants),
                    routed_rows_per_rank=[int(r) for r in all_rows],
                    timing_max_rank=max_rank,
                    ranks=gathered,
                    scope=(
                        "timing: whole forward call (runtime) or whole native backward call "
                        "(NativePersistentBackward.backward: staging copies, weight layouts, "
                        "native kernel, dW); hit = cache hit, bump = weights bumped before "
                        "each call; dist.barrier + synchronize before every call. Traced runs "
                        "use the phase trace + distributed extension (overhead ~2-4%)."
                    ),
                )
            ),
            flush=True,
        )
    runtime.close()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--trace-iters", type=int, default=8)
    parser.add_argument(
        "--variants", default="", help="name:ENV=V+ENV=V,... (both directions)"
    )
    parser.add_argument(
        "--trace-variants", default="", help="names to trace (default all)"
    )
    parser.add_argument("--keys", default="fwd/hit,fwd/bump,bwd/hit,bwd/bump")
    parser.add_argument("--stream-iters", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
