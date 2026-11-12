# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""EP8 timing of the native persistent backward with the owned-backward ring
floor off (``MONO_EP_BACKWARD_RING_WAVES=0``) against the code default, on one
eight-GPU node (SM100 or SM90). Both runtimes share weights, inputs and routing;
backward and checkpointed 2F+B are interleaved per iteration, order reversed
every other iteration; 5 warmups then 15 samples, max-rank and mean-rank
medians. ``RING_BENCH_CASE`` selects ``qwen30b`` or ``beam``,
``RING_BENCH_DTYPE`` ``bf16``, ``blockwise`` or ``blockwise_fp8_transport``
(SM90). Writes ``NVFP4_OUT`` JSON on rank 0."""

import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_nvfp4_lora_b200 import CASES, _make_run, _median  # noqa: E402
from validate_nvfp4_forward import routing  # noqa: E402

WARMUP, SAMPLES = 5, 15
DTYPES = {
    "bf16": ("bf16", {}),
    "blockwise": ("blockwise", {"MONO_EP_SM90_FP8_TRANSPORT": "0"}),
    "blockwise_fp8_transport": (
        "blockwise",
        {"MONO_EP_SM90_FP8_TRANSPORT": "1", "MONO_EP_SM90_FP8_COMBINE": "0"},
    ),
}


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    assert world == 8
    import deep_gemm
    from mono_ep.runtime import PersistentMoE

    case_name = os.environ.get("RING_BENCH_CASE", "qwen30b")
    dtype_name = os.environ.get("RING_BENCH_DTYPE", "bf16")
    compute_dtype, dtype_env = DTYPES[dtype_name]
    case = CASES[case_name]
    tokens, hidden, inter, experts, topk = (
        case[k] for k in ("tokens", "hidden", "inter", "experts", "topk")
    )
    local = experts // world
    deep_gemm.set_num_sms(torch.cuda.get_device_properties(rank).multi_processor_count)
    gen = torch.Generator(device=device).manual_seed(812 + rank)
    w1 = (
        (
            torch.randn(local, 2 * inter, hidden, device=device, generator=gen)
            / hidden**0.5
        )
        .bfloat16()
        .requires_grad_()
    )
    w2 = (
        (torch.randn(local, hidden, inter, device=device, generator=gen) / inter**0.5)
        .bfloat16()
        .requires_grad_()
    )

    os.environ.pop("MONO_EP_BACKWARD_RING_WAVES", None)
    os.environ.update(dtype_env)
    runtimes = {}
    # RING_BENCH_VARIANTS=ring0 or =default builds one runtime only (no bitwise check), which
    # keeps the larger shapes inside 80 GB.
    selected = os.environ.get("RING_BENCH_VARIANTS", "ring0,default").split(",")
    for name, waves in (("ring0", "0"), ("default", None)):
        if name not in selected:
            continue
        if waves is not None:
            os.environ["MONO_EP_BACKWARD_RING_WAVES"] = waves
        runtimes[name] = PersistentMoE(
            dist.group.WORLD,
            experts,
            tokens,
            topk,
            hidden,
            inter,
            compute_dtype=compute_dtype,
            fuse_activation=True,
            backward_backend="native",
        )
        os.environ.pop("MONO_EP_BACKWARD_RING_WAVES", None)
    variants = {
        name: (lambda x, ids, s, rt=rt: rt(x, ids, s, w1, w2))
        for name, rt in runtimes.items()
    }
    scopes = ("backward", "checkpointed_2fb")
    x = (
        torch.randn(tokens, hidden, device=device, generator=gen)
        .bfloat16()
        .requires_grad_()
    )
    dy = torch.randn(tokens, hidden, device=device, generator=gen).bfloat16()
    rank_elapsed = torch.empty(world, device=device)
    results, peak, grads_match = {}, {}, {}

    def timed(fn):
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        elapsed = torch.tensor([(time.perf_counter() - start) * 1000], device=device)
        dist.all_gather_into_tensor(rank_elapsed, elapsed)
        return rank_elapsed.tolist(), torch.cuda.max_memory_allocated()

    for route in ("balanced", "skewed"):
        ids, scores = routing(
            tokens, experts, topk, 0.5 if route == "skewed" else 0.0, device, gen
        )
        scores = scores.requires_grad_()
        grads = {}
        for name, fn in variants.items():
            for t in (x, scores, w1, w2):
                t.grad = None
            fn(x, ids, scores).backward(dy)
            grads[name] = [t.grad.clone() for t in (x, scores, w1, w2)]
        if len(grads) == 2:
            same = torch.tensor(
                [
                    int(torch.equal(a, b))
                    for a, b in zip(grads["ring0"], grads["default"], strict=True)
                ],
                device=device,
            )
            dist.all_reduce(same, op=dist.ReduceOp.MIN)
            grads_match[route] = dict(
                zip(("dx", "ds", "dw1", "dw2"), map(bool, same.tolist()), strict=True)
            )
        del grads
        combos = [(v, s) for v in variants for s in scopes]
        samples = {c: [] for c in combos}
        for iteration in range(WARMUP + SAMPLES):
            for variant, scope in combos if iteration % 2 == 0 else combos[::-1]:
                fn = variants[variant]
                for t in (x, scores, w1, w2):
                    t.grad = None
                y = fn(x, ids, scores) if scope == "backward" else None
                ranks, mem = timed(_make_run(scope, fn, x, ids, scores, y, dy))
                if iteration >= WARMUP:
                    samples[(variant, scope)].append(ranks)
                    peak[variant] = max(peak.get(variant, 0), mem)
                assert x.grad is not None and torch.isfinite(x.grad).all()
                del y
        results[route] = {
            f"{v}/{s}": dict(
                max_rank_median_ms=_median([max(r) for r in rs]),
                mean_rank_median_ms=_median([statistics.mean(r) for r in rs]),
                samples=len(rs),
            )
            for (v, s), rs in samples.items()
        }
    mem = torch.tensor([peak[v] for v in variants], device=device, dtype=torch.int64)
    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    summary = dict(
        benchmark="backward_ring_floor_ep8",
        case=case_name,
        dtype=dtype_name,
        geometry=dict(world=world, local_experts=local, **case),
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        active_sms=next(iter(runtimes.values())).num_sms,
        warmup=WARMUP,
        samples=SAMPLES,
        env={k: v for k, v in os.environ.items() if k.startswith("MONO_EP_")},
        backward_ring_tokens={
            k: rt._peer_backward.buffer.l1_acts.shape[0] for k, rt in runtimes.items()
        },
        grads_bitwise_ring0_vs_default=grads_match or None,
        peak_memory_gib={
            v: m / 2**30 for v, m in zip(variants, mem.tolist(), strict=True)
        },
        timings=results,
    )
    if rank == 0:
        print("BENCH_JSON " + json.dumps(summary), flush=True)
        out = os.environ.get("NVFP4_OUT")
        if out:
            with open(out, "w") as f:
                json.dump(summary, f, indent=1)
    dist.barrier()
    for rt in runtimes.values():
        rt.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
