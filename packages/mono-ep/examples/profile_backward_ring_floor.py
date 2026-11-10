# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-GPU ring-floor sweep of the BF16/FP8 native persistent backward
(``NativePersistentBackward``) at the Qwen3-30B and Beam per-GPU workloads
(world 1, same rows per rank as EP8). For each dtype variant and
``MONO_EP_BACKWARD_RING_WAVES`` in 0..3 (plus explicit token floors from
``RING_TOKEN_FLOORS``) it builds a backward on one runtime,
times ``backward()`` with CUDA events and the native kernel alone with the
profiler, and compares dx, ds, dW1 and dW2 bitwise against waves 0. Variants:
``bf16`` on SM100; ``bf16``, ``blockwise`` and ``blockwise_fp8_transport`` on
SM90. Writes ``NVFP4_OUT`` JSON."""

import json
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_nvfp4_forward import routing  # noqa: E402

CASES = {
    # tokens, hidden, inter, local experts (= world-1 experts), topk
    "qwen30b": dict(tokens=16384, hidden=2048, inter=768, experts=16, topk=8),
    "beam": dict(tokens=8192, hidden=5120, inter=2560, experts=32, topk=8),
}
VARIANTS = {
    10: {"bf16": ("bf16", {})},
    9: {
        "bf16": ("bf16", {}),
        "blockwise": ("blockwise", {}),
        "blockwise_fp8_transport": (
            "blockwise",
            {"MONO_EP_SM90_FP8_TRANSPORT": "1"},
        ),
    },
}
WAVES = (0, 1, 2, 3)
WARMUP, SAMPLES = 5, 15
ENV_KEYS = ("MONO_EP_BACKWARD_RING_WAVES", "MONO_EP_SM90_FP8_TRANSPORT")


def timed(fn):
    out = []
    for i in range(WARMUP + SAMPLES):
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize()
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        if i >= WARMUP:
            out.append(a.elapsed_time(b))
    return statistics.median(out)


def kernel_ms(fn, iters=5):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    times = [
        e.device_time_total / e.count
        for e in prof.key_averages()
        if "mono_ep_backward" in e.key and e.count
    ]
    return max(times) / 1000 if times else None


def run_case(name, case, variant, compute_dtype, env, device):
    from mono_ep.persistent_backward import NativePersistentBackward
    from mono_ep.runtime import PersistentMoE

    tokens, hidden, inter, experts, topk = (
        case[k] for k in ("tokens", "hidden", "inter", "experts", "topk")
    )
    gen = torch.Generator(device=device).manual_seed(91)
    w1 = (
        torch.randn(experts, 2 * inter, hidden, device=device, generator=gen)
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(experts, hidden, inter, device=device, generator=gen) / inter**0.5
    ).bfloat16()
    x = torch.randn(tokens, hidden, device=device, generator=gen).bfloat16()
    dy = torch.randn(tokens, hidden, device=device, generator=gen).bfloat16()
    ids, scores = routing(tokens, experts, topk, 0.0, device, gen)
    os.environ.update(env)
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inter,
        compute_dtype=compute_dtype,
        fuse_activation=True,
    )
    result = dict(
        case=name,
        variant=variant,
        geometry=case,
        device=torch.cuda.get_device_name(),
        sms=runtime.num_sms,
        env=dict(env),
        by_waves={},
    )
    import mono_ep.persistent_backward as pb

    floors = [int(t) for t in os.environ.get("RING_TOKEN_FLOORS", "").split(",") if t]
    points = [(str(w), w, None) for w in WAVES] + [
        (f"tokens_{t}", None, t) for t in floors
    ]
    ref = None
    for waves, value, floor in points:
        if floor is None:
            os.environ["MONO_EP_BACKWARD_RING_WAVES"] = str(value)
            backward = NativePersistentBackward(runtime)
            os.environ.pop("MONO_EP_BACKWARD_RING_WAVES")
        else:
            helper = pb.owned_backward_ring_tokens
            pb.owned_backward_ring_tokens = lambda *a, t=floor, **k: t
            try:
                backward = NativePersistentBackward(runtime)
            finally:
                pb.owned_backward_ring_tokens = helper

        def step(b=backward):
            return b.backward(dy, x, ids, scores, w1, w2)

        ms = timed(step)
        kms = kernel_ms(step)
        grads = [t.clone() for t in step()]
        torch.cuda.synchronize()
        row = dict(
            ring_tokens=backward.buffer.l1_acts.shape[0],
            backward_ms=ms,
            kernel_ms=kms,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        if ref is None:
            ref = grads
            row["finite"] = all(bool(torch.isfinite(g).all()) for g in grads)
        else:
            row["bitwise"] = {
                k: bool(torch.equal(g, r))
                for k, g, r in zip(("dx", "ds", "dw1", "dw2"), grads, ref, strict=True)
            }
            row["max_abs_diff"] = {
                k: (g.float() - r.float()).abs().max().item()
                for k, g, r in zip(("dx", "ds", "dw1", "dw2"), grads, ref, strict=True)
            }
        result["by_waves"][waves] = row
        print(name, variant, waves, json.dumps(row), flush=True)
        backward.buffer.destroy()
        del backward, grads
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    runtime.close()
    runtime.buffer.destroy()
    for key in env:
        os.environ.pop(key)
    torch.cuda.empty_cache()
    return result


def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    import deep_gemm

    deep_gemm.set_num_sms(torch.cuda.get_device_properties(rank).multi_processor_count)
    arch = torch.cuda.get_device_capability()[0]
    selected = os.environ.get("RING_CASES", ",".join(CASES)).split(",")
    variants = os.environ.get("RING_VARIANTS", ",".join(VARIANTS[arch])).split(",")
    results = []
    for name in selected:
        for variant in variants:
            compute_dtype, env = VARIANTS[arch][variant]
            results.append(
                run_case(name, CASES[name], variant, compute_dtype, env, device)
            )
    out = os.environ.get("NVFP4_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
