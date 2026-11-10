# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-B200 breakdown of the NVFP4 LoRA path at the Qwen3-30B and Beam
per-GPU workloads (world 1, same rows per rank as EP8): CUDA-event timings of
the forward, the backward kernel alone, the adapter-gradient grouped GEMMs
alone and the full autograd backward; the four-form requant; a backward
pipeline-depth sweep; the kernel's per-CTA phase trace through
``probe_diag_b200.summarize_trace``; and the torch profiler's top kernels of
one backward. Writes ``NVFP4_OUT`` JSON."""

import json
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_diag_b200 import STRIDE, summarize_trace  # noqa: E402
from validate_nvfp4_forward import RANK, SCALING, routing  # noqa: E402

CASES = {
    # tokens, hidden, inter, local experts (= world-1 experts), topk
    "qwen30b": dict(tokens=16384, hidden=2048, inter=768, experts=16, topk=8),
    "beam": dict(tokens=8192, hidden=5120, inter=2560, experts=32, topk=8),
}
WARMUP, SAMPLES = 5, 15
SMEM_CAPACITY = 232448


def _align(x, a):
    return (x + a - 1) // a * a


def backward_smem(experts, hidden, pull, stages):
    slots = 1 if pull == hidden * 4 else 2
    off = _align(experts * 4, 1024) + 4 * pull
    off = _align(off, 1024) + 16384
    off = _align(off, 128) + 32768 + 512
    off = _align(off, 1024) + stages * 8192
    off = _align(off, 1024) + stages * 16384 + stages * 4096
    off = _align(off, 16) + 64 + (4 * slots + 24 + 2 * stages) * 8 + 4
    return _align(off, 1024)


def max_backward_stages(experts, hidden, pull):
    n = 2
    while n < 8 and backward_smem(experts, hidden, pull, n + 1) <= SMEM_CAPACITY:
        n += 1
    return n


def valid_pull(hidden, pull):
    if pull == hidden * 4:
        return True
    chunk = pull // 4
    return (
        pull % 4 == 0
        and chunk % 128 == 0
        and chunk <= 4096
        and chunk < hidden * 2
        and (hidden * 2) % chunk == 0
    )


def default_pull(experts, hidden):
    """Mirrors the launcher default: whole x and dy rows."""
    return hidden * 4


def timed(fn, warmup=WARMUP, samples=SAMPLES):
    out = []
    for i in range(warmup + samples):
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize()
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        if i >= warmup:
            out.append(a.elapsed_time(b))
    return statistics.median(out)


def run_case(name, case, device):
    from mono_ep.nvfp4_lora import (
        Nvfp4LoraExperts,
        Nvfp4PersistentBackward,
        RequantCache,
        backward_for,
        lora_weight_grads,
        nvfp4_lora_moe,
        requant_modules,
    )
    from mono_ep.runtime import PersistentMoE

    tokens, hidden, inter, experts, topk = (
        case[k] for k in ("tokens", "hidden", "inter", "experts", "topk")
    )
    requant = requant_modules().requant
    gen = torch.Generator(device=device).manual_seed(77)
    w1 = torch.randn(experts, 2 * inter, hidden, device=device, generator=gen) * 0.02
    w2 = torch.randn(experts, hidden, inter, device=device, generator=gen) * 0.02
    pts1 = requant.per_tensor_scale_from_amax(w1.abs().amax(dim=(1, 2)))
    pts2 = requant.per_tensor_scale_from_amax(w2.abs().amax(dim=(1, 2)))

    def quantize_base(w, pts):
        e, n, k = w.shape
        packed, scale = requant.quantize_nvfp4_packed_ref(
            w.reshape(e * n, k), pts.repeat_interleave(n)
        )
        return packed.view(e, n, k // 2), scale.view(e, n, k // 16)

    q1, s1 = quantize_base(w1, pts1)
    q2, s2 = quantize_base(w2, pts2)
    del w1, w2
    layer = Nvfp4LoraExperts(q1, s1, pts1, q2, s2, pts2, scaling=SCALING)
    a1 = (
        (torch.randn(experts, RANK, hidden, device=device, generator=gen) / hidden**0.5)
        .bfloat16()
        .requires_grad_()
    )
    b1 = (
        (torch.randn(experts, 2 * inter, RANK, device=device, generator=gen) * 0.02)
        .bfloat16()
        .requires_grad_()
    )
    a2 = (
        (torch.randn(experts, RANK, inter, device=device, generator=gen) / inter**0.5)
        .bfloat16()
        .requires_grad_()
    )
    b2 = (
        (torch.randn(experts, hidden, RANK, device=device, generator=gen) * 0.02)
        .bfloat16()
        .requires_grad_()
    )
    adapters = (a1, b1, a2, b2)
    x = (
        torch.randn(tokens, hidden, device=device, generator=gen)
        .bfloat16()
        .requires_grad_()
    )
    dy = torch.randn(tokens, hidden, device=device, generator=gen).bfloat16()
    ids, scores = routing(tokens, experts, topk, 0.0, device, gen)
    scores.requires_grad_()
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inter,
        compute_dtype="nvfp4_lora",
    )
    cache = RequantCache()
    merged = cache.get(layer, adapters)
    result = dict(case=name, geometry=dict(case, lora_rank=RANK), sms=runtime.num_sms)

    def forward():
        return nvfp4_lora_moe(x, ids, scores, *adapters, runtime, layer, cache)

    def full_backward():
        y = forward()
        torch.cuda.synchronize()
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        a.record()
        y.backward(dy)
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b)

    with torch.no_grad():
        pass
    result["forward_ms"] = timed(forward)
    full = [full_backward() for _ in range(WARMUP + SAMPLES)][WARMUP:]
    result["backward_autograd_ms"] = statistics.median(full)
    backward = backward_for(runtime)
    xd, sd = x.detach(), scores.detach()
    result["backward_kernel_ms"] = timed(
        lambda: backward.backward(dy, xd, ids, sd, merged)
    )
    result["lora_wgrad_grouped_mm_ms"] = timed(
        lambda: lora_weight_grads(
            backward.pools, backward.starts, backward.counts, *adapters, SCALING
        )
    )
    os.environ["MONO_EP_NVFP4_LORA_WGRAD"] = "loop"
    result["lora_wgrad_loop_ms"] = timed(
        lambda: lora_weight_grads(
            backward.pools, backward.starts, backward.counts, *adapters, SCALING
        )
    )
    os.environ.pop("MONO_EP_NVFP4_LORA_WGRAD")
    result["requant_4_forms_ms"] = timed(lambda: layer.merged(*adapters))
    result["pool_rows"] = backward.rows
    pull = default_pull(experts, hidden)
    max_stages = max_backward_stages(experts, hidden, pull)
    result["backward_pull_bytes"] = pull
    result["backward_max_stages"] = max_stages
    result["backward_whole_row_pull_stages"] = max_backward_stages(
        experts, hidden, hidden * 4
    )
    sweep = {}
    for stages in range(2, max_stages + 1):
        os.environ["MONO_EP_SM100_NVFP4_BACKWARD_STAGES"] = str(stages)
        sweep[stages] = timed(lambda: backward.backward(dy, xd, ids, sd, merged))
    os.environ.pop("MONO_EP_SM100_NVFP4_BACKWARD_STAGES")
    result["backward_kernel_ms_by_stages"] = sweep
    result["ring_tokens"] = backward.buffer.l1_acts.shape[0]
    # Pull-size sweep (whole rows and the chunked pulls) at each pull's deepest pipeline, dx against the default
    dx_ref, ds_ref = backward.backward(dy, xd, ids, sd, merged)
    pull_sweep = {}
    for candidate in (hidden * 4, 8192, 4096, 2048):
        if not valid_pull(hidden, candidate) or candidate in pull_sweep:
            continue
        os.environ["MONO_EP_SM100_NVFP4_BACKWARD_PULL_BYTES"] = str(candidate)
        ms = timed(lambda: backward.backward(dy, xd, ids, sd, merged))
        dx_c, ds_c = backward.backward(dy, xd, ids, sd, merged)
        os.environ.pop("MONO_EP_SM100_NVFP4_BACKWARD_PULL_BYTES")
        pull_sweep[candidate] = dict(
            ms=ms,
            stages=max_backward_stages(experts, hidden, candidate),
            dx_bitwise=bool(torch.equal(dx_c, dx_ref)),
            ds_max_abs_diff=(ds_c - ds_ref).abs().max().item(),
        )
    result["backward_kernel_by_pull"] = pull_sweep
    # Ring-floor sweep: waves of CTA pairs the ring holds (0 = the forward's interleaved-schedule sizing)
    ring_sweep = {}
    for waves in (0, 1, 2, 3, 4):
        os.environ["MONO_EP_NVFP4_BACKWARD_RING_WAVES"] = str(waves)
        other = Nvfp4PersistentBackward(runtime)
        os.environ.pop("MONO_EP_NVFP4_BACKWARD_RING_WAVES")
        ms = timed(lambda o=other: o.backward(dy, xd, ids, sd, merged))
        dx_o, ds_o = other.backward(dy, xd, ids, sd, merged)
        ring_sweep[waves] = dict(
            ring_tokens=other.buffer.l1_acts.shape[0],
            ms=ms,
            dx_bitwise=bool(torch.equal(dx_o, dx_ref)),
            ds_max_abs_diff=(ds_o - ds_ref).abs().max().item(),
        )
        other.buffer.destroy()
        del other
        torch.cuda.empty_cache()
    result["backward_kernel_by_ring_waves"] = ring_sweep
    # Phase trace of the backward kernel (default stages)
    sms = runtime.num_sms
    buf = torch.zeros(sms * (STRIDE["bwd"] + 64), device=device, dtype=torch.int64)
    os.environ["MONO_EP_SM100_PHASE_TRACE"] = str(buf.data_ptr())
    backward.backward(dy, xd, ids, sd, merged)
    torch.cuda.synchronize()
    buf.zero_()
    traced_ms = timed(
        lambda: backward.backward(dy, xd, ids, sd, merged), warmup=0, samples=1
    )
    buf.zero_()
    backward.backward(dy, xd, ids, sd, merged)
    torch.cuda.synchronize()
    os.environ.pop("MONO_EP_SM100_PHASE_TRACE")
    result["backward_kernel_traced_ms"] = traced_ms
    result["backward_phase_trace"] = summarize_trace(
        "bwd", buf.cpu()[: sms * STRIDE["bwd"]].view(sms, -1)
    )
    # Profiler table of one full backward (kernel + adapter grads)
    from torch.profiler import ProfilerActivity, profile

    y = forward()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        y.backward(dy)
        torch.cuda.synchronize()
    rows = sorted(
        (e for e in prof.key_averages() if e.device_time_total > 0),
        key=lambda e: e.device_time_total,
        reverse=True,
    )[:15]
    result["backward_top_kernels_us"] = [
        dict(name=e.key[:90], total_us=round(e.device_time_total, 1), count=e.count)
        for e in rows
    ]
    runtime.close()
    return result


def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    assert "B200" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
    import deep_gemm

    deep_gemm.set_num_sms(torch.cuda.get_device_properties(rank).multi_processor_count)
    selected = os.environ.get("NVFP4_CASES", ",".join(CASES)).split(",")
    results = []
    for name in selected:
        results.append(run_case(name, CASES[name], device))
        print(json.dumps(results[-1], indent=1), flush=True)
    out = os.environ.get("NVFP4_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
