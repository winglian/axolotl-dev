# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-B200 GEMM-floor reference for the NVFP4 LoRA expert path: the cuDNN
frontend's grouped block-scaled GEMM (``grouped_gemm_quant_wrapper_sm100``,
CuTe DSL, NVFP4 A/B, e4m3 scales per 16, BF16 out) on the four grouped GEMMs
of one layer at the per-GPU EP8 workloads (world 1, same rows per rank as the
profile harness):

- ``l1``: ``x @ w1^T``, x static-scale NVFP4, merged ``w1`` (interleaved rows)
- ``l2``: ``h @ w2^T``, h static-scale NVFP4, merged ``w2``
- ``dh``: ``dy @ w2``, dy per-row NVFP4 (``row_scale``), merged ``w2t``
- ``dx``: ``dz @ w1``, dz per-row NVFP4 (``row_scale``), merged ``w1t``

Weights and their scale atoms are the requant cache's bytes as is (the
``[E, N/128, K/64, 512]`` atoms are the API's MMA-packed SFB); activation
scales are packed into the same atom layout over the 256-row padded groups.
Each GEMM sweeps the API's tile / cluster / scheduler options and checks the
output against a BF16 ``torch._grouped_mm`` on the dequantized operands, whose
time is also reported (the fp4_cute + all-to-all baseline's backward GEMMs).
Same allocation: fp4_cute's two forward GEMMs on the same rows, and mono-ep's
persistent forward and backward kernels with their phase-trace k-loops.

Timing: CUDA events around one call queued behind a ``torch.cuda._sleep`` so
host launch overhead is not in the interval; median of ``SAMPLES``. The
wrapper's host time per call is reported separately. Writes ``NVFP4_OUT``."""

import importlib.metadata
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_diag_b200 import STRIDE, summarize_trace  # noqa: E402
from validate_nvfp4_forward import RANK, SCALING, _sonicmoe, routing  # noqa: E402

CASES = {
    # tokens, hidden, intermediate, local experts (= world-1 experts), topk
    "qwen30b": dict(tokens=16384, hidden=2048, inter=768, experts=16, topk=8),
    "beam": dict(tokens=8192, hidden=5120, inter=2560, experts=32, topk=8),
}
ROUTES = {"balanced": 0.0, "skewed": 0.5}
WARMUP, SAMPLES = 5, 30
M_ALIGNED = 256
SLEEP_CYCLES = 2_000_000
# (mma_tiler_mn, cluster_shape_mn, use_dynamic_sched); the first is the API default
CONFIGS = (
    ((256, 256), (2, 1), False),
    ((256, 256), (2, 1), True),
    ((256, 256), (2, 2), False),
    ((128, 256), (1, 1), False),
    ((128, 256), (2, 1), False),
    ((128, 256), (1, 2), False),
)


def timed(fn, setup=None, warmup=WARMUP, samples=SAMPLES):
    out = []
    for i in range(warmup + samples):
        if setup is not None:
            setup()
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize()
        torch.cuda._sleep(SLEEP_CYCLES)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        if i >= warmup:
            out.append(a.elapsed_time(b))
    return statistics.median(out)


def host_us(fn, n=20):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    host = (time.perf_counter() - start) / n * 1e6
    torch.cuda.synchronize()
    return host


def rel_fro(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def config_name(cfg):
    (tm, tn), (cm, cn), dyn = cfg
    return f"tile{tm}x{tn}_cluster{cm}x{cn}" + ("_dynsched" if dyn else "")


class Operand:
    """One NVFP4 activation operand over the padded groups: packed bytes, row-major
    scales for the reference, the API's packed SFA, and per-row global scales."""

    def __init__(self, rows, requant, per_row):
        amax = rows.float().abs().amax(1)
        if per_row:
            # Kernel's dynamic per-row scale for dy / dz: amax / (6 * 448)
            self.pts = torch.where(amax > 0, amax / 2688.0, torch.ones_like(amax))
        else:
            # Static input scale (the checkpoint's pts_x / pts_h; 1 in the validators)
            self.pts = torch.ones_like(amax)
        packed, scale = requant.quantize_nvfp4_packed_ref(rows, self.pts)
        self.packed = packed.contiguous()
        self.scale = scale.contiguous()
        self.sfa = (
            requant_sf_layout()
            .pack_scales_blocked(scale.view(torch.uint8))
            .view(torch.float8_e4m3fn)
            .contiguous()
        )
        self.row_scale = self.pts.float().contiguous() if per_row else None
        self.reference = requant.dequantize_nvfp4_packed_ref(
            self.packed, self.scale, self.pts, torch.bfloat16
        )


def requant_sf_layout():
    from mono_ep.nvfp4_requant import sf_layout

    return sf_layout


def weight_operand(result, pts, requant):
    e, n, k2 = result.qdata.shape
    deq = requant.dequantize_nvfp4_packed_ref(
        result.qdata.view(e * n, k2),
        result.scale.view(e * n, k2 // 8),
        pts.repeat_interleave(n),
        torch.bfloat16,
    ).view(e, n, 2 * k2)
    return dict(
        b=result.qdata.contiguous().view(torch.float4_e2m1fn_x2),
        sfb=result.scale_atoms.contiguous().view(torch.float8_e4m3fn),
        reference=deq,
        n=n,
        k=2 * k2,
    )


def cudnn_gemm_case(gemm, act, weight, alpha, offsets, routed_rows):
    from cudnn import grouped_gemm_quant_wrapper_sm100

    m = act.packed.shape[0]
    n, k = weight["n"], weight["k"]
    ref = torch._grouped_mm(
        act.reference, weight["reference"].transpose(1, 2), offs=offsets
    )
    d = torch.empty(m, n, device=offsets.device, dtype=torch.bfloat16)
    counter = torch.zeros(1, device=offsets.device, dtype=torch.int32)
    a_fp4 = act.packed.view(torch.float4_e2m1fn_x2)
    out = dict(
        gemm=gemm,
        padded_rows=m,
        routed_rows=routed_rows,
        n=n,
        k=k,
        tflop_padded=2 * m * n * k / 1e12,
        bf16_grouped_mm_ms=timed(
            lambda: torch._grouped_mm(
                act.reference, weight["reference"].transpose(1, 2), offs=offsets
            )
        ),
        configs={},
    )
    for cfg in CONFIGS:
        tiler, cluster, dyn = cfg

        def call(tiler=tiler, cluster=cluster, dyn=dyn):
            return grouped_gemm_quant_wrapper_sm100(
                a_tensor=a_fp4,
                sfa_tensor=act.sfa,
                padded_offsets=offsets,
                alpha_tensor=alpha,
                b_tensor=weight["b"],
                sfb_tensor=weight["sfb"],
                row_scale_tensor=act.row_scale,
                d_dtype=torch.bfloat16,
                d_tensor=d,
                mma_tiler_mn=tiler,
                cluster_shape_mn=cluster,
                sf_vec_size=16,
                m_aligned=M_ALIGNED,
                use_dynamic_sched=dyn,
                scheduler_counter_tensor=counter if dyn else None,
                generate_amax=False,
            )

        setup = counter.zero_ if dyn else None
        entry = {}
        try:
            start = time.perf_counter()
            if setup:
                setup()
            call()
            torch.cuda.synchronize()
            entry["first_call_s"] = time.perf_counter() - start
            entry["rel_fro_vs_bf16_dequant"] = rel_fro(d, ref)
            entry["ms"] = timed(call, setup)
            entry["tflops_padded"] = out["tflop_padded"] / entry["ms"] * 1e3
            entry["host_us_per_call"] = host_us(call)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"[:400]
        out["configs"][config_name(cfg)] = entry
        print(gemm, config_name(cfg), entry, flush=True)
    ok = {
        k: v
        for k, v in out["configs"].items()
        if "ms" in v and v["rel_fro_vs_bf16_dequant"] < 1e-2
    }
    if ok:
        best = min(ok, key=lambda k: ok[k]["ms"])
        out["best_config"], out["best_ms"] = best, ok[best]["ms"]
        out["best_tflops_padded"] = ok[best]["tflops_padded"]
    return out


def mono_ep_reference(runtime, layer, adapters, cache, x, ids, scores, dy, device):
    from mono_ep.nvfp4_lora import backward_for, nvfp4_lora_moe

    merged = cache.get(layer, adapters)
    xd, sd = x.detach(), scores.detach()
    sms = runtime.num_sms
    result = {}
    with torch.no_grad():
        result["forward_ms"] = timed(
            lambda: nvfp4_lora_moe(xd, ids, sd, *adapters, runtime, layer, cache)
        )
    backward = backward_for(runtime)
    result["backward_kernel_ms"] = timed(
        lambda: backward.backward(dy, xd, ids, sd, merged)
    )
    result["pool_rows"] = backward.rows
    for direction, env, run in (
        (
            "fwd",
            "MONO_EP_SM100_NVFP4_PHASE_TRACE",
            lambda: runtime.forward_nvfp4(
                xd,
                ids,
                sd,
                merged.w1,
                merged.w1_atoms,
                merged.w2,
                merged.w2_atoms,
                merged.scales_forward,
            ),
        ),
        (
            "bwd",
            "MONO_EP_SM100_PHASE_TRACE",
            lambda: backward.backward(dy, xd, ids, sd, merged),
        ),
    ):
        buf = torch.zeros(
            sms * (STRIDE[direction] + 64), device=device, dtype=torch.int64
        )
        os.environ[env] = str(buf.data_ptr())
        try:
            run()
            torch.cuda.synchronize()
            buf.zero_()
            traced = timed(run, warmup=0, samples=1)
            buf.zero_()
            run()
            torch.cuda.synchronize()
            summary = summarize_trace(
                direction, buf.cpu()[: sms * STRIDE[direction]].view(sms, -1)
            )
            result[f"{direction}_traced_ms"] = traced
            result[f"{direction}_wall_ms"] = summary["median_over_ctas_ms"][
                "kernel_wall"
            ]
            result[f"{direction}_mma"] = summary["median_over_leader_ctas"]
        except Exception as exc:  # noqa: BLE001
            result[f"{direction}_trace_error"] = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            os.environ.pop(env)
    return result


def fp4_cute_reference(layer, adapters, r2, xg, hg, offsets_i64, padded):
    sm = _sonicmoe()
    import importlib

    fp4_cute = importlib.import_module(f"{sm.name}.fp4_cute")
    ops = importlib.import_module(f"{sm.name}.fp4_cute_ops")
    a1, b1 = adapters[:2]
    r1c = layer._requant(layer.q1, layer.s1, layer.pts1, a1, b1)
    h, i, e = layer.hidden, layer.inter, layer.experts
    l1 = fp4_cute.GroupedNvfp4Gemm(
        2 * i, h, e, gated=True, concat_b=True, store_preact=True
    )
    l2 = fp4_cute.GroupedNvfp4Gemm(h, i, e)
    l1.set_weights(r1c.qdata, r1c.scale)
    l2.set_weights(r2.qdata, r2.scale)
    a_q, sfa = ops.quantize_grouped_rows(xg, offsets_i64)
    h_q, sfh = ops.quantize_grouped_rows(hg, offsets_i64)
    colvec1 = layer.pts1.repeat_interleave(padded).float()
    colvec2 = layer.pts2.repeat_interleave(padded).float()
    return dict(
        l1_gated_swiglu_ms=timed(
            lambda: l1.forward(a_q, sfa, offsets_i64, colvec=colvec1)
        ),
        l2_ms=timed(lambda: l2.forward(h_q, sfh, offsets_i64, colvec=colvec2)),
    )


def run_case(name, case, device):
    from mono_ep.nvfp4_lora import Nvfp4LoraExperts, RequantCache, requant_modules
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
    adapters = (
        (
            torch.randn(experts, RANK, hidden, device=device, generator=gen)
            / hidden**0.5
        ).bfloat16(),
        (
            torch.randn(experts, 2 * inter, RANK, device=device, generator=gen) * 0.02
        ).bfloat16(),
        (
            torch.randn(experts, RANK, inter, device=device, generator=gen) / inter**0.5
        ).bfloat16(),
        (
            torch.randn(experts, hidden, RANK, device=device, generator=gen) * 0.02
        ).bfloat16(),
    )
    r1, r2, r1t, r2t = layer.requant_all(*adapters)
    weights = dict(
        l1=weight_operand(r1, layer.pts1, requant),
        l2=weight_operand(r2, layer.pts2, requant),
        dh=weight_operand(r2t, layer.pts2, requant),
        dx=weight_operand(r1t, layer.pts1, requant),
    )
    # Static input scales are 1 (the validators' pts_x = pts_h), so the forward alphas are the weights' pts
    alphas = dict(l1=layer.pts1, l2=layer.pts2, dh=layer.pts2, dx=layer.pts1)
    x = torch.randn(tokens, hidden, device=device, generator=gen).bfloat16()
    dy = torch.randn(tokens, hidden, device=device, generator=gen).bfloat16()
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
    result = dict(
        case=name, geometry=dict(case, lora_rank=RANK), sms=runtime.num_sms, routes={}
    )
    for route, skew in ROUTES.items():
        ids, scores = routing(tokens, experts, topk, skew, device, gen)
        flat = ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = order // topk
        counts = torch.bincount(flat, minlength=experts)
        padded = (counts + M_ALIGNED - 1) // M_ALIGNED * M_ALIGNED
        ends = padded.cumsum(0)
        starts = ends - padded
        n_rows = int(counts.sum())
        total = int(ends[-1])
        pos = (starts - (counts.cumsum(0) - counts)).repeat_interleave(
            counts
        ) + torch.arange(n_rows, device=device)

        def grouped(src, total=total, pos=pos):
            return src.new_zeros(total, src.shape[1]).index_copy_(0, pos, src)

        xg = grouped(x.index_select(0, tok))
        dyg = grouped(dy.index_select(0, tok))
        hg = grouped(
            torch.randn(n_rows, inter, device=device, generator=gen).bfloat16()
        )
        dzg = grouped(
            torch.randn(n_rows, 2 * inter, device=device, generator=gen).bfloat16()
        )
        acts = dict(
            l1=Operand(xg, requant, per_row=False),
            l2=Operand(hg, requant, per_row=False),
            dh=Operand(dyg, requant, per_row=True),
            dx=Operand(dzg, requant, per_row=True),
        )
        offsets = ends.to(torch.int32)
        offsets_i64 = torch.cat([ends.new_zeros(1), ends])
        entry = dict(
            routed_rows=n_rows,
            padded_rows=total,
            max_expert_rows=int(counts.max()),
            min_expert_rows=int(counts.min()),
            gemms={},
        )
        for gemm in ("l1", "l2", "dh", "dx"):
            entry["gemms"][gemm] = cudnn_gemm_case(
                gemm, acts[gemm], weights[gemm], alphas[gemm], offsets, n_rows
            )
        del acts
        try:
            entry["fp4_cute"] = fp4_cute_reference(
                layer, adapters, r2, xg, hg, offsets_i64, padded
            )
        except Exception as exc:  # noqa: BLE001
            entry["fp4_cute"] = dict(error=f"{type(exc).__name__}: {exc}"[:400])
        entry["mono_ep"] = mono_ep_reference(
            runtime, layer, adapters, cache, x, ids, scores, dy, device
        )
        gemms = entry["gemms"]
        best = {g: gemms[g].get("best_ms") for g in gemms}
        if all(v is not None for v in best.values()):
            # The backward recomputes L1 (RecomputeLow + RecomputeHigh), so its GEMM floor is dh + l1 + dx
            entry["cudnn_forward_floor_ms"] = best["l1"] + best["l2"]
            entry["cudnn_backward_floor_ms"] = best["dh"] + best["l1"] + best["dx"]
        result["routes"][route] = entry
        print(
            json.dumps(
                {route: {k: v for k, v in entry.items() if k != "gemms"}}, indent=1
            ),
            flush=True,
        )
        torch.cuda.empty_cache()
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
    versions = {}
    for pkg in (
        "nvidia-cudnn-frontend",
        "nvidia-cutlass-dsl",
        "torch",
        "quack-kernels",
    ):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    print("versions", versions, flush=True)
    selected = os.environ.get("NVFP4_CASES", ",".join(CASES)).split(",")
    results = dict(
        benchmark="cudnn_grouped_gemm_b200",
        device=torch.cuda.get_device_name(),
        versions=versions,
        warmup=WARMUP,
        samples=SAMPLES,
        cases=[],
    )
    for name in selected:
        results["cases"].append(run_case(name, CASES[name], device))
    out = os.environ.get("NVFP4_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
