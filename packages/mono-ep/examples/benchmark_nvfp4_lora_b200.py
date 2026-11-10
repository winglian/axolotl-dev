# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""8xB200 EP8 timing of the NVFP4 LoRA persistent path against the BF16
persistent path (full fine-tune, native backward) and an fp4_cute
requant-cache baseline behind torch all-to-all dispatch/combine.

Per routing (balanced, skewed) the variants and scopes (forward, backward,
checkpointed 2F+B) are interleaved per iteration, order reversed every other
iteration; 5 warmups then 15 samples. Reports max-rank and mean-rank medians,
peak memory, and the per-step weight rebuild cost of each LoRA variant
(requant of the merged forms; for the baseline also the BF16 dequant its
backward uses). ``NVFP4_BENCH_CASE`` selects ``qwen30b`` or ``beam``."""

import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_nvfp4_forward import routing  # noqa: E402

CASES = {
    # tokens per GPU, hidden, intermediate, global experts, top-k
    "qwen30b": dict(tokens=16384, hidden=2048, inter=768, experts=128, topk=8),
    "beam": dict(tokens=8192, hidden=5120, inter=2560, experts=256, topk=8),
}
RANK, SCALING = 16, 2.0
WARMUP, SAMPLES = 5, 15
BLOCK = 128


class Fp4CuteMergedExperts(torch.autograd.Function):
    """Local experts on fp4_cute with the requant cache's merged bytes; backward
    through BF16 dequantized merged weights (axolotl's ``grouped_dx_dequant``)
    and grouped GEMMs for the adapter gradients."""

    @staticmethod
    def forward(ctx, xg, cu, a1, b1, a2, b2, eng):
        from sonicmoe_lite.fp4_cute_ops import quantize_grouped_rows

        a_q, sfa = quantize_grouped_rows(xg, cu)
        post, z = eng.l1.forward(a_q, sfa, cu, colvec=eng.colvec1)
        h_q, sfh = quantize_grouped_rows(post.contiguous(), cu)
        y = eng.l2.forward(h_q, sfh, cu, colvec=eng.colvec2)
        ctx.save_for_backward(xg, z, post, cu, a1, b1, a2, b2)
        ctx.eng = eng
        return y

    @staticmethod
    def backward(ctx, dy):
        xg, z, post, cu, a1, b1, a2, b2 = ctx.saved_tensors
        eng = ctx.eng
        offs = cu[1:].to(torch.int32)
        dy = dy.contiguous()
        gmm = torch._grouped_mm
        s = eng.scaling
        dh = gmm(dy, eng.w2m, offs=offs)
        p2 = gmm(dy, b2, offs=offs)
        q2 = gmm(post, a2.transpose(1, 2).contiguous(), offs=offs)
        da2 = s * gmm(p2.t(), post, offs=offs).float()
        db2 = s * gmm(dy.t(), q2, offs=offs).float()
        gate, up = z.float().chunk(2, -1)
        sig = torch.sigmoid(gate)
        dhf = dh.float()
        dz = torch.cat(
            [dhf * up * (sig + gate * sig * (1 - sig)), dhf * gate * sig], -1
        ).bfloat16()
        dx = gmm(dz, eng.w1m, offs=offs)
        p1 = gmm(dz, b1, offs=offs)
        q1 = gmm(xg, a1.transpose(1, 2).contiguous(), offs=offs)
        da1 = s * gmm(p1.t(), xg, offs=offs).float()
        db1 = s * gmm(dz.t(), q1, offs=offs).float()
        return (
            dx,
            None,
            da1.to(a1.dtype),
            db1.to(b1.dtype),
            da2.to(a2.dtype),
            db2.to(b2.dtype),
            None,
        )


def _a2a(t, output_splits, input_splits):
    out = all_to_all_single_autograd(
        t.contiguous(), output_splits, input_splits, dist.group.WORLD
    )
    return out.wait() if hasattr(out, "wait") else out


class Fp4CuteA2A:
    """torch all-to-all EP dispatch/combine around :class:`Fp4CuteMergedExperts`."""

    def __init__(self, layer, experts, topk, scaling):
        from sonicmoe_lite.fp4_cute import GroupedNvfp4Gemm

        self.layer, self.experts, self.topk, self.scaling = (
            layer,
            experts,
            topk,
            scaling,
        )
        self.world, self.rank = dist.get_world_size(), dist.get_rank()
        self.local = experts // self.world
        h, i = layer.hidden, layer.inter
        self.l1 = GroupedNvfp4Gemm(
            2 * i, h, self.local, gated=True, concat_b=True, store_preact=True
        )
        self.l2 = GroupedNvfp4Gemm(h, i, self.local)
        self.colvec1 = self.colvec2 = None

    def rebuild(self, a1, b1, a2, b2):
        layer = self.layer
        r1 = layer._requant(layer.q1, layer.s1, layer.pts1, a1, b1)
        r2 = layer._requant(layer.q2, layer.s2, layer.pts2, a2, b2)
        self.l1.set_weights(r1.qdata, r1.scale)
        self.l2.set_weights(r2.qdata, r2.scale)
        self.w1m = layer.dequantized(r1, layer.pts1).bfloat16()
        self.w2m = layer.dequantized(r2, layer.pts2).bfloat16()

    def __call__(self, x, ids, scores, a1, b1, a2, b2):
        world, local, layer = self.world, self.local, self.layer
        tokens, k = ids.shape
        flat = ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = order // k
        counts = torch.bincount(flat, minlength=self.experts)
        recv_counts = torch.empty_like(counts)
        dist.all_to_all_single(recv_counts, counts)
        send_splits = counts.view(world, local).sum(1).tolist()
        recv_splits = recv_counts.view(world, local).sum(1).tolist()
        xs = x.index_select(0, tok)
        xr = _a2a(xs, recv_splits, send_splits)
        exp_ids = (
            torch.arange(local, device=x.device)
            .repeat(world)
            .repeat_interleave(recv_counts)
        )
        order2 = torch.argsort(exp_ids, stable=True)
        counts_local = recv_counts.view(world, local).sum(0)
        padded = (counts_local + BLOCK - 1) // BLOCK * BLOCK
        cu = torch.zeros(local + 1, device=x.device, dtype=torch.int64)
        cu[1:] = padded.cumsum(0)
        cu_raw = torch.zeros_like(cu)
        cu_raw[1:] = counts_local.cumsum(0)
        n = int(cu_raw[-1])
        pos = (cu[:-1] - cu_raw[:-1]).repeat_interleave(counts_local) + torch.arange(
            n, device=x.device
        )
        total = int(cu[-1])
        xg = x.new_zeros(total, x.shape[1]).index_copy(
            0, pos, xr.index_select(0, order2)
        )
        self.colvec1 = layer.pts1.repeat_interleave(padded).float()
        self.colvec2 = layer.pts2.repeat_interleave(padded).float()
        yg = Fp4CuteMergedExperts.apply(xg, cu, a1, b1, a2, b2, self)
        inv2 = torch.empty_like(order2)
        inv2[order2] = torch.arange(order2.numel(), device=x.device)
        yr = yg.index_select(0, pos).index_select(0, inv2)
        ys = _a2a(yr, send_splits, recv_splits)
        weights = scores.reshape(-1).index_select(0, order).to(ys.dtype)
        return x.new_zeros(tokens, x.shape[1]).index_add(0, tok, ys * weights[:, None])


def _median(v):
    return statistics.median(v) if v else None


def _make_run(scope, fn, x, ids, scores, y, dy):
    if scope == "forward":
        return lambda: fn(x, ids, scores)
    if scope == "backward":
        return lambda: y.backward(dy)

    def run():
        out = checkpoint(
            fn, x, ids, scores, use_reentrant=False, preserve_rng_state=False
        )
        out.backward(dy)
        return out

    return run


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    assert "B200" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
    world = dist.get_world_size()
    assert world == 8
    import deep_gemm
    from mono_ep.nvfp4_lora import (
        Nvfp4LoraExperts,
        RequantCache,
        nvfp4_lora_moe,
        requant_modules,
    )
    from mono_ep.runtime import PersistentMoE

    case_name = os.environ.get("NVFP4_BENCH_CASE", "qwen30b")
    case = CASES[case_name]
    tokens, hidden, inter, experts, topk = (
        case[k] for k in ("tokens", "hidden", "inter", "experts", "topk")
    )
    local = experts // world
    sms = torch.cuda.get_device_properties(rank).multi_processor_count
    deep_gemm.set_num_sms(sms)
    requant = requant_modules().requant
    gen = torch.Generator(device=device).manual_seed(812 + rank)

    # Frozen NVFP4 base for the LoRA variants, BF16 trainable experts for the BF16 path
    w1 = (
        torch.randn(local, 2 * inter, hidden, device=device, generator=gen)
        / hidden**0.5
    )
    w2 = torch.randn(local, hidden, inter, device=device, generator=gen) / inter**0.5
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
    layer = Nvfp4LoraExperts(q1, s1, pts1, q2, s2, pts2, scaling=SCALING)
    w1_bf = w1.bfloat16().requires_grad_()
    w2_bf = w2.bfloat16().requires_grad_()
    del w1, w2
    a1 = (
        (torch.randn(local, RANK, hidden, device=device, generator=gen) / hidden**0.5)
        .bfloat16()
        .requires_grad_()
    )
    b1 = (
        (torch.randn(local, 2 * inter, RANK, device=device, generator=gen) * 0.02)
        .bfloat16()
        .requires_grad_()
    )
    a2 = (
        (torch.randn(local, RANK, inter, device=device, generator=gen) / inter**0.5)
        .bfloat16()
        .requires_grad_()
    )
    b2 = (
        (torch.randn(local, hidden, RANK, device=device, generator=gen) * 0.02)
        .bfloat16()
        .requires_grad_()
    )
    adapters = (a1, b1, a2, b2)

    runtime_nv = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inter,
        compute_dtype="nvfp4_lora",
    )
    runtime_bf = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inter,
        compute_dtype="bf16",
        fuse_activation=True,
        backward_backend="native",
    )
    cache = RequantCache()
    a2a = Fp4CuteA2A(layer, experts, topk, SCALING)

    variants = {
        "nvfp4_lora": lambda x, ids, s: nvfp4_lora_moe(
            x, ids, s, *adapters, runtime_nv, layer, cache
        ),
        "bf16_persistent": lambda x, ids, s: runtime_bf(x, ids, s, w1_bf, w2_bf),
        "fp4_cute_a2a": lambda x, ids, s: a2a(x, ids, s, *adapters),
    }
    scopes = ("forward", "backward", "checkpointed_2fb")
    x = (
        torch.randn(tokens, hidden, device=device, generator=gen)
        .bfloat16()
        .requires_grad_()
    )
    dy = torch.randn(tokens, hidden, device=device, generator=gen).bfloat16()
    rank_elapsed = torch.empty(world, device=device)
    results = {}
    peak = {}

    def timed(fn):
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        elapsed = torch.tensor([(time.perf_counter() - start) * 1000], device=device)
        dist.all_gather_into_tensor(rank_elapsed, elapsed)
        return out, rank_elapsed.tolist(), torch.cuda.max_memory_allocated()

    # Per-step rebuilds, timed on their own (the training flow runs them in the optimizer post-step hook)
    rebuilds = {}
    for name, fn in (
        ("nvfp4_lora/requant_4_forms", lambda: layer.merged(*adapters)),
        (
            "fp4_cute_a2a/requant_2_forms_plus_bf16_dequant",
            lambda: a2a.rebuild(*adapters),
        ),
    ):
        samples = []
        for i in range(WARMUP + SAMPLES):
            _, ranks, _ = timed(fn)
            if i >= WARMUP:
                samples.append(ranks)
        rebuilds[name] = dict(
            max_rank_median_ms=_median([max(r) for r in samples]),
            mean_rank_median_ms=_median([statistics.mean(r) for r in samples]),
        )
    cache.get(layer, adapters)
    a2a.rebuild(*adapters)

    for route in ("balanced", "skewed"):
        ids, scores = routing(
            tokens, experts, topk, 0.5 if route == "skewed" else 0.0, device, gen
        )
        scores = scores.requires_grad_()
        combos = [(v, s) for v in variants for s in scopes]
        samples = {c: [] for c in combos}
        for iteration in range(WARMUP + SAMPLES):
            for variant, scope in combos if iteration % 2 == 0 else combos[::-1]:
                fn = variants[variant]
                for t in (x, scores, w1_bf, w2_bf, *adapters):
                    t.grad = None
                y = fn(x, ids, scores) if scope == "backward" else None
                run = _make_run(scope, fn, x, ids, scores, y, dy)
                out, ranks, mem = timed(run)
                if iteration >= WARMUP:
                    samples[(variant, scope)].append(ranks)
                    peak[variant] = max(peak.get(variant, 0), mem)
                if scope != "backward" and out is not None:
                    assert torch.isfinite(out).all()
                if scope != "forward":
                    assert x.grad is not None and torch.isfinite(x.grad).all()
                del out, y
        results[route] = {
            f"{v}/{s}": dict(
                max_rank_median_ms=_median([max(r) for r in rs]),
                mean_rank_median_ms=_median([statistics.mean(r) for r in rs]),
                per_rank_median_ms=[_median([r[i] for r in rs]) for i in range(world)],
                samples=len(rs),
            )
            for (v, s), rs in samples.items()
        }
    mem = torch.tensor([peak[v] for v in variants], device=device, dtype=torch.int64)
    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    summary = dict(
        benchmark="nvfp4_lora_b200_ep8",
        case=case_name,
        geometry=dict(
            world=world,
            tokens_per_gpu=tokens,
            hidden=hidden,
            inter=inter,
            experts=experts,
            local_experts=local,
            topk=topk,
            lora_rank=RANK,
            scaling=SCALING,
        ),
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        active_sms=runtime_nv.num_sms,
        warmup=WARMUP,
        samples=SAMPLES,
        env={k: v for k, v in os.environ.items() if k.startswith("MONO_EP_")},
        rebuild_ms=rebuilds,
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
    runtime_nv.close()
    runtime_bf.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
