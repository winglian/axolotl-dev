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
backward uses). ``NVFP4_BENCH_CASE`` selects ``qwen30b``, ``qwen235b`` or ``beam``."""

import importlib.util
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
from validate_nvfp4_forward import _sonicmoe, routing  # noqa: E402

CASES = {
    # tokens per GPU, hidden, intermediate, global experts, top-k
    "qwen30b": dict(tokens=16384, hidden=2048, inter=768, experts=128, topk=8),
    "qwen235b": dict(tokens=16384, hidden=4096, inter=1536, experts=128, topk=8),
    "beam": dict(tokens=8192, hidden=5120, inter=2560, experts=256, topk=8),
}
RANK, SCALING = 16, 2.0
WARMUP, SAMPLES = 5, 15
BLOCK = 128
# cuDNN frontend grouped GEMM: groups padded to m_aligned 256; tile / cluster from the
# single-B200 sweep in benchmark_cudnn_grouped_gemm_b200.py
CUDNN_BLOCK = 256
CUDNN_TILE, CUDNN_CLUSTER = (256, 256), (2, 1)


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
        # The gated engine stores the preact interleaved [g0, u0, g1, u1, ...]
        zf = z.float()
        gate, up = zf[:, 0::2], zf[:, 1::2]
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
        _sonicmoe()  # registers sonicmoe_lite from SONICMOE_DIR
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
        self.block = BLOCK

    def rebuild(self, a1, b1, a2, b2):
        layer = self.layer
        r1 = layer._requant(layer.q1, layer.s1, layer.pts1, a1, b1)
        r2 = layer._requant(layer.q2, layer.s2, layer.pts2, a2, b2)
        self.l1.set_weights(r1.qdata, r1.scale)
        self.l2.set_weights(r2.qdata, r2.scale)
        self.w1m = layer.dequantized(r1, layer.pts1).bfloat16()
        self.w2m = layer.dequantized(r2, layer.pts2).bfloat16()

    def __call__(self, x, ids, scores, a1, b1, a2, b2):
        world, local = self.world, self.local
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
        padded = (counts_local + self.block - 1) // self.block * self.block
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
        yg = self._experts(xg, cu, padded, a1, b1, a2, b2)
        inv2 = torch.empty_like(order2)
        inv2[order2] = torch.arange(order2.numel(), device=x.device)
        yr = yg.index_select(0, pos).index_select(0, inv2)
        ys = _a2a(yr, send_splits, recv_splits)
        weights = scores.reshape(-1).index_select(0, order).to(ys.dtype)
        return x.new_zeros(tokens, x.shape[1]).index_add(0, tok, ys * weights[:, None])

    def _experts(self, xg, cu, padded, a1, b1, a2, b2):
        layer = self.layer
        self.colvec1 = layer.pts1.repeat_interleave(padded).float()
        self.colvec2 = layer.pts2.repeat_interleave(padded).float()
        return Fp4CuteMergedExperts.apply(xg, cu, a1, b1, a2, b2, self)


def _quantize_rows(x, row_scale=None):
    """NVFP4 rows with the SFA in the cuDNN grouped GEMM's contiguous MMA-packed
    layout (fp4_cute's fused row quantizer with identity row placement);
    ``row_scale`` divides each row first (per-row two-level scaling, the GEMM
    multiplies it back)."""
    from sonicmoe_lite.triton_nvfp4 import _get_kernels, rowscale_inplace_triton

    x = x.contiguous()
    if row_scale is not None:
        x = rowscale_inplace_triton(x.clone(), 1.0 / row_scale)
    rows, k = x.shape
    sf_k = k // 16
    packed = torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device)
    sfa = torch.zeros(
        (rows + 127) // 128, sf_k // 4, 512, dtype=torch.uint8, device=x.device
    )
    # quantize_grouped_rows would sync on its per-expert row placement
    dest = torch.arange(rows, device=x.device, dtype=torch.int32)
    _get_kernels().quant_sfa[(rows, (sf_k + 63) // 64)](
        x, packed, sfa, dest, K=k, SF_K=sf_k, RK=sf_k // 4, BLOCK_SF=64
    )
    return packed.view(torch.float4_e2m1fn_x2), sfa.view(torch.float8_e4m3fn)


def _row_scale(t):
    """mono-ep's dynamic per-row NVFP4 scale for dy / dz: amax / (6 * 448)."""
    amax = torch.linalg.vector_norm(t, float("inf"), dim=1, dtype=torch.float32)
    return torch.where(amax > 0, amax / 2688.0, torch.ones_like(amax))


def _cudnn_gemm(a_q, sfa, offsets, alpha, form, row_scale=None):
    from cudnn import grouped_gemm_quant_wrapper_sm100

    b, sfb = form
    return grouped_gemm_quant_wrapper_sm100(
        a_tensor=a_q,
        sfa_tensor=sfa,
        padded_offsets=offsets,
        alpha_tensor=alpha,
        b_tensor=b,
        sfb_tensor=sfb,
        row_scale_tensor=row_scale,
        d_dtype=torch.bfloat16,
        mma_tiler_mn=CUDNN_TILE,
        cluster_shape_mn=CUDNN_CLUSTER,
        sf_vec_size=16,
        m_aligned=CUDNN_BLOCK,
        generate_amax=False,
    )["d_tensor"]


class CudnnMergedExperts(torch.autograd.Function):
    """Local experts on the cuDNN frontend grouped NVFP4 GEMM (four merged forms:
    concat ``w1``, ``w2``, ``w1t``, ``w2t``). Forward quantizes x and h with a
    unit static scale; backward quantizes dy and dz per row (``row_scale``) and
    runs Dh / Dx on the transposed forms; adapter gradients as the fp4_cute
    baseline (BF16 grouped GEMMs)."""

    @staticmethod
    def forward(ctx, xg, offsets, a1, b1, a2, b2, eng):
        layer = eng.layer
        a_q, sfa = _quantize_rows(xg)
        z = _cudnn_gemm(a_q, sfa, offsets, layer.pts1, eng.w1)
        gate, up = z.chunk(2, -1)
        post = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
        h_q, sfh = _quantize_rows(post)
        y = _cudnn_gemm(h_q, sfh, offsets, layer.pts2, eng.w2)
        ctx.save_for_backward(xg, z, post, offsets, a1, b1, a2, b2)
        ctx.eng = eng
        return y

    @staticmethod
    def backward(ctx, dy):
        xg, z, post, offs, a1, b1, a2, b2 = ctx.saved_tensors
        eng = ctx.eng
        layer = eng.layer
        dy = dy.contiguous()
        gmm = torch._grouped_mm
        s = eng.scaling
        rs = _row_scale(dy)
        dy_q, sfdy = _quantize_rows(dy, rs)
        dh = _cudnn_gemm(dy_q, sfdy, offs, layer.pts2, eng.w2t, rs)
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
        rz = _row_scale(dz)
        dz_q, sfdz = _quantize_rows(dz, rz)
        dx = _cudnn_gemm(dz_q, sfdz, offs, layer.pts1, eng.w1t, rz)
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


class CudnnGroupedA2A(Fp4CuteA2A):
    """The :class:`Fp4CuteA2A` dispatch/combine (groups padded to 256 rows)
    around :class:`CudnnMergedExperts`."""

    def __init__(self, layer, experts, topk, scaling):
        _sonicmoe()
        self.layer, self.experts, self.topk, self.scaling = (
            layer,
            experts,
            topk,
            scaling,
        )
        self.world, self.rank = dist.get_world_size(), dist.get_rank()
        self.local = experts // self.world
        self.block = CUDNN_BLOCK

    def rebuild(self, a1, b1, a2, b2):
        layer = self.layer

        def form(r):
            return (
                r.qdata.view(torch.float4_e2m1fn_x2),
                r.scale_atoms.view(torch.float8_e4m3fn),
            )

        a1, b1, a2, b2 = (t.detach() for t in (a1, b1, a2, b2))
        self.w1 = form(layer._requant(layer.q1, layer.s1, layer.pts1, a1, b1))
        self.w2 = form(layer._requant(layer.q2, layer.s2, layer.pts2, a2, b2))
        self.w1t = form(
            layer._requant(
                layer.q1t, layer.s1t, layer.pts1, b1.transpose(1, 2), a1.transpose(1, 2)
            )
        )
        self.w2t = form(
            layer._requant(
                layer.q2t, layer.s2t, layer.pts2, b2.transpose(1, 2), a2.transpose(1, 2)
            )
        )

    def _experts(self, xg, cu, padded, a1, b1, a2, b2):
        offsets = cu[1:].to(torch.int32)
        return CudnnMergedExperts.apply(xg, offsets, a1, b1, a2, b2, self)


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
        Nvfp4PersistentBackward,
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
    # Opt-in: NVFP4 persistent variants only (no BF16 runtimes, no all-to-all baselines), for memory headroom
    nvfp4_only = os.environ.get("NVFP4_BENCH_NVFP4_ONLY", "0") != "0"
    w1_bf = None if nvfp4_only else w1.bfloat16().requires_grad_()
    w2_bf = None if nvfp4_only else w2.bfloat16().requires_grad_()
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
    trainable = [t for t in (w1_bf, w2_bf, *adapters) if t is not None]

    runtime_nv = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inter,
        compute_dtype="nvfp4_lora",
    )
    runtime_bf = runtime_bf_ring = a2a = None
    cache = RequantCache()
    variants = {
        "nvfp4_lora": lambda x, ids, s: nvfp4_lora_moe(
            x, ids, s, *adapters, runtime_nv, layer, cache
        ),
    }
    if not nvfp4_only:
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
        # Same BF16 path without the owned-backward ring floor (the forward's ring sizing)
        os.environ["MONO_EP_BACKWARD_RING_WAVES"] = "0"
        runtime_bf_ring = PersistentMoE(
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
        os.environ.pop("MONO_EP_BACKWARD_RING_WAVES")
        a2a = Fp4CuteA2A(layer, experts, topk, SCALING)
        variants["bf16_persistent"] = lambda x, ids, s: runtime_bf(
            x, ids, s, w1_bf, w2_bf
        )
        variants["bf16_persistent_ring0"] = lambda x, ids, s: runtime_bf_ring(
            x, ids, s, w1_bf, w2_bf
        )
        variants["fp4_cute_a2a"] = lambda x, ids, s: a2a(x, ids, s, *adapters)
    # Opt-in: backward kernel option variants, each on its own runtime built like `nvfp4_lora`'s (options 2051 =
    # 3 | 2048 with the lean recompute tail, 3 without it). The instances are created after the main runtime, so
    # comparing them with each other rather than with `nvfp4_lora` keeps construction effects out.
    extra_runtimes = {}
    if os.environ.get("NVFP4_BENCH_OPTION_VARIANTS", "0") != "0":
        for name, options in (
            ("nvfp4_lora_lean", 3 | 2048),
            ("nvfp4_lora_default2", 3),
        ):
            rt = PersistentMoE(
                dist.group.WORLD,
                experts,
                tokens,
                topk,
                hidden,
                inter,
                compute_dtype="nvfp4_lora",
            )
            rt._nvfp4_backward = Nvfp4PersistentBackward(rt, options=options)
            extra_runtimes[name] = rt
            variants[name] = lambda x, ids, s, rt=rt: nvfp4_lora_moe(
                x, ids, s, *adapters, rt, layer, cache
            )
    # Only on the image with the cuDNN frontend (Modal job `timing_cudnn`)
    cudnn_a2a = None
    if not nvfp4_only and importlib.util.find_spec("cudnn") is not None:
        cudnn_a2a = CudnnGroupedA2A(layer, experts, topk, SCALING)
        variants["cudnn_grouped_a2a"] = lambda x, ids, s: cudnn_a2a(
            x, ids, s, *adapters
        )
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

    def merged_with(backend):
        # every requant kernel on the same layer; cuda_fused needs no transposed base
        current, layer.backend = layer.backend, backend
        try:
            return layer.merged(*adapters)
        finally:
            layer.backend = current

    backends = [layer.backend] + [
        b
        for b in ("triton", "cuda", "cuda_fused")
        if b != layer.backend and (b == "cuda_fused" or hasattr(layer, "q1t"))
    ]
    for name, fn in (
        ("nvfp4_lora/requant_4_forms", lambda: layer.merged(*adapters)),
        *(
            (f"nvfp4_lora/requant_4_forms[{b}]", lambda b=b: merged_with(b))
            for b in backends[1:]
        ),
        *(
            [
                (
                    "fp4_cute_a2a/requant_2_forms_plus_bf16_dequant",
                    lambda: a2a.rebuild(*adapters),
                )
            ]
            if a2a is not None
            else []
        ),
        *(
            [
                (
                    "cudnn_grouped_a2a/requant_4_forms",
                    lambda: cudnn_a2a.rebuild(*adapters),
                )
            ]
            if cudnn_a2a is not None
            else []
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
    if a2a is not None:
        a2a.rebuild(*adapters)
    agreement = {}
    if cudnn_a2a is not None:
        # Same layer through both all-to-all baselines: y and dx agreement (two NVFP4 quantizers)
        cudnn_a2a.rebuild(*adapters)
        ids, scores = routing(tokens, experts, topk, 0.0, device, gen)
        outs = {}
        for name in ("fp4_cute_a2a", "cudnn_grouped_a2a"):
            x.grad = None
            y = variants[name](x, ids, scores)
            y.backward(dy)
            outs[name] = (y.detach().float(), x.grad.float())
        x.grad = None
        (y0, dx0), (y1, dx1) = outs["fp4_cute_a2a"], outs["cudnn_grouped_a2a"]
        stats = torch.stack(
            [
                (y1 - y0).norm() ** 2,
                y0.norm() ** 2,
                (dx1 - dx0).norm() ** 2,
                dx0.norm() ** 2,
            ]
        )
        dist.all_reduce(stats)
        agreement = dict(
            y_rel_fro_cudnn_vs_fp4_cute=(stats[0] / stats[1]).sqrt().item(),
            dx_rel_fro_cudnn_vs_fp4_cute=(stats[2] / stats[3]).sqrt().item(),
        )

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
                for t in (x, scores, *trainable):
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
    # Top CUDA kernels of one backward per all-to-all variant (rank 0's view)
    from torch.profiler import ProfilerActivity, profile

    backward_kernels = {}
    for name in [v for v in variants if v.endswith("_a2a")]:
        x.grad = None
        y = variants[name](x, ids, scores)
        dist.barrier()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            y.backward(dy)
            torch.cuda.synchronize()
        rows = sorted(
            (e for e in prof.key_averages() if e.device_time_total > 0),
            key=lambda e: e.device_time_total,
            reverse=True,
        )
        backward_kernels[name] = dict(
            total_us=round(sum(e.device_time_total for e in rows), 1),
            top=[
                dict(
                    name=e.key[:90],
                    total_us=round(e.device_time_total, 1),
                    count=e.count,
                )
                for e in rows[:12]
            ],
        )
        del y
    x.grad = None
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
        cudnn_vs_fp4_cute=agreement,
        a2a_backward_kernels_rank0=backward_kernels,
        peak_memory_gib={
            v: m / 2**30 for v, m in zip(variants, mem.tolist(), strict=True)
        },
        timings=results,
        extra_runtime_creation_order=list(extra_runtimes),
        backward_ring_tokens=dict(
            nvfp4_lora=runtime_nv._nvfp4_backward.buffer.l1_acts.shape[0]
        ),
    )
    if runtime_bf is not None:
        summary["backward_ring_tokens"].update(
            bf16_persistent=runtime_bf._peer_backward.buffer.l1_acts.shape[0],
            bf16_persistent_ring0=runtime_bf_ring._peer_backward.buffer.l1_acts.shape[
                0
            ],
        )
    for name, rt in extra_runtimes.items():
        summary["backward_ring_tokens"][name] = rt._nvfp4_backward.buffer.l1_acts.shape[
            0
        ]
    if rank == 0:
        print("BENCH_JSON " + json.dumps(summary), flush=True)
        out = os.environ.get("NVFP4_OUT")
        if out:
            with open(out, "w") as f:
                json.dump(summary, f, indent=1)
    dist.barrier()
    runtime_nv.close()
    for rt in extra_runtimes.values():
        rt.close()
    if runtime_bf is not None:
        runtime_bf.close()
        runtime_bf_ring.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
