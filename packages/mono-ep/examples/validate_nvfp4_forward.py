# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-rank parity of the SM100 NVFP4 LoRA persistent forward.

Builds NVFP4 base experts, merges a small LoRA with the packed requant kernel,
runs the persistent ``nvfp4_lora`` forward and compares it with
(a) an FP32 oracle on the dequantized merged weights,
(b) a replica of the kernel's numerics (NVFP4 activations, FP32 SwiGLU, BF16
    per-expert outputs) and, when ``quack`` is installed,
(c) the fp4_cute grouped GEMM path on the same merged bytes.
"""

import json
import os
import sys
import types

import torch
import torch.distributed as dist

CASES = {
    # name: (experts, hidden, intermediate, topk, tokens, skew)
    "qwen30b_balanced": (16, 2048, 768, 8, 2048, 0.0),
    "qwen30b_skewed": (16, 2048, 768, 8, 2048, 0.5),
    "beam_balanced": (8, 5120, 2560, 4, 1024, 0.0),
}
RANK = 16
SCALING = 2.0


def _sonicmoe():
    """Requant from the vendored ``mono_ep.nvfp4_requant``; fp4_cute and friends
    from Axolotl's sonicmoe package (``SONICMOE_DIR`` or the installed axolotl)
    without executing its ``__init__``."""
    path = os.environ.get("SONICMOE_DIR")
    if path and "sonicmoe_lite" not in sys.modules:
        pkg = types.ModuleType("sonicmoe_lite")
        pkg.__path__ = [path]
        sys.modules["sonicmoe_lite"] = pkg
    name = "sonicmoe_lite" if path else "axolotl.integrations.kernels.libs.sonicmoe"
    import importlib

    from mono_ep import nvfp4_requant as vendored

    return types.SimpleNamespace(
        requant=vendored.nvfp4_requant,
        triton_requant=vendored.triton_nvfp4_requant,
        quant=importlib.import_module(f"{name}.nvfp4_quant"),
        sf_layout=importlib.import_module(f"{name}.sf_layout"),
        name=name,
    )


def interleave_perm(n2, device):
    """Row permutation to mono-ep's granularity-8 gate/up interleave: new row
    ``16j + 8h + i`` holds gate (h=0) / up (h=1) output column ``8j + i``."""
    from deep_gemm.mega import _interleave_weights

    return _interleave_weights(torch.arange(n2, device=device).view(n2, 1)).view(-1)


def fake_quant_rows(x, sf_scale, inv_pts):
    """The kernel's activation quantizer: scale = e4m3(amax * sf_scale) with
    saturation and no floor, code = e2m1(x * inv_pts / scale)."""
    xf = x.float()
    rows, k = xf.shape
    blocks = xf.view(rows, k // 16, 16)
    amax = blocks.abs().amax(-1)
    sdec = (amax * sf_scale).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    r = torch.where(sdec > 0, inv_pts / sdec, torch.zeros_like(sdec))
    q = blocks * r.unsqueeze(-1)
    # RNE onto E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    a = q.abs().clamp(max=6.0)
    idx = torch.bucketize(a, (mags[1:] + mags[:-1]) / 2)
    mid = (mags[1:] + mags[:-1]) / 2
    # ties to even code: a == midpoint -> lower when lower idx is even
    tie = torch.isin(a, mid)
    lower = idx - 1
    idx = torch.where(tie & (lower % 2 == 0), lower, idx)
    vals = mags[idx] * torch.sign(q)
    # dequantized value = code * scale (pts applied by the fold)
    return (vals * sdec.unsqueeze(-1)).view(rows, k)


def routing(tokens, experts, topk, skew, device, gen):
    logits = torch.randn(tokens, experts, device=device, generator=gen)
    if skew:
        hot = torch.rand(tokens, device=device, generator=gen) < skew
        logits[hot, 0] += 8.0
    scores, indices = torch.topk(torch.softmax(logits, -1), topk, dim=-1)
    scores = scores / scores.sum(-1, keepdim=True)
    return indices.long(), scores.float()


def oracle_fp32(
    x, indices, scores, w1c, w2, act_quant=None, out_bf16_expert=False, l2_fold=None
):
    tokens, hidden = x.shape
    inter = w2.shape[-1]
    out = torch.zeros(tokens, hidden, device=x.device, dtype=torch.float32)
    for e in range(w1c.shape[0]):
        rows, slots = (indices == e).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        xe = x[rows].float()
        if act_quant is not None:
            xe = act_quant[0](xe, e)
        z = xe @ w1c[e].T
        gate, up = z[:, :inter], z[:, inter:]
        h = torch.nn.functional.silu(gate) * up * scores[rows, slots, None]
        if act_quant is not None:
            h = act_quant[1](h, e)
        ye = h @ w2[e].T
        if l2_fold is not None:
            ye = ye * l2_fold[e]
        if out_bf16_expert:
            ye = ye.bfloat16().float()
        out.index_add_(0, rows, ye)
    return out


def errors(a, ref):
    d = a.float() - ref.float()
    return dict(
        max_abs=d.abs().max().item(),
        mean_abs=d.abs().mean().item(),
        rel_fro=(d.norm() / ref.float().norm()).item(),
    )


def run_case(name, experts, hidden, inter, topk, tokens, skew, sm, device):
    from mono_ep.runtime import PersistentMoE

    gen = torch.Generator(device=device).manual_seed(1234 + hash(name) % 1000)
    requant, triton_requant = sm.requant, sm.triton_requant

    # Base experts in BF16, quantized with a per-expert per-tensor scale
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
    perm = interleave_perm(2 * inter, device)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=device)
    q1i, s1i = (
        q1.index_select(1, perm).contiguous(),
        s1.index_select(1, perm).contiguous(),
    )

    # LoRA factors (rank-major PEFT layout), B rows permuted like the base
    a1 = torch.randn(experts * RANK, hidden, device=device, generator=gen) / hidden**0.5
    b1 = torch.randn(2 * inter, RANK * experts, device=device, generator=gen) * 0.02
    a2 = torch.randn(experts * RANK, inter, device=device, generator=gen) / inter**0.5
    b2 = torch.randn(hidden, RANK * experts, device=device, generator=gen) * 0.02
    r1 = triton_requant.requant_lora_packed(
        q1i,
        s1i,
        pts1,
        a1.bfloat16(),
        b1.index_select(0, perm).bfloat16(),
        SCALING,
        pts1,
    )
    r2 = triton_requant.requant_lora_packed(
        q2, s2, pts2, a2.bfloat16(), b2.bfloat16(), SCALING, pts2
    )
    torch.cuda.synchronize()

    # Merged weights, dequantized to FP32 (interleaved L1 rows -> concat order for the oracles)
    def dequant(res, pts):
        e, n, k2 = res.qdata.shape
        return requant.dequantize_nvfp4_packed_ref(
            res.qdata.view(e * n, k2),
            res.scale.view(e * n, k2 // 8),
            pts.repeat_interleave(n),
            torch.float32,
        ).view(e, n, 2 * k2)

    w1m_i = dequant(r1, pts1)
    w1m_c = w1m_i.index_select(1, inv_perm)
    w2m = dequant(r2, pts2)

    x = torch.randn(tokens, hidden, device=device, generator=gen).bfloat16()
    indices, scores = routing(tokens, experts, topk, skew, device, gen)

    # Kernel

    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inter,
        compute_dtype="nvfp4_lora",
    )
    expert_scales = torch.zeros(experts, 8, device=device, dtype=torch.float32)
    expert_scales[:, 0] = 1.0 / 6.0
    expert_scales[:, 1] = 1.0
    expert_scales[:, 2] = pts1
    expert_scales[:, 3] = pts1
    expert_scales[:, 4] = 1.0 / 6.0
    expert_scales[:, 5] = 1.0
    expert_scales[:, 6] = pts2
    y = runtime.forward_nvfp4(
        x,
        indices,
        scores,
        r1.qdata,
        r1.scale_atoms,
        r2.qdata,
        r2.scale_atoms,
        expert_scales,
    )
    torch.cuda.synchronize()
    y2 = runtime.forward_nvfp4(
        x,
        indices,
        scores,
        r1.qdata,
        r1.scale_atoms,
        r2.qdata,
        r2.scale_atoms,
        expert_scales,
    )
    torch.cuda.synchronize()
    deterministic = torch.equal(y, y2)
    runtime.close()

    # Oracles
    ref = oracle_fp32(x, indices, scores, w1m_c, w2m)
    quant_x = lambda t, e: fake_quant_rows(t, 1.0 / 6.0, 1.0)  # noqa: E731
    quant_h = lambda t, e: fake_quant_rows(t, 1.0 / 6.0, 1.0)  # noqa: E731
    replica = oracle_fp32(
        x,
        indices,
        scores,
        w1m_c,
        w2m,
        act_quant=(quant_x, quant_h),
        out_bf16_expert=True,
    ).bfloat16()
    result = dict(
        case=name,
        shape=dict(
            experts=experts,
            hidden=hidden,
            inter=inter,
            topk=topk,
            tokens=tokens,
            skew=skew,
        ),
        deterministic=deterministic,
        finite=bool(torch.isfinite(y).all()),
        kernel_vs_fp32=errors(y, ref),
        replica_vs_fp32=errors(replica, ref),
        kernel_vs_replica=errors(y, replica),
        bf16_rounding_vs_fp32=errors(ref.bfloat16(), ref),
    )

    # fp4_cute on the same merged bytes (optional)
    try:
        import importlib

        fp4_cute = importlib.import_module(f"{sm.name}.fp4_cute")
        fp4_cute_ops = importlib.import_module(f"{sm.name}.fp4_cute_ops")
        import quack  # noqa: F401

        flat_idx = indices.view(-1)
        order = torch.argsort(flat_idx, stable=True)
        counts = torch.bincount(flat_idx, minlength=experts)
        cu = torch.zeros(experts + 1, device=device, dtype=torch.int32)
        cu[1:] = counts.cumsum(0)
        tok = order // topk
        slot = order % topk
        xg = x[tok].contiguous()
        a_q, sfa = fp4_cute_ops.quantize_grouped_rows(xg, cu)
        l1 = fp4_cute.GroupedNvfp4Gemm(
            2 * inter, hidden, experts, gated=True, concat_b=True, store_preact=False
        )
        l1.set_weights(
            r1.qdata.index_select(1, inv_perm).contiguous(),
            r1.scale.index_select(1, inv_perm).contiguous(),
        )
        colvec = pts1[flat_idx[order]].float()
        post, _ = l1.forward(a_q, sfa, cu, colvec=colvec)
        h_q, sfh = fp4_cute_ops.quantize_grouped_rows(post.contiguous(), cu)
        l2 = fp4_cute.GroupedNvfp4Gemm(hidden, inter, experts)
        l2.set_weights(r2.qdata.contiguous(), r2.scale.contiguous())
        ye = l2.forward(h_q, sfh, cu, colvec=pts2[flat_idx[order]].float())
        y_cute = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
        y_cute.index_add_(0, tok, ye.float() * scores[tok, slot, None])
        result["fp4_cute_vs_fp32"] = errors(y_cute.bfloat16(), ref)
        result["kernel_vs_fp4_cute"] = errors(y, y_cute.bfloat16())
    except Exception as exc:  # noqa: BLE001
        result["fp4_cute"] = f"skipped: {type(exc).__name__}: {exc}"[:300]
    return result


def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    assert "B200" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    import deep_gemm

    sms = torch.cuda.get_device_properties(rank).multi_processor_count
    deep_gemm.set_num_sms(sms)
    sm = _sonicmoe()
    device = torch.device("cuda", rank)
    selected = os.environ.get("NVFP4_CASES", ",".join(CASES)).split(",")
    results = []
    for name in selected:
        results.append(run_case(name, *CASES[name], sm, device))
        print(json.dumps(results[-1], indent=1), flush=True)
    out = os.environ.get("NVFP4_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    failures = []
    for r in results:
        if not r["finite"] or not r["deterministic"]:
            failures.append(
                f"{r['case']}: finite={r['finite']} deterministic={r['deterministic']}"
            )
        bound = 1.25 * r["replica_vs_fp32"]["rel_fro"]
        if "fp4_cute_vs_fp32" in r:
            bound = max(bound, 1.25 * r["fp4_cute_vs_fp32"]["rel_fro"])
        if r["kernel_vs_fp32"]["rel_fro"] > bound:
            failures.append(
                f"{r['case']}: kernel rel_fro {r['kernel_vs_fp32']['rel_fro']:.4g} > {bound:.4g}"
            )
    if failures:
        raise SystemExit("\n".join(failures))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
