# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-GPU parity of the SM100 NVFP4 LoRA backward (``dx``, ``dscore``, ``dA``,
``dB``) against an FP32 oracle on the dequantized merged weights, next to a
numerics replica that applies the kernel's quantization points (static NVFP4
``x``, per-row NVFP4 ``dy`` and ``dz``, BF16 pools). Run under torchrun on one
B200; writes ``NVFP4_OUT`` JSON."""

import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_nvfp4_forward import (  # noqa: E402
    CASES,
    RANK,
    SCALING,
    errors,
    fake_quant_rows,
    routing,
)


def fake_quant_row_dynamic(x):
    """Per-row dynamic NVFP4: pts_row = amax / 2688, scale = e4m3(block amax * 448 / amax)."""
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    sf_scale = torch.where(amax > 0, 448.0 / amax, torch.zeros_like(amax))
    inv_pts = torch.where(amax > 0, 2688.0 / amax, torch.zeros_like(amax))
    q = fake_quant_rows(xf, sf_scale, inv_pts)
    return q * (amax / 2688.0)


def oracle_grads(x, indices, scores, w1c, w2, dy):
    """FP32 autograd of the routed SwiGLU MoE on merged weights: returns
    ``dx, dscore, dW1 (concat order), dW2``."""
    from validate_nvfp4_forward import oracle_fp32

    # detach first: .float() on an fp32 input returns the caller's tensor itself
    x32 = x.detach().float().requires_grad_()
    s32 = scores.detach().clone().float().requires_grad_()
    w1 = w1c.detach().clone().requires_grad_()
    w2r = w2.detach().clone().requires_grad_()
    y = oracle_fp32(x32, indices, s32, w1, w2r)
    return torch.autograd.grad((y * dy.float()).sum(), (x32, s32, w1, w2r))


def replica_backward(
    x, indices, scores, w1c, w2, w1t, w2t, dy, scaling, a1, b1, a2, b2
):
    """The kernel's numerics in torch: NVFP4 operands at the kernel's quantization
    points, BF16 gate/up/dh/h/dz rounding, BF16 pools for the adapter grads."""
    tokens, hidden = x.shape
    inter = w2.shape[-1]
    experts = w1c.shape[0]
    dx = torch.zeros(tokens, hidden, device=x.device, dtype=torch.float32)
    ds = torch.zeros_like(scores, dtype=torch.float32)
    da1, db1 = (
        torch.zeros_like(a1, dtype=torch.float32),
        torch.zeros_like(b1, dtype=torch.float32),
    )
    da2, db2 = (
        torch.zeros_like(a2, dtype=torch.float32),
        torch.zeros_like(b2, dtype=torch.float32),
    )
    for e in range(experts):
        rows, slots = (indices == e).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        xq = fake_quant_rows(x[rows], 1.0 / 6.0, 1.0)
        z = (xq @ w1c[e].T).bfloat16().float()
        gate, up = z[:, :inter], z[:, inter:]
        sc = scores[rows, slots, None].float()
        sig = torch.sigmoid(gate)
        silu = gate * sig
        dyq = fake_quant_row_dynamic(dy[rows])
        dh = (dyq @ w2t[e].T).bfloat16().float()
        ds[rows, slots] = (dh * silu * up).sum(-1)
        h = (silu * up * sc).bfloat16().float()
        dg = (dh * sc * up * (sig + silu * (1 - sig))).bfloat16().float()
        du = (dh * sc * silu).bfloat16().float()
        dz = torch.cat([dg, du], -1)
        dzq = fake_quant_row_dynamic(dz)
        dx.index_add_(0, rows, (dzq @ w1t[e].T).bfloat16().float())
        xb, dyb = x[rows].float(), dy[rows].float()
        da1[e] = scaling * (dz @ b1[e].float()).bfloat16().float().T @ xb
        db1[e] = scaling * dz.T @ (xb @ a1[e].float().T).bfloat16().float()
        da2[e] = scaling * (dyb @ b2[e].float()).bfloat16().float().T @ h
        db2[e] = scaling * dyb.T @ (h @ a2[e].float().T).bfloat16().float()
    return dx, ds, da1, db1, da2, db2


def run_case(name, experts, hidden, inter, topk, tokens, skew, device):
    from mono_ep.nvfp4_lora import (
        Nvfp4LoraExperts,
        RequantCache,
        nvfp4_lora_moe,
        requant_modules,
    )
    from mono_ep.runtime import PersistentMoE

    gen = torch.Generator(device=device).manual_seed(4321 + hash(name) % 1000)
    requant = requant_modules().requant

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
    layer = Nvfp4LoraExperts(q1, s1, pts1, q2, s2, pts2, scaling=SCALING)

    a1 = (
        torch.randn(experts, RANK, hidden, device=device, generator=gen) / hidden**0.5
    ).requires_grad_()
    b1 = (
        torch.randn(experts, 2 * inter, RANK, device=device, generator=gen) * 0.02
    ).requires_grad_()
    a2 = (
        torch.randn(experts, RANK, inter, device=device, generator=gen) / inter**0.5
    ).requires_grad_()
    b2 = (
        torch.randn(experts, hidden, RANK, device=device, generator=gen) * 0.02
    ).requires_grad_()

    x = torch.randn(tokens, hidden, device=device, generator=gen).bfloat16()
    dy = (torch.randn(tokens, hidden, device=device, generator=gen) * 0.1).bfloat16()
    indices, scores = routing(tokens, experts, topk, skew, device, gen)

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

    def step():
        xr = x.clone().requires_grad_()
        sr = scores.clone().requires_grad_()
        for t in (a1, b1, a2, b2):
            t.grad = None
        y = nvfp4_lora_moe(xr, indices, sr, a1, b1, a2, b2, runtime, layer, cache)
        y.backward(dy)
        torch.cuda.synchronize()
        return y, xr.grad, sr.grad, a1.grad, b1.grad, a2.grad, b2.grad

    first = step()
    second = step()
    deterministic = all(torch.equal(p, q) for p, q in zip(first, second, strict=True))
    y, dx, ds, ga1, gb1, ga2, gb2 = first
    merged = cache.entries[id(layer)]
    r1, r2, r1t, r2t = layer.requant_all(a1, b1, a2, b2)
    runtime.close()

    inv_perm = torch.empty_like(layer.perm)
    inv_perm[layer.perm] = torch.arange(layer.perm.numel(), device=device)
    w1m_c = layer.dequantized(r1, pts1).index_select(1, inv_perm)
    w2m = layer.dequantized(r2, pts2)
    w1t = layer.dequantized(r1t, pts1)
    w2t = layer.dequantized(r2t, pts2)
    # Exact merged weights for the oracle (no transposed-form rounding)
    w1x = w1c_exact = (
        torch.stack(
            [
                requant.dequantize_nvfp4_packed_ref(
                    q1[e], s1[e], pts1[e].expand(2 * inter), torch.float32
                )
                for e in range(experts)
            ]
        )
        + SCALING * b1.detach().float() @ a1.detach().float()
    )
    w2x = (
        torch.stack(
            [
                requant.dequantize_nvfp4_packed_ref(
                    q2[e], s2[e], pts2[e].expand(hidden), torch.float32
                )
                for e in range(experts)
            ]
        )
        + SCALING * b2.detach().float() @ a2.detach().float()
    )
    del w1c_exact
    dx_ref, ds_ref, dw1_ref, dw2_ref = oracle_grads(x, indices, scores, w1x, w2x, dy)
    da1_ref = SCALING * b1.detach().float().transpose(1, 2) @ dw1_ref
    db1_ref = SCALING * dw1_ref @ a1.detach().float().transpose(1, 2)
    da2_ref = SCALING * b2.detach().float().transpose(1, 2) @ dw2_ref
    db2_ref = SCALING * dw2_ref @ a2.detach().float().transpose(1, 2)
    rep = replica_backward(
        x,
        indices,
        scores,
        w1m_c,
        w2m,
        w1t,
        w2t,
        dy,
        SCALING,
        a1.detach(),
        b1.detach(),
        a2.detach(),
        b2.detach(),
    )
    outs = dict(dx=dx, dscore=ds, dA1=ga1, dB1=gb1, dA2=ga2, dB2=gb2)
    refs = dict(
        dx=dx_ref, dscore=ds_ref, dA1=da1_ref, dB1=db1_ref, dA2=da2_ref, dB2=db2_ref
    )
    reps = dict(zip(outs, rep, strict=True))
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
        finite=all(bool(torch.isfinite(t).all()) for t in outs.values()),
        transposed_weight_rounding=dict(
            w1t_vs_w1=errors(w1t.transpose(1, 2), w1m_c),
            w2t_vs_w2=errors(w2t.transpose(1, 2), w2m),
        ),
        merged_quant_vs_exact=dict(w1=errors(w1m_c, w1x), w2=errors(w2m, w2x)),
        grads={
            k: dict(
                kernel_vs_fp32=errors(outs[k], refs[k]),
                replica_vs_fp32=errors(reps[k], refs[k]),
                kernel_vs_replica=errors(outs[k], reps[k]),
            )
            for k in outs
        },
        requant_builds=cache.builds,
        merged_version=merged.version,
    )
    return result


def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    assert "B200" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    import deep_gemm

    sms = torch.cuda.get_device_properties(rank).multi_processor_count
    deep_gemm.set_num_sms(sms)
    device = torch.device("cuda", rank)
    selected = os.environ.get("NVFP4_CASES", ",".join(CASES)).split(",")
    results = []
    for name in selected:
        results.append(run_case(name, *CASES[name], device))
        print(json.dumps(results[-1], indent=1), flush=True)
    out = os.environ.get("NVFP4_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    failures = []
    for r in results:
        if not r["finite"]:
            failures.append(f"{r['case']}: non-finite gradients")
        for k, g in r["grads"].items():
            bound = 1.25 * g["replica_vs_fp32"]["rel_fro"]
            if g["kernel_vs_fp32"]["rel_fro"] > bound:
                failures.append(
                    f"{r['case']}/{k}: kernel rel_fro {g['kernel_vs_fp32']['rel_fro']:.4g} > {bound:.4g}"
                )
    if failures:
        raise SystemExit("\n".join(failures))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
