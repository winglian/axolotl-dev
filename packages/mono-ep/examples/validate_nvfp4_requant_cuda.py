# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-B200 checks and timings of the CUDA NVFP4 requant against the Triton kernel.

1. Conversion primitives, exhaustively: the SM100 ``cvt`` E2M1 encode over every
   fp32 with ``|x| <= 8`` and the E4M3 encode over every fp32 in ``[2**-11, 448]``
   against the torch encoders, and both decodes over all 256 bytes.
2. Byte parity of ``requant_lora_packed_cuda`` with the Triton kernel on odd
   shapes over the numerics grid (rank padding, bf16 / fp32 adapters, bf16 / fp32
   rounding, torchao / ModelOpt, fresh / reuse, per-projection scales, row
   interleave, expert ranges), and with an exact torch reference (sequential
   rank-order fp32 delta) for bf16 adapters.
3. The four merged forms of ``Nvfp4LoraExperts`` at the Qwen3-30B and Beam
   per-GPU shapes, CUDA against Triton, byte for byte; the fused variant's normal
   forms against the unfused ones and its transposed forms against the exact
   torch reference, plus both transposed forms' error against the fp32 merged
   weight.
4. Timings at both shapes: Triton and CUDA four-form rebuilds, per form, FFMA2 on
   and off, the fused variant, the CUDA bound probes (copy, no delta, no
   quantizer) and a torch copy of the same bytes. Writes ``NVFP4_OUT`` JSON.
"""

import json
import os
import statistics

import torch

RANK, SCALING = 16, 2.0
CASES = {
    "qwen30b": dict(hidden=2048, inter=768, experts=16),
    "beam": dict(hidden=5120, inter=2560, experts=32),
}
WARMUP, SAMPLES = 5, 20


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


def _as_int32(lo, hi):
    w = torch.arange(lo, hi, device="cuda", dtype=torch.int64)
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)


def check_conversions():
    from mono_ep.nvfp4_requant import cuda_nvfp4_requant as cr, nvfp4_requant as ref

    out = {}
    step = 1 << 27
    bad = total = 0
    for lo, hi in ((0, 0x41000001), (0x80000000, 0xC1000001)):
        for start in range(lo, hi, step):
            w = _as_int32(start, min(hi, start + step))
            got = cr.convert_probe("e2m1_encode", w) & 0xF
            x = w.view(torch.float32)
            exp = ref.encode_e2m1_rne(x.clamp(-6.0, 6.0), sign_from_bit=True).to(
                torch.int32
            )
            bad += int((got != exp).sum())
            total += w.numel()
    out["e2m1_encode"] = dict(checked=total, mismatches=bad)
    w = _as_int32(0x3A000000, 0x43E00001)
    got = cr.convert_probe("e4m3_encode", w) & 0xFF
    exp = (
        w.view(torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
    )
    out["e4m3_encode"] = dict(checked=w.numel(), mismatches=int((got != exp).sum()))
    b = torch.arange(256, device="cuda", dtype=torch.int32)
    codes = b.to(torch.uint8)
    for kind, nib in (("e2m1_decode_lo", codes & 0xF), ("e2m1_decode_hi", codes >> 4)):
        got = cr.convert_probe(kind, b)
        exp = ref.code_values(nib).view(torch.int32)
        out[kind] = dict(checked=256, mismatches=int((got != exp).sum()))
    finite = (b & 0x7F) != 0x7F
    exp = codes.view(torch.float8_e4m3fn).float().view(torch.int32)
    got = cr.convert_probe("e4m3_decode_lo", b)
    out["e4m3_decode_lo"] = dict(
        checked=int(finite.sum()), mismatches=int((got != exp)[finite].sum())
    )
    got = cr.convert_probe("e4m3_decode_hi", b << 8)
    out["e4m3_decode_hi"] = dict(
        checked=int(finite.sum()), mismatches=int((got != exp)[finite].sum())
    )
    return out


def _base(E, N, K, projections, gen, *, inflate=False):
    from mono_ep.nvfp4_requant import nvfp4_requant as ref

    w = torch.randn(E, N, K, device="cuda", generator=gen) * 0.02
    w[:, :, :16] = 0.0  # all-zero blocks
    w[0, :4] *= 1e-4  # subnormal-scale blocks
    w[:, -1, 16:32] = 0.75 * w[:, -1, 16:32].abs().amax()  # exact ties
    seg = w.reshape(E, projections, -1).abs().amax(dim=2)
    pts = ref.per_tensor_scale_from_amax(seg)
    q, s = ref.quantize_nvfp4_packed_ref(
        w.reshape(E * N, K), ref.expand_pts_rows(pts, N)
    )
    s = s.view(torch.uint8)
    if inflate:
        # calibrated-looking scales: a third of the blocks carry a 2x scale the codes do not need
        mask = torch.rand(s.shape, device="cuda", generator=gen) < 0.33
        s = torch.where(mask & (s < 0x70) & (s > 0), s + 8, s)
    return q.view(E, N, K // 2), s.view(E, N, K // 16).view(torch.float8_e4m3fn), pts


def exact_reference(q, s, pts, A, B, scaling, e0, e1, numerics, interleave_rows):
    """Torch requant whose delta is the kernels' sequential rank-order fp32 chain
    (exact for bf16 factors: every product is exact, every add is one RN)."""
    from mono_ep.nvfp4_requant import nvfp4_requant as ref

    E, N, k2 = q.shape
    K = 2 * k2
    r = A.shape[0] // E
    rows = (e1 - e0) * N
    w = ref.dequantize_nvfp4_packed_ref(
        q[e0:e1].reshape(rows, k2),
        s[e0:e1].reshape(rows, K // 16),
        ref.expand_pts_rows(pts[e0:e1], N),
        numerics.round_dtype,
    ).reshape(e1 - e0, N, K)
    A3 = A.reshape(E, r, K)[e0:e1].float()
    B3 = B.reshape(N, r, E).permute(2, 0, 1)[e0:e1].float()
    acc = torch.zeros(e1 - e0, N, K, device=q.device)
    for i in range(r):
        acc = acc + B3[:, :, i : i + 1] * A3[:, i : i + 1, :]
    if r < max(16, 1 << (r - 1).bit_length()):
        acc = acc + 0.0
    w_eff = (w.float() + acc * scaling).to(numerics.round_dtype)
    packed, sc = ref.quantize_nvfp4_packed_ref(
        w_eff.reshape(rows, K),
        ref.expand_pts_rows(pts[e0:e1], N),
        modelopt_compat=numerics.modelopt_compat,
        base_scale=s[e0:e1].reshape(rows, K // 16)
        if numerics.scale_mode == "reuse"
        else None,
    )
    packed = packed.reshape(e1 - e0, N, K // 2)
    sc = sc.view(torch.uint8).reshape(e1 - e0, N, K // 16)
    if interleave_rows:
        perm = torch.empty(N, dtype=torch.long, device=q.device)
        n = torch.arange(N, device=q.device)
        perm[torch.where(n < N // 2, 2 * n, 2 * (n - N // 2) + 1)] = n
        packed, sc = packed[:, perm], sc[:, perm]
    return packed, sc


def _same(x, y):
    return bool(torch.equal(x.view(torch.uint8), y.view(torch.uint8)))


def _check_paired(q, s, pts, A, B, numerics, e0, e1, gen):
    """``requant_paired`` against two Triton requants (normal, and a random
    transposed base with ``B^T`` / ``A^T`` as its factors)."""
    from mono_ep.nvfp4_requant import (
        cuda_nvfp4_requant as cr,
        triton_nvfp4_requant as tr,
    )

    E, N, k2 = q.shape
    K = 2 * k2
    r = A.shape[0] // E
    qt, st, _ = _base(E, K, N, 1, gen)
    a_nat = A.reshape(E, r, K)
    b_nat = B.reshape(N, r, E).permute(2, 0, 1)
    # the transposed form's rank-major factors: A' = B^T [E*r, N], B' = A^T [K, r*E]
    a_t = b_nat.permute(0, 2, 1).reshape(E * r, N)
    b_t = a_nat.permute(2, 1, 0).reshape(K, r * E)
    tn = tr.requant_lora_packed(q, s, pts, A, B, 1.75, pts, e0, e1, numerics=numerics)
    tt = tr.requant_lora_packed(
        qt, st, pts, a_t, b_t, 1.75, pts, e0, e1, numerics=numerics
    )
    cn, ct = cr.requant_paired(
        q, s, qt, st, pts, a_nat, b_nat, 1.75, e0, e1, numerics=numerics
    )
    return all(
        _same(x, y)
        for x, y in (
            (tn.qdata, cn.qdata),
            (tn.scale, cn.scale),
            (tn.scale_atoms, cn.scale_atoms),
            (tt.qdata, ct.qdata),
            (tt.scale, ct.scale),
            (tt.scale_atoms, ct.scale_atoms),
        )
    )


def check_small_shapes():
    from mono_ep.nvfp4_requant import (
        cuda_nvfp4_requant as cr,
        nvfp4_requant as ref,
        triton_nvfp4_requant as tr,
    )

    gen = torch.Generator(device="cuda").manual_seed(5)
    rows, failures = [], []
    for E, N, K in ((5, 80, 320), (3, 256, 576)):
        for projections in (1, 2):
            for inflate in (False, True):
                q, s, pts = _base(E, N, K, projections, gen, inflate=inflate)
                for r in (3, 16, 64):
                    for lora_dtype in (torch.bfloat16, torch.float32):
                        A = (
                            torch.randn(r * E, K, device="cuda", generator=gen) * 0.05
                        ).to(lora_dtype)
                        B = (
                            torch.randn(N, r * E, device="cuda", generator=gen) * 0.05
                        ).to(lora_dtype)
                        for round_dtype in (torch.bfloat16, torch.float32):
                            for modelopt in (False, True):
                                for scale_mode in ("fresh", "reuse"):
                                    numerics = ref.RequantNumerics(
                                        modelopt_compat=modelopt,
                                        round_dtype=round_dtype,
                                        scale_mode=scale_mode,
                                    )
                                    for interleave in (False, True):
                                        for e0, e1 in ((0, E), (1, E - 1)):
                                            kw = dict(
                                                numerics=numerics,
                                                interleave_rows=interleave,
                                            )
                                            t = tr.requant_lora_packed(
                                                q, s, pts, A, B, 1.75, pts, e0, e1, **kw
                                            )
                                            c = cr.requant_lora_packed_cuda(
                                                q, s, pts, A, B, 1.75, pts, e0, e1, **kw
                                            )
                                            ok = (
                                                _same(t.qdata, c.qdata)
                                                and _same(t.scale, c.scale)
                                                and _same(t.scale_atoms, c.scale_atoms)
                                                and torch.equal(
                                                    t.per_tensor_scale,
                                                    c.per_tensor_scale,
                                                )
                                            )
                                            exact = None
                                            if lora_dtype == torch.bfloat16:
                                                eq, es = exact_reference(
                                                    q,
                                                    s,
                                                    pts,
                                                    A,
                                                    B,
                                                    1.75,
                                                    e0,
                                                    e1,
                                                    numerics,
                                                    interleave,
                                                )
                                                exact = _same(c.qdata, eq) and _same(
                                                    c.scale, es
                                                )
                                            row = dict(
                                                shape=(E, N, K),
                                                projections=projections,
                                                inflated_scales=inflate,
                                                r=r,
                                                lora=str(lora_dtype),
                                                round=str(round_dtype),
                                                modelopt=modelopt,
                                                scale_mode=scale_mode,
                                                interleave=interleave,
                                                experts=(e0, e1),
                                                triton_bytes_equal=ok,
                                                exact_reference_equal=exact,
                                                code_mismatches=int(
                                                    (t.qdata != c.qdata).sum()
                                                ),
                                            )
                                            if (
                                                projections == 1
                                                and N % 64 == 0
                                                and not interleave
                                            ):
                                                row["paired_equal"] = _check_paired(
                                                    q,
                                                    s,
                                                    pts,
                                                    A,
                                                    B,
                                                    numerics,
                                                    e0,
                                                    e1,
                                                    gen,
                                                )
                                            rows.append(row)
                                            if (
                                                not ok
                                                or exact is False
                                                or row.get("paired_equal") is False
                                            ):
                                                failures.append(row)
    return dict(cases=len(rows), failures=failures[:20], num_failures=len(failures))


def _layer(case, backend, gen):
    from mono_ep.nvfp4_lora import Nvfp4LoraExperts
    from mono_ep.nvfp4_requant import nvfp4_requant as ref

    E, H, I = case["experts"], case["hidden"], case["inter"]
    w1 = torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", generator=gen) * 0.02
    pts1 = ref.per_tensor_scale_from_amax(w1.abs().amax(dim=(1, 2)))
    pts2 = ref.per_tensor_scale_from_amax(w2.abs().amax(dim=(1, 2)))

    def quantize(w, pts):
        e, n, k = w.shape
        packed, scale = ref.quantize_nvfp4_packed_ref(
            w.reshape(e * n, k), pts.repeat_interleave(n)
        )
        return packed.view(e, n, k // 2), scale.view(e, n, k // 16)

    q1, s1 = quantize(w1, pts1)
    q2, s2 = quantize(w2, pts2)
    del w1, w2
    layer = Nvfp4LoraExperts(
        q1, s1, pts1, q2, s2, pts2, scaling=SCALING, backend=backend
    )
    a1 = (torch.randn(E, RANK, H, device="cuda", generator=gen) / H**0.5).bfloat16()
    b1 = (torch.randn(E, 2 * I, RANK, device="cuda", generator=gen) * 0.02).bfloat16()
    a2 = (torch.randn(E, RANK, I, device="cuda", generator=gen) / I**0.5).bfloat16()
    b2 = (torch.randn(E, H, RANK, device="cuda", generator=gen) * 0.02).bfloat16()
    return layer, (a1, b1, a2, b2)


def _merged_exact(layer, q, s, pts, a, b):
    """fp32 merged weight and its bf16 rounding (natural rows), sequential delta."""
    e, n, k2 = q.shape
    w = layer.dequantized(types_result(q, s), pts).bfloat16().float()
    acc = torch.zeros_like(w)
    for i in range(a.shape[1]):
        acc = acc + b[:, :, i : i + 1].float() * a[:, i : i + 1, :].float()
    exact = w + acc * layer.scaling
    return exact, exact.bfloat16()


def types_result(q, s):
    import types

    return types.SimpleNamespace(qdata=q, scale=s)


def _rel(x, y):
    return float((x - y).norm() / y.norm())


def check_layer(name, case, gen):
    from mono_ep.nvfp4_requant import nvfp4_requant as ref

    layer, adapters = _layer(case, "triton", gen)
    a1, b1, a2, b2 = adapters
    out = dict(case=name)
    t = layer.requant_all(*adapters)
    layer.backend = "cuda"  # paired: w1 + w1t, w2 + w2t
    c = layer.requant_all(*adapters)
    forms = ("w1", "w2", "w1t", "w2t")
    out["cuda_vs_triton"] = {
        f: dict(
            qdata=_same(x.qdata, y.qdata),
            scale=_same(x.scale, y.scale),
            atoms=_same(x.scale_atoms, y.scale_atoms),
        )
        for f, x, y in zip(forms, t, c, strict=True)
    }
    layer.backend = "cuda_fused"
    f = layer.requant_all(*adapters)
    out["fused_normal_vs_unfused"] = {
        k: dict(
            qdata=_same(x.qdata, y.qdata),
            scale=_same(x.scale, y.scale),
            atoms=_same(x.scale_atoms, y.scale_atoms),
        )
        for k, x, y in (("w1", c[0], f[0]), ("w2", c[1], f[1]))
    }
    fused_t = {}
    for k, ft, ut, q, s, pts, a, b in (
        ("w1t", f[2], c[2], layer.q1, layer.s1, layer.pts1, a1, b1),
        ("w2t", f[3], c[3], layer.q2, layer.s2, layer.pts2, a2, b2),
    ):
        exact, merged = _merged_exact(layer, q, s, pts, a, b)
        e, n, kk = merged.shape
        mt = merged.float().transpose(1, 2).reshape(e * kk, n)
        rq, rs = ref.quantize_nvfp4_packed_ref(mt, pts.repeat_interleave(kk))
        fused_dq = layer.dequantized(ft, pts).transpose(1, 2)
        unfused_dq = layer.dequantized(ut, pts).transpose(1, 2)
        fused_t[k] = dict(
            qdata_vs_exact_reference=_same(ft.qdata.reshape(e * kk, n // 2), rq),
            scale_vs_exact_reference=_same(ft.scale.reshape(e * kk, n // 16), rs),
            atoms_vs_rowmajor=_same(
                ft.scale_atoms, ref.scale_atoms_from_rowmajor(ft.scale)
            ),
            code_bytes_differing_from_unfused=float(
                (ft.qdata != ut.qdata).float().mean()
            ),
            fused_rel_fro_vs_fp32_merged=_rel(fused_dq, exact),
            unfused_rel_fro_vs_fp32_merged=_rel(unfused_dq, exact),
            normal_form_rel_fro_vs_fp32_merged=None,
        )
        del exact, merged, mt, fused_dq, unfused_dq
    # the normal forms' own quantization error, for scale
    exact, _ = _merged_exact(layer, layer.q2, layer.s2, layer.pts2, a2, b2)
    fused_t["w2t"]["normal_form_rel_fro_vs_fp32_merged"] = _rel(
        layer.dequantized(c[1], layer.pts2), exact
    )
    del exact
    out["fused_transposed"] = fused_t
    del t, c, f
    torch.cuda.empty_cache()
    return out, layer, adapters


def time_layer(layer, adapters):
    from mono_ep.nvfp4_requant import cuda_nvfp4_requant as cr

    a1, b1, a2, b2 = adapters
    res = {}
    layer.backend = "triton"
    res["triton_4_forms_ms"] = timed(lambda: layer.requant_all(*adapters))
    bf = lambda t: t.contiguous()  # noqa: E731
    forms = dict(
        w1=(layer.q1i, layer.s1i, layer.pts1, a1, b1.index_select(1, layer.perm)),
        w2=(layer.q2, layer.s2, layer.pts2, a2, b2),
        w1t=(
            layer.q1t,
            layer.s1t,
            layer.pts1,
            bf(b1.transpose(1, 2)),
            bf(a1.transpose(1, 2)),
        ),
        w2t=(
            layer.q2t,
            layer.s2t,
            layer.pts2,
            bf(b2.transpose(1, 2)),
            bf(a2.transpose(1, 2)),
        ),
    )
    res["triton_per_form_ms"] = {
        k: timed(
            lambda v=v: layer.triton_requant.requant_lora_packed(
                v[0],
                v[1],
                v[2],
                v[3].reshape(-1, v[3].shape[2]),
                v[4].permute(1, 2, 0).reshape(v[4].shape[1], -1).contiguous(),
                layer.scaling,
                v[2],
            )
        )
        for k, v in forms.items()
    }
    from mono_ep.nvfp4_requant.nvfp4_requant import RequantNumerics

    tc = RequantNumerics(delta_kernel="bf16x3")
    res["triton_tensor_core_delta_per_form_ms"] = {
        k: timed(
            lambda v=v: layer.triton_requant.requant_lora_packed(
                v[0],
                v[1],
                v[2],
                v[3].reshape(-1, v[3].shape[2]),
                v[4].permute(1, 2, 0).reshape(v[4].shape[1], -1).contiguous(),
                layer.scaling,
                v[2],
                numerics=tc,
            )
        )
        for k, v in forms.items()
    }
    for ffma2 in (True, False):
        os.environ["MONO_EP_NVFP4_REQUANT_FFMA2"] = str(int(ffma2))
        layer.backend = "cuda"
        res[f"cuda_paired_4_forms_ms_ffma2_{int(ffma2)}"] = timed(
            lambda: layer.requant_all(*adapters)
        )
        layer.backend = "cuda_fused"
        res[f"cuda_fused_4_forms_ms_ffma2_{int(ffma2)}"] = timed(
            lambda: layer.requant_all(*adapters)
        )
    os.environ.pop("MONO_EP_NVFP4_REQUANT_FFMA2")
    res["cuda_unpaired_4_forms_ms"] = timed(
        lambda: [
            cr.requant_natural(v[0], v[1], v[2], v[3], v[4], layer.scaling, v[2])
            for v in forms.values()
        ]
    )
    for mode in ("requant", "copy", "no_delta", "no_quantizer"):
        res[f"cuda_{mode}_per_form_ms"] = {
            k: timed(
                lambda v=v, m=mode: cr.requant_natural(
                    v[0], v[1], v[2], v[3], v[4], layer.scaling, v[2], mode=m
                )
            )
            for k, v in forms.items()
        }
    nbytes = 0
    srcs = []
    for v in forms.values():
        for t in (v[0], v[1]):
            srcs.append(t)
            nbytes += t.numel() * t.element_size()
    nbytes_atoms = sum(v[1].numel() for v in forms.values())
    res["torch_copy_codes_and_scales_ms"] = timed(lambda: [t.clone() for t in srcs])
    res["bytes_read_codes_scales"] = nbytes
    res["bytes_written_codes_scales_atoms"] = nbytes + nbytes_atoms
    return res


def main():
    torch.cuda.set_device(0)
    assert "B200" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
    results = dict(device=torch.cuda.get_device_name())
    results["conversions"] = check_conversions()
    print(json.dumps(results["conversions"], indent=1), flush=True)
    if os.environ.get("NVFP4_SKIP_SMALL") != "1":
        results["small_shapes"] = check_small_shapes()
        print(
            json.dumps(
                {k: v for k, v in results["small_shapes"].items() if k != "failures"}
            ),
            flush=True,
        )
    gen = torch.Generator(device="cuda").manual_seed(77)
    for name in os.environ.get("NVFP4_CASES", ",".join(CASES)).split(","):
        parity, layer, adapters = check_layer(name, CASES[name], gen)
        print(json.dumps(parity, indent=1), flush=True)
        timing = time_layer(layer, adapters)
        print(json.dumps(timing, indent=1), flush=True)
        results[name] = dict(parity=parity, timing=timing)
        del layer, adapters
        torch.cuda.empty_cache()
    failed = [k for k, v in results["conversions"].items() if v["mismatches"]]
    if results.get("small_shapes", {}).get("num_failures"):
        failed.append("small_shapes")
    for name in CASES:
        p = results.get(name, {}).get("parity")
        if not p:
            continue
        for group in ("cuda_vs_triton", "fused_normal_vs_unfused"):
            for form, flags in p[group].items():
                if not all(flags.values()):
                    failed.append(f"{name}/{group}/{form}")
        for form, flags in p["fused_transposed"].items():
            if not (
                flags["qdata_vs_exact_reference"]
                and flags["scale_vs_exact_reference"]
                and flags["atoms_vs_rowmajor"]
            ):
                failed.append(f"{name}/fused_transposed/{form}")
    results["failed"] = failed
    out = os.environ.get("NVFP4_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    print("FAILED: " + ", ".join(failed) if failed else "ALL CHECKS PASSED", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
