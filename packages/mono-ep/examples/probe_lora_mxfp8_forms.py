# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""One B200: the per-layer merge + MXFP8 forms rebuild of mxfp8_lora, separate (BF16 merge, then
the full fine-tune's forms) against fused (merge inside the CUDA forms builder), at the EP8 local
expert shapes. Reports both costs, how many form bytes differ and the dequantized error of
either against the FP32 merged weight.

    python probe_lora_mxfp8_forms.py [out.json]
"""

import functools
import json
import statistics
import sys

import torch

SHAPES = {"30b": (2048, 768, 16), "235b": (4096, 1536, 16), "beam": (5120, 2560, 32)}
RANK, SCALING = 16, 2.0
NAMES = ("sm100_mx_w1", "sm100_mx_w1t", "sm100_mx_w2", "sm100_mx_w2t")


def timed(fn, reps=30):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    return statistics.median(samples)


def separate(w1, w2, a1, b1, a2, b2):
    from mono_ep.lora import merge
    from mono_ep.mx_weight_forms import cuda_mx_backward_forms
    from mono_ep.token_quantization import prepare_persistent_mxfp8

    m1, m2 = merge(w1, b1, a1, SCALING), merge(w2, b2, a2, SCALING)
    f1, f2t, f1t = cuda_mx_backward_forms(m1, m2)
    f2 = prepare_persistent_mxfp8(m2, weights=True)
    return dict(zip(NAMES, (f1, f1t, f2, f2t), strict=True))


def dequant(form):
    """FP32 [E, N, K] of a ``(values [E, N, K], words view [E, N, K / 128])`` form; the words of a
    128-row block sit in UTCCP order (row n at slot (n % 32) * 4 + (n % 128) // 32)."""
    values, words = form
    n = torch.arange(values.shape[1], device=values.device)
    slot = n // 128 * 128 + (n % 32) * 4 + (n % 128) // 32
    words = words.transpose(1, 2)[:, :, slot].transpose(1, 2).contiguous()
    exps = words.view(torch.uint8).view(*words.shape, 4).float()
    scale = torch.exp2(exps - 127.0).repeat_interleave(32, dim=-1).flatten(-2)
    return values.float() * scale[..., : values.shape[-1]]


def err(form, ref):
    return float((dequant(form) - ref).norm() / ref.norm())


def main():
    from mono_ep.mx_weight_forms import cuda_mx_backward_forms, cuda_mx_lora_forms
    from mono_ep.token_quantization import prepare_persistent_mxfp8

    out = []
    gen = torch.Generator(device="cuda").manual_seed(0)
    for name, (h, i, e) in SHAPES.items():
        w1 = (
            torch.randn(e, 2 * i, h, device="cuda", generator=gen) / h**0.5
        ).bfloat16()
        w2 = (torch.randn(e, h, i, device="cuda", generator=gen) / i**0.5).bfloat16()
        a1 = (torch.randn(e, RANK, h, device="cuda", generator=gen) / h**0.5).bfloat16()
        b1 = (
            torch.randn(e, 2 * i, RANK, device="cuda", generator=gen) * 0.02
        ).bfloat16()
        a2 = (torch.randn(e, RANK, i, device="cuda", generator=gen) / i**0.5).bfloat16()
        b2 = (torch.randn(e, h, RANK, device="cuda", generator=gen) * 0.02).bfloat16()
        row = dict(shape=name, local_experts=e, hidden=h, inter=i, rank=RANK)
        args = (w1, w2, a1, b1, a2, b2)
        row["separate_ms"] = timed(functools.partial(separate, *args))
        row["fused_ms"] = timed(functools.partial(cuda_mx_lora_forms, *args, SCALING))
        from mono_ep.lora import merge

        row["merge_only_ms"] = timed(
            functools.partial(
                lambda w1, w2, a1, b1, a2, b2: (
                    merge(w1, b1, a1, SCALING),
                    merge(w2, b2, a2, SCALING),
                ),
                *args,
            )
        )
        sep = separate(w1, w2, a1, b1, a2, b2)
        fus = cuda_mx_lora_forms(w1, w2, a1, b1, a2, b2, SCALING)
        exact = (
            w1.float() + SCALING * b1.float() @ a1.float(),
            w2.float() + SCALING * b2.float() @ a2.float(),
        )
        from deep_gemm.mega import _interleave_weights

        refs = dict(
            sm100_mx_w1=_interleave_weights(exact[0]),
            sm100_mx_w1t=exact[0].transpose(1, 2),
            sm100_mx_w2=exact[1],
            sm100_mx_w2t=exact[1].transpose(1, 2),
        )
        for n in NAMES:
            (qs, ss), (qf, sf) = sep[n], fus[n]
            ref = refs[n]
            row[n] = dict(
                value_bytes_differ=float(
                    (qs.view(torch.uint8) != qf.view(torch.uint8)).float().mean()
                ),
                scale_words_differ=float((ss != sf).float().mean()),
                separate_rel_fp32=err(sep[n], ref),
                fused_rel_fp32=err(fus[n], ref),
            )
        # the fused merge rounds once: compare with the forms of a single-rounded FP32 merge
        single = dict(
            zip(
                ("sm100_mx_w1", "sm100_mx_w2t", "sm100_mx_w1t"),
                cuda_mx_backward_forms(
                    (w1.float() + SCALING * b1.float() @ a1.float()).bfloat16(),
                    (w2.float() + SCALING * b2.float() @ a2.float()).bfloat16(),
                ),
                strict=True,
            )
        )
        single["sm100_mx_w2"] = prepare_persistent_mxfp8(
            (w2.float() + SCALING * b2.float() @ a2.float()).bfloat16(), weights=True
        )
        for n in NAMES:
            row[n]["scale_words_differ_vs_single_rounding"] = float(
                (single[n][1] != fus[n][1]).float().mean()
            )
            row[n]["value_bytes_differ_vs_single_rounding"] = float(
                (single[n][0].view(torch.uint8) != fus[n][0].view(torch.uint8))
                .float()
                .mean()
            )
            row[n]["single_rounding_rel_fp32"] = err(single[n], refs[n])
        zero = [torch.zeros_like(t) for t in (b1, b2)]
        plain = dict(
            zip(
                ("sm100_mx_w1", "sm100_mx_w2t", "sm100_mx_w1t"),
                cuda_mx_backward_forms(w1, w2),
                strict=True,
            )
        )
        plain["sm100_mx_w2"] = prepare_persistent_mxfp8(w2, weights=True)
        zf = cuda_mx_lora_forms(w1, w2, a1, zero[0], a2, zero[1], SCALING)
        row["zero_adapter_bitwise"] = all(
            torch.equal(zf[n][0].view(torch.uint8), plain[n][0].view(torch.uint8))
            and torch.equal(zf[n][1], plain[n][1])
            for n in NAMES
        )
        print(json.dumps(row), flush=True)
        out.append(row)
        del w1, w2, sep, fus, exact, refs, plain, zf, single
        torch.cuda.empty_cache()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
