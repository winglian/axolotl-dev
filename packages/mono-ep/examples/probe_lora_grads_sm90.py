# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""One Hopper GPU, EP8 pool shapes (16,384 tokens x top-8 routed rows per rank): the fp8_lora
adapter gradients from the SM90 blockwise backward's packed E4M3 wgrad operands, by the full FP8
dW plus projection (``project``) against the rank-r Triton passes (``packed``). Time per layer and
relative error against an FP32 per-expert reference on the dequantized operands; the compiled
TTGIR is checked for 8-bit types. Then the per-step LoRA merge of the experts by ``torch.baddbmm``
against the one-pass Triton merge, each against the single-rounded FP32 merge, next to the
five-form FP8 quantization of the merged experts.

    python probe_lora_grads_sm90.py [out.json]
"""

import functools
import json
import statistics
import sys

import torch

SHAPES = {"30b": (2048, 768, 16), "235b": (4096, 1536, 16), "beam": (5120, 2560, 32)}
ROWS, RANK, SCALING = 16384 * 8, 16, 2.0


def timed(fn, reps=20):
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


def packed_operand(x, ends, starts, padded):
    """Packed column-major E4M3 operand of the zero-padded ``x`` and its exact dequantization."""
    from mono_ep.wgrad_quantization import quantize_transposed_groups

    q, s = quantize_transposed_groups(
        x, ends, use_psum_layout=True, scales_mn_major=True
    )
    width = x.shape[1]
    deq = torch.zeros_like(x, dtype=torch.float32)
    for st, p in zip(starts.tolist(), padded.tolist(), strict=True):
        if not p:
            continue
        seg = q[st * width : (st + p) * width].view(width, p).float()
        scale = s[:, st // 128 : (st + p) // 128].repeat_interleave(128, 1)
        deq[st : st + p] = (seg * scale).t()
    return q, s, deq.bfloat16()


def reference(pools, starts, counts, a1, b1, a2, b2):
    x, dy, h, dz = pools
    out = [torch.zeros_like(t, dtype=torch.float32) for t in (a1, b1, a2, b2)]
    for e, (s, c) in enumerate(zip(starts.tolist(), counts.tolist(), strict=True)):
        xe, dye, he, dze = (p[s : s + c].float() for p in (x, dy, h, dz))
        p1 = (dze @ b1[e].float()).bfloat16().float()
        q1 = (xe @ a1[e].float().t()).bfloat16().float()
        p2 = (dye @ b2[e].float()).bfloat16().float()
        q2 = (he @ a2[e].float().t()).bfloat16().float()
        out[0][e] = SCALING * p1.t() @ xe
        out[1][e] = SCALING * dze.t() @ q1
        out[2][e] = SCALING * p2.t() @ he
        out[3][e] = SCALING * dye.t() @ q2
    return out


def project_route(packed, padded, rows, factors):
    from mono_ep.hopper_wgrad import weight_grad_fp8_hopper_packed
    from mono_ep.lora import project_weight_grads

    qx, qdy, qh, qdz, sx, sdy, sh, sdz = packed
    dw1 = weight_grad_fp8_hopper_packed(qdz, sdz, qx, sx, padded, rows)
    dw2 = weight_grad_fp8_hopper_packed(qdy, sdy, qh, sh, padded, rows)
    return project_weight_grads(dw1, dw2, *factors, SCALING)


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def main():
    from mono_ep import lora_grads

    out = []
    for name, (hidden, inter, e) in SHAPES.items():
        gen = torch.Generator(device="cuda").manual_seed(0)
        # uneven counts with a partial last block per expert, one empty expert
        per = ROWS // e
        counts = torch.full((e,), per - 128, dtype=torch.int64)
        counts[1::2] -= 37
        counts[3] = 0
        padded = (counts + 127) // 128 * 128
        starts = torch.cumsum(padded, 0) - padded
        rows = int(padded.sum())
        mask = torch.zeros(rows, dtype=torch.bool)
        for s, c in zip(starts.tolist(), counts.tolist(), strict=True):
            mask[s : s + c] = True
        mask = mask.cuda()[:, None]

        def mk(w, s=1.0, gen=gen, mask=mask, rows=rows):
            t = torch.randn(rows, w, device="cuda", generator=gen) * s
            return (t * mask).bfloat16()

        ends = (starts + padded).to(torch.int32).cuda()
        args = dict(
            starts=starts.to(torch.int32).cuda(),
            counts=counts.to(torch.int32).cuda(),
            padded=padded.to(torch.int32).cuda(),
        )
        ops = [
            packed_operand(x, ends, starts, padded)
            for x in (mk(hidden), mk(hidden, 0.01), mk(inter), mk(2 * inter, 0.01))
        ]
        packed = tuple(o[0] for o in ops) + tuple(o[1] for o in ops)
        deq = [o[2] for o in ops]
        a1 = torch.randn(e, RANK, hidden, device="cuda", generator=gen) / hidden**0.5
        b1 = torch.randn(e, 2 * inter, RANK, device="cuda", generator=gen) * 0.02
        a2 = torch.randn(e, RANK, inter, device="cuda", generator=gen) / inter**0.5
        b2 = torch.randn(e, hidden, RANK, device="cuda", generator=gen) * 0.02
        factors = [t.bfloat16() for t in (a1, b1, a2, b2)]
        project = functools.partial(
            project_route, packed, args["padded"], rows, factors
        )

        run_packed = functools.partial(
            lora_grads.lora_weight_grads_packed,
            packed,
            args["starts"],
            args["counts"],
            args["padded"],
            *factors,
            SCALING,
        )
        ref = reference(deq, starts, counts, *factors)
        keys = ("dA1", "dB1", "dA2", "dB2")
        row = dict(shape=name, rows=rows, local_experts=e, rank=RANK)
        row["project_ms"] = timed(project)
        row["packed_ms"] = timed(run_packed)
        row["project_rel"] = dict(zip(keys, map(rel, project(), ref), strict=True))
        row["packed_rel"] = dict(zip(keys, map(rel, run_packed(), ref), strict=True))
        print(json.dumps(row), flush=True)
        out.append(row)
        del ops, packed, deq
        torch.cuda.empty_cache()
    import os

    from mono_ep.lora import merge
    from mono_ep.sm90_weight_forms import quantize_all_forms

    def merge_both(mode, w1, w2, a1, b1, a2, b2):
        os.environ["MONO_EP_LORA_MERGE"] = mode
        return merge(w1, b1, a1, SCALING), merge(w2, b2, a2, SCALING)

    for name, (hidden, inter, e) in SHAPES.items():
        gen = torch.Generator(device="cuda").manual_seed(1)
        w1 = (
            torch.randn(e, 2 * inter, hidden, device="cuda", generator=gen) / 50
        ).bfloat16()
        w2 = (
            torch.randn(e, hidden, inter, device="cuda", generator=gen) / 50
        ).bfloat16()
        a1, a2 = (
            (torch.randn(e, RANK, k, device="cuda", generator=gen) / k**0.5).bfloat16()
            for k in (hidden, inter)
        )
        b1, b2 = (
            (torch.randn(e, n, RANK, device="cuda", generator=gen) * 0.02).bfloat16()
            for n in (2 * inter, hidden)
        )
        weights = (w1, w2, a1, b1, a2, b2)
        exact = (
            (w1.float() + SCALING * b1.float() @ a1.float()).bfloat16(),
            (w2.float() + SCALING * b2.float() @ a2.float()).bfloat16(),
        )
        row = dict(shape=name, local_experts=e, rank=RANK)
        for mode in ("torch", "triton"):
            row[f"{mode}_merge_ms"] = timed(
                functools.partial(merge_both, mode, *weights)
            )
            got = merge_both(mode, *weights)
            row[f"{mode}_vs_fp32_merge_differ"] = [
                float((g != x).float().mean()) for g, x in zip(got, exact, strict=True)
            ]
            del got
        row["forms_ms"] = timed(functools.partial(quantize_all_forms, *exact))
        print(json.dumps(row), flush=True)
        out.append(row)
        del w1, w2, exact, weights
        torch.cuda.empty_cache()
    os.environ.pop("MONO_EP_LORA_MERGE")
    ttgir = []
    for kernel in (lora_grads._project_packed, lora_grads._contract_packed):
        for cache in kernel.device_caches.values():
            for compiled in cache[0].values():
                ttgir.append(compiled.asm["ttgir"])
    eight_bit = sorted(
        {t for text in ttgir for t in ("i8", "f8E4M3", "f8E5M2") if t in text}
    )
    print(json.dumps(dict(compiled_kernels=len(ttgir), eight_bit_types=eight_bit)))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(dict(rows=out, eight_bit_types=eight_bit), f, indent=1)


if __name__ == "__main__":
    main()
