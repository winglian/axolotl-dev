# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""One GPU, EP8 pool shapes (16,384 tokens x top-8 routed rows per rank): the LoRA adapter
gradients from the backward's pools by torch._grouped_mm against the two-pass Triton kernels
(BF16 pools), and the Triton kernels on row-wise E4M3 x / dy / dz pools in the MXFP8 backward's
layout. Time per layer and relative error against an FP32 per-expert reference on the same
operands; the compiled TTGIR is checked for 8-bit types.

    python probe_lora_grads.py [out.json]
"""

import functools
import json
import os
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


def mx_pool(x):
    """Row-wise E4M3 pool and its ``[K / 128, rows]`` UTCCP-ordered scale words."""
    from mono_ep.token_quantization import prepare_persistent_mxfp8

    q, words = prepare_persistent_mxfp8(x)
    rows = x.shape[0]
    r = torch.arange(rows, device=x.device)
    sf_row = r - r % 128 + (r % 128 % 32) * 4 + (r % 128) // 32
    sw = torch.empty(x.shape[1] // 128, rows, device=x.device, dtype=torch.int32)
    sw[:, sf_row] = words.t()
    deq = q.float() * torch.exp2(
        (words.view(torch.uint8).view(rows, -1).float() - 127.0).repeat_interleave(
            32, 1
        )
    )
    return q, sw, deq.bfloat16()


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


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def main():
    from mono_ep import lora_grads
    from mono_ep.lora import lora_weight_grads

    out = []
    for name, (hidden, inter, e) in SHAPES.items():
        per = ROWS // e
        counts = torch.full((e,), per, dtype=torch.int32, device="cuda")
        starts = torch.arange(e, dtype=torch.int32, device="cuda") * per
        gen = torch.Generator(device="cuda").manual_seed(0)

        def mk(w, s=1.0, gen=gen):
            return (torch.randn(ROWS, w, device="cuda", generator=gen) * s).bfloat16()

        pools = [mk(hidden), mk(hidden, 0.01), mk(inter), mk(2 * inter, 0.01)]
        a1 = torch.randn(e, RANK, hidden, device="cuda", generator=gen) / hidden**0.5
        b1 = torch.randn(e, 2 * inter, RANK, device="cuda", generator=gen) * 0.02
        a2 = torch.randn(e, RANK, inter, device="cuda", generator=gen) / inter**0.5
        b2 = torch.randn(e, hidden, RANK, device="cuda", generator=gen) * 0.02
        args = (starts, counts, a1, b1, a2, b2, SCALING)
        row = dict(shape=name, rows=ROWS, local_experts=e, rank=RANK)
        os.environ["MONO_EP_LORA_WGRAD"] = "grouped"
        row["grouped_ms"] = timed(functools.partial(lora_weight_grads, pools, *args))
        grouped = lora_weight_grads(pools, *args)
        row["triton_bf16_ms"] = timed(
            functools.partial(lora_grads.lora_weight_grads_triton, pools, *args)
        )
        triton_bf16 = lora_grads.lora_weight_grads_triton(pools, *args)
        ref = reference(pools, starts, counts, a1, b1, a2, b2)
        keys = ("dA1", "dB1", "dA2", "dB2")
        row["grouped_rel"] = dict(zip(keys, map(rel, grouped, ref), strict=True))
        row["triton_bf16_rel"] = dict(
            zip(keys, map(rel, triton_bf16, ref), strict=True)
        )
        # E4M3 x / dy / dz in the backward's layout; the reference reads their exact dequantization
        qx, sx, x8 = mx_pool(pools[0])
        qdy, sdy, dy8 = mx_pool(pools[1])
        qdz, sdz, dz8 = mx_pool(pools[3])
        fp8 = (qx, qdy, qdz, sx, sdy, sdz)
        row["triton_fp8_ms"] = timed(
            functools.partial(
                lora_grads.lora_weight_grads_triton, pools, *args, fp8_pools=fp8
            )
        )
        triton_fp8 = lora_grads.lora_weight_grads_triton(pools, *args, fp8_pools=fp8)
        ref8 = reference((x8, dy8, pools[2], dz8), starts, counts, a1, b1, a2, b2)
        row["triton_fp8_rel_vs_dequant_ref"] = dict(
            zip(keys, map(rel, triton_fp8, ref8), strict=True)
        )
        row["triton_fp8_rel_vs_bf16_ref"] = dict(
            zip(keys, map(rel, triton_fp8, ref), strict=True)
        )
        print(json.dumps(row), flush=True)
        out.append(row)
        del pools, x8, dy8, dz8, qx, qdy, qdz
        torch.cuda.empty_cache()
    # 8-bit types in the compiled kernels (the E4M3 path reads int32 words)
    ttgir = []
    for kernel in (lora_grads._project, lora_grads._contract):
        for cache in kernel.device_caches.values():
            for compiled in cache[0].values():
                ttgir.append(compiled.asm["ttgir"])
    eight_bit = [
        token for text in ttgir for token in ("i8", "f8E4M3", "f8E5M2") if token in text
    ]
    print(
        json.dumps(
            dict(compiled_kernels=len(ttgir), eight_bit_types=sorted(set(eight_bit)))
        )
    )
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(
                dict(rows=out, eight_bit_types=sorted(set(eight_bit))), f, indent=1
            )


if __name__ == "__main__":
    main()
