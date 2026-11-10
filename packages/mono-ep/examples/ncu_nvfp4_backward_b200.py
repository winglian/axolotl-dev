# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Nsight Compute target: the NVFP4 LoRA backward kernel and the cuDNN frontend
grouped GEMMs for Dh and Dx on the same operands, single B200, world 1 at the
EP8 per-rank rows (``NVFP4_CASE`` = ``beam`` or ``qwen30b``). Launch order:
three mono-ep backward kernels, then three cuDNN Dh and three cuDNN Dx GEMMs
(the profiler takes the third of each)."""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_cudnn_grouped_gemm_b200 import (  # noqa: E402
    CASES,
    M_ALIGNED,
    Operand,
    weight_operand,
)
from validate_nvfp4_forward import RANK, SCALING, routing  # noqa: E402

REPEATS = 3


def main():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    import deep_gemm
    from cudnn import grouped_gemm_quant_wrapper_sm100
    from mono_ep.nvfp4_lora import (
        Nvfp4LoraExperts,
        RequantCache,
        backward_for,
        requant_modules,
    )
    from mono_ep.runtime import PersistentMoE

    deep_gemm.set_num_sms(torch.cuda.get_device_properties(0).multi_processor_count)
    case = CASES[os.environ.get("NVFP4_CASE", "beam")]
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
    x = torch.randn(tokens, hidden, device=device, generator=gen).bfloat16()
    dy = torch.randn(tokens, hidden, device=device, generator=gen).bfloat16()
    ids, scores = routing(tokens, experts, topk, 0.0, device, gen)
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inter,
        compute_dtype="nvfp4_lora",
    )
    merged = RequantCache().get(layer, adapters)
    backward = backward_for(runtime)
    for _ in range(REPEATS):
        backward.backward(dy, x, ids, scores, merged)
    torch.cuda.synchronize()

    # cuDNN Dh / Dx on the same rows, groups padded to 256
    _, r2, r1t, r2t = layer.requant_all(*adapters)
    del r2
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tok = order // topk
    counts = torch.bincount(flat, minlength=experts)
    padded = (counts + M_ALIGNED - 1) // M_ALIGNED * M_ALIGNED
    ends = padded.cumsum(0)
    starts = ends - padded
    n_rows, total = int(counts.sum()), int(ends[-1])
    pos = (starts - (counts.cumsum(0) - counts)).repeat_interleave(
        counts
    ) + torch.arange(n_rows, device=device)
    dyg = dy.new_zeros(total, hidden).index_copy_(0, pos, dy.index_select(0, tok))
    dzg = dy.new_zeros(total, 2 * inter).index_copy_(
        0, pos, torch.randn(n_rows, 2 * inter, device=device, generator=gen).bfloat16()
    )
    offsets = ends.to(torch.int32)
    for act, weight, alpha in (
        (
            Operand(dyg, requant, per_row=True),
            weight_operand(r2t, layer.pts2, requant),
            layer.pts2,
        ),
        (
            Operand(dzg, requant, per_row=True),
            weight_operand(r1t, layer.pts1, requant),
            layer.pts1,
        ),
    ):
        for _ in range(REPEATS):
            grouped_gemm_quant_wrapper_sm100(
                a_tensor=act.packed.view(torch.float4_e2m1fn_x2),
                sfa_tensor=act.sfa,
                padded_offsets=offsets,
                alpha_tensor=alpha,
                b_tensor=weight["b"],
                sfb_tensor=weight["sfb"],
                row_scale_tensor=act.row_scale,
                d_dtype=torch.bfloat16,
                sf_vec_size=16,
                m_aligned=M_ALIGNED,
                generate_amax=False,
            )
        torch.cuda.synchronize()
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
