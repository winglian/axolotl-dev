# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Launches the Triton and CUDA NVFP4 requant on the Beam ``w1`` form (32
experts, 5120 x 5120, rank 16, bf16 adapters) a few times each, for Nsight
Compute (``modal_nvfp4_requant.py ncu``)."""

import os

import torch


def main():
    from mono_ep.nvfp4_requant import (
        cuda_nvfp4_requant as cr,
        triton_nvfp4_requant as tr,
    )

    E, N, K, r = 32, 5120, 5120, 16
    gen = torch.Generator(device="cuda").manual_seed(3)
    q = torch.randint(
        0, 256, (E, N, K // 2), device="cuda", dtype=torch.uint8, generator=gen
    )
    s = torch.randint(
        0x30, 0x50, (E, N, K // 16), device="cuda", dtype=torch.uint8, generator=gen
    )
    pts = torch.rand(E, device="cuda", generator=gen) * 1e-3 + 1e-4
    a = (torch.randn(E, r, K, device="cuda", generator=gen) * 0.01).bfloat16()
    b = (torch.randn(E, N, r, device="cuda", generator=gen) * 0.02).bfloat16()
    a_rm = a.reshape(E * r, K)
    b_rm = b.permute(1, 2, 0).reshape(N, r * E).contiguous()
    which = os.environ.get("PROBE_KERNELS", "triton,cuda,cuda_fused").split(",")
    for _ in range(3):
        if "triton" in which:
            tr.requant_lora_packed(q, s, pts, a_rm, b_rm, 2.0, pts)
        if "cuda" in which:
            cr.requant_natural(q, s, pts, a, b, 2.0, pts)
        if "cuda_fused" in which:
            cr.requant_fused(q, s, pts, a, b, 2.0, interleave=8)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
