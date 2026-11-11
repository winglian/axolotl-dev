# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Shared pieces of the NVFP4 LoRA workspace probes: a random packed NVFP4 layer with
adapters at a published MoE shape, the routing cases, and reductions."""

import torch
import torch.distributed as dist
from mok_vs_mono_ep import SHAPES, _skewed_routing

RANK, SCALING = 16, 2.0
# NVFP4 LoRA training geometries (per rank): the Qwen3-30B-A3B Axolotl run of the wiring session and Beam
TRAIN_SHAPES = {
    "30b": dict(tokens=3840, hidden=2048, intermediate=768, experts=128, topk=8),
    "beam": dict(tokens=3840, hidden=5120, intermediate=2560, experts=256, topk=8),
}


def shape(name, tokens=None):
    base = dict(TRAIN_SHAPES.get(name) or SHAPES[name])
    base.setdefault("topk", 8)
    if tokens:
        base["tokens"] = tokens
    return base


def random_layer(local, hidden, inter, device, seed, backend=None, scaling=SCALING):
    """A frozen packed NVFP4 base (``Nvfp4LoraExperts``) for ``local`` experts plus fp32
    adapters ``(a1, b1, a2, b2)`` at LoRA rank ``RANK``."""
    from mono_ep.nvfp4_lora import Nvfp4LoraExperts, requant_modules

    requant = requant_modules().requant
    gen = torch.Generator(device=device).manual_seed(seed)

    def quantized(n, k):
        # one expert at a time: the fp32 round trip of a whole Beam layer is GBs
        packed = torch.empty(local, n, k // 2, device=device, dtype=torch.uint8)
        scale = torch.empty(local, n, k // 16, device=device, dtype=torch.float8_e4m3fn)
        pts = torch.empty(local, device=device, dtype=torch.float32)
        for e in range(local):
            w = torch.randn(n, k, device=device, generator=gen) * 0.02
            pts[e] = requant.per_tensor_scale_from_amax(w.abs().amax().reshape(1))[0]
            q, s = requant.quantize_nvfp4_packed_ref(w, pts[e].expand(n))
            packed[e].copy_(q)
            scale[e].copy_(s.view(torch.float8_e4m3fn))
            del w, q, s
        return packed, scale, pts

    q1, s1, pts1 = quantized(2 * inter, hidden)
    q2, s2, pts2 = quantized(hidden, inter)
    layer = Nvfp4LoraExperts(
        q1, s1, pts1, q2, s2, pts2, scaling=scaling, backend=backend
    )
    adapters = (
        torch.randn(local, RANK, hidden, device=device, generator=gen) / hidden**0.5,
        torch.randn(local, 2 * inter, RANK, device=device, generator=gen) * 0.02,
        torch.randn(local, RANK, inter, device=device, generator=gen) / inter**0.5,
        torch.randn(local, hidden, RANK, device=device, generator=gen) * 0.02,
    )
    return layer, adapters


def routing(case, tokens, experts, topk, rank):
    """``balanced``, ``skewed`` (load skew towards low expert ids) or ``adversarial``
    (every token to rank 0's experts)."""
    local = experts // dist.get_world_size()
    if case == "skewed":
        return _skewed_routing(tokens, experts, topk, rank)
    if case == "adversarial":
        return (
            torch.rand(tokens, local, device="cuda").argsort(-1)[:, :topk].contiguous()
        )
    return torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk].contiguous()


def reduce_max(values):
    t = torch.tensor(values, device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))
