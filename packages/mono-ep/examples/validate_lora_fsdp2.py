# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""dp_shard x EP training check of the fused LoRA path (``bf16_lora`` / ``mxfp8_lora``).

One decoder-layer stand-in (frozen dense projection and router, frozen BF16 experts of this
EP rank, trainable per-expert LoRA factors) is ``fully_shard``-ed as a whole on the
``dp_shard`` mesh, so the base experts are sharded frozen parameters regathered per forward
and backward and the adapters are sharded DTensors. AdamW trains the factors for ``--steps``
optimizer steps with the merged-weight cache invalidated by the post-step hook. Every step
the loss and the adapter gradients are compared with an FP32 torch reference computed from
the live (gathered) weights and the routing the layer used; the merged weights the cache
holds must equal a fresh merge of the live adapters, the base must not move, and the merge
must change after every step. Run under torchrun on 4 or 8 B200 (``--ep 2`` or ``--ep 4``).
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

EXPERTS, HIDDEN, INNER, TOPK, RANK, SCALING = 16, 1024, 512, 4, 16, 2.0


def _gather(t, group):
    t = t.detach().contiguous()
    out = t.new_empty((dist.get_world_size(group) * t.shape[0], *t.shape[1:]))
    dist.all_gather_into_tensor(out, t, group=group)
    return out


def _full(p):
    p = p.detach()
    return p.full_tensor() if hasattr(p, "full_tensor") else p


class Layer(nn.Module):
    def __init__(self, local, runtime, experts, cache):
        super().__init__()
        self.pre = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.router = nn.Linear(HIDDEN, EXPERTS, bias=False)
        self.w1 = nn.Parameter(
            torch.randn(local, 2 * INNER, HIDDEN) / HIDDEN**0.5, requires_grad=False
        )
        self.w2 = nn.Parameter(
            torch.randn(local, HIDDEN, INNER) / INNER**0.5, requires_grad=False
        )
        self.a1 = nn.Parameter(torch.randn(local, RANK, HIDDEN) / HIDDEN**0.5)
        self.b1 = nn.Parameter(torch.randn(local, 2 * INNER, RANK) * 0.02)
        self.a2 = nn.Parameter(torch.randn(local, RANK, INNER) / INNER**0.5)
        self.b2 = nn.Parameter(torch.randn(local, HIDDEN, RANK) * 0.02)
        for p in (self.pre.weight, self.router.weight):
            p.requires_grad_(False)
        self.runtime, self.experts, self.cache = runtime, experts, cache
        self.seen = None

    def forward(self, x):
        from mono_ep.lora import lora_moe

        h = self.pre(x)
        logits = self.router(h).float()
        scores, ids = logits.softmax(-1).topk(TOPK, dim=-1)
        self.seen = (h.detach(), ids, scores.detach())
        return lora_moe(
            h,
            ids,
            scores.contiguous(),
            self.w1,
            self.w2,
            self.a1,
            self.b1,
            self.a2,
            self.b2,
            self.runtime,
            self.experts,
            self.cache,
        )


def moe_fp32(x, ids, scores, w1, w2):
    y = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for e in range(w1.shape[0]):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        gate, up = (x[tok] @ w1[e].t()).chunk(2, dim=-1)
        y.index_add_(
            0, tok, ((F.silu(gate) * up) @ w2[e].t()) * scores[tok, slot, None]
        )
    return y


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--compute-dtype", choices=("bf16", "mxfp8"), default="mxfp8")
    parser.add_argument("--merged-cache", choices=("layer", "all"), default="layer")
    parser.add_argument("--reserved-sms", type=int, default=16)
    parser.add_argument("--loss-bound", type=float, default=None)
    parser.add_argument("--grad-bound", type=float, default=None)
    args = parser.parse_args()
    bounds = dict(bf16=(1e-2, 2e-2), mxfp8=(0.1, 0.1))[args.compute_dtype]
    loss_bound = args.loss_bound or bounds[0]
    grad_bound = args.grad_bound or bounds[1]
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    mesh = init_device_mesh(
        "cuda", (world // args.ep, args.ep), mesh_dim_names=("dp_shard", "ep")
    )
    ep_group, dp_group = mesh["ep"].get_group(), mesh["dp_shard"].get_group()
    ep, dp = mesh["ep"].size(), mesh["dp_shard"].size()
    ep_rank = mesh["ep"].get_local_rank()
    local = EXPERTS // ep

    import deep_gemm
    from mono_ep.api import check_nccl_env
    from mono_ep.lora import (
        LoraExperts,
        MergedWeightCache,
        attach_post_step_hook,
        merge,
    )
    from mono_ep.runtime import PersistentMoE

    check_nccl_env(reserved_sms=args.reserved_sms)
    physical = torch.cuda.get_device_properties(device).multi_processor_count
    deep_gemm.set_num_sms(physical - args.reserved_sms)
    runtime = PersistentMoE(
        ep_group,
        EXPERTS,
        args.tokens,
        TOPK,
        HIDDEN,
        INNER,
        compute_dtype=args.compute_dtype,
        fuse_activation=True,
        backward_backend="native",
        reserved_sms=args.reserved_sms,
    )
    cache = MergedWeightCache(mode=args.merged_cache)
    experts = LoraExperts(scaling=SCALING)
    # dense weights shared by every rank, experts and adapters per EP rank
    torch.manual_seed(7)
    pre = torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5
    router = torch.randn(EXPERTS, HIDDEN) / HIDDEN**0.5
    torch.manual_seed(100 + ep_rank)
    layer = Layer(local, runtime, experts, cache)
    with torch.no_grad():
        layer.pre.weight.copy_(pre)
        layer.router.weight.copy_(router)
    layer = layer.to(device)
    for p in (layer.w1, layer.w2):
        p.data = p.data.bfloat16()
    base = [layer.w1.detach().clone(), layer.w2.detach().clone()]
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
    )
    fully_shard(layer, mesh=mesh["dp_shard"], mp_policy=mp)
    if dp > 1:
        layer.set_force_sum_reduction_for_comms(True)
    layer.set_gradient_divide_factor(float(ep * dp))
    adapters = (layer.a1, layer.b1, layer.a2, layer.b2)
    optimizer = torch.optim.AdamW(adapters, lr=args.lr, weight_decay=0.0)
    attach_post_step_hook(optimizer, cache)

    gen = torch.Generator(device="cpu").manual_seed(4321 + dist.get_rank())
    x = torch.randn(args.tokens, HIDDEN, generator=gen).to(device)
    target = (torch.randn(args.tokens, HIDDEN, generator=gen) * 0.1).to(device)

    results, prev_merged = [], None
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        y = layer(x)
        loss = (y.float() * target).sum()
        loss.backward()
        torch.cuda.synchronize()

        with torch.no_grad():
            full = [_full(p).float() for p in adapters]
            grads = [_full(p.grad).float() for p in adapters]
            w1, w2 = (_full(p) for p in (layer.w1, layer.w2))
        h, ids, scores = layer.seen
        a1, b1, a2, b2 = full
        m1 = w1.float() + SCALING * b1.bfloat16().float() @ a1.bfloat16().float()
        m2 = w2.float() + SCALING * b2.bfloat16().float() @ a2.bfloat16().float()
        m1_all, m2_all = _gather(m1, ep_group), _gather(m2, ep_group)
        # loss of this rank's tokens over every expert of the EP group
        y_ref = moe_fp32(h.float(), ids, scores, m1_all, m2_all)
        loss_ref = (y_ref * target).sum()
        # adapter gradients of the local experts: all EP-group tokens, dp-summed and divided like FSDP2
        h_all, ids_all, s_all, t_all = (
            _gather(t, ep_group) for t in (h, ids, scores, target)
        )
        ref1 = m1_all.clone().requires_grad_()
        ref2 = m2_all.clone().requires_grad_()
        (moe_fp32(h_all.float(), ids_all, s_all, ref1, ref2) * t_all).sum().backward()
        dw1, dw2 = ref1.grad, ref2.grad
        for ref in (dw1, dw2):
            dist.all_reduce(ref, group=dp_group)
            ref.div_(ep * dp)
        block = slice(ep_rank * local, (ep_rank + 1) * local)
        dw1, dw2 = dw1[block], dw2[block]
        refs = (
            SCALING * b1.transpose(1, 2) @ dw1,
            SCALING * dw1 @ a1.transpose(1, 2),
            SCALING * b2.transpose(1, 2) @ dw2,
            SCALING * dw2 @ a2.transpose(1, 2),
        )
        errs = dict(
            # the loss is a dot product near zero at init: normalize by its scale
            loss=abs(float(loss) - float(loss_ref))
            / float(y_ref.norm() * target.norm()),
            y=rel(y, y_ref),
            **{
                k: rel(g, r)
                for k, g, r in zip(
                    ("dA1", "dB1", "dA2", "dB2"), grads, refs, strict=True
                )
            },
        )
        entry = cache.entries.get(id(experts))
        live = [_full(p) for p in adapters]
        fresh = (
            merge(w1, live[1], live[0], SCALING),
            merge(w2, live[3], live[2], SCALING),
        )
        cached_ok = entry is not None and all(
            torch.equal(a, b) for a, b in zip((entry.w1, entry.w2), fresh, strict=True)
        )
        changed = prev_merged is None or not torch.equal(prev_merged, fresh[0])
        base_ok = torch.equal(w1, base[0]) and torch.equal(w2, base[1])
        worst = torch.tensor(list(errs.values()), device=device)
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        flags = torch.tensor(
            [cached_ok, changed, base_ok], device=device, dtype=torch.int32
        )
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        worst = dict(zip(errs, worst.tolist(), strict=True))
        passed = (
            all(bool(f) for f in flags)
            and worst["loss"] <= loss_bound
            and worst["y"] <= loss_bound
            and all(worst[k] <= grad_bound for k in ("dA1", "dB1", "dA2", "dB2"))
            and cache.version == step
            and all(p.grad is None for p in (layer.w1, layer.w2))
        )
        results.append(
            dict(
                step=step,
                loss=float(loss),
                loss_ref=float(loss_ref),
                worst=worst,
                cached_equals_fresh_merge=bool(flags[0]),
                merged_changed_since_previous_step=bool(flags[1]),
                base_unchanged=bool(flags[2]),
                cache_version=cache.version,
                passed=passed,
            )
        )
        if dist.get_rank() == 0:
            print(json.dumps(results[-1]), flush=True)
        prev_merged = fresh[0].clone()
        if step < args.steps:
            optimizer.step()
    torch.cuda.synchronize()
    runtime.close()
    summary = dict(
        geometry=dict(
            world=world,
            ep=ep,
            dp_shard=dp,
            experts=EXPERTS,
            hidden=HIDDEN,
            inter=INNER,
            topk=TOPK,
            rank=RANK,
            tokens_per_rank=args.tokens,
            compute_dtype=args.compute_dtype,
            merged_cache=cache.mode,
            optimizer_steps=args.steps,
        ),
        cache=cache.stats(),
        steps=results,
        passed=all(r["passed"] for r in results),
    )
    if dist.get_rank() == 0:
        print("LORA_FSDP2_JSON " + json.dumps(summary), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if not summary["passed"]:
        raise SystemExit("fused LoRA FSDP2 validation failed")


if __name__ == "__main__":
    main()
