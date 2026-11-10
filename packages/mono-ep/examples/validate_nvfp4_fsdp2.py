# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""dp_shard x EP training check of the NVFP4 LoRA path: adapters under FSDP2
``fully_shard`` on the ``dp_shard`` mesh, frozen packed NVFP4 base per EP rank,
two AdamW steps with the requant cache invalidated by the optimizer post-step
hook. After every forward/backward the cached merged bytes must equal a fresh
requant of the live (gathered) adapters, and ``y``, ``dx``, ``dscore`` and the
adapter gradients are compared with an FP32 oracle on the dequantized merged
weights. Run under torchrun on 4 B200 (``--ep 2``)."""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_nvfp4_backward import oracle_grads  # noqa: E402
from validate_nvfp4_forward import RANK, SCALING, errors, oracle_fp32  # noqa: E402

EXPERTS, HIDDEN, INNER, TOPK = 16, 1024, 512, 4


def _gather(t, group):
    t = t.detach().contiguous()
    out = t.new_empty((dist.get_world_size(group) * t.shape[0], *t.shape[1:]))
    dist.all_gather_into_tensor(out, t, group=group)
    return out


class Adapters(nn.Module):
    def __init__(self, local, runtime, layer, cache):
        super().__init__()
        self.a1 = nn.Parameter(torch.randn(local, RANK, HIDDEN) / HIDDEN**0.5)
        self.b1 = nn.Parameter(torch.randn(local, 2 * INNER, RANK) * 0.02)
        self.a2 = nn.Parameter(torch.randn(local, RANK, INNER) / INNER**0.5)
        self.b2 = nn.Parameter(torch.randn(local, HIDDEN, RANK) * 0.02)
        self.runtime, self.layer, self.cache = runtime, layer, cache

    def forward(self, x, ids, scores):
        from mono_ep.nvfp4_lora import nvfp4_lora_moe

        return nvfp4_lora_moe(
            x,
            ids,
            scores,
            self.a1,
            self.b1,
            self.a2,
            self.b2,
            self.runtime,
            self.layer,
            self.cache,
        )


def merged_bytes(m):
    return [m.w1, m.w1_atoms, m.w2, m.w2_atoms, m.w1t, m.w1t_atoms, m.w2t, m.w2t_atoms]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--reserved-sms", type=int, default=16)
    parser.add_argument("--y-bound", type=float, default=0.25)
    parser.add_argument(
        "--grad-bound",
        type=float,
        default=float(os.environ.get("NVFP4_FSDP2_GRAD_BOUND", "0.35")),
    )
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    assert "B200" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
    world = dist.get_world_size()
    mesh = init_device_mesh(
        "cuda", (world // args.ep, args.ep), mesh_dim_names=("dp_shard", "ep")
    )
    ep_group = mesh["ep"].get_group()
    ep, dp = mesh["ep"].size(), mesh["dp_shard"].size()
    ep_rank = mesh["ep"].get_local_rank()
    local = EXPERTS // ep

    import deep_gemm
    from mono_ep.api import check_nccl_env
    from mono_ep.nvfp4_lora import (
        Nvfp4LoraExperts,
        RequantCache,
        attach_post_step_hook,
        requant_modules,
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
        compute_dtype="nvfp4_lora",
        reserved_sms=args.reserved_sms,
    )
    requant = requant_modules().requant

    # Frozen base per EP rank (identical across dp)
    gen = torch.Generator(device=device).manual_seed(1234 + ep_rank)
    w1 = torch.randn(local, 2 * INNER, HIDDEN, device=device, generator=gen) * 0.02
    w2 = torch.randn(local, HIDDEN, INNER, device=device, generator=gen) * 0.02
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
    base1 = torch.stack(
        [
            requant.dequantize_nvfp4_packed_ref(
                q1[e], s1[e], pts1[e].expand(2 * INNER), torch.float32
            )
            for e in range(local)
        ]
    )
    base2 = torch.stack(
        [
            requant.dequantize_nvfp4_packed_ref(
                q2[e], s2[e], pts2[e].expand(HIDDEN), torch.float32
            )
            for e in range(local)
        ]
    )
    cache = RequantCache()
    torch.manual_seed(99 + ep_rank)
    adapters = Adapters(local, runtime, layer, cache).to(device)
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    fully_shard(adapters, mesh=mesh["dp_shard"], mp_policy=mp)
    if dp > 1:
        adapters.set_force_sum_reduction_for_comms(True)
    adapters.set_gradient_divide_factor(float(ep * dp))
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=args.lr, weight_decay=0.0)
    attach_post_step_hook(optimizer, cache)

    data_gen = torch.Generator(device="cpu").manual_seed(4321 + dist.get_rank())
    x = torch.randn(args.tokens, HIDDEN, generator=data_gen).bfloat16().to(device)
    ids = (
        torch.rand(args.tokens, EXPERTS, generator=data_gen)
        .argsort(-1)[:, :TOPK]
        .to(device)
    )
    scores = torch.rand(args.tokens, TOPK, generator=data_gen).softmax(-1).to(device)
    dy = (
        (torch.randn(args.tokens, HIDDEN, generator=data_gen) * 0.1)
        .bfloat16()
        .to(device)
    )
    x_all, ids_all, scores_all, dy_all = (
        _gather(t, ep_group) for t in (x, ids, scores, dy)
    )

    results = []
    prev_merged, prev_weights = None, None
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        x_leaf = x.clone().requires_grad_()
        s_leaf = scores.clone().requires_grad_()
        y = adapters(x_leaf, ids, s_leaf)
        (y.float() * dy.float()).sum().backward()
        torch.cuda.synchronize()

        with torch.no_grad():
            full = [
                p.full_tensor()
                for p in (adapters.a1, adapters.b1, adapters.a2, adapters.b2)
            ]
            grads = [
                p.grad.full_tensor()
                for p in (adapters.a1, adapters.b1, adapters.a2, adapters.b2)
            ]
        cached = cache.entries[id(layer)]
        fresh = layer.merged(*full)
        bitwise = all(
            torch.equal(a, b)
            for a, b in zip(merged_bytes(cached), merged_bytes(fresh), strict=True)
        )
        changed = prev_merged is None or not all(
            torch.equal(a, b)
            for a, b in zip(merged_bytes(cached), prev_merged, strict=True)
        )
        a1, b1, a2, b2 = (t.float() for t in full)
        w1x = base1 + SCALING * b1 @ a1
        w2x = base2 + SCALING * b2 @ a2
        w1_all, w2_all = _gather(w1x, ep_group), _gather(w2x, ep_group)
        # Local tokens, all experts: y / dx / dscore
        y_ref = oracle_fp32(x.float(), ids, scores, w1_all, w2_all)
        dx_ref, ds_ref, _, _ = oracle_grads(x, ids, scores, w1_all, w2_all, dy)
        # All EP-group tokens, summed over dp, divided like FSDP2: adapter grads of local experts
        _, _, dw1_all, dw2_all = oracle_grads(
            x_all, ids_all, scores_all, w1_all, w2_all, dy_all
        )
        for ref in (dw1_all, dw2_all):
            dist.all_reduce(ref, group=mesh["dp_shard"].get_group())
            ref.div_(ep * dp)
        block = slice(ep_rank * local, (ep_rank + 1) * local)
        dw1, dw2 = dw1_all[block], dw2_all[block]
        refs = dict(
            dA1=SCALING * b1.transpose(1, 2) @ dw1,
            dB1=SCALING * dw1 @ a1.transpose(1, 2),
            dA2=SCALING * b2.transpose(1, 2) @ dw2,
            dB2=SCALING * dw2 @ a2.transpose(1, 2),
        )
        errs = dict(
            y=errors(y, y_ref)["rel_fro"],
            dx=errors(x_leaf.grad, dx_ref)["rel_fro"],
            dscore=errors(s_leaf.grad, ds_ref)["rel_fro"],
            **{
                k: errors(g, refs[k])["rel_fro"]
                for k, g in zip(refs, grads, strict=True)
            },
        )
        if prev_weights is not None:
            y_prev = oracle_fp32(x.float(), ids, scores, *prev_weights)
            errs["y_vs_previous_step_weights"] = errors(y, y_prev)["rel_fro"]
        worst = torch.tensor(list(errs.values()), device=device)
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        flags = torch.tensor([bitwise, changed], device=device, dtype=torch.int32)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        worst = dict(zip(errs, worst.tolist(), strict=True))
        passed = (
            bool(flags[0])
            and bool(flags[1])
            and worst["y"] <= args.y_bound
            and all(
                worst[k] <= args.grad_bound
                for k in ("dx", "dscore", "dA1", "dB1", "dA2", "dB2")
            )
            and (
                "y_vs_previous_step_weights" not in worst
                or worst["y_vs_previous_step_weights"] > worst["y"]
            )
            and cache.version == step
            and cache.builds == step + 1
        )
        results.append(
            dict(
                step=step,
                worst=worst,
                cached_equals_fresh_requant=bool(flags[0]),
                merged_changed_since_previous_step=bool(flags[1]),
                cache_version=cache.version,
                cache_builds=cache.builds,
                passed=passed,
            )
        )
        if rank == 0:
            print(json.dumps(results[-1]), flush=True)
        prev_merged = [t.clone() for t in merged_bytes(cached)]
        prev_weights = (w1_all.clone(), w2_all.clone())
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
            tokens_per_rank=args.tokens,
            rank=RANK,
            lr=args.lr,
            reserved_sms=args.reserved_sms,
            nccl_max_ctas=os.environ.get("NCCL_MAX_CTAS"),
        ),
        steps=results,
        passed=all(r["passed"] for r in results),
    )
    if rank == 0:
        print("FSDP2_JSON " + json.dumps(summary), flush=True)
        out = os.environ.get("NVFP4_OUT")
        if out:
            with open(out, "w") as f:
                json.dump(summary, f, indent=1)
    dist.barrier()
    dist.destroy_process_group()
    if not summary["passed"]:
        raise SystemExit("nvfp4 fsdp2 validation failed")


if __name__ == "__main__":
    main()
