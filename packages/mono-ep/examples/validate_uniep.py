# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Independent BF16 parity and outstanding-forward checks for fused EP."""

import argparse
import json
import os

import torch
import torch.distributed as dist


def reference(x, scores, ids, w1, w2):
    y = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        token, slot = torch.where(ids == expert)
        gate, up = (x[token] @ w1[expert].T).float().chunk(2, dim=-1)
        h = (torch.nn.functional.silu(gate) * up * scores[token, slot, None]).to(
            x.dtype
        )
        y = y.index_add(0, token, (h @ w2[expert].T).float())
    return y.to(x.dtype)


def training_check(w1, w2, group, tokens, topk, progress):
    from triton_dist.function.nvidia.ep_moe_fused import TritonDistFusedEpMoeFunction

    rank, world = dist.get_rank(group), dist.get_world_size(group)
    local_experts, _, hidden = w1.shape
    experts = local_experts * world
    reference_weights = [w.detach().clone().requires_grad_() for w in (w1, w2)]
    optimizer = torch.optim.AdamW([w1, w2], lr=1e-3, foreach=False)
    reference_optimizer = torch.optim.AdamW(reference_weights, lr=1e-3, foreach=False)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        gathered = []
        for weight in reference_weights:
            parts = [torch.empty_like(weight) for _ in range(world)]
            dist.all_gather(parts, weight.detach(), group=group)
            gathered.append(torch.cat(parts).requires_grad_())
        x = torch.randn(tokens, hidden, device="cuda", dtype=w1.dtype).requires_grad_()
        scores, ids = torch.randn(tokens, experts, device="cuda").softmax(-1).topk(topk)
        ids = ids.int()
        scores.requires_grad_()
        xr, sr = [t.detach().clone().requires_grad_() for t in (x, scores)]
        target = torch.randn_like(x)
        expected = reference(xr, sr, ids, *gathered)
        reference_loss = (expected.float() - target.float()).square().mean()
        reference_loss.backward()
        for weight, full_weight in zip(reference_weights, gathered, strict=True):
            grad = full_weight.grad.float()
            dist.all_reduce(grad, group=group)
            weight.grad = grad[rank * local_experts : (rank + 1) * local_experts].to(
                weight.dtype
            )
        actual = TritonDistFusedEpMoeFunction.apply(
            experts, scores, ids, x, w1, None, w2, group
        )
        loss = (actual.float() - target.float()).square().mean()
        loss.backward()
        gradient_errors = torch.stack(
            [
                (a.grad.float() - b.grad.float()).norm()
                / b.grad.float().norm().clamp_min(1e-8)
                for a, b in zip(
                    (x, scores, w1, w2), (xr, sr, *reference_weights), strict=True
                )
            ]
        )
        dist.all_reduce(gradient_errors, op=dist.ReduceOp.MAX, group=group)
        assert torch.isfinite(gradient_errors).all() and gradient_errors.max() < 0.03
        optimizer.step()
        reference_optimizer.step()
        weight_errors = torch.stack(
            [
                (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
                for a, b in zip((w1, w2), reference_weights, strict=True)
            ]
        )
        dist.all_reduce(weight_errors, op=dist.ReduceOp.MAX, group=group)
        assert torch.isfinite(weight_errors).all() and weight_errors.max() < 0.01
        for state in optimizer.state.values():
            assert all(torch.isfinite(value).all() for value in state.values())
        progress(
            "optimizer_step",
            step=step,
            loss=loss.item(),
            reference_loss=reference_loss.item(),
            gradient_relative_l2=gradient_errors.tolist(),
            weight_relative_l2=weight_errors.tolist(),
        )


def run(inflight, training=False):
    from triton_dist.function.nvidia.common import (
        deinit_triton_dist_ep_op,
        get_triton_dist_ep_op,
        init_triton_dist_ep_op,
    )
    from triton_dist.function.nvidia.ep_moe_fused import TritonDistFusedEpMoeFunction
    from triton_dist.kernels.nvidia.swiglu import swiglu_backward, swiglu_forward
    from triton_dist.utils import (
        finalize_distributed,
        init_nvshmem_by_torch_process_group,
    )

    rank = int(os.environ["LOCAL_RANK"])

    def progress(stage, **details):
        print(json.dumps({"rank": rank, "stage": stage, **details}), flush=True)

    progress("distributed_init", torch_version=torch.__version__)
    torch.cuda.set_device(rank)
    for margin in (0, 1):
        z = torch.randn(137, 512, device="cuda", dtype=torch.bfloat16)
        scale = torch.rand(137, device="cuda")
        dy = torch.randn(137, 256, device="cuda", dtype=torch.bfloat16)
        zr = z.float().requires_grad_()
        sr = scale.detach().clone().requires_grad_()
        gate, up = zr.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate) * up * sr[:, None]
        dz_ref, ds_ref = torch.autograd.grad(expected, (zr, sr), dy.float())
        actual, ctx = swiglu_forward(z, scale, sm_margin=margin)
        dz, ds = swiglu_backward(dy, z, scale, ctx, sm_margin=margin)
        errors = [
            ((a.float() - b.float()).norm() / b.float().norm()).item()
            for a, b in ((actual, expected), (dz, dz_ref), (ds, ds_ref))
        ]
        progress("swiglu_check", margin=margin, relative_l2=errors)
        assert all(0 <= error < 0.01 for error in errors), errors
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    world = dist.get_world_size()
    if world not in (4, 8):
        raise ValueError("Pinned UniEP fused backward requires four or eight ranks")
    experts, hidden, inner, tokens, topk = 8, 256, 256, 128, 2
    assert experts % world == 0
    local_experts = experts // world
    init_nvshmem_by_torch_process_group(group)
    progress("nvshmem_initialized")
    init_triton_dist_ep_op(
        group,
        tokens,
        hidden,
        topk,
        rank,
        experts,
        world,
        dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        num_sm=64,
        num_buffers=1,
        capacity=4.0,
    )
    torch.manual_seed(731 + rank)
    progress("operator_initialized")
    w1 = (
        (torch.randn(local_experts, 2 * inner, hidden, device="cuda") / hidden**0.5)
        .bfloat16()
        .requires_grad_()
    )
    w2 = (
        (torch.randn(local_experts, hidden, inner, device="cuda") / inner**0.5)
        .bfloat16()
        .requires_grad_()
    )
    gathered = []
    for w in (w1, w2):
        parts = [torch.empty_like(w) for _ in range(world)]
        dist.all_gather(parts, w.detach())
        gathered.append(torch.cat(parts))
    pending = []
    for step in range(2 if inflight else 1):
        x = torch.randn(
            tokens, hidden, dtype=torch.bfloat16, device="cuda"
        ).requires_grad_()
        scores, ids = torch.randn(tokens, experts, device="cuda").softmax(-1).topk(topk)
        ids = ids.int()
        scores.requires_grad_()
        dy = torch.randn_like(x)
        refs = [t.detach().clone().requires_grad_() for t in (x, scores, *gathered)]
        expected = reference(refs[0], refs[1], ids, refs[2], refs[3])
        ref_grads = list(torch.autograd.grad(expected, refs, dy))
        for i in (2, 3):
            full_grad = ref_grads[i].float()
            dist.all_reduce(full_grad)
            ref_grads[i] = full_grad[rank * local_experts : (rank + 1) * local_experts]
        actual = TritonDistFusedEpMoeFunction.apply(
            experts, scores, ids, x, w1, None, w2, group
        )
        progress("forward_enqueued", step=step)
        pending.append((step, actual, (x, scores, w1, w2), dy, expected, ref_grads))
    for step, actual, inputs, dy, expected, ref_grads in reversed(pending):
        progress("backward_start", step=step)
        grads = torch.autograd.grad(actual, inputs, dy)
        errors = torch.stack(
            [
                (got.float() - want.float()).norm()
                / want.float().norm().clamp_min(1e-8)
                for got, want in zip(
                    (actual, *grads), (expected, *ref_grads), strict=True
                )
            ]
        )
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "step": step,
                    "inflight": inflight,
                    "relative_l2": dict(
                        zip(
                            ("y", "dx", "ds", "dw1", "dw2"),
                            errors.tolist(),
                            strict=True,
                        )
                    ),
                }
            ),
            flush=True,
        )
        assert torch.isfinite(errors).all() and errors.max() < 0.03, errors
    if training:
        training_check(w1, w2, group, tokens, topk, progress)
    torch.cuda.synchronize()
    dist.barrier()
    # This fixed-capacity probe never resizes buffers; free each owned allocation once.
    operator = get_triton_dist_ep_op()
    allocator = operator._nvshmem_allocator
    import nvshmem.core

    memory = nvshmem.core.finalize.__globals__["memory"]
    owned = {
        allocation.get_underlying_tensor().data_ptr(): allocation._spec.name
        for allocation in allocator._lazy_tensors
    }
    if operator.barrier_all_ctx.is_intra_node:
        owned[operator.barrier_all_ctx.symm_barrier.data_ptr()] = "intra_node_barrier"
    operator.barrier_all_ctx.finalize()
    for allocation in allocator._lazy_tensors:
        allocator.free_tensor(allocation)
    remaining = [
        {
            "owner": owned.get(pointer, "outside_operator"),
            "type": str(record["type"]),
            "ref_count": record["ref_count"],
        }
        for resource in memory._mr_references.values()
        for pointer, record in resource._mem_references.items()
        if not record["freed"]
    ]
    progress("symmetric_cleanup", owned_count=len(owned), remaining=remaining)
    assert not remaining, remaining
    deinit_triton_dist_ep_op()
    finalize_distributed()
    progress("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inflight", action="store_true")
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    run(args.inflight, args.training)
