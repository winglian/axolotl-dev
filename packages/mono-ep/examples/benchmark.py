# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""EP=1 training-layer comparison with fixed routing and matched BF16 weights.

Run each backend in a fresh process. This measures neither multi-GPU EP nor
end-to-end model training; both need separate comparisons.
"""

import argparse
import json
import os
import statistics
import tempfile
import time

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


def torch_experts(x, scores, w1, w2, ids):
    experts, topk = w1.shape[0], ids.shape[1]
    order = ids.flatten().argsort(stable=True)
    counts = torch.zeros(experts, dtype=torch.int32, device=x.device)
    counts.scatter_add_(
        0, ids.flatten(), torch.ones_like(ids.flatten(), dtype=torch.int32)
    )
    offsets = counts.cumsum(0, dtype=torch.int32)
    packed = x[order // topk]
    z = torch._grouped_mm(packed, w1.transpose(1, 2), offs=offsets)
    gate, up = z.float().chunk(2, dim=-1)
    h = (torch.nn.functional.silu(gate) * up * scores.flatten()[order, None]).to(
        x.dtype
    )
    out = torch._grouped_mm(h, w2.transpose(1, 2), offs=offsets)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=x.device)
    return out[inverse].view(x.shape[0], topk, x.shape[1]).float().sum(1).to(x.dtype)


def validate_torch_baseline(compiled):
    torch.manual_seed(19)
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scores = torch.randn(17, 4, device="cuda", requires_grad=True)
    ids = torch.rand(17, 8, device="cuda").topk(4).indices
    w1 = (
        torch.randn(8, 512, 256, device="cuda", dtype=torch.bfloat16) / 16
    ).requires_grad_()
    w2 = (
        torch.randn(8, 256, 256, device="cuda", dtype=torch.bfloat16) / 16
    ).requires_grad_()
    expected = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(8):
        token, slot = torch.where(ids == expert)
        gate, up = (x[token] @ w1[expert].T).float().chunk(2, dim=-1)
        h = (torch.nn.functional.silu(gate) * up * scores[token, slot, None]).to(
            x.dtype
        )
        expected = expected.index_add(0, token, (h @ w2[expert].T).float())
    expected = expected.to(x.dtype)
    operation = (
        torch.compile(torch_experts, fullgraph=True) if compiled else torch_experts
    )
    actual = operation(x, scores, w1, w2, ids)
    dy = torch.randn_like(x)
    leaves = (x, scores, w1, w2)
    pairs = zip(
        (actual, *torch.autograd.grad(actual, leaves, dy)),
        (expected, *torch.autograd.grad(expected, leaves, dy)),
        strict=True,
    )
    errors = []
    for value, reference in pairs:
        error = (
            (value.float() - reference.float()).norm()
            / reference.float().norm().clamp_min(1e-8)
        ).item()
        assert error < 0.02, error
        errors.append(error)
    return errors


def run(args):
    validation = (
        validate_torch_baseline(args.compile) if args.backend == "torch" else None
    )
    torch.manual_seed(72)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(
        args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    scores, ids = (
        torch.randn(args.tokens, args.experts, device="cuda")
        .softmax(-1)
        .topk(args.topk)
    )
    scores = scores.detach().requires_grad_()
    w1 = (
        torch.randn(
            args.experts,
            args.inner * 2,
            args.hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / args.hidden**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(
            args.experts, args.hidden, args.inner, device="cuda", dtype=torch.bfloat16
        )
        / args.inner**0.5
    ).requires_grad_()
    dy = torch.randn_like(x)
    workspace_bytes = 0
    runtime = None
    if args.backend == "torch":

        def forward(x, scores, w1, w2):
            return torch_experts(x, scores, w1, w2, ids)
    elif args.backend == "sonic":
        from sonicmoe import moe_general_routing_inputs
        from sonicmoe.enums import ActivationType

        token_ids = torch.arange(
            args.tokens, device="cuda", dtype=torch.int32
        ).repeat_interleave(args.topk)
        expert_ids = ids.flatten().int()

        def forward(x, scores, w1, w2):
            return moe_general_routing_inputs(
                x,
                scores.flatten(),
                token_ids,
                expert_ids,
                w1.permute(1, 2, 0),
                None,
                w2.permute(1, 2, 0),
                None,
                args.experts,
                None,
                ActivationType.SWIGLU,
                concat_layout=True,
            )[0]
    else:
        from mono_ep import PersistentMoE

        rendezvous = os.path.join(tempfile.mkdtemp(prefix="moe-bench-"), "rendezvous")
        dist.init_process_group(
            "nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1
        )
        runtime = PersistentMoE(
            dist.group.WORLD,
            args.experts,
            args.tokens,
            args.topk,
            args.hidden,
            args.inner,
            fuse_activation=not args.unfused_backward,
        )
        workspace_bytes = runtime.buffer.buffer.numel()

        def forward(x, scores, w1, w2):
            return runtime(x, ids, scores, w1, w2)

    if args.compile:
        forward = torch.compile(forward, fullgraph=True)
    inputs = (x, scores, w1, w2)

    def step():
        y = (
            checkpoint(forward, *inputs, use_reentrant=False)
            if args.checkpoint
            else forward(*inputs)
        )
        grads = torch.autograd.grad(y, inputs, dy)
        return y, grads

    first_y, first_grads = step()
    for t in (first_y, *first_grads):
        if not torch.isfinite(t).all().item():
            raise RuntimeError("nonfinite output or gradient")
    del first_y, first_grads
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    print(
        json.dumps(
            {
                "backend": args.backend,
                "scope": "EP=1 forward+backward",
                "device": torch.cuda.get_device_name(),
                "dtype": "bfloat16",
                "checkpoint": args.checkpoint,
                "compiled": args.compile,
                "validation_relative_l2": validation,
                "fused_backward_activation": not args.unfused_backward
                if args.backend == "deepgemm"
                else None,
                "tokens": args.tokens,
                "hidden": args.hidden,
                "intermediate": args.inner,
                "experts": args.experts,
                "topk": args.topk,
                "median_ms": statistics.median(times),
                "min_ms": min(times),
                "allocated_baseline_bytes": baseline,
                "workspace_bytes": workspace_bytes,
                "peak_increment_bytes": torch.cuda.max_memory_allocated() - baseline,
                "iterations": args.iters,
            }
        )
    )
    if runtime is not None:
        runtime.buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("torch", "sonic", "deepgemm"), required=True
    )
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inner", type=int, default=768)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--unfused-backward", action="store_true")
    parser.add_argument("--compile", action="store_true")
    run(parser.parse_args())
