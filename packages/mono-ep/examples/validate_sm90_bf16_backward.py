# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Check the SM90 BF16 persistent (native) backward at one layer shape on two Hopper GPUs.

Runs forward + backward through the runtime with the native backward on balanced routing and on
skewed routing (every token to rank 0's experts, so rank 0's ring wraps) and compares dx, dscore,
dW1 and dW2 against FP32 autograd over the same routed inputs from both ranks. A second backward
in the same process is the run-to-run control (dscore and dW may use atomics); the first
backward's gradients are saved so two builds of the kernel can be compared across processes.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def reference(x, ids, scores, w1, w2, dy):
    """FP32 autograd of sum_k score_k * SwiGLU expert(x) over all experts, for all ranks' tokens."""
    x, scores, w1, w2 = (
        t.detach().float().requires_grad_() for t in (x, scores, w1, w2)
    )
    out = torch.zeros(x.shape[0], w2.shape[1], device=x.device)
    for expert in range(w1.shape[0]):
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        gate, up = (x[rows] @ w1[expert].T).chunk(2, -1)
        h = torch.nn.functional.silu(gate) * up * scores[rows, slots, None]
        out = out.index_add(0, rows, h @ w2[expert].T)
    out.backward(dy.float())
    return x.grad, scores.grad, w1.grad, w2.grad


def gather_rows(tensor, counts, capacity):
    padded = torch.zeros(capacity, *tensor.shape[1:], device="cuda", dtype=tensor.dtype)
    if tensor.dtype == torch.long:
        padded.fill_(-1)
    padded[: tensor.shape[0]] = tensor
    peers = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(peers, padded)
    return torch.cat([p[:c] for p, c in zip(peers, counts, strict=True)])


def main(args):
    from mono_ep.runtime import PersistentMoE

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    name = torch.cuda.get_device_name()
    assert torch.cuda.get_device_capability()[0] == 9, name
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    assert world == 2, world
    hidden, inner, experts, topk = (
        args.hidden,
        args.intermediate,
        args.experts,
        args.topk,
    )
    local = experts // world
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        args.tokens,
        topk,
        hidden,
        inner,
        compute_dtype="bf16",
        fuse_activation=True,
        backward_backend="native",
    )
    torch.manual_seed(77)
    w1_all = (
        torch.randn(experts, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    w2_all = (
        torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
    ).bfloat16()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = [args.tokens - 37 * r for r in range(world)]
    for routing in ("balanced", "skewed"):
        torch.manual_seed(100 + rank)
        count = counts[rank]
        x = torch.randn(count, hidden, device="cuda", dtype=torch.bfloat16)
        ids = torch.rand(count, experts, device="cuda").argsort(-1)[:, :topk]
        if routing == "skewed":
            ids = torch.rand(count, local, device="cuda").argsort(-1)[:, :topk]
        ids = ids.contiguous()
        ids[::19, -1] = -1
        scores = torch.rand(count, topk, device="cuda").softmax(-1)
        dy = torch.randn(count, hidden, device="cuda", dtype=torch.bfloat16)
        grads = []
        for _ in range(2):
            xg, sg = x.clone().requires_grad_(), scores.clone().requires_grad_()
            w1 = w1_all[rank * local : (rank + 1) * local].clone().requires_grad_()
            w2 = w2_all[rank * local : (rank + 1) * local].clone().requires_grad_()
            dist.barrier()
            runtime(xg, ids, sg, w1, w2).backward(dy)
            torch.cuda.synchronize()
            grads.append([t.grad.clone() for t in (xg, sg, w1, w2)])
        rx, rs, rw1, rw2 = reference(
            gather_rows(x, counts, args.tokens),
            gather_rows(ids, counts, args.tokens).clamp_min(0),
            gather_rows(scores * (ids >= 0), counts, args.tokens),
            w1_all,
            w2_all,
            gather_rows(dy, counts, args.tokens),
        )
        start = sum(counts[:rank])
        expected = dict(
            dx=rx[start : start + count],
            dscore=(rs[start : start + count] * (ids >= 0)),
            dw1=rw1[rank * local : (rank + 1) * local],
            dw2=rw2[rank * local : (rank + 1) * local],
        )
        actual = dict(zip(("dx", "dscore", "dw1", "dw2"), grads[0], strict=True))
        actual["dscore"] = actual["dscore"] * (ids >= 0)
        errors = {
            key: (
                (actual[key].float() - expected[key]).norm()
                / expected[key].norm().clamp_min(1e-12)
            ).item()
            for key in expected
        }
        for key, value in actual.items():
            torch.save(
                value.cpu(),
                out_dir / f"bwd_{args.shape_name}_{routing}_{key}_rank{rank}.pt",
            )
        histogram = torch.bincount(ids[ids >= 0], minlength=experts)
        dist.all_reduce(histogram)
        mine = histogram[rank * local : (rank + 1) * local]
        used_pool_rows = int((((mine + 127) // 128) * 128).sum().item())
        ring_tokens = int(runtime.buffer.l1_acts.shape[0])
        record = dict(
            shape=args.shape_name,
            rank=rank,
            device=name,
            routing=routing,
            tokens=count,
            hidden=hidden,
            intermediate=inner,
            experts=experts,
            topk=topk,
            rel_error_vs_fp32_reference=errors,
            rerun_bitwise={
                key: bool(torch.equal(a, b))
                for key, a, b in zip(
                    ("dx", "dscore", "dw1", "dw2"), *grads, strict=True
                )
            },
            finite=all(bool(torch.isfinite(t.float()).all()) for t in grads[0]),
            ring_tokens=ring_tokens,
            used_pool_rows=used_pool_rows,
            ring_wrapped=used_pool_rows > ring_tokens,
        )
        print("SM90_BF16_BACKWARD " + json.dumps(record), flush=True)
        assert record["finite"] and max(errors.values()) < 0.03, record
        if routing == "skewed" and rank == 0:
            assert record["ring_wrapped"], record
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-name", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--experts", type=int, default=64, help="total experts")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=4096, help="max tokens per rank")
    parser.add_argument("--out-dir", required=True)
    main(parser.parse_args())
