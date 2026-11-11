# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Resident memory of one fused mono-ep runtime (forward + native backward) per rank, measured.

Run under torchrun (EP8). Builds the runtime under the current ``MONO_EP_WORKSPACE_BOUND``, runs
three forward + backward steps with the given routing, and reports ``mem_get_info`` device-used and
torch-allocated growth across construction and the steps, the pool rows, and the bound's state.
Prints ``WSMEM_JSON <payload>`` on rank 0.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from mok_vs_mono_ep import SHAPES, _skewed_routing


def _used():
    free, total = torch.cuda.mem_get_info()
    return total - free


def _max(value):
    t = torch.tensor([float(value)], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument(
        "--routing", choices=("balanced", "skewed", "adversarial"), default="balanced"
    )
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    shape = SHAPES[args.shape]
    tokens, hidden, inner, experts = (
        shape[k] for k in ("tokens", "hidden", "intermediate", "experts")
    )
    topk, local = 8, experts // world

    import deep_gemm
    from mono_ep.persistent_backward import NativePersistentBackward
    from mono_ep.runtime import PersistentMoE

    torch.manual_seed(812 + rank)
    x = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    if args.routing == "skewed":
        ids = _skewed_routing(tokens, experts, topk, rank)
    elif args.routing == "adversarial":
        ids = (
            torch.rand(tokens, local, device="cuda").argsort(-1)[:, :topk].contiguous()
        )
    else:
        ids = (
            torch.rand(tokens, experts, device="cuda")
            .argsort(-1)[:, :topk]
            .contiguous()
        )
    scores = torch.rand(tokens, topk, device="cuda")
    scores = (scores / scores.sum(-1, keepdim=True)).requires_grad_()
    w1 = (
        torch.randn(local, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(local, hidden, inner, device="cuda") / inner**0.5
    ).requires_grad_()
    dy = torch.randn_like(x)
    deep_gemm.set_num_sms(
        torch.cuda.get_device_properties(local_rank).multi_processor_count
    )
    torch.cuda.synchronize()
    used0, alloc0 = _used(), torch.cuda.memory_allocated()
    runtime = PersistentMoE(
        dist.group.WORLD, experts, tokens, topk, hidden, inner, fuse_activation=True
    )
    native = NativePersistentBackward(runtime)
    runtime._peer_backward = native
    torch.cuda.synchronize()
    built = dict(
        device_used=_max(_used() - used0),
        torch_allocated=_max(torch.cuda.memory_allocated() - alloc0),
    )
    for _ in range(3):
        for t in (x, scores, w1, w2):
            t.grad = None
        runtime(x, ids, scores, w1, w2).backward(dy)
    for t in (x, scores, w1, w2):
        t.grad = None
    torch.cuda.synchronize()
    stepped = dict(
        device_used=_max(_used() - used0),
        torch_allocated=_max(torch.cuda.memory_allocated() - alloc0),
    )
    pools = sum(p.numel() * p.element_size() for p in native.pools[:4])
    bound = native.bound
    payload = dict(
        shape=args.shape,
        routing=args.routing,
        mode=os.getenv("MONO_EP_WORKSPACE_BOUND", "auto"),
        after_construction_gib=built,
        after_three_steps_gib=stepped,
        backward_pools_gib=_max(pools),
        pool_rows_max_rank=_max(native.pools[0].shape[0]) * 2**30,
        worst_rows=native.worst_rows,
        bound=None
        if bound is None
        else dict(
            rows_max_rank=_max(bound.rows) * 2**30,
            high_water_max_rank=_max(bound.high_water) * 2**30,
            grows=bound.grows,
        ),
        forward_symmetric_gib=_max(runtime.buffer.buffer.nbytes),
        backward_symmetric_gib=_max(native.buffer.buffer.nbytes),
        forward_ring_tokens=runtime.buffer.l1_acts.shape[0],
        backward_ring_tokens=native.buffer.l1_acts.shape[0],
    )
    if rank == 0:
        print("WSMEM_JSON " + json.dumps(payload), flush=True)
    dist.barrier()
    runtime.close()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
