# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Per-layer memory kept between forward and backward by each checkpointing policy, measured.

Run under torchrun (EP8) with the MoK checkout on PYTHONPATH. One fused mono-ep layer plus a
residual-style consumer (``out = y * 1.0``, which saves nothing), so the measured delta is what the
policy keeps alive for the backward beyond the layer output:

* ``ckpt``: non-reentrant checkpoint of the layer (the literal 2F+B contract);
* ``sac``: the same with a selective policy that saves ``mono_ep::persistent_forward``'s output;
* ``eager``: no checkpointing.

Deltas are ``torch.cuda.memory_allocated`` after the forward (graph alive, ``y`` dropped) minus before,
minus ``out``. Resident per-runtime memory (workspaces, pools, the stored-z buffer at the routed
high-water mark after three storing steps) is measured with ``torch.cuda.mem_get_info`` /
``memory_allocated`` around construction. MoK: bytes of its schedule (kept from forward to backward in
its checkpointed mode) and of its forward context (kept in its stored-context mode), summed over unique
storages, and its workspace from ``mem_get_info`` around construction.

Prints ``MEMPOL_JSON <payload>`` on rank 0.
"""

import argparse
import dataclasses
import gc
import json
import os

import torch
import torch.distributed as dist
from mok_vs_mono_ep import SHAPES, _save_mono_ep_output
from torch.utils.checkpoint import checkpoint


def _used():
    free, total = torch.cuda.mem_get_info()
    return total - free


def _storage_bytes(obj, seen):
    total = 0
    if isinstance(obj, torch.Tensor):
        key = obj.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            total += obj.untyped_storage().nbytes()
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            total += _storage_bytes(item, seen)
    elif dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            total += _storage_bytes(getattr(obj, field.name), seen)
    return total


def _max(value):
    t = torch.tensor([float(value)], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--routing", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--mok", action="store_true")
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
    ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk].contiguous()
    if args.routing == "skewed":
        from mok_vs_mono_ep import _skewed_routing

        ids = _skewed_routing(tokens, experts, topk, rank)
    scores = torch.rand(tokens, topk, device="cuda")
    scores = (scores / scores.sum(-1, keepdim=True)).requires_grad_()
    w1 = (
        torch.randn(local, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(local, hidden, inner, device="cuda") / inner**0.5
    ).requires_grad_()
    dy = torch.randn_like(x) * hidden**-0.5
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
    resident = dict(
        device_used=_max(_used() - used0),
        torch_allocated=_max(torch.cuda.memory_allocated() - alloc0),
    )

    def layer(policy):
        if policy == "eager":
            return runtime(x, ids, scores, w1, w2)
        return checkpoint(
            runtime,
            x,
            ids,
            scores,
            w1,
            w2,
            use_reentrant=False,
            preserve_rng_state=False,
            **({"context_fn": _save_mono_ep_output} if policy == "sac" else {}),
        )

    def step(policy):
        out = layer(policy) * 1.0
        out.backward(dy)
        for t in (x, scores, w1, w2):
            t.grad = None

    for policy in ("eager", "ckpt", "sac"):
        step(policy)
    torch.cuda.synchronize()
    saved = {}
    for policy in ("eager", "ckpt", "sac"):
        for t in (x, scores, w1, w2):
            t.grad = None
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        y = layer(policy)
        out = y * 1.0
        del y
        torch.cuda.synchronize()
        saved[policy] = _max(torch.cuda.memory_allocated() - before - out.nbytes)
        out.backward(dy)
        del out
    layer_output_bytes = tokens * hidden * 2

    # Stored z at the routed high-water mark: storing forwards plus backwards, then the polled size
    os.environ["MONO_EP_STORED_CONTEXT"] = "always"
    for t in (x, scores, w1, w2):
        t.grad = None
    for _ in range(4):
        step("eager")
    torch.cuda.synchronize()
    stored = runtime._stored
    stored._poll_high_water(enforce_memory=False)
    from mono_ep.stored_context import meta_bytes

    z_rows_worst = world * runtime.buffer.num_max_tokens_per_rank * topk + local * 128
    stored_z = dict(
        high_water_rows=_max(stored.high_water),
        capacity_rows=_max(stored.capacity),
        buffer_bytes=_max(stored.nbytes()),
        high_water_bytes=_max(stored.high_water * stored.row_bytes + stored.meta),
        worst_case_bytes=meta_bytes(world, runtime.buffer.num_max_tokens_per_rank, topk)
        + z_rows_worst * stored.row_bytes,
        hits=native.context_hits,
    )
    os.environ.pop("MONO_EP_STORED_CONTEXT")

    mok = None
    if args.mok:
        from benchmarks import bench_mok
        from mok import functional
        from mok_vs_mono_ep import MACROBATCH, MOK_KNOBS

        from tests.utils import generate_inputs

        inputs = list(
            generate_inputs(
                rank,
                torch.device("cuda", local_rank),
                experts,
                local,
                topk,
                tokens,
                hidden,
                inner,
            )
        )
        inputs[1] = ids.to(inputs[1].dtype)
        torch.cuda.synchronize()
        used1 = _used()
        bench = bench_mok.MoKBenchmark(tuple(inputs))
        torch.cuda.synchronize()
        mok = dict(workspace_and_weights_device_used=_max(_used() - used1))
        for precision in ("BF16", "MXFP8"):
            f, b, mb = MOK_KNOBS[args.shape][precision]
            cfg = functional.MoKConfig(
                fwd_num_comm_sms=f,
                bwd_num_comm_sms=b,
                minibatch_size=mb,
                macrobatch_size=MACROBATCH,
            )
            setattr(bench_mok, f"{precision}_CONFIG", cfg)
            output, (schedule, context) = getattr(
                bench, f"run_{precision.lower()}_fwd"
            )()
            torch.cuda.synchronize()
            mok[precision] = dict(
                schedule_bytes=_max(_storage_bytes(schedule, set())),
                forward_context_bytes=_max(_storage_bytes(context, set())),
                forward_context_fields={
                    field.name: _max(
                        _storage_bytes(getattr(context, field.name), set())
                    )
                    for field in dataclasses.fields(context)
                },
            )
            del output, schedule, context
        functional.clear_workspace_cache()

    payload = dict(
        shape=args.shape,
        routing=args.routing,
        tokens=tokens,
        hidden=hidden,
        intermediate=inner,
        experts=experts,
        world=world,
        layer_output_bytes=layer_output_bytes,
        saved_beyond_output_bytes=saved,
        mono_ep_resident=resident,
        stored_z=stored_z,
        mok=mok,
    )
    if rank == 0:
        print("MEMPOL_JSON " + json.dumps(payload), flush=True)
    dist.barrier()
    runtime.close()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
