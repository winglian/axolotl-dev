# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Check the SM90 FP8 quantized-weight cache under FSDP2 with real optimizer steps.

Each rank runs the persistent FP8 layer at expert-parallel world 1 (its own process group, all
16 experts local) while ``fully_shard`` shards the whole layer (a dense projection plus the
FP32 expert weights) over the ranks, so FSDP2 all-gathers the expert weights into persistent
unsharded storage under a preserved version counter every step. The checkpointed layer runs
through the compiled (fullgraph) runtime. Before the first and after every AdamW step the
cached run's output, input gradient and score gradient must be bitwise equal to an uncached run
of the same kernels on the live (``full_tensor``) weights, and the FSDP-reduced weight
gradients must match the uncached run's averaged gradients within the dW nondeterminism
envelope.

Cases: ``global_hook`` (default contract: the global torch.optim post-step hook invalidates),
``optimizer_hook`` (global hook off, ``register_optimizer_invalidation``), ``epoch`` (global
hook off, ``set_weight_cache_epoch(step)``), ``bf16_params`` (global hook, FSDP2 BF16
param_dtype), ``uncached`` (cache off), ``no_invalidation`` (global hook off, nothing invalidates; observational,
since the data pointer is still in the key) and the validator self-check ``frozen`` (a fixed caller
epoch and no invalidation, which must be caught serving stale weights).

``--ep N`` (default 1) builds a (dp_shard, ep) mesh instead: the layer is sharded over
dp_shard only, each EP group of N ranks runs the persistent kernels across peers with 16 local
experts per rank (16 N routed experts), expert weights differ per EP rank and agree across
dp_shard, and the reference dW is averaged over dp_shard. With ``--ep 1`` each rank's EP group has
one rank, so the persistent kernels never wait on peers; reserved SMs (``--reserved-sms``,
default 16) and NCCL_MAX_CTAS still leave room for the FSDP2 collectives.
"""

import argparse
import json
import os
import traceback

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint

HIDDEN, INNER, EXPERTS, TOPK = 2048, 768, 16, 8
CASES = (
    "uncached",
    "global_hook",
    "optimizer_hook",
    "epoch",
    "bf16_params",
    "no_invalidation",
    "frozen",
)


class Layer(nn.Module):
    def __init__(self, runner, ep_rank=0):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        torch.manual_seed(4321 + ep_rank)
        self.w1 = nn.Parameter(torch.randn(EXPERTS, 2 * INNER, HIDDEN) / HIDDEN**0.5)
        self.w2 = nn.Parameter(torch.randn(EXPERTS, HIDDEN, INNER) / INNER**0.5)
        self.runner = runner
        self.seen = []

    def experts(self, h, ids, scores):
        storage = self.w1.untyped_storage()
        self.seen.append((id(storage), storage.data_ptr(), self.w1._version))
        return self.runner(h, ids, scores.float(), self.w1, self.w2)

    def forward(self, x, ids, scores):
        h = self.proj(x.to(self.proj.weight.dtype)).bfloat16()
        h.retain_grad()
        y = checkpoint(self.experts, h, ids, scores, use_reentrant=False)
        return h, y


def _relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def run_case(case, args, runtime, compiled, device, rank, mesh):
    from mono_ep import weight_cache

    os.environ["MONO_EP_SM90_FP8_WEIGHT_CACHE"] = (
        "0" if case == "uncached" else args.cache_mode
    )
    weight_cache.invalidate_weight_caches()
    weight_cache.set_optimizer_hook_enabled(case in ("global_hook", "bf16_params"))
    weight_cache._EPOCH = None
    if case == "frozen":
        # Self-check of this validator: a caller epoch that never advances drops the data
        # pointer from the key and nothing invalidates, so the cache must serve stale weights.
        weight_cache.set_weight_cache_epoch("frozen")
    stats0 = weight_cache.cache_stats()

    torch.manual_seed(1234)
    ep_rank = mesh["ep"].get_local_rank()
    dp_group, dp_size = mesh["dp_shard"].get_group(), mesh["dp_shard"].size()
    layer = Layer(compiled, ep_rank).to(device)
    mp = (
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        if case == "bf16_params"
        else MixedPrecisionPolicy()
    )
    fully_shard(layer, mesh=mesh["dp_shard"], mp_policy=mp)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=args.lr, weight_decay=0.0)
    handle = None
    if case == "optimizer_hook":
        handle = weight_cache.register_optimizer_invalidation(optimizer)

    generator = torch.Generator(device=device).manual_seed(77 + rank)
    records = []
    previous_w1 = None
    for step in range(args.steps + 1):
        if case == "epoch":
            weight_cache.set_weight_cache_epoch(step)
        x = torch.randn(
            args.tokens,
            HIDDEN,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        ids = (
            torch.rand(
                args.tokens, EXPERTS * args.ep, device=device, generator=generator
            )
            .argsort(-1)[:, :TOPK]
            .contiguous()
        )
        scores = torch.rand(args.tokens, TOPK, device=device, generator=generator)
        scores = (scores / scores.sum(-1, keepdim=True)).requires_grad_()
        dy = torch.randn(
            args.tokens,
            HIDDEN,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        h, y = layer(x, ids, scores)
        (y.float() * dy.float()).sum().backward()

        # Uncached reference on the live weights (same kernels, eager runtime call).
        live_w1 = layer.w1.full_tensor().detach()
        live_w2 = layer.w2.full_tensor().detach()
        if case == "bf16_params":
            live_w1, live_w2 = live_w1.bfloat16(), live_w2.bfloat16()
        mode = os.environ["MONO_EP_SM90_FP8_WEIGHT_CACHE"]
        os.environ["MONO_EP_SM90_FP8_WEIGHT_CACHE"] = "0"
        bf16 = case == "bf16_params"
        h_ref = h.detach().clone().requires_grad_()
        # FSDP2 casts floating forward inputs to param_dtype, so the layer saw BF16 scores
        s_ref = (
            scores.detach().bfloat16() if bf16 else scores.detach().clone()
        ).requires_grad_()
        w1_ref = live_w1.clone().requires_grad_()
        w2_ref = live_w2.clone().requires_grad_()
        y_ref = runtime(h_ref, ids, s_ref.float(), w1_ref, w2_ref)
        y_ref.backward(dy)
        os.environ["MONO_EP_SM90_FP8_WEIGHT_CACHE"] = mode
        grads_ref = []
        for g in (w1_ref.grad, w2_ref.grad):
            g = g.float()
            dist.all_reduce(g, group=dp_group)
            grads_ref.append(g / dp_size)
        grads = [layer.w1.grad.full_tensor(), layer.w2.grad.full_tensor()]
        record = dict(
            step=step,
            y_bitwise=bool(torch.equal(y.detach(), y_ref.detach())),
            dh_bitwise=bool(torch.equal(h.grad, h_ref.grad)),
            dscore_bitwise=bool(torch.equal(scores.grad, s_ref.grad.to(scores.dtype))),
            y_rel=_relative(y.detach(), y_ref.detach()),
            dh_rel=_relative(h.grad, h_ref.grad),
            dscore_rel=_relative(scores.grad, s_ref.grad.to(scores.dtype)),
            dw1_rel=_relative(grads[0], grads_ref[0]),
            dw2_rel=_relative(grads[1], grads_ref[1]),
            w1_rel_change_since_previous=None
            if previous_w1 is None
            else _relative(live_w1, previous_w1),
            unsharded_w1_storage_ptr_version=[list(v[1:]) for v in layer.seen],
            unsharded_w1_storage_objects=len({v[0] for v in layer.seen}),
        )
        layer.seen.clear()
        previous_w1 = live_w1.clone()
        records.append(record)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if handle is not None:
        handle.remove()
    stats1 = weight_cache.cache_stats()
    weight_cache.set_optimizer_hook_enabled(True)
    weight_cache._EPOCH = None
    weight_cache.invalidate_weight_caches()
    exact = all(
        r["y_bitwise"] and r["dh_bitwise"] and r["dscore_bitwise"] for r in records
    )
    dw_ok = all(
        r["dw1_rel"] < args.dw_tolerance and r["dw2_rel"] < args.dw_tolerance
        for r in records
    )
    expected_stale = case in ("no_invalidation", "frozen")
    return dict(
        case=case,
        cache_mode="0" if case == "uncached" else args.cache_mode,
        records=records,
        exact=exact,
        dw_within_tolerance=dw_ok,
        stale_reproduced=expected_stale and not exact,
        # no_invalidation is observational: the pointer stays in the key, so whether it goes
        # stale depends on FSDP2 re-gathering into the same address.
        passed=True
        if case == "no_invalidation"
        else (not exact)
        if expected_stale
        else (exact and dw_ok),
        cache_stats_delta={k: stats1[k] - stats0[k] for k in stats1},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dw-tolerance", type=float, default=2e-3)
    parser.add_argument("--cache-mode", default="1")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--out", default="")
    parser.add_argument("--reserved-sms", type=int, default=16)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if not any(name in torch.cuda.get_device_name() for name in ("H100", "H200")):
        raise RuntimeError(f"expected H100 or H200, got {torch.cuda.get_device_name()}")
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    os.environ.setdefault("MONO_EP_SM90_FP8_TRANSPORT", "1")
    os.environ.setdefault("MONO_EP_SM90_FP8_COMBINE", "1")
    import deep_gemm
    from mono_ep.runtime import PersistentMoE

    physical = torch.cuda.get_device_properties(device).multi_processor_count
    deep_gemm.set_num_sms(physical - args.reserved_sms)
    if world % args.ep:
        raise ValueError("world must be a multiple of --ep")
    mesh = init_device_mesh(
        "cuda", (world // args.ep, args.ep), mesh_dim_names=("dp_shard", "ep")
    )
    runtime = PersistentMoE(
        mesh["ep"].get_group(),
        EXPERTS * args.ep,
        args.tokens,
        TOPK,
        HIDDEN,
        INNER,
        compute_dtype="blockwise",
        backward_backend="native",
        reserved_sms=args.reserved_sms,
    )
    compiled = torch.compile(runtime, fullgraph=True)
    results = []
    for case in args.cases.split(","):
        try:
            results.append(run_case(case, args, runtime, compiled, device, rank, mesh))
        except Exception:  # noqa: BLE001
            results.append(dict(case=case, passed=False, error=traceback.format_exc()))
    gathered = [None] * world
    dist.all_gather_object(gathered, results)
    if rank == 0:
        summary = dict(
            device=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            world_size=world,
            sharding=f"fully_shard(whole layer) over dp_shard={world // args.ep}, "
            f"EP world {args.ep} ({EXPERTS} local experts per rank)",
            mesh_dp_shard_x_ep=[world // args.ep, args.ep],
            env_defaults_overridden={
                k: v for k, v in os.environ.items() if k.startswith("MONO_EP_")
            },
            tokens_per_rank=args.tokens,
            steps=args.steps,
            lr=args.lr,
            reserved_sms=args.reserved_sms,
            nccl_max_ctas=os.environ.get("NCCL_MAX_CTAS"),
            cache_mode=args.cache_mode,
            per_rank=gathered,
            passed={
                case: all(
                    next(r for r in ranks if r["case"] == case)["passed"]
                    for ranks in gathered
                )
                for case in args.cases.split(",")
            },
        )
        if args.out:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=1)
        print("FSDP2_CACHE_JSON " + json.dumps(summary["passed"]), flush=True)
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
