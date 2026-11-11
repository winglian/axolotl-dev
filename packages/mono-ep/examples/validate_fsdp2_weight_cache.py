# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Check the BF16 weight-layout cache against live FSDP2-sharded expert weights.

Mirrors Axolotl's FSDP2 x EP layout: each layer is a decoder-like module with a
dense projection and an experts submodule whose (w1, w2) hold this rank's
expert-parallel slice as plain parameters. The experts submodule is wrapped with
``fully_shard`` on the ``dp_shard`` mesh (gradient divide factor ep * dp_shard,
SUM reduction), the whole layer on the flattened (dp_shard, ep) mesh, and the
root last; MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32). FSDP2
all-gathers into persistent unsharded storage with the version counter
preserved, so a version-keyed cache can serve layouts of the previous step.

For every case the stack runs ``--steps`` AdamW steps (nonzero lr) and before
the first and after each step checks the fused MoE output, its input gradient,
the score gradient and the FSDP-reduced expert gradients against an FP32 torch
reference computed from the live (all-gathered) weights at that step.

``--backend fused`` runs the persistent SM100 runtime with the native backward;
``--backend emulated`` (CPU/gloo dry run) replaces the kernel with a torch
emulation that reads the same cached layouts.
"""

import argparse
import contextlib
import json
import os
import traceback

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint

HIDDEN, INNER, EXPERTS, TOPK = 2048, 768, 16, 4


def relative(actual, expected):
    return (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-12)
    ).item()


def _deinterleave(tw1):
    g, n, k = tw1.shape
    view = tw1.view(g, n // 16, 2, 8, k)
    return torch.cat(
        (view[:, :, 0].reshape(g, n // 2, k), view[:, :, 1].reshape(g, n // 2, k)), 1
    )


class _EmulatedMoE(torch.autograd.Function):
    """Torch stand-in for the fused kernel that reads the cached layouts."""

    @staticmethod
    def forward(ctx, x, ids, scores, w1, w2, group):
        from mono_ep.weight_cache import weight_layouts

        tw1, tw2 = weight_layouts(w1, w2, "w1_interleaved", "w2")
        gx, gids, gscores = (_gather(t, group) for t in (x, ids, scores))
        y = _expert_sum(gx, gids, gscores, _deinterleave(tw1), tw2, group)
        ctx.save_for_backward(x, ids, scores, w1, w2)
        ctx.group = group
        return _reduce_rows(y, group, x.shape[0]).to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        from mono_ep.weight_cache import weight_layouts

        x, ids, scores, w1, w2 = ctx.saved_tensors
        group = ctx.group
        tw1, w2t, w1t = weight_layouts(w1, w2, "w1_interleaved", "w2t", "w1t")
        gx, gids, gscores, gdy = (_gather(t, group) for t in (x, ids, scores, dy))
        with torch.enable_grad():
            leaves = [
                t.detach().float().requires_grad_()
                for t in (gx, gscores, w1t.transpose(1, 2), w2t.transpose(1, 2))
            ]
            del tw1
            y = _expert_sum(leaves[0], gids, leaves[1], leaves[2], leaves[3], group)
            grads = torch.autograd.grad(y, leaves, gdy.float())
        dx = _reduce_rows(grads[0], group, x.shape[0]).to(x.dtype)
        ds = _reduce_rows(grads[1], group, x.shape[0])
        return dx, None, ds, grads[2].to(w1.dtype), grads[3].to(w2.dtype), None


def _check_runtime_inputs(x, ids, scores, w1, w2):
    """The dtype/layout checks PersistentMoE.__call__ applies."""
    if (
        x.dtype != torch.bfloat16
        or any(t.dtype not in (torch.bfloat16, torch.float32) for t in (w1, w2))
        or scores.dtype != torch.float32
        or ids.dtype != torch.int64
    ):
        raise ValueError(
            f"runtime dtypes: x {x.dtype} w {w1.dtype}/{w2.dtype} scores {scores.dtype} ids {ids.dtype}"
        )
    if any(not t.is_contiguous() for t in (x, ids, scores, w1, w2)):
        raise ValueError("runtime requires contiguous inputs")


def _gather(t, group):
    t = t.detach()
    if dist.get_world_size(group) == 1:
        return t
    out = t.new_empty((dist.get_world_size(group) * t.shape[0], *t.shape[1:]))
    dist.all_gather_into_tensor(out, t.contiguous(), group=group)
    return out


def _reduce_rows(t, group, rows):
    if dist.get_world_size(group) == 1:
        return t
    t = t.contiguous()
    dist.all_reduce(t, group=group)
    rank = dist.get_rank(group)
    return t[rank * rows : (rank + 1) * rows]


def _expert_sum(x, ids, scores, w1, w2, group):
    """sum_k score_k * SwiGLU expert(x) over this rank's local experts (FP32)."""
    local = w1.shape[0]
    first = dist.get_rank(group) * local if group is not None else 0
    out = x.new_zeros((x.shape[0], w2.shape[1]), dtype=torch.float32)
    for e in range(local):
        rows, slots = (ids == first + e).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        z = x[rows].float() @ w1[e].float().T
        gate, up = z.chunk(2, -1)
        h = nn.functional.silu(gate) * up * scores[rows, slots, None].float()
        out = out.index_add(0, rows, h @ w2[e].float().T)
    return out


class Experts(nn.Module):
    def __init__(self, local, runner):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(local, 2 * INNER, HIDDEN) / HIDDEN**0.5)
        self.w2 = nn.Parameter(torch.randn(local, HIDDEN, INNER) / INNER**0.5)
        self.runner = runner

    def forward(self, h, ids, scores):
        # FSDP2 mixed precision casts floating forward inputs to BF16; the runtime takes FP32 routing weights
        return self.runner(h, ids, scores.float(), self.w1, self.w2)


class Layer(nn.Module):
    def __init__(self, local, runner):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.experts = Experts(local, runner)
        self.captured = {}

    def forward(self, x, ids, scores):
        h = self.proj(x)
        h.retain_grad()
        y = checkpoint(self.experts, h, ids, scores, use_reentrant=False)
        y.retain_grad()
        self.captured = dict(h=h, y=y)
        return y


class Stack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, batches, scores):
        loss = 0
        for layer, (x, ids, _s, dy), s in zip(
            self.layers, batches, scores, strict=True
        ):
            loss = loss + (layer(x, ids, s).float() * dy.float()).sum()
        return loss


def _reference(h, ids, scores, w1_full, w2_full, dy):
    """FP32 reference over all experts for this rank's tokens."""
    leaves = [
        t.detach().float().requires_grad_() for t in (h, scores, w1_full, w2_full)
    ]
    y = _expert_sum(leaves[0], ids, leaves[1], leaves[2], leaves[3], None)
    grads = torch.autograd.grad(y, leaves, dy.float())
    return y.detach(), grads


def run_case(args, case, mesh, flat, runner_factory, device):
    from mono_ep import weight_cache

    env = case.get("env", {})
    saved_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    hook = getattr(weight_cache, "set_optimizer_hook_enabled", None)
    if hook is not None:
        hook(case.get("global_hook", False))
    weight_cache.invalidate_weight_caches()
    original_layouts = weight_cache.weight_layouts
    if case.get("freeze"):
        frozen = {}

        def frozen_layouts(w1, w2, *names):
            key = (w1.untyped_storage()._cdata, w2.untyped_storage()._cdata)
            store = frozen.setdefault(key, {})
            missing = [n for n in names if n not in store]
            for name, value in zip(
                missing, original_layouts(w1, w2, *missing), strict=True
            ):
                store[name] = value.clone()
            return tuple(store[n] for n in names)

        weight_cache.weight_layouts = frozen_layouts
    original_mx_forms = getattr(weight_cache, "mxfp8_backward_weight_forms", None)
    if case.get("freeze_mx"):
        frozen_mx = {}

        def frozen_mx_forms(w1, w2):
            key = (w1.untyped_storage()._cdata, w2.untyped_storage()._cdata)
            if key not in frozen_mx:
                frozen_mx[key] = tuple(
                    tuple(t.clone() for t in form) for form in original_mx_forms(w1, w2)
                )
            return frozen_mx[key]

        weight_cache.mxfp8_backward_weight_forms = frozen_mx_forms
    ep, dp = mesh["ep"].size(), mesh["dp_shard"].size()
    local = EXPERTS // ep
    ep_rank, dp_rank = mesh["ep"].get_local_rank(), mesh["dp_shard"].get_local_rank()
    torch.manual_seed(1234 + ep_rank)
    runner = runner_factory()
    layers = nn.ModuleList()
    for _ in range(args.layers):
        layer = Layer(local, runner)
        torch.manual_seed(99 + len(layers))
        layer.proj.weight.data.normal_(0, 1 / HIDDEN**0.5)
        layers.append(layer)
    layers.to(device)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in layers:
        fully_shard(layer.experts, mesh=mesh["dp_shard"], mp_policy=mp)
        if dp > 1:
            layer.experts.set_force_sum_reduction_for_comms(True)
        layer.experts.set_gradient_divide_factor(float(ep * dp))
        fully_shard(layer, mesh=flat, mp_policy=mp)
    model = Stack(layers)
    fully_shard(model, mesh=flat, mp_policy=mp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    if case.get("register_on_optimizer"):
        weight_cache.register_optimizer_invalidation(optimizer)
    generator = torch.Generator(device="cpu").manual_seed(4321 + dist.get_rank())
    inputs = []
    for _ in range(args.layers):
        x = torch.randn(args.tokens, HIDDEN, generator=generator).bfloat16()
        ids = torch.rand(args.tokens, EXPERTS, generator=generator).argsort(-1)[
            :, :TOPK
        ]
        scores = torch.rand(args.tokens, TOPK, generator=generator).softmax(-1)
        dy = torch.randn(args.tokens, HIDDEN, generator=generator).bfloat16()
        inputs.append([t.to(device).contiguous() for t in (x, ids, scores, dy)])
    results = []
    runtime = getattr(layers[0].experts, "runner", None)
    bound = getattr(getattr(runtime, "_peer_backward", None), "bound", None)
    bound_keys = []
    if bound is not None:
        from mono_ep.workspace_bound import layer_key

        original_record = bound.record

        def tracing_record(w1, counts):
            bound_keys[-1].append(layer_key(w1))
            original_record(w1, counts)

        bound.record = tracing_record
    for step in range(args.steps + 1):
        bound_keys.append([])
        bound_before = (
            None if bound is None else (bound.found, bound.unrecorded, bound.dropped)
        )
        if case.get("caller_epoch"):
            weight_cache.set_weight_cache_epoch(step)
        stats_before = dict(getattr(weight_cache, "cache_stats", lambda: {})())
        optimizer.zero_grad(set_to_none=True)
        leaves = [b[2].clone().requires_grad_() for b in inputs]
        model(inputs, leaves).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        stats_after = dict(getattr(weight_cache, "cache_stats", lambda: {})())
        errors = []
        for index, (layer, (_x, ids, _scores, dy)) in enumerate(
            zip(layers, inputs, strict=True)
        ):
            with torch.no_grad():
                w1_full = layer.experts.w1.full_tensor()
                w2_full = layer.experts.w2.full_tensor()
                dw1_got = layer.experts.w1.grad.full_tensor()
                dw2_got = layer.experts.w2.grad.full_tensor()
            w1_all = _gather(w1_full.contiguous(), mesh["ep"].get_group())
            w2_all = _gather(w2_full.contiguous(), mesh["ep"].get_group())
            h, y = layer.captured["h"], layer.captured["y"]
            y_ref, (dh_ref, ds_ref, dw1_ref, dw2_ref) = _reference(
                h.bfloat16(),
                ids,
                leaves[index],
                w1_all.bfloat16(),
                w2_all.bfloat16(),
                dy,
            )
            for ref in (dw1_ref, dw2_ref):
                dist.all_reduce(ref)
                ref.div_(ep * dp)
            block = slice(ep_rank * local, (ep_rank + 1) * local)
            errors.append(
                dict(
                    layer=index,
                    y=relative(y, y_ref),
                    dx=relative(h.grad, dh_ref),
                    dscore=relative(leaves[index].grad, ds_ref),
                    dw1=relative(dw1_got, dw1_ref[block]),
                    dw2=relative(dw2_got, dw2_ref[block]),
                )
            )
        worst = {
            k: max(e[k] for e in errors) for k in ("y", "dx", "dscore", "dw1", "dw2")
        }
        worst_tensor = torch.tensor(list(worst.values()), device=device)
        dist.all_reduce(worst_tensor, op=dist.ReduceOp.MAX)
        worst = dict(zip(worst, worst_tensor.tolist(), strict=True))
        results.append(
            dict(
                step=step,
                worst=worst,
                passed=max(worst.values()) < args.tolerance,
                cache_delta={
                    k: stats_after.get(k, 0) - stats_before.get(k, 0)
                    for k in stats_after
                },
                per_layer_rank0=errors if dist.get_rank() == 0 else None,
            )
        )
        if bound is not None:
            # w1's unsharded storage must stay the record key across reshard / regather and steps
            results[-1]["workspace_bound"] = dict(
                records=len(bound_keys[-1]),
                keys_match_step0=bound_keys[-1] == bound_keys[0],
                found=bound.found - bound_before[0],
                unrecorded=bound.unrecorded - bound_before[1],
                dropped=bound.dropped - bound_before[2],
                pending=sum(map(len, bound.records.values())),
            )
        if step < args.steps:
            optimizer.step()
            if case.get("explicit_invalidate"):
                weight_cache.invalidate_weight_caches()
    if bound is not None:
        del bound.record
    weight_cache.weight_layouts = original_layouts
    if original_mx_forms is not None:
        weight_cache.mxfp8_backward_weight_forms = original_mx_forms
    if hasattr(weight_cache, "set_weight_cache_epoch"):
        weight_cache.set_weight_cache_epoch(None)
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    del layers, model, optimizer
    return dict(
        case=case["name"],
        description=case["description"],
        expected=case.get("expected", "pass"),
        passed=all(r["passed"] for r in results),
        steps=results,
        mesh=dict(dp_shard=dp, ep=ep),
        dp_rank=dp_rank,
    )


def cases(weight_cache):
    has_fix = hasattr(weight_cache, "set_optimizer_hook_enabled")
    detector = dict(
        name="detector_frozen_layouts",
        description="harness self-check: layouts frozen at step 0 (must be detected as stale)",
        env={"MONO_EP_WEIGHT_CACHE": "0"},
        freeze=True,
        expected="stale",
    )
    if not has_fix:
        return [
            detector,
            dict(
                name="round1_cache",
                description="round-1 weight cache as committed (version-keyed, object identity)",
                env={"MONO_EP_WEIGHT_CACHE": "1"},
                expected="unknown",
            ),
            dict(
                name="round1_cache_lru8",
                description="round-1 weight cache as committed with MONO_EP_WEIGHT_CACHE_ENTRIES=8",
                env={
                    "MONO_EP_WEIGHT_CACHE": "1",
                    "MONO_EP_WEIGHT_CACHE_ENTRIES": "8",
                },
                expected="unknown",
            ),
            dict(
                name="round1_cache_explicit_invalidate",
                description="round-1 cache plus invalidate_weight_caches() after each optimizer step",
                env={"MONO_EP_WEIGHT_CACHE": "1"},
                explicit_invalidate=True,
            ),
            dict(
                name="uncached",
                description="MONO_EP_WEIGHT_CACHE=0 control",
                env={"MONO_EP_WEIGHT_CACHE": "0"},
            ),
        ]
    hook_off = dict(env={"MONO_EP_WEIGHT_CACHE": "1"})
    return [
        detector,
        dict(
            name="detector_frozen_mxfp8_forms",
            description="harness self-check: MXFP8 backward weight forms frozen at step 0 (must be stale)",
            env={"MONO_EP_WEIGHT_CACHE": "1"},
            global_hook=True,
            freeze_mx=True,
            expected="stale",
        ),
        dict(
            name="round1_key_no_invalidate",
            description="round-1 key (MONO_EP_WEIGHT_CACHE_KEY=object), no invalidation, hook off",
            env={
                "MONO_EP_WEIGHT_CACHE": "1",
                "MONO_EP_WEIGHT_CACHE_KEY": "object",
            },
            expected="unknown",
        ),
        dict(
            name="storage_key_no_invalidate",
            description="storage key with data pointer, no invalidation, hook off (contract violated)",
            env={
                "MONO_EP_WEIGHT_CACHE": "1",
                "MONO_EP_WEIGHT_CACHE_KEY": "storage",
            },
            expected="unknown",
        ),
        dict(
            name="round1_key_no_invalidate_lru8",
            description="round-1 key, no invalidation, hook off, MONO_EP_WEIGHT_CACHE_ENTRIES=8 (>= layers)",
            env={
                "MONO_EP_WEIGHT_CACHE": "1",
                "MONO_EP_WEIGHT_CACHE_KEY": "object",
                "MONO_EP_WEIGHT_CACHE_ENTRIES": "8",
            },
            expected="unknown",
        ),
        dict(
            name="storage_key_no_invalidate_lru8",
            description="storage key with data pointer, no invalidation, hook off, 8 cache entries",
            env={
                "MONO_EP_WEIGHT_CACHE": "1",
                "MONO_EP_WEIGHT_CACHE_KEY": "storage",
                "MONO_EP_WEIGHT_CACHE_ENTRIES": "8",
            },
            expected="unknown",
        ),
        dict(
            name="caller_epoch_lru8",
            description="auto key, set_weight_cache_epoch(step), hook off, 8 cache entries",
            env={
                "MONO_EP_WEIGHT_CACHE": "1",
                "MONO_EP_WEIGHT_CACHE_ENTRIES": "8",
            },
            caller_epoch=True,
        ),
        dict(
            name="storage_epoch_key_no_invalidate",
            description="storage key without data pointer, no invalidation, hook off (contract violated)",
            env={
                "MONO_EP_WEIGHT_CACHE": "1",
                "MONO_EP_WEIGHT_CACHE_KEY": "storage_epoch",
            },
            expected="stale",
        ),
        dict(
            name="default_global_hook",
            description="defaults: auto key, global torch.optim post-step hook only",
            env={"MONO_EP_WEIGHT_CACHE": "1"},
            global_hook=True,
        ),
        dict(
            name="explicit_invalidate",
            description="auto key, invalidate_weight_caches() after each optimizer step, hook off",
            explicit_invalidate=True,
            **hook_off,
        ),
        dict(
            name="caller_epoch",
            description="auto key, set_weight_cache_epoch(step) before each step, hook off",
            caller_epoch=True,
            **hook_off,
        ),
        dict(
            name="registered_optimizer",
            description="auto key, register_optimizer_invalidation(optimizer), hook off",
            register_on_optimizer=True,
            **hook_off,
        ),
        dict(
            name="uncached",
            description="MONO_EP_WEIGHT_CACHE=0 control",
            env={"MONO_EP_WEIGHT_CACHE": "0"},
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("fused", "emulated"), default="fused")
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--only", default="")
    parser.add_argument(
        "--precision",
        choices=("bf16", "mxfp8"),
        default="bf16",
        help="MONO_EP_SM100_BACKWARD_PRECISION for every case",
    )
    parser.add_argument(
        "--reserved-sms",
        type=int,
        default=0,
        help="SMs left free for FSDP2's NCCL kernels (the persistent kernels spin on peers)",
    )
    args = parser.parse_args()
    fused = args.backend == "fused"
    os.environ["MONO_EP_SM100_BACKWARD_PRECISION"] = args.precision
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if fused:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group("nccl", device_id=device)
        name = torch.cuda.get_device_name()
        assert "B200" in name, name
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo")
        name = "cpu"
    world = dist.get_world_size()
    assert world % args.ep == 0
    mesh = init_device_mesh(
        device.type, (world // args.ep, args.ep), mesh_dim_names=("dp_shard", "ep")
    )
    ep_group = mesh["ep"].get_group()
    flat = mesh._flatten("dp_ep")
    if fused:
        import deep_gemm
        from mono_ep.runtime import PersistentMoE

        physical = torch.cuda.get_device_properties(device).multi_processor_count
        deep_gemm.set_num_sms(physical - args.reserved_sms)
        runtime = PersistentMoE(
            ep_group,
            EXPERTS,
            args.tokens,
            TOPK,
            HIDDEN,
            INNER,
            compute_dtype="bf16",
            fuse_activation=True,
            backward_backend="native",
            reserved_sms=args.reserved_sms,
        )

        def runner_factory():
            return runtime

    else:

        def runner_factory():
            def call(x, ids, scores, w1, w2):
                _check_runtime_inputs(x, ids, scores, w1, w2)
                return _EmulatedMoE.apply(x, ids, scores, w1, w2, ep_group)

            return call

    from mono_ep import weight_cache

    selected = [
        c
        for c in cases(weight_cache)
        if not args.only or c["name"] in args.only.split(",")
    ]
    outputs = []
    for case in selected:
        try:
            outputs.append(run_case(args, case, mesh, flat, runner_factory, device))
        except Exception as error:  # noqa: BLE001
            outputs.append(
                dict(
                    case=case["name"],
                    error=repr(error)[-2000:],
                    traceback=traceback.format_exc()[-4000:],
                )
            )
            if fused:
                raise
        if dist.get_rank() == 0:
            print("FSDP2_CASE " + json.dumps(outputs[-1]), flush=True)
    if dist.get_rank() == 0:
        print(
            "FSDP2_JSON "
            + json.dumps(
                dict(
                    device=name,
                    torch=torch.__version__,
                    world=world,
                    mesh=dict(dp_shard=world // args.ep, ep=args.ep),
                    backend=args.backend,
                    layers=args.layers,
                    steps=args.steps,
                    tokens_per_rank=args.tokens,
                    shape=dict(hidden=HIDDEN, inner=INNER, experts=EXPERTS, topk=TOPK),
                    lr=args.lr,
                    tolerance=args.tolerance,
                    reserved_sms=args.reserved_sms,
                    backward_precision=args.precision,
                    nccl_max_ctas=os.environ.get("NCCL_MAX_CTAS"),
                    inkernel_dw=os.environ.get("MONO_EP_SM100_INKERNEL_DW", "0"),
                    cases=outputs,
                )
            ),
            flush=True,
        )
    with contextlib.suppress(Exception):
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
