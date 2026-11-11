# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""mono-ep fused layer and Mixture-of-Kittens in one process, interleaved per iteration.

Run under torchrun on one 8-GPU host with the MoK checkout on PYTHONPATH (MoK's own
``benchmarks.bench_mok.MoKBenchmark`` builds its workspace and weights)::

    torchrun --standalone --nproc-per-node 8 mok_vs_mono_ep.py --shape 30b --routing balanced \
        --variants 'base:,mxbwd:MONO_EP_SM100_BACKWARD_PRECISION=mxfp8'

Both paths see the same routing (mono-ep's generator, skewed or balanced) and the same
timing: every case is one synchronized host-wall span per iteration, maximum over ranks,
median over the measured iterations, cases interleaved with the order reversed every other
iteration. CUDA-event spans are recorded too (MoK's own method, median of the per-iteration
rank maximum). mono-ep cases: forward, backward (forward untimed), checkpointed 2F+B, per
env variant, plus the NCCL-backward control. MoK cases per precision: forward, backward from
the stored context, recompute+backward, and checkpointed 2F+B (forward + recompute + backward
in one span). MoK computes a shared expert inside every number; its MXFP8 weight forms are
built once (the weight-quantization cost per step is timed separately as ``mok_*/wquant``).

Prints ``MOKVS_JSON <payload>`` on rank 0.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

SHAPES = {
    "30b": dict(tokens=16384, hidden=2048, intermediate=768, experts=128),
    "235b": dict(tokens=16384, hidden=4096, intermediate=1536, experts=128),
    "beam": dict(tokens=16384, hidden=5120, intermediate=2560, experts=256),
}
# fwd comm SMs, bwd comm SMs, minibatch (the 2026-11-10 8xB200 sweep picks)
MOK_KNOBS = {
    "30b": dict(BF16=(52, 52, 8192), MXFP8=(52, 52, 8192)),
    "235b": dict(BF16=(36, 36, 4096), MXFP8=(36, 36, 4096)),
    "beam": dict(BF16=(24, 24, 16384), MXFP8=(36, 24, 4096)),
}
MACROBATCH = 163840


def _parse_variants(spec):
    variants = []
    for item in filter(None, spec.split(",")):
        name, _, env = item.partition(":")
        variants.append((name, dict(e.split("=", 1) for e in env.split("+") if e)))
    return variants or [("base", {})]


def _skewed_routing(tokens, experts, topk, rank):
    generator = torch.Generator(device="cuda").manual_seed(812 + rank)
    logits = -torch.arange(experts, device="cuda", dtype=torch.float32) / (
        1.5 * experts
    )
    uniform = torch.rand(tokens, experts, device="cuda", generator=generator)
    gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-20)))
    return (logits + gumbel).topk(topk, dim=-1).indices.contiguous()


def _rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--routing", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--variants", default="")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument(
        "--mok", default="BF16,MXFP8", help="MoK precisions, '' for none"
    )
    parser.add_argument("--no-nccl", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--reference", action="store_true", help="FP32 reference parity"
    )
    parser.add_argument("--tokens", type=int, default=0, help="override tokens per GPU")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    shape = dict(SHAPES[args.shape])
    if args.tokens:
        shape["tokens"] = args.tokens
    tokens, hidden, inner, experts = (
        shape[k] for k in ("tokens", "hidden", "intermediate", "experts")
    )
    topk = 8
    local_experts = experts // world
    variants = _parse_variants(args.variants)
    managed = {key for _, env in variants for key in env}
    launch_env = {key: os.environ[key] for key in managed if key in os.environ}

    def set_env(env):
        for key in managed:
            os.environ.pop(key, None)
        os.environ.update(launch_env)
        os.environ.update(env)

    # Inputs (mono-ep harness generator; MoK gets the same routing)
    torch.manual_seed(812 + rank)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk].contiguous()
    if args.routing == "skewed":
        ids = _skewed_routing(tokens, experts, topk, rank)
    scores = torch.rand(tokens, topk, device="cuda")
    scores = scores / scores.sum(-1, keepdim=True)
    w1 = (
        torch.randn(local_experts, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(local_experts, hidden, inner, device="cuda") / inner**0.5
    ).requires_grad_()
    dy = torch.randn_like(x) * hidden**-0.5
    x = x.requires_grad_()
    scores = scores.requires_grad_()

    import deep_gemm
    from mono_ep.persistent_backward import NativePersistentBackward
    from mono_ep.runtime import PersistentMoE

    physical = torch.cuda.get_device_properties(device).multi_processor_count
    deep_gemm.set_num_sms(physical)
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inner,
        compute_dtype="bf16",
        fuse_activation=True,
        retain_activations=False,
    )
    native = NativePersistentBackward(runtime)
    operation = runtime if args.no_compile else torch.compile(runtime, fullgraph=True)

    def bump():
        for t in (w1, w2):
            torch.autograd.graph.increment_version(t)

    def clear():
        for t in (x, scores, w1, w2):
            t.grad = None

    def mono_case(scope, backend):
        def prepare():
            set_env(case_env)
            runtime._peer_backward = backend
            clear()
            bump()
            return operation(x, ids, scores, w1, w2) if scope == "bwd" else None

        def run(y):
            if scope == "fwd":
                return operation(x, ids, scores, w1, w2)
            if scope == "2fb":
                y = checkpoint(
                    operation,
                    x,
                    ids,
                    scores,
                    w1,
                    w2,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            y.backward(dy)
            return y

        case_env = {}
        return prepare, run, case_env

    cases = []
    for name, env in variants:
        for scope in ("fwd", "bwd", "2fb"):
            prepare, run, case_env = mono_case(scope, native)
            case_env.update(env)
            cases.append((f"mono/{name}/{scope}", prepare, run))
    if not args.no_nccl:
        prepare, run, _ = mono_case("2fb", None)
        cases.append(("nccl/2fb", prepare, run))

    # MoK
    mok_payload = {}
    mok_precisions = [p for p in args.mok.split(",") if p]
    if mok_precisions:
        from benchmarks import bench_mok
        from mok import functional, ops

        from tests.utils import (
            BF16_TOLERANCE,
            MXFP8_TOLERANCE,
            generate_inputs,
            run_reference_bf16,
        )

        inputs = list(
            generate_inputs(
                rank, device, experts, local_experts, topk, tokens, hidden, inner
            )
        )
        generator = torch.Generator(device=device).manual_seed(4321 + rank)
        inputs[1] = ids.to(inputs[1].dtype)
        inputs[2] = torch.softmax(
            torch.randn(tokens, topk, device=device, generator=generator), -1
        )
        bench = bench_mok.MoKBenchmark(tuple(inputs))
        configs = {}
        for precision in mok_precisions:
            f, b, mb = MOK_KNOBS[args.shape][precision]
            configs[precision] = functional.MoKConfig(
                fwd_num_comm_sms=f,
                bwd_num_comm_sms=b,
                minibatch_size=mb,
                macrobatch_size=MACROBATCH,
            )
            mok_payload[precision] = dict(
                fwd_comm_sms=f, bwd_comm_sms=b, minibatch=mb, macrobatch=MACROBATCH
            )

        def use(precision):
            setattr(bench_mok, f"{precision}_CONFIG", configs[precision])

        # Correctness once per precision against MoK's BF16 reference (same routing)
        reference = run_reference_bf16(*inputs)
        tolerances = dict(BF16=BF16_TOLERANCE, MXFP8=MXFP8_TOLERANCE)
        from benchmarks.utils import check_benchmark_correctness

        for precision in mok_precisions:
            use(precision)
            try:
                check_benchmark_correctness(
                    precision,
                    getattr(bench, f"run_{precision.lower()}_fwd"),
                    getattr(bench, f"run_{precision.lower()}_bwd"),
                    reference,
                    tolerances[precision],
                    rank,
                )
                ok = 1
            except (AssertionError, ValueError, RuntimeError) as exc:
                ok = 0
                if rank == 0:
                    print(f"MoK {precision} correctness: {exc}", flush=True)
            flag = torch.tensor([ok], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            mok_payload[precision]["correct"] = bool(flag.item())
        del reference

        def mok_case(precision, scope):
            run_fwd = getattr(bench, f"run_{precision.lower()}_fwd")
            run_bwd = getattr(bench, f"run_{precision.lower()}_bwd")
            routed = (
                (bench.w_gate_mxfp8[:2], bench.w_up_mxfp8[:2])
                if precision == "MXFP8"
                else (bench.w_routed_gate, bench.w_routed_up)
            )
            cfg = configs[precision]

            def recompute(schedule):
                return functional.recompute_forward_context(
                    cfg,
                    bench.workspace,
                    schedule,
                    bench.x,
                    bench.w_shared_gate,
                    bench.w_shared_up,
                    *routed,
                )

            def prepare():
                set_env({})
                use(precision)
                if scope in ("bwd", "rbwd"):
                    return run_fwd()[1]
                return None

            def run(context):
                if scope == "fwd":
                    return run_fwd()
                if scope == "bwd":
                    return run_bwd(context)
                if scope == "rbwd":
                    return run_bwd((context[0], recompute(context[0])))
                if scope == "fb":
                    return run_bwd(run_fwd()[1])
                if scope == "ckpt":
                    schedule = run_fwd()[1][0]
                    return run_bwd((schedule, recompute(schedule)))
                if scope == "wquant":
                    return [
                        ops.mxfp8_quantize(w, True, True)
                        for w in (
                            bench.w_routed_gate,
                            bench.w_routed_up,
                            bench.w_routed_down,
                        )
                    ]
                raise ValueError(scope)

            return prepare, run

        for precision in mok_precisions:
            scopes = ("fwd", "bwd", "rbwd", "fb", "ckpt") + (
                ("wquant",) if precision == "MXFP8" else ()
            )
            for scope in scopes:
                prepare, run = mok_case(precision, scope)
                cases.append((f"mok_{precision.lower()}/{scope}", prepare, run))

    # Timing loop
    host = {label: [] for label, *_ in cases}
    event = {label: [] for label, *_ in cases}
    gathered = torch.empty(2 * world, device=device, dtype=torch.float64)
    captured = {}
    for iteration in range(args.warmup + args.iters):
        order = cases if iteration % 2 == 0 else cases[::-1]
        for label, prepare, run in order:
            state = prepare()
            dist.barrier()
            torch.cuda.synchronize()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start = time.perf_counter()
            begin.record()
            out = run(state)
            end.record()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - start) * 1000
            local = torch.tensor(
                [wall, begin.elapsed_time(end)], device=device, dtype=torch.float64
            )
            dist.all_gather_into_tensor(gathered, local)
            values = gathered.view(world, 2)
            if iteration >= args.warmup:
                host[label].append(values[:, 0].max().item())
                event[label].append(values[:, 1].max().item())
            if (
                iteration == args.warmup
                and label.startswith("mono/")
                and label.endswith("/bwd")
            ):
                captured[label] = dict(
                    dx=x.grad.clone(),
                    dscore=scores.grad.clone(),
                    dw1=w1.grad.clone(),
                    dw2=w2.grad.clone(),
                )
            if (
                label.startswith("mono/")
                and label.endswith("/fwd")
                and iteration == args.warmup
            ):
                captured[label] = dict(y=out.detach().clone())
            del out, state
    set_env({})

    # Parity of every mono-ep variant against the first, and optionally against FP32
    parity = {}
    first = variants[0][0]
    for label, tensors in captured.items():
        _, name, scope = label.split("/")
        reference_label = f"mono/{first}/{scope}"
        if name == first:
            continue
        entry = {}
        for key, a in tensors.items():
            b = captured[reference_label][key]
            stats = torch.tensor(
                [float(not torch.equal(a, b)), _rel(a, b)], device=device
            )
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            entry[key] = dict(bitwise=stats[0].item() == 0.0, rel=stats[1].item())
        parity[f"{name}_vs_{first}/{scope}"] = entry
    fp32 = None
    if args.reference:
        fp32 = _fp32_reference(
            x, ids, scores, w1, w2, dy, world, local_experts, captured, variants
        )

    def med(samples):
        return {k: statistics.median(v) for k, v in samples.items() if v}

    payload = dict(
        shape_name=args.shape,
        shape=dict(shape, topk=topk, world=world, local_experts=local_experts),
        routing=args.routing,
        variants={n: e for n, e in variants},
        launch_env=launch_env,
        warmup=args.warmup,
        iters=args.iters,
        compiled=not args.no_compile,
        device=torch.cuda.get_device_name(device),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nccl=".".join(map(str, torch.cuda.nccl.version())),
        mok=mok_payload,
        host_wall_median_ms=med(host),
        cuda_event_median_ms=med(event),
        host_wall_samples_ms=host,
        parity_first_measured=parity,
        fp32_reference=fp32,
        peak_alloc_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        timing="per case: synchronized host-wall span (and CUDA events), max over ranks, median over iterations; cases interleaved, order reversed every other iteration",
    )
    if rank == 0:
        print("MOKVS_JSON " + json.dumps(payload), flush=True)
    dist.barrier()
    runtime.close()
    if mok_precisions:
        functional.clear_workspace_cache()
    dist.destroy_process_group()
    os._exit(0)


def _fp32_reference(
    x, ids, scores, w1, w2, dy, world, local_experts, captured, variants
):
    """dx / dscore / dW relative L2 of each mono-ep variant against an FP32 all-gather reference.

    Every rank gathers all tokens, runs its local experts in FP32 on the rows routed to them,
    and the partial outputs are reduced; gradients follow by autograd on the FP32 graph.
    """
    rank = dist.get_rank()
    gx = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gx, x.detach())
    gi = [torch.empty_like(ids) for _ in range(world)]
    dist.all_gather(gi, ids)
    gs = [torch.empty_like(scores) for _ in range(world)]
    dist.all_gather(gs, scores.detach())
    gdy = [torch.empty_like(dy) for _ in range(world)]
    dist.all_gather(gdy, dy)
    X = torch.cat(gx).float().requires_grad_()
    S = torch.cat(gs).float().requires_grad_()
    I = torch.cat(gi)
    W1 = w1.detach().float().requires_grad_()
    W2 = w2.detach().float().requires_grad_()
    inner = W2.shape[2]
    DY = torch.cat(gdy).float()
    loss = X.new_zeros(())
    for e in range(local_experts):
        token, slot = (I == rank * local_experts + e).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        z = X[token] @ W1[e].t()
        h = torch.nn.functional.silu(z[:, :inner]) * z[:, inner:]
        loss = loss + ((h @ W2[e].t()) * S[token, slot, None] * DY[token]).sum()
    loss.backward()
    dX = X.grad.clone()
    dist.all_reduce(dX)
    dS = S.grad.clone()
    dist.all_reduce(dS)
    tokens = x.shape[0]
    ref = dict(
        dx=dX[rank * tokens : (rank + 1) * tokens],
        dscore=dS[rank * tokens : (rank + 1) * tokens],
        dw1=W1.grad,
        dw2=W2.grad,
    )
    out = {}
    for name, _ in variants:
        tensors = captured.get(f"mono/{name}/bwd")
        if tensors is None:
            continue
        stats = torch.tensor(
            [_rel(tensors[k], ref[k]) for k in ("dx", "dscore", "dw1", "dw2")],
            device=x.device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        out[name] = dict(
            zip(("dx", "dscore", "dw1", "dw2"), stats.tolist(), strict=True)
        )
    return out


if __name__ == "__main__":
    main()
