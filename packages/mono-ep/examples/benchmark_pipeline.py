# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compare chunk schedules with identical SonicMoE compute and NCCL transport."""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from mono_ep import FP8Experts, PersistentMoE
from mono_ep.pipeline import pipelined_moe
from mono_ep.sonic import SonicExperts, sonic_experts, sonic_fp8_experts


def run(args):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    if args.experts % world:
        raise ValueError("experts must divide evenly across ranks")
    torch.manual_seed(72 + rank)
    x = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    scores, ids = (
        torch.randn(args.tokens, args.experts, device="cuda")
        .softmax(-1)
        .topk(args.topk)
    )
    w1 = (
        torch.randn(
            args.experts // world,
            2 * args.inner,
            args.hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / args.hidden**0.5
    )
    w2 = (
        torch.randn(
            args.experts // world,
            args.hidden,
            args.inner,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / args.inner**0.5
    )
    if args.compare_fused or args.compare_sonic_fp8:
        w1, w2 = w1.float(), w2.float()
    inputs = [t.requires_grad_() for t in (x, scores, w1, w2)]
    dy = torch.randn_like(x)
    reference = None
    cases = (
        [
            ("sonic_pipeline", 1, False),
            ("fused_bf16", 1, False),
            ("fused_mxfp8", 1, False),
            ("fp8_pipeline", 1, False),
            ("fp8_pipeline_retained", 1, False),
            ("fp8_pipeline_retained_compiled", 1, False),
            ("fp8_pipeline_retained_legacy_compiled", 1, False),
        ]
        if args.compare_fused
        else [
            ("sonic_pipeline", c, o)
            for c, o in ((1, False), (2, False), (2, True), (4, False), (4, True))
        ]
    )
    if args.compare_sonic_fp8:
        cases = [
            ("sonic_pipeline", 1, False),
            ("sonic_fp8_pipeline", 1, False),
            ("sonic_fp8_pipeline", 2, True),
            ("sonic_pipeline_compiled", 1, False),
            ("sonic_fp8_pipeline_compiled", 1, False),
            ("sonic_fp8_pipeline_compiled", 2, True),
        ]
    if args.sonic_retained_only:
        cases = [
            ("sonic_pipeline_retained_compiled", 1, False),
            ("sonic_fp8_pipeline_compiled", 1, False),
            ("sonic_fp8_pipeline_retained_compiled", 1, False),
            ("sonic_fp8_pipeline_retained_compiled", 2, True),
        ]
    from mono_ep import column_quantization
    from mono_ep.quantization import quantize_fp8

    fused_column_quantization = column_quantization.quantize_columns

    def legacy_column_quantization(t):
        q, sf = quantize_fp8(t.T.contiguous(), block_cols=32)
        return q.T.contiguous(), sf.T.contiguous()

    for backend, chunks, overlap in cases:
        if rank == 0:
            print(
                json.dumps(
                    {
                        "starting_backend": backend,
                        "chunks": chunks,
                        "overlap_backward": overlap,
                    }
                ),
                flush=True,
            )
        column_quantization.quantize_columns = (
            legacy_column_quantization
            if "legacy" in backend
            else fused_column_quantization
        )
        runtime = None
        compute = (
            FP8Experts("mxfp8", compile_local=backend.endswith("compiled"))
            if backend.startswith("fp8_pipeline")
            else sonic_experts
        )
        if backend.startswith("sonic_fp8"):
            compute = sonic_fp8_experts
        if backend.startswith("sonic") and backend.endswith("compiled"):
            compute = torch.compile(compute, fullgraph=True, dynamic=False)
        if backend.startswith("sonic") and "retained" in backend:
            compute = SonicExperts(fp8="fp8" in backend, compile_local=True)
        if backend.startswith("fused_"):
            runtime = PersistentMoE(
                dist.group.WORLD,
                args.experts,
                args.tokens,
                args.topk,
                args.hidden,
                args.inner,
                compute_dtype="mxfp8" if backend == "fused_mxfp8" else "bf16",
            )

        def step(
            chunks=chunks,
            overlap=overlap,
            runtime=runtime,
            compute=compute,
            backend=backend,
        ):
            if runtime is not None:
                y = runtime(inputs[0], ids, inputs[1], inputs[2], inputs[3])
                return y, *torch.autograd.grad(y, inputs, dy)
            y = pipelined_moe(
                inputs[0],
                ids,
                inputs[1],
                inputs[2],
                inputs[3],
                group=dist.group.WORLD,
                local_kernel=compute,
                local_backward=compute.backward
                if isinstance(compute, (FP8Experts, SonicExperts))
                else None,
                local_forward=compute.forward_with_activations
                if "retained" in backend
                else None,
                chunks=chunks,
                overlap_backward=overlap,
            )
            return y, *torch.autograd.grad(y, inputs, dy)

        actual = step()
        if reference is None:
            reference = tuple(t.detach().clone() for t in actual)
        errors = []
        for value, expected in zip(actual, reference, strict=True):
            error = (
                (value.float() - expected.float()).norm()
                / expected.float().norm().clamp_min(1e-8)
            ).item()
            if not torch.isfinite(value).all() or error >= (
                0.15 if "fp8" in backend else 0.03
            ):
                raise AssertionError((rank, chunks, overlap, error))
            errors.append(error)
        all_errors = torch.tensor(errors, device="cuda")
        dist.all_reduce(all_errors, op=dist.ReduceOp.MAX)
        del actual, value, expected
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.iters):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            elapsed = torch.tensor((time.perf_counter() - start) * 1000, device="cuda")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            times.append(elapsed.item())
        peak = torch.tensor(torch.cuda.max_memory_allocated() - baseline, device="cuda")
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "backend": backend,
                        "world_size": world,
                        "device": torch.cuda.get_device_name(),
                        "master_dtype": str(w1.dtype),
                        "weight_grad_operand_prep": "per_tensor"
                        if backend.startswith("sonic_fp8")
                        else "transpose_reference"
                        if "legacy" in backend
                        else "fused_column"
                        if "fp8" in backend
                        else "not_applicable",
                        "compute_dtype": "per_tensor_e4m3"
                        if backend.startswith("sonic_fp8")
                        else "mxfp8"
                        if "fp8" in backend
                        else "bfloat16",
                        "scope": "dispatch+experts+combine forward+backward",
                        "local_forward_recomputed_in_backward": (
                            backend.startswith("sonic") and "retained" not in backend
                        )
                        or backend in ("fp8_pipeline", "fused_bf16", "fused_mxfp8"),
                        "retained_activations": "retained" in backend,
                        "compiled": False,
                        "local_experts_fullgraph": backend.endswith("compiled"),
                        "chunks": chunks,
                        "overlap_backward": overlap,
                        "tokens_per_rank": args.tokens,
                        "hidden": args.hidden,
                        "intermediate": args.inner,
                        "experts": args.experts,
                        "topk": args.topk,
                        "median_ms": statistics.median(times),
                        "min_ms": min(times),
                        "samples_ms": times,
                        "mean_ms": statistics.mean(times),
                        "iterations": args.iters,
                        "peak_increment_bytes": peak.item(),
                        "rank0_relative_l2_vs_sonic_one_chunk": errors,
                        "max_rank_relative_l2_vs_sonic_one_chunk": all_errors.tolist(),
                    }
                ),
                flush=True,
            )
        if args.profile and backend in (
            "fused_mxfp8",
            "fp8_pipeline_retained",
            "sonic_fp8_pipeline_compiled",
        ):
            dist.barrier()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for _ in range(3):
                    step()
                torch.cuda.synchronize()
            if rank == 0:
                events = prof.key_averages()
                for metric in ("self_device_time_total", "cpu_time_total"):
                    selected = sorted(
                        events, key=lambda e: getattr(e, metric), reverse=True
                    )[:30]
                    print(
                        json.dumps(
                            {
                                "profile_backend": backend,
                                "iterations": 3,
                                "sort": metric,
                                "events": [
                                    {
                                        "name": e.key,
                                        "calls": e.count,
                                        "self_cuda_us": e.self_device_time_total,
                                        "total_cuda_us": e.device_time_total,
                                        "self_cpu_us": e.self_cpu_time_total,
                                        "total_cpu_us": e.cpu_time_total,
                                    }
                                    for e in selected
                                ],
                            }
                        ),
                        flush=True,
                    )
        if runtime is not None:
            runtime.buffer.destroy()
        dist.barrier()
    column_quantization.quantize_columns = fused_column_quantization
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inner", type=int, default=768)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--compare-fused", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compare-sonic-fp8", action="store_true")
    parser.add_argument("--sonic-retained-only", action="store_true")
    args = parser.parse_args()
    args.compare_sonic_fp8 |= args.sonic_retained_only
    run(args)
