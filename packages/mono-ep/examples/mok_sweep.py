# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Time Mixture-of-Kittens over a list of configs in one process.

Run under torchrun with the MoK checkout on PYTHONPATH. Reuses MoK's own
``benchmarks.bench_mok.MoKBenchmark``, ``benchmark_fwd``/``benchmark_bwd`` and
``check_benchmark_correctness``; the config globals and iteration counts are
swapped per config so the inputs, reference and workspace are built once.
Adds one timing MoK's harness lacks: ``recompute_forward_context`` + backward,
the checkpointed backward that mono-ep's 2F+B contract counts.

Prints ``MOKSWEEP_JSON <payload>`` on rank 0.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def routed_tflop(t, topk, h, i):
    return 6 * t * topk * h * i / 1e12


def mok_tflop(t, topk, h, i):
    return 6 * (t + t * topk) * h * i / 1e12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--configs", required=True, help="JSON list of config dicts")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--precisions", default="BF16,MXFP8")
    parser.add_argument("--no-recompute", action="store_true")
    args = parser.parse_args()

    from benchmarks import bench_mok, utils
    from mok import functional

    from tests.utils import (
        BF16_TOLERANCE,
        MXFP8_TOLERANCE,
        generate_inputs,
        run_reference_bf16,
    )

    utils.WARMUP_ITERS = args.warmup
    utils.TIMED_ITERS = args.iters
    rank, world, device = utils.init_distributed()
    local_experts = utils.get_num_local_experts(args.experts, world)
    shape = dict(
        tokens=args.tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
        experts=args.experts,
        local_experts=local_experts,
        topk=args.topk,
        world=world,
    )
    inputs = generate_inputs(
        rank,
        device,
        args.experts,
        local_experts,
        args.topk,
        args.tokens,
        args.hidden,
        args.intermediate,
    )
    started = time.time()
    bench = bench_mok.MoKBenchmark(inputs)
    reference = run_reference_bf16(*inputs)
    torch.cuda.synchronize()
    setup_s = time.time() - started

    def recompute_bwd_ms(precision, cfg):
        routed = (
            (bench.w_gate_mxfp8[:2], bench.w_up_mxfp8[:2])
            if precision == "MXFP8"
            else (bench.w_routed_gate, bench.w_routed_up)
        )
        run_fwd = getattr(bench, f"run_{precision.lower()}_fwd")
        run_bwd = getattr(bench, f"run_{precision.lower()}_bwd")

        def step():
            schedule = run_fwd()[1][0]
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            ctx = functional.recompute_forward_context(
                cfg,
                bench.workspace,
                schedule,
                bench.x,
                bench.w_shared_gate,
                bench.w_shared_up,
                *routed,
            )
            run_bwd((schedule, ctx))
            end.record()
            return start, end

        for _ in range(args.warmup):
            step()
        barrier = dist.barrier(async_op=True)
        barrier.block_current_stream()
        events = [step() for _ in range(args.iters)]
        torch.cuda.synchronize()
        dist.barrier()
        return utils.median_rank_max_latency(
            [s.elapsed_time(e) for s, e in events], device
        )

    def agree(ok):
        flag = torch.tensor([1 if ok else 0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    tolerances = dict(BF16=BF16_TOLERANCE, MXFP8=MXFP8_TOLERANCE)
    results = []
    for raw in json.loads(args.configs):
        cfg = functional.MoKConfig(
            fwd_num_comm_sms=raw["fwd_comm_sms"],
            bwd_num_comm_sms=raw["bwd_comm_sms"],
            minibatch_size=raw["minibatch"],
            macrobatch_size=raw["macrobatch"],
        )
        bench_mok.BF16_CONFIG = cfg
        bench_mok.MXFP8_CONFIG = cfg
        for precision in args.precisions.split(","):
            run_fwd = getattr(bench, f"run_{precision.lower()}_fwd")
            run_bwd = getattr(bench, f"run_{precision.lower()}_bwd")
            row = dict(precision=precision, **raw)
            torch.cuda.reset_peak_memory_stats(device)
            try:
                utils.check_benchmark_correctness(
                    precision, run_fwd, run_bwd, reference, tolerances[precision], rank
                )
                ok, error = True, None
            except (AssertionError, ValueError, RuntimeError) as exc:
                ok, error = False, f"{type(exc).__name__}: {exc}"
            row["correct"] = agree(ok)
            row["error_rank0"] = error
            if error and "exceeded tolerance" not in error:
                results.append(row)
                continue
            row["fwd_ms"] = utils.benchmark_fwd(run_fwd, device)
            row["bwd_ms"] = utils.benchmark_bwd(run_fwd, run_bwd, device)
            if not args.no_recompute:
                row["recompute_bwd_ms"] = recompute_bwd_ms(precision, cfg)
            row["peak_alloc_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            free, total = torch.cuda.mem_get_info(device)
            row["device_used_gib"] = (total - free) / 2**30
            results.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)

    dims = (args.tokens, args.topk, args.hidden, args.intermediate)
    payload = dict(
        shape=shape,
        routed_fwd_tflop=routed_tflop(*dims),
        mok_fwd_tflop=mok_tflop(*dims),
        warmup=args.warmup,
        iters=args.iters,
        setup_s=setup_s,
        device=torch.cuda.get_device_name(device),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nccl=".".join(map(str, torch.cuda.nccl.version())),
        results=results,
    )
    if rank == 0:
        print("MOKSWEEP_JSON " + json.dumps(payload), flush=True)
    dist.barrier()
    functional.clear_workspace_cache()
    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
