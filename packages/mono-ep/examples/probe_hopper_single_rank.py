# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-rank persistent backward with the EP8 per-GPU math workload."""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist


def run(capture=False):
    import deep_gemm
    from mono_ep.runtime import PersistentMoE

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    assert dist.get_world_size() == 1
    torch.manual_seed(881)
    tokens, hidden, inner, experts, topk = 16384, 2048, 768, 16, 8
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inner,
        compute_dtype="blockwise",
        retain_activations=False,
        backward_backend="native",
    )
    backend = runtime._peer_backward
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk].contiguous()
    scores = torch.rand(tokens, topk, device="cuda").softmax(-1)
    w1 = torch.randn(experts, 2 * inner, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
    original = deep_gemm._C.fp8_mono_ep_backward_sm90
    intervals = []

    def timed_native(*args):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        result = original(*args)
        end.record()
        intervals.append((start, end))
        return result

    for _ in range(5):
        backend.backward(dy, x, ids, scores, w1, w2)
    torch.cuda.synchronize()
    if capture:
        torch.cuda.cudart().cudaProfilerStart()
        backend.backward(dy, x, ids, scores, w1, w2)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    else:
        deep_gemm._C.fp8_mono_ep_backward_sm90 = timed_native
        try:
            for _ in range(15):
                backend.backward(dy, x, ids, scores, w1, w2)
        finally:
            deep_gemm._C.fp8_mono_ep_backward_sm90 = original
        torch.cuda.synchronize()
    samples = [a.elapsed_time(b) for a, b in intervals]
    print(
        json.dumps(
            dict(
                benchmark="single_rank_persistent_backward",
                device=torch.cuda.get_device_name(),
                world_size=1,
                tokens=tokens,
                hidden=hidden,
                intermediate=inner,
                experts=experts,
                topk=topk,
                routed_rows=int(backend.counts.sum()),
                padded_rows=int(backend.padded_counts.sum()),
                timeout_handler=os.getenv("MONO_EP_SM90_FP8_TRAP_TIMEOUTS"),
                warmup_iterations=5,
                measured_iterations=len(samples),
                median_native_entry_ms=statistics.median(samples) if samples else None,
                samples_ms=samples,
                scope="native host entry CUDA events; excludes preparation and dW; retains local dispatch/combine",
                matching="EP8 per-GPU expected routed rows and 16 local experts; no inter-GPU traffic",
                ncu_capture=capture,
            )
        ),
        flush=True,
    )
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    run(parser.parse_args().capture)
