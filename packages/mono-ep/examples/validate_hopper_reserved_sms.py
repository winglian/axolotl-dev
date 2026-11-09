# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate reserved-SM persistent training alongside bounded NCCL work."""

import json
import os
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from mono_ep.runtime import PersistentMoE


def _gpu_overlap(trace):
    kernels = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel" and event.get("ph") == "X"
    ]
    nccl = [event for event in kernels if "nccl" in event["name"].lower()]
    result = {}
    for direction in ("forward", "backward"):
        persistent = [
            event
            for event in kernels
            if f"sm90_fp8_mono_ep_{direction}_impl" in event["name"]
        ]
        overlap_us = 0.0
        overlapping_kernels = 0
        streams = set()
        maximum_streams = 0
        for event in persistent:
            start, end = event["ts"], event["ts"] + event["dur"]
            edges = []
            for collective in nccl:
                left = max(start, collective["ts"])
                right = min(end, collective["ts"] + collective["dur"])
                if left < right:
                    stream = collective.get("args", {}).get("stream", "unknown")
                    streams.add(stream)
                    edges.extend(((left, 1, stream), (right, -1, stream)))
            if edges:
                overlapping_kernels += 1
            active = {}
            previous = start
            for timestamp, delta, stream in sorted(edges):
                if active:
                    overlap_us += timestamp - previous
                active[stream] = active.get(stream, 0) + delta
                if not active[stream]:
                    del active[stream]
                maximum_streams = max(maximum_streams, len(active))
                previous = timestamp
        result[direction] = dict(
            persistent_kernels=len(persistent),
            overlapping_persistent_kernels=overlapping_kernels,
            overlap_us=overlap_us,
            overlapping_nccl_streams=len(streams),
            maximum_simultaneous_nccl_streams=maximum_streams,
        )
        if not overlapping_kernels or overlap_us <= 0:
            raise AssertionError(
                f"No measured NCCL GPU-kernel overlap for {direction}: {result}"
            )
    return result


def _snapshot(operation, x, ids, scores, w1, w2, dy, collectives=None):
    tensors = tuple(
        t.detach().clone().requires_grad_(True) for t in (x, scores, w1, w2)
    )
    current_x, current_scores, current_w1, current_w2 = tensors
    if collectives is not None:
        collectives()
    y = operation(current_x, ids, current_scores, current_w1, current_w2)
    if collectives is not None:
        collectives(wait=True)
        collectives()
    y.backward(dy)
    if collectives is not None:
        collectives(wait=True)
    return (y.detach(), *(tensor.grad.detach() for tensor in tensors))


def run():
    import deep_gemm

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    timeout = timedelta(seconds=120)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", rank), timeout=timeout
    )
    world = dist.get_world_size()
    if world != 2:
        raise ValueError("reserved-SM validation requires two ranks")
    groups = [dist.new_group(timeout=timeout) for _ in range(3)]
    streams = [torch.cuda.Stream() for _ in groups]
    collective_tensors = [
        torch.full((8 * 1024 * 1024,), rank + index + 1, device="cuda")
        for index in range(3)
    ]
    works = []

    def collectives(wait=False, batches=1):
        nonlocal works
        if wait:
            for work in works:
                work.wait()
            for stream in streams:
                stream.synchronize()
            works = []
            return
        if works:
            raise RuntimeError("previous collectives are still active")
        for _ in range(batches):
            for tensor, group, stream in zip(
                collective_tensors, groups, streams, strict=True
            ):
                with torch.cuda.stream(stream):
                    tensor.fill_(rank + 1)
                    work = dist.all_reduce(tensor, group=group, async_op=True)
                    # Order the next fill after NCCL finishes reading this tensor.
                    work.wait()
                    works.append(work)

    physical_sms = torch.cuda.get_device_properties(rank).multi_processor_count
    reserved_sms = 16
    active_sms = physical_sms - reserved_sms
    deep_gemm.set_num_sms(active_sms)
    tokens, hidden, inner, experts, topk = 137, 512, 512, 8, 4
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens + 31,
        topk,
        hidden,
        inner,
        compute_dtype="blockwise",
        retain_activations=False,
        backward_backend="native",
        reserved_sms=reserved_sms,
    )
    operation = torch.compile(
        runtime, fullgraph=True, options={"triton.cudagraphs": False}
    )
    torch.manual_seed(731)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk].contiguous()
    scores = torch.rand(tokens, topk, device="cuda").softmax(-1)
    local = experts // world
    w1 = (
        torch.randn(local, 2 * inner, hidden, device="cuda", dtype=torch.bfloat16)
        / hidden**0.5
    )
    w2 = (
        torch.randn(local, hidden, inner, device="cuda", dtype=torch.bfloat16)
        / inner**0.5
    )
    dy = torch.randn_like(x)

    _snapshot(operation, x, ids, scores, w1, w2, dy)
    reference = _snapshot(operation, x, ids, scores, w1, w2, dy)
    torch.cuda.synchronize()
    overlap_batches = 128

    def sustained_collectives(wait=False):
        collectives(wait=wait, batches=overlap_batches)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        profiled = _snapshot(
            operation, x, ids, scores, w1, w2, dy, collectives=sustained_collectives
        )
        torch.cuda.synchronize()
    for observed, expected in zip(profiled, reference, strict=True):
        torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)
    with tempfile.TemporaryDirectory(prefix="reserved_sm_overlap_") as directory:
        trace_path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        overlap = _gpu_overlap(json.loads(trace_path.read_text()))
    max_errors = [0.0] * len(reference)
    started = time.monotonic()
    for _ in range(20):
        actual = _snapshot(
            operation, x, ids, scores, w1, w2, dy, collectives=collectives
        )
        for index, (observed, expected) in enumerate(
            zip(actual, reference, strict=True)
        ):
            torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)
            error = (
                observed.float() - expected.float()
            ).norm() / expected.float().norm().clamp_min(1e-8)
            max_errors[index] = max(max_errors[index], error.item())

    expected_collective = float(sum(range(1, world + 1)))
    for tensor in collective_tensors:
        torch.testing.assert_close(
            tensor, torch.full_like(tensor, expected_collective), rtol=0, atol=0
        )

    deep_gemm.set_num_sms(physical_sms)
    rejected_budget_change = False
    try:
        operation(x, ids, scores, w1, w2)
    except RuntimeError as error:
        rejected_budget_change = "SM count changed" in str(error)
    finally:
        deep_gemm.set_num_sms(active_sms)
    if not rejected_budget_change:
        raise AssertionError("post-allocation SM-count change was not rejected")

    record = dict(
        rank=rank,
        iterations=20,
        physical_sms=physical_sms,
        reserved_sms=reserved_sms,
        active_sms=active_sms,
        nccl_max_ctas=os.environ.get("NCCL_MAX_CTAS"),
        concurrent_collectives=3,
        compute_dtype="blockwise_fp8",
        separate_process_groups=True,
        separate_streams=True,
        forward_and_backward_overlap=True,
        overlap_evidence="CUPTI GPU kernel intervals; at least one NCCL kernel per direction",
        overlap_collective_batches_per_group=overlap_batches,
        gpu_overlap=overlap,
        watchdog_seconds=120,
        post_allocation_budget_change_rejected=rejected_budget_change,
        relative_errors=max_errors,
        elapsed_seconds=time.monotonic() - started,
    )
    print(json.dumps(record), flush=True)
    runtime.close()
    deep_gemm.set_num_sms(physical_sms)
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
