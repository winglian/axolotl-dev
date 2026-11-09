# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Qwen3-30B-A3B layer geometry from Prime's published eight-H200 benchmark."""

import json
import os
import statistics
import subprocess
import time
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from mono_ep.fp8 import expert_backward_fp8, expert_forward_fp8
from mono_ep.grouped import ExpertLayout
from mono_ep.pipeline import _build_send_layout, _compute_send_counts, exchange
from validate import reference


def gather(t):
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, t)
    return torch.cat(parts)


class Layer:
    def __init__(self, w1, w2, compiled, retain_activations=False):
        self.w1, self.w2 = w1, w2
        self.retain_activations = retain_activations
        self.forward_local = partial(
            expert_forward_fp8,
            format="blockwise",
            retain_activations=retain_activations,
        )
        self.backward_local = partial(expert_backward_fp8, format="blockwise")
        if compiled:
            self.forward_local = torch.compile(self.forward_local, fullgraph=True)
            self.backward_local = torch.compile(self.backward_local, fullgraph=True)

    def forward(self, x, ids, scores):
        group = dist.group.WORLD
        world = dist.get_world_size()
        dest, membership, send = _compute_send_counts(ids, world, self.w1.shape[0])
        recv = funcol.wait_tensor(exchange(send, [1] * world, [1] * world, group))
        send, recv = torch.stack((send, recv)).cpu().tolist()
        token, _, local_ids, weights = _build_send_layout(
            ids, scores, dest, membership, sum(send), self.w1.shape[0]
        )
        rx = exchange(x[token], recv, send, group)
        rw = exchange(weights, recv, send, group)
        ri = exchange(local_ids, recv, send, group)
        rx, rw, ri = (funcol.wait_tensor(t) for t in (rx, rw, ri))
        layout = ExpertLayout.build(ri, self.w1.shape[0], 128)
        local_y = self.forward_local(rx, rw, self.w1, self.w2, layout)
        activations = None
        if self.retain_activations:
            local_y, activations = local_y
        out = funcol.wait_tensor(exchange(local_y, send, recv, group))
        y = (
            torch.zeros_like(x, dtype=torch.float32)
            .index_add_(0, token, out.float())
            .to(x.dtype)
        )
        return y, (
            rx,
            rw,
            layout,
            token,
            local_ids,
            send,
            recv,
            scores.shape,
            activations,
        )

    def backward(self, dy, state):
        rx, rw, layout, token, ids, send, recv, score_shape, activations = state
        grad = funcol.wait_tensor(exchange(dy[token], recv, send, dist.group.WORLD))
        gx, gs, gw1, gw2 = self.backward_local(
            grad, rx, rw, self.w1, self.w2, layout, activations=activations
        )
        gx = exchange(gx, send, recv, dist.group.WORLD)
        gs = exchange(gs, send, recv, dist.group.WORLD)
        gx, gs = funcol.wait_tensor(gx), funcol.wait_tensor(gs)
        dx = (
            torch.zeros_like(dy, dtype=torch.float32)
            .index_add_(0, token, gx.float())
            .to(dy.dtype)
        )
        ds = torch.zeros(score_shape, device=dy.device).index_add_(
            0, token, gs.masked_fill(ids < 0, 0)
        )
        return dx, ds, gw1, gw2


def inputs(tokens):
    x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
    scores, ids = torch.rand(tokens, 128, device="cuda").topk(8)
    scores /= scores.sum(-1, keepdim=True)
    return x, ids, scores, torch.randn_like(x)


def run():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    if dist.get_world_size() != 8 or "H200" not in torch.cuda.get_device_name():
        raise RuntimeError("this comparison requires exactly eight H200 GPUs")
    torch.manual_seed(91 + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    w1 = torch.randn(16, 1536, 2048, device="cuda") / 2048**0.5
    w2 = torch.randn(16, 2048, 768, device="cuda") / 768**0.5
    layer = Layer(w1, w2, compiled=False)
    x, ids, scores, dy = inputs(33)
    ids[::7] = -1
    full_x, full_ids, full_scores, full_dy = (gather(t) for t in (x, ids, scores, dy))
    full_w1, full_w2 = gather(w1), gather(w2)
    leaves = [t.requires_grad_() for t in (full_x, full_scores, full_w1, full_w2)]
    expected = reference(
        leaves[0], leaves[1], full_ids, leaves[2].bfloat16(), leaves[3].bfloat16()
    )
    grads = torch.autograd.grad(expected, leaves, full_dy)
    y, state = layer.forward(x, ids, scores)
    actual_grads = layer.backward(dy, state)
    retained = Layer(w1, w2, compiled=True, retain_activations=True)
    retained_y, retained_state = retained.forward(x, ids, scores)
    retained_grads = retained.backward(dy, retained_state)
    for a, b in zip((retained_y, *retained_grads), (y, *actual_grads), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    del retained_y, retained_state, retained_grads
    refs = (
        expected[rank * 33 : (rank + 1) * 33],
        grads[0][rank * 33 : (rank + 1) * 33],
        grads[1][rank * 33 : (rank + 1) * 33],
        grads[2][rank * 16 : (rank + 1) * 16],
        grads[3][rank * 16 : (rank + 1) * 16],
    )
    errors = []
    for a, b in zip((y, *actual_grads), refs, strict=True):
        error = (
            (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
        ).item()
        assert torch.isfinite(a).all() and error < 0.15, (rank, error)
        errors.append(error)
    print(
        json.dumps({"validation_rank": rank, "relative_l2_vs_bf16": errors}), flush=True
    )
    del (
        leaves,
        expected,
        grads,
        actual_grads,
        refs,
        state,
        full_x,
        full_scores,
        full_ids,
        full_dy,
        full_w1,
        full_w2,
    )
    layer = Layer(w1, w2, compiled=True)
    x, ids, scores, dy = inputs(16384)
    if rank == 0:
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=False
        )
        print(
            json.dumps(
                {
                    "topology_returncode": topology.returncode,
                    "topology_stdout": topology.stdout,
                    "topology_stderr": topology.stderr,
                }
            ),
            flush=True,
        )
        print(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(),
                }
            ),
            flush=True,
        )

    def step(scope):
        if scope in ("no_checkpoint_layer", "checkpoint_retained_layer"):
            if scope == "checkpoint_retained_layer":
                _, discarded = layer.forward(x, ids, scores)
                del discarded
            _, retained_state = retained.forward(x, ids, scores)
            return retained.backward(dy, retained_state)
        if scope == "forward":
            return layer.forward(x, ids, scores)
        if scope == "backward":
            return layer.backward(dy, saved)
        _, first_state = layer.forward(x, ids, scores)
        del first_state
        _, recomputed = layer.forward(x, ids, scores)
        return layer.backward(dy, recomputed)

    # Validate compilation at the timed shape before collecting any timings.
    y, saved = layer.forward(x, ids, scores)
    check_grads = layer.backward(dy, saved)
    assert all(torch.isfinite(t).all() for t in (y, *check_grads))
    del check_grads
    retained_y, retained_state = retained.forward(x, ids, scores)
    retained_grads = retained.backward(dy, retained_state)
    for a, b in zip(
        (retained_y, *retained_grads), (y, *layer.backward(dy, saved)), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    retained_bytes = sum(t.numel() * t.element_size() for t in retained_state[-1])
    del retained_y, retained_state, retained_grads, a, b
    for scope in (
        "forward",
        "backward",
        "checkpointed_layer",
        "no_checkpoint_layer",
        "checkpoint_retained_layer",
    ):
        if scope == "no_checkpoint_layer":
            del saved, y
        times = []
        peaks = []
        baseline = torch.cuda.memory_allocated()
        dist.all_reduce(
            (baseline_tensor := torch.tensor(baseline, device="cuda")),
            op=dist.ReduceOp.MAX,
        )
        for iteration in range(20):
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = step(scope)
            torch.cuda.synchronize()
            elapsed = torch.tensor((time.perf_counter() - start) * 1000, device="cuda")
            peak = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda")
            del result
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if iteration >= 5:
                times.append(elapsed.item())
                peaks.append(peak.item())
        if rank == 0:
            print(
                json.dumps(
                    {
                        "benchmark": "qwen3_30b_a3b_h200_ep8",
                        "backend": "prototype_deepgemm_fp8_nccl",
                        "scope": scope,
                        "mean_ms": statistics.mean(times),
                        "median_ms": statistics.median(times),
                        "samples_ms": times,
                        "tokens_per_gpu": 16384,
                        "hidden": 2048,
                        "intermediate": 768,
                        "experts": 128,
                        "topk": 8,
                        "world_size": 8,
                        "master_dtype": "float32",
                        "compute": "blockwise_fp8",
                        "local_fullgraph": True,
                        "whole_layer_fullgraph": False,
                        "backward_recomputes_w1": scope
                        not in ("no_checkpoint_layer", "checkpoint_retained_layer"),
                        "baseline_allocated_bytes_max_rank": baseline_tensor.item(),
                        "peak_allocated_bytes_max_rank": max(peaks),
                        "retained_local_activation_bytes_this_rank": retained_bytes,
                        "quantization_in_timing": True,
                        "host_split_sizes": True,
                        "hopper_wgrad_quantization": "fused_triton",
                    }
                ),
                flush=True,
            )
    if os.environ.get("MOE_PROFILE") == "1":
        _, saved = layer.forward(x, ids, scores)
        dist.barrier()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as profile:
            for _ in range(3):
                result = step("backward")
                del result
            torch.cuda.synchronize()
        if rank == 0:
            events = profile.key_averages()
            for metric in ("self_device_time_total", "cpu_time_total"):
                entries = sorted(
                    events, key=lambda event: getattr(event, metric), reverse=True
                )[:30]
                print(
                    json.dumps(
                        {
                            "backward_profile": metric,
                            "iterations": 3,
                            "rank": rank,
                            "events": [
                                {
                                    "name": event.key,
                                    "calls": event.count,
                                    "self_cuda_us": event.self_device_time_total,
                                    "total_cuda_us": event.device_time_total,
                                    "self_cpu_us": event.self_cpu_time_total,
                                    "total_cpu_us": event.cpu_time_total,
                                }
                                for event in entries
                            ],
                        }
                    ),
                    flush=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
