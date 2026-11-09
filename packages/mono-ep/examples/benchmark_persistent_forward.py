# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Qwen-sized persistent forward, with prepared and complete operand scopes."""

import argparse
import json
import os
import statistics
import time

import deep_gemm
import torch
import torch.distributed as dist
from mono_ep.token_quantization import prepare_persistent_mxfp8


def run(retain_activations=False, include_router=False):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    tokens, hidden, inner, experts, topk = 16384, 2048, 768, 128, 8
    assert world == 8
    torch.manual_seed(812 + rank)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk].contiguous()
    scores = torch.rand(tokens, topk, device="cuda")
    scores /= scores.sum(-1, keepdim=True)
    w1 = torch.randn(experts // world, 2 * inner, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(experts // world, hidden, inner, device="cuda") / inner**0.5
    router_weight = torch.randn(experts, hidden, device="cuda") / hidden**0.5
    dist.broadcast(router_weight, src=0)

    def route():
        logits = torch.nn.functional.linear(x, router_weight.bfloat16())
        probabilities = logits.float().softmax(-1)
        routing_scores, routing_ids = probabilities.topk(topk, dim=-1)
        routing_scores = routing_scores / routing_scores.sum(-1, keepdim=True)
        return routing_ids.contiguous(), routing_scores.contiguous()

    if include_router:
        ids, scores = route()
    sample_count = 128

    def gather(t):
        parts = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(parts, t)
        return torch.cat(parts)

    sample_x, sample_ids, sample_scores = [
        gather(t[:sample_count].contiguous()) for t in (x, ids, scores)
    ]
    reference = torch.zeros(world * sample_count, hidden, device="cuda")
    expected_z = torch.full(
        (world * sample_count, topk, 2 * inner),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    for local in range(experts // world):
        row, slot = torch.where(sample_ids == rank * (experts // world) + local)
        z = sample_x[row] @ w1[local].bfloat16().T
        expected_z[row, slot] = z
        gate, up = z.float().chunk(2, -1)
        h = (
            torch.nn.functional.silu(gate) * up * sample_scores[row, slot, None]
        ).bfloat16()
        reference.index_add_(0, row, (h @ w2[local].bfloat16().T).float())
    dist.all_reduce(reference)
    reference = reference[rank * sample_count : (rank + 1) * sample_count]
    for dtype in ("bf16",) if retain_activations else ("bf16", "mxfp8"):
        buffer = deep_gemm.get_symm_buffer_for_mono_ep(
            dist.group.WORLD,
            experts,
            tokens,
            topk,
            hidden,
            inner,
            mma_type="bf16xbf16" if dtype == "bf16" else "fp8xfp8",
        )
        forward = (
            deep_gemm.bf16_mono_ep if dtype == "bf16" else deep_gemm.fp8_fp4_mono_ep
        )

        def prepare(dtype=dtype, buffer=buffer, compute_router=False):
            current_ids, current_scores = route() if compute_router else (ids, scores)
            if dtype == "bf16":
                buffer.x[:tokens].copy_(x)
                weights = deep_gemm.transform_weights_for_mono_ep(
                    w1.bfloat16(), w2.bfloat16()
                )
            else:
                qx, sx = prepare_persistent_mxfp8(x)
                buffer.x[:tokens].copy_(qx)
                buffer.x_sf[:tokens].copy_(sx)
                weights = (
                    prepare_persistent_mxfp8(w1, weights=True, interleave=True),
                    prepare_persistent_mxfp8(w2, weights=True),
                )
            buffer.topk_idx[:tokens].copy_(current_ids)
            buffer.topk_weights[:tokens].copy_(current_scores)
            return weights

        weights = prepare()
        output = torch.empty_like(x)
        saved_z = None
        if retain_activations:
            saved_z = x.new_empty(
                (world, buffer.num_max_tokens_per_rank, topk, 2 * inner)
            )
        options = {"saved_z": saved_z} if retain_activations else {}
        forward(output, *weights, buffer, fast_math=False, **options)
        torch.cuda.synchronize()
        if retain_activations:
            valid = torch.isfinite(expected_z)
            actual_z = saved_z[:, :sample_count].reshape_as(expected_z)
            torch.testing.assert_close(
                actual_z[valid], expected_z[valid], atol=0, rtol=0
            )
        error = (output[:sample_count].float() - reference).norm() / reference.norm()
        assert error < (0.005 if dtype == "bf16" else 0.15), error.item()
        scopes = ("prepared_forward", "complete_forward")
        if include_router:
            scopes += ("router_inclusive_forward",)
        timings = {scope: [] for scope in scopes}
        for iteration in range(20):
            for scope in scopes if iteration % 2 == 0 else scopes[::-1]:
                dist.barrier()
                torch.cuda.synchronize()
                start = time.perf_counter()
                if scope != "prepared_forward":
                    weights = prepare(
                        compute_router=scope == "router_inclusive_forward"
                    )
                    output = torch.empty_like(x)
                    if retain_activations:
                        saved_z = torch.empty_like(saved_z)
                        options = {"saved_z": saved_z}
                forward(output, *weights, buffer, fast_math=False, **options)
                torch.cuda.synchronize()
                elapsed = torch.tensor(
                    (time.perf_counter() - start) * 1000, device="cuda"
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                if iteration >= 5:
                    timings[scope].append(elapsed.item())
        assert torch.isfinite(output).all()
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        benchmark="qwen3_30b_a3b_persistent_forward",
                        device=torch.cuda.get_device_name(),
                        torch_version=torch.__version__,
                        world_size=world,
                        tokens_per_gpu=tokens,
                        hidden=hidden,
                        intermediate=inner,
                        experts=experts,
                        topk=topk,
                        compute_dtype=dtype,
                        master_dtype="float32",
                        retained_activations=retain_activations,
                        retained_bytes_per_rank=saved_z.numel() * saved_z.element_size()
                        if saved_z is not None
                        else 0,
                        router_in_timing={
                            scope: scope == "router_inclusive_forward"
                            for scope in scopes
                        },
                        routing_source="bf16_linear_fp32_softmax_topk_normalized"
                        if include_router
                        else "random_distinct_experts",
                        router_fused=False,
                        whole_layer_fullgraph=False,
                        fast_math=False,
                        warmup_iterations=5,
                        measured_iterations=15,
                        timing="host_wall_synchronized_maximum_rank",
                        sample_relative_l2=error.item(),
                        samples_ms=timings,
                        median_ms={k: statistics.median(v) for k, v in timings.items()},
                    )
                ),
                flush=True,
            )
        dist.barrier()
        buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retain-activations", action="store_true")
    parser.add_argument("--include-router", action="store_true")
    args = parser.parse_args()
    run(args.retain_activations, args.include_router)
