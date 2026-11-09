# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate the SM90 persistent BF16 mono-ep prototype."""

import argparse
import json
import os

import deep_gemm
import torch
import torch.distributed as dist


def run(tokens, experts=8, topk=4, skewed=False):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    torch.manual_seed(190 + rank)
    hidden, inner = 512, 512
    max_tokens = max(256, tokens)
    w1 = (
        torch.randn(experts, 2 * inner, hidden, device="cuda", dtype=torch.bfloat16)
        / hidden**0.5
    )
    w2 = (
        torch.randn(experts, hidden, inner, device="cuda", dtype=torch.bfloat16)
        / inner**0.5
    )
    dist.broadcast(w1, 0)
    dist.broadcast(w2, 0)
    local = experts // world
    tw1, tw2 = deep_gemm.transform_weights_for_mono_ep(
        w1[rank * local : (rank + 1) * local],
        w2[rank * local : (rank + 1) * local],
    )
    buffer = deep_gemm.get_symm_buffer_for_mono_ep(
        dist.group.WORLD,
        experts,
        max_tokens,
        topk,
        hidden,
        inner,
        mma_type="bf16xbf16",
    )
    steps = (False,) if skewed else (False, True, False)
    for step, masked in enumerate(steps):
        torch.manual_seed(410 + step)
        inputs = torch.randn(world, tokens, hidden, device="cuda", dtype=torch.bfloat16)
        ids = (
            torch.rand(world, tokens, experts, device="cuda")
            .argsort(-1)[..., :topk]
            .contiguous()
        )
        if skewed:
            ids.copy_(
                torch.arange(
                    experts // 2,
                    experts // 2 + topk,
                    device="cuda",
                    dtype=torch.int64,
                ).expand_as(ids)
            )
            active_local_experts = experts // world
            seed_tokens = active_local_experts // topk
            ids[:, :seed_tokens].copy_(
                torch.arange(
                    experts // 2,
                    experts,
                    device="cuda",
                    dtype=torch.int64,
                ).view(1, seed_tokens, topk)
            )
        scores = torch.rand(world, tokens, topk, device="cuda")
        if masked:
            ids[torch.rand(world, tokens, topk, device="cuda") < 0.25] = -1
        buffer.x[:tokens].copy_(inputs[rank])
        buffer.topk_idx[:tokens].copy_(ids[rank])
        buffer.topk_weights[:tokens].copy_(scores[rank])
        saved = torch.full(
            (world, buffer.num_max_tokens_per_rank, topk, 2 * inner),
            float("nan"),
            device="cuda",
            dtype=torch.bfloat16,
        )
        y = torch.empty_like(inputs[rank])
        deep_gemm.bf16_mono_ep(y, tw1, tw2, buffer, fast_math=False, saved_z=saved)
        torch.cuda.synchronize()

        expected = torch.zeros_like(y, dtype=torch.float32)
        projection_errors = []
        for expert in range(experts):
            source, token, slot = torch.where(ids == expert)
            z = (inputs[source, token] @ w1[expert].T).bfloat16()
            if expert // local == rank:
                actual = saved[source, token, slot]
                torch.testing.assert_close(actual, z, rtol=1e-3, atol=1e-3)
                if actual.numel():
                    projection_errors.append(
                        ((actual.float() - z.float()).norm() / z.float().norm()).item()
                    )
            owned = source == rank
            gate, up = z[owned].float().chunk(2, -1)
            h = (
                torch.nn.functional.silu(gate)
                * up
                * scores[source[owned], token[owned], slot[owned], None]
            ).bfloat16()
            expected.index_add_(0, token[owned], (h @ w2[expert].T).float())
        output_error = (y.float() - expected).norm() / expected.norm().clamp_min(1e-8)
        assert output_error < 0.003, output_error.item()
        assert torch.isnan(saved[:, :tokens][ids == -1]).all()
        ring_blocks = buffer.l1_acts.size(0) // 128
        pool_blocks = sum(
            (int((ids == expert).sum().item()) + 127) // 128
            for expert in range(experts)
        )
        if skewed:
            assert pool_blocks > ring_blocks, (pool_blocks, ring_blocks)
        result = {
            "rank": rank,
            "world": world,
            "step": step,
            "masked_routes": masked,
            "tokens": tokens,
            "skewed_to_rank": 1 if skewed else None,
            "ring_blocks": ring_blocks,
            "pool_blocks": pool_blocks,
            "projection_relative_l2": projection_errors,
            "output_relative_l2": output_error.item(),
        }
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--skewed", action="store_true")
    args = parser.parse_args()
    run(args.tokens, args.experts, args.topk, args.skewed)
