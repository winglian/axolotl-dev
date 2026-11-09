# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Check retained persistent-forward projections against independent GEMMs."""

import argparse
import json
import os

import deep_gemm
import torch
import torch.distributed as dist


def quantized_reference(x):
    blocks = x.float().unflatten(-1, (-1, 32))
    scale = torch.exp2(
        torch.ceil(
            torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)
        )
    )
    return (
        (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    ).flatten(-2)


def run(compute_dtype="bf16"):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    torch.manual_seed(81)
    hidden, inner, experts, topk = 512, 512, 8, 4
    w1 = (
        torch.randn(experts, 2 * inner, hidden, device="cuda", dtype=torch.bfloat16)
        / hidden**0.5
    )
    w2 = (
        torch.randn(experts, hidden, inner, device="cuda", dtype=torch.bfloat16)
        / inner**0.5
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    local = experts // world
    tw1, tw2 = deep_gemm.transform_weights_for_mono_ep(
        w1[rank * local : (rank + 1) * local], w2[rank * local : (rank + 1) * local]
    )
    if compute_dtype == "mxfp8":
        from mono_ep.token_quantization import prepare_persistent_mxfp8

        tw1 = prepare_persistent_mxfp8(
            w1[rank * local : (rank + 1) * local], weights=True, interleave=True
        )
        tw2 = prepare_persistent_mxfp8(
            w2[rank * local : (rank + 1) * local], weights=True
        )
    buffer = deep_gemm.get_symm_buffer_for_mono_ep(
        dist.group.WORLD,
        experts,
        769,
        topk,
        hidden,
        inner,
        mma_type="bf16xbf16" if compute_dtype == "bf16" else "fp8xfp8",
    )
    retained = []
    for step, tokens in enumerate((137, 769, 137)):
        torch.manual_seed(900 + step)
        inputs = torch.randn(world, tokens, hidden, device="cuda", dtype=torch.bfloat16)
        ids = (
            torch.rand(world, tokens, experts, device="cuda")
            .argsort(-1)[..., :topk]
            .contiguous()
        )
        scores = torch.rand(world, tokens, topk, device="cuda")
        if compute_dtype == "mxfp8":
            qx, sx = prepare_persistent_mxfp8(inputs[rank])
            buffer.x[:tokens].copy_(qx)
            buffer.x_sf[:tokens].copy_(sx)
            reference_x = quantized_reference(inputs)
            reference_w1 = quantized_reference(w1)
        else:
            buffer.x[:tokens].copy_(inputs[rank])
            reference_x, reference_w1 = inputs, w1
        buffer.topk_idx[:tokens].copy_(ids[rank])
        buffer.topk_weights[:tokens].copy_(scores[rank])
        saved = torch.full(
            (world, buffer.num_max_tokens_per_rank, topk, 2 * inner),
            float("nan"),
            device="cuda",
            dtype=torch.bfloat16,
        )
        y = torch.empty_like(inputs[rank])
        forward = (
            deep_gemm.bf16_mono_ep
            if compute_dtype == "bf16"
            else deep_gemm.fp8_fp4_mono_ep
        )
        forward(y, tw1, tw2, buffer, fast_math=False, saved_z=saved)
        torch.cuda.synchronize()
        expected_y = torch.zeros_like(y, dtype=torch.float32)
        errors = []
        for expert in range(experts):
            source, token, slot = torch.where(ids == expert)
            z = (reference_x[source, token] @ reference_w1[expert].T).bfloat16()
            if expert // local == rank:
                actual = saved[source, token, slot]
                assert torch.isfinite(actual).all()
                error = (
                    actual.float() - z.float()
                ).norm() / z.float().norm().clamp_min(1e-8)
                assert error < 0.001, (expert, error.item())
                errors.append(error.item())
            owned = source == rank
            gate, up = z[owned].float().chunk(2, -1)
            h = (
                torch.nn.functional.silu(gate)
                * up
                * scores[source[owned], token[owned], slot[owned], None]
            ).bfloat16()
            expected_y.index_add_(0, token[owned], (h @ w2[expert].T).float())
        error_y = (y.float() - expected_y).norm() / expected_y.norm().clamp_min(1e-8)
        assert error_y < (0.003 if compute_dtype == "bf16" else 0.12), error_y.item()
        mask = (ids // local) != rank
        assert torch.isnan(saved[:, :tokens][mask]).all()
        assert torch.isnan(saved[:, tokens:]).all()
        for old, snapshot in retained:
            torch.testing.assert_close(old, snapshot, equal_nan=True, atol=0, rtol=0)
        retained.append((saved, saved.clone()))
        print(
            json.dumps(
                dict(
                    rank=rank,
                    compute_dtype=compute_dtype,
                    projection_reference="independent_e4m3_per32_power_of_two"
                    if compute_dtype == "mxfp8"
                    else "bf16",
                    step=step,
                    tokens=tokens,
                    projection_relative_l2=errors,
                    output_relative_l2=error_y.item(),
                    previous_retention_unchanged=True,
                )
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-dtype", choices=("bf16", "mxfp8"), default="bf16")
    run(parser.parse_args().compute_dtype)
