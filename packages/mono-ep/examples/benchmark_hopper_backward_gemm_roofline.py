# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Measure the three grouped FP8 GEMMs in the Hopper backward math path."""

import argparse
import json
import statistics
from pathlib import Path

import torch


def _measure(operation, warmup, iterations):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _spotcheck(name, qa, sa, qb, sb, output, starts, counts):
    errors = []
    for expert, (start, count) in enumerate(zip(starts, counts, strict=True)):
        if not count:
            continue
        scale_b = sb[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
        decoded_b = qb[expert].float() * scale_b[: qb.shape[1], : qb.shape[2]]
        for row in {start, start + count - 1}:
            decoded_a = qa[row].float() * sa[row].repeat_interleave(128)[: qa.shape[1]]
            reference = torch.matmul(decoded_a, decoded_b.T).bfloat16().float()
            actual = output[row].float()
            error = (actual - reference).norm() / reference.norm().clamp_min(1e-8)
            if (
                not torch.isfinite(actual).all()
                or not torch.isfinite(reference).all()
                or not torch.isfinite(error)
                or error >= 0.01
            ):
                raise AssertionError((name, expert, row, error.item()))
            errors.append(error.item())
    return max(errors, default=0.0)


def benchmark(
    *,
    rows=131072,
    experts=16,
    hidden=2048,
    inner=768,
    master_dtype="float32",
    warmup=10,
    iterations=50,
):
    import deep_gemm
    from mono_ep.quantization import quantize_fp8
    from mono_ep.weight_quantization import quantize_transposed_weights_128x128

    if rows <= 0 or experts <= 0 or rows % experts:
        raise ValueError("rows must be positive and evenly divisible by experts")
    if hidden <= 0 or inner <= 0 or hidden % 128 or inner % 128:
        raise ValueError("hidden and inner must be positive multiples of 128")
    if rows // experts % 128:
        raise ValueError("rows per expert must be divisible by 128")
    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be positive")

    torch.manual_seed(863)
    device = torch.device("cuda")
    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    dz = torch.randn(rows, 2 * inner, device=device, dtype=torch.bfloat16)
    weight_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(master_dtype)
    if weight_dtype is None:
        raise ValueError("master_dtype must be bfloat16 or float32")
    w1 = torch.randn(experts, 2 * inner, hidden, device=device, dtype=weight_dtype)
    w2 = torch.randn(experts, hidden, inner, device=device, dtype=weight_dtype)

    # Packing, transposes, quantization, scale preparation, and output allocation
    # intentionally finish before any timed interval.
    qx, sx = quantize_fp8(x, block_rows=1, block_cols=128)
    qdy, sdy = quantize_fp8(dy, block_rows=1, block_cols=128)
    qdz, sdz = quantize_fp8(dz, block_rows=1, block_cols=128)
    qw1, sw1 = quantize_fp8(w1, block_rows=128, block_cols=128)
    qw2t, sw2t = quantize_transposed_weights_128x128(w2)
    qw1t, sw1t = quantize_transposed_weights_128x128(w1)
    sx = deep_gemm.transform_sf_into_required_layout(sx, rows, hidden, (1, 128))
    sdy = deep_gemm.transform_sf_into_required_layout(sdy, rows, hidden, (1, 128))
    sdz = deep_gemm.transform_sf_into_required_layout(sdz, rows, 2 * inner, (1, 128))

    ends = torch.arange(
        rows // experts,
        rows + 1,
        rows // experts,
        device=device,
        dtype=torch.int32,
    )
    outputs = {
        "dh": torch.empty((rows, inner), device=device, dtype=torch.bfloat16),
        "z_recompute": torch.empty(
            (rows, 2 * inner), device=device, dtype=torch.bfloat16
        ),
        "dx": torch.empty((rows, hidden), device=device, dtype=torch.bfloat16),
    }

    def grouped(a, sa, b, sb, out):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a, sa),
            (b, sb),
            out,
            ends,
            recipe_a=(1, 128),
            recipe_b=(128, 128),
            use_psum_layout=True,
        )

    operations = {
        "dh": lambda: grouped(qdy, sdy, qw2t, sw2t, outputs["dh"]),
        "z_recompute": lambda: grouped(qx, sx, qw1, sw1, outputs["z_recompute"]),
        "dx": lambda: grouped(qdz, sdz, qw1t, sw1t, outputs["dx"]),
    }
    torch.cuda.synchronize()

    flop_counts = {
        "dh": 2 * rows * hidden * inner,
        "z_recompute": 2 * rows * hidden * (2 * inner),
        "dx": 2 * rows * (2 * inner) * hidden,
    }
    records = {}
    for name, operation in operations.items():
        samples = _measure(operation, warmup, iterations)
        median_ms = statistics.median(samples)
        records[name] = {
            "samples_ms": samples,
            "median_ms": median_ms,
            "mean_ms": statistics.fmean(samples),
            "flops": flop_counts[name],
            "tflops_at_median": flop_counts[name] / (median_ms * 1.0e9),
        }

    starts = [i * (rows // experts) for i in range(experts)]
    counts = [rows // experts] * experts
    records["dh"]["max_spotcheck_relative_l2"] = _spotcheck(
        "dh", qdy, sdy, qw2t, sw2t, outputs["dh"], starts, counts
    )
    records["z_recompute"]["max_spotcheck_relative_l2"] = _spotcheck(
        "z_recompute", qx, sx, qw1, sw1, outputs["z_recompute"], starts, counts
    )
    records["dx"]["max_spotcheck_relative_l2"] = _spotcheck(
        "dx", qdz, sdz, qw1t, sw1t, outputs["dx"], starts, counts
    )

    total_flops = sum(flop_counts.values())
    summed_median_ms = sum(record["median_ms"] for record in records.values())
    return {
        "scope": "standalone_single_rank_synthetic_balanced_prepacked_math_only",
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "geometry": {
            "packed_rows": rows,
            "local_experts": experts,
            "hidden": hidden,
            "inner": inner,
            "rows_per_expert": rows // experts,
            "activation_scale": [1, 128],
            "weight_scale": [128, 128],
            "master_dtype": master_dtype,
        },
        "matched_reference": {
            "matches_default_geometry": rows == 131072
            and experts == 16
            and hidden == 2048
            and inner == 768,
            "source_tokens_per_rank": 16384,
            "world_size": 8,
            "topk": 8,
            "expected_received_rows_per_rank": 131072,
        },
        "warmup": warmup,
        "iterations": iterations,
        "gemms": records,
        "aggregate_tflops_from_summed_medians": total_flops
        / (summed_median_ms * 1.0e9),
        "summed_median_ms": summed_median_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inner", type=int, default=768)
    parser.add_argument(
        "--master-dtype", choices=("bfloat16", "float32"), default="float32"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(
        rows=args.rows,
        experts=args.experts,
        hidden=args.hidden,
        inner=args.inner,
        master_dtype=args.master_dtype,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
