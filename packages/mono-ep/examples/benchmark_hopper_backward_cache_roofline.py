# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Time grouped backward GEMMs over caches produced by the persistent kernel."""

import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist


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


def benchmark(*, master_dtype="float32", warmup=10, iterations=50):
    import deep_gemm
    from deep_gemm.mega import _interleave_weights
    from mono_ep.quantization import quantize_fp8
    from mono_ep.runtime import PersistentMoE
    from mono_ep.weight_quantization import quantize_transposed_weights_128x128

    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    weight_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(master_dtype)
    if weight_dtype is None:
        raise ValueError("master_dtype must be bfloat16 or float32")

    tokens, experts, topk, hidden, inner = 16384, 16, 8, 2048, 768
    torch.cuda.set_device(torch.cuda.current_device())
    device = torch.device("cuda", torch.cuda.current_device())
    initialized_here = not dist.is_initialized()
    init_file = None
    if initialized_here:
        init_file = tempfile.NamedTemporaryFile(delete=False)
        init_file.close()
        dist.init_process_group(
            "nccl",
            init_method=f"file://{init_file.name}",
            rank=0,
            world_size=1,
            device_id=device,
        )

    previous_transport = os.environ.get("MONO_EP_SM90_FP8_TRANSPORT")
    previous_combine = os.environ.get("MONO_EP_SM90_FP8_COMBINE")
    os.environ["MONO_EP_SM90_FP8_TRANSPORT"] = "1"
    os.environ["MONO_EP_SM90_FP8_COMBINE"] = "0"
    runtime = None
    try:
        physical_sms = torch.cuda.get_device_properties(device).multi_processor_count
        deep_gemm.set_num_sms(physical_sms)
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
        torch.manual_seed(863)
        x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
        dy = torch.randn_like(x)
        indices = (
            torch.rand(tokens, experts, device=device)
            .argsort(-1)[:, :topk]
            .contiguous()
        )
        scores = torch.rand(tokens, topk, device=device).softmax(-1)
        w1 = torch.randn(experts, 2 * inner, hidden, device=device, dtype=weight_dtype)
        w2 = torch.randn(experts, hidden, inner, device=device, dtype=weight_dtype)

        # Produce the same routed row caches and padded expert metadata used by
        # the persistent backward path. This entire call is outside timing.
        backend.backward(dy, x, indices, scores, w1, w2)
        padded_ends = backend.starts + backend.padded_counts
        live_rows = int(padded_ends[-1].item())
        counts = backend.counts.cpu().tolist()
        padded_counts = backend.padded_counts.cpu().tolist()
        qx, qdy = backend.pools[4][:live_rows], backend.pools[5][:live_rows]
        sx, sdy = backend.pools[6][:live_rows], backend.pools[7][:live_rows]
        qdz, sdz = backend.pools[8][:live_rows], backend.pools[9][:live_rows]

        prepared = (
            quantize_fp8(_interleave_weights(w1), block_rows=128, block_cols=128),
            quantize_transposed_weights_128x128(w2),
            quantize_transposed_weights_128x128(w1),
        )
        (qw1, sw1), (qw2t, sw2t), (qw1t, sw1t) = prepared
        sx = deep_gemm.transform_sf_into_required_layout(
            sx, live_rows, hidden, (1, 128), psum_layout=padded_ends
        )
        sdy = deep_gemm.transform_sf_into_required_layout(
            sdy, live_rows, hidden, (1, 128), psum_layout=padded_ends
        )
        sdz = deep_gemm.transform_sf_into_required_layout(
            sdz, live_rows, 2 * inner, (1, 128), psum_layout=padded_ends
        )
        outputs = {
            "dh": torch.empty((live_rows, inner), device=device, dtype=torch.bfloat16),
            "z_recompute": torch.empty(
                (live_rows, 2 * inner), device=device, dtype=torch.bfloat16
            ),
            "dx": torch.empty((live_rows, hidden), device=device, dtype=torch.bfloat16),
        }

        def grouped(a, sa, b, sb, out):
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (a, sa),
                (b, sb),
                out,
                padded_ends,
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
            "dh": 2 * live_rows * hidden * inner,
            "z_recompute": 2 * live_rows * hidden * (2 * inner),
            "dx": 2 * live_rows * (2 * inner) * hidden,
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
        starts = backend.starts.cpu().tolist()
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
            "scope": "standalone_single_rank_actual_persistent_row_cache_math_only",
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "geometry": {
                "source_tokens": tokens,
                "topk": topk,
                "experts": experts,
                "live_padded_rows": live_rows,
                "expert_counts": counts,
                "expert_padded_counts": padded_counts,
                "hidden": hidden,
                "inner": inner,
                "master_dtype": master_dtype,
            },
            "warmup": warmup,
            "iterations": iterations,
            "gemms": records,
            "aggregate_tflops_from_summed_medians": total_flops
            / (summed_median_ms * 1.0e9),
            "summed_median_ms": summed_median_ms,
        }
    finally:
        if runtime is not None:
            runtime.close()
        if previous_transport is None:
            os.environ.pop("MONO_EP_SM90_FP8_TRANSPORT", None)
        else:
            os.environ["MONO_EP_SM90_FP8_TRANSPORT"] = previous_transport
        if previous_combine is None:
            os.environ.pop("MONO_EP_SM90_FP8_COMBINE", None)
        else:
            os.environ["MONO_EP_SM90_FP8_COMBINE"] = previous_combine
        if initialized_here:
            dist.destroy_process_group()
            if init_file is not None:
                Path(init_file.name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master-dtype", choices=("bfloat16", "float32"), default="float32"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(
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
