# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Qwen3 MoE layer FP8 native/NCCL training timing on one eight-GPU Hopper node.

The layer geometry defaults to Qwen3-30B-A3B; ``--shapes`` runs several geometries one after another
in the same allocation (one process per shape and routing).
"""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

TIMING_GPU = os.getenv("MONO_EP_TIMING_GPU", "H200")
if TIMING_GPU not in ("H100", "H200"):
    raise ValueError("MONO_EP_TIMING_GPU must be H100 or H200")
app = modal.App(f"axolotl-{TIMING_GPU.lower()}-persistent-fp8-timing")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    source = Path(os.environ.get("MONO_EP_SOURCE_DIR", root / "src"))
    image = base_image.add_local_dir(source, "/probe/src").add_local_file(
        root / "examples/benchmark_persistent_training.py", "/probe/benchmark.py"
    )


def _benchmark(
    profile: bool = False,
    compare_forward_layouts: bool = False,
    compare_kernel_layouts: bool = False,
    fp8_transport: bool = False,
    fp8_combine: bool = False,
    trap_timeouts: bool = False,
    expected_gpu: str = "H200",
    variants: str = "",
    routings: str = "",
    routing_override: str = "",
    bump_versions: bool = False,
    four_layer_variants: str = "",
    shapes: str = "",
    shape_override: str = "",
    time_weight_build: bool = False,
    shape_config: str = "",
    shape_env: str = "",
):
    import torch

    if shapes and not shape_override:
        # One pass per shape (each with its routings), one after another in this allocation.
        actual_gpu = torch.cuda.get_device_name()
        if expected_gpu not in actual_gpu:
            raise RuntimeError(f"Expected {expected_gpu}, allocated {actual_gpu}")
        passes = []
        config = json.loads(shape_config) if shape_config else {}
        for shape in shapes.split(","):
            entry = config.get(_parse_shape(shape)[0], {})
            passes.append(
                dict(
                    shape=shape,
                    **_benchmark(
                        profile,
                        compare_forward_layouts,
                        compare_kernel_layouts,
                        fp8_transport,
                        fp8_combine,
                        trap_timeouts,
                        expected_gpu,
                        entry.get("variants", variants),
                        routings,
                        bump_versions=bump_versions,
                        four_layer_variants=four_layer_variants,
                        shape_override=shape,
                        time_weight_build=time_weight_build,
                        shape_env=entry.get("env", ""),
                    ),
                    shape_variants=entry.get("variants", variants),
                    shape_env=entry.get("env", ""),
                )
            )
        return dict(
            returncode=max(p["returncode"] for p in passes),
            device=actual_gpu,
            modal_task_id=os.environ.get("MODAL_TASK_ID"),
            shapes=passes,
        )

    if routings and not routing_override:
        # One process per routing, one after another in this allocation.
        runs = [
            dict(
                routing=routing,
                **_benchmark(
                    profile,
                    compare_forward_layouts,
                    compare_kernel_layouts,
                    fp8_transport,
                    fp8_combine,
                    trap_timeouts,
                    expected_gpu,
                    variants,
                    routing_override=routing,
                    bump_versions=bump_versions,
                    four_layer_variants=four_layer_variants,
                    shape_override=shape_override,
                    time_weight_build=time_weight_build,
                    shape_env=shape_env,
                ),
            )
            for routing in routings.split(",")
        ]
        return dict(
            returncode=max(r["returncode"] for r in runs),
            device=torch.cuda.get_device_name(),
            modal_task_id=os.environ.get("MODAL_TASK_ID"),
            runs=runs,
        )

    actual_gpu = torch.cuda.get_device_name()
    print(f"device {actual_gpu}", flush=True)
    if expected_gpu not in actual_gpu:
        raise RuntimeError(f"Expected {expected_gpu}, allocated {actual_gpu}")
    if fp8_combine and not fp8_transport:
        raise ValueError("FP8 combine requires FP8 input transport")
    geometry = []
    if shape_override:
        name, tokens, hidden, inner, experts, topk = _parse_shape(shape_override)
        geometry = [
            "--shape-name",
            name,
            "--tokens",
            str(tokens),
            "--hidden",
            str(hidden),
            "--intermediate",
            str(inner),
            "--experts",
            str(experts),
            "--topk",
            str(topk),
        ]
    result = subprocess.run(  # nosec B603 B607
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            "/probe/benchmark.py",
            "--compute-dtype",
            "blockwise",
            "--recompute",
            "--compare-native",
            "--compiled",
            *(["--profile"] if profile else []),
            *(["--compare-forward-layouts"] if compare_forward_layouts else []),
            *(["--compare-kernel-layouts"] if compare_kernel_layouts else []),
            *(["--variants", variants] if variants else []),
            *(["--routing", routing_override] if routing_override else []),
            *(["--bump-versions"] if bump_versions else []),
            *(["--time-weight-build"] if time_weight_build else []),
            *geometry,
            *(
                ["--four-layer-variants", four_layer_variants]
                if four_layer_variants
                else []
            ),
        ],
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src",
            "OMP_NUM_THREADS": "1",
            "MONO_EP_SM90_FP8_TRAP_TIMEOUTS": "1" if trap_timeouts else "0",
            "MONO_EP_SM90_FP8_TRANSPORT": "1" if fp8_transport else "0",
            "MONO_EP_SM90_FP8_COMBINE": "1" if fp8_combine else "0",
            **dict(item.split("=", 1) for item in shape_env.split("+") if item),
        },
        capture_output=True,
        text=True,
        timeout=3000,
        check=False,
    )
    # Echo into the container log so a run is recoverable from `modal app logs` if the runner dies.
    print(
        f"ROUTING_RESULT {shape_override} {routing_override} rc={result.returncode}",
        flush=True,
    )
    print(result.stdout[-200000:], flush=True)
    print(result.stderr[-6000:], flush=True)
    return dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


@app.function(image=image, gpu="H100!:8", cpu=32, memory=262144, timeout=7200)
def benchmark_h100(*args, **kwargs):
    return _benchmark(*args, expected_gpu="H100", **kwargs)


@app.function(image=image, gpu="H200:8", cpu=32, memory=131072, timeout=7200)
def benchmark_h200(*args, **kwargs):
    return _benchmark(*args, expected_gpu="H200", **kwargs)


SHAPES = {
    "qwen3_30b_a3b": "qwen3_30b_a3b:16384x2048x768x128x8",
    "qwen3_235b_a22b": "qwen3_235b_a22b:16384x4096x1536x128x8",
}


def _parse_shape(spec):
    """'name:TxHxIxExK' (tokens per GPU, hidden, intermediate, total experts, top-k) or a SHAPES key."""
    name, _, dims = SHAPES.get(spec, spec).partition(":")
    values = [int(v) for v in dims.split("x")]
    if len(values) != 5:
        raise ValueError(f"shape {spec!r} must be name:TxHxIxExK")
    return (name, *values)


@app.local_entrypoint()
def main(
    gpu: str = "H200",
    profile: bool = False,
    compare_forward_layouts: bool = False,
    compare_kernel_layouts: bool = False,
    fp8_transport: bool = False,
    fp8_combine: bool = False,
    trap_timeouts: bool = False,
    variants: str = "",
    routings: str = "",
    out: str = "",
    bump_versions: bool = False,
    four_layer_variants: str = "",
    shapes: str = "",
    time_weight_build: bool = False,
    shape_config: str = "",
):
    """variants: native env variants 'name:ENV=V+ENV2=V2,...'; routings: e.g. 'balanced,skewed'.

    shapes: comma list of SHAPES keys or 'name:TxHxIxExK'; empty keeps the Qwen3-30B-A3B default.
    shape_config: JSON {shape name: {"variants": ..., "env": "K=V+K2=V2"}}; per-shape variant list and
    process environment (applies to the NCCL control too), overriding --variants for that shape.
    """
    for shape in filter(None, shapes.split(",")):
        _parse_shape(shape)
    if compare_forward_layouts and compare_kernel_layouts:
        raise ValueError(
            "Forward-layout and kernel-layout comparisons are separate runs"
        )
    if fp8_combine and not fp8_transport:
        raise ValueError("--fp8-combine requires --fp8-transport")
    if gpu not in ("H100", "H200"):
        raise ValueError("gpu must be H100 or H200")
    benchmark = benchmark_h100 if gpu == "H100" else benchmark_h200
    result = benchmark.remote(
        profile,
        compare_forward_layouts,
        compare_kernel_layouts,
        fp8_transport,
        fp8_combine,
        trap_timeouts,
        variants=variants,
        routings=routings,
        bump_versions=bump_versions,
        four_layer_variants=four_layer_variants,
        shapes=shapes,
        time_weight_build=time_weight_build,
        shape_config=shape_config,
    )
    result["modal_app_id"] = os.environ.get("MODAL_APP_ID") or app.app_id
    if out:
        Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError(f"{gpu} native FP8 timing failed")
