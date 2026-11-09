# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compare retained BF16/FP8 Qwen MoE layer schedules on eight B200 GPUs."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-qwen-peer")
image = None
if modal.is_local():
    from modal_validate import base_image

    image = (
        base_image.add_local_file(
            Path(__file__).resolve().parents[1] / "tests" / "test_ready_bf16.py",
            "/probe/test_ready_bf16.py",
        )
        .add_local_dir(Path(__file__).resolve().parents[1] / "src", "/probe/src")
        .add_local_dir(Path(__file__).resolve().parents[3] / "src", "/probe/framework")
        .add_local_file(
            Path(__file__).with_name("benchmark_peer_runtime.py"),
            "/probe/benchmark_peer_runtime.py",
        )
        .add_local_file(
            Path(__file__).with_name("validate_peer_resources.py"),
            "/probe/validate_peer_resources.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests" / "test_swiglu_epilogue.py",
            "/probe/test_swiglu_epilogue.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests" / "test_peer_quantization.py",
            "/probe/test_peer_quantization.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests" / "test_grouped_scatter.py",
            "/probe/test_grouped_scatter.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1]
            / "tests"
            / "test_column_quantization.py",
            "/probe/test_column_quantization.py",
        )
    )


@app.function(image=image, gpu="B200:8", cpu=32, memory=131072, timeout=3600)
def benchmark(
    profile=False,
    sweep_sms=False,
    qwen_scatter=False,
    qwen_columns=False,
    qwen_dispatch=False,
    qwen_swiglu=False,
    qwen_padding=False,
    qwen_bounds=False,
    routing="random",
    qwen_activation=False,
    compute_dtype="fp8",
    qwen_bf16_weights=False,
    qwen_bf16_forward=False,
    timing_batch_size=1,
    qwen_incremental=False,
):
    if qwen_scatter:
        tests = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "/probe/test_grouped_scatter.py",
                "-k",
                "separate_scatter_store",
            ],
            env={**os.environ, "PYTHONPATH": "/probe/src:/probe/framework"},
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        print(tests.stdout, flush=True)
        if tests.returncode:
            return dict(
                returncode=tests.returncode, stdout=tests.stdout, stderr=tests.stderr
            )
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "/probe/benchmark_peer_runtime.py",
        "--qwen",
        "--compute-dtype",
        compute_dtype,
        "--routing",
        routing,
    ]
    command.extend(["--timing-batch-size", str(timing_batch_size)])
    if qwen_incremental:
        command.append("--qwen-incremental")
    if qwen_bf16_forward:
        command.append("--qwen-bf16-forward")
    if qwen_bf16_weights:
        command.append("--qwen-bf16-weights")
    if qwen_activation:
        command.append("--qwen-activation")
    if qwen_bounds:
        command.append("--qwen-bounds")
    if qwen_padding:
        command.append("--qwen-padding")
    if qwen_scatter:
        command.append("--qwen-scatter")
    if qwen_swiglu:
        command.append("--qwen-swiglu")
    if qwen_dispatch:
        command.append("--qwen-dispatch")
    if qwen_columns:
        command.append("--qwen-columns")
    if sweep_sms:
        command.append("--sweep-sms")
    if profile:
        command.append("--profile")
    result = subprocess.run(
        command,
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src:/probe/framework",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=3500,
        check=False,
    )
    return dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=600)
def validate_scatter(test_filter=""):
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "/probe/test_grouped_scatter.py",
            "/probe/test_column_quantization.py",
            "/probe/test_peer_quantization.py",
            "/probe/test_swiglu_epilogue.py",
            "/probe/test_ready_bf16.py",
            *(["-k", test_filter] if test_filter else []),
        ],
        env={**os.environ, "PYTHONPATH": "/probe/src:/probe/framework"},
        capture_output=True,
        text=True,
        timeout=550,
        check=False,
    )
    return dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=1200)
def inspect_resources(compute_dtype="fp8"):
    result = subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            "/probe/validate_peer_resources.py",
            "--dtype",
            compute_dtype,
        ],
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src:/probe/framework",
            "OMP_NUM_THREADS": "1",
            "DG_JIT_PTXAS_VERBOSE": "1",
        },
        capture_output=True,
        text=True,
        timeout=1100,
        check=False,
    )
    return dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


@app.local_entrypoint()
def main(
    profile: bool = False,
    sweep_sms: bool = False,
    qwen_scatter: bool = False,
    tests_only: bool = False,
    test_filter: str = "",
    resources_only: bool = False,
    qwen_columns: bool = False,
    qwen_dispatch: bool = False,
    qwen_swiglu: bool = False,
    qwen_padding: bool = False,
    qwen_bounds: bool = False,
    routing: str = "random",
    qwen_activation: bool = False,
    compute_dtype: str = "fp8",
    qwen_bf16_weights: bool = False,
    qwen_bf16_forward: bool = False,
    timing_batch_size: int = 1,
    qwen_incremental: bool = False,
):
    if test_filter and not tests_only:
        raise ValueError("--test-filter requires --tests-only")
    if resources_only:
        result = inspect_resources.remote(compute_dtype)
        print(json.dumps(result))
        if result["returncode"]:
            raise RuntimeError("Peer resource inspection failed")
        return
    result = (
        validate_scatter.remote(test_filter)
        if tests_only
        else benchmark.remote(
            profile,
            sweep_sms,
            qwen_scatter,
            qwen_columns,
            qwen_dispatch,
            qwen_swiglu,
            qwen_padding,
            qwen_bounds,
            routing,
            qwen_activation,
            compute_dtype,
            qwen_bf16_weights,
            qwen_bf16_forward,
            timing_batch_size,
            qwen_incremental,
        )
    )
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Qwen peer benchmark failed")
