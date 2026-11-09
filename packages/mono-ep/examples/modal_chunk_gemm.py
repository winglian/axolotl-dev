# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Build and validate the native DeepGEMM readiness-queue extension."""

import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-chunk-gemm")
image = None
if modal.is_local():
    from modal_validate import base_image

    image = (
        base_image.add_local_dir(
            Path(__file__).resolve().parents[1] / "src", "/probe/src"
        )
        .add_local_file(
            Path(__file__).with_name("validate_chunk_gemm.py"),
            "/probe/validate_chunk_gemm.py",
        )
        .add_local_file(
            Path(__file__).with_name("validate_chunk_remote.py"),
            "/probe/validate_chunk_remote.py",
        )
        .add_local_file(
            Path(__file__).with_name("validate_chunk_fp8.py"),
            "/probe/validate_chunk_fp8.py",
        )
        .add_local_file(
            Path(__file__).with_name("validate_chunk_fp8_remote.py"),
            "/probe/validate_chunk_fp8_remote.py",
        )
    )


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=1200)
def validate(compiled: bool = False, fp8: bool = False):
    command = [
        "python",
        "/probe/validate_chunk_fp8.py" if fp8 else "/probe/validate_chunk_gemm.py",
    ]
    if compiled:
        command.append("--compiled")
    result = subprocess.run(
        command,
        env={**os.environ, "PYTHONPATH": "/probe/src"},
        capture_output=True,
        text=True,
        timeout=1100,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.local_entrypoint()
def main(
    compiled: bool = False,
    remote: bool = False,
    fused: bool = False,
    fp8: bool = False,
    quantized_producer: bool = False,
    routed_producer: bool = False,
):
    import json

    report = (
        validate_remote.remote(
            fused, compiled, fp8, quantized_producer, routed_producer
        )
        if remote
        else validate.remote(compiled, fp8)
    )
    print(json.dumps(report))
    if report["returncode"]:
        raise RuntimeError("Chunk GEMM validation failed; see captured report")


@app.function(image=image, gpu="B200:2", cpu=8, memory=32768, timeout=1200)
def validate_remote(
    fused: bool = False,
    compiled: bool = False,
    fp8: bool = False,
    quantized_producer: bool = False,
    routed_producer: bool = False,
):
    options = (["--fused"] if fused and not fp8 else []) + (
        ["--compiled"] if compiled else []
    )
    if routed_producer:
        if not fp8:
            raise ValueError("Routed producer requires FP8")
        options.append("--routed-producer")
    if quantized_producer:
        if not fp8:
            raise ValueError("Quantized producer requires FP8")
        options.append("--quantized-producer")
    result = subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "/probe/validate_chunk_fp8_remote.py"
            if fp8
            else "/probe/validate_chunk_remote.py",
        ]
        + options,
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src",
            "OMP_NUM_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/mono-ep-inductor",
        },
        capture_output=True,
        text=True,
        timeout=1100,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
