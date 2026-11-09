# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate GPU-offset weight-gradient packing on Hopper."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-wgrad")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_file(
            root / "tests/test_quantization.py", "/probe/test_quantization.py"
        )
        .add_local_file(
            root / "tests/test_wgrad_quantization.py",
            "/probe/test_wgrad_quantization.py",
        )
        .add_local_file(
            root / "tests/test_hopper_wgrad.py", "/probe/test_hopper_wgrad.py"
        )
        .add_local_file(
            root / "tests/test_paired_weight_quantization.py",
            "/probe/test_paired_weight_quantization.py",
        )
    )


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=900)
def validate(
    gemm=False,
    activation_quantization=False,
    weight_quantization=False,
    paired_weights=False,
):
    if paired_weights and (gemm or activation_quantization or weight_quantization):
        raise ValueError("Paired-weight validation is a separate selection")
    if weight_quantization and (gemm or activation_quantization):
        raise ValueError("Weight quantization validation is a separate selection")
    result = subprocess.run(
        ["python", "-m", "pytest", "-q"]
        + (
            ["/probe/test_paired_weight_quantization.py"]
            if paired_weights
            else [
                "/probe/test_quantization.py",
                "-k",
                "quantize_transposed_weights_128x128",
            ]
            if weight_quantization
            else [
                "/probe/test_quantization.py",
                "-k",
                "activation_cache or tokens_into",
            ]
            if activation_quantization
            else ["/probe/test_wgrad_quantization.py"]
        )
        + (["/probe/test_hopper_wgrad.py"] if gemm else []),
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=800,
        check=False,
    )
    return dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


@app.local_entrypoint()
def main(
    gemm: bool = False,
    activation_quantization: bool = False,
    weight_quantization: bool = False,
    paired_weights: bool = False,
):
    if paired_weights and (gemm or activation_quantization or weight_quantization):
        raise ValueError("Paired-weight validation is a separate selection")
    if weight_quantization and (gemm or activation_quantization):
        raise ValueError("Weight quantization validation is a separate selection")
    result = validate.remote(
        gemm, activation_quantization, weight_quantization, paired_weights
    )
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Hopper weight-gradient validation failed")
