# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate the opt-in persistent backward kernel on two H100s."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-persistent-backward")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(
            root / "vendor/deepgemm/deep_gemm/include/deep_gemm",
            "/probe/native/deep_gemm",
        )
        .add_local_file(
            root / "tests/native/hopper_persistent_backward_probe.cu",
            "/probe/hopper_persistent_backward_probe.cu",
        )
        .add_local_file(
            root / "examples/validate_hopper_persistent_backward.py",
            "/probe/validate.py",
        )
    )


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=900)
def validate():
    result = subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "/probe/validate.py",
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MAX_JOBS": "4"},
        capture_output=True,
        text=True,
        timeout=800,
        check=False,
    )
    return dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


@app.local_entrypoint()
def main():
    result = validate.remote()
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Hopper persistent backward validation failed")
