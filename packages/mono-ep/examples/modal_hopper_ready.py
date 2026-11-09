# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate native Hopper readiness admission."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-readiness")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_file(root / "tests/test_ready_bf16.py", "/probe/test_ready_bf16.py")
        .add_local_file(
            root / "tests/test_ready_fp8_hopper.py", "/probe/test_ready_fp8_hopper.py"
        )
    )


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=1200)
def validate():
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "/probe/test_ready_bf16.py",
            "/probe/test_ready_fp8_hopper.py",
            "-k",
            "test_ready_grouped_bf16 or test_ready_bf16_fused_forward or test_ready_grouped_fp8_hopper",
        ],
        env={**os.environ, "PYTHONPATH": "/probe/src"},
        capture_output=True,
        text=True,
        timeout=1100,
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
        raise RuntimeError("Hopper readiness validation failed")
