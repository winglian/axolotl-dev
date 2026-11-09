# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Isolate delayed Hopper FP8 readiness operands in fresh CUDA processes."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-fp8-debug")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(root / "src", "/probe/src").add_local_file(
        root / "tests/test_ready_fp8_hopper.py", "/probe/test_ready_fp8_hopper.py"
    )


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=1200)
def validate():
    results = {}
    modes = (
        ("immediate", "none", "0", False),
        ("priority", "none", "10000000", True),
        ("delayed", "none", "10000000", False),
        ("all", "all", "10000000", False),
    )
    for name, delayed, sleep, priority in modes:
        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "/probe/test_ready_fp8_hopper.py::test_ready_grouped_fp8_hopper[False-128-counts0]",
            ],
            env={
                **os.environ,
                "PYTHONPATH": "/probe/src",
                "ULTRA_FP8_DELAY": delayed,
                "ULTRA_FP8_SLEEP": sleep,
                **({"ULTRA_FP8_PRODUCER_PRIORITY": "1"} if priority else {}),
            },
            capture_output=True,
            text=True,
            timeout=150,
            check=False,
        )
        results[name] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    return results


@app.local_entrypoint()
def main():
    result = validate.remote()
    print(json.dumps(result))
    if any(item["returncode"] for item in result.values()):
        raise RuntimeError("Hopper FP8 debug validation failed")
