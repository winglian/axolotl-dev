# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate the MonoEP backward scheduler on Hopper."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-backward-schedule")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(root / "src", "/probe/src").add_local_file(
        root / "tests/test_backward_schedule.py",
        "/probe/test_backward_schedule.py",
    )


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=1800)
def validate():
    result = subprocess.run(
        ["python", "-m", "pytest", "-q", "/probe/test_backward_schedule.py"],
        env={**os.environ, "PYTHONPATH": "/probe/src"},
        capture_output=True,
        text=True,
        timeout=1500,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.local_entrypoint()
def main():
    result = validate.remote()
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Hopper backward scheduler validation failed")
