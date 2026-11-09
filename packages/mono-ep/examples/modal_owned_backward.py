# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate owned-block SM100 backward independently of transport regressions."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-owned-backward")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(root / "src", "/probe/src").add_local_file(
        root / "tests/test_owned_backward.py", "/probe/test_owned_backward.py"
    )


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=600)
def validate():
    result = subprocess.run(
        ["python", "-m", "pytest", "-x", "-vv", "-s", "/probe/test_owned_backward.py"],
        env={**os.environ, "PYTHONPATH": "/probe/src", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=540,
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
        raise RuntimeError("Owned-block backward validation failed")
