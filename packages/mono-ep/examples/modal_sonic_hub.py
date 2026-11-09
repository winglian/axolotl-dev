# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate the standard Hub-loaded Sonic adapter with Axolotl compatibility setup."""

import json
import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("axolotl-mono-ep-hub")
image = (
    modal.Image.from_registry("axolotlai/axolotl-uv:main-latest")
    .add_local_dir(ROOT / "src", "/probe/src")
    .add_local_dir(ROOT / "../../src", "/probe/framework")
    .add_local_file(ROOT / "tests/test_sonic_hub.py", "/probe/test_sonic_hub.py")
)


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=1200)
def validate():
    result = subprocess.run(
        ["python", "-m", "pytest", "-q", "/probe/test_sonic_hub.py"],
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src:/probe/framework",
            "OMP_NUM_THREADS": "1",
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
def main():
    result = validate.remote()
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Hub adapter validation failed")
