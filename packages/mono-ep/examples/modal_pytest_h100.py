# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run package tests on strict H100s: modal run modal_pytest_h100.py --gpus 1 --tests '...'."""

import json
import os
import shlex
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("mono-ep-pytest-h100")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_dir(root / "tests", "/probe/tests")
        .add_local_dir(root / "examples", "/probe/examples")
    )


def _run(tests, script=False):
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not all("H100" in n for n in names):
        return dict(returncode=2, devices=names, stdout="", stderr="wrong devices")
    result = subprocess.run(  # nosec B603 B607
        ["python", *tests]
        if script
        else ["python", "-m", "pytest", "-v", "-p", "no:cacheprovider", *tests],
        cwd="/probe",
        env={**os.environ, "PYTHONPATH": "/probe/src"},
        capture_output=True,
        text=True,
        timeout=3000,
        check=False,
    )
    return dict(
        returncode=result.returncode,
        devices=names,
        stdout=result.stdout[-20000:],
        stderr=result.stderr[-6000:],
    )


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, timeout=3600)
def pytest1(tests, script=False):
    return _run(tests, script)


@app.function(image=image, gpu="H100!:2", cpu=16, memory=131072, timeout=3600)
def pytest2(tests, script=False):
    return _run(tests, script)


@app.local_entrypoint()
def main(tests: str, gpus: int = 1, out: str = "", script: bool = False):
    """--script runs ``python <tests...>`` (a probe script) instead of pytest."""
    result = (pytest2 if gpus == 2 else pytest1).remote(shlex.split(tests), script)
    result["modal_app_id"] = os.environ.get("MODAL_APP_ID") or app.app_id
    if out:
        Path(out).write_text(json.dumps(result, indent=1))
    print(result["stdout"][-8000:])
    print(result["stderr"][-3000:])
    print(
        json.dumps(
            dict(
                returncode=result["returncode"],
                devices=result["devices"],
                modal_app_id=result["modal_app_id"],
            )
        )
    )
    if result["returncode"]:
        raise RuntimeError("pytest failed")
