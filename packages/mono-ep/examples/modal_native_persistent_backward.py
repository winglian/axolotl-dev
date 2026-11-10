# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate the native persistent backward kernel on two B200s."""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-native-persistent-backward")
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
            root / "tests/native/persistent_backward_probe.cu",
            "/probe/persistent_backward_probe.cu",
        )
        .add_local_file(
            root / "examples/validate_native_persistent_backward.py",
            "/probe/validate.py",
        )
        .add_local_file(
            root / "examples/validate_native_forward_pull.py",
            "/probe/validate_forward_pull.py",
        )
        .add_local_dir(root / "src", "/probe/src")
    )


@app.function(image=image, gpu="B200:2", cpu=16, memory=65536, timeout=1800)
def validate(stages="6", options="11", pulls="", forward_pull=False):
    import torch

    runs = []
    pull_values = pulls.split(",") if pulls else ["4096"] * len(stages.split(","))
    for value, option, pull in zip(
        stages.split(","), options.split(","), pull_values, strict=True
    ):
        result = subprocess.run(  # nosec B603 B607
            [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "/probe/validate.py",
            ],
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MAX_JOBS": "4",
                "MONO_EP_SM100_BACKWARD_STAGES": value,
                "MONO_EP_SM100_BACKWARD_OPTIONS": option,
                "MONO_EP_SM100_BACKWARD_PULL_BYTES": pull,
            },
            capture_output=True,
            text=True,
            timeout=800,
            check=False,
        )
        runs.append(
            dict(
                stages=int(value),
                options=int(option),
                pull=int(pull),
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr[-20000:],
            )
        )
    if forward_pull:
        result = subprocess.run(  # nosec B603 B607
            [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "/probe/validate_forward_pull.py",
            ],
            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": "/probe/src"},
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        runs.append(
            dict(
                check="forward_pull_variants",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr[-20000:],
            )
        )
    return dict(
        returncode=max(r["returncode"] for r in runs),
        device=torch.cuda.get_device_name(),
        runs=runs,
    )


@app.local_entrypoint()
def main(
    stages: str = "6",
    options: str = "11",
    pulls: str = "",
    forward_pull: bool = False,
    out: str = "",
):
    result = validate.remote(stages, options, pulls, forward_pull)
    result["modal_app_id"] = app.app_id
    if out:
        Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Native persistent backward validation failed")
