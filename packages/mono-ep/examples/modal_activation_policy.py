# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate retained experts against Axolotl activation-memory contexts."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-mono-ep-activation-policy")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_dir(root / "../../src", "/probe/framework")
        .add_local_dir(root / "../../tests", "/probe/tests")
        .add_local_file(
            root / "examples/validate_activation_policy.py", "/probe/validate.py"
        )
    )


@app.function(image=image, gpu="B200:2", cpu=8, memory=32768, timeout=1200)
def validate(compile_local: bool = False):
    result = subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "/probe/validate.py",
            *(["--compile-local"] if compile_local else []),
        ],
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


@app.function(image=image, cpu=4, memory=16384, timeout=300)
def config_tests():
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "/probe/tests/integrations/expert_parallel/test_plugin.py",
            "/probe/tests/utils/schemas/validation/test_expert_parallel_validation.py",
            "-k",
            "FP8",
            "-o",
            "addopts=",
        ],
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src:/probe/framework:/probe",
            "OMP_NUM_THREADS": "1",
        },
        cwd="/probe",
        capture_output=True,
        text=True,
        timeout=250,
        check=False,
    )
    return dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


@app.local_entrypoint()
def main(config_only: bool = False, compile_local: bool = False):
    result = config_tests.remote() if config_only else validate.remote(compile_local)
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Activation-policy validation failed")
