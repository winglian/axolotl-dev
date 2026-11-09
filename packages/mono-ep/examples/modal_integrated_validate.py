# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""2x strict-H100 validation of the integrated SM90 FP8 kernels with an explicit env set.

suite=fp32ref: validate_hopper_fp8_persistent_backward.py (independent FP32 reference).
suite=fp8combine: validate_hopper_fp8_combine.py (independently decoded FP8 combine reference).
suite=forward: validate_hopper_fp8_persistent_forward.py (FP32 reference, original-W1 exactness).
"""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-integrated-validate")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_dir(
            root / "vendor/deepgemm/deep_gemm/include/deep_gemm",
            "/probe/native/deep_gemm",
        )
        .add_local_file(
            root / "tests/native/hopper_fp8_persistent_backward_probe.cu",
            "/probe/hopper_fp8_persistent_backward_probe.cu",
        )
        .add_local_file(
            root / "examples/validate_hopper_fp8_persistent_backward.py",
            "/probe/fp32ref.py",
        )
        .add_local_file(
            root / "examples/validate_hopper_fp8_combine.py", "/probe/fp8combine.py"
        )
        .add_local_file(
            root / "examples/validate_hopper_fp8_persistent_forward.py",
            "/probe/forward.py",
        )
    )


@app.function(image=image, gpu="H100!:2", cpu=16, memory=65536, timeout=1800)
def validate(suite, env):
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not all("H100" in n for n in names):
        return dict(returncode=2, stdout="", stderr=f"wrong devices {names}")
    result = subprocess.run(  # nosec B603 B607
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            f"/probe/{suite}.py",
        ],
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MAX_JOBS": "4",
            "PYTHONPATH": "/probe/src",
            **env,
        },
        capture_output=True,
        text=True,
        timeout=1700,
        check=False,
    )
    return dict(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr[-20000:],
        devices=names,
    )


@app.local_entrypoint()
def main(out: str, suite: str, env: str = ""):
    """env: "KEY=V+KEY2=V2" passed to the validator processes."""
    if suite not in ("fp32ref", "fp8combine", "forward"):
        raise ValueError("suite must be fp32ref, fp8combine or forward")
    values = dict(item.split("=", 1) for item in env.split("+") if item)
    result = validate.remote(suite, values)
    result.update(
        suite=suite,
        env=values,
        modal_app_id=os.environ.get("MODAL_APP_ID") or app.app_id,
        records=[
            json.loads(line)
            for line in result["stdout"].splitlines()
            if line.startswith("{")
        ],
    )
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("returncode", "devices", "modal_app_id")}))
    if result["returncode"]:
        print(result["stderr"][-5000:])
        raise RuntimeError(f"{suite} validation failed")
