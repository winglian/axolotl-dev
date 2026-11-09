# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Independent-reference validation of the SM90 FP8 backward dscore reduction on two strict H100s."""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-dscore-validate")
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
            "/probe/validate.py",
        )
    )


@app.function(image=image, gpu="H100!:2", cpu=16, memory=65536, timeout=900)
def validate(
    bound=False,
    cached=False,
    cache_dz=False,
    compact=False,
    trap_timeouts=False,
    role_registers="0",
):
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
            "/probe/validate.py",
        ],
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MAX_JOBS": "4",
            "MONO_EP_NATIVE_BINDING": "1" if bound else "0",
            "MONO_EP_CACHE_ACTIVATIONS": "1" if cached else "0",
            "MONO_EP_TRAP_TIMEOUTS": "1" if trap_timeouts else "0",
            "MONO_EP_COMPACT_BACKWARD": "1" if compact else "0",
            "MONO_EP_CACHE_DZ": "1" if cache_dz else "0",
            "PYTHONPATH": "/probe/src",
            "MONO_EP_SM90_FP8_ROLE_REGISTERS": role_registers,
        },
        capture_output=True,
        text=True,
        timeout=800,
        check=False,
    )
    return dict(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        devices=names,
        role_registers=role_registers,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
    )


@app.local_entrypoint()
def main(
    bound: bool = False,
    cached: bool = False,
    cache_dz: bool = False,
    compact: bool = False,
    trap_timeouts: bool = False,
    role_registers: str = "0",
):
    if trap_timeouts and bound:
        raise ValueError("Trap-only validation requires the raw probe")
    if compact and bound:
        raise ValueError("Compact validation requires the raw probe")
    if cache_dz and not cached:
        raise ValueError("dZ cache requires --cached")
    result = validate.remote(
        bound, cached, cache_dz, compact, trap_timeouts, role_registers
    )
    result["modal_app_id"] = os.environ.get("MODAL_APP_ID") or app.app_id
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Hopper persistent backward validation failed")
