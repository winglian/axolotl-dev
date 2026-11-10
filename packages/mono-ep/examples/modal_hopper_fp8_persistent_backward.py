# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate the opt-in persistent backward kernel on two H100s."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-fp8-persistent-backward")
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


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=900)
def validate(
    bound=False, cached=False, cache_dz=False, compact=False, trap_timeouts=False
):
    result = subprocess.run(
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
        },
        capture_output=True,
        text=True,
        timeout=800,
        check=False,
    )
    return dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


@app.local_entrypoint()
def main(
    bound: bool = False,
    cached: bool = False,
    cache_dz: bool = False,
    compact: bool = False,
    trap_timeouts: bool = False,
):
    if trap_timeouts and bound:
        raise ValueError("Trap-only validation requires the raw probe")
    if compact and bound:
        raise ValueError("Compact validation requires the raw probe")
    if cache_dz and not cached:
        raise ValueError("dZ cache requires --cached")
    result = validate.remote(bound, cached, cache_dz, compact, trap_timeouts)
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Hopper persistent backward validation failed")
