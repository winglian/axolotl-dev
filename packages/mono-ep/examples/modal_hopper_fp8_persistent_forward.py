# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate the opt-in persistent FP8 forward kernel on two H100s."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-fp8-persistent-forward")
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
            root / "tests/native/hopper_fp8_persistent_forward_probe.cu",
            "/probe/hopper_fp8_persistent_forward_probe.cu",
        )
        .add_local_file(
            root / "examples/validate_hopper_fp8_persistent_forward.py",
            "/probe/validate.py",
        )
    )


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=900)
def validate(bound=False, compact=False, trap_timeouts=False, original_w1_layout=False):
    if bound and compact:
        raise ValueError("compact forward is available only through the raw probe")
    if bound and trap_timeouts:
        raise ValueError(
            "trap-only timeout handling is available only through the raw probe"
        )
    if original_w1_layout and (not bound or compact or trap_timeouts):
        raise ValueError("original W1 parity requires the bound default kernel")
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
            "MONO_EP_COMPACT_FORWARD": "1" if compact else "0",
            "MONO_EP_TRAP_TIMEOUTS": "1" if trap_timeouts else "0",
            "MONO_EP_ORIGINAL_W1_LAYOUT": "1" if original_w1_layout else "0",
            "PYTHONPATH": "/probe/src",
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
        selection=dict(
            binding="bound" if bound else "raw_probe",
            layout="compact384" if compact else "baseline512",
            timeout_handler=(
                "trap_only_no_diagnostic_calls" if trap_timeouts else "diagnostic_calls"
            ),
            original_w1_layout=original_w1_layout,
        ),
    )


@app.local_entrypoint()
def main(
    bound: bool = False,
    compact: bool = False,
    trap_timeouts: bool = False,
    original_w1_layout: bool = False,
):
    if bound and trap_timeouts:
        raise ValueError("--bound and --trap-timeouts cannot be combined")
    if original_w1_layout and (not bound or compact or trap_timeouts):
        raise ValueError(
            "--original-w1-layout requires --bound without compact or trap timeouts"
        )
    result = validate.remote(bound, compact, trap_timeouts, original_w1_layout)
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Hopper persistent FP8 forward validation failed")
