# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate packed BF16 peer composition on two B200s."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-bf16-peer")
image = None
if modal.is_local():
    from modal_validate import base_image

    image = (
        base_image.add_local_file(
            Path(__file__).with_name("validate_peer_autograd.py"),
            "/probe/validate_peer_autograd.py",
        )
        .add_local_dir(Path(__file__).resolve().parents[1] / "src", "/probe/src")
        .add_local_file(
            Path(__file__).with_name("validate_bf16_peer.py"),
            "/probe/validate_bf16_peer.py",
        )
    )


def _validate(
    compiled=False,
    skip_padding=False,
    runtime=False,
    atomic=False,
    world_size=2,
    compute_lanes=0,
    incremental_ready=False,
    dtype="bf16",
    fused_forward=False,
):
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "/probe/validate_bf16_peer.py",
    ]
    if runtime:
        command[-1] = "/probe/validate_peer_autograd.py"
        command.extend(["--dtype", dtype, "--experts", "2", "--gpu-barrier"])
        if not atomic:
            command.append("--slots")
    if fused_forward:
        command.append("--fused-forward")
    if incremental_ready:
        command.append("--incremental-ready")
    if compute_lanes:
        command.extend(["--compute-lanes", str(compute_lanes)])
    if skip_padding:
        command.append("--skip-padding")
    if compiled:
        command.append("--compiled")
    result = subprocess.run(
        command,
        env={**os.environ, "PYTHONPATH": "/probe/src", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=1100,
        check=False,
    )
    return dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


@app.function(image=image, gpu="B200:2", cpu=16, memory=65536, timeout=1200)
def validate(
    compiled=False,
    skip_padding=False,
    runtime=False,
    atomic=False,
    compute_lanes=0,
    incremental_ready=False,
    dtype="bf16",
    fused_forward=False,
):
    return _validate(
        compiled,
        skip_padding,
        runtime,
        atomic,
        compute_lanes=compute_lanes,
        incremental_ready=incremental_ready,
        dtype=dtype,
        fused_forward=fused_forward,
    )


@app.function(image=image, gpu="B200:4", cpu=24, memory=98304, timeout=1200)
def validate_four(
    compiled=False,
    skip_padding=False,
    atomic=False,
    compute_lanes=0,
    incremental_ready=False,
    dtype="bf16",
    fused_forward=False,
):
    return _validate(
        compiled,
        skip_padding,
        True,
        atomic,
        world_size=4,
        compute_lanes=compute_lanes,
        incremental_ready=incremental_ready,
        dtype=dtype,
        fused_forward=fused_forward,
    )


@app.local_entrypoint()
def main(
    compiled: bool = False,
    skip_padding: bool = False,
    runtime: bool = False,
    atomic: bool = False,
    world_size: int = 2,
    compute_lanes: int = 0,
    incremental_ready: bool = False,
    dtype: str = "bf16",
    fused_forward: bool = False,
):
    if (dtype != "bf16" or fused_forward) and not runtime:
        raise ValueError("--dtype and --fused-forward require --runtime")
    if incremental_ready and not runtime:
        raise ValueError("--incremental-ready requires --runtime")
    if compute_lanes and not runtime:
        raise ValueError("--compute-lanes requires --runtime")
    if atomic and not runtime:
        raise ValueError("--atomic requires --runtime")
    if world_size == 4 and runtime:
        result = validate_four.remote(
            compiled,
            skip_padding,
            atomic,
            compute_lanes,
            incremental_ready,
            dtype,
            fused_forward,
        )
    elif world_size == 2:
        result = validate.remote(
            compiled,
            skip_padding,
            runtime,
            atomic,
            compute_lanes,
            incremental_ready,
            dtype,
            fused_forward,
        )
    else:
        raise ValueError("Use two GPUs, or four with --runtime")
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("BF16 peer composition failed")
