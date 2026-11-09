# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate routed FP8 peer training on a single B200 node."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-peer-layer")
image = None
if modal.is_local():
    from modal_validate import base_image

    image = (
        base_image.add_local_dir(
            Path(__file__).resolve().parents[1] / "src", "/probe/src"
        )
        .add_local_file(
            Path(__file__).with_name("validate_peer_layer.py"),
            "/probe/validate_peer_layer.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests" / "test_activation.py",
            "/probe/test_activation.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests" / "test_peer_quantization.py",
            "/probe/test_peer_quantization.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests" / "test_packing.py",
            "/probe/test_packing.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests" / "test_grouped_scatter.py",
            "/probe/test_grouped_scatter.py",
        )
        .add_local_file(
            Path(__file__).with_name("validate_peer_autograd.py"),
            "/probe/validate_peer_autograd.py",
        )
    )


def _validate(
    compiled: bool = False,
    backward: bool = False,
    training: bool = False,
    autograd: bool = False,
    experts: int = 1,
    slots: bool = False,
    gpu_barrier: bool = False,
    world_size: int = 2,
    skip_padding: bool = False,
):
    environment = {**os.environ, "PYTHONPATH": "/probe/src", "OMP_NUM_THREADS": "1"}
    activation = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "/probe/test_activation.py",
            "/probe/test_peer_quantization.py",
            "/probe/test_packing.py",
            "/probe/test_grouped_scatter.py",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if activation.returncode:
        return dict(
            returncode=activation.returncode,
            stdout=activation.stdout,
            stderr=activation.stderr,
        )
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "/probe/validate_peer_autograd.py"
        if autograd
        else "/probe/validate_peer_layer.py",
    ]
    if skip_padding:
        if not autograd:
            raise ValueError("--skip-padding requires --autograd")
        command.append("--skip-padding")
    if gpu_barrier:
        if not autograd:
            raise ValueError("--gpu-barrier requires --autograd")
        command.append("--gpu-barrier")
    if slots:
        if not autograd:
            raise ValueError("--slots requires --autograd")
        command.append("--slots")
    if autograd:
        command.extend(["--experts", str(experts)])
    if compiled:
        command.append("--compiled")
    if backward and not autograd:
        command.append("--backward")
    if training and not autograd:
        command.append("--training")
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1100,
        check=False,
    )
    return dict(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        activation_tests=activation.stdout,
    )


@app.function(image=image, gpu="B200:2", cpu=8, memory=32768, timeout=1200)
def validate(
    compiled=False,
    backward=False,
    training=False,
    autograd=False,
    experts=1,
    slots=False,
    gpu_barrier=False,
    skip_padding=False,
):
    return _validate(
        compiled,
        backward,
        training,
        autograd,
        experts,
        slots,
        gpu_barrier,
        skip_padding=skip_padding,
    )


@app.function(image=image, gpu="B200:4", cpu=16, memory=65536, timeout=1200)
def validate_four(
    compiled=False, experts=1, slots=False, gpu_barrier=False, skip_padding=False
):
    return _validate(
        compiled=compiled,
        autograd=True,
        experts=experts,
        slots=slots,
        gpu_barrier=gpu_barrier,
        world_size=4,
        skip_padding=skip_padding,
    )


@app.function(image=image, gpu="B200:8", cpu=16, memory=65536, timeout=1200)
def validate_eight(
    compiled=False, experts=1, slots=False, gpu_barrier=False, skip_padding=False
):
    return _validate(
        compiled=compiled,
        autograd=True,
        experts=experts,
        slots=slots,
        gpu_barrier=gpu_barrier,
        world_size=8,
        skip_padding=skip_padding,
    )


@app.local_entrypoint()
def main(
    compiled: bool = False,
    backward: bool = False,
    training: bool = False,
    autograd: bool = False,
    experts: int = 1,
    slots: bool = False,
    gpu_barrier: bool = False,
    world_size: int = 2,
    skip_padding: bool = False,
):
    if world_size in (4, 8):
        if not autograd:
            raise ValueError("Four/eight-rank validation requires --autograd")
        runner = validate_four if world_size == 4 else validate_eight
        result = runner.remote(compiled, experts, slots, gpu_barrier, skip_padding)
    elif world_size == 2:
        result = validate.remote(
            compiled,
            backward,
            training,
            autograd,
            experts,
            slots,
            gpu_barrier,
            skip_padding,
        )
    else:
        raise ValueError("Supported world sizes are 2, 4 and 8")
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Peer layer validation failed")
