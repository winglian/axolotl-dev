# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate compact multi-expert dispatch on two B200s."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-packed-peer")
image = None
if modal.is_local():
    from modal_validate import base_image

    image = (
        base_image.add_local_dir(
            Path(__file__).resolve().parents[1] / "src", "/probe/src"
        )
        .add_local_file(
            Path(__file__).with_name("validate_packed_peer.py"),
            "/probe/validate_packed_peer.py",
        )
        .add_local_file(
            Path(__file__).with_name("benchmark_packed_dispatch.py"),
            "/probe/benchmark_packed_dispatch.py",
        )
        .add_local_file(
            Path(__file__).with_name("benchmark_peer_runtime.py"),
            "/probe/benchmark_peer_runtime.py",
        )
    )


@app.function(image=image, gpu="B200:2", cpu=8, memory=32768, timeout=1200)
def validate(
    compiled: bool = False,
    benchmark: bool = False,
    tokens: int = 137,
    hidden: int = 256,
    layer: bool = False,
    profile: bool = False,
    compare_order: bool = False,
    compare_layout: bool = False,
    compare_acquire: bool = False,
    compare_scatter: bool = False,
    compare_routing: bool = False,
    compare_sms: bool = False,
    compare_backward_scatter: bool = False,
    compare_slots: bool = False,
    compare_slot_dtype: bool = False,
    compare_reset: bool = False,
    compare_boundary: bool = False,
    compare_epoch: bool = False,
):
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "/probe/benchmark_peer_runtime.py"
        if layer
        else "/probe/benchmark_packed_dispatch.py"
        if benchmark
        else "/probe/validate_packed_peer.py",
    ]
    if compare_epoch:
        if not layer:
            raise ValueError("--compare-epoch requires --layer")
        command.append("--compare-epoch")
    if compare_boundary:
        if not layer:
            raise ValueError("--compare-boundary requires --layer")
        command.append("--compare-boundary")
    if compare_reset:
        if not layer:
            raise ValueError("--compare-reset requires --layer")
        command.append("--compare-reset")
    if compare_slot_dtype:
        if not layer:
            raise ValueError("--compare-slot-dtype requires --layer")
        command.append("--compare-slot-dtype")
    if compare_slots:
        if not layer:
            raise ValueError("--compare-slots requires --layer")
        command.append("--compare-slots")
    if compare_backward_scatter:
        if not layer:
            raise ValueError("--compare-backward-scatter requires --layer")
        command.append("--compare-backward-scatter")
    if compare_sms:
        if not layer:
            raise ValueError("--compare-sms requires --layer")
        command.append("--compare-sms")
    if compare_routing:
        if not layer:
            raise ValueError("--compare-routing requires --layer")
        command.append("--compare-routing")
    if compare_scatter:
        if not layer:
            raise ValueError("--compare-scatter requires --layer")
        command.append("--compare-scatter")
    if compare_acquire:
        if not layer:
            raise ValueError("--compare-acquire requires --layer")
        command.append("--compare-acquire")
    if compare_layout:
        if not layer:
            raise ValueError("--compare-layout requires --layer")
        command.append("--compare-layout")
    if compare_order:
        if not layer:
            raise ValueError("--compare-order requires --layer")
        command.append("--compare-order")
    if profile:
        if not layer:
            raise ValueError("--profile requires --layer")
        command.append("--profile")
    if not benchmark and not layer:
        command.extend(["--tokens", str(tokens), "--hidden", str(hidden)])
    if compiled and not benchmark and not layer:
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


@app.local_entrypoint()
def main(
    compiled: bool = False,
    benchmark: bool = False,
    tokens: int = 137,
    hidden: int = 256,
    layer: bool = False,
    profile: bool = False,
    compare_order: bool = False,
    compare_layout: bool = False,
    compare_acquire: bool = False,
    compare_scatter: bool = False,
    compare_routing: bool = False,
    compare_sms: bool = False,
    compare_backward_scatter: bool = False,
    compare_slots: bool = False,
    compare_slot_dtype: bool = False,
    compare_reset: bool = False,
    compare_boundary: bool = False,
    compare_epoch: bool = False,
):
    result = validate.remote(
        compiled,
        benchmark,
        tokens,
        hidden,
        layer,
        profile,
        compare_order,
        compare_layout,
        compare_acquire,
        compare_scatter,
        compare_routing,
        compare_sms,
        compare_backward_scatter,
        compare_slots,
        compare_slot_dtype,
        compare_reset,
        compare_boundary,
        compare_epoch,
    )
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Packed peer validation failed")
