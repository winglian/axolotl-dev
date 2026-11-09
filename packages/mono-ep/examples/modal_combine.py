# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate peer scatter-add combine and GPU completion on two B200s."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-peer-combine")
image = None
if modal.is_local():
    from modal_validate import base_image

    image = (
        base_image.add_local_dir(
            Path(__file__).resolve().parents[1] / "src", "/probe/src"
        )
        .add_local_file(
            Path(__file__).with_name("validate_peer_combine.py"),
            "/probe/validate_peer_combine.py",
        )
        .add_local_file(
            Path(__file__).with_name("benchmark_peer_combine.py"),
            "/probe/benchmark_peer_combine.py",
        )
    )


@app.function(image=image, gpu="B200:2", cpu=8, memory=32768, timeout=1200)
def validate(compiled: bool = False, benchmark: bool = False):
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "/probe/benchmark_peer_combine.py"
        if benchmark
        else "/probe/validate_peer_combine.py",
    ]
    if compiled and not benchmark:
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
def main(compiled: bool = False, benchmark: bool = False):
    result = validate.remote(compiled, benchmark)
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Peer combine validation failed")
