# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate all queued MXFP8 expert derivatives on B200."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-scheduled-backward")
image = None
if modal.is_local():
    from modal_validate import base_image

    image = base_image.add_local_dir(
        Path(__file__).resolve().parents[1] / "src", "/probe/src"
    ).add_local_file(
        Path(__file__).with_name("validate_peer_layer.py"),
        "/probe/validate_peer_layer.py",
    )

    image = image.add_local_file(
        Path(__file__).with_name("validate_scheduled_backward.py"),
        "/probe/validate_scheduled_backward.py",
    ).add_local_file(
        Path(__file__).with_name("validate_chunk_gemm.py"),
        "/probe/validate_chunk_gemm.py",
    )


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=1200)
def validate(compiled: bool = False):
    command = [
        "python",
        "/probe/validate_scheduled_backward.py",
    ]
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


@app.local_entrypoint()
def main(compiled: bool = False):
    result = validate.remote(compiled)
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Peer layer validation failed")
