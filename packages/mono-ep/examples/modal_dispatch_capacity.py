# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Measure padding amplification in bidirectional B200 peer dispatch."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-dispatch-capacity")
image = None
if modal.is_local():
    from modal_validate import base_image

    image = base_image.add_local_dir(
        Path(__file__).resolve().parents[1] / "src", "/probe/src"
    ).add_local_file(
        Path(__file__).with_name("benchmark_dispatch_capacity.py"),
        "/probe/benchmark_dispatch_capacity.py",
    )


@app.function(image=image, gpu="B200:2", cpu=8, memory=32768, timeout=600)
def benchmark():
    result = subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "/probe/benchmark_dispatch_capacity.py",
        ],
        env={**os.environ, "PYTHONPATH": "/probe/src", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=540,
        check=False,
    )
    return dict(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


@app.local_entrypoint()
def main():
    result = benchmark.remote()
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Dispatch capacity benchmark failed")
