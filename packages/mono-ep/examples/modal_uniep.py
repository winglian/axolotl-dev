# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Isolated upstream fused-EP smoke test; does not alter the Sonic runtime."""

import os
import subprocess

import modal

REVISION = "7908e4ea9010bb2238f9f05b28c904fd4b71c60a"
app = modal.App("axolotl-uniep-probe")
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3")
    .apt_install("git", "ninja-build", "libnuma-dev")
    .run_commands(
        "git clone https://github.com/ByteDance-Seed/Triton-distributed.git /opt/triton-dist",
        f"git -C /opt/triton-dist checkout {REVISION}",
        "python -m pip install 'setuptools>=77' wheel 'cmake>=3.20,<4' ninja 'pybind11>=2.13.1' packaging",
        "cd /opt/triton-dist && MAX_JOBS=8 TRITON_BUILD_LITTLE_KERNEL=OFF python -m pip install --no-build-isolation -v './python[tests]'",
    )
)


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=1200)
def probe():
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "/opt/triton-dist/python/triton_dist/test/nvidia/test_ep_moe_fused.py",
        "--ntokens",
        "1024",
        "--hidden_dim",
        "256",
        "--ffn_dim",
        "256",
        "--num_experts",
        "8",
        "--topk",
        "2",
        "--warmup",
        "1",
        "--iters",
        "2",
    ]
    result = subprocess.run(
        command,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=1100,
        check=False,
    )
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    return {
        "revision": REVISION,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(probe.remote()))
