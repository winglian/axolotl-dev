# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Build and validate the SM90 persistent BF16 fused-MoE prototype."""

import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-persistent-bf16")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_file(
        root / "examples/validate_hopper_persistent.py",
        "/probe/validate_hopper_persistent.py",
    )


def _run(world, tokens, experts=8, topk=4, skewed=False):
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        "/probe/validate_hopper_persistent.py",
        "--tokens",
        str(tokens),
        "--experts",
        str(experts),
        "--topk",
        str(topk),
    ]
    if skewed:
        command.append("--skewed")
    result = subprocess.run(  # nosec B603
        command,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=1500,
        check=False,
    )
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    result.check_returncode()
    return result.stdout


def _run_diagnostic():
    command = [
        "/usr/local/cuda/bin/compute-sanitizer",
        "--tool",
        "memcheck",
        "--target-processes",
        "all",
        "--show-backtrace",
        "yes",
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        "/probe/validate_hopper_persistent.py",
        "--tokens",
        "137",
    ]
    result = subprocess.run(  # nosec B603
        command,
        env={
            **os.environ,
            "CUDA_LAUNCH_BLOCKING": "1",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=1500,
        check=False,
    )
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    for cubin in Path("/root/.dj/tmp").glob("*/kernel.cubin"):
        sass = subprocess.run(  # nosec B603
            ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(cubin)],
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"===== SASS {cubin} =====", flush=True)
        print(sass.stdout, flush=True)
        print(sass.stderr, flush=True)
    return {"returncode": result.returncode}


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=1800)
def validate_single():
    return _run(1, 137)


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=1800)
def validate_pair():
    return _run(2, 64)


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=1800)
def validate_ring_wrap():
    return _run(2, 1900, experts=128, topk=8, skewed=True)


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=1800)
def diagnose_single():
    return _run_diagnostic()


@app.local_entrypoint()
def main(pair: bool = False, diagnostic: bool = False, ring_wrap: bool = False):
    if diagnostic:
        print(diagnose_single.remote())
    elif ring_wrap:
        print(validate_ring_wrap.remote())
    else:
        print(validate_pair.remote() if pair else validate_single.remote())
