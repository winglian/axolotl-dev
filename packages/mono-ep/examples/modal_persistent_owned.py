# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Exercise cluster-owned M blocks inside the native persistent forward."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-persistent-owned-blocks")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    native = root / "vendor/deepgemm/deep_gemm/include/deep_gemm"
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_file(
            root / "examples/validate_persistent_training.py", "/probe/training.py"
        )
        .add_local_file(native / "scheduler/mono_ep.cuh", "/probe/native/scheduler.cuh")
        .add_local_file(
            native / "impls/sm100_bf16_mono_ep.cuh", "/probe/native/kernel.cuh"
        )
        .add_local_file(
            native / "epilogue/sm100_moe_backward.cuh", "/probe/native/epilogue.cuh"
        )
    )


@app.function(image=image, gpu="B200:2", cpu=16, memory=65536, timeout=900)
def validate():
    import deep_gemm

    include = Path(deep_gemm.__file__).parent / "include/deep_gemm"
    (include / "scheduler/mono_ep.cuh").write_text(
        Path("/probe/native/scheduler.cuh").read_text()
    )
    (include / "epilogue/sm100_moe_backward.cuh").write_text(
        Path("/probe/native/epilogue.cuh").read_text()
    )
    kernel = Path("/probe/native/kernel.cuh").read_text()
    assert kernel.count("bool kOwnMBlocks = false") == 1
    (include / "impls/sm100_bf16_mono_ep.cuh").write_text(
        kernel.replace("bool kOwnMBlocks = false", "bool kOwnMBlocks = true")
    )
    result = subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "/probe/training.py",
            "--recompute",
            "--peer-backward",
            "--compiled",
            "--checkpointed",
        ],
        env={**os.environ, "PYTHONPATH": "/probe/src", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=800,
        check=False,
    )
    return dict(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        native_owned_m_blocks=True,
        backward="existing_grouped_peer_control",
    )


@app.local_entrypoint()
def main():
    result = validate.remote()
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Native persistent M-block ownership validation failed")
