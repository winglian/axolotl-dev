# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate persistent activation policies and training on two to eight B200s."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-persistent-retention")
image = None
if modal.is_local():
    from modal_validate import base_image

    image = (
        base_image.add_local_dir(
            Path(__file__).resolve().parents[1] / "src", "/probe/src"
        )
        .add_local_file(
            Path(__file__).with_name("validate_persistent_retention.py"),
            "/probe/validate.py",
        )
        .add_local_file(
            Path(__file__).with_name("validate_persistent_training.py"),
            "/probe/training.py",
        )
        .add_local_file(
            Path(__file__).with_name("validate_peer_packing.py"),
            "/probe/peer_packing.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests/test_retained_packing.py",
            "/probe/test_retained_packing.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests/test_owned_backward.py",
            "/probe/test_owned_backward.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests/test_backward_schedule.py",
            "/probe/test_backward_schedule.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests/test_peer_scatter.py",
            "/probe/test_peer_scatter.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests/test_peer_packing.py",
            "/probe/test_peer_packing.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests/test_recompute_epilogue.py",
            "/probe/test_recompute_epilogue.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1] / "tests/test_fp8_swiglu_epilogue.py",
            "/probe/test_fp8_swiglu_epilogue.py",
        )
        .add_local_file(
            Path(__file__).resolve().parents[1]
            / "tests/test_transposed_quantization.py",
            "/probe/test_transposed_quantization.py",
        )
    )


def _validate(
    training=False,
    compute_dtype="bf16",
    checkpointed=False,
    recompute=False,
    compiled=False,
    peer_backward=False,
    owned_blocks=False,
    native_backward=False,
    master_dtype="float32",
    world_size=2,
):
    packing = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "/probe/test_retained_packing.py",
            "/probe/test_peer_scatter.py",
            "/probe/test_backward_schedule.py",
            "/probe/test_owned_backward.py",
            "/probe/test_peer_packing.py",
            "/probe/test_recompute_epilogue.py",
            "/probe/test_fp8_swiglu_epilogue.py",
            "/probe/test_transposed_quantization.py",
        ],
        env={**os.environ, "PYTHONPATH": "/probe/src"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if packing.returncode:
        return dict(
            returncode=packing.returncode, stdout=packing.stdout, stderr=packing.stderr
        )
    peer = subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "/probe/peer_packing.py",
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": "/probe/src"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if peer.returncode:
        return dict(
            returncode=peer.returncode,
            stdout=peer.stdout,
            stderr=peer.stderr,
            packing_stdout=packing.stdout,
        )
    result = subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            "/probe/training.py" if training else "/probe/validate.py",
            "--compute-dtype",
            compute_dtype,
            *(["--checkpointed"] if checkpointed else []),
            *(["--recompute"] if recompute else []),
            *(["--compiled"] if compiled else []),
            *(["--peer-backward"] if peer_backward else []),
            *(["--owned-blocks"] if owned_blocks else []),
            *(["--native-backward"] if native_backward else []),
            *(["--master-dtype", master_dtype] if training else []),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": "/probe/src"},
        capture_output=True,
        text=True,
        timeout=1100,
        check=False,
    )
    return dict(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        packing_stdout=packing.stdout,
        peer_stdout=peer.stdout,
    )


_validators = {
    size: app.function(
        image=image,
        gpu=f"B200:{size}",
        cpu=max(16, size * 4),
        memory=65536,
        timeout=1800,
        name=f"validate_{size}",
    )(_validate)
    for size in (2, 4, 8)
}


@app.local_entrypoint()
def main(
    training: bool = False,
    compute_dtype: str = "bf16",
    checkpointed: bool = False,
    recompute: bool = False,
    compiled: bool = False,
    peer_backward: bool = False,
    owned_blocks: bool = False,
    native_backward: bool = False,
    master_dtype: str = "float32",
    world_size: int = 2,
):
    if world_size not in (2, 4, 8) or (not training and world_size != 2):
        raise ValueError("Training world_size must be 2, 4 or 8; retention requires 2")
    if master_dtype not in ("float32", "bf16") or (
        not training and master_dtype != "float32"
    ):
        raise ValueError("Master dtype selection requires training: float32 or bf16")
    if compute_dtype not in ("bf16", "mxfp8"):
        raise ValueError("Unknown compute dtype")
    if checkpointed and not training:
        raise ValueError("Checkpoint validation requires --training")
    if (recompute or compiled) and not training:
        raise ValueError("Recompute/compile validation requires training")
    if peer_backward and (not training or not recompute):
        raise ValueError("Peer backward validation requires recomputing training")
    if native_backward and (
        not peer_backward or compute_dtype != "bf16" or owned_blocks
    ):
        raise ValueError("Native backward requires isolated BF16 peer validation")
    if owned_blocks and (not peer_backward or compute_dtype != "bf16"):
        raise ValueError("Owned blocks require BF16 peer training")
    result = _validators[world_size].remote(
        training,
        compute_dtype,
        checkpointed,
        recompute,
        compiled,
        peer_backward,
        owned_blocks,
        native_backward,
        master_dtype,
        world_size,
    )
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Persistent retention validation failed")
