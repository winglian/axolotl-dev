# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate compiled checkpointed native BF16 training on two H100s."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-native-training")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(root / "src", "/probe/src").add_local_file(
        root / "examples/validate_persistent_training.py", "/probe/training.py"
    )


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=1800)
def validate():
    records = {}
    for master_dtype in ("float32", "bf16"):
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
                "--native-backward",
                "--compiled",
                "--checkpointed",
                "--master-dtype",
                master_dtype,
            ],
            env={**os.environ, "PYTHONPATH": "/probe/src", "OMP_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=840,
            check=False,
        )
        records[master_dtype] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode:
            break
    return records


@app.local_entrypoint()
def main():
    result = validate.remote()
    print(json.dumps(result))
    if any(record["returncode"] for record in result.values()):
        raise RuntimeError("Hopper native training validation failed")
