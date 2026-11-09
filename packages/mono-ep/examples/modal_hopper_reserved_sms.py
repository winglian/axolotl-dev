# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run reserved-SM persistent F/B with concurrent NCCL collectives."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-reserved-sms")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(root / "src", "/probe/src").add_local_file(
        root / "examples/validate_hopper_reserved_sms.py", "/probe/validate.py"
    )


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=1500)
def validate():
    result = subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "/probe/validate.py",
        ],
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src",
            "OMP_NUM_THREADS": "1",
            "NCCL_MAX_CTAS": "4",
        },
        capture_output=True,
        text=True,
        timeout=1380,
        check=False,
    )
    return dict(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        selection=dict(
            reserved_sms=16,
            nccl_max_ctas=4,
            concurrent_collectives=3,
            iterations=20,
        ),
    )


@app.local_entrypoint()
def main():
    result = validate.remote()
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Hopper reserved-SM validation failed")
