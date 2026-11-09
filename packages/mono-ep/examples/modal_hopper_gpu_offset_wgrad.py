# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run SM90 GPU-offset grouped BF16 weight-gradient parity."""

import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-gpu-offset-wgrad")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_file(
        root / "examples/validate_hopper_gpu_offset_wgrad.py", "/probe/validate.py"
    )


@app.function(image=image, gpu="H100", timeout=1200)
def validate():
    subprocess.run(["python", "/probe/validate.py"], check=True)


@app.local_entrypoint()
def main():
    validate.remote()
