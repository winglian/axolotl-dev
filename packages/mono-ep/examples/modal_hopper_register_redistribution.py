# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compile and disassemble frozen SM90 full-CTA register experiments."""

import json
import os
from pathlib import Path

import modal

app = modal.App("mono-ep-hopper-register-redistribution")
image = None
if modal.is_local():
    from modal_validate import base_image

    vendor = Path(os.environ["MONO_EP_VENDOR_DIR"])
    image = base_image.add_local_dir(
        vendor / "deepgemm/deep_gemm/include", "/probe/native", copy=True
    ).add_local_file(
        Path(__file__).with_name("probe_hopper_register_redistribution.py"),
        "/probe/probe_hopper_register_redistribution.py",
        copy=True,
    )


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=3000)
def run():
    import sys

    sys.path.insert(0, "/probe")
    from probe_hopper_register_redistribution import probe

    return probe("/probe/native", "/tmp/register-probe")


@app.local_entrypoint()
def main():
    result = run.remote()
    print(json.dumps(result, indent=2))
    if any(
        r["compile_returncode"] or r["disassembly_error"] for r in result["variants"]
    ):
        raise RuntimeError(
            "Register compiler probe failed; inspect variant diagnostics"
        )
