# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""FP32-reference validation of SM90 FP8 forward opt-ins on two strict H100s."""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-diag-forward-validate")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_dir(
            root / "vendor/deepgemm/deep_gemm/include/deep_gemm",
            "/probe/native/deep_gemm",
        )
        .add_local_file(
            root / "tests/native/hopper_fp8_persistent_forward_probe.cu",
            "/probe/hopper_fp8_persistent_forward_probe.cu",
        )
        .add_local_file(
            root / "examples/validate_hopper_fp8_persistent_forward.py",
            "/probe/validate.py",
        )
    )


@app.function(image=image, gpu="H100!:2", cpu=16, memory=65536, timeout=1800)
def validate(selectors, original_w1_layout=False):
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not all("H100" in n for n in names):
        return dict(returncode=2, stdout="", stderr=f"wrong devices {names}")
    result = subprocess.run(  # nosec B603 B607
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
            "OMP_NUM_THREADS": "1",
            "MAX_JOBS": "4",
            "MONO_EP_NATIVE_BINDING": "1",
            "MONO_EP_ORIGINAL_W1_LAYOUT": "1" if original_w1_layout else "0",
            "PYTHONPATH": "/probe/src",
            **selectors,
        },
        capture_output=True,
        text=True,
        timeout=1700,
        check=False,
    )
    records = []
    for line in result.stdout.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return dict(
        returncode=result.returncode,
        selectors=selectors,
        original_w1_layout=original_w1_layout,
        devices=names,
        records=records,
        stdout=result.stdout[-20000:],
        stderr=result.stderr[-20000:],
    )


@app.local_entrypoint()
def main(out: str, selectors: str = ""):
    candidate = dict(item.split("=", 1) for item in filter(None, selectors.split("+")))
    runs = []
    for name, chosen in (("baseline", {}), ("candidate", candidate)):
        for original in (False, True):
            result = validate.remote(chosen, original)
            result["name"] = name
            runs.append(result)
            print(
                json.dumps(
                    dict(
                        name=name,
                        original_w1_layout=original,
                        returncode=result["returncode"],
                        devices=result.get("devices"),
                        output_errors=[
                            r.get("output_error")
                            for r in result["records"]
                            if "output_error" in r
                        ],
                    )
                )
            )
    Path(out).write_text(json.dumps(runs, indent=1))
    if any(r["returncode"] for r in runs):
        raise RuntimeError("forward validation failed")
