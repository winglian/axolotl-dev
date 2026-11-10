# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run validate_mono_ep_activations.py on one strict H100."""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("mono-ep-activations-elementwise")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(
        root / "vendor/deepgemm/deep_gemm/include", "/probe/include"
    ).add_local_file(
        root / "examples/validate_mono_ep_activations.py", "/probe/activations.py"
    )


@app.function(image=image, gpu="H100!", cpu=8, memory=32768, timeout=1800)
def check():
    import torch

    name = torch.cuda.get_device_name()
    if "H100" not in name:
        return dict(returncode=2, device=name, stdout="", stderr="wrong device")
    result = subprocess.run(  # nosec B603 B607
        ["python", "/probe/activations.py", "--include-dir", "/probe/include"],
        capture_output=True,
        text=True,
        timeout=1700,
        check=False,
        env={**os.environ, "MAX_JOBS": "8"},
    )
    payload = None
    for line in result.stdout.splitlines():
        if line.startswith("ACT_JSON "):
            payload = json.loads(line[len("ACT_JSON ") :])
    return dict(
        returncode=result.returncode,
        device=name,
        result=payload,
        stderr=result.stderr[-6000:],
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
    )


@app.local_entrypoint()
def main(out: str):
    result = check.remote()
    result["modal_app_id"] = os.environ.get("MODAL_APP_ID") or app.app_id
    Path(out).write_text(json.dumps(result, indent=1))
    summary = {
        k: (v["passed"], v["dx"]["rel_norm"], v["dx"]["max_abs_over_absmax"])
        for k, v in ((result.get("result") or {}).get("results") or {}).items()
    }
    print(
        json.dumps(
            dict(
                returncode=result["returncode"],
                device=result["device"],
                modal_app_id=result["modal_app_id"],
                passed=(result.get("result") or {}).get("passed"),
                summary=summary,
            ),
            indent=1,
        )
    )
    if result["returncode"]:
        print(result["stderr"])
