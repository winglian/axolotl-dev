# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run the phase-trace / staging A/B probe on a strict H100."""

import json
import os
from pathlib import Path

import modal

app = modal.App("axolotl-diag-staging")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(root / "src", "/probe/src").add_local_file(
        root / "examples/probe_diag_staging.py", "/probe/diag.py"
    )


@app.function(image=image, gpu="H100!", cpu=16, memory=65536, timeout=3000)
def probe(args):
    import subprocess  # nosec B404

    import torch

    name = torch.cuda.get_device_name()
    if "H100" not in name:
        return dict(returncode=2, stderr=f"wrong device {name}", stdout="")
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        "/probe/diag.py",
        *args,
    ]
    result = subprocess.run(  # nosec B603
        command,
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src",
            "OMP_NUM_THREADS": "1",
            "MONO_EP_SM90_FP8_TRAP_TIMEOUTS": "1",
            "MONO_EP_SM90_FP8_TRANSPORT": "1",
            "MONO_EP_SM90_FP8_COMBINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=2900,
        check=False,
    )
    return dict(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr[-20000:],
        command=command,
        device=name,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
    )


@app.local_entrypoint()
def main(out: str, args: str = ""):
    result = probe.remote(args.split())
    payload = None
    for line in result["stdout"].splitlines():
        if line.startswith("DIAG_JSON "):
            payload = json.loads(line[len("DIAG_JSON ") :])
    result["diag"] = payload
    result["modal_app_id"] = os.environ.get("MODAL_APP_ID") or app.app_id
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("returncode", "device", "modal_app_id")}))
    if result["returncode"] or payload is None:
        print(result["stderr"][-5000:])
        raise RuntimeError("diag probe failed")
