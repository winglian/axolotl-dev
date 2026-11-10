# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run validate_fsdp2_sm90_weight_cache.py on two strict H100s and save the JSON locally."""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("mono-ep-fsdp2-sm90-weight-cache")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(root / "src", "/probe/src").add_local_file(
        root / "examples/validate_fsdp2_sm90_weight_cache.py", "/probe/validate.py"
    )


@app.function(image=image, gpu="H100!:2", cpu=16, memory=65536, timeout=2400)
def validate(extra):
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != 2 or not all("H100" in n for n in names):
        return dict(returncode=2, devices=names)
    result = subprocess.run(  # nosec B603 B607
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "/probe/validate.py",
            "--out",
            "/tmp/fsdp2.json",  # nosec B108
            *extra,
        ],
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src",
            "OMP_NUM_THREADS": "1",
            "MONO_EP_SM90_FP8_TRANSPORT": "1",
            "MONO_EP_SM90_FP8_COMBINE": "1",
            "NCCL_MAX_CTAS": "16",
        },
        capture_output=True,
        text=True,
        timeout=2200,
        check=False,
    )
    payload = (
        json.loads(Path("/tmp/fsdp2.json").read_text())  # nosec B108
        if Path("/tmp/fsdp2.json").exists()  # nosec B108
        else None
    )
    return dict(
        returncode=result.returncode,
        devices=names,
        result=payload,
        stdout=result.stdout[-4000:],
        stderr=result.stderr[-8000:],
    )


@app.local_entrypoint()
def main(out: str, extra_args: str = ""):
    result = validate.remote(extra_args.split())
    result["modal_app_id"] = os.environ.get("MODAL_APP_ID") or app.app_id
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            dict(
                returncode=result["returncode"],
                devices=result["devices"],
                modal_app_id=result["modal_app_id"],
                passed=(result.get("result") or {}).get("passed"),
            )
        )
    )
    if result["returncode"]:
        print(result.get("stderr", "")[-6000:])
        raise RuntimeError("FSDP2 weight-cache validation failed")
