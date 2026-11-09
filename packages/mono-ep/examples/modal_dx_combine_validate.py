# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""FP8 combine validation (independently decoded BF16 reference) with dx-combine/dscore opt-ins on strict H100s."""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-dx-combine-validate")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(root / "src", "/probe/src").add_local_file(
        root / "examples/validate_hopper_fp8_combine.py", "/probe/validate.py"
    )


@app.function(image=image, gpu="H100!:2", cpu=16, memory=65536, timeout=1500)
def validate(ring_wrap=False, topk=8, dx_combine=0, dscore_warp=True):
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
            "PYTHONPATH": "/probe/src",
            "OMP_NUM_THREADS": "1",
            "NCCL_MAX_CTAS": "4",
            "MONO_EP_TEST_RING_WRAP": "1" if ring_wrap else "0",
            "MONO_EP_TEST_TOPK": str(topk),
            "MONO_EP_SM90_FP8_DX_COMBINE": str(dx_combine),
            "MONO_EP_SM90_FP8_DSCORE_WARP_REDUCE": "1" if dscore_warp else "0",
        },
        capture_output=True,
        text=True,
        timeout=1380,
        check=False,
    )
    return dict(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr[-20000:],
        devices=names,
        dx_combine=dx_combine,
        dscore_warp=dscore_warp,
    )


@app.local_entrypoint()
def main(
    out: str,
    ring_wrap: bool = False,
    topk: int = 8,
    dx_combine: int = 0,
    dscore_warp: bool = True,
):
    if topk not in (1, 8) or (ring_wrap and topk != 8):
        raise ValueError("Use top-k 1 or 8; forced ring wrap requires top-k 8")
    result = validate.remote(ring_wrap, topk, dx_combine, dscore_warp)
    result["modal_app_id"] = app.app_id
    result["records"] = [
        json.loads(line)
        for line in result["stdout"].splitlines()
        if line.startswith("{")
    ]
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("returncode", "devices", "modal_app_id")}))
    if result["returncode"]:
        raise RuntimeError("Hopper FP8 combine validation failed")
