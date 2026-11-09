# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Same-allocation single-rank A/B of env-selected variants of this tree on one strict H100.

All variants run in one probe process and alternate within every round (warmups per variant per
round, then timed iterations), so every sample shares the GPU allocation. Backward runs
probe_dispatch_warps_backward.py (packed x/dy pools checked against each run's own row pool);
forward runs probe_integrated_forward.py.
"""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-dispatch-warps-ab")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_file(
            root / "examples/probe_dispatch_warps_backward.py", "/probe/backward.py"
        )
        .add_local_file(
            root / "examples/probe_integrated_forward.py", "/probe/forward.py"
        )
    )


@app.function(image=image, gpu="H100!", cpu=16, memory=65536, timeout=3000)
def ab(direction, routing, variants, warmup, iters, rounds):
    import torch

    name = torch.cuda.get_device_name()
    if "H100" not in name:
        return dict(returncode=2, stderr=f"wrong device {name}", stdout="", device=name)
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        f"/probe/{direction}.py",
        "--routing",
        routing,
        "--variants",
        variants,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--rounds",
        str(rounds),
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
        timeout=2800,
        check=False,
    )
    payload = None
    for line in result.stdout.splitlines():
        if line.startswith("DIAG_JSON "):
            payload = json.loads(line[len("DIAG_JSON ") :])
    return dict(
        returncode=result.returncode if payload is not None else 1,
        device=name,
        diag=payload,
        stderr=result.stderr[-8000:],
        variant_log=[
            ln for ln in result.stderr.splitlines() if ln.startswith("VARIANT ")
        ],
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
    )


@app.local_entrypoint()
def main(
    out: str,
    variants: str,
    direction: str = "backward",
    routing: str = "balanced",
    warmup: int = 5,
    iters: int = 5,
    rounds: int = 3,
):
    """variants: "name:ENV=V+ENV2=V2,..." (the first variant is the parity reference)."""
    result = ab.remote(direction, routing, variants, warmup, iters, rounds)
    result.update(
        direction=direction,
        routing=routing,
        variants=variants,
        modal_app_id=os.environ.get("MODAL_APP_ID") or app.app_id,
    )
    Path(out).write_text(json.dumps(result, indent=1))
    medians = {}
    if result.get("diag"):
        medians = {
            k: (v["median_ms"], v["samples"])
            for k, v in result["diag"]["results"].items()
        }
    print(
        json.dumps(
            dict(
                returncode=result["returncode"],
                device=result.get("device"),
                modal_app_id=result["modal_app_id"],
                medians=medians,
            )
        )
    )
    if result["returncode"]:
        print(result["stderr"][-6000:])
        raise RuntimeError("dispatch-warps A/B failed")
