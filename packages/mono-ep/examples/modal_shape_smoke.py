# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single strict-H100 smoke of the integrated SM90 FP8 forward and backward at several layer shapes.

Runs probe_integrated_{forward,backward}.py (world 1, all local experts on one GPU) once per
(shape, direction, routing) in one container with the package defaults, and records timing and
peak memory. Shapes are 'name:TxHxIxE' (tokens, hidden, intermediate, local experts; top-k 8).
"""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-mono-ep-shape-smoke")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_file(
            root / "examples/probe_integrated_backward.py", "/probe/backward.py"
        )
        .add_local_file(
            root / "examples/probe_integrated_forward.py", "/probe/forward.py"
        )
    )

SHAPES = {
    "qwen3_30b_a3b": "qwen3_30b_a3b:16384x2048x768x16",
    "qwen3_235b_a22b": "qwen3_235b_a22b:16384x4096x1536x16",
}


@app.function(image=image, gpu="H100!", cpu=16, memory=65536, timeout=3600)
def smoke(configs, warmup, iters, rounds):
    import torch

    device = torch.cuda.get_device_name()
    if "H100" not in device:
        return dict(returncode=2, device=device, runs=[])
    runs = []
    for shape, direction, routing, variants in configs:
        name, _, dims = SHAPES.get(shape, shape).partition(":")
        tokens, hidden, inner, experts = dims.split("x")
        result = subprocess.run(  # nosec B603 B607
            [
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
                "--tokens",
                tokens,
                "--hidden",
                hidden,
                "--intermediate",
                inner,
                "--experts",
                experts,
            ],
            env={
                **os.environ,
                "PYTHONPATH": "/probe/src",
                "OMP_NUM_THREADS": "1",
                "MONO_EP_SM90_FP8_TRANSPORT": "1",
                "MONO_EP_SM90_FP8_COMBINE": "1",
            },
            capture_output=True,
            text=True,
            timeout=1500,
            check=False,
        )
        payload = None
        for line in result.stdout.splitlines():
            if line.startswith("DIAG_JSON "):
                payload = json.loads(line[len("DIAG_JSON ") :])
        runs.append(
            dict(
                shape=name,
                direction=direction,
                routing=routing,
                variants=variants,
                returncode=result.returncode,
                diag=payload,
                stderr=result.stderr[-8000:] if result.returncode else "",
                error_lines=[
                    line[:2000]
                    for line in (result.stdout + result.stderr).splitlines()
                    if any(
                        k in line
                        for k in ("error", "Error", "Assertion", "assert", "failed")
                    )
                ][:40],
            )
        )
        print(f"SMOKE {name} {direction} {routing} rc={result.returncode}", flush=True)
        if result.returncode:
            print(result.stderr[-8000:], flush=True)
    return dict(
        returncode=max(r["returncode"] for r in runs),
        device=device,
        runs=runs,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
    )


@app.local_entrypoint()
def main(
    out: str,
    shapes: str = "qwen3_30b_a3b,qwen3_235b_a22b",
    directions: str = "forward,backward",
    routings: str = "balanced,skewed",
    variants: str = "defaults:",
    warmup: int = 5,
    iters: int = 5,
    rounds: int = 2,
    plan: str = "",
):
    """plan: JSON list of [shape, direction, routing, variants] replacing the shape/direction grid."""
    configs = (
        [tuple(c) for c in json.loads(plan)]
        if plan
        else [
            (s, d, r, variants)
            for s in shapes.split(",")
            for d in directions.split(",")
            for r in routings.split(",")
        ]
    )
    result = smoke.remote(configs, warmup, iters, rounds)
    result["modal_app_id"] = os.environ.get("MODAL_APP_ID") or app.app_id
    result["warmup_per_round"], result["iters_per_round"] = warmup, iters
    result["rounds"] = rounds
    Path(out).write_text(json.dumps(result, indent=1))
    summary = {}
    for run in result["runs"]:
        diag = run["diag"] or {}
        for variant, value in diag.get("results", {}).items():
            summary[f"{run['shape']}/{run['direction']}/{run['routing']}/{variant}"] = (
                dict(
                    native_ms=value["median_ms"],
                    call_ms=value["call_median_ms"],
                    samples=value["samples"],
                    peak_allocated_gb=diag["peak_allocated_bytes"] / 1e9,
                    device_used_gb=diag["device_used_bytes"] / 1e9,
                )
            )
    print(
        json.dumps(
            dict(
                returncode=result["returncode"],
                device=result["device"],
                modal_app_id=result["modal_app_id"],
                summary=summary,
            )
        )
    )
    if result["returncode"]:
        raise RuntimeError("shape smoke failed")
