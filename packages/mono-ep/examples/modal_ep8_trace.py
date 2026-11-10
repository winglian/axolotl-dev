# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Multi-rank phase trace of the SM90 FP8 persistent training step on strict H100s.

gpus=8: EP8 balanced and skewed runs plus a same-allocation world-1 run of the same script (GPU 0).
gpus=2: short functional check of the trace on two ranks.
gpus=1: world-1 runs (balanced and skewed) on one strict H100, for host-side A/Bs.
Each run writes a gzip JSON (benchmark_ep8_trace.py --out) that is returned and saved locally.
"""

import base64
import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("mono-ep-ep8-trace")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_file(root / "examples/benchmark_ep8_trace.py", "/probe/ep_trace.py")
        .add_local_file(
            root / "examples/benchmark_persistent_training.py", "/probe/paired.py"
        )
    )


def _run(runs, env):
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not all("H100" in n for n in names):
        return dict(returncode=2, devices=names, runs=[])
    out = []
    for run in runs:
        path = f"/tmp/{run['name']}.json.gz"  # nosec B108
        paired = run.get("script") == "paired"
        command = [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={run['nproc']}",
            *(
                ["/probe/paired.py", *run["args"]]
                if paired
                else ["/probe/ep_trace.py", "--out", path, *run["args"]]
            ),
        ]
        result = subprocess.run(  # nosec B603
            command,
            env={
                **os.environ,
                "PYTHONPATH": "/probe/src",
                "OMP_NUM_THREADS": "1",
                "DG_JIT_CACHE_DIR": "/root/dg_jit",
                "MONO_EP_SM90_FP8_TRAP_TIMEOUTS": "0",
                "MONO_EP_SM90_FP8_TRANSPORT": "1",
                "MONO_EP_SM90_FP8_COMBINE": "1",
                **env,
                **run.get("env", {}),
            },
            capture_output=True,
            text=True,
            timeout=2400,
            check=False,
        )
        payload = None
        if os.path.exists(path):
            payload = base64.b64encode(Path(path).read_bytes()).decode()
        brief = None
        for line in result.stdout.splitlines():
            if line.startswith("EP_TRACE_JSON "):
                brief = json.loads(line[len("EP_TRACE_JSON ") :])
            elif paired and line.startswith('{"benchmark"'):
                brief = json.loads(line)
        out.append(
            dict(
                name=run["name"],
                command=command,
                returncode=result.returncode,
                brief=brief,
                payload=payload,
                stderr=result.stderr[-8000:] if result.returncode else "",
            )
        )
        if result.returncode:
            break
    return dict(
        returncode=max((r["returncode"] for r in out), default=1),
        devices=names,
        runs=out,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
    )


@app.function(image=image, gpu="H100!:8", cpu=32, memory=131072, timeout=3600)
def trace8(runs, env):
    return _run(runs, env)


@app.function(image=image, gpu="H100!:2", cpu=16, memory=65536, timeout=2400)
def trace2(runs, env):
    return _run(runs, env)


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, timeout=3600)
def trace1(runs, env):
    return _run(runs, env)


@app.local_entrypoint()
def main(
    out_dir: str,
    gpus: int = 8,
    variants: str = "default:",
    env: str = "",
    extra_args: str = "",
    runs: str = "ep8_balanced,ep8_skewed,ep1_balanced",
    paired_control: bool = False,
):
    """variants: benchmark_ep8_trace.py --variants (same env applied on every rank)."""
    extra = dict(item.split("=", 1) for item in env.split("+") if item)
    more = extra_args.split()
    if gpus == 8:
        plan = [
            dict(
                name="ep8_balanced",
                nproc=8,
                args=[
                    "--routing",
                    "balanced",
                    "--variants",
                    variants,
                    "--profile",
                    *more,
                ],
            ),
            dict(
                name="ep8_skewed",
                nproc=8,
                args=[
                    "--routing",
                    "skewed",
                    "--variants",
                    variants,
                    "--profile",
                    *more,
                ],
            ),
            dict(
                name="ep1_balanced",
                nproc=1,
                args=[
                    "--routing",
                    "balanced",
                    "--variants",
                    variants,
                    "--profile",
                    *more,
                ],
            ),
        ]
        selected = runs.split(",")
        plan = [r for r in plan if r["name"] in selected]
        if paired_control:
            # Same allocation: the standard paired native/NCCL timing at the default config.
            plan.append(
                dict(
                    name="paired_native_nccl",
                    nproc=8,
                    script="paired",
                    args=[
                        "--compute-dtype",
                        "blockwise",
                        "--recompute",
                        "--compare-native",
                        "--compiled",
                    ],
                )
            )
        result = trace8.remote(plan, extra)
    elif gpus == 1:
        plan = [
            dict(
                name=f"ep1_{routing}",
                nproc=1,
                args=["--routing", routing, "--variants", variants, *more],
            )
            for routing in ("balanced", "skewed")
        ]
        selected = runs.split(",")
        if not any(r["name"] in selected for r in plan):
            selected = [f"ep1_{routing}" for routing in ("balanced", "skewed")]
        result = trace1.remote([r for r in plan if r["name"] in selected], extra)
    elif gpus == 2:
        plan = [
            dict(
                name="ep2_check",
                nproc=2,
                args=[
                    "--routing",
                    "skewed",
                    "--variants",
                    variants,
                    "--warmup",
                    "2",
                    "--iters",
                    "3",
                    "--trace-iters",
                    "2",
                    "--profile",
                    *more,
                ],
            )
        ]
        result = trace2.remote(plan, extra)
    else:
        raise ValueError("gpus must be 8, 2 or 1")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    app_id = os.environ.get("MODAL_APP_ID") or app.app_id
    for run in result["runs"]:
        if run["payload"]:
            (target / f"{run['name']}.json.gz").write_bytes(
                base64.b64decode(run.pop("payload"))
            )
        else:
            run.pop("payload")
    result.update(modal_app_id=app_id, gpus=gpus, variants=variants, env=extra)
    (target / "launch.json").write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            dict(
                returncode=result["returncode"],
                devices=result["devices"],
                modal_app_id=app_id,
                briefs=[r["brief"] for r in result["runs"]],
            )
        )
    )
    if result["returncode"]:
        for run in result["runs"]:
            if run["stderr"]:
                print(run["stderr"][-6000:])
        raise RuntimeError("EP trace failed")
