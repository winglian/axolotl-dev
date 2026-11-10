# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run the multi-rank SM100 trace on eight B200s (plus a single-rank control).

``--gpus 8`` runs EP8 and then the same per-GPU workload on one of the eight
GPUs (world 1) in the same container; ``--gpus 1`` runs only the single-rank
control. The worktree's kernel headers are overlaid onto the installed
DeepGEMM include tree, so header-only changes do not rebuild the image (build
it from a byte-identical vendor snapshot with ``MONO_EP_VENDOR_DIR``).
"""

import json
import os
import shutil
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-ep-trace-b200")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(
            root / "vendor/deepgemm/deep_gemm/include/deep_gemm",
            "/probe/headers/deep_gemm",
        )
        .add_local_dir(root / "src", "/probe/src")
        .add_local_file(
            root / "examples/probe_diag_b200.py", "/probe/probe_diag_b200.py"
        )
        .add_local_file(
            root / "examples/benchmark_ep_trace_b200.py", "/probe/ep_trace.py"
        )
    )


def _run(expected, worlds, args):
    import deep_gemm
    import torch

    names = {torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
    if torch.cuda.device_count() != expected or any("B200" not in n for n in names):
        return dict(returncode=2, stderr=f"wrong devices {names}", runs=[])
    installed = Path(deep_gemm.__file__).parent / "include/deep_gemm"
    shutil.copytree("/probe/headers/deep_gemm", installed, dirs_exist_ok=True)
    cache = Path("/tmp/dg_jit_cache")  # nosec B108
    shutil.rmtree(cache, ignore_errors=True)
    runs = []
    argsets = (
        [a.split() for a in " ".join(args).split(";;")]
        if ";;" in " ".join(args)
        else [args]
    )
    for run_args, world in [(a, w) for a in argsets for w in worlds]:
        command = [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world}",
            "/probe/ep_trace.py",
            *run_args,
        ]
        result = subprocess.run(  # nosec B603
            command,
            env={
                **os.environ,
                "PYTHONPATH": "/probe/src:/probe",
                "OMP_NUM_THREADS": "1",
                "DG_JIT_CACHE_DIR": str(cache),
                "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(world)),
            },
            capture_output=True,
            text=True,
            timeout=2400,
            check=False,
        )
        payload = None
        for line in result.stdout.splitlines():
            if line.startswith("EPTRACE_JSON "):
                payload = json.loads(line[len("EPTRACE_JSON ") :])
        runs.append(
            dict(
                world=world,
                args=run_args,
                returncode=result.returncode,
                result=payload,
                stdout=result.stdout[-4000:],
                stderr=result.stderr[-20000:],
            )
        )
        if result.returncode:
            break
    return dict(
        returncode=max(r["returncode"] for r in runs),
        device=sorted(names),
        torch=torch.__version__,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
        runs=runs,
    )


@app.function(image=image, gpu="B200:8", cpu=32, memory=262144, timeout=5400)
def trace8(args, worlds=(8, 1)):
    return _run(8, tuple(worlds), args)


@app.function(image=image, gpu="B200", cpu=16, memory=65536, timeout=3600)
def trace1(args):
    return _run(1, (1,), args)


@app.local_entrypoint()
def main(out: str, gpus: int = 8, args: str = "", worlds: str = "8,1"):
    """``args`` may hold several argument sets separated by ``;;`` (each runs at every world)."""
    result = (
        trace8.remote(args.split(), [int(w) for w in worlds.split(",")])
        if gpus == 8
        else trace1.remote(args.split())
    )
    result["modal_app_id"] = app.app_id
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("returncode", "device", "modal_app_id")}))
    for run in result["runs"]:
        print("world", run["world"], "returncode", run["returncode"])
        if run["result"]:
            print(" ".join(run["args"]))
            print(json.dumps(run["result"]["timing_max_rank"]))
        if run["returncode"]:
            print(run["stderr"][-6000:])
    if result["returncode"]:
        raise RuntimeError("EP trace failed")
