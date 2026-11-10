# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run the single-rank SM100 diagnosis probe on exactly one B200.

The worktree's kernel headers are overlaid onto the installed DeepGEMM include
tree at runtime, so header-only instrumentation changes do not rebuild the image.
"""

import json
import os
import shutil
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-diag-b200")
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
        .add_local_file(root / "examples/probe_diag_b200.py", "/probe/diag.py")
    )


@app.function(image=image, gpu="B200", cpu=16, memory=65536, timeout=7200)
def probe(args):
    import deep_gemm
    import torch

    name = torch.cuda.get_device_name()
    if "B200" not in name or torch.cuda.device_count() != 1:
        return dict(returncode=2, stderr=f"wrong device {name}", stdout="")
    installed = Path(deep_gemm.__file__).parent / "include/deep_gemm"
    shutil.copytree("/probe/headers/deep_gemm", installed, dirs_exist_ok=True)
    cache = Path("/tmp/dg_jit_cache")  # nosec B108
    shutil.rmtree(cache, ignore_errors=True)
    runs = []
    for run_args in args:
        command = [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            "/probe/diag.py",
            *run_args,
        ]
        result = subprocess.run(  # nosec B603
            command,
            env={
                **os.environ,
                "PYTHONPATH": "/probe/src",
                "OMP_NUM_THREADS": "1",
                "DG_JIT_CACHE_DIR": str(cache),
            },
            capture_output=True,
            text=True,
            timeout=2400,
            check=False,
        )
        runs.append(
            dict(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr[-20000:],
                command=command,
            )
        )
        if result.returncode:
            break
    jit_kernels = sorted(
        {path.parent.name for path in cache.rglob("*.cubin")}
        | {path.name for path in cache.glob("*") if path.is_dir()}
    )
    return dict(
        runs=runs,
        jit_kernels=jit_kernels,
        returncode=max(r["returncode"] for r in runs),
        device=name,
        torch=torch.__version__,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
    )


@app.local_entrypoint()
def main(out: str, args: str = ""):
    """``args`` holds one or more probe invocations separated by ``;;``, run in one container."""
    result = probe.remote([part.split() for part in args.split(";;")])
    payloads = []
    for run in result["runs"]:
        run["diag"] = None
        for line in run["stdout"].splitlines():
            if line.startswith("DIAG_JSON "):
                run["diag"] = json.loads(line[len("DIAG_JSON ") :])
        run["stdout"] = run["stdout"][-20000:]
        payloads.append(run["diag"])
    result["diag"] = payloads[0] if len(payloads) == 1 else payloads
    result["modal_app_id"] = app.app_id
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("returncode", "device", "modal_app_id")}))
    if result["returncode"] or any(p is None for p in payloads):
        print(result["runs"][-1]["stderr"][-6000:])
        raise RuntimeError("diag probe failed")
