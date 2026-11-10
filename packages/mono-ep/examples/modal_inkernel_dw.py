# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run in-kernel dW checks on one, two or eight B200s.

The worktree's kernel headers are overlaid onto the installed DeepGEMM include
tree at runtime, so header-only changes reuse the built image (set
``MONO_EP_VENDOR_DIR`` to a frozen vendor snapshot to keep it cached). The
app name carries ``INKERNEL_DW_PURPOSE``.

``--runs`` holds ``;;``-separated invocations ``script|nproc|ENV=V,ENV=V|args``
(``script`` relative to ``examples/``), all run in one container.
"""

import json
import os
import shutil
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App(
    "axolotl-mono-ep-inkernel-dw-" + os.environ.get("INKERNEL_DW_PURPOSE", "dev")
)
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
        .add_local_dir(root / "examples", "/probe/examples")
    )


def _records(stdout):
    """JSON payload lines (``{...}`` or ``NAME_JSON {...}``), parsed before any truncation."""
    records = []
    for line in stdout.splitlines():
        head, _, rest = line.partition(" ")
        text = line if line.startswith("{") else rest
        if line.startswith("{") or (head.endswith("_JSON") and rest.startswith("{")):
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                pass
    return records


def _run(runs, gpus):
    import deep_gemm
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != gpus or not all("B200" in n for n in names):
        return dict(returncode=2, stderr=f"wrong devices {names}", runs=[])
    installed = Path(deep_gemm.__file__).parent / "include/deep_gemm"
    shutil.copytree("/probe/headers/deep_gemm", installed, dirs_exist_ok=True)
    cache = Path("/tmp/dg_jit_cache")  # nosec B108
    shutil.rmtree(cache, ignore_errors=True)
    results = []
    for spec in runs:
        script, nproc, env, args = spec.split("|", 3)
        extra = dict(item.split("=", 1) for item in filter(None, env.split(",")))
        command = [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            f"/probe/examples/{script}",
            *args.split(),
        ]
        result = subprocess.run(  # nosec B603
            command,
            env={
                **os.environ,
                "PYTHONPATH": "/probe/src:/probe/examples",
                "OMP_NUM_THREADS": "1",
                "DG_JIT_CACHE_DIR": str(cache),
                **extra,
            },
            capture_output=True,
            text=True,
            timeout=3000,
            check=False,
        )
        results.append(
            dict(
                spec=spec,
                returncode=result.returncode,
                records=_records(result.stdout),
                stdout=result.stdout[-20000:],
                stderr=result.stderr[-20000:],
            )
        )
        print(spec, "->", result.returncode, flush=True)
        if result.returncode:
            print(result.stdout[-4000:], result.stderr[-8000:], flush=True)
    return dict(
        returncode=max((r["returncode"] for r in results), default=0),
        devices=names,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        jit_kernels=sorted(p.name for p in cache.glob("*") if p.is_dir()),
        runs=results,
    )


@app.function(image=image, gpu="B200", cpu=16, memory=65536, timeout=7200)
def single(runs):
    return _run(runs, 1)


@app.function(image=image, gpu="B200:2", cpu=16, memory=131072, timeout=7200)
def pair(runs):
    return _run(runs, 2)


@app.function(image=image, gpu="B200:8", cpu=64, memory=262144, timeout=7200)
def octo(runs):
    return _run(runs, 8)


@app.local_entrypoint()
def main(out: str, runs: str, gpus: int = 1):
    result = {1: single, 2: pair, 8: octo}[gpus].remote(runs.split(";;"))
    result["modal_app_id"] = app.app_id
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("returncode", "devices", "torch", "modal_app_id")}
        )
    )
    if result["returncode"]:
        raise RuntimeError("in-kernel dW run failed; see the artifact")
