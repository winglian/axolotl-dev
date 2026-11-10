# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Modal runners for the BF16/FP8 native persistent backward ring-floor sweep
(one B200 or H100) and the EP8 timing (8xB200 or 8xH200), on the NVFP4 LoRA
image."""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-mono-ep-backward-ring-floor")
image = None
if modal.is_local():
    from modal_nvfp4_lora import image
ENV = {
    "PYTHONPATH": "/probe/src",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1",
}


def _run(script, nproc, extra_env, timeout):
    import shutil

    import deep_gemm

    shutil.copytree(
        "/probe/include",
        Path(deep_gemm.__file__).parent / "include",
        dirs_exist_ok=True,
    )
    out = "/tmp/result.json"  # nosec B108
    Path(out).unlink(missing_ok=True)
    env = {**os.environ, **ENV, **extra_env, "NVFP4_OUT": out}
    cmd = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc}",
        f"/probe/examples/{script}",
    ]
    lines = []
    with subprocess.Popen(  # nosec B603
        cmd,
        cwd="/probe",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait(timeout=timeout)
    result = json.loads(Path(out).read_text()) if Path(out).exists() else None
    return dict(returncode=proc.returncode, result=result, tail="".join(lines[-80:]))


def _sweep(cases, variants, floors):
    env = {"RING_TOKEN_FLOORS": floors} if floors else {}
    if cases:
        env["RING_CASES"] = cases
    if variants:
        env["RING_VARIANTS"] = variants
    return _run("profile_backward_ring_floor.py", 1, env, 3400)


def _timing(cases, dtypes):
    runs = {}
    for case in (cases or "qwen30b,beam").split(","):
        for dtype in dtypes.split(","):
            runs[f"{case}/{dtype}"] = _run(
                "benchmark_backward_ring_floor.py",
                8,
                {"RING_BENCH_CASE": case, "RING_BENCH_DTYPE": dtype},
                2400,
            )
    return dict(
        returncode=max(r["returncode"] for r in runs.values()),
        result={k: r["result"] for k, r in runs.items()},
        tail="\n".join(f"== {k}\n{r['tail']}" for k, r in runs.items()),
    )


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=3600)
def sweep_b200(cases: str = "", variants: str = "", floors: str = ""):
    return _sweep(cases, variants, floors)


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=3600)
def sweep_h100(cases: str = "", variants: str = "", floors: str = ""):
    return _sweep(cases, variants, floors)


@app.function(image=image, gpu="B200:8", cpu=32, memory=262144, timeout=7200)
def timing_b200(cases: str = ""):
    return _timing(cases, "bf16")


@app.function(image=image, gpu="H200:8", cpu=32, memory=262144, timeout=7200)
def timing_h200(cases: str = "", dtypes: str = ""):
    return _timing(cases, dtypes or "bf16,blockwise,blockwise_fp8_transport")


@app.local_entrypoint()
def main(
    job: str = "sweep_b200",
    out: str = "",
    cases: str = "",
    variants: str = "",
    floors: str = "",
):
    if job == "sweep_b200":
        result = sweep_b200.remote(cases, variants, floors)
    elif job == "sweep_h100":
        result = sweep_h100.remote(cases, variants, floors)
    elif job == "timing_b200":
        result = timing_b200.remote(cases)
    elif job == "timing_h200":
        result = timing_h200.remote(cases, variants)
    else:
        raise ValueError(job)
    result["modal_app_id"] = app.app_id
    if out:
        Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tail"}, indent=1))
    if result["returncode"]:
        raise SystemExit(result["tail"])
