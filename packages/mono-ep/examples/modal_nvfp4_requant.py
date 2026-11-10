# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Modal runners for the CUDA NVFP4 requant on one B200: ``validate``
(``validate_nvfp4_requant_cuda.py``: conversions, byte parity, timings) and
``ncu`` (Nsight Compute of the Triton and CUDA kernels on the Beam ``w1`` form)."""

import csv
import io
import json
import os
import shutil
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-mono-ep-nvfp4-requant")
image = None
if modal.is_local():
    from modal_nvfp4_lora import ROOT, image as lora_image

    image = lora_image.add_local_dir(ROOT / "tests", "/probe/tests")

ENV = {"PYTHONPATH": "/probe/src", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"}
SECTIONS = (
    "SpeedOfLight",
    "ComputeWorkloadAnalysis",
    "InstructionStats",
    "WarpStateStats",
    "SchedulerStats",
    "MemoryWorkloadAnalysis",
    "LaunchStats",
    "Occupancy",
)


def _stream(cmd, env, timeout):
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
    return proc.returncode, lines


@app.function(image=image, gpu="B200", cpu=8, memory=131072, timeout=5400)
def validate(cases: str = "", skip_small: bool = False):
    out = "/tmp/result.json"  # nosec B108
    env = {
        **os.environ,
        **ENV,
        "NVFP4_OUT": out,
        "NVFP4_SKIP_SMALL": str(int(skip_small)),
    }
    if cases:
        env["NVFP4_CASES"] = cases
    code, lines = _stream(
        ["python", "/probe/examples/validate_nvfp4_requant_cuda.py"], env, 5200
    )
    result = json.loads(Path(out).read_text()) if Path(out).exists() else None
    return dict(returncode=code, result=result, tail="".join(lines[-80:]))


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=1800)
def script(name: str):
    env = {**os.environ, **ENV}
    code, lines = _stream(["python", f"/probe/examples/{name}"], env, 1700)
    return dict(returncode=code, result=None, tail="".join(lines[-80:]))


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=1800)
def pytest_run(path: str):
    env = {**os.environ, **ENV}
    cmd = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", f"/probe/{path}"]
    code, lines = _stream(cmd, env, 1700)
    return dict(returncode=code, result=None, tail="".join(lines[-40:]))


def _parse_raw(text):
    rows = [line for line in text.splitlines() if line.startswith('"')]
    if len(rows) < 3:
        return []
    table = list(csv.reader(io.StringIO("\n".join(rows))))
    header, units = table[0], table[1]
    return [
        {
            name: (value, unit)
            for name, unit, value in zip(header, units, row, strict=False)
        }
        for row in table[2:]
    ]


@app.function(image=image, gpu="B200", cpu=8, memory=131072, timeout=3600)
def ncu(kernels: str = "triton,cuda,cuda_fused"):
    executable = next(
        (
            c
            for c in ("/usr/local/cuda/bin/ncu", shutil.which("ncu"))
            if c and os.path.exists(c)
        ),
        None,
    )
    if executable is None:
        return dict(returncode=127, result=None, tail="ncu not found")
    env = {
        **os.environ,
        **ENV,
        "PROBE_KERNELS": kernels,
        "MONO_EP_NVFP4_REQUANT_VERBOSE": "1",
    }
    # build the extension and the Triton kernel outside the profiler
    code, lines = _stream(
        ["python", "/probe/examples/probe_nvfp4_requant.py"], env, 1800
    )
    if code:
        return dict(returncode=code, result=None, tail="".join(lines[-80:]))
    sections = [arg for s in SECTIONS for arg in ("--section", s)]
    cmd = [
        executable,
        "--target-processes", "all",
        "--replay-mode", "kernel",
        "--clock-control", "none",
        "--kernel-name-base", "function",
        "--kernel-name", "regex:requant_kernel",
        "--launch-skip", str(len(kernels.split(","))),
        "--launch-count", str(len(kernels.split(","))),
        *sections,
        "--csv", "--page", "raw",
        "python", "/probe/examples/probe_nvfp4_requant.py",
    ]  # fmt: skip
    result = subprocess.run(
        cmd,
        cwd="/probe",
        env=env,
        capture_output=True,
        text=True,
        timeout=3000,
        check=False,
    )  # nosec B603
    return dict(
        returncode=result.returncode,
        result=_parse_raw(result.stdout),
        build="".join(
            line
            for line in lines
            if "registers" in line or "spill" in line or "Compiling entry" in line
        ),
        tail=result.stdout[-2000:] + result.stderr[-4000:],
    )


@app.local_entrypoint()
def main(
    job: str = "validate",
    out: str = "",
    cases: str = "",
    skip_small: bool = False,
    kernels: str = "triton,cuda,cuda_fused",
):
    if job == "validate":
        result = validate.remote(cases, skip_small)
    elif job.startswith("tests/"):
        result = pytest_run.remote(job)
    elif job.endswith(".py"):
        result = script.remote(job)
    elif job == "ncu":
        result = ncu.remote(kernels)
    else:
        raise ValueError(job)
    result["modal_app_id"] = app.app_id
    if out:
        Path(out).write_text(json.dumps(result, indent=1))
    summary = {k: v for k, v in result.items() if k not in ("tail", "result")}
    print(json.dumps(summary, indent=1))
    print(result["tail"][-3000:])
    if result["returncode"]:
        raise SystemExit(1)
