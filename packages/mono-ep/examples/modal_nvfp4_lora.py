# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Modal runners for the NVFP4 LoRA persistent path on B200 (Blackwell image)."""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
# Resolved only on the client; the remote container imports this file from /root.
AXOLOTL_SRC = ROOT.parents[1] / "src" if len(ROOT.parents) > 1 else ROOT / "src"
BASE_IMAGE = os.environ.get(
    "MONO_EP_BASE_IMAGE",
    "axolotlai/axolotl-cloud-uv:main-py3.14-cu130-2.14.1-blackwell",
)
# cuDNN frontend OSS grouped block-scaled GEMM, the GEMM-floor reference (job `cudnn` only)
CUDNN_FRONTEND = "nvidia-cudnn-frontend[cutedsl]==1.32.2"
app = modal.App("axolotl-mono-ep-nvfp4-lora")
base_image = (
    modal.Image.from_registry(BASE_IMAGE)
    .run_commands(
        'uv pip install --python "$(command -v python)" setuptools wheel packaging ninja',
        "apt-get update && apt-get install -y libdw-dev libelf-dev software-properties-common",
        "add-apt-repository -y ppa:ubuntu-toolchain-r/test && apt-get update && apt-get install -y g++-13",
    )
    # The JIT-only kernel headers are overlaid at run time (see `_overlay_headers`), so editing them
    # does not rebuild the host extension.
    .add_local_dir(
        ROOT / "vendor",
        "/opt/mono-ep/vendor",
        copy=True,
        ignore=["deepgemm/deep_gemm/include/deep_gemm/impls/**"],
    )
    .add_local_file(
        ROOT / "tools/fetch_build_dependencies.py",
        "/opt/mono-ep/tools/fetch_build_dependencies.py",
        copy=True,
    )
    .run_commands(
        "python /opt/mono-ep/tools/fetch_build_dependencies.py",
        "cd /opt/mono-ep/vendor/deepgemm && DG_USE_LOCAL_VERSION=0 DG_FORCE_BUILD=1 "
        'CC=gcc-13 CXX=g++-13 MAX_JOBS=4 uv pip install --python "$(command -v python)" --no-build-isolation --no-deps .',
    )
)


def _with_sources(img):
    return (
        img.add_local_dir(ROOT / "src", "/probe/src")
        .add_local_dir(
            AXOLOTL_SRC / "axolotl/integrations/kernels/libs/sonicmoe",
            "/probe/sonicmoe",
        )
        .add_local_dir(ROOT / "examples", "/probe/examples")
        .add_local_dir(ROOT / "vendor/deepgemm/deep_gemm/include", "/probe/include")
    )


image = _with_sources(base_image)
# A separate image so the frontend's dependency resolution cannot move the other jobs' packages
cudnn_image = _with_sources(
    base_image.run_commands(
        f'uv pip install --python "$(command -v python)" "{CUDNN_FRONTEND}"'
    )
)
ENV = {
    "PYTHONPATH": "/probe/src",
    "SONICMOE_DIR": "/probe/sonicmoe",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1",
}


def _overlay_headers():
    import shutil

    import deep_gemm

    shutil.copytree(
        "/probe/include",
        Path(deep_gemm.__file__).parent / "include",
        dirs_exist_ok=True,
    )


def _run(script, nproc, extra_env, timeout):
    _overlay_headers()
    out = "/tmp/result.json"  # nosec B108
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


def _requant_env(requant):
    """``MONO_EP_NVFP4_REQUANT`` for the remote run (the requant kernel selector)."""
    return {"MONO_EP_NVFP4_REQUANT": requant} if requant else {}


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=3600)
def forward_parity(cases: str = "", requant: str = ""):
    env = _requant_env(requant) | ({"NVFP4_CASES": cases} if cases else {})
    return _run("validate_nvfp4_forward.py", 1, env, 3400)


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=3600)
def backward_parity(cases: str = "", requant: str = ""):
    env = _requant_env(requant) | ({"NVFP4_CASES": cases} if cases else {})
    return _run("validate_nvfp4_backward.py", 1, env, 3400)


@app.function(image=image, gpu="B200:4", cpu=16, memory=131072, timeout=3600)
def fsdp2_training(requant: str = ""):
    return _run(
        "validate_nvfp4_fsdp2.py",
        4,
        {"NCCL_MAX_CTAS": "8"} | _requant_env(requant),
        3400,
    )


@app.function(image=image, gpu="B200:8", cpu=32, memory=262144, timeout=7200)
def timing(cases: str = "", requant: str = ""):
    runs = {}
    for case in (cases or "qwen30b,beam").split(","):
        env = {"NVFP4_BENCH_CASE": case} | _requant_env(requant)
        runs[case] = _run("benchmark_nvfp4_lora_b200.py", 8, env, 3000)
    return dict(
        returncode=max(r["returncode"] for r in runs.values()),
        result={k: r["result"] for k, r in runs.items()},
        tail="\n".join(f"== {k}\n{r['tail']}" for k, r in runs.items()),
    )


@app.function(image=cudnn_image, gpu="B200:8", cpu=32, memory=262144, timeout=7200)
def timing_cudnn(cases: str = "", requant: str = ""):
    """``timing`` plus the ``cudnn_grouped_a2a`` variant (the runner adds it when the
    cuDNN frontend is importable)."""
    return timing.local(cases, requant)


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=7200)
def profile(cases: str = "", requant: str = ""):
    env = _requant_env(requant) | ({"NVFP4_CASES": cases} if cases else {})
    return _run("profile_nvfp4_lora_b200.py", 1, env, 7000)


@app.function(image=cudnn_image, gpu="B200", cpu=8, memory=131072, timeout=7200)
def cudnn(cases: str = ""):
    env = {"NVFP4_CASES": cases} if cases else {}
    return _run("benchmark_cudnn_grouped_gemm_b200.py", 1, env, 7000)


NCU_METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sector_op_read_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__m_xbar2l1tex_read_bytes.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
)


@app.function(image=cudnn_image, gpu="B200", cpu=8, memory=131072, timeout=7200)
def ncu(cases: str = ""):
    """Nsight Compute memory metrics of the NVFP4 backward kernel and cuDNN's Dh / Dx
    GEMMs on the same operands (``ncu_nvfp4_backward_b200.py``)."""
    import csv
    import glob
    import io
    import shutil

    _overlay_headers()
    found = [
        p
        for p in (
            "/usr/local/cuda/bin/ncu",
            shutil.which("ncu"),
            *glob.glob("/opt/nvidia/nsight-compute/*/ncu"),
        )
        if p and os.path.exists(p)
    ]
    if not found:
        return dict(returncode=127, result=None, tail="ncu not found")
    env = {
        **os.environ,
        **ENV,
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29593",
    }

    def parse(text):
        lines = [line for line in text.splitlines() if line.startswith('"')]
        if len(lines) < 3:
            return []
        rows = list(csv.reader(io.StringIO("\n".join(lines))))
        header, units = rows[0], rows[1]
        return [
            {k: (v, u) for k, u, v in zip(header, units, row, strict=False)}
            for row in rows[2:]
        ]

    runs, tails = {}, []
    for case in (cases or "beam,qwen30b").split(","):
        for label, regex, skip, count in (
            ("mono_ep_backward", "sm100_nvfp4_mono_ep_backward", 2, 1),
            ("cudnn_dh_dx", "cudnn_kernel", 0, 6),
        ):
            command = [
                found[0], "--target-processes", "all", "--replay-mode", "kernel",
                "--clock-control", "none", "--kernel-name-base", "demangled",
                "--kernel-name", f"regex:{regex}", "--launch-skip", str(skip),
                "--launch-count", str(count), "--metrics", ",".join(NCU_METRICS),
                "--csv", "--page", "raw",
                "python", "/probe/examples/ncu_nvfp4_backward_b200.py",
            ]  # fmt: skip
            proc = subprocess.run(  # nosec B603
                command,
                cwd="/probe",
                env={**env, "NVFP4_CASE": case},
                capture_output=True,
                text=True,
                timeout=3000,
                check=False,
            )
            kernels = parse(proc.stdout)
            runs[f"{case}/{label}"] = dict(returncode=proc.returncode, kernels=kernels)
            tails.append(
                f"== {case}/{label} rc={proc.returncode}\n{proc.stderr[-3000:]}"
            )
    version = subprocess.run(  # nosec B603
        [found[0], "--version"], capture_output=True, text=True, check=False
    ).stdout
    ok = all(r["returncode"] == 0 and r["kernels"] for r in runs.values())
    return dict(
        returncode=0 if ok else 1,
        result=dict(ncu=found[0], ncu_version=version, runs=runs),
        tail="\n".join(tails),
    )


@app.local_entrypoint()
def main(job: str = "forward", out: str = "", cases: str = "", requant: str = ""):
    if job == "forward":
        result = forward_parity.remote(cases, requant)
    elif job == "backward":
        result = backward_parity.remote(cases, requant)
    elif job == "fsdp2":
        result = fsdp2_training.remote(requant)
    elif job == "timing":
        result = timing.remote(cases, requant)
    elif job == "profile":
        result = profile.remote(cases, requant)
    elif job == "timing_cudnn":
        result = timing_cudnn.remote(cases, requant)
    elif job == "ncu":
        result = ncu.remote(cases)
    elif job == "cudnn":
        result = cudnn.remote(cases)
    else:
        raise ValueError(job)
    result["modal_app_id"] = app.app_id
    if out:
        Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "tail"}, indent=1))
    if result["returncode"]:
        raise SystemExit(result["tail"])
