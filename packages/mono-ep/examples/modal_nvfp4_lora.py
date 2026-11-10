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
app = modal.App("axolotl-mono-ep-nvfp4-lora")
image = (
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
    .add_local_dir(ROOT / "src", "/probe/src")
    .add_local_dir(
        AXOLOTL_SRC / "axolotl/integrations/kernels/libs/sonicmoe", "/probe/sonicmoe"
    )
    .add_local_dir(ROOT / "examples", "/probe/examples")
    .add_local_dir(ROOT / "vendor/deepgemm/deep_gemm/include", "/probe/include")
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


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=3600)
def forward_parity(cases: str = ""):
    env = {"NVFP4_CASES": cases} if cases else {}
    return _run("validate_nvfp4_forward.py", 1, env, 3400)


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=3600)
def backward_parity(cases: str = ""):
    env = {"NVFP4_CASES": cases} if cases else {}
    return _run("validate_nvfp4_backward.py", 1, env, 3400)


@app.local_entrypoint()
def main(job: str = "forward", out: str = "", cases: str = ""):
    if job == "forward":
        result = forward_parity.remote(cases)
    elif job == "backward":
        result = backward_parity.remote(cases)
    else:
        raise ValueError(job)
    result["modal_app_id"] = app.app_id
    if out:
        Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "tail"}, indent=1))
    if result["returncode"]:
        raise SystemExit(result["tail"])
