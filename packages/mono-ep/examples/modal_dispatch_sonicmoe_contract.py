# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Modal driver for ``dispatch_sonicmoe_contract.py``: one strict 8 x H100 job and one
8 x B200 job. Writes ``benchmarks/dispatch_sonicmoe_contract_{h100,b200}.json``."""

import json
import os
import subprocess  # nosec B404
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("axolotl-mono-ep-dispatch-sonicmoe-contract")


def _with_deepgemm(image):
    """The vendored DeepGEMM build the fused runtime needs (the same steps, in the same
    order, as ``modal_validate.base_image``, so the layers are shared)."""
    return (
        image.run_commands(
            'uv pip install --python "$(command -v python)" setuptools wheel packaging ninja',
            "apt-get update && apt-get install -y libdw-dev libelf-dev software-properties-common",
            "add-apt-repository -y ppa:ubuntu-toolchain-r/test && apt-get update && apt-get install -y g++-13",
        )
        .add_local_dir(
            Path(os.environ.get("MONO_EP_VENDOR_DIR", ROOT / "vendor")),
            "/opt/mono-ep/vendor",
            copy=True,
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


def _with_sources(image):
    return (
        image.add_local_dir(ROOT / "src", "/probe/mono_ep_src")
        .add_local_dir(ROOT / "../../src", "/probe/framework")
        .add_local_file(
            ROOT / "examples/dispatch_sonicmoe_contract.py", "/probe/contract.py"
        )
        .add_local_file(
            ROOT / "examples/sonicmoe_contract_profiles.py",
            "/probe/sonicmoe_contract_profiles.py",
        )
    )


# H100: stock image plus the vendored DeepGEMM build and the SonicMoE hub kernel's CuteDSL
# build deps.
hopper_image = _with_sources(
    _with_deepgemm(
        modal.Image.from_registry("axolotlai/axolotl-uv:main-latest")
    ).run_commands(
        'uv pip install --python "$(command -v python)" '
        '"nvidia-cutlass-dsl==4.8.0" "apache-tvm-ffi>=0.1.10"'
    )
)
# B200: the kernel lane's image, which carries axolotl's `blackwell` extra (quack, flash-attn-4,
# CuteDSL 4.8.1). The hub-vendored quack 0.6.1 still uses the pre-4.8 `nvvm.mbarrier_arrive`
# call form on its SM100 2-CTA path; the worker applies the kernels plugin's shim for it.
blackwell_image = _with_sources(
    _with_deepgemm(
        modal.Image.from_registry(
            "axolotlai/axolotl-cloud-uv:main-py3.14-cu130-2.14.1-blackwell"
        )
    )
)


def _run_worker(expected_gpu):
    import torch

    actual = torch.cuda.get_device_name()
    if expected_gpu not in actual:
        raise RuntimeError(f"expected {expected_gpu}, allocated {actual}")
    print(f"allocated {actual}", flush=True)
    started = time.time()
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "/probe/contract.py",
        "--worker",
        "--profiles",
        expected_gpu,
    ]
    lines = []
    with subprocess.Popen(  # nosec B603
        command,
        env={
            **os.environ,
            "PYTHONPATH": "/probe/framework:/probe/mono_ep_src",
            "OMP_NUM_THREADS": "1",
            # the fused H100 reference line's settings; read only by the SM90 blockwise
            # backward, inert on B200 / BF16
            "MONO_EP_SM90_FP8_TRANSPORT": "1",
            "MONO_EP_SM90_FP8_COMBINE": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        returncode = process.wait()
    result = [line for line in lines if line.startswith("RESULT ")]
    if not result:
        raise RuntimeError(f"worker printed no RESULT line (exit {returncode})")
    payload = json.loads(result[-1][len("RESULT ") :])
    payload["worker_returncode"] = returncode
    payload["wall_seconds"] = time.time() - started
    payload["gpu_hours"] = 8 * payload["wall_seconds"] / 3600
    payload["modal_task_id"] = os.environ.get("MODAL_TASK_ID")
    return payload


@app.function(image=hopper_image, gpu="H100!:8", cpu=32, memory=131072, timeout=5400)
def contract_h100():
    return _run_worker("H100")


@app.function(image=blackwell_image, gpu="B200:8", cpu=32, memory=131072, timeout=5400)
def contract_b200():
    return _run_worker("B200")


@app.local_entrypoint()
def main(gpu: str = "H100"):
    function = {"H100": contract_h100, "B200": contract_b200}[gpu]
    result = function.remote()
    result["modal_app_id"] = app.app_id
    result["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = ROOT / f"benchmarks/dispatch_sonicmoe_contract_{gpu.lower()}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out}")
    if result["worker_returncode"]:
        raise RuntimeError(f"{gpu} worker exited {result['worker_returncode']}")
