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


def _with_sources(image):
    return (
        image.add_local_dir(ROOT / "src", "/probe/mono_ep_src")
        .add_local_dir(ROOT / "../../src", "/probe/framework")
        .add_local_file(
            ROOT / "examples/dispatch_sonicmoe_contract.py", "/probe/contract.py"
        )
    )


# H100: stock image plus the SonicMoE hub kernel's CuteDSL build deps.
hopper_image = _with_sources(
    modal.Image.from_registry("axolotlai/axolotl-uv:main-latest").run_commands(
        'uv pip install --python "$(command -v python)" '
        '"nvidia-cutlass-dsl==4.8.0" "apache-tvm-ffi>=0.1.10"'
    )
)
# B200: the kernel lane's image, which carries axolotl's `blackwell` extra (quack, flash-attn-4).
# Its unpinned CuteDSL (4.8.1) changed `mbarrier_arrive`, which breaks the hub-vendored quack
# SM100 GEMM; the lane itself ends up on 4.8.0 through mono-ep's `sonic` extra, so pin that.
blackwell_image = _with_sources(
    modal.Image.from_registry(
        "axolotlai/axolotl-cloud-uv:main-py3.14-cu130-2.14.1-blackwell"
    ).run_commands(
        'uv pip install --python "$(command -v python)" "nvidia-cutlass-dsl==4.8.0"'
    )
)


def _run_worker(expected_gpu):
    import torch

    actual = torch.cuda.get_device_name()
    if expected_gpu not in actual:
        raise RuntimeError(f"expected {expected_gpu}, allocated {actual}")
    started = time.time()
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "/probe/contract.py",
        "--worker",
    ]
    lines = []
    with subprocess.Popen(  # nosec B603
        command,
        env={
            **os.environ,
            "PYTHONPATH": "/probe/framework:/probe/mono_ep_src",
            "OMP_NUM_THREADS": "1",
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
