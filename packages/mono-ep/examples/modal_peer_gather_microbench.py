# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Modal driver for ``peer_gather_microbench.py``: one strict 2 x H100 job and one 2 x B200
job. Writes ``benchmarks/peer_gather_microbench_{h100,b200}.json``."""

import json
import os
import subprocess  # nosec B404
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("axolotl-mono-ep-peer-gather-microbench")


def _with_sources(image):
    return (
        image.add_local_dir(ROOT / "src", "/probe/mono_ep_src")
        .add_local_dir(ROOT / "../../src", "/probe/framework")
        .add_local_file(
            ROOT / "examples/peer_gather_microbench.py", "/probe/microbench.py"
        )
    )


# H100: stock image plus the SonicMoE hub kernel's CuteDSL build deps.
hopper_image = _with_sources(
    modal.Image.from_registry("axolotlai/axolotl-uv:main-latest").run_commands(
        'uv pip install --python "$(command -v python)" '
        '"nvidia-cutlass-dsl==4.8.0" "apache-tvm-ffi>=0.1.10"'
    )
)
# B200: the kernel lane's image (axolotl `blackwell` extra: quack, CuteDSL 4.8.1).
blackwell_image = _with_sources(
    modal.Image.from_registry(
        "axolotlai/axolotl-cloud-uv:main-py3.14-cu130-2.14.1-blackwell"
    )
)


def _run_worker(expected_gpu):
    import torch

    actual = torch.cuda.get_device_name()
    if expected_gpu not in actual:
        raise RuntimeError(f"expected {expected_gpu}, allocated {actual}")
    print(f"allocated {actual} x {torch.cuda.device_count()}", flush=True)
    started = time.time()
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "/probe/microbench.py",
        "--worker",
        "--gpu",
        expected_gpu,
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
    if result:
        payload = json.loads(result[-1][len("RESULT ") :])
    else:
        payload = {"benchmark": "peer_gather_microbench", "stages_completed": []}
    # the worker prints a snapshot after every stage; keep the tail of the log when it died
    if returncode or not result or payload["stages_completed"][-1:] != ["done"]:
        payload["worker_log_tail"] = "".join(lines[-120:])
    payload["worker_returncode"] = returncode
    payload["wall_seconds"] = time.time() - started
    payload["gpu_hours"] = 2 * payload["wall_seconds"] / 3600
    payload["modal_task_id"] = os.environ.get("MODAL_TASK_ID")
    return payload


@app.function(image=hopper_image, gpu="H100!:2", cpu=8, memory=65536, timeout=1800)
def microbench_h100():
    return _run_worker("H100")


@app.function(image=blackwell_image, gpu="B200:2", cpu=8, memory=65536, timeout=1800)
def microbench_b200():
    return _run_worker("B200")


@app.local_entrypoint()
def main(gpu: str = "H100"):
    function = {"H100": microbench_h100, "B200": microbench_b200}[gpu]
    result = function.remote()
    result["modal_app_id"] = app.app_id
    result["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = ROOT / f"benchmarks/peer_gather_microbench_{gpu.lower()}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out}")
    if result["worker_returncode"]:
        raise RuntimeError(f"{gpu} worker exited {result['worker_returncode']}")
