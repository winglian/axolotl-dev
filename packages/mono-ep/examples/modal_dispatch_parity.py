# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Modal driver for ``dispatch_parity.py``: 2 x H100, EP2 x FSDP2, grouped_mm, scattermoe
and SonicMoE local kernels.

The dispatch mode needs only the package's Triton kernels and torch symmetric memory,
so the image is the stock Axolotl image plus the checkout's ``src`` trees; no DeepGEMM
build. Writes ``benchmarks/dispatch_mode_parity_h100.json``."""

import json
import os
import subprocess  # nosec B404
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("axolotl-mono-ep-dispatch-parity")
image = (
    modal.Image.from_registry("axolotlai/axolotl-uv:main-latest")
    # the SonicMoE hub kernel's CuteDSL build deps (axolotl's blackwell extra pins quack on
    # top of these; the plain swiglu path needs only the DSL and its FFI)
    .run_commands(
        'uv pip install --python "$(command -v python)" '
        '"nvidia-cutlass-dsl==4.8.0" "apache-tvm-ffi>=0.1.10"'
    )
    .add_local_dir(ROOT / "src", "/probe/mono_ep_src")
    .add_local_dir(ROOT / "../../src", "/probe/framework")
    .add_local_file(ROOT / "examples/dispatch_parity.py", "/probe/dispatch_parity.py")
)


def _run_worker():
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "/probe/dispatch_parity.py",
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
        if process.wait():
            raise RuntimeError("dispatch parity worker failed; see streamed logs")
    result = [line for line in lines if line.startswith("RESULT ")]
    if not result:
        raise RuntimeError("worker printed no RESULT line")
    return json.loads(result[-1][len("RESULT ") :])


@app.function(image=image, gpu="H100!:2", cpu=16, memory=65536, timeout=2400)
def hopper_parity():
    started = time.time()
    result = _run_worker()
    result["wall_seconds"] = time.time() - started
    result["device"] = subprocess.run(  # nosec B603 B607
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split("\n")[0]
    return result


@app.local_entrypoint()
def main():
    result = hopper_parity.remote()
    result["modal_app_id"] = app.app_id
    result["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = ROOT / "benchmarks/dispatch_mode_parity_h100.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
