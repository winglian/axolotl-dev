# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Isolated, pinned evaluation of the upstream SonicMoE FP8 proposal."""

import os
import signal
import subprocess
import threading
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1630585e1ff121cf21df9d700f37d8d4ee88e445"
app = modal.App("axolotl-sonic-fp8-probe")
image = (
    modal.Image.from_registry("axolotlai/axolotl-uv:main-latest")
    .run_commands(
        'uv pip install --python "$(command -v python)" '
        f'"sonic-moe @ git+https://github.com/bdepyzy/sonic-moe.git@{REVISION}" '
        '"quack-kernels==0.6.5" "nvidia-cutlass-dsl==4.8.0"'
    )
    .add_local_dir(ROOT / "src", "/probe/src")
    .add_local_file(ROOT / "tests/test_sonic_fp8.py", "/probe/test_sonic_fp8.py")
    .add_local_file(ROOT / "examples/validate.py", "/probe/validate.py")
    .add_local_file(
        ROOT / "examples/benchmark_pipeline.py", "/probe/benchmark_pipeline.py"
    )
    .add_local_file(ROOT / "examples/probe_sonic_fp8.py", "/probe/probe.py")
)


@app.function(image=image, gpu="B200", cpu=8, timeout=1200)
def probe(
    uncached=False, untuned=False, interleaved=False, standalone=False, edge_tests=False
):
    results = []
    flags = ["--uncached", "--qwen"] if uncached else []
    if untuned:
        flags.append("--untuned")
    if interleaved:
        flags.append("--interleaved")
    if standalone:
        flags = ["--standalone", "--qwen"]
    modes = (
        [["-m", "pytest", "-q", "/probe/test_sonic_fp8.py"]]
        if edge_tests
        else [["/probe/probe.py", *flags], ["/probe/probe.py", *flags, "--compiled"]]
    )
    for mode in modes:
        result = subprocess.run(
            ["python", *mode],
            env={**os.environ, "PYTHONPATH": "/probe/src", "OMP_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=550,
            check=False,
        )
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
        results.append(
            {
                "mode": mode,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return {"revision": REVISION, "results": results}


@app.function(image=image, gpu="B200:2", cpu=16, timeout=1800)
def distributed(profile=False, retained=False, quick=False):
    with subprocess.Popen(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "/probe/benchmark_pipeline.py",
            "--compare-sonic-fp8",
            *(["--profile"] if profile else []),
            *(["--sonic-retained-only"] if retained else []),
            *(
                [
                    "--tokens",
                    "137",
                    "--hidden",
                    "256",
                    "--inner",
                    "256",
                    "--experts",
                    "8",
                    "--topk",
                    "2",
                    "--warmup",
                    "0",
                    "--iters",
                    "1",
                ]
                if quick
                else []
            ),
        ],
        env={**os.environ, "PYTHONPATH": "/probe/src", "OMP_NUM_THREADS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:

        def terminate_group():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(1700, terminate_group)
        timer.start()
        lines = []
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                lines.append(line)
            returncode = process.wait()
        finally:
            timer.cancel()
    return {"returncode": returncode, "stdout": "".join(lines), "stderr": ""}


@app.local_entrypoint()
def main(
    uncached: bool = False,
    untuned: bool = False,
    interleaved: bool = False,
    standalone: bool = False,
    edge_tests: bool = False,
    ep: bool = False,
    profile: bool = False,
    retained: bool = False,
    quick: bool = False,
):
    import json

    if ep:
        print(json.dumps(distributed.remote(profile, retained, quick)))
        return
    print(
        json.dumps(probe.remote(uncached, untuned, interleaved, standalone, edge_tests))
    )
