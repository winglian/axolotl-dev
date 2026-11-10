# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""H100 development timing and Nsight Compute stall attribution."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-single-rank-backward")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    source = Path(os.environ.get("MONO_EP_SOURCE_DIR", root / "src"))
    image = (
        base_image.run_commands(
            "apt-get update && apt-get install -y nsight-compute-2026.4.0"
        )
        .add_local_dir(source, "/probe/src")
        .add_local_file(
            root / "examples/probe_hopper_single_rank.py", "/probe/single_rank.py"
        )
    )


def _ncu_executable():
    candidates = sorted(Path("/opt/nvidia").glob("nsight-compute/**/ncu"))
    return str(candidates[-1]) if candidates else shutil.which("ncu")


@app.function(image=image, gpu="H100", cpu=16, memory=65536, timeout=1800)
def probe(ncu=False, trap_timeouts=True):
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        "/probe/single_rank.py",
    ]
    version = None
    if ncu:
        executable = _ncu_executable()
        if executable is None:
            candidates = sorted(Path("/opt/nvidia").glob("nsight-compute*/ncu"))
            executable = str(candidates[-1]) if candidates else None
        if executable is None:
            return dict(returncode=127, stderr="Nsight Compute is absent", stdout="")
        version = subprocess.run(
            [executable, "--version"], capture_output=True, text=True
        ).stdout
        command = [
            executable,
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--clock-control",
            "none",
            "--replay-mode",
            "application",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            "regex:sm90_fp8_mono_ep_backward_impl",
            "--launch-count",
            "1",
            "--section",
            "WarpStateStats",
            "--section",
            "SchedulerStats",
            "--section",
            "SpeedOfLight",
            "--section",
            "LaunchStats",
            "--section",
            "Occupancy",
            "--section",
            "MemoryWorkloadAnalysis",
            "--section",
            "InstructionStats",
            "--csv",
            *command,
            "--capture",
        ]
    result = subprocess.run(
        command,
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src",
            "OMP_NUM_THREADS": "1",
            "MONO_EP_SM90_FP8_TRAP_TIMEOUTS": "1" if trap_timeouts else "0",
            "MONO_EP_SM90_FP8_TRANSPORT": "1",
            "MONO_EP_SM90_FP8_COMBINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=1700,
        check=False,
    )
    return dict(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=command,
        ncu_version=version,
    )


@app.function(image=image, cpu=2, memory=4096, timeout=60)
def profiler_capabilities():
    executable = _ncu_executable()
    if executable is None:
        return dict(returncode=127, stderr="Nsight Compute is absent")
    help_text = subprocess.run(
        [executable, "--help"], capture_output=True, text=True, check=False
    ).stdout
    return dict(
        returncode=0,
        version=subprocess.run(
            [executable, "--version"], capture_output=True, text=True, check=False
        ).stdout,
        concurrent_options=[
            line
            for line in help_text.splitlines()
            if any(name in line for name in ("communicator", "lockstep", "app-range"))
        ],
        available_packages=subprocess.run(
            ["apt-cache", "search", "^nsight-compute"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout,
    )


@app.local_entrypoint()
def main(ncu: bool = False, diagnostic: bool = False, capabilities: bool = False):
    if capabilities:
        print(json.dumps(profiler_capabilities.remote()))
        return
    result = probe.remote(ncu, not diagnostic)
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Single-rank backward probe failed")
