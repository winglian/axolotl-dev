# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Nsight Compute (kernel replay) of the single-rank SM100 persistent kernels on one B200.

Profiles the native backward and forward entries (probe ``entry`` mode) and
the plain grouped BF16 GEMMs on the backward's own operands (probe ``floors``
mode) for contrast, with the memory-workload sections (L2 hit rate, DRAM and
L2 bytes, per-source L2 tables) and warp-state statistics. A second pass asks
for explicit L2/DRAM/TMA metrics and is allowed to fail on names this chip
does not expose. Raw per-kernel metrics are returned for offline reduction.
"""

import csv
import io
import json
import os
import shutil
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-ncu-b200")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(
            root / "vendor/deepgemm/deep_gemm/include/deep_gemm",
            "/probe/headers/deep_gemm",
        )
        .add_local_dir(root / "src", "/probe/src")
        .add_local_file(root / "examples/probe_diag_b200.py", "/probe/diag.py")
    )

SECTIONS = (
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Tables",
    "MemoryWorkloadAnalysis_Chart",
    "WarpStateStats",
    "SchedulerStats",
    "ComputeWorkloadAnalysis",
    "LaunchStats",
    "Occupancy",
)
METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sector_op_read_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__m_xbar2l1tex_read_bytes.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
)


def _ncu():
    for candidate in ("/usr/local/cuda/bin/ncu", shutil.which("ncu")):
        if candidate and os.path.exists(candidate) and "/target/" not in candidate:
            return candidate
    return None


def _parse_raw(text):
    lines = [line for line in text.splitlines() if line.startswith('"')]
    if len(lines) < 3:
        return []
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    header, units = rows[0], rows[1]
    kernels = []
    for row in rows[2:]:
        kernels.append(
            {
                name: (value, unit)
                for name, unit, value in zip(header, units, row, strict=False)
            }
        )
    return kernels


def _profile(executable, probe_args, kernel_regex, count, extra, env):
    command = [
        executable,
        "--target-processes",
        "all",
        "--replay-mode",
        "kernel",
        "--clock-control",
        "none",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        f"regex:{kernel_regex}",
        "--launch-count",
        str(count),
        *extra,
        "--csv",
        "--page",
        "raw",
        "python",
        "/probe/diag.py",
        *probe_args,
    ]
    result = subprocess.run(  # nosec B603
        command,
        env=env,
        capture_output=True,
        text=True,
        timeout=3000,
        check=False,
    )
    return dict(
        command=command,
        returncode=result.returncode,
        kernels=_parse_raw(result.stdout),
        stdout_tail=result.stdout[-3000:],
        stderr_tail=result.stderr[-6000:],
    )


@app.function(image=image, gpu="B200", cpu=16, memory=65536, timeout=7200)
def profile_variants(variants):
    """Metrics pass on the third launch of each backward variant (probe ``entry`` syntax)."""
    import deep_gemm
    import torch

    name = torch.cuda.get_device_name()
    if "B200" not in name or torch.cuda.device_count() != 1:
        return dict(returncode=2, error=f"wrong device {name}")
    installed = Path(deep_gemm.__file__).parent / "include/deep_gemm"
    shutil.copytree("/probe/headers/deep_gemm", installed, dirs_exist_ok=True)
    executable = _ncu()
    if executable is None:
        return dict(returncode=127, error="ncu not found")
    env = {
        **os.environ,
        "PYTHONPATH": "/probe/src",
        "OMP_NUM_THREADS": "1",
        "DG_JIT_CACHE_DIR": "/tmp/dg_jit_cache",  # nosec B108
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29591",
    }
    sections = [arg for s in SECTIONS for arg in ("--section", s)]
    runs = {}
    for spec in variants:
        entry = ["entry", "--variants", spec, "--warmup", "2", "--iters", "1"]
        for label, extra in (
            ("metrics", ["--metrics", ",".join(METRICS)]),
            ("sections", sections),
        ):
            runs[f"{spec.split(':')[0]}_{label}"] = _profile(
                executable,
                entry,
                "sm100_bf16_mono_ep_backward_impl",
                1,
                ["--launch-skip", "2", *extra],
                env,
            )
    return dict(
        returncode=max(r["returncode"] for r in runs.values()),
        device=name,
        torch=torch.__version__,
        ncu=executable,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
        variants=variants,
        runs=runs,
    )


@app.function(image=image, gpu="B200", cpu=16, memory=65536, timeout=7200)
def profile():
    import deep_gemm
    import torch

    name = torch.cuda.get_device_name()
    if "B200" not in name or torch.cuda.device_count() != 1:
        return dict(returncode=2, error=f"wrong device {name}")
    installed = Path(deep_gemm.__file__).parent / "include/deep_gemm"
    shutil.copytree("/probe/headers/deep_gemm", installed, dirs_exist_ok=True)
    executable = _ncu()
    if executable is None:
        return dict(returncode=127, error="ncu not found")
    env = {
        **os.environ,
        "PYTHONPATH": "/probe/src",
        "OMP_NUM_THREADS": "1",
        "DG_JIT_CACHE_DIR": "/tmp/dg_jit_cache",  # nosec B108
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29591",
    }
    sections = [arg for s in SECTIONS for arg in ("--section", s)]
    entry = ["entry", "--variants", "bwd_base:bwd:,fwd_base:fwd:"]
    entry += ["--warmup", "2", "--iters", "1", "--rounds", "1"]
    runs = {
        "native_sections": _profile(
            executable,
            entry,
            "sm100_bf16_mono_ep_backward_impl|sm100_bf16_mono_ep_impl",
            6,
            sections,
            env,
        ),
        "native_metrics": _profile(
            executable,
            entry,
            "sm100_bf16_mono_ep_backward_impl|sm100_bf16_mono_ep_impl",
            6,
            ["--metrics", ",".join(METRICS)],
            env,
        ),
        "floors_sections": _profile(
            executable,
            ["floors", "--warmup", "1", "--iters", "1"],
            "sm100_bf16_gemm",
            10,
            sections,
            env,
        ),
    }
    version = subprocess.run(  # nosec B603
        [executable, "--version"], capture_output=True, text=True, check=False
    ).stdout
    return dict(
        returncode=max(r["returncode"] for r in runs.values()),
        device=name,
        torch=torch.__version__,
        ncu=executable,
        ncu_version=version,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
        runs=runs,
    )


@app.local_entrypoint()
def main(out: str, variants: str = ""):
    """``variants``: backward specs in the probe's ``name:bwd:env`` syntax separated by ``;``."""
    result = (
        profile_variants.remote([v for v in variants.split(";") if v])
        if variants
        else profile.remote()
    )
    result["modal_app_id"] = app.app_id
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("returncode", "device", "modal_app_id", "ncu_version")
            }
        )
    )
    for label, run in result.get("runs", {}).items():
        print(label, run["returncode"], len(run["kernels"]), "kernels")
        if run["returncode"] or not run["kernels"]:
            print(run["stderr_tail"][-3000:])
