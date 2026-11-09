# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Same-allocation Nsight Compute comparison of dispatch spin variants on one H100."""

import glob
import json
import os
import shutil
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-dispatch-spin-ncu")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_file(
            root / "examples/probe_hopper_single_rank.py", "/probe/single_rank.py"
        )
        .add_local_file(
            root / "examples/ncu_dump_source_counters.py", "/probe/ncu_dump.py"
        )
        .add_local_file(
            root / "examples/aggregate_ncu_stalls.py", "/probe/aggregate.py"
        )
    )

KERNEL = "regex:sm90_fp8_mono_ep_backward_impl"
SECTIONS = (
    "SourceCounters",
    "WarpStateStats",
    "SpeedOfLight",
    "LaunchStats",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Tables",
)
PROBE = [
    "python",
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nproc-per-node=1",
    "/probe/single_rank.py",
]
WORK = "/tmp/dispatch_spin_ncu"  # nosec B108


def _ncu():
    for candidate in ("/usr/local/cuda/bin/ncu", shutil.which("ncu")):
        if candidate and os.path.exists(candidate):
            return candidate
    for candidate in sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu")):
        if "/target/" not in candidate:
            return candidate
    return None


def _run(command, env, timeout):
    result = subprocess.run(  # nosec B603
        command, env=env, capture_output=True, text=True, timeout=timeout, check=False
    )
    return dict(
        returncode=result.returncode,
        stdout=result.stdout[-20000:],
        stderr=result.stderr[-20000:],
    )


@app.function(image=image, gpu="H100!", cpu=16, memory=65536, timeout=4 * 3600)
def profile(variants):
    device = subprocess.run(  # nosec B603 B607
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if "H100" not in device:
        return dict(device=device, error="not an H100 allocation")
    executable = _ncu()
    out = dict(device=device, ncu_executable=executable, variants={})
    for name, extra in variants.items():
        work = Path(WORK) / name
        work.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "PYTHONPATH": "/probe/src",
            "OMP_NUM_THREADS": "1",
            "MONO_EP_SM90_FP8_TRAP_TIMEOUTS": "1",
            "MONO_EP_SM90_FP8_TRANSPORT": "1",
            "MONO_EP_SM90_FP8_COMBINE": "1",
            "DG_JIT_CACHE_DIR": str(work / "cache"),
            "DG_JIT_WITH_LINEINFO": "1",
            **extra,
        }
        record = dict(env=extra)
        record["timing"] = _run(PROBE, env, 1800)
        report = work / "report"
        command = [executable, "--target-processes", "all", "--profile-from-start"]
        command += ["off", "--clock-control", "none", "--replay-mode", "application"]
        command += ["--kernel-name-base", "demangled", "--kernel-name", KERNEL]
        command += ["--launch-count", "1", "--import-source", "yes", "--export"]
        command += [str(report), "--force-overwrite"]
        for section in SECTIONS:
            command += ["--section", section]
        command += [*PROBE, "--capture"]
        ncu = _run(command, env, 3 * 3600)
        record["ncu"] = dict(returncode=ncu["returncode"], stderr=ncu["stderr"][-4000:])
        rep = str(report) + ".ncu-rep"
        if not os.path.exists(rep):
            record["ncu"]["stdout"] = ncu["stdout"][-4000:]
            out["variants"][name] = record
            continue
        sass_csv = work / "source_sass.csv"
        sass_csv.write_text(
            subprocess.run(  # nosec B603
                [executable, "-i", rep, "--page", "source", "--csv"]
                + ["--print-source", "sass"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
        )
        dump = _run(
            ["python", "/probe/ncu_dump.py", rep, str(work / "per_pc.json")],
            os.environ.copy(),
            1800,
        )
        record["dump"] = dict(
            returncode=dump["returncode"], stderr=dump["stderr"][-2000:]
        )
        cubins = [
            c
            for c in glob.glob(str(work / "cache/cache/*/kernel.cubin"))
            if "mono_ep_backward" in Path(c).with_name("kernel.cu").read_text()
        ]
        record["cubins"] = cubins
        if cubins and dump["returncode"] == 0:
            disasm = work / "nvdisasm.txt"
            disasm.write_text(
                subprocess.run(  # nosec B603
                    ["/usr/local/cuda/bin/nvdisasm", "-gi", "-c", "-hex", cubins[0]],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
            )
            agg = _run(
                [
                    "python",
                    "/probe/aggregate.py",
                    str(work / "per_pc.json"),
                    str(disasm),
                    str(sass_csv),
                    str(work / "stalls.json"),
                    "--top",
                    "25",
                ],
                os.environ.copy(),
                1800,
            )
            record["aggregate"] = dict(
                returncode=agg["returncode"], stderr=agg["stderr"][-2000:]
            )
            if agg["returncode"] == 0:
                record["stalls"] = json.loads((work / "stalls.json").read_text())
            scalars = json.loads((work / "per_pc.json").read_text())["scalars"]
            record["scalars"] = {
                k: v
                for k, v in scalars.items()
                if any(
                    tag in k
                    for tag in (
                        "hit_rate",
                        "gpu__time_duration",
                        "l1tex__t_sectors",
                        "l1tex__t_requests",
                        "lts__t_sector",
                        "smsp__average_warp",
                        "smsp__pcsamp_sample_count",
                        "sm__throughput",
                        "l1tex__throughput",
                        "smsp__inst_executed.sum",
                    )
                )
            }
        out["variants"][name] = record
    return out


@app.local_entrypoint()
def main(out: str, variants: str):
    parsed = {}
    for spec in variants.split(","):
        name, _, env = spec.partition(":")
        parsed[name] = dict(
            item.partition("=")[::2] for item in filter(None, env.split("+"))
        )
    data = profile.remote(parsed)
    data["modal_app_id"] = app.app_id
    Path(out).write_text(json.dumps(data, indent=1))
    print(json.dumps({k: data[k] for k in ("device", "modal_app_id")}))
