# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""B200 runners for the SM100 MXFP8 persistent backward (``MONO_EP_SM100_BACKWARD_PRECISION``).

``--mode validate`` runs ``validate_sm100_mxfp8_backward.py`` on ``--world`` (1 or 2) B200s; every
mode asserts the device names and records the app id, torch and CUDA versions. The worktree's
kernel headers are overlaid onto the installed DeepGEMM include tree, so header-only changes do
not rebuild the image (launcher changes do).
"""

import json
import os
import shutil
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App(os.environ.get("MONO_EP_MODAL_APP", "axolotl-mono-ep-mxfp8-validate"))
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
        .add_local_dir(root / "examples", "/probe/examples")
    )


def _prepare(world):
    import deep_gemm
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != world or not all("B200" in n for n in names):
        raise RuntimeError(f"expected {world} B200, got {names}")
    installed = Path(deep_gemm.__file__).parent / "include/deep_gemm"
    shutil.copytree("/probe/headers/deep_gemm", installed, dirs_exist_ok=True)
    cache = Path("/tmp/dg_jit_cache")  # nosec B108
    shutil.rmtree(cache, ignore_errors=True)
    return dict(
        devices=names,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        jit_cache=str(cache),
    )


def _torchrun(world, script, args, env=None, timeout=3600):
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        f"/probe/examples/{script}",
        *args,
    ]
    lines = []
    with subprocess.Popen(  # nosec B603
        command,
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src:/probe/examples",
            "OMP_NUM_THREADS": "1",
            "DG_JIT_CACHE_DIR": "/tmp/dg_jit_cache",  # nosec B108
            **(env or {}),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        code = process.wait(timeout=timeout)
    return code, lines


def _validate(world, args):
    info = _prepare(world)
    code, lines = _torchrun(world, "validate_sm100_mxfp8_backward.py", args)
    records = [json.loads(line[6:]) for line in lines if line.startswith("MXVAL ")]
    parity = [json.loads(line[9:]) for line in lines if line.startswith("MXPARITY ")]
    jit = sorted(p.name for p in Path("/tmp/dg_jit_cache").glob("**/*") if p.is_dir())  # nosec B108
    return dict(
        **info,
        returncode=code,
        records=records,
        quantizer_parity=parity,
        jit_kernels=jit[:50],
        tail="".join(lines[-80:]) if code else "",
    )


@app.function(image=image, gpu="B200", cpu=16, memory=131072, timeout=5400)
def validate_1(args):
    return _validate(1, args)


def _trace(world, args):
    info = _prepare(world)
    code, lines = _torchrun(world, "benchmark_ep_trace_b200.py", args)
    payload = [
        json.loads(line[len("EPTRACE_JSON ") :])
        for line in lines
        if line.startswith("EPTRACE_JSON ")
    ]
    return dict(
        **info,
        returncode=code,
        result=payload[-1] if payload else None,
        tail="".join(lines[-80:]) if code else "",
    )


@app.function(image=image, gpu="B200", cpu=16, memory=131072, timeout=5400)
def trace_1(args):
    return _trace(1, args)


@app.function(image=image, gpu="B200:2", cpu=32, memory=262144, timeout=5400)
def validate_2(args):
    return _validate(2, args)


PAIRED_VARIANTS = (
    "bf16:MONO_EP_SM100_BACKWARD_PRECISION=bf16,"
    "mxfp8:MONO_EP_SM100_BACKWARD_PRECISION=mxfp8"
)


@app.function(image=image, gpu="B200:8", cpu=64, memory=524288, timeout=3 * 3600)
def paired_8(routings):
    """bf16 vs mxfp8 backward interleaved per iteration (5 warmups, 15 samples), NCCL control,
    weights bumped every step, compiled checkpointed layer; then one EP8 phase trace."""
    info = _prepare(8)
    runs = {}
    for routing in routings:
        code, lines = _torchrun(
            8,
            "benchmark_persistent_training.py",
            [
                "--recompute",
                "--peer-backward",
                "--compare-native",
                "--compiled",
                "--compute-dtype",
                "bf16",
                "--bump-versions",
                "--routing",
                routing,
                "--variants",
                PAIRED_VARIANTS,
            ],
        )
        payload = [
            json.loads(line) for line in lines if line.startswith('{"benchmark"')
        ]
        runs[routing] = dict(
            returncode=code,
            result=payload[-1] if payload else None,
            tail="".join(lines[-80:]) if code else "",
        )
    code, lines = _torchrun(
        8,
        "benchmark_ep_trace_b200.py",
        [
            "--variants",
            PAIRED_VARIANTS,
            "--keys",
            "bwd/bump",
            "--rounds",
            "2",
            "--iters",
            "8",
            "--trace-iters",
            "6",
        ],
    )
    trace = [
        json.loads(line[len("EPTRACE_JSON ") :])
        for line in lines
        if line.startswith("EPTRACE_JSON ")
    ]
    return dict(
        **info,
        returncode=max([r["returncode"] for r in runs.values()] + [code]),
        runs=runs,
        trace=trace[-1] if trace else None,
        trace_returncode=code,
        tail="".join(lines[-80:]) if code else "",
    )


@app.local_entrypoint()
def main(out: str, mode: str = "validate", world: int = 1, args: str = ""):
    parsed = args.split()
    if mode == "validate":
        result = (validate_1 if world == 1 else validate_2).remote(parsed)
    elif mode == "trace" and world == 1:
        result = trace_1.remote(parsed)
    elif mode == "paired":
        result = paired_8.remote(parsed or ["balanced", "skewed"])
    else:
        raise ValueError(mode)
    result["modal_app_id"] = app.app_id
    result["mode"] = mode
    Path(out).write_text(json.dumps(result, indent=1))
    print(
        json.dumps({k: v for k, v in result.items() if k != "tail"}, indent=1)[:20000]
    )
    if result["returncode"]:
        print(result["tail"])
        raise RuntimeError("run failed")
