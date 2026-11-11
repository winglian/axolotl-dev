# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Modal runner for the bounded Hopper workspace (MONO_EP_WORKSPACE_BOUND) on 8xH100 / 8xH200: parity and
same-process backward timing (workspace_parity.py), resident memory per mode (workspace_memory.py), the
checkpointed-layer record probe (workspace_layers.py) and the FSDP2 dp_shard x EP validator, for the SM90
BF16 and blockwise FP8 backwards, on the NVFP4 LoRA image."""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-mono-ep-bounded-hopper")
image = None
if modal.is_local():
    from modal_nvfp4_lora import image
ENV = {
    "PYTHONPATH": "/probe/src:/probe/examples",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1",
}
PREFIXES = (
    "WSPAR_JSON ",
    "WSMEM_JSON ",
    "WSLAYERS_JSON ",
    "FSDP2_CASE ",
    "FSDP2_JSON ",
)


def _run(args, nproc, extra_env, timeout=1800):
    import shutil

    import deep_gemm

    shutil.copytree(
        "/probe/include",
        Path(deep_gemm.__file__).parent / "include",
        dirs_exist_ok=True,
    )
    env = {**os.environ, **ENV, **extra_env}
    cmd = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc}",
        *args,
    ]
    lines = []
    with subprocess.Popen(  # nosec B603
        cmd,
        cwd="/probe/examples",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait(timeout=timeout)
    payloads = [
        json.loads(line[len(p) :])
        for line in lines
        for p in PREFIXES
        if line.startswith(p)
    ]
    return dict(
        args=" ".join(args),
        env=extra_env,
        returncode=proc.returncode,
        payloads=payloads,
        tail="".join(lines[-60:]),
    )


def _suite(shapes, dtypes, parts):
    runs = []
    for dtype in dtypes.split(","):
        for shape in shapes.split(","):
            # The SM90 BF16 forward does not compile at hidden 5120
            if dtype == "bf16" and shape == "beam":
                continue
            if "parity" in parts:
                runs.append(
                    _run(
                        ["workspace_parity.py", "--shape", shape, "--dtype", dtype],
                        8,
                        {},
                    )
                )
            if "memory" in parts:
                for mode in ("max", "auto"):
                    runs.append(
                        _run(
                            ["workspace_memory.py", "--shape", shape, "--dtype", dtype],
                            8,
                            {"MONO_EP_WORKSPACE_BOUND": mode},
                        )
                    )
        if "layers" in parts:
            runs.append(
                _run(["workspace_layers.py", "--shape", "30b", "--dtype", dtype], 8, {})
            )
    if "fsdp2" in parts:
        for mode in ("auto", "max"):
            runs.append(
                _run(
                    [
                        "validate_fsdp2_weight_cache.py",
                        "--ep",
                        "2",
                        "--reserved-sms",
                        "24",
                        "--steps",
                        "3",
                    ],
                    4,
                    {"MONO_EP_WORKSPACE_BOUND": mode, "NCCL_MAX_CTAS": "16"},
                    2400,
                )
            )
    return dict(returncode=max(r["returncode"] for r in runs), runs=runs)


@app.function(image=image, gpu="H100:8", cpu=32, memory=262144, timeout=10800)
def suite_h100(shapes: str, dtypes: str, parts: str):
    return _suite(shapes, dtypes, parts)


@app.function(image=image, gpu="H200:8", cpu=32, memory=262144, timeout=10800)
def suite_h200(shapes: str, dtypes: str, parts: str):
    return _suite(shapes, dtypes, parts)


@app.local_entrypoint()
def main(
    gpu: str = "h100",
    shapes: str = "30b,235b,beam",
    dtypes: str = "bf16,blockwise",
    parts: str = "parity,memory,layers,fsdp2",
    out: str = "",
):
    fn = suite_h100 if gpu == "h100" else suite_h200
    result = fn.remote(shapes, dtypes, parts)
    result["modal_app_id"] = app.app_id
    if out:
        Path(out).write_text(json.dumps(result, indent=1) + "\n")
    for run in result["runs"]:
        print(run["returncode"], run["args"], run["env"])
    if result["returncode"]:
        raise SystemExit("a run failed; see the per-run tails in the output file")
