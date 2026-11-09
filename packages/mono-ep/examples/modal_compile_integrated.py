# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""ptxas register/spill report for the integrated SM90 FP8 backward and forward (compile only)."""

import json
import os
import re
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("mono-ep-compile-integrated")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(
        root / "vendor/deepgemm/deep_gemm/include", "/probe/native"
    )

# Single-rank runtime instantiations (trap timeouts, FP8 transport + combine, cached dZ).
BACKWARD = (
    "17280, 2048, 768, 16, 0, 8, 128, 128, 128, 32, 44160, {stages}, 2048, "
    "128, 128, 256, 132, 1, true, true, true, false, true, true{suffix}"
)
# EP8 per-GPU forward with the original W1 layout.
FORWARD = (
    "16384, 2048, 768, 16, 0, 8, 128, 128, 128, 32, 44160, {stages}, 2048, 128, 128, 256, "
    "132, 1, 0x7f7fffffu, false, false, true{suffix}"
)


@app.function(image=image, cpu=4, memory=16384, timeout=1800)
def compile_one(record):
    import importlib.util

    from torch.utils.cpp_extension import CUDA_HOME

    work = Path("/tmp/integrated") / record["name"]  # nosec B108
    work.mkdir(parents=True, exist_ok=True)
    direction = record["direction"]
    template = BACKWARD if direction == "backward" else FORWARD
    args = template.format(stages=record["stages"], suffix=record["suffix"])
    source = work / "probe.cu"
    source.write_text(
        "#define DG_TRAP_ONLY_TIMEOUTS 1\n"
        + "".join(f"#define {d}\n" for d in record["defines"])
        + f"#include <deep_gemm/impls/sm90_fp8_mono_ep_{direction}.cuh>\n"
        + "static void instantiate() {\n"
        + "    auto ptr = reinterpret_cast<void*>("
        + f"&deep_gemm::sm90_fp8_mono_ep_{direction}_impl<{args}>);\n}}\n"
    )
    cuda_bin = Path(CUDA_HOME) / "bin"
    root_include = Path(importlib.util.find_spec("deep_gemm").origin).parent / "include"
    command = [
        str(cuda_bin / "nvcc"),
        "--gpu-architecture=sm_90a",
        "-O3",
        "--ptxas-options=--verbose",
        "--ptxas-options=--register-usage-level=10",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--diag-suppress=39,161,174,177,186,940",
        "-I/probe/native",
        f"-I{root_include}",
        "-cubin",
        str(source),
        "-o",
        str(work / "probe.cubin"),
    ]
    result = subprocess.run(  # nosec B603
        command, capture_output=True, text=True, timeout=1500, check=False
    )
    stack = re.findall(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
        result.stderr,
    )
    return dict(
        **record,
        instantiation=args,
        returncode=result.returncode,
        nvcc=subprocess.run(  # nosec B603
            [str(cuda_bin / "nvcc"), "--version"], capture_output=True, text=True
        ).stdout.splitlines()[-1:],
        ptxas_registers=[
            int(x) for x in re.findall(r"Used (\d+) registers", result.stderr)
        ],
        ptxas_stack_spill_store_load=[list(map(int, s)) for s in stack],
        c7507="C7507" in result.stderr,
        c7510="C7510" in result.stderr,
        warnings=[ln for ln in result.stderr.splitlines() if "warning" in ln][:20],
        stderr_tail=result.stderr[-4000:],
    )


@app.local_entrypoint()
def main(out: str, variants: str):
    """variants: ';'-separated name|direction|stages|DEFINE+DEFINE|template suffix."""
    records = []
    for spec in variants.split(";"):
        name, direction, stages, defines, suffix = spec.split("|")
        records.append(
            dict(
                name=name,
                direction=direction,
                stages=int(stages),
                defines=[d for d in defines.split("+") if d],
                suffix=suffix,
            )
        )
    results = list(compile_one.map(records))
    payload = dict(
        modal_app_id=os.environ.get("MODAL_APP_ID") or app.app_id,
        results=results,
    )
    Path(out).write_text(json.dumps(payload, indent=1))
    for r in results:
        print(
            r["name"],
            r["returncode"],
            r["ptxas_registers"],
            r["ptxas_stack_spill_store_load"],
            "C7507" if r["c7507"] else "",
            "C7510" if r["c7510"] else "",
        )
        if r["returncode"]:
            print(r["stderr_tail"])
