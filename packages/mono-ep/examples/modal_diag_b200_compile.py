# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compile-only check of the SM100 diagnostic instantiations (CPU, no launch).

Compiles the persistent backward and forward with the original headers and with
the instrumented headers, compares the default instantiations' SASS, and checks
that the phase-trace and ablation instantiations build.
"""

import difflib
import json
import os
import re
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-diag-b200-compile")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(
        root / "vendor/deepgemm/deep_gemm/include/deep_gemm", "/probe/new/deep_gemm"
    ).add_local_dir(os.environ["MONO_EP_OLD_HEADERS"], "/probe/old")

BACKWARD = """
#include <deep_gemm/impls/sm100_bf16_mono_ep_backward.cuh>
static void instantiate() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm100_bf16_mono_ep_backward_impl<
        16384, 2048, 768, {experts}, 0, 8, 128, 128, 64, 32,
        {ring}, {stages}, 4096, 128, 128, 256, 148, {world}{diag}>);
}}
"""
FORWARD = """
#include <deep_gemm/impls/sm100_bf16_mono_ep.cuh>
using namespace deep_gemm;
static void instantiate() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mono_ep_impl<
        16384, 2048, 768, {experts}, 0, 8, 128, 128, 64, 32, {ring}, 6, 4096,
        128, 128, 256, 148, {world}, cute::numeric_limits<float>::infinity(), false{diag}>);
}}
"""


def _compile(name, source, include, workdir):
    import deep_gemm
    from torch.utils.cpp_extension import CUDA_HOME

    cu = workdir / f"{name}.cu"
    cubin = workdir / f"{name}.cubin"
    cu.write_text(source)
    result = subprocess.run(  # nosec B603
        [
            str(Path(CUDA_HOME) / "bin/nvcc"),
            "-std=c++20",
            "-arch=sm_100a",
            "--expt-relaxed-constexpr",
            "--diag-suppress=39,161,174,177,186,940",
            "--ptxas-options=-v",
            f"-I{include}",
            f"-I{Path(deep_gemm.__file__).parent / 'include'}",
            "-cubin",
            str(cu),
            "-o",
            str(cubin),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    sass = None
    if cubin.exists():
        sass = subprocess.run(  # nosec B603
            [str(Path(CUDA_HOME) / "bin/cuobjdump"), "-sass", str(cubin)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    resources = [
        line.strip()
        for line in result.stderr.splitlines()
        if "registers" in line or "spill" in line or "stack" in line
    ]
    return dict(
        returncode=result.returncode,
        stderr=result.stderr[-6000:] if result.returncode else None,
        resources=resources,
        cubin_bytes=cubin.stat().st_size if cubin.exists() else 0,
        sass_lines=len(sass.splitlines()) if sass else 0,
    ), sass


@app.function(image=image, cpu=16, memory=65536, timeout=3000)
def compile_all(world: int = 1):
    from concurrent.futures import ThreadPoolExecutor

    workdir = Path("/tmp/diag_compile")  # nosec B108
    workdir.mkdir(exist_ok=True)
    experts, ring = 16 * world, 16384
    jobs = {}
    for kernel, template in (("backward", BACKWARD), ("forward", FORWARD)):
        own = ", true" if kernel == "backward" else ", false"
        ablate_all = 127 if kernel == "backward" else 31
        variants = {
            "old_default": ("/probe/old", ""),
            "new_default": ("/probe/new", ""),
            "new_trace": ("/probe/new", f"{own}, true, 0u"),
            "new_ablate_all": ("/probe/new", f"{own}, false, {ablate_all}u"),
        }
        bits = (1, 2, 4, 8, 16, 32, 64) if kernel == "backward" else (1, 2, 4, 8, 16)
        for bit in bits:
            variants[f"new_ablate_{bit}"] = ("/probe/new", f"{own}, false, {bit}u")
        if kernel == "backward":
            variants["new_stages6"] = ("/probe/new", "")
            variants["new_stages6_trace"] = ("/probe/new", f"{own}, true, 0u")
        for name, (include, diag) in variants.items():
            stages = 6 if "stages6" in name else 2
            source = template.format(
                experts=experts, ring=ring, world=world, diag=diag, stages=stages
            )
            jobs[f"{kernel}_{name}"] = (source, include)
    with ThreadPoolExecutor(8) as pool:
        futures = {
            name: pool.submit(_compile, name, source, include, workdir)
            for name, (source, include) in jobs.items()
        }
        outputs = {name: future.result() for name, future in futures.items()}
    results = {name: out[0] for name, out in outputs.items()}

    def instructions(sass):
        # Mangled names carry the two new defaulted template arguments; compare code only.
        return [
            re.sub(r"\b_Z\w+", "<sym>", line.split("*/", 1)[-1]).strip()
            for line in sass.splitlines()
            if line.strip().startswith("/*") and "*/" in line
        ]

    for kernel in ("backward", "forward"):
        old = instructions(outputs[f"{kernel}_old_default"][1] or "")
        new = instructions(outputs[f"{kernel}_new_default"][1] or "")
        diff = list(difflib.unified_diff(old, new, lineterm="", n=0))
        results[f"{kernel}_default_sass_identical"] = bool(old) and old == new
        results[f"{kernel}_default_sass_instructions"] = [len(old), len(new)]
        results[f"{kernel}_default_sass_diff_head"] = diff[:40]
        results[f"{kernel}_default_sass_opcode_counts"] = {
            op: sum(op in line for line in new)
            for op in (
                "CCTL.IVALL",
                "LDG.E.STRONG.GPU",
                "LDG.E.STRONG.SYS",
                "ATOMS.CAST.SPIN",
                "ATOMS.ADD",
                "ATOMS.CAS",
                "STG.E.U16",
                "STG.E.128",
                "UTCHMMA",
                "UTCBAR",
                "SYNCS",
            )
        }
    return dict(world=world, results=results, validation="compile_only_not_executed")


@app.local_entrypoint()
def main(out: str, world: int = 1):
    result = compile_all.remote(world)
    result["modal_app_id"] = app.app_id
    Path(out).write_text(json.dumps(result, indent=1))
    summary = {
        k: (v if not isinstance(v, dict) else (v["returncode"], v["resources"][-1:]))
        for k, v in result["results"].items()
    }
    print(json.dumps(summary, indent=1))
