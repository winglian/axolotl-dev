# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compile-only check of the SM100 backward round-1 variants (CPU, no launch).

Builds the persistent backward for each kernel-option bitmask and pipeline
shape at the EP8 per-GPU geometry, reports registers/spills and selected
opcode counts, and checks that options 0 at 6 stages is SASS-identical to the
previous headers (``MONO_EP_OLD_HEADERS``) at 6 stages.
"""

import json
import os
import re
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-b200-r1-compile")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(
        root / "vendor/deepgemm/deep_gemm/include/deep_gemm", "/probe/new/deep_gemm"
    ).add_local_dir(os.environ["MONO_EP_OLD_HEADERS"], "/probe/old")

OLD = """
#include <deep_gemm/impls/sm100_bf16_mono_ep_backward.cuh>
static void instantiate() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm100_bf16_mono_ep_backward_impl<
        16384, 2048, 768, {experts}, 0, 8, 128, 128, {bk}, 32,
        {ring}, {stages}, 4096, 128, 128, 256, 148, {world}>);
}}
"""
NEW = """
#include <deep_gemm/impls/sm100_bf16_mono_ep_backward.cuh>
static void instantiate() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm100_bf16_mono_ep_backward_impl<
        16384, 2048, 768, {experts}, 0, 8, 128, 128, {bk}, 32,
        {ring}, {stages}, 4096, 128, 128, 256, 148, {world}, true, {trace}, 0u, {options}u>);
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
        stderr=result.stderr[:8000] if result.returncode else None,
        resources=resources,
        cubin_bytes=cubin.stat().st_size if cubin.exists() else 0,
        sass_lines=len(sass.splitlines()) if sass else 0,
    ), sass


@app.function(image=image, cpu=16, memory=65536, timeout=3000)
def compile_all(world: int = 1, options: str = "0,1,2,3,4,8,11,15"):
    from concurrent.futures import ThreadPoolExecutor

    workdir = Path("/tmp/r1_compile")  # nosec B108
    workdir.mkdir(exist_ok=True)
    experts, ring = 16 * world, 16384
    jobs = {
        "old_stages6": (
            OLD.format(experts=experts, ring=ring, world=world, bk=64, stages=6),
            "/probe/old",
        )
    }
    for opt in [int(o) for o in options.split(",")]:
        for bk, stages in ((64, 6), (128, 3)):
            for trace in (
                ("false", "true") if opt in (0, 3) and bk == 64 else ("false",)
            ):
                name = f"opt{opt}_bk{bk}_s{stages}" + (
                    "_trace" if trace == "true" else ""
                )
                jobs[name] = (
                    NEW.format(
                        experts=experts,
                        ring=ring,
                        world=world,
                        bk=bk,
                        stages=stages,
                        trace=trace,
                        options=opt,
                    ),
                    "/probe/new",
                )
    with ThreadPoolExecutor(8) as pool:
        futures = {
            name: pool.submit(_compile, name, source, include, workdir)
            for name, (source, include) in jobs.items()
        }
        outputs = {name: future.result() for name, future in futures.items()}
    results = {name: out[0] for name, out in outputs.items()}

    def instructions(sass):
        return [
            re.sub(r"\b_Z\w+", "<sym>", line.split("*/", 1)[-1]).strip()
            for line in (sass or "").splitlines()
            if line.strip().startswith("/*") and "*/" in line
        ]

    old = instructions(outputs["old_stages6"][1])
    new = instructions(outputs["opt0_bk64_s6"][1])
    results["opt0_sass_identical_to_previous_stages6"] = bool(old) and old == new
    for name, (_, sass) in outputs.items():
        code = instructions(sass)
        results[name]["opcode_counts"] = {
            op: sum(op in line for line in code)
            for op in (
                "STG.E.U16",
                "STG.E.128",
                "STS.U16",
                "LDS.128",
                "UTCHMMA",
                "LDTM",
            )
        }
    return dict(world=world, results=results, validation="compile_only_not_executed")


@app.local_entrypoint()
def main(out: str, world: int = 1, options: str = "0,1,2,3,4,8,11,15"):
    if not os.environ.get("MONO_EP_OLD_HEADERS"):
        raise ValueError("set MONO_EP_OLD_HEADERS to the previous include tree")
    result = compile_all.remote(world, options)
    result["modal_app_id"] = app.app_id
    Path(out).write_text(json.dumps(result, indent=1))
    summary = {
        k: (v if not isinstance(v, dict) else (v["returncode"], v["resources"][-1:]))
        for k, v in result["results"].items()
    }
    print(json.dumps(summary, indent=1))
    if any(isinstance(v, dict) and v["returncode"] for v in result["results"].values()):
        for k, v in result["results"].items():
            if isinstance(v, dict) and v["returncode"]:
                print(k, v["stderr"][:6000])
        raise RuntimeError("compile failed")
