# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compile-only check of the SM100 round-2 header changes (CPU, no launch).

Builds the persistent backward (6 stages, options 11) and forward at the EP8
geometry from the previous headers (``MONO_EP_OLD_HEADERS``) and the
current ones, checks that the default and phase-trace instantiations are
SASS-identical, and that the distributed-trace instantiation (diag bit 1024)
and any extra variants build without spills.
"""

import json
import os
import re
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-b200-r2-compile")
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
        {ring}, {stages}, {pull}, 128, 128, 256, 148, {world}, true, {trace}, {ablate}u, {options}u>);
}}
"""
FORWARD = """
#include <deep_gemm/impls/sm100_bf16_mono_ep.cuh>
using namespace deep_gemm;
static void instantiate() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mono_ep_impl<
        16384, 2048, 768, {experts}, 0, 8, 128, 128, 64, 32, {ring}, {stages}, {pull},
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
        timeout=1500,
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


@app.function(image=image, cpu=16, memory=65536, timeout=3600)
def compile_all(world: int, extra: list):
    from concurrent.futures import ThreadPoolExecutor

    workdir = Path("/tmp/r2_compile")  # nosec B108
    workdir.mkdir(exist_ok=True)
    experts, ring = 16 * world, 16384

    def backward(trace, ablate, options=11, pull=4096, stages=6):
        return BACKWARD.format(
            experts=experts,
            ring=ring,
            world=world,
            trace=trace,
            ablate=ablate,
            options=options,
            pull=pull,
            stages=stages,
        )

    def forward(diag, pull=4096, stages=6):
        return FORWARD.format(
            experts=experts, ring=ring, world=world, diag=diag, pull=pull, stages=stages
        )

    jobs = {}
    for side in ("old", "new"):
        include = f"/probe/{side}"
        jobs[f"backward_{side}_default"] = (backward("false", 0), include)
        jobs[f"backward_{side}_trace"] = (backward("true", 0), include)
        jobs[f"forward_{side}_default"] = (forward(""), include)
        jobs[f"forward_{side}_trace"] = (forward(", false, true, 0u"), include)
    jobs["backward_new_dist_trace"] = (backward("true", 1024), "/probe/new")
    jobs["forward_new_dist_trace"] = (forward(", false, true, 1024u"), "/probe/new")
    for name, kernel, trace, ablate, options, pull, stages in extra:
        source = (
            backward(trace, ablate, options, pull, stages)
            if kernel == "backward"
            else forward(f", false, {trace}, {ablate}u", pull, stages)
        )
        jobs[name] = (source, "/probe/new")
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

    for kernel in ("backward", "forward"):
        for variant in ("default", "trace"):
            old = instructions(outputs[f"{kernel}_old_{variant}"][1])
            new = instructions(outputs[f"{kernel}_new_{variant}"][1])
            results[f"{kernel}_{variant}_sass_identical"] = bool(old) and old == new
    return dict(world=world, results=results, validation="compile_only_not_executed")


@app.local_entrypoint()
def main(out: str, world: int = 8, extra: str = ""):
    """``extra``: ``name:kernel:trace:ablate:options[:pull:stages]`` items separated by commas."""
    if not os.environ.get("MONO_EP_OLD_HEADERS"):
        raise ValueError("set MONO_EP_OLD_HEADERS to the previous include tree")
    parsed = []
    for item in filter(None, extra.split(",")):
        name, kernel, trace, ablate, options, *rest = item.split(":")
        pull, stages = (int(rest[0]), int(rest[1])) if rest else (4096, 6)
        parsed.append((name, kernel, trace, int(ablate), int(options), pull, stages))
    result = compile_all.remote(world, parsed)
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
