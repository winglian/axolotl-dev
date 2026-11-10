# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compile-only gate for the SM100 in-kernel dW option (CPU, no launch).

Builds the persistent backward at the EP8 contract geometry (6 stages, 8 KB
pulls) from the previous headers (``MONO_EP_OLD_HEADERS``) and the current
ones, checks that the default (options 11) and phase-trace instantiations are
SASS-identical, and reports ptxas resources (registers, spills, stack) for the
default and in-kernel dW (options 11 | 512) instantiations, plus any extra
variants. Building the image also compiles the vendored DeepGEMM extension.
"""

import json
import os
import re
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-mono-ep-inkernel-dw-compile")
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
        16384, {hidden}, {inner}, {experts}, 0, 8, 128, 128, 64, 32,
        {ring}, {stages}, {pull}, 128, 128, 256, 148, {world}, true, {trace}, {ablate}u, {options}u>);
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
            "-O3",
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

    import torch

    workdir = Path("/tmp/inkernel_dw_compile")  # nosec B108
    workdir.mkdir(exist_ok=True)

    def backward(trace, options, hidden=2048, inner=768, experts=16, pull=8192):
        return BACKWARD.format(
            hidden=hidden,
            inner=inner,
            experts=experts * world,
            ring=16384,
            world=world,
            trace=trace,
            ablate=0,
            options=options,
            pull=pull,
            stages=6,
        )

    jobs = {}
    for side in ("old", "new"):
        include = f"/probe/{side}"
        jobs[f"backward_{side}_default"] = (backward("false", 11), include)
        jobs[f"backward_{side}_trace"] = (backward("true", 11), include)
    jobs["backward_new_inkernel_dw"] = (backward("false", 523), "/probe/new")
    jobs["backward_new_inkernel_dw_trace"] = (backward("true", 523), "/probe/new")
    for name, trace, options, hidden, inner, experts, pull in extra:
        jobs[name] = (
            backward(trace, options, hidden, inner, experts, pull),
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

    for variant in ("default", "trace"):
        old = instructions(outputs[f"backward_old_{variant}"][1])
        new = instructions(outputs[f"backward_new_{variant}"][1])
        results[f"backward_{variant}_sass_identical"] = bool(old) and old == new
        results[f"backward_{variant}_sass_raw_identical"] = (
            outputs[f"backward_old_{variant}"][1]
            == outputs[f"backward_new_{variant}"][1]
        )
    # Local-memory accesses with a window of SASS around each, to locate spills
    for name, (_, sass) in outputs.items():
        if "inkernel_dw" not in name or not sass:
            continue
        lines = sass.splitlines()
        hits = [i for i, line in enumerate(lines) if re.search(r"\b(STL|LDL)\b", line)]
        results[name]["local_memory_sites"] = [
            "\n".join(lines[max(0, i - 25) : i + 3]) for i in hits[:12]
        ]
    return dict(
        world=world,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        results=results,
        validation="compile_only_not_executed",
    )


@app.local_entrypoint()
def main(out: str, world: int = 8, extra: str = ""):
    """``extra``: ``name:trace:options:hidden:inner:experts:pull`` items separated by commas."""
    if not os.environ.get("MONO_EP_OLD_HEADERS"):
        raise ValueError("set MONO_EP_OLD_HEADERS to the previous include tree")
    parsed = []
    for item in filter(None, extra.split(",")):
        name, trace, options, hidden, inner, experts, pull = item.split(":")
        parsed.append(
            (
                name,
                trace,
                int(options),
                int(hidden),
                int(inner),
                int(experts),
                int(pull),
            )
        )
    result = compile_all.remote(world, parsed)
    result["modal_app_id"] = app.app_id
    Path(out).write_text(json.dumps(result, indent=1))
    summary = {
        k: (v if not isinstance(v, dict) else (v["returncode"], v["resources"][-3:]))
        for k, v in result["results"].items()
    }
    print(json.dumps(summary, indent=1))
    if any(isinstance(v, dict) and v["returncode"] for v in result["results"].values()):
        for k, v in result["results"].items():
            if isinstance(v, dict) and v["returncode"]:
                print(k, v["stderr"][:6000])
        raise RuntimeError("compile failed")
