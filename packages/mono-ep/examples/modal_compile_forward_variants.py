# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""ptxas report for SM90 FP8 persistent forward instantiations at the EP8 per-GPU shape."""

import json
import os
import re
import subprocess  # nosec B404
import tempfile
from pathlib import Path

import modal

app = modal.App("axolotl-compile-forward-variants")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    include = os.environ.get(
        "MONO_EP_COMPILE_INCLUDE_DIR",
        str(root / "vendor/deepgemm/deep_gemm/include/deep_gemm"),
    )
    image = base_image.add_local_dir(include, "/probe/native/deep_gemm")


@app.function(image=image, gpu="H100!", cpu=8, memory=32768, timeout=1800)
def compile_variants(ring, suffixes):
    import deep_gemm
    from torch.utils.cpp_extension import CUDA_HOME

    workdir = Path(tempfile.mkdtemp(prefix="forward_ptxas_"))
    reports = {}
    for name, suffix in suffixes.items():
        source = workdir / f"forward_{name}.cu"
        source.write_text(
            "#include <deep_gemm/impls/sm90_fp8_mono_ep_forward.cuh>\n"
            "static void instantiate() {\n"
            "    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mono_ep_forward_impl<\n"
            f"        16384, 2048, 768, 16, 0, 8, 128, 128, 128, 32, {ring}, 2, 2048, 128, 128, 256,\n"
            f"        132, 1, 0x7f7fffffu, false, false, true{suffix}>);\n"
            "}\n"
        )
        command = [
            str(Path(CUDA_HOME) / "bin/nvcc"),
            "-std=c++20",
            "-arch=sm_90a",
            "--expt-relaxed-constexpr",
            "--ptxas-options=-v",
            "-I/probe/native",
            f"-I{Path(deep_gemm.__file__).parent / 'include'}",
            "-cubin",
            str(source),
            "-o",
            str(workdir / f"forward_{name}.cubin"),
        ]
        result = subprocess.run(  # nosec B603
            command, capture_output=True, text=True, timeout=1500, check=False
        )
        text = result.stdout + result.stderr
        usage = re.findall(r"Used (\d+) registers", text)
        spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", text)
        stack = re.findall(r"(\d+) bytes stack frame", text)
        reports[name] = dict(
            suffix=suffix,
            returncode=result.returncode,
            registers=usage,
            spill_store_load_bytes=spills,
            stack_frame_bytes=stack,
            c7507="C7507" in text,
            c7510="C7510" in text,
            ptxas_output=text[-6000:],
        )
    return reports


@app.local_entrypoint()
def main(out: str, ring: int, variants: str):
    suffixes = {}
    for spec in variants.split(";"):
        name, _, suffix = spec.partition(":")
        suffixes[name] = suffix
    reports = compile_variants.remote(ring, suffixes)
    Path(out).write_text(json.dumps(reports, indent=1))
    for name, report in reports.items():
        print(
            name,
            report["returncode"],
            report["registers"],
            report["spill_store_load_bytes"],
            report["stack_frame_bytes"],
            report["c7507"],
            report["c7510"],
        )
