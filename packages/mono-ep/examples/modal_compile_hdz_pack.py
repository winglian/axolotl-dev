# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""ptxas report for the h/dZ pack and dZ quantize opt-ins at the single-rank probe instantiation."""

import json
import re
import subprocess  # nosec B404
import tempfile
from pathlib import Path

import modal

app = modal.App("axolotl-compile-hdz-pack")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(
        root / "vendor/deepgemm/deep_gemm/include/deep_gemm", "/probe/native/deep_gemm"
    )

# Runtime arguments of the FP8-transport, FP8-combine, cached-dZ, packed-wgrad backward.
PREFIX = (
    "{max_tokens}, 2048, 768, 16, 0, 8, 128, 128, 128, 32, {ring}, 2, 2048, 128, 128, 256, "
    "132, 1, true, true, true, false, true, true"
)
VARIANTS = {
    "baseline_dscorewarp": "false, 0u, true",
    "vechdz2": "false, 0u, true, 2u, 0u",
    "vecdzq1": "false, 0u, true, 0u, 1u",
    "vecdzq2": "false, 0u, true, 0u, 2u",
    "vechdz2_vecdzq1": "false, 0u, true, 2u, 1u",
    "vechdz2_vecdzq2": "false, 0u, true, 2u, 2u",
}


@app.function(image=image, gpu="H100!", cpu=16, memory=32768, timeout=1800)
def compile_variants(max_tokens: int, ring: int):
    import deep_gemm
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = str(Path(CUDA_HOME) / "bin/nvcc")
    records = {}
    for name, suffix in VARIANTS.items():
        work = Path(tempfile.mkdtemp(prefix=name))
        source = work / "probe.cu"
        define, _, suffix = suffix.rpartition("|")
        args = PREFIX.format(max_tokens=max_tokens, ring=ring) + ", " + suffix
        source.write_text(
            define + "\n#define DG_TRAP_ONLY_TIMEOUTS 1\n"
            "#include <deep_gemm/impls/sm90_fp8_mono_ep_backward.cuh>\n"
            "static void instantiate() {\n"
            "    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mono_ep_backward_impl<"
            f"{args}>);\n}}\n"
        )
        command = [
            nvcc,
            "-std=c++20",
            "-arch=sm_90a",
            "--expt-relaxed-constexpr",
            "--ptxas-options=-v",
            "-I/probe/native",
            f"-I{Path(deep_gemm.__file__).parent / 'include'}",
            "-cubin",
            str(source),
            "-o",
            str(work / "probe.cubin"),
        ]
        result = subprocess.run(  # nosec B603
            command, capture_output=True, text=True, timeout=1500, check=False
        )
        text = result.stdout + result.stderr
        records[name] = dict(
            template_args=args,
            returncode=result.returncode,
            registers=re.findall(r"Used (\d+) registers", text),
            spill_lines=[line for line in text.splitlines() if "spill" in line],
            stack=re.findall(r"(\d+) bytes stack frame", text),
            c7507="C7507" in text,
            c7510="C7510" in text,
            ptxas=text[-6000:],
        )
    return dict(
        device=torch.cuda.get_device_name(),
        nvcc=subprocess.run(  # nosec B603
            [nvcc, "--version"], capture_output=True, text=True, check=False
        ).stdout,
        records=records,
    )


@app.local_entrypoint()
def main(out: str, max_tokens: int = 16384, ring: int = 16384):
    result = compile_variants.remote(max_tokens, ring)
    result["modal_app_id"] = app.app_id
    Path(out).write_text(json.dumps(result, indent=1))
    for name, record in result["records"].items():
        print(
            name,
            record["returncode"],
            record["registers"],
            record["stack"],
            record["spill_lines"],
            record["c7507"],
            record["c7510"],
        )
        if record["returncode"]:
            print(record["ptxas"])
