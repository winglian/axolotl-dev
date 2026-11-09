# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""ptxas report for the SM90 FP8 backward at the single-rank runtime instantiation, per dx-combine mode."""

import json
import re
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-compile-dx-combine")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(
        root / "vendor/deepgemm/deep_gemm/include/deep_gemm", "/probe/native/deep_gemm"
    )

# Single-rank runtime instantiation (fp8 transport + fp8 combine, cached x/dy/dz, 512 threads).
PREFIX = (
    "17280, 2048, 768, 16, 0, 8, 128, 128, 128, 32, 44160, 2, 2048, "
    "128, 128, 256, 132, 1, true, true, true, false, true, true"
)


@app.function(image=image, cpu=8, memory=32768, timeout=1800)
def compile_modes(modes):
    from concurrent.futures import ThreadPoolExecutor

    from torch.utils.cpp_extension import CUDA_HOME

    def one(mode):
        suffix = ", false, 0u, true" + (f", {mode}u" if mode else "")
        work = Path(f"/tmp/dxc{mode}")  # nosec B108
        work.mkdir(parents=True, exist_ok=True)
        source = work / "probe.cu"
        source.write_text(
            "#define DG_TRAP_ONLY_TIMEOUTS 1\n"
            "#include <deep_gemm/impls/sm90_fp8_mono_ep_backward.cuh>\n"
            "static void instantiate() {\n"
            "    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mono_ep_backward_impl<"
            + PREFIX
            + suffix
            + ">);\n}\n"
        )
        import deep_gemm

        command = [
            str(Path(CUDA_HOME) / "bin/nvcc"),
            "--gpu-architecture=sm_90a",
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "--ptxas-options=--verbose",
            "--ptxas-options=--register-usage-level=10",
            "-I/probe/native",
            f"-I{Path(deep_gemm.__file__).parent / 'include'}",
            "-cubin",
            str(source),
            "-o",
            str(work / "probe.cubin"),
        ]
        result = subprocess.run(  # nosec B603
            command, capture_output=True, text=True, timeout=1700, check=False
        )
        text = result.stdout + result.stderr
        regs = re.findall(r"Used (\d+) registers", text)
        spill = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", text)
        stack = re.findall(r"(\d+) bytes stack frame", text)
        return dict(
            mode=mode,
            returncode=result.returncode,
            registers=regs,
            spill_store_load=spill,
            stack=stack,
            c7507="C7507" in text,
            c7510="C7510" in text,
            ptxas=text[-6000:],
        )

    with ThreadPoolExecutor(len(modes)) as pool:
        return list(pool.map(one, modes))


@app.local_entrypoint()
def main(out: str, modes: str = "0,1,2,4,7"):
    result = compile_modes.remote([int(m) for m in modes.split(",")])
    Path(out).write_text(json.dumps(result, indent=1))
    for r in result:
        print(json.dumps({k: r[k] for k in r if k != "ptxas"}))
        if r["returncode"]:
            print(r["ptxas"])
