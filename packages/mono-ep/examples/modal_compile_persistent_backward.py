# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compile the native persistent backward executor without launching it."""

import json
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-compile-persistent-backward")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(
        root / "vendor/deepgemm/deep_gemm/include/deep_gemm", "/probe/native/deep_gemm"
    )


@app.function(image=image, cpu=8, memory=32768, timeout=600)
def compile_kernel():
    import deep_gemm
    from torch.utils.cpp_extension import CUDA_HOME

    source = Path("/tmp/persistent_backward.cu")
    source.write_text("""
#include <deep_gemm/impls/sm100_bf16_mono_ep_backward.cuh>
static void instantiate() {
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm100_bf16_mono_ep_backward_impl<
        16384, 2048, 768, 128, 0, 8, 128, 128, 64, 32,
        16384, 2, 4096, 128, 128, 256, 148, 8>);
}
""")
    result = subprocess.run(
        [
            str(Path(CUDA_HOME) / "bin/nvcc"),
            "-std=c++20",
            "-arch=sm_100a",
            "--expt-relaxed-constexpr",
            "--ptxas-options=-v",
            "-I/probe/native",
            f"-I{Path(deep_gemm.__file__).parent / 'include'}",
            "-cubin",
            str(source),
            "-o",
            "/tmp/persistent_backward.cubin",
        ],
        capture_output=True,
        text=True,
        timeout=540,
        check=False,
    )
    cubin = Path("/tmp/persistent_backward.cubin")
    return dict(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        cubin_bytes=cubin.stat().st_size if cubin.exists() else 0,
        validation="compile_only_not_executed",
    )


@app.local_entrypoint()
def main():
    result = compile_kernel.remote()
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("Native persistent backward compile failed")
