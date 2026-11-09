# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compile the opt-in SM90 persistent backward prototype."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-compile-hopper-backward")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    vendor = Path(os.environ.get("MONO_EP_VENDOR_DIR", root / "vendor"))
    image = base_image.add_local_dir(
        vendor / "deepgemm/deep_gemm/include/deep_gemm", "/probe/native/deep_gemm"
    )


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=900)
def compile_kernel(
    fp8=False,
    forward=False,
    cached=False,
    inspect_calls=False,
    trap_timeouts=False,
    compact_forward=False,
    compact_backward=False,
):
    import deep_gemm
    from torch.utils.cpp_extension import CUDA_HOME

    source = Path("/tmp/sm90_persistent_backward.cu")
    source_text = """
#include <deep_gemm/impls/sm90_bf16_mono_ep_backward.cuh>
static void instantiate() {
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_bf16_mono_ep_backward_impl<
        16384, 2048, 768, 128, 0, 8, 128, 128, 64, 32,
        16384, 2, 4096, 128, 128, 256, 132, 8, true>);
}
"""
    if fp8:
        source_text = source_text.replace("sm90_bf16_mono_ep", "sm90_fp8_mono_ep")
        source_text = source_text.replace("128, 128, 64, 32", "128, 128, 128, 32")
    if forward:
        if not fp8:
            raise ValueError("Forward compile probe currently requires FP8")
        source_text = source_text.replace(
            "sm90_fp8_mono_ep_backward", "sm90_fp8_mono_ep_forward"
        )
        source_text = source_text.replace("132, 8, true", "132, 8, 0x7f7fffffu, false")
        source_text = source_text.replace("16384, 2, 4096, 128", "16384, 2, 2048, 128")
    if cached:
        if not fp8 or forward:
            raise ValueError("Cached compile selection is for FP8 backward")
        source_text = source_text.replace("132, 8, true", "132, 8, true, true, true")
    if compact_forward:
        if not fp8 or not forward:
            raise ValueError("Compact non-math WG selection requires FP8 forward")
        original = source_text
        source_text = source_text.replace("2048, 128, 128, 256", "2048, 32, 96, 256")
        source_text = source_text.replace(
            "0x7f7fffffu, false>", "0x7f7fffffu, false, true>"
        )
        if source_text == original or "2048, 32, 96, 256" not in source_text:
            raise RuntimeError("Compact forward compile probe substitution failed")
    if compact_backward:
        if not fp8 or forward or not cached:
            raise ValueError(
                "Compact backward compile probe requires --fp8 --cached without --forward"
            )
        original = source_text
        source_text = source_text.replace("4096, 128, 128, 256", "4096, 32, 96, 256")
        source_text = source_text.replace(
            "132, 8, true, true, true>", "132, 8, true, true, true, true>"
        )
        if source_text == original or "4096, 32, 96, 256" not in source_text:
            raise RuntimeError("Compact backward compile probe substitution failed")
    if trap_timeouts:
        source_text = "#define DG_TRAP_ONLY_TIMEOUTS 1\n" + source_text
    source.write_text(source_text)
    command = [
        str(Path(CUDA_HOME) / "bin/nvcc"),
        "-std=c++20",
        "-arch=sm_90a",
        "--expt-relaxed-constexpr",
        "--ptxas-options=-v",
        *(["--keep", "--keep-dir", "/tmp"] if inspect_calls else []),
        "-I/probe/native",
        f"-I{Path(deep_gemm.__file__).parent / 'include'}",
        "-cubin",
        str(source),
        "-o",
        "/tmp/sm90_persistent_backward.cubin",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=840,
        check=False,
    )
    cubin = Path("/tmp/sm90_persistent_backward.cubin")
    ptx_path = source.with_suffix(".ptx")
    ptx_lines = ptx_path.read_text().splitlines() if ptx_path.exists() else []
    return {
        "returncode": result.returncode,
        "cached_activations": cached or forward,
        "trap_only_timeouts": trap_timeouts,
        "compact_non_math_wg": compact_forward or compact_backward,
        "nvcc_command": command,
        "nvcc_version": subprocess.run(
            [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout,
        "instantiation_source": source_text,
        "instantiation_source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "ptx_register_directives": [
            line
            for line in ptx_lines
            if any(
                marker in line
                for marker in ("setmaxnreg", ".maxntid", ".reqntid", ".minnctapersm")
            )
        ],
        "ptx_call_sites": [
            "\n".join(ptx_lines[max(0, i - 2) : i + 12])
            for i, line in enumerate(ptx_lines)
            if line.strip().startswith(("call ", "call.", "call\t"))
        ],
        "ptx_functions": [line for line in ptx_lines if ".func " in line],
        "ptx_strings": [line for line in ptx_lines if "$str" in line and ".b8" in line],
        "stdout": result.stdout,
        "stderr": result.stderr,
        "cubin_bytes": cubin.stat().st_size if cubin.exists() else 0,
        "validation": "compile_only_not_executed",
        "direction": "forward" if forward else "backward",
        "compute_dtype": "blockwise_fp8" if fp8 else "bf16",
    }


@app.local_entrypoint()
def main(
    fp8: bool = False,
    forward: bool = False,
    cached: bool = False,
    inspect_calls: bool = False,
    trap_timeouts: bool = False,
    compact_forward: bool = False,
    compact_backward: bool = False,
):
    result = compile_kernel.remote(
        fp8,
        forward,
        cached,
        inspect_calls,
        trap_timeouts,
        compact_forward,
        compact_backward,
    )
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("SM90 persistent backward compile failed")
