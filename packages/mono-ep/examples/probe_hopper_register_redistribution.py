# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compile isolated full-512-thread register experiments without launching kernels."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

HEADER = Path("deep_gemm/impls/sm90_fp8_mono_ep_backward.cuh")
SOURCE = """#define DG_TRAP_ONLY_TIMEOUTS 1
#include <deep_gemm/impls/sm90_fp8_mono_ep_backward.cuh>
static void instantiate() {
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mono_ep_backward_impl<
        16384, 2048, 768, 128, 0, 8, 128, 128, 128, 32,
        16384, 2, 2048, 128, 128, 256, 132, 8,
        true, true, true, false, true, true>);
}
"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one source marker: {old}")
    return text.replace(old, new, 1)


def variant_header(original, name):
    if "__launch_bounds__(kNumThreads, 1)" not in original:
        raise ValueError("Unexpected launch bounds")
    text = original
    if name != "baseline_48_40_208":
        for old, new in ((48, 64), (40, 56), (208, 192)):
            text = replace_once(
                text,
                f"kUseMoreEpilogueRegisters ? {old} :",
                f"kUseMoreEpilogueRegisters ? {new} :",
            )
    if name == "early_64_56_192":
        start = text.index("    // Adjust registers\n")
        end = text.index("    // Grid sync index assignments", start)
        allocation = text[start:end]
        text = text[:start] + text[end:]
        text = replace_once(
            text, "    // Task scheduler\n", allocation + "    // Task scheduler\n"
        )
    return text


def probe(native_include, output):
    import deep_gemm
    from torch.utils.cpp_extension import CUDA_HOME

    native_include, output = Path(native_include), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = (native_include / HEADER).read_text()
    cuda_bin = Path(CUDA_HOME) / "bin"
    version = subprocess.run(
        [str(cuda_bin / "nvcc"), "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    records = []
    for name in ("baseline_48_40_208", "budget_64_56_192", "early_64_56_192"):
        work = output / name
        work.mkdir()
        header = work / HEADER
        header.parent.mkdir(parents=True)
        modified = variant_header(original, name)
        header.write_text(modified)
        source = work / "probe.cu"
        source.write_text(SOURCE)
        cubin = work / "probe.cubin"
        command = [
            str(cuda_bin / "nvcc"),
            "-std=c++20",
            "-arch=sm_90a",
            "--expt-relaxed-constexpr",
            "--ptxas-options=-v",
            "--keep",
            "--keep-dir",
            str(work),
            f"-I{work}",
            f"-I{native_include}",
            f"-I{Path(deep_gemm.__file__).parent / 'include'}",
            "-cubin",
            str(source),
            "-o",
            str(cubin),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=840, check=False
        )
        (work / "compile.stderr").write_text(result.stderr)
        ptx = (work / "probe.ptx").read_text() if (work / "probe.ptx").exists() else ""
        sass = ""
        disassembly_error = None
        if result.returncode == 0:
            tool = cuda_bin / "cuobjdump"
            disasm_command = [str(tool), "--dump-sass", str(cubin)]
            if not tool.exists():
                disasm_command = [str(cuda_bin / "nvdisasm"), str(cubin)]
            disasm = subprocess.run(
                disasm_command, capture_output=True, text=True, timeout=120, check=False
            )
            sass = disasm.stdout
            disassembly_error = disasm.stderr if disasm.returncode else None
            (work / "probe.sass").write_text(sass)
        registers = [
            line
            for line in sass.splitlines()
            if re.search(r"SETMAXNREG|SETMAXREG", line)
        ]
        records.append(
            dict(
                variant=name,
                compile_returncode=result.returncode,
                source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                variant_sha256=hashlib.sha256(modified.encode()).hexdigest(),
                command=command,
                stderr=result.stderr,
                disassembly_error=disassembly_error,
                sass_register_reallocation=registers,
                ptx_directives=[
                    line
                    for line in ptx.splitlines()
                    if any(
                        marker in line
                        for marker in (
                            "setmaxnreg",
                            ".maxntid",
                            ".reqntid",
                            ".minnctapersm",
                            ".maxnreg",
                        )
                    )
                ],
                sass_calls=[
                    line for line in sass.splitlines() if re.search(r"\bCALL\b", line)
                ],
                c7507="C7507" in result.stderr,
                c7510="C7510" in result.stderr,
                reallocation_observed=bool(registers),
            )
        )
    result = dict(
        nvcc_version=version,
        threads=512,
        transport_fp8=True,
        fp8_combine=True,
        trap_only=True,
        instantiation_source=SOURCE,
        validation="compile_and_disassemble_only_not_executed",
        variants=records,
    )
    (output / "results.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--native-include", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.native_include, args.output), indent=2))
