# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""ptxas + SASS report for the SM90 FP8 backward epilogue-overlap variants (compile only).

Compiles the single-rank runtime instantiation of this tree's header with each requested
kOverlapEpilogue value, and optionally the same instantiation of a baseline header tree
(MONO_EP_BASE_INCLUDE_DIR, without the trailing template argument) so the default
instantiation's SASS can be compared instruction for instruction.
"""

import hashlib
import json
import os
import re
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("mono-ep-compile-overlap")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(
        root / "vendor/deepgemm/deep_gemm/include", "/probe/native"
    )
    base_include = os.environ.get("MONO_EP_BASE_INCLUDE_DIR")
    if base_include:
        image = image.add_local_dir(base_include, "/probe/base_native")

DEFINES = (
    "DG_TRAP_ONLY_TIMEOUTS 1",
    "DG_MonoEP_ROLE_SCOPED_REGISTERS 1",
    "DG_MonoEP_SPLIT_COMBINE_REGISTERS 160",
)
# Single-rank runtime instantiation: FP8 transport + combine, cached x/dy/dZ, 3 FP8 stages,
# every integrated epilogue default.
ARGS = (
    "17280, 2048, 768, 16, 0, 8, 128, 128, 128, 32, 44160, {stages}, 2048, "
    "128, 128, 256, 132, 1, true, true, true, false, true, true, "
    "{trace}, 0u, true, 2u, 1u, 7u, 4u, true"
)


def _sass_report(cuobjdump, cubin):
    sass = subprocess.run(  # nosec B603
        [cuobjdump, "-sass", str(cubin)], capture_output=True, text=True, check=False
    ).stdout
    body = []
    for line in sass.splitlines():
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", line)
        if m:
            body.append(m.group(1))
    regs = [int(r) for r in re.findall(r"\bR(\d+)\b", "\n".join(body))]
    return dict(
        sass_instructions=len(body),
        sass_sha256=hashlib.sha256("\n".join(body).encode()).hexdigest(),
        max_register_index=max(regs) if regs else None,
        hgmma=sum("HGMMA" in b for b in body),
        warpgroup_depbar=sum("WARPGROUP.DEPBAR" in b for b in body),
        warpgroup_arrive=sum("WARPGROUP.ARRIVE" in b for b in body),
        stl=sum(b.split()[0].startswith("STL") for b in body if b.split()),
        ldl=sum(b.split()[0].startswith("LDL") for b in body if b.split()),
        setmaxreg=sum("SETMAXREG" in b for b in body),
    ), body


@app.function(image=image, cpu=4, memory=16384, timeout=1800)
def compile_one(record):
    import importlib.util

    from torch.utils.cpp_extension import CUDA_HOME

    work = Path("/tmp/overlap") / record["name"]  # nosec B108
    work.mkdir(parents=True, exist_ok=True)
    args = ARGS.format(
        trace="true" if record["trace"] else "false", stages=record["stages"]
    )
    if record["tree"] == "this":
        args += f", {record['overlap']}u"
    source = work / "probe.cu"
    source.write_text(
        "".join(f"#define {d}\n" for d in (*DEFINES, *record["defines"]))
        + "#include <deep_gemm/impls/sm90_fp8_mono_ep_backward.cuh>\n"
        + "static void instantiate() {\n"
        + "    auto ptr = reinterpret_cast<void*>("
        + f"&deep_gemm::sm90_fp8_mono_ep_backward_impl<{args}>);\n}}\n"
    )
    cuda_bin = Path(CUDA_HOME) / "bin"
    root_include = Path(importlib.util.find_spec("deep_gemm").origin).parent / "include"
    native = "/probe/native" if record["tree"] == "this" else "/probe/base_native"
    command = [
        str(cuda_bin / "nvcc"),
        "--gpu-architecture=sm_90a",
        "-O3",
        "--ptxas-options=--verbose",
        "--ptxas-options=--register-usage-level=10",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--diag-suppress=39,161,174,177,186,940",
        f"-I{native}",
        f"-I{root_include}",
        "-cubin",
        str(source),
        "-o",
        str(work / "probe.cubin"),
    ]
    result = subprocess.run(  # nosec B603
        command, capture_output=True, text=True, timeout=1500, check=False
    )
    stack = re.findall(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
        result.stderr,
    )
    report, body = (
        _sass_report(str(cuda_bin / "cuobjdump"), work / "probe.cubin")
        if result.returncode == 0
        else ({}, [])
    )
    return dict(
        **record,
        instantiation=args,
        returncode=result.returncode,
        nvcc=subprocess.run(  # nosec B603
            [str(cuda_bin / "nvcc"), "--version"], capture_output=True, text=True
        ).stdout.splitlines()[-1:],
        ptxas_registers=[
            int(x) for x in re.findall(r"Used (\d+) registers", result.stderr)
        ],
        ptxas_stack_spill_store_load=[list(map(int, s)) for s in stack],
        c7507="C7507" in result.stderr,
        c7510="C7510" in result.stderr,
        wgmma_serialized=sorted(
            set(re.findall(r"\((C75\d\d)\) Potential Performance Loss", result.stderr))
        ),
        warnings=[ln for ln in result.stderr.splitlines() if "arning" in ln][:20],
        stderr_tail=result.stderr[-6000:],
        sass=report,
        sass_body=body if record.get("keep_sass") else None,
    )


@app.local_entrypoint()
def main(out: str, variants: str = "this0:this:0:0,this1:this:1:0,base:base:0:0"):
    """variants: comma-separated name:tree(this|base):overlap:trace[:keep_sass[:DEFINE=V+...[:stages]]]."""
    records = []
    for spec in variants.split(","):
        parts = spec.split(":")
        records.append(
            dict(
                name=parts[0],
                tree=parts[1],
                overlap=int(parts[2]),
                trace=parts[3] == "1",
                keep_sass=len(parts) > 4 and parts[4] == "1",
                defines=[
                    d.replace("=", " ")
                    for d in (parts[5].split("+") if len(parts) > 5 else [])
                    if d
                ],
                stages=int(parts[6]) if len(parts) > 6 else 3,
            )
        )
    results = list(compile_one.map(records))
    payload = dict(
        modal_app_id=os.environ.get("MODAL_APP_ID") or app.app_id, results=results
    )
    Path(out).write_text(json.dumps(payload, indent=1))
    for r in results:
        print(
            r["name"],
            r["returncode"],
            r["ptxas_registers"],
            r["ptxas_stack_spill_store_load"],
            "C7507" if r["c7507"] else "",
            "C7510" if r["c7510"] else "",
            r["wgmma_serialized"],
            json.dumps(r["sass"]),
        )
        if r["returncode"]:
            print(r["stderr_tail"])
