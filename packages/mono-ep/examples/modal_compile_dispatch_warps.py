# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""ptxas report plus SASS CCTL.IVALL loop scan for the dispatch-warp x/dy pack and relaxed spins (compile only)."""

import json
import os
import re
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("mono-ep-compile-dispatch-warps")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(
        root / "vendor/deepgemm/deep_gemm/include", "/probe/native"
    )

# Single-rank runtime instantiations (trap timeouts, FP8 transport + combine, cached dZ).
BACKWARD = (
    "17280, 2048, 768, 16, 0, 8, 128, 128, 128, 32, 44160, {stages}, 2048, "
    "128, 128, 256, 132, 1, true, true, true, false, true, true{suffix}"
)
# EP8 per-GPU forward with the original W1 layout.
FORWARD = (
    "16384, 2048, 768, 16, 0, 8, 128, 128, 128, 32, 44160, {stages}, 2048, 128, 128, 256, "
    "132, 1, 0x7f7fffffu, false, false, true{suffix}"
)


def scan_cctl(sass):
    """Count CCTL.IVALL and those inside a backward-branch loop of at most 64 instructions."""
    lines = []
    for line in sass.splitlines():
        match = re.search(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", line)
        if match:
            lines.append((int(match.group(1), 16), match.group(2).strip()))
    branches = []
    for addr, text in lines:
        target = re.search(r"\bBRA\b.*?0x([0-9a-f]+)", text)
        if target and int(target.group(1), 16) <= addr:
            branches.append((int(target.group(1), 16), addr))
    cctl = [i for i, (_, text) in enumerate(lines) if "CCTL.IVALL" in text]
    index = {addr: i for i, (addr, _) in enumerate(lines)}

    def relaxed_inner_poll(t, addr):
        # A nested backward-branch loop before the CCTL that polls with a strong load and no CCTL:
        # the acquire then runs once per poll exit, not once per iteration.
        for t2, b2 in branches:
            if t <= t2 and b2 < addr and t2 in index and b2 in index:
                body = [text for _, text in lines[index[t2] : index[b2] + 1]]
                if any("STRONG" in x for x in body) and not any(
                    "CCTL" in x for x in body
                ):
                    return True
        return False

    in_loop = []
    for i in cctl:
        addr = lines[i][0]
        loops = [(t, b) for t, b in branches if t <= addr <= b and b - t <= 64 * 16]
        if loops:
            t, b = min(loops, key=lambda tb: tb[1] - tb[0])
            lo = max(0, i - 6)
            in_loop.append(
                dict(
                    addr=hex(addr),
                    loop=[hex(t), hex(b)],
                    acquire_after_relaxed_poll=relaxed_inner_poll(t, addr),
                    load=lines[i - 1][1],
                    context=[x for _, x in lines[lo : i + 4]],
                )
            )
    return dict(
        instructions=len(lines),
        cctl_ivall=len(cctl),
        cctl_ivall_in_tight_loops=len(in_loop),
        cctl_ivall_per_iteration=sum(
            not e["acquire_after_relaxed_poll"] for e in in_loop
        ),
        stl=sum(t.startswith("STL") or " STL" in t for _, t in lines),
        ldl=sum(t.startswith("LDL") or " LDL" in t for _, t in lines),
        in_loop=in_loop,
        ld_strong_gpu=sum("STRONG.GPU" in t for _, t in lines),
        ld_strong_sys=sum("STRONG.SYS" in t for _, t in lines),
        nanosleep=sum("NANOSLEEP" in t for _, t in lines),
    )


@app.function(image=image, cpu=4, memory=16384, timeout=1800)
def compile_one(record):
    import importlib.util

    from torch.utils.cpp_extension import CUDA_HOME

    work = Path("/tmp/dispatch_warps") / record["name"]  # nosec B108
    work.mkdir(parents=True, exist_ok=True)
    direction = record["direction"]
    template = BACKWARD if direction == "backward" else FORWARD
    args = template.format(stages=record["stages"], suffix=record["suffix"])
    source = work / "probe.cu"
    source.write_text(
        "#define DG_TRAP_ONLY_TIMEOUTS 1\n"
        + "".join(f"#define {d}\n" for d in record["defines"])
        + f"#include <deep_gemm/impls/sm90_fp8_mono_ep_{direction}.cuh>\n"
        + "static void instantiate() {\n"
        + "    auto ptr = reinterpret_cast<void*>("
        + f"&deep_gemm::sm90_fp8_mono_ep_{direction}_impl<{args}>);\n}}\n"
    )
    cuda_bin = Path(CUDA_HOME) / "bin"
    root_include = Path(importlib.util.find_spec("deep_gemm").origin).parent / "include"
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
        "-I/probe/native",
        f"-I{root_include}",
        "-cubin",
        str(source),
        "-o",
        str(work / "probe.cubin"),
    ]
    result = subprocess.run(  # nosec B603
        command, capture_output=True, text=True, timeout=1500, check=False
    )
    sass = ""
    if result.returncode == 0:
        sass = subprocess.run(  # nosec B603
            [str(cuda_bin / "cuobjdump"), "-sass", str(work / "probe.cubin")],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        ).stdout
    stack = re.findall(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
        result.stderr,
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
        warnings=[ln for ln in result.stderr.splitlines() if "warning" in ln][:20],
        sass_scan=scan_cctl(sass),
        sass=sass if "dump" in record["name"] else "",
        stderr_tail=result.stderr[-4000:],
    )


@app.local_entrypoint()
def main(out: str, variants: str):
    """variants: ';'-separated name|direction|stages|DEFINE+DEFINE|template suffix."""
    records = []
    for spec in variants.split(";"):
        name, direction, stages, defines, suffix = spec.split("|")
        records.append(
            dict(
                name=name,
                direction=direction,
                stages=int(stages),
                defines=[d for d in defines.split("+") if d],
                suffix=suffix,
            )
        )
    results = list(compile_one.map(records))
    payload = dict(
        modal_app_id=os.environ.get("MODAL_APP_ID") or app.app_id,
        results=results,
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
            "CCTL.IVALL",
            r["sass_scan"]["cctl_ivall"],
            "in tight loops",
            r["sass_scan"]["cctl_ivall_in_tight_loops"],
            "per-iteration",
            r["sass_scan"]["cctl_ivall_per_iteration"],
            "STL/LDL",
            r["sass_scan"]["stl"],
            r["sass_scan"]["ldl"],
        )
        if r["returncode"]:
            print(r["stderr_tail"])
