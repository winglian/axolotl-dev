# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""SM90 BF16 combine chunking across layer shapes: compile probe, 2xH100 validator, 8xH100 timing.

Two builds of the SM90 BF16 kernels run in every job: ``after`` is the tree the image was built
from, ``before`` overlays the headers in ``MONO_EP_BEFORE_HEADERS`` (the pre-change
``sm90_bf16_mono_ep*.cuh``) on the installed DeepGEMM include tree, each with its own JIT cache.

stage=compile: nvcc/ptxas report (registers, spills, stack, C7510) for both kernels at hidden
2048 / 4096 / 5120 / 7168, both builds.
stage=validate: ``validate_sm90_bf16_forward.py`` and, with ``--kinds forward,backward``,
``validate_sm90_bf16_backward.py`` (FP32 reference, forced ring wrap) at the
Qwen3-30B-A3B, Qwen3-235B-A22B and Beam shapes on two strict H100s, both builds; outputs
compared bitwise across builds and the JIT cache cubins compared by hash and SASS.
stage=timing: ``benchmark_persistent_training.py`` on eight strict H100s, one process per pass
(shape, precision, routing, build), all in one allocation.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("mono-ep-sm90-bf16-combine-chunks")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_dir(root / "vendor/deepgemm/deep_gemm/include", "/probe/native")
        .add_local_dir(
            Path(os.environ["MONO_EP_BEFORE_HEADERS"]), "/probe/headers-before"
        )
        .add_local_file(
            root / "examples/validate_sm90_bf16_forward.py", "/probe/validate.py"
        )
        .add_local_file(
            root / "examples/validate_sm90_bf16_backward.py",
            "/probe/validate_backward.py",
        )
        .add_local_file(
            root / "examples/benchmark_persistent_training.py", "/probe/benchmark.py"
        )
    )

HEADERS = ("sm90_bf16_mono_ep.cuh", "sm90_bf16_mono_ep_backward.cuh")
BUILDS = ("before", "after")
# hidden, intermediate, total experts at EP8
SHAPES = {
    "qwen3_30b_a3b": (2048, 768, 128),
    "qwen3_235b_a22b": (4096, 1536, 128),
    "beam": (5120, 2560, 256),
    "hidden7168": (7168, 2048, 256),
}
# The runtime's instantiations: block 128x128x64, store block 32, 2 stages, 512 threads, 132 SMs, EP8.
FORWARD = (
    "16384, {hidden}, {inner}, {experts}, 0, 8, 128, 128, 64, 32, 16384, 2, {pull}, 128, 128, 256, "
    "132, 8, cute::numeric_limits<float>::infinity(), false"
)
BACKWARD = (
    "16384, {hidden}, {inner}, {experts}, 0, 8, 128, 128, 64, 32, 16384, 2, {pull}, 128, 128, 256, "
    "132, 8, true"
)


def _pull_bytes(hidden):
    """BF16 dispatch pull: the row halved until it is at most 4 KB (heuristics/mono_ep.hpp)."""
    pull = hidden * 2
    while pull > 4096:
        pull //= 2
    return pull


def _installed_impls():
    import importlib.util

    return (
        Path(importlib.util.find_spec("deep_gemm").origin).parent
        / "include/deep_gemm/impls"
    )


def _select_build(build):
    """Point the installed DeepGEMM include tree (the JIT's include root) at one build's headers."""
    impls = _installed_impls()
    stash = Path("/tmp/headers-after")  # nosec B108
    if not stash.exists():
        stash.mkdir()
        for header in HEADERS:
            shutil.copy2(impls / header, stash / header)
    source = Path("/probe/headers-before") if build == "before" else stash
    for header in HEADERS:
        shutil.copy2(source / header, impls / header)


def _records(stdout, prefix):
    # Two ranks' prints can share a line; decode each JSON object after every prefix
    decoder = json.JSONDecoder()
    out = []
    for chunk in stdout.split(prefix)[1:]:
        try:
            out.append(decoder.raw_decode(chunk)[0])
        except json.JSONDecodeError:
            continue
    return out


def _cuda_bin():
    from torch.utils.cpp_extension import CUDA_HOME

    return Path(CUDA_HOME) / "bin"


def _cache_inventory(cache):
    """Relative paths, cubin hashes and SASS of every compiled kernel under a JIT cache dir."""
    cache = Path(cache)
    entries = {}
    if not cache.exists():
        return entries
    for path in sorted(cache.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(cache))
        entry = dict(
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        if path.suffix == ".cubin":
            sass = subprocess.run(  # nosec B603
                [str(_cuda_bin() / "cuobjdump"), "-sass", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            # Drop the lines that name the file so two caches of one kernel compare equal
            text = "\n".join(
                ln
                for ln in sass.stdout.splitlines()
                if not ln.startswith("Fatbin ") and "cubin" not in ln
            )
            entry.update(
                sass_sha256=hashlib.sha256(text.encode()).hexdigest(),
                sass_lines=len(text.splitlines()),
            )
        if path.name == "kernel.cu":
            entry["source"] = path.read_text()
        entries[relative] = entry
    return entries


def _compare_caches(caches):
    """Pair the two builds' kernels by their generated kernel.cu (the cache dir name also hashes the
    headers, so it differs between builds) and compare the cubins by hash and by SASS text."""
    by_source = {}
    for build, entries in caches.items():
        for relative, entry in entries.items():
            if "source" in entry:
                by_source.setdefault(entry["sha256"], {})[build] = relative.rsplit(
                    "/", 1
                )[0]
    result = {}
    for source_sha, dirs in by_source.items():
        after_dir = dirs.get("after")
        if after_dir is None:
            continue
        after = caches["after"][f"{after_dir}/kernel.cubin"]
        before = (
            caches["before"].get(f"{dirs['before']}/kernel.cubin")
            if "before" in dirs
            else None
        )
        source = caches["after"][f"{after_dir}/kernel.cu"]["source"]
        template = re.search(r"&(sm90_\w+)<([^>]*)>", source, re.S)
        result[after_dir] = dict(
            kernel=template.group(1) if template else None,
            template_args=" ".join(template.group(2).split()) if template else None,
            source_sha256=source_sha,
            after_dir=after_dir,
            before_dir=dirs.get("before"),
            after_cubin_sha256=after["sha256"],
            before_cubin_sha256=before["sha256"] if before else None,
            cubin_equal=bool(before and before["sha256"] == after["sha256"]),
            sass_equal=bool(before and before["sass_sha256"] == after["sass_sha256"]),
            sass_lines=after["sass_lines"],
        )
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=1800)
def compile_one(record):
    import importlib.util

    build, direction, shape = record["build"], record["direction"], record["shape"]
    hidden, inner, experts = SHAPES[shape]
    args = (FORWARD if direction == "forward" else BACKWARD).format(
        hidden=hidden, inner=inner, experts=experts, pull=_pull_bytes(hidden)
    )
    include = Path("/probe/native")
    if build == "before":
        include = Path("/tmp/native-before")  # nosec B108
        if not include.exists():
            shutil.copytree("/probe/native", include)
            for header in HEADERS:
                shutil.copy2(
                    Path("/probe/headers-before") / header,
                    include / "deep_gemm/impls" / header,
                )
    work = Path("/tmp/combine-chunks") / f"{build}-{direction}-{shape}"  # nosec B108
    work.mkdir(parents=True, exist_ok=True)
    suffix = "" if direction == "forward" else "_backward"
    source = work / "probe.cu"
    source.write_text(
        f"#include <deep_gemm/impls/sm90_bf16_mono_ep{suffix}.cuh>\n"
        "static void instantiate() {\n"
        "    auto ptr = reinterpret_cast<void*>("
        f"&deep_gemm::sm90_bf16_mono_ep{suffix}_impl<{args}>);\n}}\n"
    )
    root_include = Path(importlib.util.find_spec("deep_gemm").origin).parent / "include"
    command = [
        str(_cuda_bin() / "nvcc"),
        "--gpu-architecture=sm_90a",
        "-O3",
        "--ptxas-options=--verbose",
        "--ptxas-options=--register-usage-level=10",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--diag-suppress=39,161,174,177,186,940",
        f"-I{include}",
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
    return dict(
        **record,
        hidden=hidden,
        intermediate=inner,
        experts=experts,
        pull_bytes=_pull_bytes(hidden),
        instantiation=args,
        returncode=result.returncode,
        nvcc=subprocess.run(  # nosec B603
            [str(_cuda_bin() / "nvcc"), "--version"], capture_output=True, text=True
        ).stdout.splitlines()[-1:],
        ptxas_registers=[
            int(x) for x in re.findall(r"Used (\d+) registers", result.stderr)
        ],
        ptxas_stack_spill_store_load=[list(map(int, s)) for s in stack],
        c7507="C7507" in result.stderr,
        c7510="C7510" in result.stderr,
        static_assert_messages=re.findall(
            r'static assertion failed[^\n]*"([^"]*)"', result.stderr
        )
        + re.findall(r"static_assert[^\n]*\"([^\"]*)\"", result.stderr),
        warnings=[ln for ln in result.stderr.splitlines() if "warning" in ln][:20],
        stderr_tail=result.stderr[-4000:],
    )


VALIDATE_SHAPES = {
    # name: (hidden, intermediate, total experts at 2 ranks, top-k, max tokens per rank)
    # 64 experts keep 32 local experts per rank, the Beam EP8 count.
    "qwen3_30b_a3b": (2048, 768, 64, 8, 4096),
    "qwen3_235b_a22b": (4096, 1536, 64, 8, 4096),
    "beam": (5120, 2560, 64, 8, 4096),
}


@app.function(image=image, gpu="H100!:2", cpu=16, memory=65536, timeout=3600)
def validate(shapes, kinds=("forward",)):
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not all("H100" in n for n in names):
        return dict(returncode=2, devices=names, error="wrong devices")
    runs = []
    scripts = {
        "forward": ("/probe/validate.py", "SM90_BF16_FORWARD "),
        "backward": ("/probe/validate_backward.py", "SM90_BF16_BACKWARD "),
    }
    for build in BUILDS:
        _select_build(build)
        for kind, shape in ((k, s) for k in kinds for s in shapes):
            script, prefix = scripts[kind]
            hidden, inner, experts, topk, tokens = VALIDATE_SHAPES[shape]
            out_dir = f"/tmp/y-{build}"  # nosec B108
            result = subprocess.run(  # nosec B603 B607
                [
                    "python",
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=2",
                    script,
                    "--shape-name",
                    shape,
                    "--hidden",
                    str(hidden),
                    "--intermediate",
                    str(inner),
                    "--experts",
                    str(experts),
                    "--topk",
                    str(topk),
                    "--tokens",
                    str(tokens),
                    "--out-dir",
                    out_dir,
                ],
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MAX_JOBS": "4",
                    "PYTHONPATH": "/probe/src",
                    "DG_JIT_CACHE_DIR": f"/tmp/jit-{build}",
                    "DG_PRINT_CONFIGS": "1",
                },
                capture_output=True,
                text=True,
                timeout=1500,
                check=False,
            )
            print(f"VALIDATE {build} {kind} {shape} rc={result.returncode}", flush=True)
            print(result.stdout[-20000:], flush=True)
            print(result.stderr[-6000:], flush=True)
            runs.append(
                dict(
                    build=build,
                    kind=kind,
                    shape=shape,
                    returncode=result.returncode,
                    records=_records(result.stdout, prefix),
                    configs=[
                        ln for ln in result.stdout.splitlines() if "MonoEPConfig(" in ln
                    ],
                    stdout_tail=result.stdout[-4000:],
                    stderr_tail=result.stderr[-6000:],
                )
            )
    _select_build("after")
    bitwise = {}
    for path in sorted(Path("/tmp/y-after").glob("*.pt")):  # nosec B108
        other = Path("/tmp/y-before") / path.name  # nosec B108
        if other.exists():
            a, b = torch.load(path), torch.load(other)
            bitwise[path.stem] = dict(
                equal=bool(torch.equal(a, b)),
                max_abs_diff=(a.float() - b.float()).abs().max().item(),
            )
        else:
            bitwise[path.stem] = dict(
                equal=None, note="no before output (build failed)"
            )
    caches = {build: _cache_inventory(f"/tmp/jit-{build}") for build in BUILDS}
    return dict(
        returncode=max(r["returncode"] for r in runs),
        devices=names,
        runs=runs,
        bitwise_after_vs_before=bitwise,
        jit_cache_files={b: sorted(c) for b, c in caches.items()},
        cubins_after_vs_before=_compare_caches(caches),
    )


# shape spec "name:TxHxIxExK" (tokens per GPU, hidden, intermediate, total experts, top-k)
TIMING_SHAPES = {
    "qwen3_30b_a3b": "qwen3_30b_a3b:16384x2048x768x128x8",
    "beam": "beam:16384x5120x2560x256x8",
    "beam8k": "beam:8192x5120x2560x256x8",
}


def _parse_shape(spec):
    name, _, dims = TIMING_SHAPES.get(spec, spec).partition(":")
    values = [int(v) for v in dims.split("x")]
    if len(values) != 5:
        raise ValueError(f"shape {spec!r} must be name:TxHxIxExK")
    return (name, *values)


@app.function(image=image, gpu="H100!:8", cpu=32, memory=262144, timeout=7200)
def timing(passes):
    """passes: list of dicts(name, build, compute_dtype, shape, routing)."""
    import torch

    device = torch.cuda.get_device_name()
    if "H100" not in device:
        return dict(returncode=2, device=device, error="wrong device")
    results = []
    for item in passes:
        _select_build(item["build"])
        name, tokens, hidden, inner, experts, topk = _parse_shape(item["shape"])
        dtype = item["compute_dtype"]
        env = {
            **os.environ,
            "PYTHONPATH": "/probe/src",
            "OMP_NUM_THREADS": "1",
            "DG_JIT_CACHE_DIR": f"/tmp/jit-{item['build']}",
        }
        for key in ("MONO_EP_SM90_FP8_TRANSPORT", "MONO_EP_SM90_FP8_COMBINE"):
            env.pop(key, None)
        if dtype == "blockwise":
            env.update(MONO_EP_SM90_FP8_TRANSPORT="1", MONO_EP_SM90_FP8_COMBINE="1")
        command = [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            "/probe/benchmark.py",
            "--compute-dtype",
            dtype,
            "--recompute",
            "--compare-native",
            "--compiled",
            "--paths",
            "native",
            "--routing",
            item["routing"],
            "--shape-name",
            name,
            "--tokens",
            str(tokens),
            "--hidden",
            str(hidden),
            "--intermediate",
            str(inner),
            "--experts",
            str(experts),
            "--topk",
            str(topk),
            *(["--peer-backward"] if dtype == "bf16" else []),
        ]
        result = subprocess.run(  # nosec B603 B607
            command, env=env, capture_output=True, text=True, timeout=3000, check=False
        )
        print(
            f"PASS {item['name']} {item['routing']} rc={result.returncode}", flush=True
        )
        print(result.stdout[-30000:], flush=True)
        print(result.stderr[-6000:], flush=True)
        record = None
        for line in result.stdout.splitlines():
            if line.startswith("{") and '"benchmark"' in line:
                record = json.loads(line)
        results.append(
            dict(
                **item,
                returncode=result.returncode,
                record=record,
                stdout_tail=result.stdout[-3000:],
                stderr_tail=result.stderr[-6000:],
            )
        )
    _select_build("after")
    caches = {build: _cache_inventory(f"/tmp/jit-{build}") for build in BUILDS}
    return dict(
        returncode=max(r["returncode"] for r in results),
        device=device,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
        passes=results,
        jit_cache_files={b: sorted(c) for b, c in caches.items()},
        cubins_after_vs_before=_compare_caches(caches),
    )


DEFAULT_PASSES = [
    dict(
        name="qwen3_30b_bf16_after",
        build="after",
        compute_dtype="bf16",
        shape="qwen3_30b_a3b",
    ),
    dict(
        name="qwen3_30b_bf16_before",
        build="before",
        compute_dtype="bf16",
        shape="qwen3_30b_a3b",
    ),
    dict(name="beam_bf16", build="after", compute_dtype="bf16", shape="beam"),
    dict(name="beam_fp8", build="after", compute_dtype="blockwise", shape="beam"),
    dict(name="beam8k_bf16", build="after", compute_dtype="bf16", shape="beam8k"),
    dict(name="beam8k_fp8", build="after", compute_dtype="blockwise", shape="beam8k"),
]


@app.local_entrypoint()
def main(
    out: str,
    stage: str,
    shapes: str = "",
    passes: str = "",
    routings: str = "balanced,skewed",
    kinds: str = "forward",
):
    """stage: compile | validate | timing.

    shapes (compile, validate): comma list of shape names; default all.
    passes (timing): comma list of DEFAULT_PASSES names; default all, each on every routing.
    """
    if stage == "compile":
        selected = shapes.split(",") if shapes else list(SHAPES)
        records = [
            dict(build=build, direction=direction, shape=shape)
            for shape in selected
            for direction in ("forward", "backward")
            for build in BUILDS
        ]
        results = list(compile_one.map(records))
        payload = dict(
            modal_app_id=os.environ.get("MODAL_APP_ID") or app.app_id, results=results
        )
        for r in results:
            print(
                r["build"],
                r["direction"],
                r["shape"],
                r["returncode"],
                r["ptxas_registers"],
                r["ptxas_stack_spill_store_load"],
                "C7510" if r["c7510"] else "",
                r["static_assert_messages"],
            )
    elif stage == "validate":
        selected = shapes.split(",") if shapes else list(VALIDATE_SHAPES)
        payload = validate.remote(selected, tuple(kinds.split(",")))
        payload["modal_app_id"] = os.environ.get("MODAL_APP_ID") or app.app_id
        for run in payload.get("runs", []):
            for record in run["records"]:
                print(
                    run["build"],
                    run.get("kind", "forward"),
                    record["shape"],
                    record["routing"],
                    record["rank"],
                    record["rel_error_vs_fp32_reference"],
                    record["ring_wrapped"],
                )
        print(json.dumps(payload.get("bitwise_after_vs_before"), indent=1))
        print(json.dumps(payload.get("cubins_after_vs_before"), indent=1))
    elif stage == "timing":
        names = passes.split(",") if passes else [p["name"] for p in DEFAULT_PASSES]
        timed = [
            dict(p, routing=routing)
            for p in DEFAULT_PASSES
            if p["name"] in names
            for routing in routings.split(",")
        ]
        payload = timing.remote(timed)
        payload["modal_app_id"] = os.environ.get("MODAL_APP_ID") or app.app_id
        for item in payload.get("passes", []):
            record = item["record"] or {}
            print(
                item["name"],
                item["routing"],
                item["returncode"],
                record.get("median_ms"),
            )
    else:
        raise ValueError("stage must be compile, validate or timing")
    Path(out).write_text(json.dumps(payload, indent=1))
    print(
        json.dumps(
            {
                k: payload[k]
                for k in payload
                if k in ("returncode", "device", "devices", "modal_app_id")
            }
        )
    )
