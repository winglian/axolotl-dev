# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-rank H100 A/B of the dispatch ring-slot spin, with ptxas and SASS capture."""

import glob
import json
import os
import re
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-dispatch-spin")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(root / "src", "/probe/src").add_local_file(
        root / "examples/probe_diag_staging.py", "/probe/diag.py"
    )

JIT_CACHE = "/tmp/dg_cache"  # nosec B108
KERNEL_FILE = "sm90_fp8_mono_ep_backward.cuh"
FILE_RE = re.compile(r'//## File "([^"]+)", line (\d+)')


def _sass(cubin, lines):
    """Return CCTL counts and the SASS attributed to kernel source lines in [lo, hi]."""
    lo, hi = lines
    text = subprocess.run(  # nosec B603
        ["/usr/local/cuda/bin/nvdisasm", "-gi", "-c", cubin],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    snippet, inside = [], False
    for line in text.splitlines():
        match = FILE_RE.search(line)
        if match:
            chain = [(match.group(1), int(match.group(2)))]
            chain += [
                (f, int(n))
                for f, n in re.findall(r'inlined at "([^"]+)", line (\d+)', line)
            ]
            inside = any(f.endswith(KERNEL_FILE) and lo <= n <= hi for f, n in chain)
            if inside:
                snippet.append(line.strip())
            continue
        if inside and "/*" in line:
            snippet.append(line.strip())
    return dict(
        cctl_ivall_total=text.count("CCTL.IVALL"),
        ldg_strong_gpu_total=text.count("LDG.E.STRONG.GPU"),
        nanosleep_total=text.count("NANOSLEEP"),
        spin_region_sass=snippet[:400],
    )


@app.function(image=image, gpu="H100!", cpu=16, memory=65536, timeout=3000)
def probe(args, sass_lines=None):
    import torch

    name = torch.cuda.get_device_name()
    if "H100" not in name:
        return dict(returncode=2, stderr=f"wrong device {name}", stdout="")
    env = {
        **os.environ,
        "PYTHONPATH": "/probe/src",
        "OMP_NUM_THREADS": "1",
        "MONO_EP_SM90_FP8_TRAP_TIMEOUTS": "1",
        "MONO_EP_SM90_FP8_TRANSPORT": "1",
        "MONO_EP_SM90_FP8_COMBINE": "1",
        "DG_JIT_CACHE_DIR": JIT_CACHE,
        "DG_JIT_PTXAS_VERBOSE": "1",
    }
    if sass_lines:
        env["DG_JIT_WITH_LINEINFO"] = "1"
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        "/probe/diag.py",
        *args,
    ]
    result = subprocess.run(  # nosec B603
        command, env=env, capture_output=True, text=True, timeout=2900, check=False
    )
    ptxas = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if "ptxas" in line or "bytes stack frame" in line
    ]
    cubins = {}
    for cubin in glob.glob(f"{JIT_CACHE}/cache/*/kernel.cubin"):
        source = Path(cubin).with_name("kernel.cu")
        text = source.read_text() if source.exists() else ""
        if "mono_ep_backward" not in text:
            continue
        entry = dict(instantiation=text[-700:])
        if sass_lines:
            entry.update(_sass(cubin, sass_lines))
        cubins[cubin] = entry
    return dict(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr[-20000:],
        command=command,
        device=name,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
        ptxas=ptxas,
        cubins=cubins,
    )


@app.local_entrypoint()
def main(out: str, args: str = "", sass_lines: str = ""):
    lines = [int(v) for v in sass_lines.split("-")] if sass_lines else None
    result = probe.remote(args.split(), lines)
    payload = None
    for line in result["stdout"].splitlines():
        if line.startswith("DIAG_JSON "):
            payload = json.loads(line[len("DIAG_JSON ") :])
    result["diag"] = payload
    result["stdout"] = "\n".join(
        line
        for line in result["stdout"].splitlines()
        if not line.startswith("DIAG_JSON ")
    )[-20000:]
    result["modal_app_id"] = os.environ.get("MODAL_APP_ID") or app.app_id
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("returncode", "device", "modal_app_id")}))
    if result["returncode"] or payload is None:
        print(result["stderr"][-5000:])
        raise RuntimeError("dispatch spin probe failed")
