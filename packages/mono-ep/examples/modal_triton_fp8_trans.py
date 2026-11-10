# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Triton FP8 tile-transpose repro on a strict H100: ``modal run modal_triton_fp8_trans.py --out DIR``.

Runs ``tools/triton_fp8_trans_repro.py`` and ``tools/probe_prefix_fused_forms.py`` on the
``axolotlai/axolotl-uv:main-latest`` stack and brings back the JSON reports and IR dumps.
"""

import io
import json
import os
import subprocess  # nosec B404
import tarfile
import tempfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("mono-ep-triton-fp8-trans")
image = (
    modal.Image.from_registry("axolotlai/axolotl-uv:main-latest")
    .add_local_dir(ROOT / "src", "/probe/src")
    .add_local_dir(ROOT / "tools", "/probe/tools")
    .add_local_dir(ROOT / "tests", "/probe/tests")
)


def _run(cmd, cwd="/probe", **env):
    result = subprocess.run(  # nosec B603
        cmd,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": "/probe/src", **env},
        capture_output=True,
        text=True,
        timeout=2400,
        check=False,
    )
    return dict(
        cmd=cmd,
        returncode=result.returncode,
        stdout=result.stdout[-30000:],
        stderr=result.stderr[-8000:],
    )


@app.function(image=image, gpu="H100!", cpu=4, memory=32768, timeout=3000)
def run(gpu_name="H100", full=False):
    import torch
    import triton

    name = torch.cuda.get_device_name()
    out = dict(device=name, torch=torch.__version__, triton=triton.__version__, runs=[])
    if gpu_name not in name:
        out["error"] = "wrong device"
        return out, b""
    work = Path(tempfile.mkdtemp(prefix="fp8-trans-"))
    repro_json, prefix_json, dumps = (
        work / "repro.json",
        work / "prefix.json",
        work / "dumps",
    )
    out["runs"].append(
        _run(
            [
                "python",
                "-m",
                "pytest",
                "-v",
                "-p",
                "no:cacheprovider",
                "tests/test_triton_fp8_transpose.py",
                "tests/test_sm90_weight_cache.py",
                "tests/test_transposed_quantization.py",
                "-k",
                "not fsdp2",
            ]
        )
    )
    if full:
        out["runs"].append(
            _run(
                [
                    "python",
                    "tools/triton_fp8_trans_repro.py",
                    "--out",
                    str(repro_json),
                    "--dump",
                    str(dumps / "repro"),
                    "--diagnose",
                    "--ptxas-matrix",
                ]
            )
        )
    out["runs"].append(
        _run(
            [
                "python",
                "tools/probe_prefix_fused_forms.py",
                "--src",
                "tools/sm90_weight_forms_prefix_src.py.txt",
                "--out",
                str(prefix_json),
                "--dump",
                str(dumps / "prefix"),
            ]
        )
    )
    for key, path in (("repro", repro_json), ("prefix", prefix_json)):
        if path.exists():
            out[key] = json.loads(path.read_text())
    buf = io.BytesIO()
    if dumps.exists():
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(dumps, arcname="dumps")
    return out, buf.getvalue()


@app.local_entrypoint()
def main(out: str, gpu_name: str = "H100", full: bool = False):
    result, dumps = run.remote(gpu_name, full)
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    (target / "result.json").write_text(json.dumps(result, indent=1))
    if dumps:
        (target / "dumps.tar.gz").write_bytes(dumps)
    for r in result.get("runs", []):
        print(" ".join(r["cmd"]), "->", r["returncode"])
        print(r["stdout"][-6000:])
        print(r["stderr"][-2000:])
    print(
        json.dumps({k: result.get(k) for k in ("device", "torch", "triton", "error")})
    )
