# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Benchmark Cursor's Mixture-of-Kittens (MoK) on Modal B200s at mono-ep shapes.

The image is the mono-ep B200 base (``axolotlai/axolotl-uv:main-latest`` plus
build tools, without the DeepGEMM layer) with MoK cloned at a pinned commit and
compiled for SM100 in an image layer, so GPU time is spent only on benchmarks.

    modal run modal_mok_bench.py::single --out ../benchmarks/mok_single_b200.json
    modal run modal_mok_bench.py::ep8 --out-dir ../benchmarks --shapes 30b,235b,beam
"""

import json
import os
import re
import subprocess  # nosec B404
from pathlib import Path

import modal

MOK_COMMIT = "f3a8bd2432457c5a2a2ea13915f0dcd0224581e4"
REPO = "/opt/mok/repo"
SHAPES = {
    "30b": dict(tokens=16384, hidden=2048, intermediate=768, experts=128),
    "235b": dict(tokens=16384, hidden=4096, intermediate=1536, experts=128),
    "beam": dict(tokens=16384, hidden=5120, intermediate=2560, experts=256),
}
MINIBATCHES = (2048, 4096, 8192, 16384)
COMM_SMS = (16, 24, 36, 52)
# Smallest multiple of every swept minibatch above 1.25 x tokens x topk, so
# each rank's routed rows fit one ring-buffer pass.
MACROBATCH = 163840

app = modal.App("axolotl-mok-bench")
image = (
    modal.Image.from_registry("axolotlai/axolotl-uv:main-latest")
    .run_commands(
        'uv pip install --python "$(command -v python)" setuptools wheel packaging ninja "pytest>=9,<10"',
        "apt-get update && apt-get install -y git make",
    )
    .run_commands(
        f"git clone https://github.com/cursor/mixture-of-kittens {REPO} && cd {REPO} "
        f"&& git checkout {MOK_COMMIT} && git submodule update --init third_party/ThunderKittens "
        "&& git -C third_party/ThunderKittens rev-parse HEAD > /opt/mok/tk_commit",
        # Unqualified ``detail::`` is ambiguous under torch 2.14 / CUDA 13.0 headers.
        f"cd {REPO} && sed -i 's/^\\( *\\)detail::tcgen05::commit/\\1kittens::detail::tcgen05::commit/' "
        "csrc/megakernel/grouped_gemm.cuh && git diff > /opt/mok/patch.diff && test -s /opt/mok/patch.diff",
        f"cd {REPO} && export PATH=/usr/local/cuda/bin:$PATH "
        "LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-} "
        "&& nvcc --version > /opt/mok/nvcc.txt && start=$(date +%s) "
        '&& (make ARCH=SM100 PYTHON="$(command -v python)" > /opt/mok/build.log 2>&1; rc=$?; '
        'echo "compile_seconds $(( $(date +%s) - start )) rc $rc" >> /opt/mok/build.log; '
        "tail -40 /opt/mok/build.log; exit $rc)",
    )
    .add_local_file(Path(__file__).with_name("mok_sweep.py"), "/probe/mok_sweep.py")
)


def _env(**extra):
    return {
        **os.environ,
        "PYTHONPATH": REPO,
        "OMP_NUM_THREADS": "1",
        "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
        **{k: str(v) for k, v in extra.items()},
    }


def _run(command, timeout, **env):
    try:
        result = subprocess.run(  # nosec B603
            command,
            cwd=REPO,
            env=_env(**env),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out, err, rc = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
        err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
        out, err, rc = out or "", (err or "") + f"\nTIMEOUT after {timeout}s", -9
    print(f"$ {' '.join(command)} -> rc {rc}", flush=True)
    print(out[-3000:], err[-3000:], sep="\n", flush=True)
    return dict(command=command, env=env, returncode=rc, stdout=out, stderr=err)


def _torchrun(world, *args):
    return [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        *args,
    ]


def _sweep(world, shape, configs, warmup, iters, timeout, **flags):
    command = _torchrun(
        world,
        "/probe/mok_sweep.py",
        f"--tokens={shape['tokens']}",
        f"--hidden={shape['hidden']}",
        f"--intermediate={shape['intermediate']}",
        f"--experts={shape['experts']}",
        "--topk=8",
        f"--configs={json.dumps(configs)}",
        f"--warmup={warmup}",
        f"--iters={iters}",
        *(f"--{k}={v}" for k, v in flags.items()),
    )
    run = _run(command, timeout)
    run["payload"] = None
    run["rows"] = []
    for line in run["stdout"].splitlines():
        if line.startswith("MOKSWEEP_JSON "):
            run["payload"] = json.loads(line[len("MOKSWEEP_JSON ") :])
        elif line.startswith('{"precision"'):
            run["rows"].append(json.loads(line))
    run["stdout"] = run["stdout"][-6000:]
    run["stderr"] = run["stderr"][-12000:]
    return run


def _bench_env(shape, cfg):
    return dict(
        NUM_LOCAL_TOKENS=shape["tokens"],
        HIDDEN_DIM=shape["hidden"],
        INTERMEDIATE_DIM=shape["intermediate"],
        NUM_EXPERTS=shape["experts"],
        TOPK=8,
        MINIBATCH_SIZE=cfg["minibatch"],
        MACROBATCH_SIZE=cfg["macrobatch"],
        BF16_FWD_COMM_SMS=cfg["fwd_comm_sms"],
        BF16_BWD_COMM_SMS=cfg["bwd_comm_sms"],
        MXFP8_FWD_COMM_SMS=cfg["fwd_comm_sms"],
        MXFP8_BWD_COMM_SMS=cfg["bwd_comm_sms"],
    )


def _devices(expected):
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != expected or any("B200" not in n for n in names):
        raise RuntimeError(f"wrong devices {names}")
    return dict(
        device=sorted(set(names)),
        gpus=len(names),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        nvcc=Path("/opt/mok/nvcc.txt").read_text().strip().splitlines()[-1],
        mok_commit=MOK_COMMIT,
        thunderkittens_commit=Path("/opt/mok/tk_commit").read_text().strip(),
        mok_patch=Path("/opt/mok/patch.diff").read_text(),
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
    )


def _build_facts():
    log = Path("/opt/mok/build.log").read_text()
    spills = [
        line.strip()
        for line in log.splitlines()
        if "spill" in line and re.search(r"[1-9]\d* bytes spill", line)
    ]
    match = re.search(r"compile_seconds (\d+) rc (\d+)", log)
    return dict(
        compile_seconds=int(match.group(1)) if match else None,
        build_rc=int(match.group(2)) if match else None,
        nonzero_spill_lines=spills,
        spill_lines_total=sum("spill" in line for line in log.splitlines()),
        warnings=[line for line in log.splitlines() if "warning" in line.lower()][:40],
        log_tail=log[-4000:],
    )


@app.function(image=image, gpu="B200", cpu=16, memory=131072, timeout=3600)
def run_single():
    info = _devices(1)
    info["build"] = _build_facts()
    info["import"] = _run(["python", "-c", "import mok; print(mok.__version__)"], 300)
    shape = SHAPES["30b"]
    default = dict(fwd_comm_sms=24, bwd_comm_sms=28, minibatch=4096, macrobatch=131072)
    info["bench_mok"] = _run(
        _torchrun(1, "-m", "benchmarks.bench_mok"), 1800, **_bench_env(shape, default)
    )
    info["sweep_default_full"] = _sweep(1, shape, [default], 500, 100, 1800)
    info["test_misc"] = _run(
        _torchrun(1, "-m", "pytest", "-q", "-x", "tests/test_misc.py"), 1500
    )
    for key in ("import", "bench_mok", "test_misc"):
        info[key]["stdout"] = info[key]["stdout"][-8000:]
        info[key]["stderr"] = info[key]["stderr"][-8000:]
    return info


def _best(rows, key):
    rows = [r for r in rows if r.get(key) is not None]
    return min(rows, key=lambda r: r[key]) if rows else None


def _pick(rows):
    """Per minibatch, pick fwd and bwd comm SMs independently; keep the minibatch with the lowest sum."""
    best = None
    for minibatch in sorted({r["minibatch"] for r in rows}):
        group = [r for r in rows if r["minibatch"] == minibatch]
        fwd, bwd = _best(group, "fwd_ms"), _best(group, "bwd_ms")
        if fwd is None or bwd is None:
            continue
        total = fwd["fwd_ms"] + bwd["bwd_ms"]
        if best is None or total < best[0]:
            best = (
                total,
                dict(
                    fwd_comm_sms=fwd["fwd_comm_sms"],
                    bwd_comm_sms=bwd["bwd_comm_sms"],
                    minibatch=minibatch,
                    macrobatch=MACROBATCH,
                ),
            )
    return best[1] if best else None


@app.function(image=image, gpu="B200:8", cpu=32, memory=262144, timeout=5400)
def run_ep8(name, sweep_warmup=50, sweep_iters=30):
    info = _devices(8)
    info["shape_name"] = name
    shape = SHAPES[name]
    grid = [
        dict(fwd_comm_sms=c, bwd_comm_sms=c, minibatch=m, macrobatch=MACROBATCH)
        for m in MINIBATCHES
        for c in COMM_SMS
    ]
    info["sweep"] = _sweep(8, shape, grid, sweep_warmup, sweep_iters, 2400)
    rows = (info["sweep"]["payload"] or {}).get("results") or info["sweep"]["rows"]
    info["picked"] = {}
    finals = []
    for precision in ("BF16", "MXFP8"):
        picked = _pick(
            [r for r in rows if r["precision"] == precision and r["correct"]]
        )
        info["picked"][precision] = picked
        if picked:
            finals.append((precision, picked))
    info["final"] = {
        precision: _sweep(8, shape, [cfg], 500, 100, 1800, precisions=precision)
        for precision, cfg in finals
    }
    if finals:
        info["bench_mok_unmodified"] = _run(
            _torchrun(8, "-m", "benchmarks.bench_mok"),
            1800,
            **_bench_env(shape, finals[0][1]),
        )
    info["nccl_control"] = _run(
        _torchrun(8, "-m", "benchmarks.bench_nccl_torch"),
        2400,
        NUM_LOCAL_TOKENS=shape["tokens"],
        HIDDEN_DIM=shape["hidden"],
        INTERMEDIATE_DIM=shape["intermediate"],
        NUM_EXPERTS=shape["experts"],
        TOPK=8,
    )
    for key in ("bench_mok_unmodified", "nccl_control"):
        if key in info:
            info[key]["stdout"] = info[key]["stdout"][-8000:]
            info[key]["stderr"] = info[key]["stderr"][-8000:]
    return info


@app.local_entrypoint()
def single(out: str):
    result = run_single.remote()
    result["modal_app_id"] = app.app_id
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps({k: result[k] for k in ("device", "torch", "nvcc", "modal_app_id")})
    )
    print(
        "build",
        {
            k: result["build"][k]
            for k in ("compile_seconds", "build_rc", "spill_lines_total")
        },
    )
    print("nonzero spills", result["build"]["nonzero_spill_lines"])
    for key in ("import", "bench_mok", "test_misc"):
        print(key, "rc", result[key]["returncode"])
        print(result[key]["stdout"][-2500:])
        if result[key]["returncode"]:
            print(result[key]["stderr"][-4000:])
    print(json.dumps((result["sweep_default_full"]["payload"] or {}).get("results")))


@app.local_entrypoint()
def ep8(out_dir: str, shapes: str = "30b,235b,beam"):
    for name in shapes.split(","):
        result = run_ep8.remote(name)
        result["modal_app_id"] = app.app_id
        path = Path(out_dir) / f"mok_ep8_{name}_b200.json"
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(name, json.dumps(result["picked"]))
        for precision, run in result["final"].items():
            print(name, precision, json.dumps((run["payload"] or {}).get("results")))
        for key in ("bench_mok_unmodified", "nccl_control"):
            if key in result:
                print(name, key, "rc", result[key]["returncode"])
                print(result[key]["stdout"][-2000:])
