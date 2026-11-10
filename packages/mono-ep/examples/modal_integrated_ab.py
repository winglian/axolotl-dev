# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Same-allocation single-rank A/B of two source trees (baseline vendor vs this tree) on one strict H100.

Both trees are built into the image. Each round runs one probe process per tree (order alternates
per round), so samples from both trees interleave on the same GPU. The first baseline process
saves its outputs; every later process reports bitwise/relative parity against them.
"""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-integrated-ab")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    base_vendor = os.environ["MONO_EP_BASE_VENDOR_DIR"]
    image = (
        base_image.add_local_dir(base_vendor, "/opt/base-mono-ep/vendor", copy=True)
        .add_local_file(
            root / "tools/fetch_build_dependencies.py",
            "/opt/base-mono-ep/tools/fetch_build_dependencies.py",
            copy=True,
        )
        .run_commands(
            "python /opt/base-mono-ep/tools/fetch_build_dependencies.py",
            "cd /opt/base-mono-ep/vendor/deepgemm && DG_USE_LOCAL_VERSION=0 DG_FORCE_BUILD=1 "
            'CC=gcc-13 CXX=g++-13 MAX_JOBS=4 uv pip install --python "$(command -v python)" '
            "--no-build-isolation --no-deps --target /opt/dg_base .",
        )
        .add_local_dir(root / "src", "/probe/src")
        .add_local_file(
            root / "examples/probe_integrated_backward.py", "/probe/backward.py"
        )
        .add_local_file(
            root / "examples/probe_integrated_forward.py", "/probe/forward.py"
        )
    )

TREES = {
    "base": dict(pythonpath="/opt/dg_base:/probe/src", cache="/root/dj_base"),
    "integrated": dict(pythonpath="/probe/src", cache="/root/dj_integrated"),
}


@app.function(image=image, gpu="H100!", cpu=16, memory=65536, timeout=7200)
def ab_many(configs, warmup, iters, rounds):
    """Several (direction, routing, variants) A/Bs in one container, one after another."""
    import torch

    name = torch.cuda.get_device_name()
    out = []
    for direction, routing, variants in configs:
        result = _ab(direction, routing, variants, warmup, iters, rounds)
        out.append(dict(direction=direction, routing=routing, **result))
        if result["returncode"]:
            break
    return dict(
        returncode=max(r["returncode"] for r in out),
        device=name,
        configs=out,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
    )


@app.function(image=image, gpu="H100!", cpu=16, memory=65536, timeout=5400)
def ab(direction, routing, variants, warmup, iters, rounds):
    return _ab(direction, routing, variants, warmup, iters, rounds)


def _ab(direction, routing, variants, warmup, iters, rounds):
    import torch

    name = torch.cuda.get_device_name()
    if "H100" not in name:
        return dict(returncode=2, stderr=f"wrong device {name}", stdout="", device=name)
    ref = f"/root/ref_{direction}_{routing}.pt"
    runs = []
    for round_idx in range(rounds):
        order = ["base", "integrated"] if round_idx % 2 == 0 else ["integrated", "base"]
        for tree in order:
            if not variants.get(tree):
                continue
            command = [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=1",
                f"/probe/{direction}.py",
                "--routing",
                routing,
                "--variants",
                variants[tree],
                "--warmup",
                str(warmup),
                "--iters",
                str(iters),
                "--rounds",
                "1",
                "--ref-path",
                ref,
                "--ref-tag",
                f"{tree}/round{round_idx}",
            ]
            result = subprocess.run(  # nosec B603
                command,
                env={
                    **os.environ,
                    "PYTHONPATH": TREES[tree]["pythonpath"],
                    "DG_JIT_CACHE_DIR": TREES[tree]["cache"],
                    "OMP_NUM_THREADS": "1",
                    "MONO_EP_SM90_FP8_TRAP_TIMEOUTS": "1",
                    "MONO_EP_SM90_FP8_TRANSPORT": "1",
                    "MONO_EP_SM90_FP8_COMBINE": "1",
                },
                capture_output=True,
                text=True,
                timeout=2400,
                check=False,
            )
            payload = None
            for line in result.stdout.splitlines():
                if line.startswith("DIAG_JSON "):
                    payload = json.loads(line[len("DIAG_JSON ") :])
            runs.append(
                dict(
                    tree=tree,
                    round=round_idx,
                    returncode=result.returncode,
                    diag=payload,
                    stderr=result.stderr[-6000:] if result.returncode else "",
                )
            )
            if result.returncode or payload is None:
                return dict(returncode=1, device=name, runs=runs)
    return dict(
        returncode=0,
        device=name,
        runs=runs,
        modal_task_id=os.environ.get("MODAL_TASK_ID"),
    )


def _median(values):
    values = sorted(values)
    n = len(values)
    return (
        None
        if not n
        else (values[n // 2] if n % 2 else 0.5 * sum(values[n // 2 - 1 : n // 2 + 1]))
    )


def _summarize(result):
    summary: dict[str, dict] = {}
    for run in result.get("runs", []):
        if not run["diag"]:
            continue
        for variant, value in run["diag"]["results"].items():
            entry = summary.setdefault(
                f"{run['tree']}/{variant}", dict(samples_ms=[], parity_vs_ref=[])
            )
            entry["samples_ms"].extend(value["samples_ms"])
            entry["parity_vs_ref"].append(
                dict(round=run["round"], parity=value.get("parity_vs_ref"))
            )
    for entry in summary.values():
        entry["samples"] = len(entry["samples_ms"])
        entry["median_ms"] = _median(entry["samples_ms"])
    return summary


@app.local_entrypoint()
def main(
    out: str,
    direction: str = "backward",
    routing: str = "balanced",
    base: str = "orig:",
    integrated: str = "default:",
    forward_base: str = "",
    forward_integrated: str = "",
    warmup: int = 5,
    iters: int = 5,
    rounds: int = 4,
):
    """base/integrated: variant lists "name:ENV=V+ENV2=V2,..." for each tree ('' skips a tree).

    direction and routing may be comma lists; more than one (direction, routing) pair runs them all in
    one container (same GPU), forward pairs using forward_base/forward_integrated when given.
    """
    pairs = [(d, r) for d in direction.split(",") for r in routing.split(",")]
    if len(pairs) > 1:
        configs = [
            (
                d,
                r,
                dict(
                    base=forward_base or base if d == "forward" else base,
                    integrated=forward_integrated or integrated
                    if d == "forward"
                    else integrated,
                ),
            )
            for d, r in pairs
        ]
        result = ab_many.remote(configs, warmup, iters, rounds)
        for config in result["configs"]:
            config["summary"] = _summarize(config)
        result.update(
            warmup_per_process=warmup,
            iters_per_process=iters,
            rounds=rounds,
            interleaving="per (direction, routing): one probe process per tree per round, tree order "
            "alternates per round, variants alternate within each process; all pairs in one container "
            "and GPU, run one after another",
            modal_app_id=os.environ.get("MODAL_APP_ID") or app.app_id,
        )
        Path(out).write_text(json.dumps(result, indent=1))
        print(
            json.dumps(
                dict(
                    returncode=result["returncode"],
                    device=result.get("device"),
                    modal_app_id=result["modal_app_id"],
                    medians={
                        f"{c['direction']}/{c['routing']}": {
                            k: (v["median_ms"], v["samples"])
                            for k, v in c["summary"].items()
                        }
                        for c in result["configs"]
                    },
                )
            )
        )
        if result["returncode"]:
            for config in result["configs"]:
                for run in config.get("runs", []):
                    if run["stderr"]:
                        print(run["stderr"][-4000:])
            raise RuntimeError("integrated A/B failed")
        return
    result = ab.remote(
        direction,
        routing,
        dict(base=base, integrated=integrated),
        warmup,
        iters,
        rounds,
    )
    summary: dict[str, dict] = {}
    for run in result.get("runs", []):
        if not run["diag"]:
            continue
        for variant, value in run["diag"]["results"].items():
            entry = summary.setdefault(
                f"{run['tree']}/{variant}", dict(samples_ms=[], parity_vs_ref=[])
            )
            entry["samples_ms"].extend(value["samples_ms"])
            entry["parity_vs_ref"].append(
                dict(round=run["round"], parity=value.get("parity_vs_ref"))
            )
    for entry in summary.values():
        entry["samples"] = len(entry["samples_ms"])
        entry["median_ms"] = _median(entry["samples_ms"])
    result.update(
        direction=direction,
        routing=routing,
        warmup_per_process=warmup,
        iters_per_process=iters,
        rounds=rounds,
        interleaving="one probe process per tree per round, tree order alternates per round, "
        "variants alternate within each process; same container and GPU",
        summary=summary,
        modal_app_id=os.environ.get("MODAL_APP_ID") or app.app_id,
    )
    Path(out).write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            dict(
                returncode=result["returncode"],
                device=result.get("device"),
                modal_app_id=result["modal_app_id"],
                medians={k: (v["median_ms"], v["samples"]) for k, v in summary.items()},
            )
        )
    )
    if result["returncode"]:
        for run in result.get("runs", []):
            if run["stderr"]:
                print(run["stderr"][-4000:])
        raise RuntimeError("integrated A/B failed")
