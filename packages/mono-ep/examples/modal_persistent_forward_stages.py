# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Time the persistent MoE training step on eight B200s with a chosen SM100 backward pipeline depth."""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App("axolotl-persistent-forward-stages")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_file(
            root / "examples/benchmark_persistent_forward.py", "/probe/benchmark.py"
        )
        .add_local_file(
            root / "examples/benchmark_persistent_training.py", "/probe/training.py"
        )
    )


@app.function(image=image, gpu="B200:8", cpu=32, memory=131072, timeout=7200)
def benchmark(
    retain_activations=False,
    training=False,
    include_router=False,
    profile=False,
    compute_dtype="bf16",
    compare_fusion=False,
    recompute=False,
    peer_backward=False,
    compare_ready=False,
    compare_peer_return=False,
    compare_owned=False,
    compare_nccl=False,
    compare_native=False,
    compiled=False,
    stages=0,
    variants=(),
    bump_versions=False,
):
    """``variants``: extra (label, env) runs in the same allocation, executed before the default run."""
    import torch

    name = torch.cuda.get_device_name()
    if "B200" not in name or torch.cuda.device_count() != 8:
        return dict(
            returncode=2, stdout="", stderr=f"wrong devices {name}", device=name
        )
    env = {**os.environ, "PYTHONPATH": "/probe/src", "OMP_NUM_THREADS": "1"}
    if stages:
        env["MONO_EP_SM100_BACKWARD_STAGES"] = str(stages)
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "/probe/training.py" if training else "/probe/benchmark.py",
        *(["--retain-activations"] if retain_activations and not training else []),
        *(["--include-router"] if include_router and not training else []),
        *(["--profile"] if profile else []),
        *(["--compare-fusion"] if compare_fusion else []),
        *(["--recompute"] if recompute else []),
        *(["--peer-backward"] if peer_backward else []),
        *(["--compare-ready"] if compare_ready else []),
        *(["--compare-peer-return"] if compare_peer_return else []),
        *(["--compare-owned"] if compare_owned else []),
        *(["--compare-nccl"] if compare_nccl else []),
        *(["--compare-native"] if compare_native else []),
        *(["--compiled"] if compiled else []),
        *(["--compute-dtype", compute_dtype] if training else []),
        *(["--bump-versions"] if bump_versions else []),
    ]
    runs = []
    for label, extra_env in [*variants, ("default", {})]:
        result = subprocess.run(  # nosec B603 B607
            command,
            env={**env, **extra_env},
            capture_output=True,
            text=True,
            timeout=2400,
            check=False,
        )
        runs.append(
            dict(
                label=label,
                env=extra_env,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr[-20000:],
            )
        )
    final = runs[-1]
    return dict(
        returncode=max(r["returncode"] for r in runs),
        stdout=final["stdout"],
        stderr=final["stderr"],
        device=name,
        stages=stages,
        runs=runs,
        order="variants first in the given order, then the default build",
    )


@app.local_entrypoint()
def main(
    retain_activations: bool = False,
    training: bool = False,
    include_router: bool = False,
    profile: bool = False,
    compute_dtype: str = "bf16",
    compare_fusion: bool = False,
    recompute: bool = False,
    peer_backward: bool = False,
    compare_ready: bool = False,
    compare_peer_return: bool = False,
    compare_owned: bool = False,
    compare_nccl: bool = False,
    compare_native: bool = False,
    compiled: bool = False,
    stages: int = 0,
    variants: str = "",
    out: str = "",
    bump_versions: bool = False,
):
    """``variants``: ``label=KEY:VAL+KEY:VAL`` items separated by commas, run before the default."""
    if compiled and (not training or not recompute or profile):
        raise ValueError(
            "Compiled timing requires recomputing training without profiling"
        )
    if compare_fusion and (not training or profile):
        raise ValueError("Fusion comparison requires training timing without profiling")
    if profile and not training:
        raise ValueError("Profiling currently requires --training")
    if training and include_router:
        raise ValueError("Router timing currently applies to forward-only runs")
    if compute_dtype not in ("bf16", "mxfp8") or (
        compute_dtype != "bf16" and not training
    ):
        raise ValueError("--compute-dtype bf16|mxfp8 applies to training")
    if recompute and not training:
        raise ValueError("Recompute requires training")
    if peer_backward and (not recompute):
        raise ValueError("Peer backward requires recomputing training")
    if compare_ready and (
        not peer_backward or compare_fusion or profile or compute_dtype != "bf16"
    ):
        raise ValueError("Readiness comparison requires BF16 peer timing")
    if compare_peer_return and (
        not peer_backward
        or compare_ready
        or compare_fusion
        or profile
        or compute_dtype != "bf16"
    ):
        raise ValueError("Peer-return comparison requires isolated BF16 peer timing")
    if compare_native and (
        not peer_backward
        or compare_ready
        or compare_peer_return
        or compare_fusion
        or compare_owned
        or profile
        or compute_dtype != "bf16"
    ):
        raise ValueError(
            "Native comparison requires isolated BF16 recomputing peer timing"
        )
    if compare_nccl and (
        not peer_backward
        or compare_ready
        or compare_peer_return
        or compare_fusion
        or compare_owned
        or profile
    ):
        raise ValueError("NCCL comparison requires isolated recomputing peer timing")
    if compare_owned and (
        not peer_backward
        or compare_ready
        or compare_peer_return
        or compare_fusion
        or profile
        or compute_dtype != "bf16"
    ):
        raise ValueError("Owned-block comparison requires isolated BF16 peer timing")
    result = benchmark.remote(
        retain_activations,
        training,
        include_router,
        profile,
        compute_dtype,
        compare_fusion,
        recompute,
        peer_backward,
        compare_ready,
        compare_peer_return,
        compare_owned,
        compare_nccl,
        compare_native,
        compiled,
        stages,
        [
            (
                item.split("=", 1)[0],
                dict(kv.split(":", 1) for kv in item.split("=", 1)[1].split("+")),
            )
            for item in filter(None, variants.split(","))
        ],
        bump_versions,
    )
    result["modal_app_id"] = app.app_id
    result["launch"] = dict(
        training=training,
        recompute=recompute,
        peer_backward=peer_backward,
        compare_nccl=compare_nccl,
        compare_native=compare_native,
        compiled=compiled,
        compute_dtype=compute_dtype,
        variants=variants,
        bump_versions=bump_versions,
    )
    if out:
        Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}))
    if result["returncode"]:
        raise RuntimeError("Persistent forward benchmark failed")
