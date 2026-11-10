# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run the FSDP2 weight-layout cache check on B200s.

``--gpus 4`` runs dp_shard 2 x ep 2 on all four GPUs and dp_shard 2 x ep 1 on
two of them; ``--gpus 1`` is a single-GPU smoke run (dp_shard 1 x ep 1). Each
shape runs against the current package source and, when
``MONO_EP_FSDP2_BASELINE_SRC`` names a directory, against that source too
(e.g. the round-1 weight cache), in the same container.
"""

import json
import os
import subprocess  # nosec B404
from pathlib import Path

import modal

app = modal.App(os.environ.get("MONO_EP_MODAL_APP", "axolotl-fsdp2-weight-cache"))
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(root / "src", "/probe/src").add_local_file(
        root / "examples/validate_fsdp2_weight_cache.py", "/probe/validate.py"
    )
    baseline = os.environ.get("MONO_EP_FSDP2_BASELINE_SRC")
    if baseline:
        image = image.add_local_dir(baseline, "/probe/src_baseline")


def _validate(expected_gpus, shapes, extra, nccl_ctas="", sweep=()):
    import torch

    names = {torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
    if torch.cuda.device_count() != expected_gpus or any(
        "B200" not in n for n in names
    ):
        return dict(returncode=2, stderr=f"wrong devices {names}", runs=[])
    sources = [("current", "/probe/src")]
    if Path("/probe/src_baseline").exists():
        sources.insert(0, ("baseline", "/probe/src_baseline"))
    runs = []
    plans = [(s, src, list(extra), None) for s in shapes for src in sources]
    # Deadlock-freedom sweep: each entry is (reserved_sms, --only cases), current source only
    for reserved, only in sweep:
        plans += [
            (
                s,
                sources[-1],
                [*extra, "--reserved-sms", str(reserved), "--only", only],
                reserved,
            )
            for s in shapes
        ]
    for (ep, gpus), (label, source), run_extra, sweep_reserved in plans:
        command = [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={gpus}",
            "/probe/validate.py",
            "--ep",
            str(ep),
            *run_extra,
        ]
        result = subprocess.run(  # nosec B603
            command,
            env={
                **os.environ,
                "PYTHONPATH": source,
                "OMP_NUM_THREADS": "1",
                "MONO_EP_WEIGHT_CACHE_COMPILED": os.environ.get(
                    "MONO_EP_WEIGHT_CACHE_COMPILED", "1"
                ),
                "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpus)),
                **({"NCCL_MAX_CTAS": nccl_ctas} if nccl_ctas else {}),
            },
            capture_output=True,
            text=True,
            timeout=1500 if sweep_reserved is None else 600,
            check=False,
        )
        payload = None
        for line in result.stdout.splitlines():
            if line.startswith("FSDP2_JSON "):
                payload = json.loads(line[len("FSDP2_JSON ") :])
        runs.append(
            dict(
                source=label,
                sweep_reserved_sms=sweep_reserved,
                extra=run_extra,
                ep=ep,
                gpus=gpus,
                returncode=result.returncode,
                result=payload,
                stdout=result.stdout[-20000:],
                stderr=result.stderr[-20000:],
            )
        )
    return dict(
        returncode=max([r["returncode"] for r in runs] or [2]),
        device=sorted(names),
        torch=torch.__version__,
        runs=runs,
    )


@app.function(image=image, gpu="B200:4", cpu=32, memory=131072, timeout=3600)
def validate4(shapes, extra, nccl_ctas="", sweep=()):
    return _validate(4, shapes, extra, nccl_ctas, sweep)


@app.function(image=image, gpu="B200", cpu=16, memory=65536, timeout=3600)
def validate1(shapes, extra):
    return _validate(1, shapes, extra)


@app.local_entrypoint()
def main(
    out: str,
    gpus: int = 4,
    shapes: str = "",
    extra: str = "",
    nccl_ctas: str = "",
    sweep: str = "",
):
    """``--sweep 16:default_global_hook,uncached;32:...`` adds reserved-SM runs."""
    shapes = shapes or ("2:4,1:2" if gpus == 4 else "1:1")
    parsed = [tuple(int(v) for v in part.split(":")) for part in shapes.split(",")]
    sweep_plan = [
        (int(part.split(":", 1)[0]), part.split(":", 1)[1])
        for part in sweep.split(";")
        if part
    ]
    result = (
        validate4.remote(parsed, extra.split(), nccl_ctas, sweep_plan)
        if gpus == 4
        else validate1.remote(parsed, extra.split())
    )
    result["nccl_max_ctas"] = nccl_ctas or None
    result["modal_app_id"] = app.app_id
    result["baseline_source"] = os.environ.get("MONO_EP_FSDP2_BASELINE_SRC")
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("returncode", "device", "modal_app_id")}))
    for run in result["runs"]:
        print(
            run["source"],
            "ep",
            run["ep"],
            "sweep_reserved_sms",
            run.get("sweep_reserved_sms"),
            "returncode",
            run["returncode"],
        )
        for case in (run["result"] or {}).get("cases", []):
            print(
                " ",
                case["case"],
                "passed" if case.get("passed") else "FAILED",
                case.get("expected"),
                case.get("error", "")[:300],
            )
            for step in case.get("steps", []):
                print("   ", step["step"], step["worst"], step["cache_delta"])
        if run["returncode"]:
            print(run["stderr"][-4000:])
