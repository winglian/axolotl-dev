# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate compiled checkpointed native FP8 training on two H100s."""

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("axolotl-hopper-native-fp8-training")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = base_image.add_local_dir(root / "src", "/probe/src").add_local_file(
        root / "examples/validate_persistent_training.py", "/probe/training.py"
    )


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=1800)
def validate(
    control=False,
    compact=False,
    trap_timeouts=False,
    fp8_transport=False,
    fp8_combine=False,
):
    if fp8_combine and not fp8_transport:
        raise ValueError("FP8 combine requires FP8 input transport")
    if control and fp8_transport:
        raise ValueError("FP8 transport requires the native backward path")
    if control and compact:
        raise ValueError("Compact layout validation requires the native backward path")
    if control and trap_timeouts:
        raise ValueError(
            "Trap-only timeout validation requires the native backward path"
        )
    child_env = {**os.environ, "PYTHONPATH": "/probe/src", "OMP_NUM_THREADS": "1"}
    child_env["MONO_EP_SM90_FP8_TRANSPORT"] = "1" if fp8_transport else "0"
    child_env["MONO_EP_SM90_FP8_COMBINE"] = "1" if fp8_combine else "0"
    for name in (
        "MONO_EP_SM90_FP8_COMPACT_FORWARD",
        "MONO_EP_SM90_FP8_COMPACT_BACKWARD",
        "MONO_EP_SM90_FP8_TRAP_TIMEOUTS",
    ):
        selected = trap_timeouts if name.endswith("TRAP_TIMEOUTS") else compact
        if selected:
            child_env[name] = "1"
        else:
            child_env.pop(name, None)
    records = {}
    for master_dtype in ("float32", "bf16"):
        result = subprocess.run(
            [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "/probe/training.py",
                "--compute-dtype",
                "blockwise",
                "--recompute",
                *([] if control else ["--peer-backward", "--native-backward"]),
                "--compiled",
                "--checkpointed",
                "--master-dtype",
                master_dtype,
            ],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=840,
            check=False,
        )
        records[master_dtype] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode:
            break
    return {
        "selection": {
            "control": control,
            "backward_input_transport": "fp8" if fp8_transport else "bf16",
            "backward_combine_transport": "fp8" if fp8_combine else "bf16",
            "forward_layout": "compact384" if compact else "baseline512",
            "backward_layout": "compact384" if compact else "baseline512",
            "timeout_handler": (
                "trap_only_no_diagnostic_calls" if trap_timeouts else "diagnostic_calls"
            ),
            "compiled": True,
            "checkpointed": True,
            "master_dtypes": ["float32", "bf16"],
        },
        "records": records,
    }


@app.local_entrypoint()
def main(
    control: bool = False,
    compact: bool = False,
    trap_timeouts: bool = False,
    fp8_transport: bool = False,
    fp8_combine: bool = False,
):
    if control and compact:
        raise ValueError("--control and --compact cannot be combined")
    if control and trap_timeouts:
        raise ValueError("--control and --trap-timeouts cannot be combined")
    if control and fp8_transport:
        raise ValueError("--control and --fp8-transport cannot be combined")
    if fp8_combine and not fp8_transport:
        raise ValueError("--fp8-combine requires --fp8-transport")
    result = validate.remote(
        control, compact, trap_timeouts, fp8_transport, fp8_combine
    )
    print(json.dumps(result))
    if any(record["returncode"] for record in result["records"].values()):
        raise RuntimeError("Hopper native training validation failed")
