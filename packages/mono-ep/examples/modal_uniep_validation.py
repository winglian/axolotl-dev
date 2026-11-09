# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Independent fused-EP validation using the isolated UniEP build image."""

import json
import os
import signal
import subprocess
import threading
from pathlib import Path

import modal

app = modal.App("axolotl-uniep-validation")
REVISION = "7908e4ea9010bb2238f9f05b28c904fd4b71c60a"
validation_image = None
if modal.is_local():
    from modal_uniep import image

    validation_image = (
        image.add_local_file(
            Path(__file__).resolve().parents[1] / "tools/patch_uniep_swiglu.py",
            "/opt/patch_uniep_swiglu.py",
            copy=True,
        )
        .run_commands(
            "env -u PIP_CONSTRAINT python -m pip install 'cuda.core==1.0.1' 'cuda-python>=12.9,<13' "
            "'nvidia-nvshmem-cu12==3.6.5' 'nvshmem4py-cu12==0.3.0' "
            "'Cython>=0.29.24' 'nvidia-ml-py>=12.0'",
            "python -c 'import importlib.util, pathlib, shutil; "
            'root = pathlib.Path(importlib.util.find_spec("triton_dist").origin).parent; '
            'shutil.copytree("/opt/triton-dist/python/triton_dist/language/tma", '
            'root / "language/tma", dirs_exist_ok=True)'
            "'",
            "python -c 'from cuda.core import Device; import nvshmem.core'",
            "python /opt/patch_uniep_swiglu.py",
        )
        .add_local_file(
            Path(__file__).with_name("validate_uniep.py"), "/opt/validate_uniep.py"
        )
    )


@app.function(image=validation_image, gpu="H100:4", cpu=16, memory=65536, timeout=1800)
def validate(inflight: bool = False, training: bool = False):
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "/opt/validate_uniep.py",
    ]
    if inflight:
        command.append("--inflight")
    if training:
        command.append("--training")
    output = []
    expired = threading.Event()
    process = subprocess.Popen(
        command,
        env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    def terminate():
        expired.set()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(1700, terminate)
    timer.start()
    try:
        for line in process.stdout:
            output.append(line)
            print(line, end="", flush=True)
        returncode = process.wait()
    finally:
        timer.cancel()
        timer.join()
        process.stdout.close()
        if process.poll() is None:
            terminate()
            process.wait()
    return {
        "revision": REVISION,
        "world_size": 4,
        "inflight": inflight,
        "training": training,
        "returncode": returncode,
        "timed_out": expired.is_set(),
        "stdout": "".join(output),
        "stderr": "",
    }


@app.local_entrypoint()
def main(inflight: bool = False, inspect_memory: bool = False, training: bool = False):
    if inspect_memory:
        print(inspect_memory_runtime.remote())
        return
    report = validate.remote(inflight, training)
    print(json.dumps(report))
    if report["returncode"]:
        raise RuntimeError("UniEP independent validation failed; see captured report")


@app.function(image=validation_image, timeout=60)
def inspect_memory_runtime():
    import inspect

    import nvshmem.core

    functions = {
        name: getattr(nvshmem.core, name)
        for name in ("free_tensor", "finalize", "free", "get_peer_tensor")
    }
    functions["free_all_buffers"] = nvshmem.core.finalize.__globals__[
        "memory"
    ]._free_all_buffers
    functions["tensor_get_buffer"] = nvshmem.core.free_tensor.__globals__[
        "tensor_get_buffer"
    ]
    return {name: inspect.getsource(fn) for name, fn in functions.items()}
