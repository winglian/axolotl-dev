# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Modal driver for ``dispatch_sonicmoe_contract.py``: one strict 8 x H100 job and one
8 x B200 job. Writes ``benchmarks/dispatch_sonicmoe_contract_{h100,b200}.json``.

``MONO_EP_CONTRACT_DEEP_EP=1`` adds a DeepEP layer to both images (a no-op where the image
already ships it), passes ``--deep-ep`` to the worker and writes
``benchmarks/dispatch_sonicmoe_contract_deep_ep_{h100,b200}.json`` instead; without it the
images and the artifacts are the ones of the earlier runs."""

import json
import os
import subprocess  # nosec B404
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
# ``MONO_EP_CONTRACT_RETIME=1``: the compact receive rows (``--retime --fsdp2`` plus the
# DeepEP rows at 20 / 40 / 64 SMs), written to ``dispatch_sonicmoe_contract_overlap_*``;
# ``MONO_EP_CONTRACT_SMOKE=1`` runs the same code on two GPUs (``--smoke``, no DeepEP)
RETIME = os.environ.get("MONO_EP_CONTRACT_RETIME") == "1"
SMOKE = os.environ.get("MONO_EP_CONTRACT_SMOKE") == "1"
# ``MONO_EP_CONTRACT_VARIANTS=a,b`` keeps only those variant labels and
# ``MONO_EP_CONTRACT_TAG=name`` writes ``dispatch_sonicmoe_contract_<name>_*`` instead
VARIANT_FILTER = os.environ.get("MONO_EP_CONTRACT_VARIANTS")
TAG = os.environ.get("MONO_EP_CONTRACT_TAG")
GAP_CPU = os.environ.get("MONO_EP_CONTRACT_GAP_CPU") == "1"
DEEP_EP = os.environ.get("MONO_EP_CONTRACT_DEEP_EP") == "1" or (RETIME and not SMOKE)
# v1.2.1: the commit docker/Dockerfile-cloud-blackwell-uv (by tag) and cicd/deep_ep.py pin;
# DeepEP main needs CUDA 13.1, this tag builds on the images' CUDA 13.0 with the relocated
# cuda/std headers on the include path
DEEP_EP_COMMIT = "9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee"
app = modal.App(
    "axolotl-mono-ep-dispatch-sonicmoe-contract"
    + ("-overlap" if RETIME else "")
    + ("-smoke" if SMOKE else "")
    + (f"-{TAG}" if TAG else "")
    + ("-deep-ep" if DEEP_EP and not RETIME else "")
)


def _with_deepgemm(image):
    """The vendored DeepGEMM build the fused runtime needs (the same steps, in the same
    order, as ``modal_validate.base_image``, so the layers are shared)."""
    return (
        image.run_commands(
            'uv pip install --python "$(command -v python)" setuptools wheel packaging ninja',
            "apt-get update && apt-get install -y libdw-dev libelf-dev software-properties-common",
            "add-apt-repository -y ppa:ubuntu-toolchain-r/test && apt-get update && apt-get install -y g++-13",
        )
        .add_local_dir(
            Path(os.environ.get("MONO_EP_VENDOR_DIR", ROOT / "vendor")),
            "/opt/mono-ep/vendor",
            copy=True,
        )
        .add_local_file(
            ROOT / "tools/fetch_build_dependencies.py",
            "/opt/mono-ep/tools/fetch_build_dependencies.py",
            copy=True,
        )
        .run_commands(
            "python /opt/mono-ep/tools/fetch_build_dependencies.py",
            "cd /opt/mono-ep/vendor/deepgemm && DG_USE_LOCAL_VERSION=0 DG_FORCE_BUILD=1 "
            'CC=gcc-13 CXX=g++-13 MAX_JOBS=4 uv pip install --python "$(command -v python)" --no-build-isolation --no-deps .',
        )
    )


def _with_deep_ep(image, arch):
    """DeepEP at ``DEEP_EP_COMMIT`` built for ``arch`` with the Blackwell cloud image's
    recipe (NVSHMEM from torch's ``nvidia-nvshmem-cu13`` wheel, which setup.py finds on its
    own; a wheel built with setuptools directly, the tag's build backend fails under
    ``pip install .``). Skipped where the image already imports ``deep_ep``."""
    cccl = "-I/usr/local/cuda/include/cccl"
    build = " && ".join(
        [
            "(command -v git >/dev/null || (apt-get update && apt-get install -y git))",
            "mkdir -p /tmp/DeepEP && cd /tmp/DeepEP && git init -q",
            f"git fetch -q --depth=1 https://github.com/deepseek-ai/DeepEP.git {DEEP_EP_COMMIT}",
            "git checkout -q FETCH_HEAD",
            f"TORCH_CUDA_ARCH_LIST={arch} MAX_JOBS=16 CFLAGS={cccl} CXXFLAGS={cccl} "
            f"NVCC_PREPEND_FLAGS={cccl} python setup.py bdist_wheel -d /tmp/deep-ep-wheels",
            'uv pip install --python "$(command -v python)" --no-deps /tmp/deep-ep-wheels/deep_ep-*.whl',
            "cd / && rm -rf /tmp/DeepEP /tmp/deep-ep-wheels",
        ]
    )
    return image.run_commands(
        f'(python -c "import deep_ep" 2>/dev/null && echo "DeepEP shipped by the image") || ({build})',
        "python -c \"import deep_ep, importlib.metadata as m; print('DeepEP', m.version('deep_ep'), deep_ep.Buffer)\"",
    )


def _with_sources(image):
    return (
        image.add_local_dir(ROOT / "src", "/probe/mono_ep_src")
        .add_local_dir(ROOT / "../../src", "/probe/framework")
        .add_local_file(
            ROOT / "examples/dispatch_sonicmoe_contract.py", "/probe/contract.py"
        )
        .add_local_file(
            ROOT / "examples/sonicmoe_contract_profiles.py",
            "/probe/sonicmoe_contract_profiles.py",
        )
    )


# H100: stock image plus the vendored DeepGEMM build and the SonicMoE hub kernel's CuteDSL
# build deps.
hopper_image = _with_deepgemm(
    modal.Image.from_registry("axolotlai/axolotl-uv:main-latest")
).run_commands(
    'uv pip install --python "$(command -v python)" '
    '"nvidia-cutlass-dsl==4.8.0" "apache-tvm-ffi>=0.1.10"'
)
if DEEP_EP:
    hopper_image = _with_deep_ep(hopper_image, "9.0")
hopper_image = _with_sources(hopper_image)
# B200: the kernel lane's image, which carries axolotl's `blackwell` extra (quack, flash-attn-4,
# CuteDSL 4.8.1). The hub-vendored quack 0.6.1 still uses the pre-4.8 `nvvm.mbarrier_arrive`
# call form on its SM100 2-CTA path; the worker applies the kernels plugin's shim for it.
# The cloud Blackwell image ships DeepEP v1.2.1 (sm_100), so the DeepEP layer only checks it.
blackwell_image = _with_deepgemm(
    modal.Image.from_registry(
        "axolotlai/axolotl-cloud-uv:main-py3.14-cu130-2.14.1-blackwell"
    )
)
if DEEP_EP:
    blackwell_image = _with_deep_ep(blackwell_image, "10.0")
blackwell_image = _with_sources(blackwell_image)


def _run_worker(
    expected_gpu,
    deep_ep=False,
    deep_ep_num_sms_max=40,
    gpus=8,
    retime=False,
    smoke=False,
    variants=None,
    gap_cpu=False,
):
    import torch

    actual = torch.cuda.get_device_name()
    if expected_gpu not in actual:
        raise RuntimeError(f"expected {expected_gpu}, allocated {actual}")
    print(f"allocated {actual}", flush=True)
    started = time.time()
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={gpus}",
        "/probe/contract.py",
        "--worker",
        "--profiles",
        expected_gpu,
        *(
            ["--deep-ep", "--deep-ep-num-sms-max", str(deep_ep_num_sms_max)]
            if deep_ep
            else []
        ),
        *(["--retime", "--fsdp2"] if retime or smoke else []),
        *(["--smoke"] if smoke else []),
        *(["--variants", variants] if variants else []),
        *(["--gap-cpu"] if gap_cpu else []),
    ]
    lines = []
    with subprocess.Popen(  # nosec B603
        command,
        env={
            **os.environ,
            "PYTHONPATH": "/probe/framework:/probe/mono_ep_src",
            "OMP_NUM_THREADS": "1",
            # the fused H100 reference line's settings; read only by the SM90 blockwise
            # backward, inert on B200 / BF16
            "MONO_EP_SM90_FP8_TRANSPORT": "1",
            "MONO_EP_SM90_FP8_COMBINE": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        returncode = process.wait()
    result = [line for line in lines if line.startswith("RESULT ")]
    if not result:
        raise RuntimeError(f"worker printed no RESULT line (exit {returncode})")
    payload = json.loads(result[-1][len("RESULT ") :])
    payload["worker_returncode"] = returncode
    payload["wall_seconds"] = time.time() - started
    payload["gpu_hours"] = gpus * payload["wall_seconds"] / 3600
    payload["modal_task_id"] = os.environ.get("MODAL_TASK_ID")
    return payload


@app.function(image=hopper_image, gpu="H100!:8", cpu=32, memory=131072, timeout=5400)
def contract_h100(
    deep_ep: bool = False,
    deep_ep_num_sms_max: int = 40,
    retime: bool = False,
    variants: str | None = None,
):
    return _run_worker(
        "H100", deep_ep, deep_ep_num_sms_max, retime=retime, variants=variants
    )


@app.function(image=blackwell_image, gpu="B200:8", cpu=32, memory=131072, timeout=5400)
def contract_b200(
    deep_ep: bool = False,
    deep_ep_num_sms_max: int = 40,
    retime: bool = False,
    variants: str | None = None,
):
    return _run_worker(
        "B200", deep_ep, deep_ep_num_sms_max, retime=retime, variants=variants
    )


@app.function(image=hopper_image, gpu="H100!:2", cpu=16, memory=65536, timeout=2400)
def contract_smoke_h100(variants: str | None = None, gap_cpu: bool = False):
    return _run_worker("H100", gpus=2, smoke=True, variants=variants, gap_cpu=gap_cpu)


@app.local_entrypoint()
def main(gpu: str = "H100", deep_ep_num_sms_max: int = 40):
    """``--deep-ep-num-sms-max 64`` extends the DeepEP SM sweep past the plugin's 256 MiB
    buffer (the H100 artifact ran the default 40); the retime always sizes it for 64."""
    if SMOKE:
        result = contract_smoke_h100.remote(variants=VARIANT_FILTER, gap_cpu=GAP_CPU)
    else:
        function = {"H100": contract_h100, "B200": contract_b200}[gpu]
        result = function.remote(
            deep_ep=DEEP_EP,
            deep_ep_num_sms_max=64 if RETIME else deep_ep_num_sms_max,
            retime=RETIME,
            variants=VARIANT_FILTER,
        )
    result["modal_app_id"] = app.app_id
    result["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    suffix = "_deep_ep" if DEEP_EP and not RETIME else ""
    if RETIME or SMOKE:
        suffix = "_overlap_smoke" if SMOKE else "_overlap"
    if TAG:
        suffix = f"_{TAG}_smoke" if SMOKE else f"_{TAG}"
    if GAP_CPU:
        suffix += "_gapcpu"
    out = ROOT / f"benchmarks/dispatch_sonicmoe_contract{suffix}_{gpu.lower()}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out}")
    if result["worker_returncode"]:
        raise RuntimeError(f"{gpu} worker exited {result['worker_returncode']}")
