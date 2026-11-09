# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Reproducible DeepGEMM kernel validation on Hopper and Blackwell."""

import os
import subprocess  # nosec B404
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("axolotl-mono-ep-training")
base_image = (
    modal.Image.from_registry("axolotlai/axolotl-uv:main-latest")
    .run_commands(
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
image = base_image.add_local_dir(ROOT / "src", "/probe/src").add_local_file(
    ROOT / "examples/validate.py", "/probe/validate.py"
)
_ADAPTER = ROOT / "../../src/axolotl/integrations/expert_parallel/mono_ep.py"
if _ADAPTER.exists():
    image = image.add_local_file(_ADAPTER, "/probe/axolotl_mono_ep_adapter.py")
comparison_image = (
    base_image.run_commands(
        'uv pip install --python "$(command -v python)" '
        '"sonic-moe @ git+https://github.com/Dao-AILab/sonic-moe.git@7396f3e604827d8186c2e16e64b28ee33d3defd0"',
    )
    .add_local_dir(ROOT / "src", "/probe/src")
    .add_local_file(ROOT / "examples/benchmark.py", "/probe/benchmark.py")
    .add_local_file(ROOT / "examples/validate.py", "/probe/validate.py")
    .add_local_file(
        ROOT / "examples/benchmark_pipeline.py", "/probe/benchmark_pipeline.py"
    )
)
h200_image = image.add_local_file(
    ROOT / "examples/benchmark_h200.py", "/probe/benchmark_h200.py"
)


@app.function(image=h200_image, gpu="H200:8", cpu=32, memory=131072, timeout=3600)
def compare_h200(profile=False):
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "/probe/benchmark_h200.py",
    ]
    with subprocess.Popen(  # nosec B603
        command,
        env={
            **os.environ,
            "PYTHONPATH": "/probe/src",
            "OMP_NUM_THREADS": "1",
            "MOE_PROFILE": "1" if profile else "0",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        lines = []
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        if process.wait():
            raise RuntimeError("H200 comparison failed; see streamed logs")
    return "".join(lines)


fft_image = image.add_local_dir(ROOT / "../../src", "/probe/framework").add_local_file(
    ROOT / "../../tests/e2e/multigpu/test_ep_fft_pipeline.py", "/probe/fft.py"
)


def run_fp8_fft(format, resume=False):
    result = subprocess.run(  # nosec B603 B607
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            "/probe/fft.py",
            "--worker",
        ],
        env={
            **os.environ,
            "PYTHONPATH": "/probe/framework:/probe/src",
            "AXOLOTL_EP_FP8_FORMAT": format,
            "AXOLOTL_EP_FP8_RESUME": "1" if resume else "0",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=1500,
        check=False,
    )
    print(result.stdout)
    print(result.stderr)
    result.check_returncode()
    return result.stdout


@app.function(image=fft_image, gpu="H100:4", cpu=16, timeout=1800)
def fp8_fft_hopper(resume=False):
    return run_fp8_fft("blockwise", resume)


@app.function(image=fft_image, gpu="B200:4", cpu=16, timeout=1800)
def fp8_fft_blackwell(resume=False):
    return run_fp8_fft("mxfp8", resume)


def validate(
    distributed=False,
    quantization=False,
    sonic=False,
    fp8_gemm=False,
    fused_fp8=False,
    changing_routing=False,
    framework_ep=False,
):
    command = (
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "/probe/validate.py",
            "--distributed",
        ]
        if distributed
        else ["python", "/probe/validate.py"]
    )
    if quantization:
        command.append("--quantization")
    if sonic:
        command.append("--sonic")
    if fp8_gemm:
        command.append("--fp8-gemm")
    if fused_fp8:
        command.append("--fused-fp8")
    if framework_ep:
        command.append("--framework-ep")
    if changing_routing:
        command.append("--changing-routing")
    result = subprocess.run(  # nosec B603
        command,
        env={**os.environ, "PYTHONPATH": "/probe/src"},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    print(result.stdout)
    print(result.stderr)
    result.check_returncode()
    return result.stdout


@app.function(image=image, gpu="H100", timeout=1200)
def hopper():
    return validate()


@app.function(image=image, gpu="B200", timeout=1200)
def blackwell():
    return validate()


@app.function(image=image, gpu="B200:2", timeout=1200)
def blackwell_ep(fused_fp8=False):
    return validate(distributed=True, fused_fp8=fused_fp8)


@app.function(image=image, gpu="B200", timeout=1200)
def blackwell_quantization():
    return validate(quantization=True)


@app.function(image=image, gpu="H100", timeout=1200)
def hopper_fp8_gemm():
    return validate(fp8_gemm=True)


@app.function(image=image, gpu="B200", timeout=1200)
def blackwell_fp8_gemm():
    return validate(fp8_gemm=True)


@app.function(image=image, gpu="H100:2", timeout=1200)
def hopper_fp8_ep(changing_routing=False, framework_ep=False):
    return validate(
        distributed=True,
        fp8_gemm=True,
        changing_routing=changing_routing,
        framework_ep=framework_ep,
    )


@app.function(image=image, gpu="B200:2", timeout=1200)
def blackwell_fp8_ep(changing_routing=False, framework_ep=False):
    return validate(
        distributed=True,
        fp8_gemm=True,
        changing_routing=changing_routing,
        framework_ep=framework_ep,
    )


@app.function(image=comparison_image, gpu="B200", timeout=1200)
def blackwell_sonic():
    return validate(sonic=True)


@app.function(image=comparison_image, gpu="B200:2", timeout=1200)
def blackwell_sonic_ep():
    return validate(distributed=True, sonic=True)


@app.function(image=comparison_image, gpu="B200:2", timeout=3600)
def compare_sonic_ep(compare_fused=False, profile=False):
    result = subprocess.run(  # nosec B603 B607
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "/probe/benchmark_pipeline.py",
            *(["--compare-fused"] if compare_fused else []),
            *(["--profile"] if profile else []),
        ],
        env={**os.environ, "PYTHONPATH": "/probe/src"},
        capture_output=True,
        text=True,
        timeout=3300,
        check=False,
    )
    print(result.stdout)
    print(result.stderr)
    result.check_returncode()
    return result.stdout


@app.function(image=comparison_image, gpu="B200", timeout=3600)
def compare_blackwell():
    results = []
    for backend, flags in (
        ("torch", []),
        ("torch", ["--compile"]),
        ("sonic", []),
        ("sonic", ["--compile"]),
        ("deepgemm", ["--unfused-backward"]),
        ("deepgemm", []),
    ):
        result = subprocess.run(  # nosec B603 B607
            ["python", "/probe/benchmark.py", "--backend", backend, *flags],
            env={**os.environ, "PYTHONPATH": "/probe/src"},
            capture_output=True,
            text=True,
            timeout=1100,
            check=False,
        )
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()
        results.append(result.stdout)
    return results


@app.local_entrypoint()
def main(
    gpu: str = "B200",
    ep: bool = False,
    compare: bool = False,
    quantization: bool = False,
    sonic: bool = False,
    fp8_gemm: bool = False,
    profile: bool = False,
    fused_fp8: bool = False,
    changing_routing: bool = False,
    framework_ep: bool = False,
    fft: bool = False,
    resume: bool = False,
):
    if fft:
        if gpu not in ("H100", "B200"):
            raise ValueError("FP8 FFT validation requires H100/B200")
        print((fp8_fft_hopper if gpu == "H100" else fp8_fft_blackwell).remote(resume))
        return
    if framework_ep:
        if gpu not in ("H100", "B200") or not ep:
            raise ValueError("framework adapter validation requires H100/B200 and --ep")
        print(
            (hopper_fp8_ep if gpu == "H100" else blackwell_fp8_ep).remote(False, True)
        )
        return
    if changing_routing:
        if gpu not in ("H100", "B200") or not ep:
            raise ValueError("changing-routing validation requires H100/B200 and --ep")
        print((hopper_fp8_ep if gpu == "H100" else blackwell_fp8_ep).remote(True))
        return
    if fused_fp8:
        if gpu != "B200" or not ep:
            raise ValueError("fused MXFP8 training validation requires --gpu B200 --ep")
        print(
            compare_sonic_ep.remote(True, profile)
            if compare
            else blackwell_ep.remote(True)
        )
        return
    if gpu == "H200":
        if not (ep and compare) or sonic or quantization or fp8_gemm:
            raise ValueError(
                "use --gpu H200 --ep --compare for the Qwen3 layer benchmark"
            )
        print(compare_h200.remote(profile))
        return
    if gpu not in ("H100", "B200"):
        raise ValueError("choose H100, H200, or B200")
    if fp8_gemm:
        if compare or quantization or sonic:
            raise ValueError("FP8 GEMM checks run independently")
        target = (
            (hopper_fp8_ep if gpu == "H100" else blackwell_fp8_ep)
            if ep
            else (hopper_fp8_gemm if gpu == "H100" else blackwell_fp8_gemm)
        )
        print(target.remote())
        return
    if ep and gpu != "B200":
        raise ValueError("the public fused forward requires B200")
    if sonic:
        if gpu != "B200" or quantization or (compare and not ep):
            raise ValueError("the SonicMoE checks run on B200")
        target = (
            compare_sonic_ep
            if compare
            else blackwell_sonic_ep
            if ep
            else blackwell_sonic
        )
        print(target.remote())
        return
    if quantization:
        if gpu != "B200" or ep or compare:
            raise ValueError("the format compilation check runs alone on B200")
        print(blackwell_quantization.remote())
        return
    if compare:
        if gpu != "B200" or ep:
            raise ValueError("the initial comparison is EP=1 on B200")
        print(compare_blackwell.remote())
        return
    print((blackwell_ep if ep else hopper if gpu == "H100" else blackwell).remote())
