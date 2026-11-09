# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run the standalone Hopper backward GEMM roofline probe on Modal."""

import json
from pathlib import Path

import modal

app = modal.App("mono-ep-hopper-backward-gemm-roofline")
image = None
if modal.is_local():
    from modal_validate import base_image

    root = Path(__file__).resolve().parents[1]
    image = (
        base_image.add_local_dir(root / "src", "/probe/src")
        .add_local_file(
            root / "examples" / "benchmark_hopper_backward_gemm_roofline.py",
            "/probe/benchmark_hopper_backward_gemm_roofline.py",
        )
        .add_local_file(
            root / "examples" / "benchmark_hopper_backward_cache_roofline.py",
            "/probe/benchmark_hopper_backward_cache_roofline.py",
        )
    )


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=1200)
def run(master_dtype: str, actual_cache: bool):
    import sys

    sys.path.insert(0, "/probe")
    sys.path.insert(0, "/probe/src")
    if actual_cache:
        from benchmark_hopper_backward_cache_roofline import benchmark
    else:
        from benchmark_hopper_backward_gemm_roofline import benchmark

    return benchmark(master_dtype=master_dtype)


@app.local_entrypoint()
def main(actual_cache: bool = False):
    results = {
        dtype: run.remote(dtype, actual_cache) for dtype in ("bfloat16", "float32")
    }
    print(json.dumps(results, indent=2))
