# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compare activation tiling on one B200 using fixed output allocations."""

import json
from pathlib import Path

import modal

app = modal.App("axolotl-activation-tiles")
image = None
if modal.is_local():
    from modal_validate import base_image

    image = base_image.add_local_dir(
        Path(__file__).resolve().parents[1] / "src", "/probe/src"
    )


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=600)
def benchmark():
    import statistics
    import sys

    sys.path.insert(0, "/probe/src")
    import torch
    import triton
    from mono_ep.activation import _activation_fp8

    records = []
    for rows, inner in ((137, 257), (133120, 768)):
        torch.manual_seed(841)
        z = torch.randn(rows, 2 * inner, device="cuda", dtype=torch.bfloat16)
        scores = torch.rand(rows, device="cuda")
        groups = triton.cdiv(inner, 32)
        outputs = {
            tile: (
                torch.empty(rows, inner, device="cuda", dtype=torch.float8_e4m3fn),
                torch.empty(rows, groups, device="cuda"),
            )
            for tile in (1, 4)
        }

        def run(
            tile,
            outputs=outputs,
            rows=rows,
            inner=inner,
            groups=groups,
            z=z,
            scores=scores,
        ):
            q, scales = outputs[tile]
            _activation_fp8[(triton.cdiv(rows, tile), triton.cdiv(groups, tile))](
                z,
                scores,
                q,
                scales,
                inner,
                groups,
                rows,
                tile,
                tile,
                num_warps=4,
                enable_fp_fusion=False,
            )

        for tile in outputs:
            run(tile)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            outputs[1][0].float(), outputs[4][0].float(), atol=0, rtol=0
        )
        torch.testing.assert_close(outputs[1][1], outputs[4][1], atol=0, rtol=0)
        samples = {tile: [] for tile in outputs}
        for trial in range(25):
            for tile in (1, 4) if trial % 2 else (4, 1):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run(tile)
                end.record()
                end.synchronize()
                if trial >= 5:
                    samples[tile].append(start.elapsed_time(end))
        records.append(
            dict(
                rows=rows,
                inner=inner,
                gpu=torch.cuda.get_device_name(),
                torch_version=torch.__version__,
                exact_parity=True,
                median_ms={tile: statistics.median(v) for tile, v in samples.items()},
                samples_ms=samples,
            )
        )
    return records


@app.local_entrypoint()
def main():
    print(json.dumps(benchmark.remote()))
