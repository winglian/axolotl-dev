# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Measure GPU-bounded column quantization at the Qwen peer-buffer capacity."""

import json
from pathlib import Path

import modal

app = modal.App("axolotl-column-bounds")
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
    from mono_ep.column_quantization import (
        quantize_columns,
        quantize_packed_columns,
    )

    rows, cols, end = 133120, 2048, 17408
    torch.manual_seed(859)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    x[end:] = 0
    row_end = torch.tensor(end, device="cuda", dtype=torch.int32)
    records = []
    for packed in (False, True):
        values = x.to(torch.float8_e4m3fn) if packed else x
        scales = (
            torch.full(
                (cols // 128, rows), 0x7F7F7F7F, device="cuda", dtype=torch.int32
            ).T
            if packed
            else None
        )

        def run(bounded, values=values, scales=scales, packed=packed):
            bound = row_end if bounded else None
            return (
                quantize_packed_columns(values, scales, bound)
                if packed
                else quantize_columns(values, bound)
            )

        expected, actual = run(False), run(True)
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got.float(), want.float(), atol=0, rtol=0)
        del expected, actual
        samples = {"full": [], "bounded": []}
        for trial in range(25):
            for bounded in (False, True) if trial % 2 else (True, False):
                start = torch.cuda.Event(enable_timing=True)
                finish = torch.cuda.Event(enable_timing=True)
                start.record()
                output = run(bounded)
                finish.record()
                finish.synchronize()
                if trial >= 5:
                    samples["bounded" if bounded else "full"].append(
                        start.elapsed_time(finish)
                    )
                del output
        records.append(
            dict(
                packed_input=packed,
                rows=rows,
                cols=cols,
                row_end=end,
                gpu=torch.cuda.get_device_name(),
                torch_version=torch.__version__,
                exact_full_output_parity=True,
                samples_ms=samples,
                median_ms={
                    name: statistics.median(times) for name, times in samples.items()
                },
                scope="Isolated column quantization; full output capacity is allocated and initialized in both variants.",
            )
        )
    return records


@app.local_entrypoint()
def main():
    print(json.dumps(benchmark.remote()))
