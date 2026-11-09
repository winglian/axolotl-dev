# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Isolate readiness-aware SwiGLU fusion with prepublished local inputs."""

import json
from pathlib import Path

import modal

app = modal.App("axolotl-swiglu-isolation")
image = None
if modal.is_local():
    from modal_validate import base_image

    image = base_image.add_local_dir(
        Path(__file__).resolve().parents[1] / "src", "/probe/src"
    )


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=1200)
def benchmark():
    import shutil
    import statistics
    import sys

    sys.path.insert(0, "/probe/src")
    import deep_gemm
    import torch
    import triton
    from mono_ep.activation import _activation
    from mono_ep.quantization import quantize_fp8

    torch.manual_seed(863)
    active, hidden, inner, experts = 16384, 2048, 768, 16
    weight = (
        torch.randn(experts, inner, hidden, device="cuda", dtype=torch.bfloat16)
        / hidden**0.5
    )
    qw, sw = quantize_fp8(weight)
    sw = deep_gemm.transform_sf_into_required_layout(
        sw, inner, hidden, (1, 32), num_groups=experts
    )
    records = []
    previous_sms = deep_gemm.get_num_sms()
    try:
        for rows in (active, 133120):
            x = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
            x[:active].normal_()
            qx, sx = quantize_fp8(x)
            sx = deep_gemm.transform_sf_into_required_layout(sx, rows, hidden, (1, 32))
            ids = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
            ids[:active] = torch.arange(
                experts, device="cuda", dtype=torch.int32
            ).repeat_interleave(active // experts)
            ready = torch.tensor(
                [
                    [i, i * (active // experts), active // experts, 1]
                    for i in range(experts)
                ],
                device="cuda",
                dtype=torch.int32,
            )
            z = torch.zeros(rows, 2 * inner, device="cuda", dtype=torch.bfloat16)
            z[:active].normal_()
            scores = torch.zeros(rows, device="cuda")
            scores[:active].uniform_()
            dh = torch.zeros(rows, inner, device="cuda", dtype=torch.bfloat16)
            buffers = {
                name: (
                    torch.zeros_like(dh),
                    torch.zeros_like(z),
                    torch.zeros_like(scores),
                )
                for name in ("fused", "separate")
            }
            partial = torch.zeros(rows, inner // 8, device="cuda")
            epilogue = deep_gemm.epilogue.SwiGLUBackward(
                z, scores, buffers["fused"][1], partial
            )

            def run(
                name,
                buffers=buffers,
                epilogue=epilogue,
                qx=qx,
                sx=sx,
                ids=ids,
                ready=ready,
                z=z,
                scores=scores,
                dh=dh,
                rows=rows,
                partial=partial,
            ):
                h, dz, ds = buffers[name]
                if name == "fused":
                    deep_gemm.fp8_ready_grouped_gemm_nt(
                        (qx, sx), (qw, sw), h, ids, ready, epilogue=epilogue
                    )
                    torch.sum(partial, dim=-1, out=ds)
                else:
                    deep_gemm.fp8_ready_grouped_gemm_nt(
                        (qx, sx), (qw, sw), dh, ids, ready
                    )
                    _activation[(rows,)](
                        z,
                        scores,
                        h,
                        dh,
                        dz,
                        ds,
                        inner,
                        True,
                        True,
                        triton.next_power_of_2(inner),
                        enable_fp_fusion=False,
                    )

            for sms in (16, 132):
                deep_gemm.set_num_sms(sms)
                run("fused")
                run("separate")
                torch.cuda.synchronize()
                errors = []
                for got, want in zip(
                    buffers["fused"], buffers["separate"], strict=True
                ):
                    error = (
                        (got.float() - want.float()).norm()
                        / want.float().norm().clamp_min(1e-8)
                    ).item()
                    assert torch.isfinite(got).all() and error < 0.01, error
                    assert not got[active:].count_nonzero().item()
                    errors.append(error)
                samples = {name: [] for name in buffers}
                for trial in range(25):
                    for name in (
                        ("fused", "separate") if trial % 2 else ("separate", "fused")
                    ):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        run(name)
                        end.record()
                        end.synchronize()
                        if trial >= 5:
                            samples[name].append(start.elapsed_time(end))
                records.append(
                    dict(
                        rows=rows,
                        active_rows=active,
                        sms=sms,
                        gpu=torch.cuda.get_device_name(),
                        torch_version=torch.__version__,
                        relative_l2_h_dz_ds=errors,
                        samples_ms=samples,
                        median_ms={
                            name: statistics.median(times)
                            for name, times in samples.items()
                        },
                        scope="Local inputs and descriptors prepublished; no peers, other streams or producer waits. Fused includes score reduction; separate includes activation. Inputs/outputs preallocated.",
                    )
                )
    finally:
        deep_gemm.set_num_sms(previous_sms)
    return dict(records=records, ncu_path=shutil.which("ncu"))


@app.local_entrypoint()
def main():
    print(json.dumps(benchmark.remote()))
