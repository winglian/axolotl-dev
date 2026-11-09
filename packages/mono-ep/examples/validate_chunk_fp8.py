# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Exercise readiness of both FP8 values and packed scales on SM100."""

import argparse
import json

import deep_gemm
import torch
from mono_ep.chunk import chunk_fp8_out, chunk_fp8_swiglu_backward_out
from mono_ep.quantization import quantize_fp8
from validate_chunk_gemm import publish


def forward_operation(*args):
    chunk_fp8_out(*args)


def backward_operation(*args):
    chunk_fp8_swiglu_backward_out(*args)


def backward_check(x, sx, w, sw, queue, expected_dh, gran, compiled):
    operation = backward_operation
    if compiled:
        operation = torch.compile(operation, fullgraph=True, dynamic=False)
    rows, inner = x.shape[0], w.shape[1]
    z = torch.randn(rows, inner * 2, device="cuda", dtype=torch.bfloat16)
    scores = torch.rand(rows, device="cuda")
    h = torch.full((rows, inner), 7, device="cuda", dtype=torch.bfloat16)
    dz = torch.full_like(z, 7)
    partial = torch.full((rows, inner // 8), 7, device="cuda", dtype=torch.float32)
    zr = z[128:384].float().requires_grad_()
    sr = scores[128:384].detach().clone().requires_grad_()
    gate, up = zr.chunk(2, dim=-1)
    expected_h = torch.nn.functional.silu(gate) * up * sr[:, None]
    expected_dz, expected_ds = torch.autograd.grad(
        expected_h, (zr, sr), expected_dh.bfloat16().float()
    )

    def check():
        errors = []
        for actual, expected in (
            (h[128:384], expected_h),
            (dz[128:384], expected_dz),
            (partial[128:384].sum(-1), expected_ds),
        ):
            error = (actual.float() - expected.float()).norm() / expected.float().norm()
            assert torch.isfinite(error) and error < 0.01, error
            errors.append(error.item())
        for output in (h, dz, partial):
            assert (output[:128] == 7).all() and (output[384:] == 7).all()
        return errors

    operation(x, sx, w, sw, h, queue, 0, z, scores, dz, partial, gran)
    ready = check()
    previous = [t.clone() for t in (h, dz, partial)]
    operation(x, sx, w, sw, h, queue, 1, z, scores, dz, partial, gran)
    for actual, expected in zip((h, dz, partial), previous, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (x, sx, w, sw, z, scores)
    sources = [t.clone(memory_format=torch.preserve_format) for t in inputs]
    torch.cuda._sleep(1)
    for destination, source in zip(inputs, sources, strict=True):
        destination.copy_(source)
    publish[(1,)](queue, 0)
    torch.cuda.synchronize()
    for t in inputs:
        t.zero_()
    for t in (h, dz, partial):
        t.fill_(7)
    queue[0, 3] = 0
    torch.cuda.synchronize()
    consumer, producer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        operation(x, sx, w, sw, h, queue, 0, z, scores, dz, partial, gran)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(10_000_000)
        for destination, source in zip(inputs, sources, strict=True):
            destination.copy_(source)
        publish[(1,)](queue, 0)
    torch.cuda.synchronize()
    return {"ready": ready, "delayed": check()}


def run(compiled):
    deep_gemm.set_num_sms(32)
    operation = forward_operation
    if compiled:
        operation = torch.compile(operation, fullgraph=True, dynamic=False)
    for gran in (32, 128):
        for n in (128, 256):
            x, xs = quantize_fp8(
                torch.randn(512, 512, device="cuda", dtype=torch.bfloat16),
                block_cols=gran,
            )
            w, ws = quantize_fp8(
                torch.randn(3, n, 512, device="cuda", dtype=torch.bfloat16),
                block_cols=gran,
            )
            sx = deep_gemm.transform_sf_into_required_layout(xs, 512, 512, (1, gran))
            sw = deep_gemm.transform_sf_into_required_layout(
                ws, n, 512, (1, gran), num_groups=3
            )
            xd = x.float() * xs.repeat_interleave(gran, -1)
            wd = w.float() * ws.repeat_interleave(gran, -1)
            expected = xd[128:384] @ wd[2].T
            out = torch.full((512, n), 7, device="cuda", dtype=torch.bfloat16)
            queue = torch.tensor(
                [[2, 128, 256, 1], [1, 0, 0, 1]], device="cuda", dtype=torch.int32
            )

            def check(out=out, expected=expected):
                error = (out[128:384].float() - expected).norm() / expected.norm()
                assert torch.isfinite(error) and error < 0.01, error
                assert (out[:128] == 7).all() and (out[384:] == 7).all()
                return error.item()

            regression = torch.empty_like(out)
            deep_gemm.fp8_gemm_nt(
                (x, sx), (w[2], sw[2]), regression, recipe=(1, 1, gran)
            )
            normal_error = (regression.float() - xd @ wd[2].T).norm() / (
                xd @ wd[2].T
            ).norm()
            assert torch.isfinite(normal_error) and normal_error < 0.01
            layout = torch.tensor(
                [0] * 128 + [1] * 128 + [2] * 256, device="cuda", dtype=torch.int32
            )
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (x, sx), (w, sw), regression, layout, recipe=(1, 1, gran)
            )
            grouped_reference = torch.cat(
                (xd[:128] @ wd[0].T, xd[128:256] @ wd[1].T, xd[256:] @ wd[2].T)
            )
            grouped_error = (
                regression.float() - grouped_reference
            ).norm() / grouped_reference.norm()
            assert torch.isfinite(grouped_error) and grouped_error < 0.01
            operation(x, sx, w, sw, out, queue, 0, gran)
            ready = check()
            previous = out.clone()
            operation(x, sx, w, sw, out, queue, 1, gran)
            torch.testing.assert_close(out, previous, atol=0, rtol=0)
            sources = [
                t.clone(memory_format=torch.preserve_format) for t in (x, sx, w, sw)
            ]
            torch.cuda._sleep(1)
            for destination, source in zip((x, sx, w, sw), sources, strict=True):
                destination.copy_(source)
            publish[(1,)](queue, 0)
            torch.cuda.synchronize()
            for destination in (x, sx, w, sw):
                destination.zero_()
            queue[0, 3] = 0
            out.fill_(7)
            torch.cuda.synchronize()
            consumer, producer = torch.cuda.Stream(), torch.cuda.Stream()
            with torch.cuda.stream(consumer):
                operation(x, sx, w, sw, out, queue, 0, gran)
            with torch.cuda.stream(producer):
                torch.cuda._sleep(10_000_000)
                for destination, source in zip((x, sx, w, sw), sources, strict=True):
                    destination.copy_(source)
                publish[(1,)](queue, 0)
            torch.cuda.synchronize()
            delayed = check()
            backward = backward_check(x, sx, w, sw, queue, expected, gran, compiled)
            print(
                json.dumps(
                    dict(
                        gran_k=gran,
                        n=n,
                        compiled=compiled,
                        backward_relative_l2=backward,
                        normal_relative_l2=normal_error.item(),
                        grouped_relative_l2=grouped_error.item(),
                        ready_relative_l2=ready,
                        delayed_relative_l2=delayed,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    run(parser.parse_args().compiled)
