# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Check all explicit queued MXFP8 expert derivatives against independent math."""

import argparse
import json

import deep_gemm
import torch
from mono_ep.quantization import quantize_fp8
from mono_ep.scheduled import fp8_expert_chunk_backward
from validate_chunk_gemm import publish
from validate_peer_layer import reference_quantize


def invoke(*args):
    return fp8_expert_chunk_backward(*args)


def pack(x):
    q, s = quantize_fp8(x)
    return q, deep_gemm.transform_sf_into_required_layout(
        s,
        x.shape[-2],
        x.shape[-1],
        (1, 32),
        num_groups=x.shape[0] if x.ndim == 3 else None,
    )


def run(compiled):
    torch.manual_seed(1131)
    torch.backends.cuda.matmul.allow_tf32 = False
    deep_gemm.set_num_sms(32)
    rows, hidden, inner = 256, 256, 128
    raw = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(raw)
    raw[137:] = 0
    upstream[137:] = 0
    scores = torch.rand(rows, device="cuda")
    scores[137:] = 0
    w1 = (
        torch.randn(1, 2 * inner, hidden, device="cuda", dtype=torch.bfloat16)
        / hidden**0.5
    )
    w2 = torch.randn(1, hidden, inner, device="cuda", dtype=torch.bfloat16) / inner**0.5
    x, sx = pack(raw)
    dy, sy = pack(upstream)
    w1t, s1t = quantize_fp8(w1.transpose(1, 2).contiguous())
    w2t, s2t = pack(w2.transpose(1, 2).contiguous())
    decoded_x = reference_quantize(raw).bfloat16()
    decoded_dy = reference_quantize(upstream).bfloat16()
    z = (decoded_x.float() @ reference_quantize(w1[0]).T).bfloat16()
    zr, sr = z.float().requires_grad_(), scores.detach().clone().requires_grad_()
    gate, up = zr.chunk(2, -1)
    h = torch.nn.functional.silu(gate) * up * sr[:, None]
    dh = (decoded_dy.float() @ reference_quantize(w2[0].T.contiguous()).T).bfloat16()
    dz, ds = torch.autograd.grad(h, (zr, sr), dh.float())
    dz = dz.bfloat16()
    dx = (
        reference_quantize(dz) @ reference_quantize(w1[0].T.contiguous()).T
    ).bfloat16()
    dw1 = (
        reference_quantize(dz.T.contiguous())
        @ reference_quantize(decoded_x.T.contiguous()).T
    )
    dw2 = (
        reference_quantize(decoded_dy.T.contiguous())
        @ reference_quantize(h.bfloat16().T.contiguous()).T
    )
    expected = (dx, ds, dw1[None], dw2[None])
    queue = torch.tensor([[0, 0, rows, 1]], device="cuda", dtype=torch.int32)
    operation = (
        torch.compile(invoke, fullgraph=True, dynamic=False) if compiled else invoke
    )
    args = (dy, sy, x, sx, w1t, s1t, w2t, s2t, queue, scores, z)

    def check(actual):
        errors = []
        for got, want in zip(actual, expected, strict=True):
            error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(
                1e-8
            )
            assert torch.isfinite(error) and error < 0.01, error
            errors.append(error.item())
        assert (actual[0][137:] == 0).all() and (actual[1][137:] == 0).all()
        assert actual[2].dtype == torch.float32 and actual[3].dtype == torch.float32
        return errors

    ready_errors = check(operation(*args))
    source, source_scale = dy.clone(), sy.clone(memory_format=torch.preserve_format)
    torch.cuda._sleep(1)
    dy.copy_(source)
    sy.copy_(source_scale)
    publish[(1,)](queue, 0)
    torch.cuda.synchronize()
    dy.zero_()
    sy.zero_()
    queue[0, 3] = 0
    torch.cuda.synchronize()
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        delayed = operation(*args)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(10_000_000)
        dy.copy_(source)
        sy.copy_(source_scale)
        publish[(1,)](queue, 0)
    torch.cuda.synchronize()
    delayed_errors = check(delayed)
    print(
        json.dumps(
            dict(
                compiled=compiled,
                order=["dx", "ds", "dw1", "dw2"],
                ready_relative_l2=ready_errors,
                delayed_relative_l2=delayed_errors,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    run(parser.parse_args().compiled)
