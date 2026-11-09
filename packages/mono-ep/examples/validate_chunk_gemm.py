# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Check readiness, row ownership, and BF16 GEMM regressions on SM100."""

import argparse
import json

import deep_gemm
import torch
import triton
import triton.language as tl


@triton.jit
def publish(queue, task: tl.constexpr):
    tl.atomic_xchg(queue + task * 4 + 3, 1, sem="release", scope="sys")


def backward_check(x, w, queue, compiled):
    from mono_ep.chunk import chunk_swiglu_backward_out

    operation = chunk_swiglu_backward_out
    if compiled:
        operation = torch.compile(operation, fullgraph=True, dynamic=False)
    rows, inner = x.shape[0], w.shape[1]
    z = torch.randn(rows, 2 * inner, device="cuda", dtype=torch.bfloat16)
    scores = torch.rand(rows, device="cuda")
    h = torch.full((rows, inner), 7, device="cuda", dtype=torch.bfloat16)
    dz = torch.full_like(z, 7)
    partial = torch.full((rows, inner // 8), 7, device="cuda", dtype=torch.float32)
    zr = z[128:384].float().requires_grad_()
    sr = scores[128:384].detach().clone().requires_grad_()
    gate, up = zr.chunk(2, dim=-1)
    expected_h = torch.nn.functional.silu(gate) * up * sr[:, None]
    dh = x[128:384] @ w[2].T
    expected_dz, expected_ds = torch.autograd.grad(expected_h, (zr, sr), dh.float())

    def check():
        errors = []
        for got, want in (
            (h[128:384], expected_h),
            (dz[128:384], expected_dz),
            (partial[128:384].sum(-1), expected_ds),
        ):
            error = (got.float() - want.float()).norm() / want.float().norm()
            assert torch.isfinite(error) and error < 0.03, error
            errors.append(error.item())
        for out in (h, dz, partial):
            assert (out[:128] == 7).all() and (out[384:] == 7).all()
        return errors

    operation(x, w, h, queue, 0, z, scores, dz, partial)
    ready_errors = check()
    previous = tuple(t.clone() for t in (h, dz, partial))
    operation(x, w, h, queue, 1, z, scores, dz, partial)
    for got, want in zip((h, dz, partial), previous, strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    source = x.clone()
    x.zero_()
    for out in (h, dz, partial):
        out.fill_(7)
    queue[0, 3] = 0
    torch.cuda.synchronize()
    consumer, producer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        operation(x, w, h, queue, 0, z, scores, dz, partial)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(10_000_000)
        x.copy_(source)
        publish[(1,)](queue, 0)
    torch.cuda.synchronize()
    return {"ready_relative_l2": ready_errors, "delayed_relative_l2": check()}


def run(compiled=False):
    torch.manual_seed(31)
    deep_gemm.set_num_sms(32)
    operation = deep_gemm.bf16_chunk_gemm_nt
    if compiled:
        from mono_ep.chunk import chunk_bf16_out

        operation = torch.compile(chunk_bf16_out, fullgraph=True, dynamic=False)
    records = []
    for n in (128, 256):
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(3, n, 256, device="cuda", dtype=torch.bfloat16) / 16
        d = torch.full((512, n), 7, device="cuda", dtype=torch.bfloat16)
        queue = torch.tensor(
            [[2, 128, 256, 1], [1, 0, 0, 1]], device="cuda", dtype=torch.int32
        )
        expected = x[128:384] @ w[2].T
        operation(x, w, d, queue, 0)
        torch.testing.assert_close(d[128:384], expected, atol=0.03, rtol=0.02)
        assert (d[:128] == 7).all() and (d[384:] == 7).all()
        previous = d.clone()
        operation(x, w, d, queue, 1)
        torch.testing.assert_close(d, previous, atol=0, rtol=0)

        # Warm both kernels before launching a consumer that waits on another stream.
        publish[(1,)](queue, 0)
        source = torch.randn_like(x)
        expected = source[128:384] @ w[2].T
        # Lazy CUDA module loading can synchronize the device and deadlock a waiter.
        torch.cuda._sleep(1)
        x.copy_(source)
        x.zero_()
        d.fill_(7)
        queue[0, 3] = 0
        torch.cuda.synchronize()
        consumer = torch.cuda.Stream()
        producer = torch.cuda.Stream()
        with torch.cuda.stream(consumer):
            operation(x, w, d, queue, 0)
        with torch.cuda.stream(producer):
            torch.cuda._sleep(10_000_000)
            x.copy_(source)
            publish[(1,)](queue, 0)
        torch.cuda.synchronize()
        torch.testing.assert_close(d[128:384], expected, atol=0.03, rtol=0.02)
        assert (d[:128] == 7).all() and (d[384:] == 7).all()

        regular = torch.empty_like(d)
        deep_gemm.bf16_gemm_nt(x, w[0], regular)
        torch.testing.assert_close(regular, x @ w[0].T, atol=0.03, rtol=0.02)
        layout = torch.full((512,), 1, dtype=torch.int32, device="cuda")
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(x, w, regular, layout)
        torch.testing.assert_close(regular, x @ w[1].T, atol=0.03, rtol=0.02)
        backward = backward_check(x, w, queue, compiled)
        records.append(
            {
                "n": n,
                "ready": True,
                "delayed": True,
                "empty": True,
                "backward": backward,
            }
        )
    print(json.dumps({"compiled": compiled, "chunk_gemm": records}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    run(parser.parse_args().compiled)
