# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""BF16 grouped consumers acquire published expert payloads on the GPU."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]


@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("counts", [(128, 256), (0, 128), (0, 0)])
@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("compiled", [False, True])
def test_ready_grouped_bf16(counts, width, compiled, backward, incremental):
    if not torch.cuda.is_available():
        pytest.skip("Readiness-aware BF16 grouped GEMM requires CUDA")
    arch = torch.cuda.get_device_capability()[0]
    if arch not in (9, 10):
        pytest.skip("Readiness-aware BF16 grouped GEMM requires SM90 or SM100")
    if arch == 9 and not incremental:
        pytest.skip("Hopper readiness requires incremental scheduling")
    deep_gemm = pytest.importorskip("deep_gemm")
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from mono_ep.chunk import (
        ready_grouped_bf16_out,
        ready_grouped_bf16_swiglu_backward_out,
    )
    from mono_ep.ops import weight_grad_bf16

    @triton.jit
    def publish(queue):
        task = tl.program_id(0)
        tl.atomic_xchg(queue + task * 4 + 3, 1, sem="release", scope="sys")

    torch.compiler.reset()
    rows, hidden = 512, 256
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(2, width, hidden, device="cuda", dtype=torch.bfloat16)
    ids_source = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    expected = torch.zeros(rows, width, device="cuda", dtype=torch.bfloat16)
    descriptors = []
    start = 0
    for expert, count in enumerate(counts):
        ids_source[start : start + count] = expert
        expected[start : start + count] = (
            source[start : start + count] @ weights[expert].T
        )
        descriptors.append([expert, start, count, 0])
        start += count
    descriptor_source = torch.tensor(descriptors, device="cuda", dtype=torch.int32)
    queue = descriptor_source.clone()
    queue[:, 3] = 1
    payload, ids = source.clone(), ids_source.clone()
    out = torch.zeros_like(expected)
    z_source = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16)
    scores_source = torch.rand(rows, device="cuda")
    z_source[start:] = 0
    scores_source[start:] = 0
    gate, up = z_source.float().chunk(2, dim=1)
    sigmoid = torch.sigmoid(gate)
    silu = gate * sigmoid
    score = scores_source[:, None]
    dh = expected.float()
    expected_backward = (
        (silu * up * score).bfloat16(),
        torch.cat(
            (dh * score * up * (sigmoid + silu * (1 - sigmoid)), dh * score * silu),
            dim=1,
        ).bfloat16(),
        (dh * silu * up).sum(-1),
    )
    z_source[start:] = float("nan")
    scores_source[start:] = float("nan")
    z, scores = z_source.clone(), scores_source.clone()
    dz = torch.zeros_like(z)
    partial = torch.zeros(rows, width // 8, device="cuda")
    kernel = (
        ready_grouped_bf16_swiglu_backward_out if backward else ready_grouped_bf16_out
    )
    arguments = (payload, weights, out, queue, ids)
    if backward:
        arguments += (z, scores, dz, partial)
    arguments += (incremental,)
    operation = torch.compile(kernel, fullgraph=True) if compiled else kernel
    old_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(32)
    try:
        publish[(2,)](queue)
        torch.cuda._sleep(1)
        operation(*arguments)
        torch.cuda.synchronize()
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        for _ in range(3):
            payload.fill_(float("nan"))
            ids.fill_(-1)
            queue.zero_()
            out.fill_(float("nan") if backward else 0)
            dz.fill_(float("nan"))
            partial.zero_()
            z.fill_(float("nan"))
            scores.fill_(float("nan"))
            current = torch.cuda.current_stream()
            producer.wait_stream(current)
            consumer.wait_stream(current)
            with torch.cuda.stream(consumer):
                operation(*arguments)
            with torch.cuda.stream(producer):
                torch.cuda._sleep(10_000_000)
                payload.copy_(source)
                ids.copy_(ids_source)
                z.copy_(z_source)
                scores.copy_(scores_source)
                queue.copy_(descriptor_source)
                publish[(2,)](queue)
            current.wait_stream(producer)
            current.wait_stream(consumer)
            torch.cuda.synchronize()
            if backward:
                for actual, reference in zip(
                    (out, dz, partial.sum(-1)), expected_backward, strict=True
                ):
                    assert torch.isfinite(actual[:start]).all()
                    if actual.ndim == 2:
                        assert torch.isnan(actual[start:]).all()
                    else:
                        assert not actual[start:].count_nonzero().item()
                    error = (
                        actual[:start].float() - reference[:start].float()
                    ).norm() / reference.float().norm().clamp_min(1e-8)
                    assert error.item() < 1e-3
                if arch == 10:
                    ends = torch.tensor(
                        [counts[0], sum(counts)], device="cuda", dtype=torch.int32
                    )
                    for operand, reference in zip(
                        (out, dz), expected_backward[:2], strict=True
                    ):
                        gradients = weight_grad_bf16(
                            operand, source, ends, None, 2, True
                        )
                        assert torch.isfinite(gradients).all()
                        offset = 0
                        for expert, count in enumerate(counts):
                            block = slice(offset, offset + count)
                            expected_grad = (
                                reference[block].float().T @ source[block].float()
                            )
                            error = (
                                gradients[expert] - expected_grad
                            ).norm() / expected_grad.norm().clamp_min(1e-8)
                            assert error < 2e-3, error
                            offset += count
            else:
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
    finally:
        deep_gemm.set_num_sms(old_sms)


@pytest.mark.parametrize("counts", [(137, 65), (0, 65), (0, 0)])
@pytest.mark.parametrize("compiled", [False, True])
def test_packed_bf16_derivatives(counts, compiled):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Packed BF16 readiness requires SM100")
    pytest.importorskip("deep_gemm")
    from mono_ep.bf16 import bf16_packed_backward, bf16_packed_forward

    torch.compiler.reset()
    torch.manual_seed(341)
    rows, hidden, inner = 512, 256, 128
    x = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
    dy = torch.zeros_like(x)
    scores = torch.zeros(rows, device="cuda")
    ids = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    w1 = (torch.randn(2, 2 * inner, hidden, device="cuda") / hidden**0.5).bfloat16()
    w2 = (torch.randn(2, hidden, inner, device="cuda") / inner**0.5).bfloat16()
    expected_y, expected_dx = torch.zeros_like(x), torch.zeros_like(x)
    expected_ds = torch.zeros_like(scores)
    expected_dw1, expected_dw2 = (
        torch.zeros_like(w1, dtype=torch.float32),
        torch.zeros_like(w2, dtype=torch.float32),
    )
    start, descriptors, ends = 0, [], []
    for expert, count in enumerate(counts):
        padded = (count + 127) // 128 * 128
        block = slice(start, start + count)
        x[block] = torch.randn(count, hidden, device="cuda", dtype=torch.bfloat16)
        dy[block] = torch.randn_like(x[block])
        scores[block] = torch.rand(count, device="cuda")
        ids[start : start + padded] = expert
        gate, up = (x[block] @ w1[expert].T).float().chunk(2, dim=1)
        sigmoid = torch.sigmoid(gate)
        silu = gate * sigmoid
        score = scores[block, None]
        h = (silu * up * score).bfloat16()
        dh = (dy[block] @ w2[expert]).float()
        dz = torch.cat(
            (dh * score * up * (sigmoid + silu * (1 - sigmoid)), dh * score * silu),
            dim=1,
        ).bfloat16()
        expected_y[block] = h @ w2[expert].T
        expected_dx[block] = dz @ w1[expert]
        expected_ds[block] = (dh * silu * up).sum(-1)
        expected_dw1[expert] = dz.float().T @ x[block].float()
        expected_dw2[expert] = dy[block].float().T @ h.float()
        descriptors.append([expert, start, padded, 1])
        ends.append(start + count)
        start += padded
    ready = torch.tensor(descriptors, device="cuda", dtype=torch.int32)
    ends = torch.tensor(ends, device="cuda", dtype=torch.int32)
    forward = (
        torch.compile(bf16_packed_forward, fullgraph=True)
        if compiled
        else bf16_packed_forward
    )
    backward = (
        torch.compile(bf16_packed_backward, fullgraph=True)
        if compiled
        else bf16_packed_backward
    )
    y, z = forward(x, w1, w2, ready, scores, ids)
    z[ids < 0] = float("nan")
    gradients = backward(dy, x, w1, w2, ready, scores, z, ids, ends)
    for name, actual, expected in zip(
        ("output", "input_gradient", "score_gradient", "w1_gradient", "w2_gradient"),
        (y, *gradients),
        (expected_y, expected_dx, expected_ds, expected_dw1, expected_dw2),
        strict=True,
    ):
        assert torch.isfinite(actual).all(), name
        error = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm().clamp_min(1e-8)
        assert error.item() < 2e-3, (name, error.item())


@pytest.mark.parametrize("counts", [(128, 256), (0, 128), (0, 0)])
@pytest.mark.parametrize("inner", [128, 768])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
def test_ready_bf16_fused_forward(counts, inner, compiled, incremental):
    if not torch.cuda.is_available():
        pytest.skip("Fused BF16 forward requires CUDA")
    arch = torch.cuda.get_device_capability()[0]
    if arch not in (9, 10):
        pytest.skip("Fused BF16 forward requires SM90 or SM100")
    if arch == 9 and not incremental:
        pytest.skip("Hopper readiness requires incremental scheduling")
    deep_gemm = pytest.importorskip("deep_gemm")
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from mono_ep.chunk import ready_grouped_bf16_swiglu_forward_out
    from mono_ep.combine import grouped_bf16_scatter

    @triton.jit
    def publish(queue):
        task = tl.program_id(0)
        tl.atomic_xchg(queue + task * 4 + 3, 1, sem="release", scope="sys")

    torch.manual_seed(117)
    rows, hidden = 512, 256
    x_source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2, 2 * inner, hidden, device="cuda", dtype=torch.bfloat16) / 16
    interleaved = w.unflatten(1, (2, inner)).transpose(1, 2).flatten(1, 2).contiguous()
    score_source = torch.rand(rows, device="cuda")
    ids_source = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    expected_z = torch.zeros(rows, 2 * inner, device="cuda", dtype=torch.bfloat16)
    descriptors, start = [], 0
    for expert, count in enumerate(counts):
        ids_source[start : start + count] = expert
        expected_z[start : start + count] = (
            x_source[start : start + count] @ w[expert].T
        )
        descriptors.append([expert, start, count, 0])
        start += count
    score_source[start:] = float("nan")
    gate, up = expected_z.float().chunk(2, -1)
    expected_h = (
        torch.nn.functional.silu(gate) * up * score_source[:, None]
    ).bfloat16()
    expected_h[start:] = 0
    descriptor_source = torch.tensor(descriptors, device="cuda", dtype=torch.int32)
    queue = descriptor_source.clone()
    queue[:, 3] = 1
    x, ids, scores = x_source.clone(), ids_source.clone(), score_source.clone()
    z = torch.zeros_like(expected_z)
    h = torch.full_like(expected_h, float("nan"))
    w2 = (torch.randn(2, hidden, inner, device="cuda") / inner**0.5).bfloat16()
    token_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    token_ids[start:] = -1
    destination = torch.zeros(rows, hidden, device="cuda")
    completion = torch.zeros(1, 4, device="cuda", dtype=torch.int32)
    expected_y = torch.zeros_like(destination)
    offset = 0
    for expert, count in enumerate(counts):
        expected_y[offset : offset + count] = (
            expected_h[offset : offset + count] @ w2[expert].T
        ).float()
        offset += count
    scatter = grouped_bf16_scatter
    if compiled:
        scatter = torch.compile(scatter, fullgraph=True)
    op = ready_grouped_bf16_swiglu_forward_out
    if compiled:
        op = torch.compile(op, fullgraph=True)
    old_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(32)
    try:
        publish[(2,)](queue)
        torch.cuda._sleep(1)
        op(x, interleaved, z, h, scores, queue, ids, incremental)
        torch.cuda.synchronize()
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        for _ in range(3):
            x.fill_(float("nan"))
            scores.fill_(float("nan"))
            ids.fill_(-1)
            queue.zero_()
            z.zero_()
            h.fill_(float("nan"))
            current = torch.cuda.current_stream()
            producer.wait_stream(current)
            consumer.wait_stream(current)
            with torch.cuda.stream(consumer):
                op(x, interleaved, z, h, scores, queue, ids, incremental)
            with torch.cuda.stream(producer):
                torch.cuda._sleep(10_000_000)
                x.copy_(x_source)
                ids.copy_(ids_source)
                scores.copy_(score_source)
                queue.copy_(descriptor_source)
                publish[(2,)](queue)
            current.wait_stream(producer)
            current.wait_stream(consumer)
            torch.cuda.synchronize()
            torch.testing.assert_close(z, expected_z, atol=0, rtol=0)
            assert torch.isfinite(h[:start]).all()
            assert torch.isnan(h[start:]).all()
            error = (
                h[:start].float() - expected_h[:start].float()
            ).norm() / expected_h.float().norm().clamp_min(1e-8)
            assert error < 1e-3, error
            if arch == 10:
                destination.zero_()
                completion.zero_()
                scatter(
                    h,
                    w2,
                    ids,
                    token_ids,
                    destination,
                    completion,
                    0,
                    True,
                    (queue[:, 1] + queue[:, 2]).max(),
                )
                torch.cuda.synchronize()
                assert torch.isfinite(destination).all()
                output_error = (
                    destination - expected_y
                ).norm() / expected_y.norm().clamp_min(1e-8)
                assert output_error < 2e-3, output_error
    finally:
        deep_gemm.set_num_sms(old_sms)


@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_ready_bf16_staggered_progress(incremental, compiled):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Incremental readiness requires SM100")
    deep_gemm = pytest.importorskip("deep_gemm")
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from mono_ep.chunk import ready_grouped_bf16_out

    @triton.jit
    def observe_then_publish(out, payload, queue, observed, WIDTH: tl.constexpr):
        value = tl.load(out, volatile=True).to(tl.float32)
        spins = 0
        while (value != WIDTH) & (spins < 100000):
            tl.inline_asm_elementwise(
                "nanosleep.u32 256; mov.u32 $0, $1;",
                constraints="=r,r",
                args=[spins],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            value = tl.load(out, volatile=True).to(tl.float32)
            spins += 1
        tl.store(observed, value == WIDTH)
        offsets = tl.arange(0, 256)
        for start in range(128 * WIDTH, 256 * WIDTH, 256):
            tl.store(payload + start + offsets, 1)
        tl.inline_asm_elementwise(
            "fence.acq_rel.sys; mov.u32 $0, $1;",
            constraints="=r,r",
            args=[offsets],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.debug_barrier()
        tl.atomic_xchg(queue + 7, 1, sem="release", scope="sys")

    torch.compiler.reset()
    width = 256
    x = torch.ones(256, width, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(2, 256, width, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(256, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(128)
    queue = torch.tensor(
        [[0, 0, 128, 1], [1, 128, 128, 1]], device="cuda", dtype=torch.int32
    )
    observed = torch.zeros((), device="cuda", dtype=torch.int32)
    operation = (
        torch.compile(ready_grouped_bf16_out, fullgraph=True)
        if compiled
        else ready_grouped_bf16_out
    )
    previous_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(32)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    try:
        operation(x, w, out, queue, ids, incremental)
        observe_then_publish[(1,)](out, x, queue, observed, width)
        torch.cuda.synchronize()
        for _ in range(3):
            x[128:].fill_(float("nan"))
            out.fill_(float("nan"))
            queue[1, 3] = 0
            observed.zero_()
            current = torch.cuda.current_stream()
            producer.wait_stream(current)
            consumer.wait_stream(current)
            with torch.cuda.stream(consumer):
                operation(x, w, out, queue, ids, incremental)
            with torch.cuda.stream(producer):
                observe_then_publish[(1,)](out, x, queue, observed, width)
            current.wait_stream(producer)
            current.wait_stream(consumer)
            torch.cuda.synchronize()
            assert observed.item() == int(incremental)
            torch.testing.assert_close(out, torch.full_like(out, width), atol=0, rtol=0)
    finally:
        deep_gemm.set_num_sms(previous_sms)
