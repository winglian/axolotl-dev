# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Grouped GEMM peer scatter and publication against separate operations."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.gpu_blackwell,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize(
    "assign,output_dtype",
    [(False, torch.float32), (True, torch.float32), (True, torch.bfloat16)],
)
@pytest.mark.parametrize("hidden,inner", [(256, 128), (2048, 768)])
@pytest.mark.parametrize("case", ["mixed", "skewed", "masked"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_grouped_scatter(hidden, inner, case, compiled, assign, output_dtype, bounded):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 required")
    deep_gemm = pytest.importorskip("deep_gemm")
    if not hasattr(deep_gemm, "fp8_grouped_scatter_gemm_nt"):
        pytest.skip("MonoEP vendored DeepGEMM build required")
    from mono_ep.combine import (
        combine_wait_copy,
        combine_wait_reduce,
        grouped_fp8_scatter,
    )
    from mono_ep.ops import grouped_fp8
    from mono_ep.packing import DeviceExpertLayout
    from mono_ep.quantization import quantize_fp8

    torch.manual_seed(947)
    tokens, experts, topk = 137, 4, 3
    ids = torch.randint(experts, (tokens, topk), device="cuda", dtype=torch.int32)
    ids[::5] = -1
    if case == "skewed":
        ids[ids >= 0] = 0
    elif case == "masked":
        ids.fill_(-1)
    layout = DeviceExpertLayout.build(ids, experts, 128)
    source = torch.randn(tokens, inner, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
    x, sx = quantize_fp8(layout.pack(source))
    w, sw = quantize_fp8(weights)
    sx = deep_gemm.transform_sf_into_required_layout(sx, x.shape[0], inner, (1, 32))
    sw = deep_gemm.transform_sf_into_required_layout(
        sw, hidden, inner, (1, 32), num_groups=experts
    )
    row_end = layout.ends[-1:] if bounded else None
    token_ids = layout.row_assignments.int() if assign else layout.token_ids()
    values = grouped_fp8(x, sx, w, sw, layout.expert_ids, 32, 1)
    valid = token_ids >= 0
    expected = torch.zeros(tokens * (topk if assign else 1), hidden, device="cuda")
    expected.index_add_(0, token_ids[valid].long(), values[valid].float())
    if compiled:
        torch.compiler.reset()
    operation = (
        torch.compile(grouped_fp8_scatter, fullgraph=True)
        if compiled
        else grouped_fp8_scatter
    )
    destination = torch.zeros_like(expected, dtype=output_dtype)
    queue = torch.zeros(1, 4, device="cuda", dtype=torch.int32)
    result = torch.empty_like(expected)

    def consume():
        if output_dtype == torch.bfloat16:
            combine_wait_reduce(destination, result, queue, 1)
        else:
            combine_wait_copy(destination, result, queue)

    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    previous_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(32)
    try:
        scratch = torch.full_like(values, 7)
        deep_gemm.fp8_grouped_scatter_gemm_nt(
            (x, sx),
            (w, sw),
            scratch,
            layout.expert_ids,
            token_ids,
            destination,
            queue,
            0,
            assign=assign,
            row_end=row_end,
        )
        torch.cuda.synchronize()
        assert torch.all(scratch == 7)
        torch.testing.assert_close(destination.float(), expected, atol=5e-5, rtol=1e-5)
        destination.zero_()
        queue.zero_()
        operation(
            x,
            sx,
            w,
            sw,
            layout.expert_ids,
            token_ids,
            destination,
            queue,
            0,
            assign,
            row_end,
        )
        consume()
        torch.cuda._sleep(1)
        torch.cuda.synchronize()
        for _ in range(3):
            destination.zero_()
            queue.zero_()
            producer.wait_stream(torch.cuda.current_stream())
            consumer.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(consumer):
                consume()
            with torch.cuda.stream(producer):
                torch.cuda._sleep(1_000_000)
                operation(
                    x,
                    sx,
                    w,
                    sw,
                    layout.expert_ids,
                    token_ids,
                    destination,
                    queue,
                    0,
                    assign,
                    row_end,
                )
            producer.synchronize()
            consumer.synchronize()
            torch.testing.assert_close(result, expected, atol=5e-5, rtol=1e-5)
            assert queue[0, 3].item() == 1
            assert queue[0, 2].item() > 0
    finally:
        deep_gemm.set_num_sms(previous_sms)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rows", [0, 137])
def test_separate_scatter_store(dtype, compiled, rows):
    from mono_ep.combine import combine_wait_reduce, peer_scatter_store_publish

    source = torch.randn(rows, 257, device="cuda", dtype=torch.bfloat16)
    indices = torch.randperm(rows, device="cuda").int()
    indices[::5] = -1
    output = torch.zeros(max(1, rows), 257, device="cuda", dtype=dtype)
    queue = torch.zeros(1, 4, device="cuda", dtype=torch.int32)
    reference = torch.zeros_like(output)
    valid = indices >= 0
    reference[indices[valid].long()] = source[valid].to(dtype)

    def operation(source, indices, destination, queue, task):
        peer_scatter_store_publish(source, indices, destination, queue, task)

    if compiled:
        operation = torch.compile(operation, fullgraph=True)
    operation(source, indices, output, queue, 0)
    result = torch.empty_like(output)
    combine_wait_reduce(output, result, queue, 1)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert queue[0, 2].item() == max(1, (rows + 15) // 16)
    assert queue[0, 3].item() == 1


@pytest.mark.parametrize(
    "assign,output_dtype",
    [(False, torch.float32), (True, torch.float32), (True, torch.bfloat16)],
)
@pytest.mark.parametrize("hidden,inner", [(256, 128), (2048, 768)])
@pytest.mark.parametrize("case", ["mixed", "skewed", "masked"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_grouped_bf16_scatter(
    hidden, inner, case, compiled, assign, output_dtype, bounded
):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 required")
    deep_gemm = pytest.importorskip("deep_gemm")
    if not hasattr(deep_gemm, "bf16_grouped_scatter_gemm_nt"):
        pytest.skip("MonoEP vendored DeepGEMM build required")
    from mono_ep.combine import (
        combine_wait_copy,
        combine_wait_reduce,
        grouped_bf16_scatter,
    )
    from mono_ep.ops import grouped_bf16
    from mono_ep.packing import DeviceExpertLayout

    torch.manual_seed(947)
    tokens, experts, topk = 137, 4, 3
    ids = torch.randint(experts, (tokens, topk), device="cuda", dtype=torch.int32)
    ids[::5] = -1
    if case == "skewed":
        ids[ids >= 0] = 0
    elif case == "masked":
        ids.fill_(-1)
    layout = DeviceExpertLayout.build(ids, experts, 128)
    source = torch.randn(tokens, inner, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
    x, w = layout.pack(source), weights.bfloat16()
    row_end = layout.ends[-1:] if bounded else None
    token_ids = layout.row_assignments.int() if assign else layout.token_ids()
    values = grouped_bf16(x, w, layout.expert_ids)
    valid = token_ids >= 0
    expected = torch.zeros(tokens * (topk if assign else 1), hidden, device="cuda")
    expected.index_add_(0, token_ids[valid].long(), values[valid].float())
    if compiled:
        torch.compiler.reset()
    operation = (
        torch.compile(grouped_bf16_scatter, fullgraph=True)
        if compiled
        else grouped_bf16_scatter
    )
    destination = torch.zeros_like(expected, dtype=output_dtype)
    queue = torch.zeros(1, 4, device="cuda", dtype=torch.int32)
    result = torch.empty_like(expected)

    def consume():
        if output_dtype == torch.bfloat16:
            combine_wait_reduce(destination, result, queue, 1)
        else:
            combine_wait_copy(destination, result, queue)

    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    previous_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(32)
    try:
        scratch = torch.full_like(values, 7)
        deep_gemm.bf16_grouped_scatter_gemm_nt(
            x,
            w,
            scratch,
            layout.expert_ids,
            token_ids,
            destination,
            queue,
            0,
            assign=assign,
            row_end=row_end,
        )
        torch.cuda.synchronize()
        assert torch.all(scratch == 7)
        torch.testing.assert_close(destination.float(), expected, atol=5e-5, rtol=1e-5)
        destination.zero_()
        queue.zero_()
        operation(
            x,
            w,
            layout.expert_ids,
            token_ids,
            destination,
            queue,
            0,
            assign,
            row_end,
        )
        consume()
        torch.cuda._sleep(1)
        torch.cuda.synchronize()
        for _ in range(3):
            destination.zero_()
            queue.zero_()
            producer.wait_stream(torch.cuda.current_stream())
            consumer.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(consumer):
                consume()
            with torch.cuda.stream(producer):
                torch.cuda._sleep(1_000_000)
                operation(
                    x,
                    w,
                    layout.expert_ids,
                    token_ids,
                    destination,
                    queue,
                    0,
                    assign,
                    row_end,
                )
            producer.synchronize()
            consumer.synchronize()
            torch.testing.assert_close(result, expected, atol=5e-5, rtol=1e-5)
            assert queue[0, 3].item() == 1
            assert queue[0, 2].item() > 0
    finally:
        deep_gemm.set_num_sms(previous_sms)


@pytest.mark.parametrize("rows,masked", [(0, False), (137, False), (137, True)])
@pytest.mark.parametrize("width", [16, 96, 192])
@pytest.mark.parametrize("compiled", [False, True])
def test_reduce_scatter_publish(rows, masked, width, compiled):
    from mono_ep.combine import combine_wait_copy, peer_reduce_scatter_publish

    torch.manual_seed(772)
    slots = 31
    partial = torch.randn(rows, width, device="cuda")
    indices = torch.randint(slots, (rows,), device="cuda", dtype=torch.int32)
    indices[::3] = -1
    if masked:
        indices.fill_(-1)
    valid = indices >= 0
    partial[~valid] = float("nan")
    expected = torch.zeros(slots, 1, device="cuda")
    expected.index_add_(0, indices[valid].long(), partial[valid].sum(-1, keepdim=True))
    expected *= 2
    destination = torch.zeros_like(expected)
    queue = torch.zeros(2, 4, device="cuda", dtype=torch.int32)
    result = torch.empty_like(expected)
    operation = peer_reduce_scatter_publish
    if compiled:
        torch.compiler.reset()
        operation = torch.compile(operation, fullgraph=True)
    for task in range(2):
        operation(partial, indices, destination, queue, task)
    torch.cuda.synchronize()
    producers = [torch.cuda.Stream(), torch.cuda.Stream()]
    consumer = torch.cuda.Stream()
    for _ in range(3):
        destination.zero_()
        queue.zero_()
        result.fill_(float("nan"))
        current = torch.cuda.current_stream()
        consumer.wait_stream(current)
        for producer in producers:
            producer.wait_stream(current)
        with torch.cuda.stream(consumer):
            combine_wait_copy(destination, result, queue)
        for task, producer in enumerate(producers):
            with torch.cuda.stream(producer):
                torch.cuda._sleep(1_000_000 * (task + 1))
                operation(partial, indices, destination, queue, task)
        current.wait_stream(consumer)
        for producer in producers:
            current.wait_stream(producer)
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)
        assert torch.all(queue[:, 3] == 1)
        assert torch.all(queue[:, 2] == max(1, (rows + 15) // 16))
