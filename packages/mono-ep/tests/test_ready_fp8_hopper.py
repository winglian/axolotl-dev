# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Blockwise FP8 grouped consumers acquire published Hopper payloads."""

import os

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]


@pytest.mark.parametrize("counts", [(128, 256), (0, 128), (0, 0)])
@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("compiled", [False, True])
def test_ready_grouped_fp8_hopper(counts, width, compiled):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Blockwise FP8 readiness requires SM90")
    deep_gemm = pytest.importorskip("deep_gemm")
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from mono_ep.chunk import ready_grouped_fp8_out
    from mono_ep.quantization import quantize_fp8

    @triton.jit
    def publish(queue):
        task = tl.program_id(0)
        tl.atomic_xchg(queue + task * 4 + 3, 1, sem="release", scope="sys")

    torch.compiler.reset()
    torch.manual_seed(921)
    rows, hidden = 512, 256
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weights = (
        torch.randn(2, width, hidden, device="cuda", dtype=torch.bfloat16) / hidden**0.5
    )
    qx_source, sx = quantize_fp8(source, block_cols=128)
    qw_source, raw_sw = quantize_fp8(weights, block_rows=128, block_cols=128)
    sx_source = deep_gemm.transform_sf_into_required_layout(sx, rows, hidden, (1, 128))
    sw_source = deep_gemm.transform_sf_into_required_layout(
        raw_sw, width, hidden, (128, 128), num_groups=2
    )

    ids_source = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    expected = torch.zeros(rows, width, device="cuda", dtype=torch.bfloat16)
    x_dequant = qx_source.float() * sx.repeat_interleave(128, dim=-1)[:, :hidden]
    w_dequant = (
        qw_source.float()
        * raw_sw.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)[
            :, :width, :hidden
        ]
    )
    descriptors = []
    start = 0
    for expert, count in enumerate(counts):
        ids_source[start : start + count] = expert
        expected[start : start + count] = (
            x_dequant[start : start + count] @ w_dequant[expert].T
        ).bfloat16()
        descriptors.append([expert, start, count, 0])
        start += count

    descriptor_source = torch.tensor(descriptors, device="cuda", dtype=torch.int32)
    queue = descriptor_source.clone()
    queue[:, 3] = 1
    qx, transformed_sx, qw, sw, ids = (
        qx_source.clone(),
        sx_source.clone(),
        qw_source.clone(),
        sw_source.clone(),
        ids_source.clone(),
    )
    out = torch.zeros_like(expected)
    operation = (
        torch.compile(ready_grouped_fp8_out, fullgraph=True)
        if compiled
        else ready_grouped_fp8_out
    )
    arguments = (qx, transformed_sx, qw, sw, out, queue, ids, True)
    old_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(32)
    try:
        # Lazy module loading of a producer kernel waits for the spinning consumer to exit
        publish[(2,)](queue)
        torch.cuda._sleep(1)
        operation(*arguments)
        torch.cuda.synchronize()
        delayed = set(os.environ.get("ULTRA_FP8_DELAY", "all").split(","))
        producer = torch.cuda.Stream(
            priority=-1 if os.environ.get("ULTRA_FP8_PRODUCER_PRIORITY") else 0
        )
        consumer = torch.cuda.Stream()
        for _ in range(3):
            if "all" in delayed or "a" in delayed:
                qx.fill_(float("nan"))
            if "all" in delayed or "sfa" in delayed:
                transformed_sx.fill_(float("nan"))
            if "all" in delayed or "b" in delayed:
                qw.fill_(float("nan"))
            if "all" in delayed or "sfb" in delayed:
                sw.fill_(float("nan"))
            if "all" in delayed or "ids" in delayed:
                ids.fill_(-1)
            queue.zero_()
            out.zero_()
            current = torch.cuda.current_stream()
            producer.wait_stream(current)
            consumer.wait_stream(current)
            with torch.cuda.stream(consumer):
                operation(*arguments)
            with torch.cuda.stream(producer):
                torch.cuda._sleep(int(os.environ.get("ULTRA_FP8_SLEEP", "10000000")))
                if "all" in delayed or "a" in delayed:
                    qx.copy_(qx_source)
                if "all" in delayed or "sfa" in delayed:
                    transformed_sx.copy_(sx_source)
                if "all" in delayed or "b" in delayed:
                    qw.copy_(qw_source)
                if "all" in delayed or "sfb" in delayed:
                    sw.copy_(sw_source)
                if "all" in delayed or "ids" in delayed:
                    ids.copy_(ids_source)
                queue.copy_(descriptor_source)
                publish[(2,)](queue)
            current.wait_stream(producer)
            current.wait_stream(consumer)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, expected, atol=4e-2, rtol=3e-2)
    finally:
        deep_gemm.set_num_sms(old_sms)


def test_ready_grouped_fp8_hopper_rejects_untransformed_scales():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Blockwise FP8 readiness requires SM90")
    pytest.importorskip("deep_gemm")
    from mono_ep.chunk import ready_grouped_fp8_out
    from mono_ep.quantization import quantize_fp8

    rows = width = 128
    hidden = 256
    x, sx = quantize_fp8(
        torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16),
        block_cols=128,
    )
    w, sw = quantize_fp8(
        torch.randn(1, width, hidden, device="cuda", dtype=torch.bfloat16),
        block_rows=128,
        block_cols=128,
    )
    out = torch.zeros(rows, width, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(rows, device="cuda", dtype=torch.int32)
    queue = torch.tensor([[0, 0, rows, 1]], device="cuda", dtype=torch.int32)
    padded_stride_sx = torch.empty_strided(
        sx.shape, (1, rows + 4), device="cuda", dtype=torch.float32
    )
    for invalid_sx in (sx, padded_stride_sx):
        with pytest.raises(RuntimeError):
            ready_grouped_fp8_out(x, invalid_sx, w, sw, out, queue, ids, True)
