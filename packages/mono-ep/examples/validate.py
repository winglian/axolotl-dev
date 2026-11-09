# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate DeepGEMM training primitives against independent PyTorch autograd."""

import argparse
import json
import os
from itertools import product

import torch
import torch.distributed as dist
from mono_ep.grouped import (
    ExpertLayout,
    expert_backward,
    expert_forward,
)


def reference(x, scores, indices, w1, w2):
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        token, slot = torch.where(indices == expert)
        z = x[token] @ w1[expert].T
        gate, up = z.float().chunk(2, dim=-1)
        h = (torch.nn.functional.silu(gate) * up * scores[token, slot, None]).to(
            x.dtype
        )
        values = h @ w2[expert].T
        result = result.index_add(0, token, values.float())
    return result.to(x.dtype)


def validate_fp8_gemm():
    from mono_ep.fp8 import expert_gemm_fp8
    from mono_ep.quantization import quantize_fp8

    major = torch.cuda.get_device_capability()[0]
    if major not in (9, 10):
        raise RuntimeError("FP8 grouped validation requires SM90 or SM100")
    format = "mxfp8" if major == 10 else "blockwise"
    cols, rows = (32, 1) if major == 10 else (128, 128)
    torch.manual_seed(124)
    torch.backends.cuda.matmul.allow_tf32 = False
    for tokens in (0, 512):
        x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2, 256, 256, device="cuda", dtype=torch.bfloat16) / 16
        ids = torch.arange(tokens, device="cuda", dtype=torch.int32) // 256
        if tokens:
            ids[239:256] = -1
            ids[495:] = -1

        def dequantize(t, block_rows):
            q, s = quantize_fp8(t, block_rows=block_rows, block_cols=cols)
            return (
                q.float()
                * s.repeat_interleave(block_rows, -2).repeat_interleave(cols, -1)[
                    ..., : t.shape[-2], : t.shape[-1]
                ]
            )

        qx, qw = dequantize(x, 1), dequantize(w, rows)
        expected = torch.zeros(tokens, 256, device="cuda", dtype=torch.bfloat16)
        bf16 = torch.zeros_like(expected)
        for expert in range(2):
            mask = ids == expert
            expected[mask] = (qx[mask] @ qw[expert].T).bfloat16()
            bf16[mask] = x[mask] @ w[expert].T
        for compiled in (False, True):
            operation = (
                torch.compile(expert_gemm_fp8, fullgraph=True)
                if compiled
                else expert_gemm_fp8
            )
            actual = operation(x, w, ids, format=format)
            errors = {}
            for name, ref, limit in (
                ("quantized_reference", expected, 0.01),
                ("bf16_reference", bf16, 0.06),
            ):
                error = (
                    (actual.float() - ref.float()).norm()
                    / ref.float().norm().clamp_min(1e-8)
                ).item()
                assert torch.isfinite(actual).all() and error < limit, (
                    format,
                    compiled,
                    name,
                    error,
                )
                errors[name] = error
            assert torch.count_nonzero(actual[ids < 0]) == 0
            print(
                json.dumps(
                    {
                        "fp8_gemm": format,
                        "tokens": tokens,
                        "compiled": compiled,
                        "relative_l2": errors,
                    }
                ),
                flush=True,
            )
    validate_fp8_weight_grad(format)
    validate_fp8_experts(format)


def validate_fp8_experts(format):
    from mono_ep.fp8 import expert_backward_fp8, expert_forward_fp8
    from mono_ep.fp8_autograd import fp8_experts
    from mono_ep.packing import DeviceExpertLayout

    torch.manual_seed(157)
    for unrouted in (False, True):
        x = torch.randn(137, 256, device="cuda", dtype=torch.bfloat16).requires_grad_()
        scores = torch.rand(137, 2, device="cuda").requires_grad_()
        ids = torch.rand(137, 4, device="cuda").topk(2).indices
        ids[::7] = -1
        if unrouted:
            ids.fill_(-1)
        w1 = (
            torch.randn(4, 512, 256, device="cuda", dtype=torch.bfloat16) / 16
        ).requires_grad_()
        w2 = (
            torch.randn(4, 256, 256, device="cuda", dtype=torch.bfloat16) / 16
        ).requires_grad_()
        dy = torch.randn_like(x)
        expected = reference(x, scores, ids, w1, w2)
        refs = (expected, *torch.autograd.grad(expected, (x, scores, w1, w2), dy))
        layout_type = DeviceExpertLayout if format == "mxfp8" else ExpertLayout
        layout = layout_type.build(ids, 4, 128)
        for bf16_forward in (False, True):
            for compiled in (False, True):
                forward = (
                    torch.compile(expert_forward_fp8, fullgraph=True)
                    if compiled
                    else expert_forward_fp8
                )
                backward = (
                    torch.compile(expert_backward_fp8, fullgraph=True)
                    if compiled
                    else expert_backward_fp8
                )
                inputs = [t.detach() for t in (x, scores, w1, w2)]
                actual = forward(
                    *inputs, layout, format=format, bf16_forward=bf16_forward
                )
                grads = backward(
                    dy, *inputs, layout, format=format, bf16_forward=bf16_forward
                )
                errors = {}
                for name, value, ref in zip(
                    ("y", "dx", "dscore", "dw1", "dw2"),
                    (actual, *grads),
                    refs,
                    strict=True,
                ):
                    error = (
                        (value.float() - ref.float()).norm()
                        / ref.float().norm().clamp_min(1e-8)
                    ).item()
                    assert torch.isfinite(value).all() and error < 0.15, (
                        format,
                        unrouted,
                        bf16_forward,
                        compiled,
                        name,
                        error,
                    )
                    errors[name] = error
                assert torch.count_nonzero(grads[1][ids < 0]) == 0
                autograd_operation = (
                    torch.compile(fp8_experts, fullgraph=True)
                    if compiled
                    else fp8_experts
                )
                for master_dtype in (torch.bfloat16, torch.float32):
                    leaves = [
                        inputs[0].requires_grad_(),
                        inputs[1].requires_grad_(),
                        inputs[2].to(master_dtype).detach().requires_grad_(),
                        inputs[3].to(master_dtype).detach().requires_grad_(),
                    ]
                    y_autograd = autograd_operation(
                        leaves[0],
                        ids,
                        leaves[1],
                        leaves[2],
                        leaves[3],
                        format,
                        bf16_forward,
                    )
                    g_autograd = torch.autograd.grad(y_autograd, leaves, dy)
                    torch.testing.assert_close(y_autograd, actual, rtol=1e-5, atol=1e-5)
                    for g, ref, leaf in zip(g_autograd, grads, leaves, strict=True):
                        torch.testing.assert_close(
                            g, ref.to(leaf.dtype), rtol=1e-5, atol=1e-5
                        )
                    print(
                        json.dumps(
                            {
                                "fp8_autograd": format,
                                "compiled": compiled,
                                "master_dtype": str(master_dtype),
                                "bf16_forward": bf16_forward,
                                "unrouted": unrouted,
                                "matches_explicit": True,
                            }
                        ),
                        flush=True,
                    )
                print(
                    json.dumps(
                        {
                            "fp8_experts": format,
                            "unrouted": unrouted,
                            "bf16_forward": bf16_forward,
                            "compiled": compiled,
                            "relative_l2_vs_bf16": errors,
                        }
                    ),
                    flush=True,
                )


def validate_fp8_weight_grad(format):
    from mono_ep.ops import weight_grad_fp8
    from mono_ep.quantization import quantize_fp8

    block = 32 if format == "mxfp8" else 128
    for sizes in ([0, 0, 0], [256, 0, 128]):
        total = sum(sizes)
        x = torch.randn(total, 512, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn(total, 256, device="cuda", dtype=torch.bfloat16)
        if total:
            x[239:256] = 0
            dy[239:256] = 0
            x[-19:] = 0
            dy[-19:] = 0
        if format == "blockwise":
            from mono_ep.wgrad_quantization import quantize_transposed_groups

            for operand in (x, dy):
                quantized = [
                    quantize_fp8(part.T.contiguous(), block_cols=128)
                    for part in operand.split(sizes)
                    if part.shape[0]
                ]
                ref_q = (
                    torch.cat([q.flatten() for q, _ in quantized])
                    if quantized
                    else torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
                )
                ref_s = (
                    torch.cat([s for _, s in quantized], dim=1)
                    if quantized
                    else torch.empty(operand.shape[1], 0, device="cuda")
                )
                for compiled in (False, True):
                    operation = (
                        torch.compile(quantize_transposed_groups, fullgraph=True)
                        if compiled
                        else quantize_transposed_groups
                    )
                    q, s = operation(
                        operand, torch.tensor(sizes, device="cuda", dtype=torch.int32)
                    )
                    torch.testing.assert_close(q.float(), ref_q.float(), rtol=0, atol=0)
                    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)
                    print(
                        json.dumps(
                            {
                                "fused_wgrad_quantization": "exact",
                                "sizes": sizes,
                                "channels": operand.shape[1],
                                "compiled": compiled,
                            }
                        ),
                        flush=True,
                    )
        expected = torch.zeros(3, 256, 512, device="cuda")
        bf16 = torch.zeros_like(expected)
        start = 0
        for expert, size in enumerate(sizes):
            if size:
                a, b = (
                    dy[start : start + size].T.contiguous(),
                    x[start : start + size].T.contiguous(),
                )
                qa, sa = quantize_fp8(a, block_cols=block)
                qb, sb = quantize_fp8(b, block_cols=block)
                da = qa.float() * sa.repeat_interleave(block, -1)
                db = qb.float() * sb.repeat_interleave(block, -1)
                expected[expert] = da @ db.T
                bf16[expert] = a.float() @ b.float().T
            start += size
        for psum in (False, True) if format == "mxfp8" else (False,):
            layout = torch.tensor(sizes, device="cuda", dtype=torch.int32)
            if psum:
                layout = layout.cumsum(0, dtype=torch.int32)
            for compiled in (False, True):
                operation = (
                    torch.compile(weight_grad_fp8, fullgraph=True)
                    if compiled
                    else weight_grad_fp8
                )
                actual = operation(
                    dy, x, layout, None if psum else sizes, 3, psum, format
                )
                errors = {}
                for name, ref, limit in (
                    ("quantized_reference", expected, 0.01),
                    ("bf16_reference", bf16, 0.06),
                ):
                    error = ((actual - ref).norm() / ref.norm().clamp_min(1e-8)).item()
                    assert torch.isfinite(actual).all() and error < limit, (
                        format,
                        sizes,
                        psum,
                        compiled,
                        name,
                        error,
                    )
                    errors[name] = error
                assert torch.count_nonzero(actual[1]) == 0
                print(
                    json.dumps(
                        {
                            "fp8_weight_grad": format,
                            "sizes": sizes,
                            "psum": psum,
                            "compiled": compiled,
                            "relative_l2": errors,
                        }
                    ),
                    flush=True,
                )


def validate_epilogue():
    from mono_ep.activation import swiglu
    from mono_ep.ops import grouped_bf16, grouped_swiglu_backward

    torch.manual_seed(907)
    compiled = torch.compile(grouped_swiglu_backward, fullgraph=True)
    for rows, hidden, inner in ((0, 256, 256), (512, 256, 256), (2048, 512, 768)):
        dy = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(3, hidden, inner, device="cuda", dtype=torch.bfloat16)
        w2 /= hidden**0.5
        ids = (torch.arange(rows, device="cuda", dtype=torch.int32) // 128) % 4
        ids[ids == 3] = -1
        z = torch.randn(rows, 2 * inner, device="cuda", dtype=torch.bfloat16)
        scores = torch.randn(rows, device="cuda", dtype=torch.float32)
        scores[::5] = 0
        if rows > 1:
            edges = torch.tensor(
                [-100, -90, -88, -20, -1, -0.0001, 0, 0.0001, 1, 20, 88, 90, 100],
                device="cuda",
                dtype=torch.bfloat16,
            )
            z[1, :inner] = edges.repeat((inner + edges.numel() - 1) // edges.numel())[
                :inner
            ]
        z[ids < 0] = 0
        scores[ids < 0] = 0
        dh = grouped_bf16(dy, w2.transpose(1, 2).contiguous(), ids)
        expected = swiglu(z, scores, dh)
        z[ids < 0] = float("nan")
        scores[ids < 0] = float("nan")
        for mode, operation in (
            ("eager", grouped_swiglu_backward),
            ("fullgraph", compiled),
        ):
            actual = operation(dy, w2, ids, z, scores)
            errors = {}
            for name, value, ref in zip(
                ("h", "dz", "ds"), actual, expected, strict=True
            ):
                assert torch.isfinite(value).all(), (mode, name)
                assert torch.count_nonzero(value[ids < 0]) == 0, (mode, name, "padding")
                relative = (
                    (value.float() - ref.float()).norm()
                    / ref.float().norm().clamp_min(1e-8)
                ).item()
                assert relative < 0.01, (mode, rows, inner, name, relative)
                errors[name] = relative
            print(
                json.dumps(
                    {
                        "epilogue": mode,
                        "rows": rows,
                        "inner": inner,
                        "relative_l2": errors,
                    }
                )
            )


def validate_quantization():
    from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
    from mono_ep.quantization import quantize_fp8

    torch.manual_seed(35)
    compiled = torch.compile(quantize_fp8, fullgraph=True)
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(2, 137, 257, dtype=dtype, device="cuda")
        x[1] *= 64
        x[:, 0] = 0
        for rows, cols in ((1, 32), (1, 128), (128, 128)):
            operation = per_token_cast_to_fp8 if rows == 1 else per_block_cast_to_fp8
            pairs = [operation(expert, use_ue8m0=True, gran_k=cols) for expert in x]
            expected = [torch.stack([pair[i] for pair in pairs]) for i in (0, 1)]
            for mode, quantize in (("eager", quantize_fp8), ("fullgraph", compiled)):
                actual = quantize(x, block_rows=rows, block_cols=cols)
                for value, reference_value in zip(actual, expected, strict=True):
                    torch.testing.assert_close(
                        value.float(), reference_value.float(), atol=0, rtol=0
                    )
                print(
                    json.dumps(
                        {
                            "quantization": mode,
                            "dtype": str(dtype),
                            "block": [rows, cols],
                            "exact": True,
                        }
                    ),
                    flush=True,
                )

    from mono_ep.column_quantization import quantize_columns

    column_compiled = torch.compile(quantize_columns, fullgraph=True)
    for dtype in (torch.bfloat16, torch.float32):
        for shape in ((137, 257), (128, 256), (0, 256)):
            x = torch.randn(shape, dtype=dtype, device="cuda")
            if shape[0]:
                x[:32, 0] = 0
            if shape[0]:
                q, sf = per_token_cast_to_fp8(
                    x.T.contiguous(), use_ue8m0=True, gran_k=32
                )
                expected = (q.T.contiguous(), sf.T.contiguous())
            else:
                expected = (x.to(torch.float8_e4m3fn), x.float())
            for mode, operation in (
                ("eager", quantize_columns),
                ("fullgraph", column_compiled),
            ):
                for actual, ref in zip(operation(x), expected, strict=True):
                    torch.testing.assert_close(
                        actual.float(), ref.float(), atol=0, rtol=0
                    )
                print(
                    json.dumps(
                        {
                            "column_quantization": mode,
                            "dtype": str(dtype),
                            "shape": shape,
                            "exact": True,
                        }
                    ),
                    flush=True,
                )

    from deep_gemm import transform_weights_for_mono_ep
    from deep_gemm.utils.math import pack_ue8m0_to_int
    from mono_ep.token_quantization import prepare_mega_fp8

    for dtype in (torch.bfloat16, torch.float32):
        w1 = torch.randn(3, 512, 256, device="cuda", dtype=dtype)
        w2 = torch.randn(3, 256, 256, device="cuda", dtype=dtype)
        w1[:, 0] = 0
        pairs = [quantize_fp8(w) for w in (w1, w2)]
        expected = transform_weights_for_mono_ep(
            *[(q, pack_ue8m0_to_int(s)) for q, s in pairs]
        )
        for mode, op in (
            ("eager", prepare_mega_fp8),
            ("fullgraph", torch.compile(prepare_mega_fp8, fullgraph=True)),
        ):
            for w, interleave, ref in zip(
                (w1, w2), (True, False), expected, strict=True
            ):
                q, sf = op(w, weights=True, interleave=interleave)
                torch.testing.assert_close(q.float(), ref[0].float(), atol=0, rtol=0)
                torch.testing.assert_close(sf, ref[1], atol=0, rtol=0)
                assert sf.stride(-2) == 1
            x = w2[0]
            q, sf = op(x)
            rq, rs = quantize_fp8(x)
            torch.testing.assert_close(q.float(), rq.float(), atol=0, rtol=0)
            torch.testing.assert_close(sf, pack_ue8m0_to_int(rs), atol=0, rtol=0)
            print(
                json.dumps(
                    {"mega_operand_layout": mode, "dtype": str(dtype), "exact": True}
                ),
                flush=True,
            )


def validate_sonic():
    from mono_ep.sonic import sonic_experts

    torch.manual_seed(53)
    compiled = torch.compile(sonic_experts, fullgraph=True)
    for pattern in ("random", "masked", "unrouted", "empty"):
        tokens = 0 if pattern == "empty" else 37
        x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
        scores, indices = torch.randn(tokens, 8, device="cuda").softmax(-1).topk(4)
        if pattern == "masked":
            indices[:, 1::2] = -1
        elif pattern == "unrouted":
            indices[:] = -1
        for dtype in (torch.bfloat16, torch.float32):
            w1 = (torch.randn(8, 512, 256, device="cuda") / 16).to(dtype)
            w2 = (torch.randn(8, 256, 256, device="cuda") / 16).to(dtype)
            leaves = [t.detach().requires_grad_() for t in (x, scores, w1, w2)]
            expected = reference(
                leaves[0],
                leaves[1],
                indices,
                leaves[2].bfloat16(),
                leaves[3].bfloat16(),
            )
            dy = torch.randn_like(x)
            expected_grads = torch.autograd.grad(expected, leaves, dy)
            for mode, operation in (("eager", sonic_experts), ("fullgraph", compiled)):
                actual = operation(leaves[0], indices, leaves[1], leaves[2], leaves[3])
                actual_grads = torch.autograd.grad(actual, leaves, dy)
                errors = {}
                for name, value, ref in zip(
                    ("y", "dx", "ds", "dw1", "dw2"),
                    (actual, *actual_grads),
                    (expected, *expected_grads),
                    strict=True,
                ):
                    relative = (
                        (value.float() - ref.float()).norm()
                        / ref.float().norm().clamp_min(1e-8)
                    ).item()
                    assert relative < 0.02, (pattern, str(dtype), mode, name, relative)
                    errors[name] = relative
                assert torch.count_nonzero(actual_grads[1][indices < 0]) == 0
                print(
                    json.dumps(
                        {
                            "sonic_adapter": mode,
                            "routing": pattern,
                            "master_dtype": str(dtype),
                            "relative_l2": errors,
                        }
                    ),
                    flush=True,
                )


def validate():
    import deep_gemm

    if torch.cuda.get_device_capability()[0] == 10:
        validate_epilogue()
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    reports = []
    from mono_ep.packing import DeviceExpertLayout

    def device_pipeline(x, scores, indices, w1, w2, dy):
        layout = DeviceExpertLayout.build(indices, w1.shape[0], alignment)
        y = expert_forward(x, scores, w1, w2, layout)
        return y, *expert_backward(dy, x, scores, w1, w2, layout)

    compiled_pipeline = torch.compile(device_pipeline, fullgraph=True)
    for pattern in ("random", "skewed", "masked", "empty"):
        tokens, hidden, inner, experts, topk = 137, 256, 256, 8, 4
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        w1 = (
            torch.randn(experts, 2 * inner, hidden, device="cuda", dtype=torch.bfloat16)
            / hidden**0.5
        )
        w2 = (
            torch.randn(experts, hidden, inner, device="cuda", dtype=torch.bfloat16)
            / inner**0.5
        )
        scores, indices = (
            torch.randn(tokens, experts, device="cuda").softmax(-1).topk(topk)
        )
        if pattern == "skewed":
            indices[:] = torch.arange(topk, device="cuda")
        elif pattern == "masked":
            indices[::3] = -1
        elif pattern == "empty":
            indices[:] = -1
        inputs = [t.detach().requires_grad_() for t in (x, scores, w1, w2)]
        expected = reference(inputs[0], inputs[1], indices, inputs[2], inputs[3])
        dy = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(expected, inputs, dy)
        layout = ExpertLayout.build(indices, experts, alignment)
        actual = expert_forward(x, scores, w1, w2, layout)
        actual_grads = expert_backward(dy, x, scores, w1, w2, layout)
        candidates = [("host_layout_eager", (actual, *actual_grads))]
        if torch.cuda.get_device_capability()[0] == 10:
            candidates.append(
                (
                    "device_layout_fullgraph",
                    compiled_pipeline(x, scores, indices, w1, w2, dy),
                )
            )
        for mode, outputs in candidates:
            errors = {}
            for name, actual_t, expected_t in zip(
                ("y", "dx", "dscore", "dw1", "dw2"),
                outputs,
                (expected, *expected_grads),
                strict=True,
            ):
                assert torch.isfinite(actual_t).all(), (pattern, name)
                delta = (actual_t.float() - expected_t.float()).norm()
                relative = (delta / expected_t.float().norm().clamp_min(1e-8)).item()
                assert relative < 0.02, (pattern, name, relative)
                errors[name] = relative
            reports.append({"routing": pattern, "mode": mode, "relative_l2": errors})
    print(json.dumps({"device": torch.cuda.get_device_name(), "checks": reports}))


def validate_distributed(
    sonic_pipeline=False, fp8_pipeline=False, fused_fp8=False, framework_ep=False
):
    from mono_ep.runtime import MonoEP

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    tokens, hidden, inner, experts, topk = 137, 512, 512, 8, 4
    torch.manual_seed(43)
    w1 = (
        torch.randn(experts, 2 * inner, hidden, device="cuda", dtype=torch.bfloat16)
        / hidden**0.5
    )
    w2 = (
        torch.randn(experts, hidden, inner, device="cuda", dtype=torch.bfloat16)
        / inner**0.5
    )
    torch.manual_seed(100 + rank)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    scores, indices = torch.randn(tokens, experts, device="cuda").softmax(-1).topk(topk)
    indices[::7] = -1
    if sonic_pipeline or fp8_pipeline:
        indices[: tokens // 3] = -1
    dy = torch.randn_like(x)
    if not (sonic_pipeline or fp8_pipeline):
        from mono_ep.pipeline import _compute_send_counts

        _, _, token_counts = _compute_send_counts(indices, world, experts // world)
        print(
            json.dumps(
                {
                    "traffic_rank": rank,
                    "assignment_rows": int((indices >= 0).sum().item()),
                    "token_destination_rows": int(token_counts.sum().item()),
                }
            ),
            flush=True,
        )

    def gather(t):
        parts = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(parts, t)
        return torch.cat(parts)

    all_x, all_scores, all_indices, all_dy = [
        gather(t) for t in (x, scores, indices, dy)
    ]
    runtime = (
        None
        if sonic_pipeline or fp8_pipeline
        else MonoEP(
            dist.group.WORLD,
            experts,
            tokens,
            topk,
            hidden,
            inner,
            compute_dtype="mxfp8" if fused_fp8 else "bf16",
        )
    )
    cases = (
        product(
            (torch.bfloat16, torch.float32), (False, True), (False, True), (False, True)
        )
        if fp8_pipeline
        else ((dtype, True, False, False) for dtype in (torch.bfloat16, torch.float32))
    )
    if framework_ep:
        cases = product(
            (torch.bfloat16, torch.float32), (True,), (False,), (False, True)
        )
    if sonic_pipeline:
        cases = product(
            (torch.bfloat16, torch.float32), (True,), (False,), (False, True)
        )
    for master_dtype, overlap, bf16_forward, retain in cases:
        if (fp8_pipeline or sonic_pipeline) and retain:
            torch._dynamo.reset()
        full_inputs = [
            t.detach().requires_grad_()
            for t in (all_x, all_scores, w1.to(master_dtype), w2.to(master_dtype))
        ]
        expected = reference(
            full_inputs[0],
            full_inputs[1],
            all_indices,
            full_inputs[2].bfloat16(),
            full_inputs[3].bfloat16(),
        )
        expected_grads = torch.autograd.grad(expected, full_inputs, all_dy)
        start, end = rank * (experts // world), (rank + 1) * (experts // world)
        inputs = [
            t.detach().clone().requires_grad_()
            for t in (
                x,
                scores,
                w1[start:end].to(master_dtype),
                w2[start:end].to(master_dtype),
            )
        ]
        if sonic_pipeline or fp8_pipeline:
            from mono_ep.pipeline import pipelined_moe

            if fp8_pipeline:
                from mono_ep.fp8_autograd import FP8Experts

                class CountedExperts(FP8Experts):
                    forwards = 0
                    backwards = 0

                    def __call__(self, *args):
                        self.forwards += 1
                        return super().__call__(*args)

                    def forward_with_activations(self, *args):
                        self.forwards += 1
                        return super().forward_with_activations(*args)

                    def backward(self, *args, **kwargs):
                        self.backwards += 1
                        assert (kwargs.get("activations") is not None) == self.retain
                        return super().backward(*args, **kwargs)

                format = (
                    "mxfp8"
                    if torch.cuda.get_device_capability()[0] == 10
                    else "blockwise"
                )
                kernel = CountedExperts(
                    format, bf16_forward=bf16_forward, compile_local=retain
                )
                kernel.retain = retain
                local_backward = kernel.backward
            else:
                from mono_ep import MonoEP

                runtime = MonoEP(
                    dtype="bf16",
                    compile_local=retain,
                    retain_activations=retain,
                    chunks=3,
                )

            if framework_ep:
                from axolotl_mono_ep_adapter import fp8_forward
                from transformers.models.qwen3_moe.configuration_qwen3_moe import (
                    Qwen3MoeConfig,
                )
                from transformers.models.qwen3_moe.modeling_qwen3_moe import (
                    Qwen3MoeExperts,
                )

                module = Qwen3MoeExperts(
                    Qwen3MoeConfig(
                        hidden_size=hidden,
                        moe_intermediate_size=inner,
                        num_experts=experts // world,
                        num_experts_per_tok=topk,
                    )
                )
                module.gate_up_proj = torch.nn.Parameter(inputs[2])
                module.down_proj = torch.nn.Parameter(inputs[3])
                inputs[2:] = [module.gate_up_proj, module.down_proj]
                actual = fp8_forward(
                    module,
                    inputs[0],
                    indices,
                    inputs[1],
                    group=dist.group.WORLD,
                    options=(format, retain, retain),
                    chunks=3,
                )
            elif sonic_pipeline:
                actual = runtime(
                    inputs[0],
                    indices,
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    group=dist.group.WORLD,
                )
            else:
                actual = pipelined_moe(
                    inputs[0],
                    indices,
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    group=dist.group.WORLD,
                    local_kernel=kernel,
                    local_backward=local_backward,
                    local_forward=kernel.forward_with_activations if retain else None,
                    chunks=3,
                    overlap_backward=overlap,
                )
        else:
            actual = runtime(inputs[0], indices, inputs[1], inputs[2], inputs[3])
        actual_grads = torch.autograd.grad(actual, inputs, dy)
        if fp8_pipeline and not framework_ep:
            assert kernel.forwards == kernel.backwards == 3
        expected_local = (
            expected[rank * tokens : (rank + 1) * tokens],
            expected_grads[0][rank * tokens : (rank + 1) * tokens],
            expected_grads[1][rank * tokens : (rank + 1) * tokens],
            expected_grads[2][start:end],
            expected_grads[3][start:end],
        )
        errors = {}
        for name, actual_t, expected_t in zip(
            ("y", "dx", "dscore", "dw1", "dw2"),
            (actual, *actual_grads),
            expected_local,
            strict=True,
        ):
            assert torch.isfinite(actual_t).all(), (rank, name)
            relative = (
                (actual_t.float() - expected_t.float()).norm()
                / expected_t.float().norm().clamp_min(1e-8)
            ).item()
            assert relative < (0.15 if fp8_pipeline or fused_fp8 else 0.03), (
                rank,
                name,
                relative,
            )
            errors[name] = relative
        print(
            json.dumps(
                {
                    "rank": rank,
                    "master_dtype": str(master_dtype),
                    "framework_adapter": framework_ep,
                    "backend": "fp8_pipeline"
                    if fp8_pipeline
                    else "sonic_pipeline"
                    if sonic_pipeline
                    else "deepgemm_fused_mxfp8"
                    if fused_fp8
                    else "deepgemm",
                    "overlap_backward": overlap,
                    "bf16_forward": bf16_forward,
                    "retain_activations": retain,
                    "local_experts_fullgraph": (fp8_pipeline or sonic_pipeline)
                    and retain,
                    "relative_l2": errors,
                }
            )
        )
    dist.barrier()
    if runtime is not None and not sonic_pipeline:
        runtime.buffer.destroy()
    dist.destroy_process_group()


def validate_changing_routing():
    from torch._dynamo.utils import counters
    from mono_ep import FP8Experts, pipelined_moe

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    format = "mxfp8" if torch.cuda.get_device_capability()[0] == 10 else "blockwise"
    torch.manual_seed(125 + rank)
    local_experts, hidden, inner, tokens, topk = 4, 256, 256, 769, 2
    params = [
        torch.nn.Parameter(torch.randn(shape, device="cuda") / hidden**0.5)
        for shape in (
            (local_experts, 2 * inner, hidden),
            (local_experts, hidden, inner),
        )
    ]
    copies = [torch.nn.Parameter(p.detach().clone()) for p in params]
    optimizers = [torch.optim.AdamW(ps, lr=1e-4) for ps in (params, copies)]
    eager = FP8Experts(format)
    compiled = FP8Experts(format, compile_local=True)
    torch._dynamo.reset()
    counters.clear()
    for step in range(20):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        max_error = 0.0
        for micro in range(2):
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
            scores, ids = torch.rand(
                tokens, local_experts * dist.get_world_size(), device="cuda"
            ).topk(topk)
            scores /= scores.sum(-1, keepdim=True)
            ids[:: (3 + step % 7)] = -1
            if (step + micro) % 5 == 0:
                ids[: tokens // 3] = -1
            dy = torch.randn_like(x) / tokens
            results = []
            for kernel, weights in ((eager, params), (compiled, copies)):
                leaves = [
                    x.detach().requires_grad_(),
                    scores.detach().requires_grad_(),
                    *weights,
                ]
                y = pipelined_moe(
                    leaves[0],
                    ids,
                    leaves[1],
                    *weights,
                    group=dist.group.WORLD,
                    local_kernel=kernel,
                    local_forward=kernel.forward_with_activations,
                    local_backward=kernel.backward,
                    chunks=3,
                )
                grads = torch.autograd.grad(y, leaves, dy)
                results.append((y, *grads))
                for p, grad in zip(weights, grads[2:], strict=True):
                    if p.grad is None:
                        p.grad = grad / 2
                    else:
                        p.grad.add_(grad / 2)
            for actual, expected in zip(results[1], results[0], strict=True):
                error = (
                    (actual.float() - expected.float()).norm()
                    / expected.float().norm().clamp_min(1e-8)
                ).item()
                assert torch.isfinite(actual).all() and error < 0.02, (
                    rank,
                    step,
                    micro,
                    error,
                )
                max_error = max(max_error, error)
        for opt in optimizers:
            opt.step()
        weight_errors = [
            (a - b).norm().item() / a.norm().item()
            for a, b in zip(params, copies, strict=True)
        ]
        assert max(weight_errors) < 0.001, (rank, step, weight_errors)
        print(
            json.dumps(
                {
                    "changing_routing_rank": rank,
                    "tokens_per_rank": tokens,
                    "chunks": 3,
                    "optimizer": "AdamW",
                    "learning_rate": 1e-4,
                    "format": format,
                    "optimizer_step": step,
                    "microbatches": 2,
                    "relative_l2_eager_compiled_max": max_error,
                    "weight_relative_l2": weight_errors,
                    "unique_graphs": counters["stats"]["unique_graphs"],
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument("--sonic", action="store_true")
    parser.add_argument("--fp8-gemm", action="store_true")
    parser.add_argument("--fused-fp8", action="store_true")
    parser.add_argument("--changing-routing", action="store_true")
    parser.add_argument("--framework-ep", action="store_true")
    args = parser.parse_args()
    if args.framework_ep:
        validate_distributed(fp8_pipeline=True, framework_ep=True)
    elif args.changing_routing:
        validate_changing_routing()
    elif args.fused_fp8:
        validate_distributed(fused_fp8=True)
    elif args.fp8_gemm:
        if args.distributed:
            validate_distributed(fp8_pipeline=True)
        else:
            validate_fp8_gemm()
    elif args.sonic:
        if args.distributed:
            validate_distributed(sonic_pipeline=True)
        else:
            validate_sonic()
    elif args.quantization:
        validate_quantization()
    else:
        (validate_distributed if args.distributed else validate)()
