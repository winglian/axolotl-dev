# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Evaluate upstream SonicMoE PR 65 without enabling it in the training backend."""

import argparse
import json
from functools import partial
from importlib.metadata import version

import torch
from sonicmoe import moe_general_routing_inputs
from sonicmoe.enums import ActivationType
from validate import reference


def operation(x, scores, indices, w1, w2, interleaved=False):
    if interleaved:
        w1 = w1.unflatten(1, (2, -1)).transpose(1, 2).flatten(1, 2)
    token_ids = torch.arange(x.shape[0], device=x.device, dtype=torch.int32)
    return moe_general_routing_inputs(
        x,
        scores.masked_fill(indices < 0, 0).flatten(),
        token_ids.repeat_interleave(indices.shape[1]),
        indices.clamp_min(0).flatten().int(),
        w1.bfloat16().permute(1, 2, 0),
        None,
        w2.bfloat16().permute(1, 2, 0),
        None,
        w1.shape[0],
        None,
        ActivationType.SWIGLU,
        concat_layout=not interleaved,
        use_fp8=True,
    )[0]


def run(
    compiled,
    uncached=False,
    qwen=False,
    untuned=False,
    interleaved=False,
    standalone=False,
):
    import sonicmoe.functional as functional

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "quack": version("quack-kernels"),
                "cutlass": version("nvidia-cutlass-dsl"),
                "gpu": torch.cuda.get_device_name(),
                "untuned": untuned,
            }
        ),
        flush=True,
    )
    if untuned:
        functional.gemm = partial(functional.gemm, tuned=False)
        functional.gemm_dgated = partial(functional.gemm_dgated, tuned=False)
    if uncached:
        import sonicmoe.functional as functional

        functional._to_fp8_e4m3_weight = lambda x, perm: functional._to_fp8_e4m3(
            x.permute(*perm).contiguous()
        )
    tokens, hidden, inner = (4096, 2048, 768) if qwen else (256, 256, 256)
    torch.manual_seed(53)
    fn = partial(operation, interleaved=interleaved)
    if standalone:
        from mono_ep.sonic import sonic_fp8_experts

        def fn(x, scores, indices, w1, w2):
            return sonic_fp8_experts(x, indices, scores, w1, w2)

    if compiled:
        fn = torch.compile(fn, fullgraph=True)
    for dtype in (torch.bfloat16, torch.float32):
        w1 = (torch.randn(8, 2 * inner, hidden, device="cuda") / hidden**0.5).to(dtype)
        w2 = (torch.randn(8, hidden, inner, device="cuda") / inner**0.5).to(dtype)
        w1.requires_grad_()
        w2.requires_grad_()
        optimizer = torch.optim.AdamW([w1, w2], lr=0.01)
        for step in range(3):
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
            scores, indices = torch.randn(tokens, 8, device="cuda").softmax(-1).topk(4)
            if step == 1:
                indices[:, 1::2] = -1
            x.requires_grad_()
            scores.requires_grad_()
            inputs = (x, scores, w1, w2)
            dy = torch.randn_like(x)
            expected = reference(x, scores, indices, w1.bfloat16(), w2.bfloat16())
            expected_grads = torch.autograd.grad(expected, inputs, dy)
            actual = fn(x, scores, indices, w1, w2)
            grads = torch.autograd.grad(actual, inputs, dy)
            errors = {
                name: (
                    (got.float() - want.float()).norm()
                    / want.float().norm().clamp_min(1e-8)
                ).item()
                for name, got, want in zip(
                    ("y", "dx", "ds", "dw1", "dw2"),
                    (actual, *grads),
                    (expected, *expected_grads),
                    strict=True,
                )
            }
            print(
                json.dumps(
                    {
                        "compiled": compiled,
                        "dtype": str(dtype),
                        "step": step,
                        "relative_l2": errors,
                    }
                ),
                flush=True,
            )
            assert max(errors.values()) < 0.15, errors
            assert torch.count_nonzero(grads[1][indices < 0]) == 0
            optimizer.zero_grad(set_to_none=True)
            w1.grad, w2.grad = grads[2:]
            optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--uncached", action="store_true")
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--untuned", action="store_true")
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--standalone", action="store_true")
    args = parser.parse_args()
    run(
        args.compiled,
        args.uncached,
        args.qwen,
        args.untuned,
        args.interleaved,
        args.standalone,
    )
