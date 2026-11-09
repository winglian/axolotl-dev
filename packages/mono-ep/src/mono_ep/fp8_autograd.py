# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Autograd/compiler boundary for local FP8 expert computation."""

import torch

from .fp8 import expert_backward_fp8, expert_forward_fp8
from .grouped import ExpertLayout
from .packing import DeviceExpertLayout


def _layout(indices, experts, format):
    layout_type = DeviceExpertLayout if format == "mxfp8" else ExpertLayout
    return layout_type.build(indices, experts, 128)


@torch.library.custom_op("mono_ep::fp8_experts", mutates_args=())
def fp8_experts(
    x: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    format: str = "blockwise",
    bf16_forward: bool = False,
) -> torch.Tensor:
    """Local expert layer with approximate FP8 derivatives and FP32 master support.

    IDs are local; -1 marks unrouted slots. Hopper's host layout construction
    is inside the compiler boundary, so full-graph capture does not remove its
    synchronization. No distributed communication is performed here.
    """
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("FP8 experts require contiguous BF16 activations")
    if scores.dtype != torch.float32 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("routing requires FP32 scores and integer expert IDs")
    if any(w.dtype not in (torch.bfloat16, torch.float32) for w in (w1, w2)):
        raise ValueError("expert master weights must be BF16 or FP32")
    layout = _layout(indices, w1.shape[0], format)
    return expert_forward_fp8(
        x, scores, w1, w2, layout, format=format, bf16_forward=bf16_forward
    )


@fp8_experts.register_fake
def _forward_fake(x, indices, scores, w1, w2, format="blockwise", bf16_forward=False):
    return torch.empty_like(x)


@torch.library.custom_op("mono_ep::fp8_experts_backward", mutates_args=())
def _backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    format: str,
    bf16_forward: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = _layout(indices, w1.shape[0], format)
    dx, ds, dw1, dw2 = expert_backward_fp8(
        dy, x, scores, w1, w2, layout, format=format, bf16_forward=bf16_forward
    )
    return dx, ds, dw1.to(w1.dtype), dw2.to(w2.dtype)


@_backward.register_fake
def _backward_fake(dy, x, indices, scores, w1, w2, format, bf16_forward):
    return tuple(torch.empty_like(t) for t in (x, scores, w1, w2))


def _setup_context(ctx, inputs, output):
    x, indices, scores, w1, w2, format, bf16_forward = inputs
    ctx.save_for_backward(x, indices, scores, w1, w2)
    ctx.format = format
    ctx.bf16_forward = bf16_forward


def _autograd_backward(ctx, dy):
    dx, ds, dw1, dw2 = _backward(
        dy.contiguous(), *ctx.saved_tensors, ctx.format, ctx.bf16_forward
    )
    return dx, None, ds, dw1, dw2, None, None


fp8_experts.register_autograd(_autograd_backward, setup_context=_setup_context)


class FP8Experts:
    """Local FP8 compute with explicit derivatives for a communication scheduler."""

    def __init__(self, format="blockwise", *, bf16_forward=False, compile_local=False):
        if format not in ("blockwise", "mxfp8"):
            raise ValueError("choose blockwise or mxfp8")
        self.format = format
        self.bf16_forward = bf16_forward
        self._forward = expert_forward_fp8
        self._backward = expert_backward_fp8
        if compile_local:
            self._forward = torch.compile(self._forward, fullgraph=True, dynamic=True)
            self._backward = torch.compile(self._backward, fullgraph=True, dynamic=True)

    def __call__(self, x, indices, scores, w1, w2):
        return fp8_experts(x, indices, scores, w1, w2, self.format, self.bf16_forward)

    def forward_with_activations(self, x, indices, scores, w1, w2):
        """Return output and tensors to retain in the scheduler autograd context."""
        layout = _layout(indices, w1.shape[0], self.format)
        return self._forward(
            x,
            scores,
            w1,
            w2,
            layout,
            format=self.format,
            bf16_forward=self.bf16_forward,
            retain_activations=True,
        )

    def backward(self, dy, x, indices, scores, w1, w2, *, activations=None):
        layout = _layout(indices, w1.shape[0], self.format)
        return self._backward(
            dy,
            x,
            scores,
            w1,
            w2,
            layout,
            format=self.format,
            bf16_forward=self.bf16_forward,
            activations=activations,
        )
