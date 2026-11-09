# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Packed BF16 expert compute behind GPU readiness boundaries."""

import torch

from .activation import swiglu
from .chunk import (
    _check_swiglu_metadata,
    _ready_grouped_bf16_fake,
    ready_grouped_bf16_out,
    ready_grouped_bf16_swiglu_backward_out,
    ready_grouped_bf16_swiglu_forward_out,
)
from .ops import grouped_bf16, weight_grad_bf16


def bf16_packed_forward(x, w1, w2, ready, scores, expert_ids):
    """Consume published packed inputs, returning output and saved gate/up values."""
    z = x.new_zeros((x.shape[0], w1.shape[1]))
    ready_grouped_bf16_out(x, w1, z, ready, expert_ids)
    h = swiglu(z, scores)
    return grouped_bf16(h, w2, expert_ids), z


def _backward_shapes(dy, x, w1, w2, ready, scores, z, expert_ids, ends):
    rows, hidden = x.shape
    experts, inner = w2.shape[0], w2.shape[2]
    if (
        dy.shape != x.shape
        or w1.shape != (experts, 2 * inner, hidden)
        or w2.shape != (experts, hidden, inner)
        or ends.shape != (experts,)
        or ends.dtype != torch.int32
        or ends.device != x.device
        or not ends.is_contiguous()
        or x.dtype != torch.bfloat16
        or w1.dtype != torch.bfloat16
        or any(t.device != x.device or not t.is_contiguous() for t in (x, w1, w2))
    ):
        raise ValueError("Incompatible packed BF16 backward operands")
    h = x.new_empty((rows, inner))
    _ready_grouped_bf16_fake(dy, w2.transpose(1, 2), h, ready, expert_ids)
    _check_swiglu_metadata(
        h, z, scores, torch.empty_like(z), scores.new_empty((rows, inner // 8))
    )
    return rows, hidden, inner, experts


@torch.library.custom_op("mono_ep::bf16_packed_backward", mutates_args=())
def bf16_packed_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    ready: torch.Tensor,
    scores: torch.Tensor,
    z: torch.Tensor,
    expert_ids: torch.Tensor,
    ends: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed dX, routing-score gradients, and FP32 weight gradients.

    Incoming dY must be published before every ready flag. Other operands and
    unpadded expert ends are retained from forward. The opaque boundary keeps
    incoming-gradient reads behind acquisition, including under torch.compile.
    """
    return _backward_impl(dy, x, w1, w2, ready, scores, z, expert_ids, ends)


def _check_transposed_weights(w1, w2, w1t, w2t):
    for weight, transposed in ((w1, w1t), (w2, w2t)):
        if transposed is not None and (
            transposed.shape != (weight.shape[0], weight.shape[2], weight.shape[1])
            or transposed.dtype != torch.bfloat16
            or transposed.device != weight.device
            or not transposed.is_contiguous()
        ):
            raise ValueError("Expected contiguous BF16 transposed expert weights")


def _backward_impl(
    dy,
    x,
    w1,
    w2,
    ready,
    scores,
    z,
    expert_ids,
    ends,
    scatter=None,
    w1t=None,
    w2t=None,
    fused_scores=False,
    incremental=False,
):
    rows, _, inner, experts = _backward_shapes(
        dy, x, w1, w2, ready, scores, z, expert_ids, ends
    )
    _check_transposed_weights(w1, w2, w1t, w2t)
    w1t = w1.transpose(1, 2).contiguous() if w1t is None else w1t
    w2t = w2.transpose(1, 2).contiguous() if w2t is None else w2t
    h = x.new_empty((rows, inner))
    # The standalone dX GEMM computes inactive rows; peer scatter masks them.
    dz = torch.zeros_like(z) if scatter is None else torch.empty_like(z)
    allocate_partial = (
        scores.new_zeros if scatter is None or not fused_scores else scores.new_empty
    )
    partial = allocate_partial((rows, inner // 8))
    ready_grouped_bf16_swiglu_backward_out(
        dy,
        w2t,
        h,
        ready,
        expert_ids,
        z,
        scores,
        dz,
        partial,
        incremental,
    )
    if scatter is None:
        ds = partial.sum(-1)
        dx = grouped_bf16(dz, w1t, expert_ids)
    else:
        from .combine import (
            grouped_bf16_scatter,
            peer_reduce_scatter_publish,
            peer_scatter_add_publish,
        )

        tokens, slots, dx_acc, dx_flags, ds_acc, ds_flags, task, assign = scatter
        grouped_bf16_scatter(
            dz,
            w1t,
            expert_ids,
            slots if assign else tokens,
            dx_acc,
            dx_flags,
            task,
            assign,
            ends[-1:],
        )
        if fused_scores:
            peer_reduce_scatter_publish(partial, slots, ds_acc, ds_flags, task)
        else:
            peer_scatter_add_publish(
                partial.sum(-1, keepdim=True), slots, ds_acc, ds_flags, task
            )
        dx = ds = None
    dw1 = weight_grad_bf16(dz, x, ends, None, experts, True)
    dw2 = weight_grad_bf16(dy, h, ends, None, experts, True)
    return dx, ds, dw1, dw2


@bf16_packed_backward.register_fake
def _backward_fake(dy, x, w1, w2, ready, scores, z, expert_ids, ends):
    _backward_shapes(dy, x, w1, w2, ready, scores, z, expert_ids, ends)
    return (
        torch.empty_like(x),
        torch.empty_like(scores),
        w1.float().new_empty(w1.shape),
        w2.float().new_empty(w2.shape),
    )


def bf16_packed_forward_combine(
    x, w1, w2, ready, scores, expert_ids, token_ids, destination, complete, task
):
    """Consume packed peer inputs and publish weighted output to token owners."""
    from .combine import peer_scatter_add_publish

    output, z = bf16_packed_forward(x, w1, w2, ready, scores, expert_ids)
    peer_scatter_add_publish(output, token_ids, destination, complete, task)
    return z


def bf16_packed_backward_combine(
    dy,
    x,
    w1,
    w2,
    ready,
    scores,
    z,
    expert_ids,
    ends,
    token_ids,
    score_ids,
    dx_destination,
    ds_destination,
    dx_complete,
    ds_complete,
    task,
):
    """Publish packed dX/dScores to token owners and return local FP32 dWeights."""
    from .combine import peer_backward_scatter_publish

    dx, ds, dw1, dw2 = bf16_packed_backward(
        dy, x, w1, w2, ready, scores, z, expert_ids, ends
    )
    peer_backward_scatter_publish(
        dx,
        token_ids,
        ds,
        score_ids,
        dx_destination,
        ds_destination,
        dx_complete,
        ds_complete,
        task,
    )
    return dw1, dw2


def _peer_forward(
    x,
    _sx,
    w1,
    _s1,
    w2,
    _s2,
    ready,
    scores,
    z,
    ids,
    tokens,
    destination,
    complete,
    task,
    assign=False,
    incremental=False,
):
    from .combine import grouped_bf16_scatter

    ready_grouped_bf16_out(x, w1, z, ready, ids, incremental)
    h = swiglu(z, scores)
    grouped_bf16_scatter(
        h,
        w2,
        ids,
        tokens,
        destination,
        complete,
        task,
        assign,
        (ready[:, 1] + ready[:, 2]).max(),
    )


def _peer_forward_fused(
    x,
    _sx,
    w1,
    _s1,
    w2,
    _s2,
    ready,
    scores,
    z,
    ids,
    tokens,
    destination,
    complete,
    task,
    assign=False,
    initialize_h=False,
    incremental=False,
):
    from .combine import grouped_bf16_scatter

    h = x.new_empty((x.shape[0], w2.shape[2]))
    if initialize_h:
        h.zero_()
    ready_grouped_bf16_swiglu_forward_out(x, w1, z, h, scores, ready, ids, incremental)
    grouped_bf16_scatter(
        h,
        w2,
        ids,
        tokens,
        destination,
        complete,
        task,
        assign,
        (ready[:, 1] + ready[:, 2]).max(),
    )


def _peer_backward(
    dy,
    _sy,
    x,
    _sx,
    w1,
    _s1,
    w2,
    _s2,
    ready,
    scores,
    z,
    ids,
    ends,
    tokens,
    slots,
    dx_acc,
    dx_flags,
    ds_acc,
    ds_flags,
    task,
    assign=False,
    w1t=None,
    w2t=None,
    fused_scores=False,
    incremental=False,
):
    return bf16_packed_backward_scatter(
        dy,
        x,
        w1,
        w2,
        ready,
        scores,
        z,
        ids,
        ends,
        tokens,
        slots,
        dx_acc,
        dx_flags,
        ds_acc,
        ds_flags,
        task,
        assign,
        w1t,
        w2t,
        fused_scores,
        incremental,
    )


@torch.library.custom_op(
    "mono_ep::bf16_packed_backward_scatter",
    mutates_args=("dx_acc", "dx_flags", "ds_acc", "ds_flags"),
)
def bf16_packed_backward_scatter(
    dy: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    ready: torch.Tensor,
    scores: torch.Tensor,
    z: torch.Tensor,
    ids: torch.Tensor,
    ends: torch.Tensor,
    tokens: torch.Tensor,
    slots: torch.Tensor,
    dx_acc: torch.Tensor,
    dx_flags: torch.Tensor,
    ds_acc: torch.Tensor,
    ds_flags: torch.Tensor,
    task: int,
    assign: bool = False,
    w1t: torch.Tensor | None = None,
    w2t: torch.Tensor | None = None,
    fused_scores: bool = False,
    incremental: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Acquire BF16 dY and fuse dX GEMM with peer publication; return dWeights."""
    _, _, dw1, dw2 = _backward_impl(
        dy,
        x,
        w1,
        w2,
        ready,
        scores,
        z,
        ids,
        ends,
        (tokens, slots, dx_acc, dx_flags, ds_acc, ds_flags, task, assign),
        w1t,
        w2t,
        fused_scores,
        incremental,
    )
    return dw1, dw2


@bf16_packed_backward_scatter.register_fake
def _backward_scatter_fake(
    dy,
    x,
    w1,
    w2,
    ready,
    scores,
    z,
    ids,
    ends,
    tokens,
    slots,
    dx_acc,
    dx_flags,
    ds_acc,
    ds_flags,
    task,
    assign=False,
    w1t=None,
    w2t=None,
    fused_scores=False,
    incremental=False,
):
    _backward_shapes(dy, x, w1, w2, ready, scores, z, ids, ends)
    _check_transposed_weights(w1, w2, w1t, w2t)
    return torch.empty_like(w1, dtype=torch.float32), torch.empty_like(
        w2, dtype=torch.float32
    )
