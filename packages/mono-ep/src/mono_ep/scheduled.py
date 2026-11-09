# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Expert compute composed with GPU-driven chunk readiness and peer combine."""

import torch
import triton
import triton.language as tl

from .activation import swiglu_fp8
from .chunk import chunk_fp8_out
from .combine import (
    grouped_fp8_scatter,
    peer_backward_scatter_publish,
    peer_scatter_add_publish,
)
from .ops import grouped_fp8
from .quantization import quantize_fp8


def fp8_expert_chunk_combine(
    x,
    sx,
    w1,
    s1,
    w2,
    s2,
    ready,
    scores,
    z,
    expert_ids,
    token_ids,
    accumulation,
    complete,
    producer,
):
    """Consume one ready routed chunk and combine its FP8 SwiGLU expert output.

    x/sx and first-projection weights use E4M3 with packed UE8M0 scales.
    Second-projection weights are E4M3 with per-row 1x32 scales. z is a
    caller-owned BF16 gate/up buffer retained for explicit backward. Scores
    must be FP32 and visible before ready is published. token_ids maps expert
    rows to destination tokens, with -1 for padding; expert_ids selects the
    second-projection expert for each aligned row tile.

    Each invocation owns one completion row in the peer FP32 accumulator.
    Stream ordering, peer mappings, buffer lifetimes and SM reservations are
    caller-owned. This is an explicit forward primitive without autograd;
    compile this function to capture its local tensor/kernel sequence.
    """
    chunk_fp8_out(x, sx, w1, s1, z, ready, 0)
    qh, sh = swiglu_fp8(z, scores)
    output = grouped_fp8(qh, sh, w2, s2, expert_ids, 32, 1)
    peer_scatter_add_publish(output, token_ids, accumulation, complete, producer)


@triton.jit
def _decode_packed(
    X,
    SF,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // K, offsets % K
    packed = tl.load(SF + row * S0 + (col // 128) * S1, offsets < N, 0)
    exponent = (packed >> (((col // 32) % 4) * 8)) & 255
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    values = tl.load(X + offsets, offsets < N, 0.0).to(tl.float32)
    tl.store(OUT + offsets, values * scale, offsets < N)


def _dequantize_packed(x, scales):
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    _decode_packed[(triton.cdiv(x.numel(), 1024),)](
        x,
        scales,
        out,
        x.numel(),
        x.shape[1],
        *scales.stride(),
        1024,
    )
    return out


def _backward_shapes(dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z, experts=1):
    import torch

    from .chunk import _check_fp8_metadata

    rows, hidden = dy.shape
    if z.ndim != 2 or z.shape[0] != rows or z.shape[1] % 16:
        raise ValueError("Expected aligned gate/up activations for one full chunk")
    inner = z.shape[1] // 2
    if (
        x.shape != dy.shape
        or w1t.shape != (experts, hidden, 2 * inner)
        or w2t.shape != (experts, inner, hidden)
    ):
        raise ValueError(
            "Queued backward weight shapes must match the expected expert count"
        )
    if (
        ready.shape != (experts, 4)
        or scores.shape != (rows,)
        or scores.dtype != torch.float32
    ):
        raise ValueError(
            "Expected one readiness descriptor per expert and FP32 per-row scores"
        )
    if z.dtype != torch.bfloat16 or z.device != dy.device or scores.device != dy.device:
        raise ValueError("Saved activations/scores must be on the gradient device")
    h = z.new_empty((rows, inner))
    _check_fp8_metadata(dy, sy, w2t, s2t, h, ready, 0, 32)
    _check_fp8_metadata(x, sx, w2t, s2t, h, ready, 0, 32)
    if (
        w1t.dtype != torch.float8_e4m3fn
        or w1t.device != dy.device
        or s1t.device != dy.device
    ):
        raise ValueError(
            "First-projection transpose must use E4M3 on the gradient device"
        )
    return rows, hidden, inner


@torch.library.custom_op("mono_ep::fp8_expert_chunk_backward", mutates_args=())
def fp8_expert_chunk_backward(
    dy: torch.Tensor,
    sy: torch.Tensor,
    x: torch.Tensor,
    sx: torch.Tensor,
    w1t: torch.Tensor,
    s1t: torch.Tensor,
    w2t: torch.Tensor,
    s2t: torch.Tensor,
    ready: torch.Tensor,
    scores: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit MXFP8 derivatives for one fully dispatched expert chunk.

    ready must describe [expert=0, start=0, count=rows, ready_flag]. Incoming
    dy/scales are published together; x/scales and z/scores are retained from
    forward. Weight transposes are independently quantized from master weights.
    Returns dx, ds, dw1 and dw2; weight gradients are FP32. Saved quantized
    inputs are dequantized then requantized by column for weight gradients.
    This is an explicit FP8 derivative rule, not differentiation of rounding.

    The opaque boundary preserves stream order: no dequantization of incoming
    gradients may execute before the queued GEMM has acquired readiness.
    """
    from .chunk import chunk_fp8_swiglu_backward_out
    from .ops import weight_grad_fp8

    rows, hidden, inner = _backward_shapes(
        dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z
    )
    h = z.new_empty((rows, inner))
    dz = torch.empty_like(z)
    partial = scores.new_empty((rows, inner // 8))
    chunk_fp8_swiglu_backward_out(dy, sy, w2t, s2t, h, ready, 0, z, scores, dz, partial)
    decoded_dy = _dequantize_packed(dy, sy)
    decoded_x = _dequantize_packed(x, sx)
    qdz, sdz = quantize_fp8(dz)
    expert_ids = torch.zeros(rows, device=dy.device, dtype=torch.int32)
    dx = grouped_fp8(qdz, sdz, w1t, s1t, expert_ids, 32, 1)
    sizes = torch.full((1,), rows, device=dy.device, dtype=torch.int32)
    dw1 = weight_grad_fp8(dz, decoded_x, sizes, [rows], 1, False, "mxfp8")
    dw2 = weight_grad_fp8(decoded_dy, h, sizes, [rows], 1, False, "mxfp8")
    return dx, partial.sum(-1), dw1, dw2


@fp8_expert_chunk_backward.register_fake
def _fp8_expert_chunk_backward_fake(
    dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z
):
    rows, hidden, inner = _backward_shapes(
        dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z
    )
    return (
        z.new_empty((rows, hidden)),
        scores.new_empty((rows,)),
        scores.new_empty((1, 2 * inner, hidden)),
        scores.new_empty((1, hidden, inner)),
    )


def fp8_expert_chunk_backward_combine(
    dy,
    sy,
    x,
    sx,
    w1t,
    s1t,
    w2t,
    s2t,
    ready,
    scores,
    z,
    token_ids,
    score_ids,
    dx_accumulation,
    dx_complete,
    ds_accumulation,
    ds_complete,
    producer,
):
    """Return local weight gradients while combining dx/ds into source peers.

    score_ids maps expert rows to flattened source routing-score slots;
    ds_accumulation has shape [number_of_score_slots, 1]. Each destination
    uses separate producer completion flags for dx and ds. The caller sums
    returned FP32 weight gradients across source chunks before updating weights.
    """
    dx, ds, dw1, dw2 = fp8_expert_chunk_backward(
        dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z
    )
    peer_backward_scatter_publish(
        dx,
        token_ids,
        ds,
        score_ids,
        dx_accumulation,
        ds_accumulation,
        dx_complete,
        ds_complete,
        producer,
    )
    return dw1, dw2


def _packed_backward_shapes(
    dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z, expert_ids, ends
):
    experts = w1t.shape[0]
    if experts < 1:
        raise ValueError("Packed backward requires at least one expert")
    rows, hidden, inner = _backward_shapes(
        dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z, experts
    )
    if (
        expert_ids.shape != (rows,)
        or ends.shape != (experts,)
        or any(
            t.dtype != torch.int32 or t.device != dy.device or not t.is_contiguous()
            for t in (expert_ids, ends)
        )
    ):
        raise ValueError("Expected packed INT32 expert IDs and unpadded expert ends")
    return rows, hidden, inner, experts


@torch.library.custom_op("mono_ep::fp8_packed_backward", mutates_args=())
def fp8_packed_backward(
    dy: torch.Tensor,
    sy: torch.Tensor,
    x: torch.Tensor,
    sx: torch.Tensor,
    w1t: torch.Tensor,
    s1t: torch.Tensor,
    w2t: torch.Tensor,
    s2t: torch.Tensor,
    ready: torch.Tensor,
    scores: torch.Tensor,
    z: torch.Tensor,
    expert_ids: torch.Tensor,
    ends: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit derivatives over one compact arena containing multiple experts.

    Descriptors contain expert/start/padded-count; ends contains the unpadded
    ends from the same 128-aligned DeviceExpertLayout. Incoming gradients/scales
    must be published before readiness; forward operands and metadata are retained.
    Grouped dH/SwiGLU acquires readiness inside the GEMM before reading payloads.
    Grouped dx and weight-gradient GEMMs follow; unused rows return zeros.
    The opaque boundary prevents reads of incoming operands before acquisition.
    """
    return _packed_backward_impl(
        dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z, expert_ids, ends
    )


def _packed_backward_impl(
    dy,
    sy,
    x,
    sx,
    w1t,
    s1t,
    w2t,
    s2t,
    ready,
    scores,
    z,
    expert_ids,
    ends,
    scatter=None,
    incremental=False,
):
    import deep_gemm

    from .ops import weight_grad_fp8

    rows, hidden, inner, experts = _packed_backward_shapes(
        dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z, expert_ids, ends
    )
    h = z.new_zeros((rows, inner))
    dz = torch.zeros_like(z)
    partial = scores.new_zeros((rows, inner // 8))
    epilogue = deep_gemm.epilogue.SwiGLUBackward(z, scores, dz, partial)
    deep_gemm.fp8_ready_grouped_gemm_nt(
        (dy, sy),
        (w2t, s2t),
        h,
        expert_ids,
        ready,
        epilogue=epilogue,
        incremental=incremental,
    )
    qdz, sdz = quantize_fp8(dz)
    ds = partial.sum(-1)
    if scatter is None:
        dx = grouped_fp8(qdz, sdz, w1t, s1t, expert_ids, 32, 1)
    else:
        tokens, slots, dx_dest, ds_dest, dx_queue, ds_queue, task, assign = scatter
        grouped_fp8_scatter(
            qdz,
            sdz,
            w1t,
            s1t,
            expert_ids,
            slots if assign else tokens,
            dx_dest,
            dx_queue,
            task,
            assign,
            ends[-1:],
        )
        peer_scatter_add_publish(ds[:, None], slots, ds_dest, ds_queue, task)
        dx = None
    dw1 = weight_grad_fp8(dz, x, ends, None, experts, True, "mxfp8", x_scales=sx)
    dw2 = weight_grad_fp8(dy, h, ends, None, experts, True, "mxfp8", dy_scales=sy)
    return dx, ds, dw1, dw2


@fp8_packed_backward.register_fake
def _packed_backward_fake(
    dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z, expert_ids, ends
):
    rows, hidden, inner, experts = _packed_backward_shapes(
        dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z, expert_ids, ends
    )
    return (
        z.new_empty((rows, hidden)),
        scores.new_empty((rows,)),
        scores.new_empty((experts, 2 * inner, hidden)),
        scores.new_empty((experts, hidden, inner)),
    )


@torch.library.custom_op(
    "mono_ep::fp8_packed_backward_scatter",
    mutates_args=("dx_dest", "ds_dest", "dx_queue", "ds_queue"),
)
def fp8_packed_backward_scatter(
    dy: torch.Tensor,
    sy: torch.Tensor,
    x: torch.Tensor,
    sx: torch.Tensor,
    w1t: torch.Tensor,
    s1t: torch.Tensor,
    w2t: torch.Tensor,
    s2t: torch.Tensor,
    ready: torch.Tensor,
    scores: torch.Tensor,
    z: torch.Tensor,
    expert_ids: torch.Tensor,
    ends: torch.Tensor,
    tokens: torch.Tensor,
    slots: torch.Tensor,
    dx_dest: torch.Tensor,
    ds_dest: torch.Tensor,
    dx_queue: torch.Tensor,
    ds_queue: torch.Tensor,
    task: int,
    assign: bool = False,
    incremental: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish input gradients from the GEMM epilogue and return weight gradients.

    Input and score completion flags are independent; each consumer must acquire
    its own flag. Weight gradients finish before this operation returns.
    """
    _, _, dw1, dw2 = _packed_backward_impl(
        dy,
        sy,
        x,
        sx,
        w1t,
        s1t,
        w2t,
        s2t,
        ready,
        scores,
        z,
        expert_ids,
        ends,
        (tokens, slots, dx_dest, ds_dest, dx_queue, ds_queue, task, assign),
        incremental,
    )
    return dw1, dw2


@fp8_packed_backward_scatter.register_fake
def _packed_backward_scatter_fake(
    dy,
    sy,
    x,
    sx,
    w1t,
    s1t,
    w2t,
    s2t,
    ready,
    scores,
    z,
    expert_ids,
    ends,
    tokens,
    slots,
    dx_dest,
    ds_dest,
    dx_queue,
    ds_queue,
    task,
    assign=False,
    incremental=False,
):
    from .combine import _check_backward_scatter

    dx, ds, dw1, dw2 = _packed_backward_fake(
        dy, sy, x, sx, w1t, s1t, w2t, s2t, ready, scores, z, expert_ids, ends
    )
    _check_backward_scatter(
        dx, tokens, ds, slots, dx_dest, ds_dest, dx_queue, ds_queue, task, assign
    )
    return dw1, dw2
