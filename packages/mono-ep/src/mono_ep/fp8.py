# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""FP8 grouped expert GEMMs with explicit architecture-specific recipes."""

from .ops import grouped_fp8
from .quantization import quantize_fp8


def expert_gemm_fp8(x, w, expert_ids, *, format, group_ends=None):
    """Quantize and multiply packed rows by [expert, output, input] weights.

    ``mxfp8`` uses 1x32 scaling on SM100; ``blockwise`` uses 1x128
    activations and 128x128 weights on SM90. This primitive has no autograd
    registration: training must supply explicit quantized derivatives.
    """
    if format == "mxfp8":
        cols, rows = 32, 1
    elif format == "blockwise":
        cols, rows = 128, 128
    else:
        raise ValueError("choose mxfp8 or blockwise")
    # Grouped GEMM selects experts by tile; padding rows must contain zeros.
    if x.is_cuda and x.is_contiguous():
        from .token_quantization import quantize_tokens

        qx, sx = quantize_tokens(x, cols, expert_ids)
    else:
        x = x.masked_fill((expert_ids < 0).unsqueeze(-1), 0)
        qx, sx = quantize_fp8(x, block_cols=cols)
    qw, sw = quantize_fp8(w, block_rows=rows, block_cols=cols)
    return grouped_fp8(qx, sx, qw, sw, expert_ids, cols, rows, group_ends)


def expert_forward_fp8(
    x, scores, w1, w2, layout, *, format, bf16_forward=False, retain_activations=False
):
    from .activation import swiglu
    from .ops import grouped_bf16

    packed = layout.pack(x)
    z = (
        grouped_bf16(packed, w1.bfloat16(), layout.expert_ids)
        if bf16_forward
        else expert_gemm_fp8(packed, w1, layout.expert_ids, format=format)
    )
    h = swiglu(z, layout.scores(scores))
    out = (
        grouped_bf16(h, w2.bfloat16(), layout.expert_ids)
        if bf16_forward
        else expert_gemm_fp8(h, w2, layout.expert_ids, format=format)
    )
    y = layout.combine(out, x.shape[0])
    return (y, (packed, z, h)) if retain_activations else y


def expert_backward_fp8(
    dy,
    x,
    scores,
    w1,
    w2,
    layout,
    *,
    format,
    bf16_forward=False,
    activations=None,
    fuse_activation=False,
    output_exchange=None,
):
    """Explicit quantized derivatives; this is not differentiation through rounding.

    Forward intermediates are reused when supplied, otherwise recomputed.
    All input- and weight-gradient GEMMs use FP8, with FP32 weight-gradient output.
    """
    pdy = layout.pack(dy)
    px = layout.pack(x) if activations is None else activations[0]

    def combine_and_exchange(dx, ds):
        dx, ds = layout.combine(dx, x.shape[0]), layout.restore_scores(ds, scores)
        return (dx, ds) if output_exchange is None else output_exchange(dx, ds)

    return expert_backward_fp8_packed(
        pdy,
        px,
        layout.scores(scores),
        w1,
        w2,
        layout,
        format=format,
        bf16_forward=bf16_forward,
        activations=activations,
        fuse_activation=fuse_activation,
        output_exchange=combine_and_exchange,
    )


def expert_backward_fp8_packed(
    pdy,
    px,
    packed_scores,
    w1,
    w2,
    layout,
    *,
    format,
    bf16_forward=False,
    activations=None,
    fuse_activation=False,
    output_exchange=None,
):
    """Consume packed rows with quantized GEMMs and optional early gradient return."""
    from .activation import swiglu
    from .ops import grouped_bf16, grouped_mxfp8_swiglu_backward, weight_grad_fp8

    group_ends = layout.starts + layout.padded_sizes if layout.use_psum_layout else None
    if activations is None:
        z = (
            grouped_bf16(px, w1.bfloat16(), layout.expert_ids)
            if bf16_forward
            else expert_gemm_fp8(
                px, w1, layout.expert_ids, format=format, group_ends=group_ends
            )
        )
        cached_h = None
    else:
        px, z, cached_h = activations
    if fuse_activation and format == "mxfp8":
        h, dz, ds = grouped_mxfp8_swiglu_backward(
            pdy, w2, layout.expert_ids, z, packed_scores
        )
    else:
        dh = expert_gemm_fp8(
            pdy,
            w2.transpose(1, 2),
            layout.expert_ids,
            format=format,
            group_ends=group_ends,
        )
        h, dz, ds = swiglu(z, packed_scores, dh, cached_h=cached_h)
    dx = expert_gemm_fp8(
        dz, w1.transpose(1, 2), layout.expert_ids, format=format, group_ends=group_ends
    )
    if output_exchange is not None:
        dx, ds = output_exchange(dx, ds)
    if format == "blockwise" and layout.use_psum_layout:
        from .hopper_wgrad import weight_grad_fp8_hopper

        dw2 = weight_grad_fp8_hopper(pdy, h, group_ends, layout.k_sizes)
        dw1 = weight_grad_fp8_hopper(dz, px, group_ends, layout.k_sizes)
    else:
        dw2 = weight_grad_fp8(
            pdy,
            h,
            layout.k_sizes,
            layout.sizes,
            w2.shape[0],
            layout.use_psum_layout,
            format,
        )
        dw1 = weight_grad_fp8(
            dz,
            px,
            layout.k_sizes,
            layout.sizes,
            w1.shape[0],
            layout.use_psum_layout,
            format,
        )
    return dx, ds, dw1, dw2
