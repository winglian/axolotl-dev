# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Functional compiler boundaries for DeepGEMM's output-buffer interfaces."""

import torch


@torch.library.custom_op("mono_ep::grouped_fp8", mutates_args=())
def grouped_fp8(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    block_cols: int,
    weight_block_rows: int,
    group_ends: torch.Tensor | None = None,
) -> torch.Tensor:
    import deep_gemm

    out = torch.zeros((x.shape[0], w.shape[1]), device=x.device, dtype=torch.bfloat16)
    if x.shape[0]:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (x, x_scale),
            (w, w_scale),
            out,
            expert_ids if group_ends is None else group_ends,
            use_psum_layout=group_ends is not None,
            recipe_a=(1, block_cols),
            recipe_b=(weight_block_rows, block_cols),
        )
    return out


@grouped_fp8.register_fake
def _grouped_fp8_fake(
    x, x_scale, w, w_scale, expert_ids, block_cols, weight_block_rows, group_ends=None
):
    return x.new_empty((x.shape[0], w.shape[1]), dtype=torch.bfloat16)


@torch.library.custom_op("mono_ep::grouped_bf16", mutates_args=())
def grouped_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    expert_ids: torch.Tensor,
    group_ends: torch.Tensor | None = None,
) -> torch.Tensor:
    import deep_gemm

    out = x.new_zeros((x.shape[0], w.shape[1]))
    if x.shape[0]:
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            x,
            w,
            out,
            expert_ids if group_ends is None else group_ends,
            use_psum_layout=group_ends is not None,
        )
    return out


@grouped_bf16.register_fake
def _grouped_bf16_fake(x, w, expert_ids, group_ends=None):
    return x.new_empty((x.shape[0], w.shape[1]))


@torch.library.custom_op("mono_ep::grouped_swiglu_backward", mutates_args=())
def grouped_swiglu_backward(
    dy: torch.Tensor,
    w2: torch.Tensor,
    expert_ids: torch.Tensor,
    z: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import deep_gemm

    from .activation import swiglu

    wt = w2.transpose(1, 2).contiguous()
    if torch.cuda.get_device_capability(dy.device)[0] != 10:
        return swiglu(z, scores, grouped_bf16(dy, wt, expert_ids))
    rows, inner = z.shape[0], z.shape[1] // 2
    h = z.new_zeros((rows, inner))
    dz = torch.zeros_like(z)
    partial = scores.new_zeros((rows, inner // 8))
    if rows:
        epilogue = deep_gemm.epilogue.SwiGLUBackward(z, scores, dz, partial)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            dy, wt, h, expert_ids, epilogue=epilogue
        )
    return h, dz, partial.sum(-1)


@grouped_swiglu_backward.register_fake
def _grouped_swiglu_backward_fake(dy, w2, expert_ids, z, scores):
    return (
        z.new_empty((z.shape[0], z.shape[1] // 2)),
        torch.empty_like(z),
        torch.empty_like(scores),
    )


@torch.library.custom_op("mono_ep::grouped_mxfp8_swiglu_backward", mutates_args=())
def grouped_mxfp8_swiglu_backward(
    dy: torch.Tensor,
    w2: torch.Tensor,
    expert_ids: torch.Tensor,
    z: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import deep_gemm

    from .quantization import quantize_fp8
    from .token_quantization import quantize_tokens

    rows, inner = z.shape[0], z.shape[1] // 2
    h = z.new_zeros((rows, inner))
    dz = torch.zeros_like(z)
    partial = scores.new_zeros((rows, inner // 8))
    if rows:
        qdy, sdy = quantize_tokens(dy.contiguous(), 32, expert_ids)
        qw, sw = quantize_fp8(w2.transpose(1, 2), block_rows=1, block_cols=32)
        epilogue = deep_gemm.epilogue.SwiGLUBackward(z, scores, dz, partial)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (qdy, sdy),
            (qw, sw),
            h,
            expert_ids,
            recipe_a=(1, 32),
            recipe_b=(1, 32),
            epilogue=epilogue,
        )
    return h, dz, partial.sum(-1)


@grouped_mxfp8_swiglu_backward.register_fake
def _grouped_mxfp8_swiglu_backward_fake(dy, w2, expert_ids, z, scores):
    return (
        z.new_empty((z.shape[0], z.shape[1] // 2)),
        torch.empty_like(z),
        torch.empty_like(scores),
    )


@torch.library.custom_op("mono_ep::weight_grad_bf16", mutates_args=())
def weight_grad_bf16(
    dy: torch.Tensor,
    x: torch.Tensor,
    k_sizes: torch.Tensor,
    sizes: list[int] | None,
    experts: int,
    use_psum_layout: bool,
) -> torch.Tensor:
    import deep_gemm

    out = torch.zeros(
        (experts, dy.shape[1], x.shape[1]), device=x.device, dtype=torch.float32
    )
    if x.shape[0]:
        deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
            dy,
            x,
            out,
            sizes,
            k_sizes,
            out,
            use_psum_layout=use_psum_layout,
        )
    return out


@weight_grad_bf16.register_fake
def _weight_grad_bf16_fake(dy, x, k_sizes, sizes, experts, use_psum_layout):
    return x.new_empty((experts, dy.shape[1], x.shape[1]), dtype=torch.float32)


@torch.library.custom_op("mono_ep::weight_grad_bf16_gpu_offsets", mutates_args=())
def weight_grad_bf16_gpu_offsets(
    dy: torch.Tensor,
    x: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    expected_k: int,
) -> torch.Tensor:
    import deep_gemm

    out = torch.zeros(
        (starts.numel(), dy.shape[1], x.shape[1]),
        device=x.device,
        dtype=torch.float32,
    )
    if x.shape[0]:
        deep_gemm._C.k_grouped_bf16_gemm_tn_gpu_offsets(
            dy, x, out, starts, ends, expected_k, out
        )
    return out


@weight_grad_bf16_gpu_offsets.register_fake
def _weight_grad_bf16_gpu_offsets_fake(dy, x, starts, ends, expected_k):
    return x.new_empty((starts.numel(), dy.shape[1], x.shape[1]), dtype=torch.float32)


@torch.library.custom_op("mono_ep::weight_grad_fp8", mutates_args=())
def weight_grad_fp8(
    dy: torch.Tensor,
    x: torch.Tensor,
    k_sizes: torch.Tensor,
    sizes: list[int] | None,
    experts: int,
    use_psum_layout: bool,
    format: str,
    dy_scales: torch.Tensor | None = None,
    x_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    import deep_gemm

    if format not in ("mxfp8", "blockwise"):
        raise ValueError("choose mxfp8 or blockwise")
    if format == "blockwise" and (sizes is None or use_psum_layout):
        raise ValueError("Hopper FP8 weight gradients require padded host group sizes")
    if format != "mxfp8" and (dy_scales is not None or x_scales is not None):
        raise ValueError("Packed operand scales require MXFP8")
    out = torch.zeros(
        (experts, dy.shape[1], x.shape[1]), device=x.device, dtype=torch.float32
    )
    if not x.shape[0]:
        return out
    if format == "mxfp8":
        from .column_quantization import quantize_columns, quantize_packed_columns

        row_end = k_sizes[-1:] if use_psum_layout else None
        deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
            quantize_columns(dy, row_end)
            if dy_scales is None
            else quantize_packed_columns(dy, dy_scales, row_end),
            quantize_columns(x, row_end)
            if x_scales is None
            else quantize_packed_columns(x, x_scales, row_end),
            out,
            sizes,
            k_sizes,
            out,
            recipe=(1, 1, 32),
            use_psum_layout=use_psum_layout,
        )
    else:
        from .wgrad_quantization import quantize_transposed_groups

        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            quantize_transposed_groups(dy, k_sizes),
            quantize_transposed_groups(x, k_sizes),
            out,
            sizes,
            k_sizes,
            out,
            recipe=(1, 1, 128),
        )
    return out


@weight_grad_fp8.register_fake
def _weight_grad_fp8_fake(
    dy,
    x,
    k_sizes,
    sizes,
    experts,
    use_psum_layout,
    format,
    dy_scales=None,
    x_scales=None,
):
    return x.new_empty((experts, dy.shape[1], x.shape[1]), dtype=torch.float32)
