# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Hopper grouped FP8 weight gradients with GPU-resident padded offsets."""

import torch


@torch.library.custom_op("mono_ep::weight_grad_fp8_hopper_packed", mutates_args=())
def weight_grad_fp8_hopper_packed(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    sizes: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """Consume column-scaled expert operands produced by the persistent kernel."""
    import deep_gemm

    if sfa.ndim != 2 or sfb.ndim != 2 or sizes.ndim != 1:
        raise ValueError("expected matrix scales and one-dimensional group sizes")
    if capacity < 0 or capacity % 128:
        raise ValueError("packed row capacity must be nonnegative and 128-aligned")
    if sfa.shape[1] != capacity // 128 or sfb.shape[1] != capacity // 128:
        raise ValueError("scale capacity does not match packed operands")
    out = torch.zeros(
        (sizes.numel(), sfa.shape[0], sfb.shape[0]),
        dtype=torch.float32,
        device=a.device,
    )
    if capacity:
        deep_gemm._C.k_grouped_fp8_gemm_nt_gpu_sizes(
            a, sfa, b, sfb, out, sizes, capacity
        )
    return out


@weight_grad_fp8_hopper_packed.register_fake
def _weight_grad_fp8_hopper_packed_fake(a, sfa, b, sfb, sizes, capacity):
    return a.new_empty((sizes.numel(), sfa.shape[0], sfb.shape[0]), dtype=torch.float32)


@torch.library.custom_op("mono_ep::weight_grad_fp8_hopper", mutates_args=())
def weight_grad_fp8_hopper(
    dy: torch.Tensor,
    x: torch.Tensor,
    padded_ends: torch.Tensor,
    valid_ends: torch.Tensor | None = None,
) -> torch.Tensor:
    import deep_gemm

    from .wgrad_quantization import quantize_transposed_groups

    if dy.ndim != 2 or x.ndim != 2 or dy.shape[0] != x.shape[0]:
        raise ValueError("expected two matrices with equal row capacity")
    if dy.device != x.device or padded_ends.device != x.device:
        raise ValueError("operands and offsets must share a CUDA device")
    out = torch.zeros(
        (padded_ends.numel(), dy.shape[1], x.shape[1]),
        dtype=torch.float32,
        device=x.device,
    )
    if not x.shape[0]:
        return out
    a, sfa = quantize_transposed_groups(
        dy,
        padded_ends,
        use_psum_layout=True,
        scales_mn_major=True,
        valid_ends=valid_ends,
    )
    b, sfb = quantize_transposed_groups(
        x,
        padded_ends,
        use_psum_layout=True,
        scales_mn_major=True,
        valid_ends=valid_ends,
    )
    sizes = torch.diff(padded_ends, prepend=padded_ends.new_zeros(1))
    deep_gemm._C.k_grouped_fp8_gemm_nt_gpu_sizes(a, sfa, b, sfb, out, sizes, x.shape[0])
    return out


@weight_grad_fp8_hopper.register_fake
def _weight_grad_fp8_hopper_fake(dy, x, padded_ends, valid_ends=None):
    return x.new_empty(
        (padded_ends.numel(), dy.shape[1], x.shape[1]), dtype=torch.float32
    )
