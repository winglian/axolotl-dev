# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""SwiGLU derivatives from recomputed gate/up accumulators without saved z."""

import torch


@torch.library.custom_op("mono_ep::grouped_recompute_swiglu", mutates_args=())
def grouped_recompute_swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_ids: torch.Tensor,
    dh: torch.Tensor,
    scores: torch.Tensor,
    group_ends: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute BF16 gate/up and emit weighted h, concatenated dz and ds."""
    import deep_gemm

    if torch.cuda.get_device_capability(x.device)[0] != 10:
        raise ValueError("Recomputing SwiGLU epilogue requires SM100")
    rows, inner = dh.shape
    if w1.shape[1] != 2 * inner or x.shape[0] != rows:
        raise ValueError("Incompatible recompute projection geometry")
    gate, up = w1.bfloat16().chunk(2, dim=1)
    interleaved = torch.stack((gate, up), dim=2).flatten(1, 2)
    h = dh.new_zeros((rows, inner))
    dz = dh.new_zeros((rows, 2 * inner))
    partial = scores.new_zeros((rows, inner // 4))
    if rows:
        epilogue = deep_gemm.epilogue.SwiGLURecomputeBackward(
            dh, scores, h, dz, partial
        )
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            x,
            interleaved,
            dz,
            expert_ids if group_ends is None else group_ends,
            use_psum_layout=group_ends is not None,
            epilogue=epilogue,
        )
    return h, dz, partial.sum(-1)


@grouped_recompute_swiglu.register_fake
def _grouped_recompute_swiglu_fake(x, w1, expert_ids, dh, scores, group_ends=None):
    return (
        torch.empty_like(dh),
        dh.new_empty((dh.shape[0], w1.shape[1])),
        torch.empty_like(scores),
    )
