# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""GEMM epilogue stores to unique source token/top-k slots in peer mappings."""

import torch


@torch.library.custom_op("mono_ep::peer_gemm_out", mutates_args=("outputs",))
def peer_gemm_out(
    x: torch.Tensor,
    weights: torch.Tensor,
    group_ends: torch.Tensor,
    assignments: torch.Tensor,
    outputs: list[torch.Tensor],
    pointers: torch.Tensor,
) -> torch.Tensor:
    """Return completion metadata after assigning BF16 GEMM values to peer slots.

    Pointers must match outputs in rank order; assignments encode rank * capacity
    + slot, or -1. Valid slots must be unique across all producers. Caller owns
    peer publication, zero initialization, synchronization and mapping lifetime.
    The scratch tensor is allocated for GEMM shape metadata, but is not written.
    """
    _peer_gemm_fake(x, weights, group_ends, assignments, outputs, pointers)
    import deep_gemm

    if torch.cuda.get_device_capability(x.device)[0] != 10:
        raise ValueError("Peer scatter epilogue requires SM100")
    queue = torch.zeros((1, 4), device=x.device, dtype=torch.int32)
    if x.shape[0]:
        scratch = x.new_empty((x.shape[0], weights.shape[1]))
        epilogue = deep_gemm.epilogue.PeerScatterStoreBF16(
            assignments, pointers, queue, outputs[0].shape[0]
        )
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            x, weights, scratch, group_ends, use_psum_layout=True, epilogue=epilogue
        )
    return queue


@peer_gemm_out.register_fake
def _peer_gemm_fake(x, weights, group_ends, assignments, outputs, pointers):
    if (
        x.ndim != 2
        or weights.ndim != 3
        or not outputs
        or x.dtype != torch.bfloat16
        or weights.dtype != x.dtype
        or x.device.type != "cuda"
        or weights.device != x.device
        or not x.is_contiguous()
        or not weights.is_contiguous()
        or x.shape[1] != weights.shape[2]
        or assignments.shape != (x.shape[0],)
        or assignments.dtype != torch.int32
        or assignments.device != x.device
        or not assignments.is_contiguous()
        or group_ends.shape != (weights.shape[0],)
        or group_ends.dtype != torch.int32
        or group_ends.device != x.device
        or not group_ends.is_contiguous()
        or pointers.shape != (len(outputs),)
        or pointers.dtype != torch.int64
        or pointers.device != x.device
        or not pointers.is_contiguous()
    ):
        raise ValueError("Invalid peer GEMM input or metadata")
    shape = outputs[0].shape
    if len(shape) != 2 or shape[0] < 1 or shape[1] != weights.shape[1]:
        raise ValueError("Invalid peer output geometry")
    if any(
        t.shape != shape
        or t.dtype != x.dtype
        or t.device != x.device
        or not t.is_contiguous()
        for t in outputs
    ):
        raise ValueError("Expected equal-capacity BF16 peer outputs")
    return torch.empty((1, 4), device=x.device, dtype=torch.int32)
