# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Fixed-capacity NCCL baseline without routing-count transfers to the host."""

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F

from .packing import DeviceExpertLayout, pack_retained_projections
from .pipeline import exchange


def fixed_backward(dy, x, indices, scores, local_experts, runtime, retained, compute):
    world = dist.get_world_size(runtime.group)
    capacity = runtime.max_tokens
    tokens = x.shape[0]
    padding = capacity - tokens
    ranks = torch.arange(world, device=x.device, dtype=indices.dtype)
    padded_ids = F.pad(indices, (0, 0, 0, padding), value=-1)
    owner = torch.div(padded_ids, local_experts, rounding_mode="floor")
    local_ids = torch.where(
        owner[None] == ranks[:, None, None],
        padded_ids[None] - ranks[:, None, None] * local_experts,
        -1,
    ).flatten(0, 1)
    padded_scores = F.pad(scores, (0, 0, 0, padding))
    weights = padded_scores.expand(world, -1, -1).reshape(world * capacity, -1)
    weights = weights.masked_fill(local_ids < 0, 0)

    def broadcast_rows(tensor):
        return (
            F.pad(tensor, (0, 0, 0, padding))
            .expand(world, -1, -1)
            .reshape(world * capacity, -1)
        )

    pending = [
        exchange(tensor, None, None, runtime.group)
        for tensor in (broadcast_rows(x), broadcast_rows(dy), weights, local_ids)
    ]
    rx, rdy, rs, ri = (funcol.wait_tensor(tensor) for tensor in pending)
    layout = DeviceExpertLayout.build(ri, local_experts, runtime.alignment)
    packed_z = None
    if retained:
        source_ranks = ranks.repeat_interleave(capacity)
        source_tokens = torch.arange(capacity, device=x.device).repeat(world)
        packed_z = pack_retained_projections(
            retained[0], source_ranks, source_tokens, layout.row_assignments
        )

    def return_gradients(dx, ds):
        return (
            exchange(dx, None, None, runtime.group),
            exchange(ds, None, None, runtime.group),
        )

    dx, ds, dw1, dw2 = compute(rdy, rx, rs, layout, packed_z, return_gradients)
    dx, ds = funcol.wait_tensor(dx), funcol.wait_tensor(ds)
    dx = dx.reshape(world, capacity, x.shape[1]).sum(0, dtype=torch.float32)[:tokens]
    ds = ds.reshape(world, capacity, scores.shape[1]).sum(0)[:tokens]
    return dx.to(x.dtype), ds, dw1, dw2
