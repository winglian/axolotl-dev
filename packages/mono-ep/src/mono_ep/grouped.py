# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Explicit DeepGEMM expert derivatives with FP32 weight-gradient output.

The packing implementation deliberately exposes host synchronization. It is a
correctness harness for the eventual GPU-resident dispatch/backward scheduler.
"""

from dataclasses import dataclass

import torch

from .activation import swiglu
from .ops import grouped_bf16, grouped_swiglu_backward, weight_grad_bf16
from .packing import combine_rows, pack_rows


@dataclass
class ExpertLayout:
    use_psum_layout = False
    assignments: torch.Tensor
    positions: torch.Tensor
    expert_ids: torch.Tensor
    sizes: list[int]
    k_sizes: torch.Tensor
    topk: int
    assignment_rows: torch.Tensor
    row_assignments: torch.Tensor

    @classmethod
    def build(cls, indices, experts, alignment):
        if indices.ndim != 2 or alignment < 128 or alignment % 128:
            raise ValueError(
                "expected [tokens, topk] and an alignment divisible by 128"
            )
        flat = indices.flatten()
        if ((flat < -1) | (flat >= experts)).any().item():
            raise ValueError("expert index outside [-1, num_experts)")
        order = flat.argsort(stable=True)
        order = order[flat[order] >= 0]
        counts = torch.bincount(flat[order].long(), minlength=experts).tolist()
        sizes = [(n + alignment - 1) // alignment * alignment for n in counts]
        ids = torch.full((sum(sizes),), -1, device=indices.device, dtype=torch.int32)
        positions = []
        start = 0
        for expert, (count, size) in enumerate(zip(counts, sizes, strict=True)):
            ids[start : start + count] = expert
            positions.append(torch.arange(start, start + count, device=indices.device))
            start += size
        positions = torch.cat(positions)
        assignment_rows = torch.full_like(flat, -1, dtype=torch.int64)
        assignment_rows[order] = positions
        row_assignments = torch.full_like(ids, -1, dtype=torch.int64)
        row_assignments[positions] = order
        return cls(
            order,
            positions,
            ids,
            sizes,
            torch.tensor(sizes, device=indices.device, dtype=torch.int32),
            indices.shape[1],
            assignment_rows,
            row_assignments,
        )

    def pack(self, x):
        return pack_rows(x, self.row_assignments, self.topk)

    def scores(self, scores):
        out = scores.new_zeros(self.expert_ids.shape)
        out[self.positions] = scores.flatten()[self.assignments]
        return out

    def combine(self, values, tokens):
        return combine_rows(values, self.assignment_rows, tokens, self.topk)

    def restore_scores(self, values, scores):
        result = torch.zeros_like(scores)
        result.view(-1)[self.assignments] = values[self.positions]
        return result


def _gemm(x, w, layout, transpose=False):
    # Hopper accepts K-major operands, so materialize the derivative transpose.
    w = w.transpose(1, 2).contiguous() if transpose else w
    ends = layout.starts + layout.padded_sizes if layout.use_psum_layout else None
    return grouped_bf16(x, w, layout.expert_ids, ends)


def _weight_grad(dy, x, layout, experts):
    return weight_grad_bf16(
        dy, x, layout.k_sizes, layout.sizes, experts, layout.use_psum_layout
    )


def expert_forward(x, scores, w1, w2, layout):
    packed_x = layout.pack(x)
    z = _gemm(packed_x, w1, layout)
    h = swiglu(z, layout.scores(scores))
    return layout.combine(_gemm(h, w2, layout), x.shape[0])


def expert_backward(
    dy,
    x,
    scores,
    w1,
    w2,
    layout,
    *,
    fuse_activation=True,
    saved_z=None,
    output_exchange=None,
):
    """Return expert derivatives, optionally consuming packed saved projections."""

    def combine_and_exchange(packed_dx, packed_ds):
        dx = layout.combine(packed_dx, x.shape[0])
        ds = layout.restore_scores(packed_ds, scores)
        return (dx, ds) if output_exchange is None else output_exchange(dx, ds)

    return expert_backward_packed(
        layout.pack(dy),
        layout.pack(x),
        layout.scores(scores),
        w1,
        w2,
        layout,
        fuse_activation=fuse_activation,
        saved_z=saved_z,
        output_exchange=combine_and_exchange,
    )


def expert_backward_packed(
    packed_dy,
    packed_x,
    packed_scores,
    w1,
    w2,
    layout,
    *,
    fuse_activation=True,
    saved_z=None,
    output_exchange=None,
    input_ready=None,
    precomputed_dh=None,
    output_gemm=None,
    owned_blocks=False,
):
    """Consume expert-packed rows and start gradient return before weight GEMMs."""
    if owned_blocks:
        if (
            saved_z is not None
            or precomputed_dh is not None
            or output_gemm is not None
            or not layout.use_psum_layout
        ):
            raise ValueError("Owned blocks require recomputation and GPU expert counts")
        from .owned_backward import owned_backward

        if input_ready is not None:
            input_ready()
        h, dz, dx, ds = owned_backward(
            packed_x,
            packed_dy,
            w1,
            w2,
            packed_scores,
            layout.starts,
            layout.ends - layout.starts,
        )
    elif (
        fuse_activation
        and saved_z is None
        and torch.cuda.get_device_capability(packed_x.device)[0] == 10
    ):
        from .recompute import grouped_recompute_swiglu

        dh = (
            _gemm(packed_dy, w2, layout, transpose=True)
            if precomputed_dh is None
            else precomputed_dh
        )
        if input_ready is not None:
            input_ready()
        h, dz, ds = grouped_recompute_swiglu(
            packed_x,
            w1,
            layout.expert_ids,
            dh,
            packed_scores,
            layout.starts + layout.padded_sizes if layout.use_psum_layout else None,
        )
    else:
        if input_ready is not None:
            input_ready()
        z = _gemm(packed_x, w1, layout) if saved_z is None else saved_z
        if fuse_activation:
            h, dz, ds = grouped_swiglu_backward(
                packed_dy, w2, layout.expert_ids, z, packed_scores
            )
        else:
            dh = _gemm(packed_dy, w2, layout, transpose=True)
            h, dz, ds = swiglu(z, packed_scores, dh)
    if not owned_blocks:
        dx = (
            _gemm(dz, w1, layout, transpose=True)
            if output_gemm is None
            else output_gemm(dz, w1, layout)
        )
    if output_exchange is not None:
        dx, ds = output_exchange(dx, ds)
    dw2 = _weight_grad(packed_dy, h, layout, w2.shape[0])
    dw1 = _weight_grad(dz, packed_x, layout, w1.shape[0])
    return dx, ds, dw1, dw2
