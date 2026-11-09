# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Fixed-capacity expert packing with GPU-resident counts and deterministic combine."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _layout_metadata(
    IDS,
    ORDER,
    PRECEDING,
    BASES,
    ENDS,
    SIZES,
    N: tl.constexpr,
    EXPERTS: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
):
    expert = tl.arange(0, BLOCK)
    lower = tl.full((BLOCK,), 0, tl.int32)
    upper = tl.full((BLOCK,), N, tl.int32)
    for _ in range(STEPS):
        middle = (lower + upper) // 2
        assignment = tl.load(ORDER + middle, middle < N, 0)
        value = tl.load(IDS + assignment, middle < N, 2147483647)
        advance = (lower < upper) & (value < expert)
        lower = tl.where(advance, middle + 1, lower)
        upper = tl.where((lower < upper) & ~advance, middle, upper)
    next_position = tl.gather(lower, tl.minimum(expert + 1, BLOCK - 1), axis=0)
    count = tl.where(expert < EXPERTS, next_position - lower, 0)
    padded = ((count + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    base = tl.cumsum(padded) - padded
    tl.store(PRECEDING + expert, lower, expert < EXPERTS)
    tl.store(BASES + expert, base, expert < EXPERTS)
    tl.store(ENDS + expert, base + count, expert < EXPERTS)
    tl.store(SIZES + expert, padded, expert < EXPERTS)


@triton.jit
def _permutation(
    IDS,
    ORDER,
    COUNTS,
    BASES,
    ASSIGNMENT_ROWS,
    ROW_ASSIGNMENTS,
    EXPERT_IDS,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    assignment = tl.load(ORDER + offset, offset < N, 0)
    expert = tl.load(IDS + assignment, offset < N, -1)
    valid = (offset < N) & (expert >= 0)
    preceding = tl.load(COUNTS + expert, valid, 0)
    base = tl.load(BASES + expert, valid, 0)
    row = base + offset - preceding
    tl.store(ASSIGNMENT_ROWS + assignment, row, valid)
    tl.store(ROW_ASSIGNMENTS + row, assignment, valid)
    tl.store(EXPERT_IDS + row, expert, valid)


@triton.jit
def _pack(
    X, ROW_ASSIGNMENTS, OUT, H: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    assignment = tl.load(ROW_ASSIGNMENTS + row).to(tl.int64)
    value = tl.load(X + (assignment // K) * H + col, (assignment >= 0) & (col < H), 0)
    tl.store(OUT + row * H + col, value, col < H)


@triton.jit
def _combine(VALUES, ROWS, OUT, H: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for slot in range(K):
        row = tl.load(ROWS + token * K + slot).to(tl.int64)
        value = tl.load(VALUES + row * H + col, (row >= 0) & (col < H), 0)
        total += value.to(tl.float32)
    tl.store(OUT + token * H + col, total, col < H)


def pack_rows(x, row_assignments, topk):
    out = x.new_empty((row_assignments.numel(), x.shape[1]))
    if out.shape[0]:
        _pack[(out.shape[0], triton.cdiv(x.shape[1], 256))](
            x,
            row_assignments,
            out,
            x.shape[1],
            topk,
            256,
        )
    return out


@triton.jit
def _pack_retained(
    Z,
    RANKS,
    TOKENS,
    ASSIGNMENTS,
    OUT,
    CAPACITY: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    assignment = tl.load(ASSIGNMENTS + row).to(tl.int64)
    valid = assignment >= 0
    received = assignment // TOPK
    rank = tl.load(RANKS + received, valid, 0).to(tl.int64)
    token = tl.load(TOKENS + received, valid, 0).to(tl.int64)
    source = ((rank * CAPACITY + token) * TOPK + assignment % TOPK) * WIDTH
    value = tl.load(Z + source + col, valid & (col < WIDTH), 0)
    tl.store(OUT + row * WIDTH + col, value, col < WIDTH)


def pack_retained_projections(saved_z, source_ranks, source_tokens, row_assignments):
    """Gather routed saved projections directly into expert-packed storage."""
    out = saved_z.new_empty((row_assignments.numel(), saved_z.shape[-1]))
    if out.shape[0]:
        _pack_retained[(out.shape[0], triton.cdiv(out.shape[1], 256))](
            saved_z,
            source_ranks,
            source_tokens,
            row_assignments,
            out,
            saved_z.shape[1],
            saved_z.shape[2],
            out.shape[1],
            256,
        )
    return out


def combine_rows(values, assignment_rows, tokens, topk):
    out = values.new_empty((tokens, values.shape[1]))
    if tokens:
        _combine[(tokens, triton.cdiv(values.shape[1], 256))](
            values,
            assignment_rows,
            out,
            values.shape[1],
            topk,
            256,
        )
    return out


@dataclass
class DeviceExpertLayout:
    use_psum_layout = True
    sizes = None
    assignment_rows: torch.Tensor
    row_assignments: torch.Tensor
    expert_ids: torch.Tensor
    ends: torch.Tensor
    topk: int
    starts: torch.Tensor
    padded_sizes: torch.Tensor

    @property
    def k_sizes(self):
        return self.ends

    @classmethod
    def build(cls, indices, experts, alignment):
        """Indices must be valid expert IDs or -1; no host-side data reads occur."""
        if indices.ndim != 2 or alignment < 128 or alignment % 128 or experts < 1:
            raise ValueError("invalid expert layout geometry")
        flat = indices.flatten()
        n = flat.numel()
        order = flat.argsort(stable=True)
        preceding = torch.empty(experts, dtype=torch.int32, device=indices.device)
        bases = torch.empty_like(preceding)
        ends = torch.empty_like(preceding)
        sizes = torch.empty_like(preceding)
        _layout_metadata[(1,)](
            flat,
            order,
            preceding,
            bases,
            ends,
            sizes,
            n,
            experts,
            alignment,
            triton.next_power_of_2(experts + 1),
            (n + 1).bit_length(),
        )
        capacity = triton.cdiv(n + experts * (alignment - 1), alignment) * alignment
        assignment_rows = torch.full((n,), -1, device=indices.device, dtype=torch.int64)
        row_assignments = torch.full(
            (capacity,), -1, device=indices.device, dtype=torch.int64
        )
        expert_ids = torch.full(
            (capacity,), -1, device=indices.device, dtype=torch.int32
        )
        if n:
            _permutation[(triton.cdiv(n, 256),)](
                flat,
                order,
                preceding,
                bases,
                assignment_rows,
                row_assignments,
                expert_ids,
                n,
                256,
            )
        return cls(
            assignment_rows,
            row_assignments,
            expert_ids,
            ends,
            indices.shape[1],
            bases,
            sizes,
        )

    def chunk_descriptors(self, chunks=1, alignment=128):
        """GPU-resident expert/start/padded-count descriptors with unpublished flags."""
        if chunks < 1 or alignment < 128 or alignment % 128:
            raise ValueError("Invalid readiness chunk geometry")
        expert = torch.arange(
            self.starts.numel(), device=self.starts.device, dtype=torch.int32
        )
        chunk = torch.arange(chunks, device=self.starts.device, dtype=torch.int32)
        rows = (
            (self.padded_sizes + chunks * alignment - 1) // (chunks * alignment)
        ) * alignment
        offset = torch.minimum(
            rows[:, None] * chunk[None, :], self.padded_sizes[:, None]
        )
        count = torch.minimum(rows[:, None], self.padded_sizes[:, None] - offset)
        return torch.stack(
            (
                expert[:, None].expand_as(offset),
                self.starts[:, None] + offset,
                count,
                torch.zeros_like(count),
            ),
            dim=-1,
        ).reshape(-1, 4)

    def token_ids(self):
        """Map packed rows to source tokens; padding remains -1."""
        return torch.where(
            self.row_assignments >= 0, self.row_assignments // self.topk, -1
        ).int()

    def pack(self, x):
        return pack_rows(x, self.row_assignments, self.topk)

    def scores(self, scores):
        if scores.numel() == 0:
            return scores.new_zeros(self.row_assignments.shape)
        return torch.where(
            self.row_assignments >= 0,
            scores.flatten()[self.row_assignments.clamp_min(0)],
            0,
        )

    def combine(self, values, tokens):
        return combine_rows(values, self.assignment_rows, tokens, self.topk)

    def restore_scores(self, values, scores):
        return torch.where(
            self.assignment_rows >= 0, values[self.assignment_rows.clamp_min(0)], 0
        ).reshape(scores.shape)
