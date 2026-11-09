# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Backend-independent NCCL chunk pipeline, extracted from Axolotl's EP scheduler.

This is a composable overlap prototype, not a fused communication kernel.
It synchronizes routing counts to the host; local activations may be retained.
"""

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.autograd.function import once_differentiable


def _compute_send_counts(
    topk_idx: torch.Tensor, num_ranks: int, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(dest [T,K], is_in_rank [T,P], send_counts [P])`` for global ``topk_idx``
    (``-1`` = unrouted). ``dest`` is the owning rank per slot, ``-1`` where unrouted."""
    dest = torch.where(
        topk_idx >= 0,
        torch.div(topk_idx, num_local_experts, rounding_mode="floor"),
        torch.full_like(topk_idx, -1),
    )
    ranks = torch.arange(num_ranks, device=topk_idx.device, dtype=dest.dtype)
    is_in_rank = (dest.unsqueeze(-1) == ranks).any(dim=1)
    return dest, is_in_rank, is_in_rank.sum(dim=0)


def _build_send_layout(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    dest: torch.Tensor,
    is_in_rank: torch.Tensor,
    num_send: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank-major, token-ascending send order with per-destination local expert ids.

    Returns ``(send_token_idx [N], send_rank [N], send_idx [N,K], send_w [N,K])``. ``num_send``
    must equal ``is_in_rank.sum()``; passing it from the host avoids a device sync.
    """
    T = is_in_rank.shape[0]
    flat = is_in_rank.t().reshape(-1)
    # a stable sort of the (rank, token)-ordered membership bits puts the selected pairs first
    order = torch.argsort((~flat).to(torch.int8), stable=True)[:num_send]
    send_rank = torch.div(order, T, rounding_mode="floor")
    send_token_idx = order - send_rank * T
    rank_col = send_rank.unsqueeze(-1).to(topk_idx.dtype)
    send_idx = torch.where(
        dest[send_token_idx] == rank_col,
        topk_idx[send_token_idx] - rank_col * num_local_experts,
        torch.full_like(rank_col, -1),
    )
    w = topk_weights[send_token_idx]
    send_w = torch.where(send_idx >= 0, w, torch.zeros_like(w))
    return send_token_idx, send_rank, send_idx, send_w


def _chunk_bounds(num_tokens: int, chunks: int) -> list[tuple[int, int]]:
    """``chunks`` contiguous ``[start, end)`` token ranges; the last takes the remainder,
    so some are empty when ``num_tokens < chunks``."""
    size = num_tokens // chunks
    return [
        (i * size, num_tokens if i == chunks - 1 else (i + 1) * size)
        for i in range(chunks)
    ]


def exchange(x, recv, send, group):
    return funcol.all_to_all_single(x.contiguous(), recv, send, group)


class _PipelinedMoE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        idx,
        scores,
        w1,
        w2,
        chunks,
        overlap,
        group,
        local_kernel,
        local_backward,
        local_forward,
    ):
        bounds = _chunk_bounds(x.shape[0], chunks)
        dest, membership, _ = _compute_send_counts(
            idx, dist.get_world_size(group), w1.shape[0]
        )
        send = torch.stack([membership[s:e].sum(0) for s, e in bounds])
        world = dist.get_world_size(group)
        recv = funcol.wait_tensor(
            exchange(send.t().contiguous(), [1] * world, [1] * world, group)
        ).t()
        splits = torch.stack((send, recv)).cpu().tolist()
        saved = []
        handles = []
        state_sizes = []
        outputs = [None] * chunks

        def dispatch_chunk(i):
            s, e = bounds[i]
            send_i, recv_i = splits[0][i], splits[1][i]
            token, _, ids, weights = _build_send_layout(
                idx[s:e],
                scores[s:e],
                dest[s:e],
                membership[s:e],
                sum(send_i),
                w1.shape[0],
            )
            rx = exchange(x[s:e][token], recv_i, send_i, group)
            rw = exchange(weights, recv_i, send_i, group)
            ri = exchange(ids, recv_i, send_i, group)
            return rx, rw, ri, token, ids

        def finish_chunk(i, out, token):
            s, e = bounds[i]
            out = funcol.wait_tensor(out)
            outputs[i] = x.new_zeros((e - s, x.shape[1])).index_add_(0, token, out)

        pending = dispatch_chunk(0)
        pending_combine = None
        for i, (s, e) in enumerate(bounds):
            rx, rw, ri, token, ids = pending
            if i + 1 < chunks:
                pending = dispatch_chunk(i + 1)
            rx, rw, ri = (funcol.wait_tensor(t) for t in (rx, rw, ri))
            local_state = ()
            if local_forward is None:
                y = local_kernel(rx, ri, rw, w1, w2)
            else:
                y, local_state = local_forward(rx, ri, rw, w1, w2)
                if not isinstance(local_state, tuple) or not all(
                    isinstance(t, torch.Tensor) for t in local_state
                ):
                    raise TypeError(
                        "local_forward must return a tuple of saved tensors"
                    )
            send_i, recv_i = splits[0][i], splits[1][i]
            out = exchange(y, send_i, recv_i, group)
            if pending_combine is not None:
                finish_chunk(i - 1, *pending_combine)
            pending_combine = out, token
            saved.extend((rx, rw, ri, token, ids, *local_state))
            state_sizes.append(len(local_state))
            handles.append((s, e, send_i, recv_i))
        finish_chunk(chunks - 1, *pending_combine)
        ctx.save_for_backward(w1, w2, *saved)
        ctx.group = group
        ctx.local_kernel = local_kernel
        ctx.local_backward = local_backward
        ctx.handles = handles
        ctx.state_sizes = state_sizes
        ctx.retained = local_forward is not None
        ctx.overlap = overlap
        ctx.score_shape = scores.shape
        return torch.cat(outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        w1, w2, *saved = ctx.saved_tensors
        group = ctx.group
        states, local_states = [], []
        offset = 0
        for size in ctx.state_sizes:
            states.append(saved[offset : offset + 5])
            local_states.append(tuple(saved[offset + 5 : offset + 5 + size]))
            offset += 5 + size
        dx = torch.zeros_like(dy)
        ds = torch.zeros(ctx.score_shape, device=dy.device, dtype=torch.float32)
        dw1 = torch.zeros_like(w1, dtype=torch.float32) if len(states) > 1 else None
        dw2 = torch.zeros_like(w2, dtype=torch.float32) if len(states) > 1 else None

        def dispatch_grad(i):
            s, e, send, recv = ctx.handles[i]
            token = states[i][3]
            return exchange(dy[s:e][token], recv, send, group)

        def finish(i, pending):
            gx, gs = (funcol.wait_tensor(t) for t in pending)
            s, e, _, _ = ctx.handles[i]
            token, ids = states[i][3:]
            dx[s:e].index_add_(0, token, gx)
            ds[s:e].index_add_(0, token, gs.masked_fill(ids < 0, 0))

        pending_grad = dispatch_grad(0)
        pending_return = None
        for i, (rx, rw, ri, _, _) in enumerate(states):
            grad = pending_grad
            if ctx.overlap and i + 1 < len(states):
                pending_grad = dispatch_grad(i + 1)
            grad = funcol.wait_tensor(grad)
            if ctx.retained:
                gx, gs, g1, g2 = ctx.local_backward(
                    grad, rx, ri, rw, w1, w2, activations=local_states[i]
                )
            elif ctx.local_backward is not None:
                gx, gs, g1, g2 = ctx.local_backward(grad, rx, ri, rw, w1, w2)
            else:
                with torch.enable_grad():
                    inputs = [t.detach().requires_grad_(True) for t in (rx, rw, w1, w2)]
                    out = ctx.local_kernel(
                        inputs[0], ri, inputs[1], inputs[2], inputs[3]
                    )
                    gx, gs, g1, g2 = torch.autograd.grad(out, inputs, grad)
            if len(states) == 1:
                dw1, dw2 = g1, g2
            else:
                dw1.add_(g1.float())
                dw2.add_(g2.float())
            _, _, send, recv = ctx.handles[i]
            returned = (
                exchange(gx, send, recv, group),
                exchange(gs, send, recv, group),
            )
            if pending_return is not None:
                finish(i - 1, pending_return)
            if ctx.overlap:
                pending_return = returned
            else:
                finish(i, returned)
                if i + 1 < len(states):
                    pending_grad = dispatch_grad(i + 1)
        if pending_return is not None:
            finish(len(states) - 1, pending_return)
        return (
            dx,
            None,
            ds,
            dw1.to(w1.dtype),
            dw2.to(w2.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def pipelined_moe(
    x,
    indices,
    scores,
    w1,
    w2,
    *,
    group,
    local_kernel,
    chunks=1,
    overlap_backward=True,
    local_backward=None,
    local_forward=None,
):
    """Route global expert IDs, invoke local compute, and return combined tokens.

    ``local_kernel(x, local_indices, scores, w1, w2)`` must be differentiable;
    -1 marks routing slots that belong to another rank. Each token is sent once
    per destination rank. All ranks must use identical chunk counts and call order.
    Forward always pipelines chunks; ``overlap_backward`` controls backward overlap.
    Multiple chunks are opt-in; they can cost more than communication overlap saves.
    Optional ``local_backward(dy, x, ids, scores, w1, w2)`` returns dx, dscore,
    dw1, dw2 directly, avoiding reconstruction of a local forward autograd graph.
    Optional ``local_forward`` returns (output, tuple_of_saved_tensors); its
    paired ``local_backward`` receives those tensors as ``activations=``.
    Saved tensors use autograd hooks and version checks.
    Only first-order derivatives are supported.
    """
    if chunks < 1 or indices.shape != scores.shape or indices.shape[0] != x.shape[0]:
        raise ValueError("invalid chunk count or routing geometry")
    if local_forward is not None and local_backward is None:
        raise ValueError("retained local forward requires an explicit backward")
    if dist.get_world_size(group) == 1 and local_forward is None:
        return local_kernel(x, indices, scores, w1, w2)
    return _PipelinedMoE.apply(
        x,
        indices,
        scores,
        w1,
        w2,
        chunks,
        overlap_backward,
        group,
        local_kernel,
        local_backward,
        local_forward,
    )
