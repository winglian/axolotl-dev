# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""DeepEP-layout peer dispatch/combine ops, simulated on CPU with shared buffers."""

import threading

import pytest
import torch
from mono_ep.peer_dispatch import (
    LocalWorkspace,
    PeerDispatch,
    local_routing,
    producer_mask,
)

WORLD, CAP, K, H, E_LOCAL = 3, 7, 2, 16, 4


def _runtimes(world=WORLD, cap=CAP, barrier=None):
    specs = PeerDispatch.buffer_specs(world, cap, K, H)
    workspaces = LocalWorkspace.create(world, "cpu", specs, barrier)
    return [
        PeerDispatch(capacity=cap, top_k=K, hidden=H, workspace=w) for w in workspaces
    ]


def _routing(rank, tokens, generator):
    idx = torch.randint(0, WORLD * E_LOCAL, (tokens, K), generator=generator)
    idx[idx[:, 0] % 5 == rank, 1] = -1  # some dropped slots
    if rank == 0:
        idx[0] = -1  # an unrouted token
    weights = torch.rand(tokens, K, generator=generator)
    return idx, weights.masked_fill(idx < 0, 0)


def _local_kernel(experts, recv_x, recv_idx, recv_w):
    """Eager per-expert linear; ids outside ``[0, E_LOCAL)`` are skipped."""
    out = torch.zeros_like(recv_x, dtype=torch.float32)
    for e in range(E_LOCAL):
        for k in range(K):
            rows = (recv_idx[:, k] == e).nonzero(as_tuple=True)[0]
            if rows.numel():
                out[rows] += recv_w[rows, k].unsqueeze(-1) * (
                    recv_x[rows].float() @ experts[e]
                )
    return out.to(recv_x.dtype)


def _reference(x, idx, w, experts_global):
    out = torch.zeros(x.shape[0], H, dtype=torch.float32)
    for k in range(K):
        for e in range(WORLD * E_LOCAL):
            rows = (idx[:, k] == e).nonzero(as_tuple=True)[0]
            if rows.numel():
                out[rows] += w[rows, k].unsqueeze(-1) * (
                    x[rows].float() @ experts_global[e]
                )
    return out


def test_local_routing_and_producer_mask():
    all_idx = torch.tensor([[[0, 5], [-1, 3]], [[4, 7], [11, -1]]])  # [W=2, cap=2, K=2]
    local, routed = local_routing(all_idx, rank=1, num_local_experts=4)
    assert local.tolist() == [[-1, 1], [-1, -1], [0, 3], [-1, -1]]
    assert routed.tolist() == [True, False, True, False]
    mask = producer_mask(torch.tensor([[0, 5], [-1, 3], [11, -1]]), 4, 3)
    assert mask.tolist() == [
        [True, True, False],
        [True, False, False],
        [False, False, True],
    ]


def test_receive_layout_matches_published_routing():
    runtimes = _runtimes()
    generator = torch.Generator().manual_seed(1)
    tokens = [CAP, 3, 5]
    inputs = {}
    for rank, runtime in enumerate(runtimes):
        idx, w = _routing(rank, tokens[rank], generator)
        x = torch.randn(tokens[rank], H, generator=generator).bfloat16()
        inputs[rank] = (x, idx, w)
        runtime.publish(x, idx, w)
    for rank, runtime in enumerate(runtimes):
        local, weights, routed = runtime.collect_routing(E_LOCAL)
        recv_x = runtime.pull(routed)
        assert recv_x.shape == (WORLD * CAP, H)
        for source in range(WORLD):
            x, idx, w = inputs[source]
            for t in range(CAP):
                row = source * CAP + t
                if t >= tokens[source]:
                    assert not routed[row] and local[row].tolist() == [-1, -1]
                    continue
                mine = (idx[t] // E_LOCAL == rank) & (idx[t] >= 0)
                assert routed[row] == mine.any()
                assert (
                    local[row].tolist()
                    == torch.where(
                        mine, idx[t] - rank * E_LOCAL, torch.full_like(idx[t], -1)
                    ).tolist()
                )
                assert torch.equal(
                    weights[row], torch.where(mine, w[t], torch.zeros_like(w[t]))
                )
                expected = x[t] if mine.any() else torch.zeros(H, dtype=torch.bfloat16)
                assert torch.equal(recv_x[row], expected)


def _rank_forward_backward(rank, runtime, x, idx, w, experts, results):
    from mono_ep.peer_dispatch import peer_combine, peer_dispatch

    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    recv_x, recv_idx, recv_w = peer_dispatch(x.bfloat16(), idx, w, E_LOCAL, runtime.key)
    assert recv_x.shape == (WORLD * CAP, H) and recv_idx.dtype == torch.int64
    local_out = _local_kernel(experts, recv_x, recv_idx, recv_w)
    y = peer_combine(local_out, (recv_idx >= 0).any(-1), idx, E_LOCAL, runtime.key)
    assert y.shape == (x.shape[0], H) and y.dtype == torch.bfloat16
    (y.float() * results["grad_out"][rank]).sum().backward()
    results[rank] = (y.detach(), x.grad, w.grad)


def test_threaded_ranks_match_single_rank_reference():
    barrier = threading.Barrier(WORLD)
    runtimes = _runtimes(barrier=barrier)
    generator = torch.Generator().manual_seed(2)
    experts_global = torch.randn(WORLD * E_LOCAL, H, H, generator=generator) * 0.1
    tokens = [CAP, 2, 6]
    inputs = [_routing(r, tokens[r], generator) for r in range(WORLD)]
    xs = [
        torch.randn(tokens[r], H, generator=generator).bfloat16().float()
        for r in range(WORLD)
    ]
    results = {
        "grad_out": [
            torch.randn(tokens[r], H, generator=generator) for r in range(WORLD)
        ]
    }
    threads = [
        threading.Thread(
            target=_rank_forward_backward,
            args=(
                r,
                runtimes[r],
                xs[r],
                inputs[r][0],
                inputs[r][1],
                experts_global[r * E_LOCAL : (r + 1) * E_LOCAL],
                results,
            ),
        )
        for r in range(WORLD)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive()

    for rank in range(WORLD):
        idx, w = inputs[rank]
        x = xs[rank].clone().requires_grad_(True)
        wr = w.clone().requires_grad_(True)
        y_ref = _reference(x.bfloat16(), idx, wr, experts_global)
        (y_ref * results["grad_out"][rank]).sum().backward()
        y, grad_x, grad_w = results[rank]
        torch.testing.assert_close(
            y.float(), y_ref.bfloat16().float(), atol=2e-2, rtol=2e-2
        )
        assert grad_x is not None and grad_w is not None
        torch.testing.assert_close(grad_x, x.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(grad_w, wr.grad, atol=5e-2, rtol=5e-2)
        assert torch.equal(
            y[idx.eq(-1).all(-1)], torch.zeros_like(y[idx.eq(-1).all(-1)])
        )


def test_compiles_fullgraph_with_fake_ops():
    (runtime,) = _runtimes(world=1)
    from mono_ep.peer_dispatch import peer_combine, peer_dispatch

    def layer(x, idx, w):
        recv_x, recv_idx, recv_w = peer_dispatch(
            x.bfloat16(), idx, w, E_LOCAL, runtime.key
        )
        local_out = recv_w.sum(-1, keepdim=True).bfloat16() * recv_x
        return peer_combine(
            local_out, (recv_idx >= 0).any(-1), idx, E_LOCAL, runtime.key
        )

    torch._dynamo.reset()
    compiled = torch.compile(layer, fullgraph=True)
    x = torch.randn(CAP, H, requires_grad=True)
    idx = torch.randint(0, E_LOCAL, (CAP, K))
    w = torch.rand(CAP, K, requires_grad=True)
    y = compiled(x, idx, w)
    torch.testing.assert_close(y, layer(x, idx, w))
    y.float().sum().backward()
    assert x.grad is not None and w.grad is not None


def test_capacity_and_destroy():
    (runtime,) = _runtimes(world=1)
    from mono_ep.peer_dispatch import peer_dispatch

    with pytest.raises(ValueError, match="exceed"):
        peer_dispatch(
            torch.zeros(CAP + 1, H, dtype=torch.bfloat16),
            torch.zeros(CAP + 1, K, dtype=torch.int64),
            torch.zeros(CAP + 1, K),
            E_LOCAL,
            runtime.key,
        )
    runtime.destroy()
    with pytest.raises(RuntimeError, match="destroy"):
        peer_dispatch(
            torch.zeros(1, H, dtype=torch.bfloat16),
            torch.zeros(1, K, dtype=torch.int64),
            torch.zeros(1, K),
            E_LOCAL,
            runtime.key,
        )
