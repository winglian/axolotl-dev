# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The streamed peer pull/push/reduce against the legacy row kernels, bitwise.

One GPU: ``world`` simulated ranks share plain CUDA tensors (``LocalWorkspace``), which
exercises the kernels' pointer arrays, masks, compact prefix sums and the FP32 reduction
order. Two GPUs: the same comparison over real torch symmetric memory and barriers, in
two spawned processes.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.gpu

K, H, E_LOCAL = 8, 2048, 16


def _skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("peer streams require CUDA")
    pytest.importorskip("triton")


def _routing(kind, tokens, world, rank, generator, device):
    """Global expert ids ``[tokens, K]`` with ``-1`` drops; ``all_unrouted`` sends nothing
    to ``rank``, ``skewed`` sends every slot to rank 0."""
    experts = world * E_LOCAL
    if kind == "skewed":
        idx = torch.randint(0, E_LOCAL, (tokens, K), generator=generator, device=device)
    elif kind == "all_unrouted":
        others = torch.tensor(
            [e for e in range(experts) if e // E_LOCAL != rank] or [0], device=device
        )
        pick = torch.randint(
            0, others.numel(), (tokens, K), generator=generator, device=device
        )
        idx = others[pick]
    else:
        idx = torch.randint(0, experts, (tokens, K), generator=generator, device=device)
    drop = torch.rand(tokens, K, generator=generator, device=device) < 0.1
    idx = idx.masked_fill(drop, -1)
    if kind != "skewed":
        idx[::7] = -1
    weights = torch.rand(tokens, K, generator=generator, device=device)
    return idx, weights.masked_fill(idx < 0, 0)


def _run_pair(runtimes, kind, tokens, generator):
    """Forward + both backwards on every simulated rank; returns the per-rank tensors
    every variant must reproduce bitwise."""
    from mono_ep.peer_dispatch import dispatch_compact, peer_combine, peer_dispatch

    world = len(runtimes)
    device = runtimes[0].device
    inputs = []
    for rank in range(world):
        idx, w = _routing(kind, tokens[rank], world, rank, generator, device)
        x = torch.randn(tokens[rank], H, generator=generator, device=device)
        inputs.append((x.bfloat16(), idx, w))
    outs = []
    for rank, runtime in enumerate(runtimes):
        x, idx, w = inputs[rank]
        x = x.clone().requires_grad_(True)
        w = w.clone().requires_grad_(True)
        if runtime.compact:
            recv_x, recv_idx, recv_w, slots = dispatch_compact(
                x, idx, w, E_LOCAL, runtime.key
            )
            rows = slots
        else:
            recv_x, recv_idx, recv_w = peer_dispatch(x, idx, w, E_LOCAL, runtime.key)
            rows = (recv_idx >= 0).any(-1)
        local_out = (recv_x.float() * (1 + recv_w.sum(-1, keepdim=True))).bfloat16()
        y = peer_combine(local_out, rows, idx, E_LOCAL, runtime.key)
        grad_out = torch.randn(y.shape, generator=generator, device=device)
        (y.float() * grad_out).sum().backward()
        if runtime.compact:
            # back into the fixed layout, where the legacy pull is the reference
            fixed = torch.zeros(runtime.rows, H, dtype=recv_x.dtype, device=device)
            fixed[rows] = recv_x.detach()
            recv_x = fixed
        outs.append(
            {
                "recv_x": recv_x.detach(),
                "recv_rows": recv_x.shape[0],
                "y": y.detach(),
                "grad_x": x.grad,
                "grad_w": w.grad,
                "extra_barriers": runtime.extra_barriers,
            }
        )
    return outs


def _single_device_runtimes(world, cap, transfer, compact):
    from mono_ep.peer_dispatch import LocalWorkspace, PeerDispatch

    specs = PeerDispatch.buffer_specs(world, cap, K, H)
    workspaces = LocalWorkspace.create(world, "cuda", specs)
    return [
        PeerDispatch(
            capacity=cap,
            top_k=K,
            hidden=H,
            workspace=w,
            transfer=transfer,
            compact=compact,
        )
        for w in workspaces
    ]


def _compare(reference, candidate, compact):
    for ref, out in zip(reference, candidate, strict=True):
        assert torch.equal(ref["y"], out["y"])
        assert torch.equal(ref["grad_x"], out["grad_x"])
        assert torch.equal(ref["grad_w"], out["grad_w"])
        assert torch.equal(ref["recv_x"], out["recv_x"])


@pytest.mark.parametrize("kind", ["random", "all_unrouted", "skewed"])
@pytest.mark.parametrize("world", [1, 3])
def test_streamed_matches_legacy_single_device(kind, world):
    _skip_without_cuda()
    cap = 300
    tokens = [cap, max(1, cap - 37), 5][:world]
    results = {}
    for transfer, compact in (
        ("legacy", False),
        ("streamed", False),
        ("streamed", True),
    ):
        generator = torch.Generator(device="cuda").manual_seed(3)
        runtimes = _single_device_runtimes(world, cap, transfer, compact)
        try:
            results[(transfer, compact)] = _run_pair(runtimes, kind, tokens, generator)
        finally:
            for runtime in runtimes:
                runtime.destroy()
    reference = results[("legacy", False)]
    _compare(reference, results[("streamed", False)], compact=False)
    _compare(reference, results[("streamed", True)], compact=True)
    # each rank ran dispatch, combine, combine backward, dispatch backward: alternating
    # families, so the per-op barrier is the only one
    assert all(out["extra_barriers"] == 0 for out in results[("streamed", False)])


def test_reduce_slots_matches_masked_sum():
    _skip_without_cuda()
    from mono_ep.peer_stream import reduce_slots

    world, cap, tokens = 4, 50, 41
    generator = torch.Generator(device="cuda").manual_seed(5)
    for width, dtype in (
        (H, torch.bfloat16),
        (K, torch.float32),
        (1000, torch.bfloat16),
    ):
        buffer = torch.randn(world * cap, width, generator=generator, device="cuda").to(
            dtype
        )
        mask = torch.rand(world, tokens, generator=generator, device="cuda") < 0.6
        # masked slots may hold garbage
        buffer.view(world, cap, width)[:, :tokens][~mask] = float("nan")
        expected = torch.zeros(tokens, width, device="cuda")
        slots = buffer.view(world, cap, width)[:, :tokens]
        for p in range(world):
            expected += slots[p].float().masked_fill(~mask[p].unsqueeze(-1), 0)
        for out_dtype in (dtype, torch.float32):
            assert torch.equal(
                reduce_slots(buffer, mask.contiguous(), world, out_dtype),
                expected.to(out_dtype),
            )


def test_same_family_ops_insert_a_barrier():
    _skip_without_cuda()
    from mono_ep.peer_dispatch import peer_combine_backward

    (runtime,) = _single_device_runtimes(1, 8, "streamed", False)
    try:
        grad_y = torch.randn(8, H, device="cuda", dtype=torch.bfloat16)
        rows = torch.ones(8, dtype=torch.bool, device="cuda")
        peer_combine_backward(grad_y, rows, runtime.key)
        peer_combine_backward(grad_y, rows, runtime.key)
        assert runtime.extra_barriers == 1
    finally:
        runtime.destroy()


def _two_gpu_worker(rank, world, port, results):
    import torch.distributed as dist
    from mono_ep.peer_dispatch import PeerDispatch

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    device = torch.device("cuda", rank)
    cap = 257
    tokens = [cap, cap - 31][:world]
    outs = {}
    for transfer, compact in (
        ("legacy", False),
        ("streamed", False),
        ("streamed", True),
    ):
        for kind in ("random", "all_unrouted", "skewed"):
            generator = torch.Generator(device=device).manual_seed(11)
            runtime = PeerDispatch(
                capacity=cap,
                top_k=K,
                hidden=H,
                group=dist.group.WORLD,
                device=device,
                transfer=transfer,
                compact=compact,
            )
            try:
                # every rank draws the same routing stream so the published layout is
                # identical across variants; only this rank's inputs are used here
                inputs = []
                for r in range(world):
                    idx, w = _routing(kind, tokens[r], world, r, generator, device)
                    x = torch.randn(tokens[r], H, generator=generator, device=device)
                    inputs.append((x.bfloat16(), idx, w))
                x, idx, w = inputs[rank]
                x = x.clone().requires_grad_(True)
                w = w.clone().requires_grad_(True)
                from mono_ep.peer_dispatch import (
                    dispatch_compact,
                    peer_combine,
                    peer_dispatch,
                )

                if compact:
                    recv_x, recv_idx, recv_w, rows = dispatch_compact(
                        x, idx, w, E_LOCAL, runtime.key
                    )
                else:
                    recv_x, recv_idx, recv_w = peer_dispatch(
                        x, idx, w, E_LOCAL, runtime.key
                    )
                    rows = (recv_idx >= 0).any(-1)
                local_out = (
                    recv_x.float() * (1 + recv_w.sum(-1, keepdim=True))
                ).bfloat16()
                y = peer_combine(local_out, rows, idx, E_LOCAL, runtime.key)
                grad_out = torch.randn(y.shape, generator=generator, device=device)
                (y.float() * grad_out).sum().backward()
                if compact:
                    fixed = torch.zeros(
                        runtime.rows, H, dtype=recv_x.dtype, device=device
                    )
                    fixed[rows] = recv_x.detach()
                    recv_x = fixed
                torch.cuda.synchronize()
                outs[(transfer, compact, kind)] = (
                    y.detach().cpu(),
                    x.grad.cpu(),
                    w.grad.cpu(),
                    recv_x.detach().cpu(),
                )
            finally:
                dist.barrier()
                runtime.destroy()
    results[rank] = outs
    dist.destroy_process_group()


def test_streamed_matches_legacy_two_gpus():
    _skip_without_cuda()
    if torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    import torch.multiprocessing as mp

    world = 2
    manager = mp.Manager()
    results = manager.dict()
    port = 29000 + os.getpid() % 1000
    mp.spawn(_two_gpu_worker, args=(world, port, results), nprocs=world, join=True)
    for rank in range(world):
        outs = results[rank]
        for kind in ("random", "all_unrouted", "skewed"):
            reference = outs[("legacy", False, kind)]
            for variant in (("streamed", False, kind), ("streamed", True, kind)):
                for a, b in zip(reference, outs[variant], strict=True):
                    assert torch.equal(a, b), (rank, variant)
