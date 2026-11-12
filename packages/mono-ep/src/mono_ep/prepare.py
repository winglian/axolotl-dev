# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compile and load a runtime's kernels before any rank waits for a peer.

The persistent kernels wait for their peers inside the kernel and trap after 60 s
(``comm/barrier.cuh``). A rank JIT-compiles each kernel at its first call, so a rank
whose compile finishes much later than a peer's (cold cache, loaded host) leaves the
peer spinning into the trap. :func:`launch` guards the first forward and the first
backward launch of every runtime: each rank compiles and loads the kernel without
launching it, the ranks meet on the host, then the kernel launches. :func:`prepare`
runs one guarded step on synthetic inputs, so the first training step compiles nothing.
"""

import contextlib

import torch
import torch.distributed as dist

FORWARD = "forward"
BACKWARD = "backward"


@contextlib.contextmanager
def _prepare_only():
    import deep_gemm

    deep_gemm._C.set_mono_ep_prepare_only(True)
    try:
        yield
    finally:
        deep_gemm._C.set_mono_ep_prepare_only(False)


def host_barrier(group):
    """Return once every rank of ``group`` got here; blocks the host, not only the stream."""
    device = "cpu" if dist.get_backend(group) == "gloo" else "cuda"
    flag = torch.ones(1, device=device)
    dist.all_reduce(flag, group=group)
    flag.item()


def launch(runtime, direction, kernel, *args, **kwargs):
    """``kernel(*args, **kwargs)``, a persistent entry point of ``runtime``. Its first
    ``direction`` launch compiles and loads the kernel on every rank and meets the other
    ranks first; the first launch per direction is the same collective on every rank,
    so the rendezvous is too. Later variants (another token count's tiling, the stored
    context) compile at their first call without one."""
    launched = getattr(runtime, "launched", None)
    if launched is not None and direction in launched:
        return kernel(*args, **kwargs)
    import deep_gemm

    if launched is None:
        launched = runtime.launched = set()
    if (
        dist.get_world_size(runtime.group) > 1
        and hasattr(deep_gemm._C, "set_mono_ep_prepare_only")
        and not torch.cuda.is_current_stream_capturing()
    ):
        with _prepare_only():
            kernel(*args, **kwargs)
        host_barrier(runtime.group)
    launched.add(direction)
    return kernel(*args, **kwargs)


def _lora_factors(local, hidden, intermediate, rank, dtype, device, generator):
    shapes = (
        (local, rank, hidden),
        (local, 2 * intermediate, rank),
        (local, rank, intermediate),
        (local, hidden, rank),
    )
    return [
        (torch.randn(s, device=device, generator=generator, dtype=torch.float32) * 0.01)
        .to(dtype)
        .requires_grad_()
        for s in shapes
    ]


def prepare(
    runtime,
    *,
    num_tokens=None,
    weight_dtype=torch.bfloat16,
    lora_rank=None,
    lora_dtype=torch.float32,
    lora_scaling=1.0,
    nvfp4_backend=None,
):
    """One forward and backward of ``runtime`` (a :class:`mono_ep.MonoEPRuntime`) on
    synthetic inputs of ``num_tokens`` (default the runtime's token capacity) tokens,
    through :func:`launch`, then one host barrier over the expert-parallel group.

    Every kernel of a training step with that token count, weight dtype and LoRA
    rank / dtype is compiled and loaded afterwards, on every rank. Nothing the step
    touches is kept besides the workspace the first step would allocate anyway (the
    backward pools); the synthetic inputs come from a private generator, so the global
    RNG state is unchanged. Repeat calls with the same arguments return at once."""
    from . import lora
    from .weight_cache import invalidate_weight_caches

    moe = runtime._moe
    if moe is None:
        raise RuntimeError("mono_ep runtime used after destroy()")
    tokens = int(num_tokens or moe.max_tokens)
    if not 0 < tokens <= moe.max_tokens:
        raise ValueError(
            f"num_tokens={tokens} must be in [1, {moe.max_tokens}] for this runtime"
        )
    key = (tokens, weight_dtype, lora_rank, lora_dtype, nvfp4_backend)
    if key in moe.prepared:
        return
    precision = runtime.precision
    is_lora = precision.endswith("_lora")
    if is_lora and not lora_rank:
        raise ValueError(f"precision {precision} needs lora_rank")
    world = dist.get_world_size(moe.group)
    local = moe.experts // world
    hidden, inter, topk = moe.hidden, moe.intermediate, moe.topk
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device).manual_seed(dist.get_rank(moe.group))

    def randn(*shape, dtype=torch.bfloat16, scale=0.02):
        return (torch.randn(shape, device=device, generator=generator) * scale).to(
            dtype
        )

    x = randn(tokens, hidden, scale=1.0).requires_grad_()
    indices = (
        torch.rand(tokens, moe.experts, device=device, generator=generator)
        .topk(topk, dim=-1)
        .indices
    )
    scores = torch.rand(
        tokens, topk, device=device, generator=generator
    ).requires_grad_()
    stored = moe._stored
    experts = None
    if precision == "nvfp4_lora":
        from .nvfp4_lora import Nvfp4LoraExperts, RequantCache

        ones = torch.ones(local, device=device)
        experts = Nvfp4LoraExperts(
            torch.zeros(
                local, 2 * inter, hidden // 2, device=device, dtype=torch.uint8
            ),
            torch.ones(local, 2 * inter, hidden // 16, device=device).to(
                torch.float8_e4m3fn
            ),
            ones,
            torch.zeros(local, hidden, inter // 2, device=device, dtype=torch.uint8),
            torch.ones(local, hidden, inter // 16, device=device).to(
                torch.float8_e4m3fn
            ),
            ones,
            scaling=lora_scaling,
            backend=nvfp4_backend,
        )
        factors = _lora_factors(
            local, hidden, inter, lora_rank, lora_dtype, device, generator
        )
        y = runtime.forward_lora(x, indices, scores, *factors, experts, RequantCache())
        inputs = [x, scores, *factors]
    elif is_lora:
        w1 = randn(local, 2 * inter, hidden, dtype=weight_dtype)
        w2 = randn(local, hidden, inter, dtype=weight_dtype)
        experts = lora.LoraExperts(scaling=lora_scaling)
        factors = _lora_factors(
            local, hidden, inter, lora_rank, lora_dtype, device, generator
        )
        y = runtime.forward_lora(
            x,
            indices,
            scores,
            *factors,
            experts,
            lora.MergedWeightCache(),
            w1=w1,
            w2=w2,
        )
        inputs = [x, scores, *factors]
    else:
        w1 = randn(local, 2 * inter, hidden, dtype=weight_dtype).requires_grad_()
        w2 = randn(local, hidden, inter, dtype=weight_dtype).requires_grad_()
        y = runtime.forward(x, indices, scores, w1, w2)
        inputs = [x, scores, w1, w2]
    dy = randn(*y.shape, dtype=y.dtype, scale=1.0)
    torch.autograd.grad(y, inputs, dy)
    del x, scores, inputs, y, dy
    if experts is not None:
        lora._LAYERS.pop(getattr(experts, "key", None), None)
    if stored is None:
        moe._stored = None
    invalidate_weight_caches()
    host_barrier(moe.group)
    moe.prepared.add(key)
