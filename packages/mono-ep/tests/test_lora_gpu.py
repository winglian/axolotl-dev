# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Fused LoRA on the SM100 BF16 / MXFP8 and SM90 BF16 / blockwise FP8 runtimes: adapter
gradients against an FP32 merge-aware reference (next to the full fine-tune dW envelope of
the same runtime), frozen base, merged-weight rebuild per optimizer step, ring wrap and skewed routing."""

import json

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]

HIDDEN, INNER, EXPERTS, RANK, SCALING = 1024, 512, 8, 16, 2.0


@pytest.fixture
def group(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (
        9,
        10,
    ):
        pytest.skip("the fused LoRA path needs SM90 or SM100")
    pytest.importorskip("deep_gemm")
    import torch.distributed as dist

    dist.init_process_group(
        "nccl",
        store=dist.FileStore(str(tmp_path / "store"), 1),
        rank=0,
        world_size=1,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    yield dist.group.WORLD
    dist.destroy_process_group()


ARCH_DTYPES = {9: ("bf16", "blockwise"), 10: ("bf16", "mxfp8")}


def _require(dtype):
    if dtype not in ARCH_DTYPES[torch.cuda.get_device_capability()[0]]:
        pytest.skip(f"{dtype} does not run on this architecture")


def _runtime(group, compute_dtype, tokens, topk):
    import deep_gemm
    from mono_ep.runtime import PersistentMoE

    deep_gemm.set_num_sms(torch.cuda.get_device_properties(0).multi_processor_count)
    return PersistentMoE(
        group,
        EXPERTS,
        tokens,
        topk,
        HIDDEN,
        INNER,
        compute_dtype=compute_dtype,
        fuse_activation=True,
        backward_backend="native",
    )


def _inputs(tokens, topk, routing="balanced", seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, device="cuda", generator=gen).bfloat16()
    ids = torch.rand(tokens, EXPERTS, device="cuda", generator=gen).argsort(-1)[
        :, :topk
    ]
    if routing == "skewed":
        # expert 0 takes every token, the rest share the remaining slots
        ids = torch.where(ids == 0, ids[:, :1], ids)
        ids[:, 0] = 0
    scores = torch.rand(tokens, topk, device="cuda", generator=gen).softmax(-1)
    w1 = (
        torch.randn(EXPERTS, 2 * INNER, HIDDEN, device="cuda", generator=gen)
        / HIDDEN**0.5
    ).bfloat16()
    w2 = (
        torch.randn(EXPERTS, HIDDEN, INNER, device="cuda", generator=gen) / INNER**0.5
    ).bfloat16()
    adapters = [
        torch.randn(EXPERTS, RANK, HIDDEN, device="cuda", generator=gen) / HIDDEN**0.5,
        torch.randn(EXPERTS, 2 * INNER, RANK, device="cuda", generator=gen) * 0.02,
        torch.randn(EXPERTS, RANK, INNER, device="cuda", generator=gen) / INNER**0.5,
        torch.randn(EXPERTS, HIDDEN, RANK, device="cuda", generator=gen) * 0.02,
    ]
    dy = torch.randn(tokens, HIDDEN, device="cuda", generator=gen).bfloat16()
    return x, ids.contiguous(), scores, w1, w2, adapters, dy


def _moe_fp32(x, ids, scores, w1, w2):
    y = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for e in range(EXPERTS):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        gate, up = (x[tok] @ w1[e].t()).chunk(2, dim=-1)
        out = (F.silu(gate) * up) @ w2[e].t()
        y.index_add_(0, tok, out * scores[tok, slot, None])
    return y


def reference(x, ids, scores, w1, w2, adapters, dy):
    """FP32 merge-aware forward/backward: ``y``, ``dx``, ``dscore``, ``dA1 dB1 dA2 dB2``
    and the full ``dW1 dW2`` of the merged weights."""
    xf = x.detach().float().clone().requires_grad_()
    sf = scores.detach().float().clone().requires_grad_()
    a1, b1, a2, b2 = (t.detach().float().clone().requires_grad_() for t in adapters)
    m1 = (w1.float() + SCALING * b1 @ a1).detach().requires_grad_()
    m2 = (w2.float() + SCALING * b2 @ a2).detach().requires_grad_()
    y = _moe_fp32(xf, ids, sf, m1, m2)
    y.backward(dy.float())
    dw1, dw2 = m1.grad, m2.grad
    grads = (
        SCALING * b1.detach().transpose(1, 2) @ dw1,
        SCALING * dw1 @ a1.detach().transpose(1, 2),
        SCALING * b2.detach().transpose(1, 2) @ dw2,
        SCALING * dw2 @ a2.detach().transpose(1, 2),
    )
    return y.detach(), xf.grad, sf.grad, grads, (dw1, dw2)


def rel(got, want):
    return float((got.float() - want.float()).norm() / want.float().norm())


def _lora_step(runtime, cache, layer, inputs, adapters):
    from mono_ep.lora import lora_moe

    x, ids, scores, w1, w2, _, dy = inputs
    xl, sl = x.clone().requires_grad_(), scores.clone().requires_grad_()
    y = lora_moe(xl, ids, sl, w1, w2, *adapters, runtime, layer, cache)
    y.backward(dy)
    torch.cuda.synchronize()
    return y.detach(), xl.grad, sl.grad, [a.grad for a in adapters]


def _full_ft_dw(runtime, inputs):
    """Full fine-tune dW of the same runtime on the merged weights (the envelope)."""
    from mono_ep.lora import merge

    x, ids, scores, w1, w2, adapters, dy = inputs
    m1 = merge(w1, adapters[1], adapters[0], SCALING).detach().requires_grad_()
    m2 = merge(w2, adapters[3], adapters[2], SCALING).detach().requires_grad_()
    sl = scores.clone().requires_grad_()
    runtime(x, ids, sl, m1, m2).backward(dy)
    torch.cuda.synchronize()
    return m1.grad, m2.grad, sl.grad


CASES = [
    # (compute dtype, tokens, topk, routing, stored context)
    ("bf16", 256, 4, "balanced", "off"),
    ("bf16", 256, 4, "skewed", "always"),
    ("mxfp8", 256, 4, "balanced", "off"),
    ("mxfp8", 256, 4, "skewed", "off"),
    # more routed rows than the forward ring and the backward's one-wave ring floor
    ("bf16", 4096, 8, "skewed", "off"),
    ("mxfp8", 4096, 8, "balanced", "off"),
    ("blockwise", 256, 4, "balanced", "off"),
    ("blockwise", 256, 4, "skewed", "off"),
    ("blockwise", 4096, 8, "skewed", "off"),
]


@pytest.mark.parametrize("dtype,tokens,topk,routing,stored", CASES)
def test_lora_grads_match_fp32_reference(
    group, monkeypatch, dtype, tokens, topk, routing, stored
):
    from mono_ep.lora import LoraExperts, MergedWeightCache

    _require(dtype)
    monkeypatch.setenv("MONO_EP_STORED_CONTEXT", stored)
    if tokens > 1024:
        monkeypatch.setenv("MONO_EP_FORWARD_RING_ROWS", "8192")
    inputs = _inputs(tokens, topk, routing)
    runtime = _runtime(group, dtype, tokens, topk)
    adapters = [a.clone().requires_grad_() for a in inputs[5]]
    cache = MergedWeightCache(mode="layer")
    y, dx, ds, grads = _lora_step(
        runtime, cache, LoraExperts(scaling=SCALING), inputs, adapters
    )
    y_ref, dx_ref, ds_ref, grads_ref, dw_ref = reference(*inputs)
    dw = _full_ft_dw(runtime, inputs)
    runtime.close()
    errs = dict(
        y=rel(y, y_ref),
        dx=rel(dx, dx_ref),
        dscore=rel(ds, ds_ref),
        **{
            k: rel(g, r)
            for k, g, r in zip(
                ("dA1", "dB1", "dA2", "dB2"), grads, grads_ref, strict=True
            )
        },
        full_ft_dW1=rel(dw[0], dw_ref[0]),
        full_ft_dW2=rel(dw[1], dw_ref[1]),
        full_ft_dscore=rel(dw[2], ds_ref),
        lora_vs_full_ft_dscore=rel(ds, dw[2]),
    )
    print(
        "LORA_ERRS "
        + json.dumps(dict(case=[dtype, tokens, topk, routing, stored], **errs))
    )
    envelope = max(errs["full_ft_dW1"], errs["full_ft_dW2"])
    # the adapter gradients come from the BF16 pools with BF16 rank-r intermediates
    bound = max(2.0 * envelope, 1e-2)
    for key in ("dA1", "dB1", "dA2", "dB2"):
        assert errs[key] < bound, (key, errs)
    assert errs["y"] < (2e-2 if dtype == "bf16" else 0.1), errs
    assert errs["dx"] < (2e-2 if dtype == "bf16" else 0.1), errs


@pytest.mark.parametrize("dtype", ["bf16", "mxfp8", "blockwise"])
def test_base_frozen_and_merge_rebuilt_per_step(group, monkeypatch, dtype):
    from mono_ep.lora import (
        LoraExperts,
        MergedWeightCache,
        _fused_mxfp8_forms,
        attach_post_step_hook,
    )

    _require(dtype)
    monkeypatch.setenv("MONO_EP_STORED_CONTEXT", "off")
    tokens, topk = 256, 4
    inputs = _inputs(tokens, topk)
    x, ids, scores, w1, w2, init, dy = inputs
    w1_before, w2_before = w1.clone(), w2.clone()
    runtime = _runtime(group, dtype, tokens, topk)
    adapters = [torch.nn.Parameter(a.clone()) for a in init]
    optimizer = torch.optim.AdamW(adapters, lr=1e-2)
    cache = MergedWeightCache(mode="all")
    attach_post_step_hook(optimizer, cache)
    layer = LoraExperts(scaling=SCALING)
    outputs = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        y, _, _, grads = _lora_step(runtime, cache, layer, inputs, adapters)
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        entry = cache.entries[id(layer)]
        assert entry.version == step == cache.version
        fresh = (
            LoraExperts(scaling=SCALING)
            .bind(w1, w2, _fused_mxfp8_forms(runtime, w1))
            .merged(*adapters)
        )
        assert type(fresh) is type(entry)
        assert all(
            torch.equal(a, b)
            for a, b in zip(entry.tensors(), fresh.tensors(), strict=True)
        )
        outputs.append(y)
        optimizer.step()
    runtime.close()
    # one merge per step: the backward reused the forward's build
    assert cache.builds == 3 and cache.hits == 3
    assert torch.equal(w1, w1_before) and torch.equal(w2, w2_before)
    assert not w1.requires_grad and w1.grad is None and w2.grad is None
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[1], outputs[2])


def _stack(runtime, cache, layers, inputs, adapters, checkpoint_policy=None):
    """``len(layers)`` MoE layers in sequence on one runtime, each with its own adapters."""
    import functools

    from mono_ep.lora import lora_moe
    from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

    x, ids, scores, w1, w2, _, dy = inputs
    h = x.clone().requires_grad_()
    out = h
    for layer, factors in zip(layers, adapters, strict=True):

        def run(t, layer=layer, factors=factors):
            return lora_moe(t, ids, scores, w1, w2, *factors, runtime, layer, cache)

        if checkpoint_policy is None:
            out = run(out)
        else:
            out = checkpoint(
                run,
                out,
                use_reentrant=False,
                context_fn=functools.partial(
                    create_selective_checkpoint_contexts, checkpoint_policy
                ),
            )
    out.backward(dy)
    torch.cuda.synchronize()
    return out.detach(), h.grad, [[a.grad for a in f] for f in adapters]


@pytest.mark.parametrize("dtype", ["bf16", "mxfp8", "blockwise"])
def test_layer_cache_eviction_and_sac(group, monkeypatch, dtype):
    import mono_ep.runtime as runtime_mod
    from mono_ep.lora import LoraExperts, MergedWeightCache
    from torch.utils.checkpoint import CheckpointPolicy

    _require(dtype)
    monkeypatch.setenv("MONO_EP_STORED_CONTEXT", "off")
    tokens, topk = 256, 4
    inputs = _inputs(tokens, topk)
    runtime = _runtime(group, dtype, tokens, topk)
    layers = [LoraExperts(scaling=SCALING) for _ in range(3)]

    def fresh_adapters():
        return [
            [(a * (1 + 0.1 * i)).clone().requires_grad_() for a in inputs[5]]
            for i in range(3)
        ]

    # every layer's forms held across the step: the reference
    all_cache = MergedWeightCache(mode="all")
    y_ref, dx_ref, grads_ref = _stack(
        runtime, all_cache, layers, inputs, fresh_adapters()
    )
    # two layers resident: the first layer's backward rebuilds its evicted merge
    layer_cache = MergedWeightCache(mode="layer")
    y, dx, grads = _stack(runtime, layer_cache, layers, inputs, fresh_adapters())
    assert layer_cache.evictions >= 1 and layer_cache.backward_builds >= 1
    assert torch.equal(y, y_ref) and torch.equal(dx, dx_ref)
    for got, want in zip(sum(grads, []), sum(grads_ref, []), strict=True):
        assert rel(got, want) < 1e-3

    # SAC saves the fused LoRA forward: its recompute does not launch it again
    calls = []
    forward_impl = runtime_mod._forward_impl

    def counting(*args, **kwargs):
        calls.append(1)
        return forward_impl(*args, **kwargs)

    monkeypatch.setattr(runtime_mod, "_forward_impl", counting)
    saved = {torch.ops.mono_ep.lora_forward.default}

    def policy(ctx, op, *args, **kwargs):
        if op in saved:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    sac_cache = MergedWeightCache(mode="layer")
    y_sac, dx_sac, grads_sac = _stack(
        runtime, sac_cache, layers, inputs, fresh_adapters(), checkpoint_policy=policy
    )
    runtime.close()
    assert len(calls) == 3
    assert torch.equal(y_sac, y_ref) and torch.equal(dx_sac, dx_ref)
    for got, want in zip(sum(grads_sac, []), sum(grads_ref, []), strict=True):
        assert rel(got, want) < 1e-3


def test_fused_mxfp8_forms_match_a_single_rounded_merge(group):
    """The MXFP8 forms with the merge fused into the CUDA builder: bitwise the plain forms for zero
    adapters, and the forms of the single-rounded FP32 merge up to the rank-r summation order."""
    _require("mxfp8")
    from mono_ep.mx_weight_forms import cuda_mx_backward_forms, cuda_mx_lora_forms
    from mono_ep.token_quantization import prepare_persistent_mxfp8

    def plain(m1, m2):
        f1, f2t, f1t = cuda_mx_backward_forms(m1, m2)
        return dict(
            sm100_mx_w1=f1,
            sm100_mx_w1t=f1t,
            sm100_mx_w2=prepare_persistent_mxfp8(m2, weights=True),
            sm100_mx_w2t=f2t,
        )

    _, _, _, w1, w2, _, _ = _inputs(256, 4)
    gen = torch.Generator(device="cuda").manual_seed(1)
    for rank in (8, 16, 40):
        a1, a2 = (
            torch.randn(EXPERTS, rank, k, device="cuda", generator=gen) / k**0.5
            for k in (HIDDEN, INNER)
        )
        b1, b2 = (
            torch.randn(EXPERTS, n, rank, device="cuda", generator=gen) * 0.02
            for n in (2 * INNER, HIDDEN)
        )
        zero = cuda_mx_lora_forms(w1, w2, a1, b1 * 0, a2, b2 * 0, SCALING)
        base = plain(w1, w2)
        for name in base:
            assert torch.equal(
                zero[name][0].view(torch.uint8), base[name][0].view(torch.uint8)
            )
            assert torch.equal(zero[name][1], base[name][1])
        fused = cuda_mx_lora_forms(w1, w2, a1, b1, a2, b2, SCALING)
        single = plain(
            (
                w1.float() + SCALING * b1.bfloat16().float() @ a1.bfloat16().float()
            ).bfloat16(),
            (
                w2.float() + SCALING * b2.bfloat16().float() @ a2.bfloat16().float()
            ).bfloat16(),
        )
        for name in single:
            differ = fused[name][0].view(torch.uint8) != single[name][0].view(
                torch.uint8
            )
            assert differ.float().mean() < 1e-5, (rank, name)
            assert (fused[name][1] != single[name][1]).float().mean() < 1e-4, (
                rank,
                name,
            )


def _pool_case(
    rows_per_expert=(300, 0, 517, 129), hidden=HIDDEN, inter=INNER, rank=RANK
):
    """Synthetic backward pools: experts packed at 128-row aligned starts, padded rows zero in h / dz."""
    counts = torch.tensor(rows_per_expert, device="cuda", dtype=torch.int32)
    padded = (counts + 127) // 128 * 128
    starts = torch.cumsum(padded, 0, dtype=torch.int32) - padded
    rows = int(padded.sum()) + 128
    gen = torch.Generator(device="cuda").manual_seed(3)
    pools = [
        torch.randn(rows, w, device="cuda", generator=gen).mul(s).bfloat16()
        for w, s in ((hidden, 1.0), (hidden, 0.01), (inter, 1.0), (2 * inter, 0.01))
    ]
    valid = torch.zeros(rows, dtype=torch.bool, device="cuda")
    for s, c in zip(starts.tolist(), counts.tolist(), strict=True):
        valid[s : s + c] = True
    for p in pools[2:]:
        p[~valid] = 0
    e = len(rows_per_expert)
    factors = (
        torch.randn(e, rank, hidden, device="cuda", generator=gen) / hidden**0.5,
        torch.randn(e, 2 * inter, rank, device="cuda", generator=gen) * 0.02,
        torch.randn(e, rank, inter, device="cuda", generator=gen) / inter**0.5,
        torch.randn(e, hidden, rank, device="cuda", generator=gen) * 0.02,
    )
    return pools, starts, counts, factors


@pytest.mark.parametrize("rank", [8, 16, 64])
def test_triton_adapter_grads_match_the_grouped_route(group, monkeypatch, rank):
    """BF16 pools: the two-pass Triton kernels against the per-expert reference loop."""
    from mono_ep.lora import lora_weight_grads
    from mono_ep.lora_grads import lora_weight_grads_triton

    pools, starts, counts, factors = _pool_case(rank=rank)
    monkeypatch.setenv("MONO_EP_LORA_WGRAD", "loop")
    want = lora_weight_grads(pools, starts, counts, *factors, SCALING)
    got = lora_weight_grads_triton(pools, starts, counts, *factors, SCALING)
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape
        assert rel(g, w) < 5e-3
