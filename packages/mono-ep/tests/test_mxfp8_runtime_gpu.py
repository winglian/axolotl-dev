# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The SM100 MXFP8 runtime runs the fused MXFP8 backward and shares its weight forms with the forward."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]

HIDDEN, INNER, EXPERTS, TOPK, TOKENS = 1024, 512, 8, 4, 256


@pytest.fixture
def group(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the MXFP8 runtime needs SM100")
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


def _runtime(group, compute_dtype, hidden=HIDDEN):
    import deep_gemm
    from mono_ep.runtime import PersistentMoE

    deep_gemm.set_num_sms(torch.cuda.get_device_properties(0).multi_processor_count)
    return PersistentMoE(
        group,
        EXPERTS,
        TOKENS,
        TOPK,
        hidden,
        INNER,
        compute_dtype=compute_dtype,
        fuse_activation=True,
        backward_backend="native",
    )


def _step(runtime, inputs):
    x, ids, scores, w1, w2, dy = inputs
    x, scores = x.clone().requires_grad_(), scores.clone().requires_grad_()
    w1, w2 = w1.clone().requires_grad_(), w2.clone().requires_grad_()
    y = runtime(x, ids, scores, w1, w2)
    y.backward(dy)
    torch.cuda.synchronize()
    return y.detach(), x.grad, w1.grad, w2.grad


def _inputs(hidden=HIDDEN):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(TOKENS, hidden, device="cuda", generator=gen).bfloat16()
    ids = torch.rand(TOKENS, EXPERTS, device="cuda", generator=gen).argsort(-1)[
        :, :TOPK
    ]
    scores = torch.rand(TOKENS, TOPK, device="cuda", generator=gen).softmax(-1)
    w1 = (
        torch.randn(EXPERTS, 2 * INNER, hidden, device="cuda", generator=gen)
        / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(EXPERTS, hidden, INNER, device="cuda", generator=gen) / INNER**0.5
    ).bfloat16()
    dy = torch.randn(TOKENS, hidden, device="cuda", generator=gen).bfloat16()
    return x, ids.contiguous(), scores, w1, w2, dy


def test_mxfp8_runtime_takes_the_fused_mxfp8_backward(group, monkeypatch):
    monkeypatch.delenv("MONO_EP_SM100_BACKWARD_PRECISION", raising=False)
    monkeypatch.delenv("MONO_EP_SM100_DW_PRECISION", raising=False)
    inputs = _inputs()
    mx = _runtime(group, "mxfp8")
    assert mx._peer_backward.precision() == "mxfp8"
    y, dx, dw1, _ = _step(mx, inputs)
    assert all(torch.isfinite(t.float()).all() for t in (y, dx, dw1))
    # the same backward kernel on the same operands as the BF16 runtime's MXFP8 backward
    bf = _runtime(group, "bf16")
    monkeypatch.setenv("MONO_EP_SM100_BACKWARD_PRECISION", "mxfp8")
    _, dx_bf, _, _ = _step(bf, inputs)
    assert torch.equal(dx, dx_bf)
    monkeypatch.delenv("MONO_EP_SM100_BACKWARD_PRECISION")
    _, dx_again, _, _ = _step(mx, inputs)
    assert torch.equal(dx, dx_again)
    for runtime in (mx, bf):
        runtime.close()


def _launched_dw(monkeypatch):
    import deep_gemm

    flags = []
    native = deep_gemm._C.mxfp8_mono_ep_backward

    def record(*args):
        flags.append(args[-1])
        return native(*args)

    monkeypatch.setattr(deep_gemm._C, "mxfp8_mono_ep_backward", record)
    return flags


def test_mxfp8_dw_follows_the_runtime_default_and_the_selector(group, monkeypatch):
    monkeypatch.delenv("MONO_EP_SM100_BACKWARD_PRECISION", raising=False)
    monkeypatch.delenv("MONO_EP_SM100_DW_PRECISION", raising=False)
    monkeypatch.setenv("MONO_EP_STORED_CONTEXT", "off")
    flags = _launched_dw(monkeypatch)
    inputs = _inputs()
    mx = _runtime(group, "mxfp8")
    # precision mxfp8 defaults to MXFP8 dW
    _, dx, dw1, dw2 = _step(mx, inputs)
    monkeypatch.setenv("MONO_EP_SM100_DW_PRECISION", "mxfp8")
    _, dx_env, dw1_env, dw2_env = _step(mx, inputs)
    # the selector restores BF16 dW
    monkeypatch.setenv("MONO_EP_SM100_DW_PRECISION", "bf16")
    _, dx_off, dw1_off, dw2_off = _step(mx, inputs)
    assert flags == [True, True, False]
    # dx never depends on the dW precision
    assert torch.equal(dx, dx_env) and torch.equal(dx, dx_off)

    def gap(got, want):
        return (got.float() - want.float()).norm() / want.float().norm()

    # dW pool row order varies run to run, so one path repeats to rounding, not bytes
    for same in ((dw1, dw1_env), (dw2, dw2_env)):
        assert gap(*same) < 1e-3
    for got, want in ((dw1, dw1_off), (dw2, dw2_off)):
        assert torch.isfinite(got).all()
        assert 1e-3 < gap(got, want) < 0.1
    # the BF16 runtime's opt-in MXFP8 backward keeps BF16 dW by default
    monkeypatch.delenv("MONO_EP_SM100_DW_PRECISION")
    monkeypatch.setenv("MONO_EP_SM100_BACKWARD_PRECISION", "mxfp8")
    bf = _runtime(group, "bf16")
    _step(bf, inputs)
    assert flags[-1] is False
    for runtime in (mx, bf):
        runtime.close()


def test_mxfp8_forward_forms_are_the_backward_and_direct_bytes(group):
    from mono_ep import weight_cache
    from mono_ep.token_quantization import prepare_persistent_mxfp8

    _, _, _, w1, w2, _ = _inputs()
    forward = weight_cache.mxfp8_forward_weight_forms(w1, w2)
    backward = weight_cache.mxfp8_backward_weight_forms(w1, w2)
    direct = (
        prepare_persistent_mxfp8(w1, weights=True, interleave=True),
        prepare_persistent_mxfp8(w2, weights=True),
    )
    assert forward[0][0] is backward[0][0]
    for got, want in zip(forward, direct, strict=True):
        assert torch.equal(got[0].view(torch.uint8), want[0].view(torch.uint8))
        assert torch.equal(got[1], want[1]) and got[1].stride() == want[1].stride()


def test_mxfp8_runtime_falls_back_to_bf16_backward_off_512(group, monkeypatch, caplog):
    monkeypatch.delenv("MONO_EP_SM100_BACKWARD_PRECISION", raising=False)
    mx = _runtime(group, "mxfp8", hidden=768)
    with caplog.at_level("WARNING", logger="mono_ep"):
        assert mx._peer_backward.precision() == "bf16"
        y, dx, _, _ = _step(mx, _inputs(hidden=768))
    assert "BF16 persistent backward" in caplog.text
    assert "odd multiple of 256" in caplog.text
    assert torch.isfinite(dx.float()).all() and torch.isfinite(y.float()).all()
    mx.close()


def test_lazy_bf16_pools_stay_unallocated_under_mxfp8_dw(group, monkeypatch):
    """Lazy BF16 pools: the MXFP8 backward with MXFP8 dW never allocates the BF16 x / dy / dz copies
    (dx bitwise, dW to run-to-run rounding against eager pools), BF16 dW allocates them on demand,
    and a launch that would write them while they are absent fails in the launcher."""
    monkeypatch.delenv("MONO_EP_SM100_BACKWARD_PRECISION", raising=False)
    monkeypatch.delenv("MONO_EP_SM100_DW_PRECISION", raising=False)
    monkeypatch.setenv("MONO_EP_STORED_CONTEXT", "off")
    inputs = _inputs()

    def gap(got, want):
        return (got.float() - want.float()).norm() / want.float().norm()

    monkeypatch.setenv("MONO_EP_SM100_BF16_POOLS", "eager")
    eager = _runtime(group, "mxfp8")
    _, dx_eager, dw1_eager, dw2_eager = _step(eager, inputs)
    assert eager._peer_backward.pools[0].numel() > 0
    eager.close()

    monkeypatch.setenv("MONO_EP_SM100_BF16_POOLS", "lazy")
    lazy = _runtime(group, "mxfp8")
    native = lazy._peer_backward
    _, dx, dw1, dw2 = _step(lazy, inputs)
    assert all(native.pools[i].numel() == 0 for i in (0, 1, 3))
    assert native.pools[2].numel() > 0
    assert torch.equal(dx, dx_eager)
    assert gap(dw1, dw1_eager) < 1e-3 and gap(dw2, dw2_eager) < 1e-3

    # the launcher refuses absent copies when the launch would write them
    monkeypatch.setenv("MONO_EP_SM100_DW_PRECISION", "bf16")
    ensure = native._ensure_copies
    monkeypatch.setattr(native, "_ensure_copies", lambda: None)
    with pytest.raises(RuntimeError):
        _step(lazy, inputs)
    monkeypatch.setattr(native, "_ensure_copies", ensure)
    _, dx_bf, dw1_bf, _ = _step(lazy, inputs)
    assert all(native.pools[i].numel() > 0 for i in (0, 1, 3))
    assert torch.equal(dx_bf, dx_eager)
    assert torch.isfinite(dw1_bf).all()
    lazy.close()
