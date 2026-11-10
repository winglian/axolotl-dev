# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""SM90 FP8 quantized-weight cache: bitwise forms, refresh after optimizer steps, invalidation."""

import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture
def cache():
    from mono_ep import weight_cache

    weight_cache.invalidate_weight_caches()
    weight_cache.set_optimizer_hook_enabled(True)
    weight_cache._EPOCH = None
    yield weight_cache
    weight_cache.invalidate_weight_caches()
    weight_cache.set_optimizer_hook_enabled(True)
    weight_cache._EPOCH = None


def _counting_forms(cache, w1, w2):
    built = []

    def build(missing):
        built.append(list(missing))
        return {name: (w1 if name == "a" else w2).clone() for name in ("a", "b")}

    forms = cache.cached_weight_forms(w1, w2, ["a"], build)
    return forms[0], built


def test_cache_contract_cpu(cache):
    """Version bumps miss, preserved-version writes need explicit invalidation (CPU)."""
    w1 = torch.nn.Parameter(torch.randn(2, 4, 4))
    w2 = torch.nn.Parameter(torch.randn(2, 4, 4))
    first, built = _counting_forms(cache, w1, w2)
    again, built_again = _counting_forms(cache, w1, w2)
    assert built == [["a"]] and built_again == [] and again is first
    # forms built alongside the requested one are served later without a build
    b = cache.cached_weight_forms(w1, w2, ["b"], lambda m: pytest.fail("rebuilt"))
    assert torch.equal(b[0], w2)

    with torch.no_grad():
        w1.add_(1.0)
    fresh, built = _counting_forms(cache, w1, w2)
    assert built == [["a"]] and torch.equal(fresh, w1)

    # A write that preserves the version counter (FSDP2 all-gather) is invisible to the key.
    with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(w1):
        w1.mul_(2.0)
    stale, built = _counting_forms(cache, w1, w2)
    assert built == [] and not torch.equal(stale, w1)
    cache.invalidate_weight_caches()
    fresh, built = _counting_forms(cache, w1, w2)
    assert built == [["a"]] and torch.equal(fresh, w1)

    # Caller epochs end the cache too.
    cache.set_weight_cache_epoch(1)
    _counting_forms(cache, w1, w2)
    with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(w1):
        w1.mul_(2.0)
    cache.set_weight_cache_epoch(2)
    fresh, built = _counting_forms(cache, w1, w2)
    assert built == [["a"]] and torch.equal(fresh, w1)


@pytest.mark.parametrize("hook", ["global", "optimizer", "none"])
def test_optimizer_step_hooks_cpu(cache, hook):
    w1 = torch.nn.Parameter(torch.randn(2, 4, 4))
    w2 = torch.nn.Parameter(torch.randn(2, 4, 4))
    optimizer = torch.optim.SGD([w1, w2], lr=0.5)
    cache.set_optimizer_hook_enabled(hook == "global")
    handle = (
        cache.register_optimizer_invalidation(optimizer)
        if hook == "optimizer"
        else None
    )
    _counting_forms(cache, w1, w2)
    w1.grad = torch.ones_like(w1)
    w2.grad = torch.ones_like(w2)
    before = cache.cache_stats()["invalidations"]
    optimizer.step()
    after = cache.cache_stats()["invalidations"]
    assert (after > before) == (hook != "none")
    # The in-place step also bumps the version, so the next call rebuilds either way.
    fresh, built = _counting_forms(cache, w1, w2)
    assert built == [["a"]] and torch.equal(fresh, w1)
    if handle is not None:
        handle.remove()


def _cuda_fp8():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 conversion requires SM89 or newer")
    pytest.importorskip("triton")


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(2, 1536, 2048), (3, 512, 384)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sliced", [False, True])
def test_fused_forms_bitwise(shape, dtype, sliced):
    _cuda_fp8()
    from mono_ep.sm90_weight_forms import quantize_all_forms, quantize_form

    torch.manual_seed(5)
    experts, twice_inner, hidden = shape
    inner = twice_inner // 2
    w1 = torch.randn(experts, twice_inner, hidden * (2 if sliced else 1), device="cuda")
    w2 = torch.randn(experts, hidden, inner * (2 if sliced else 1), device="cuda")
    if sliced:
        w1, w2 = w1[:, :, ::2], w2[:, :, ::2]
    w1, w2 = w1.to(dtype), w2.to(dtype)
    # distinct amax in the two 64-row halves of a paired block, a zero block, large outliers
    w1[0, :64] *= 64
    w1[0, inner + 64 : inner + 128] *= 0.01
    w1[-1, :128, :128] = 0
    w1[-1, inner : inner + 128, :128] = 0
    w2[0, :128, :128] *= 300
    forms = quantize_all_forms(w1, w2)
    for name, (values, scales) in forms.items():
        expected_values, expected_scales = quantize_form(w1, w2, name)
        assert values.shape == expected_values.shape, name
        assert torch.equal(
            values.view(torch.uint8), expected_values.view(torch.uint8)
        ), name
        assert scales.shape == expected_scales.shape, name
        assert torch.equal(scales, expected_scales), name


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["1", "forms"])
def test_cached_forms_follow_optimizer_steps(cache, mode):
    """Cached forms equal fresh ones after in-place optimizer steps and explicit invalidation."""
    _cuda_fp8()
    from mono_ep.sm90_weight_forms import (
        BACKWARD_FORMS,
        FORWARD_FORMS,
        cached_forms,
        quantize_form,
    )

    torch.manual_seed(9)
    w1 = torch.nn.Parameter(torch.randn(4, 512, 256, device="cuda"))
    w2 = torch.nn.Parameter(torch.randn(4, 256, 256, device="cuda"))
    optimizer = torch.optim.AdamW([w1, w2], lr=1e-1)
    cache.set_optimizer_hook_enabled(False)

    def check():
        for names in (FORWARD_FORMS, BACKWARD_FORMS):
            for name, (values, scales) in zip(
                names, cached_forms(w1, w2, names, mode), strict=True
            ):
                expected = quantize_form(w1.detach(), w2.detach(), name)
                assert torch.equal(
                    values.view(torch.uint8), expected[0].view(torch.uint8)
                ), name
                assert torch.equal(scales, expected[1]), name

    for _ in range(2):
        check()
        builds = cache.cache_stats()["builds"]
        check()
        assert cache.cache_stats()["builds"] == builds
        w1.grad = torch.randn_like(w1)
        w2.grad = torch.randn_like(w2)
        optimizer.step()
    check()
    with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(w2):
        w2.mul_(-3.0)
    cache.invalidate_weight_caches()
    check()


@pytest.mark.gpu
@pytest.mark.slow
def test_fsdp2_whole_layer_two_steps():
    """2xH100: FSDP2 whole-layer sharding, two AdamW steps, cached vs uncached kernels bitwise."""
    if torch.cuda.device_count() < 2 or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs two SM90 GPUs")
    deep_gemm = pytest.importorskip("deep_gemm")
    if not hasattr(deep_gemm._C, "fp8_mono_ep_backward_sm90"):
        pytest.skip("vendored DeepGEMM with the MonoEP SM90 kernels is not built")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(root / "examples/validate_fsdp2_sm90_weight_cache.py"),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src"),
            "MONO_EP_SM90_FP8_TRANSPORT": "1",
            "MONO_EP_SM90_FP8_COMBINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = next(
        x for x in result.stdout.splitlines() if x.startswith("FSDP2_CACHE_JSON ")
    )
    assert "false" not in line, line
