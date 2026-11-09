# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Hardware selection without requiring the CUDA extension on the test host."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from mono_ep import MonoEP


@pytest.mark.parametrize(
    "capability,format", [((9, 0), "blockwise"), ((10, 0), "mxfp8")]
)
def test_fp8_hardware_selection(capability, format):
    factory = Mock()
    runtime = MonoEP(dtype="fp8", compile_local=True)
    with (
        patch("torch.cuda.get_device_capability", return_value=capability),
        patch.dict(
            sys.modules,
            {"mono_ep.fp8_autograd": SimpleNamespace(FP8Experts=factory)},
        ),
    ):
        first = runtime._local_compute(torch.device("cuda:0"))
        assert runtime._local_compute(torch.device("cuda:0")) is first
    factory.assert_called_once_with(format, compile_local=True)


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_unsupported_hardware_does_not_silently_fallback(dtype):
    with patch("torch.cuda.get_device_capability", return_value=(12, 0)):
        with pytest.raises(ValueError, match="SM90 and SM100"):
            MonoEP(dtype=dtype)._local_compute(torch.device("cuda:0"))


@pytest.mark.parametrize("chunks", [0, -1, 1.5, True])
def test_invalid_chunk_count(chunks):
    with pytest.raises(ValueError, match="positive integer"):
        MonoEP(chunks=chunks)


def test_cpu_inputs_fail_before_cuda_initialization():
    with pytest.raises(ValueError, match="CUDA BF16"):
        MonoEP()(
            torch.empty(1, 128, dtype=torch.bfloat16),
            None,
            None,
            None,
            None,
            group=None,
        )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"retain_activations": True}, "recomputation"),
        ({"chunks": 2}, "chunks=1"),
        ({"max_tokens": None}, "positive max_tokens"),
        ({"max_tokens": True}, "positive max_tokens"),
        ({"max_tokens": -1}, "positive max_tokens"),
        ({"compile_local": True}, "compile_local applies"),
    ],
)
def test_persistent_configuration_rejects_unsupported_combinations(options, match):
    config = dict(backend="persistent", retain_activations=False, max_tokens=128)
    config.update(options)
    with pytest.raises(ValueError, match=match):
        MonoEP(**config)


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_persistent_preparation_reuses_geometry_and_closes_handles(monkeypatch, dtype):
    from contextlib import nullcontext

    from mono_ep import runtime as persistent_module

    factory = Mock(side_effect=lambda *args, **kwargs: Mock())
    monkeypatch.setattr(persistent_module, "PersistentMoE", factory)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    engine = MonoEP(
        backend="persistent", dtype=dtype, retain_activations=False, max_tokens=128
    )
    group = object()
    geometry = dict(
        group=group, device="cuda:0", experts=8, topk=4, hidden=512, intermediate=512
    )
    first = engine.prepare_persistent(**geometry)
    assert engine.prepare_persistent(**geometry) is first
    factory.assert_called_once_with(
        group,
        8,
        128,
        4,
        512,
        512,
        compute_dtype="blockwise" if dtype == "fp8" else "bf16",
        retain_activations=False,
        backward_backend="native",
    )
    second = engine.prepare_persistent(**{**geometry, "intermediate": 768})
    assert second is not first
    engine.close()
    first.close.assert_called_once()
    second.close.assert_called_once()
    assert not engine._persistent


def test_pipeline_does_not_accept_persistent_capacity():
    with pytest.raises(ValueError, match="only to the persistent"):
        MonoEP(max_tokens=128)
