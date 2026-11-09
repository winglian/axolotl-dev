# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""CPU coverage for the framework integration surface."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from mono_ep import (
    MonoEPRuntime,
    check_nccl_env,
    nccl_env,
    resolve_precision,
    unsupported_reason,
)

SUPPORTED = dict(
    capability=(9, 0),
    hidden=2048,
    intermediate=768,
    has_bias=False,
    activation="silu",
    precision="auto",
)


@pytest.mark.parametrize(
    "precision,capability,expected",
    [
        ("auto", (9, 0), "bf16"),
        ("auto", (10, 0), "mxfp8"),
        ("fp8", (9, 0), "fp8"),
        ("bf16", (10, 0), "bf16"),
    ],
)
def test_resolve_precision(precision, capability, expected):
    assert resolve_precision(precision, capability) == expected


def test_supported_geometry_has_no_reason():
    assert unsupported_reason(**SUPPORTED) is None
    assert unsupported_reason(**{**SUPPORTED, "capability": (10, 0)}) is None


@pytest.mark.parametrize(
    "override,match",
    [
        ({"capability": (8, 0)}, "sm_80"),
        ({"capability": (12, 0)}, "sm_120"),
        ({"precision": "fp4"}, "unknown precision"),
        ({"precision": "fp8_bwd"}, "not implemented"),
        ({"precision": "mxfp8"}, "requires SM100"),
        ({"precision": "fp8", "capability": (10, 0)}, "requires SM90"),
        ({"hidden": 2048 + 128}, "multiple of 256"),
        ({"intermediate": 768 + 64}, "multiple of 128"),
        ({"has_bias": True}, "biases"),
        ({"activation": "gelu"}, "SwiGLU"),
        ({"activation_clamp": 7.0}, "clamped"),
    ],
)
def test_unsupported_reasons(override, match):
    reason = unsupported_reason(**{**SUPPORTED, **override})
    assert reason is not None and match in reason


@pytest.mark.parametrize("reserved,replicate,ctas", [(16, 1, 8), (16, 2, 5), (2, 4, 1)])
def test_nccl_env(reserved, replicate, ctas):
    assert nccl_env(reserved_sms=reserved, dp_replicate=replicate) == {
        "NCCL_MAX_CTAS": str(ctas)
    }


@pytest.mark.parametrize(
    "value,ok", [(None, False), ("x", False), ("0", False), ("8", True), ("9", False)]
)
def test_check_nccl_env(monkeypatch, value, ok):
    if value is None:
        monkeypatch.delenv("NCCL_MAX_CTAS", raising=False)
    else:
        monkeypatch.setenv("NCCL_MAX_CTAS", value)
    if ok:
        check_nccl_env(reserved_sms=16, dp_replicate=1)
    else:
        with pytest.raises(RuntimeError, match="NCCL_MAX_CTAS"):
            check_nccl_env(reserved_sms=16, dp_replicate=1)


def _fake_device(monkeypatch, arch, *, native=True, current=132):
    state = {"num_sms": current}

    def set_num_sms(value):
        state["num_sms"] = value

    c_ext = SimpleNamespace()
    if native:
        c_ext.bf16_mono_ep_backward_sm90 = object()
        c_ext.bf16_mono_ep_backward = object()
    deep_gemm = SimpleNamespace(
        _C=c_ext, get_num_sms=lambda: state["num_sms"], set_num_sms=set_num_sms
    )
    monkeypatch.setitem(sys.modules, "deep_gemm", deep_gemm)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (arch, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=132),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    persistent = Mock()
    persistent.return_value.buffer.num_max_tokens_per_rank = 4096
    persistent.return_value._peer_backward = None
    monkeypatch.setattr("mono_ep.runtime.PersistentMoE", persistent)
    return state, persistent


@pytest.mark.parametrize(
    "arch,precision,native,intermediate,backend,dtype",
    [
        (9, "auto", True, 768, "native", "bf16"),
        (10, "bf16", True, 768, "native", "bf16"),
        (10, "auto", True, 768, "nccl", "mxfp8"),
        (9, "bf16", False, 768, "nccl", "bf16"),
        (9, "bf16", True, 640, "nccl", "bf16"),
    ],
)
def test_runtime_lifecycle(
    monkeypatch, arch, precision, native, intermediate, backend, dtype
):
    state, persistent = _fake_device(monkeypatch, arch, native=native)
    runtime = MonoEPRuntime(
        group=None,
        num_experts=8,
        max_tokens_per_rank=4000,
        top_k=2,
        hidden=512,
        intermediate=intermediate,
        precision=precision,
    )
    assert state["num_sms"] == 116
    args, kwargs = persistent.call_args
    assert args == (None, 8, 4000, 2, 512, intermediate)
    assert kwargs == {
        "compute_dtype": dtype,
        "retain_activations": False,
        "backward_backend": backend,
        "reserved_sms": 16,
    }
    assert runtime.backward_backend == backend
    assert runtime.max_tokens_per_rank == 4096
    x = torch.zeros(3, 512, dtype=torch.bfloat16)
    runtime.forward(x, torch.zeros(3, 2, dtype=torch.long), torch.ones(3, 2), x, x)
    persistent.return_value.assert_called_once()
    runtime.destroy()
    runtime.destroy()
    persistent.return_value.close.assert_called_once()
    persistent.return_value.buffer.destroy.assert_called_once()
    assert state["num_sms"] == 132
    with pytest.raises(RuntimeError, match="after destroy"):
        runtime.forward(x, x, x, x, x)


def test_runtime_rejects_before_touching_sm_budget(monkeypatch):
    state, persistent = _fake_device(monkeypatch, 9)
    with pytest.raises(ValueError, match="multiple of 256"):
        MonoEPRuntime(
            group=None,
            num_experts=8,
            max_tokens_per_rank=128,
            top_k=2,
            hidden=384,
            intermediate=768,
        )
    assert state["num_sms"] == 132
    persistent.assert_not_called()


def test_runtime_restores_sm_budget_on_failure(monkeypatch):
    state, persistent = _fake_device(monkeypatch, 9)
    persistent.side_effect = ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        MonoEPRuntime(
            group=None,
            num_experts=8,
            max_tokens_per_rank=128,
            top_k=2,
            hidden=512,
            intermediate=768,
        )
    assert state["num_sms"] == 132
