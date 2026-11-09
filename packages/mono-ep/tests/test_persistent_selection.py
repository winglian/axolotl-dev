# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Backend selection fails explicitly before allocating unsupported workspaces."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from mono_ep import PersistentMoE
from mono_ep.runtime import _RUNTIMES, _check_sm_budget


@pytest.mark.parametrize("current,allocated", [(116, 116), (132, 116), (116, 132)])
def test_workspace_sm_budget_is_frozen(monkeypatch, current, allocated):
    monkeypatch.setitem(
        sys.modules, "deep_gemm", SimpleNamespace(get_num_sms=lambda: current)
    )
    runtime = SimpleNamespace(num_sms=116, buffer=SimpleNamespace(num_sms=allocated))
    if current == allocated == 116:
        _check_sm_budget(runtime)
    else:
        with pytest.raises(RuntimeError, match="SM count changed"):
            _check_sm_budget(runtime)


@pytest.mark.parametrize(
    "arch,reserved,current,match",
    [
        (9, -1, 132, "leave at least two"),
        (9, 131, 1, "leave at least two"),
        (10, 15, 117, "even active"),
        (9, 16, 132, "set_num_sms"),
    ],
)
def test_invalid_sm_reservation_before_allocation(
    monkeypatch, arch, reserved, current, match
):
    allocate = Mock()
    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(
            get_num_sms=lambda: current,
            get_symm_buffer_for_mono_ep=allocate,
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (arch, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=132),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    with pytest.raises(ValueError, match=match):
        PersistentMoE(None, 8, 128, 4, 512, 512, reserved_sms=reserved)
    allocate.assert_not_called()


@pytest.mark.parametrize(
    "options,match",
    [
        ({"backward_backend": "unknown"}, "backward_backend"),
        (
            {"compute_dtype": "blockwise", "retain_activations": True},
            "requires recomputation",
        ),
        ({"backward_backend": "native", "compute_dtype": "mxfp8"}, "recomputing BF16"),
        (
            {"backward_backend": "native", "retain_activations": True},
            "recomputing BF16",
        ),
    ],
)
def test_invalid_selection_before_extension_import(monkeypatch, options, match):
    monkeypatch.setitem(sys.modules, "deep_gemm", None)
    with pytest.raises(ValueError, match=match):
        PersistentMoE(None, 8, 128, 4, 512, 512, **options)


@pytest.mark.parametrize("backend", ["nccl", "native"])
@pytest.mark.parametrize("arch", [9, 10])
@pytest.mark.parametrize("reserved_sms", [0, 16])
def test_constructor_selects_backend_and_close_releases_handle(
    monkeypatch, backend, arch, reserved_sms
):
    native = Mock()
    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(
            get_mk_alignment_for_contiguous_layout=lambda: 128,
            get_num_sms=lambda: 132 - reserved_sms,
            get_symm_buffer_for_mono_ep=Mock(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mono_ep.persistent_backward",
        SimpleNamespace(NativePersistentBackward=native),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (arch, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=132),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    runtime = PersistentMoE(
        None,
        8,
        128,
        4,
        512,
        512,
        backward_backend=backend,
        reserved_sms=reserved_sms,
    )
    try:
        assert runtime.num_sms == 132 - reserved_sms
        assert _RUNTIMES[runtime._key] is runtime
        if backend == "native":
            native.assert_called_once_with(runtime)
            assert runtime._peer_backward is native.return_value
        else:
            native.assert_not_called()
            assert runtime._peer_backward is None
    finally:
        runtime.close()
    assert runtime._key not in _RUNTIMES


@pytest.mark.parametrize(
    "dtype,arch,message",
    [("mxfp8", 9, "mxfp8.*SM100"), ("blockwise", 10, "blockwise.*SM90")],
)
def test_fp8_format_rejected_on_wrong_architecture(monkeypatch, dtype, arch, message):
    monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (arch, 0))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    with pytest.raises(ValueError, match=message):
        PersistentMoE(
            None,
            8,
            128,
            4,
            512,
            512,
            compute_dtype=dtype,
            backward_backend="native" if dtype == "blockwise" else "nccl",
        )
