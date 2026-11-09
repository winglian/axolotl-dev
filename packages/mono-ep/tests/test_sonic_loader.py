# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The standard adapter resolves the Transformers Hub kernel without sonicmoe."""

import sys
from types import ModuleType, SimpleNamespace

import torch


def test_standard_sonic_uses_hub_loader(monkeypatch):
    from mono_ep.sonic import SonicExperts

    calls = []
    activation = object()

    def forward(*args, **kwargs):
        calls.append((args, kwargs))
        return args[0], None

    kernel = SimpleNamespace(
        moe_general_routing_inputs=forward,
        activation_type_enum=SimpleNamespace(SWIGLU=activation),
    )
    loads = []

    def load():
        loads.append(True)
        return kernel

    for name in ("transformers", "transformers.integrations"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    module = ModuleType("transformers.integrations.sonicmoe")
    module.load_sonicmoe_kernel = load
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "sonicmoe", None)
    compute = SonicExperts()
    assert len(loads) == 1
    x = torch.ones(3, 128, dtype=torch.bfloat16)
    ids = torch.zeros(3, 1, dtype=torch.int32)
    scores = torch.ones(3, 1)
    w1 = torch.ones(1, 256, 128)
    w2 = torch.ones(1, 128, 128)
    for _ in range(2):
        torch.testing.assert_close(compute(x, ids, scores, w1, w2), x)
    assert len(calls) == 2
    assert calls[0][0][10] is activation
    assert calls[0][1]["concat_layout"] is True
    assert "use_fp8" not in calls[0][1]
