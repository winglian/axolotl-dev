# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Experimental DeepGEMM training primitives; not yet a production EP backend."""

__all__ = [
    "FP8Experts",
    "MonoEP",
    "PersistentMoE",
    "SonicExperts",
    "MonoEP",
    "pipelined_moe",
    "sonic_experts",
    "sonic_fp8_experts",
]


def __getattr__(name):
    if name == "MonoEP":
        from .mono_ep import MonoEP

        return MonoEP
    if name in ("PersistentMoE", "MonoEP"):
        from .runtime import PersistentMoE

        return PersistentMoE
    if name == "FP8Experts":
        from .fp8_autograd import FP8Experts

        return FP8Experts
    if name == "SonicExperts":
        from .sonic import SonicExperts

        return SonicExperts
    if name == "pipelined_moe":
        from .pipeline import pipelined_moe

        return pipelined_moe
    if name == "sonic_experts":
        from .sonic import sonic_experts

        return sonic_experts
    if name == "sonic_fp8_experts":
        from .sonic import sonic_fp8_experts

        return sonic_fp8_experts
    raise AttributeError(name)
