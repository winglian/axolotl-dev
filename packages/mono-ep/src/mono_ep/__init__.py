# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Fused expert-parallel MoE training kernels for Hopper and Blackwell."""

__all__ = [
    "DEFAULT_RESERVED_SMS",
    "FP8Experts",
    "PersistentMoE",
    "SonicExperts",
    "MonoEP",
    "MonoEPRuntime",
    "check_nccl_env",
    "nccl_env",
    "pipelined_moe",
    "resolve_precision",
    "sonic_experts",
    "sonic_fp8_experts",
    "unsupported_reason",
]

_API = frozenset(
    {
        "DEFAULT_RESERVED_SMS",
        "MonoEPRuntime",
        "check_nccl_env",
        "nccl_env",
        "resolve_precision",
        "unsupported_reason",
    }
)


def __getattr__(name):
    if name in _API:
        from . import api

        return getattr(api, name)
    if name == "MonoEP":
        from .mono_ep import MonoEP

        return MonoEP
    if name == "PersistentMoE":
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
