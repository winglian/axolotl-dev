# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""NVFP4 merge-aware requant kernel and per-step cache (temporary copy).

Copied from Axolotl ``src/axolotl/integrations/kernels/libs/sonicmoe/``
(``nvfp4_requant.py``, ``triton_nvfp4_requant.py``, ``nvfp4_requant_cache.py``,
``sf_layout.py`` and the two ``triton_nvfp4`` helpers) at commit 227adde535 of
``features/nvfp4-requant-default``, so that ``mono_ep`` keeps no Axolotl import.
The follow-up replaces the Axolotl modules with imports from this package;
until then, kernel-layout and transposed-output additions land here, not in
Axolotl.
"""

from . import (
    cuda_nvfp4_requant,
    nvfp4_requant,
    nvfp4_requant_cache,
    triton_nvfp4_requant,
)
from .nvfp4_requant import (
    RequantBuffers,
    RequantNumerics,
    RequantResult,
    dequantize_nvfp4_packed_ref,
    per_tensor_scale_from_amax,
    quantize_nvfp4_packed_ref,
)
from .nvfp4_requant_cache import RequantCache, RequantKey, attach_post_step_hook
from .triton_nvfp4_requant import requant_lora_packed

__all__ = [
    "RequantBuffers",
    "RequantCache",
    "RequantKey",
    "RequantNumerics",
    "RequantResult",
    "attach_post_step_hook",
    "cuda_nvfp4_requant",
    "dequantize_nvfp4_packed_ref",
    "nvfp4_requant",
    "nvfp4_requant_cache",
    "per_tensor_scale_from_amax",
    "quantize_nvfp4_packed_ref",
    "requant_lora_packed",
    "triton_nvfp4_requant",
]
