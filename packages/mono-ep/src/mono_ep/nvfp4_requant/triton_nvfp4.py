# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""The two helpers ``triton_nvfp4_requant`` takes from Axolotl's ``triton_nvfp4``."""

import torch


def triton_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except ImportError:
        return False
    return True


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
