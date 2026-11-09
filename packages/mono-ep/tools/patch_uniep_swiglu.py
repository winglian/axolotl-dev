# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Repair gradient row addressing in the pinned MIT UniEP implementation."""

import hashlib
import importlib.util
from pathlib import Path

root = Path(importlib.util.find_spec("triton_dist").origin).parent
path = root / "kernels/nvidia/swiglu.py"
source = path.read_text()
expected = "7b35a846f2756ac2077e9f806b26dfa140f7dc3278a532245ea03372de2ac798"
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
normal_start = source.index("def _swiglu_backward_kernel(")
persistent_start = source.index("def _swiglu_backward_kernel_persistent(")
end = source.index("def swiglu_forward(")
normal = source[normal_start:persistent_start]
persistent = source[persistent_start:end]
normal = normal.replace(
    "    dC_ptr += row_idx * dC_row_stride\n",
    "    dC_ptr += row_idx * dC_row_stride\n"
    "    dA_ptr += row_idx * dA_row_stride\n"
    "    dB_ptr += row_idx * dB_row_stride\n",
)
persistent = persistent.replace(
    "    dC_ptr += row_start * dC_row_stride\n",
    "    dC_ptr += row_start * dC_row_stride\n"
    "    dA_ptr += row_start * dA_row_stride\n"
    "    dB_ptr += row_start * dB_row_stride\n",
).replace(
    "        dC_ptr += dC_row_stride\n",
    "        dC_ptr += dC_row_stride\n"
    "        dA_ptr += dA_row_stride\n"
    "        dB_ptr += dB_row_stride\n",
)
patched = source[:normal_start] + normal + persistent + source[end:]
assert patched.count("dA_ptr +=") == 3
assert patched.count("dB_ptr +=") == 3
path.write_text(patched)
print(
    "Patched UniEP SwiGLU gradient row addressing:",
    hashlib.sha256(path.read_bytes()).hexdigest(),
)
