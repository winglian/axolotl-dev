# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Experimental SM100 owned-block backward math, before peer transport integration."""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _load_kernel():
    import deep_gemm
    from torch.utils.cpp_extension import load

    include = Path(__file__).parent / "include"
    return load(
        name="mono_ep_owned_backward",
        sources=[
            str(include / "owned_backward_bindings.cpp"),
            str(include / "owned_backward_launch.cu"),
        ],
        extra_include_paths=[
            str(include),
            str(Path(deep_gemm.__file__).parent / "include"),
        ],
        extra_cuda_cflags=[
            "-O2",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
    )


@torch.library.custom_op("mono_ep::owned_backward", mutates_args=())
def owned_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    scores: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Emit weighted h, dZ, dX and dscore for GPU-count expert-packed rows."""
    _validate(x, dy, w1, w2, scores, starts, counts)
    if torch.cuda.get_device_capability(x.device)[0] != 10:
        raise ValueError("Owned backward requires SM100")
    rows, hidden = x.shape
    experts, twice_inner, _ = w1.shape
    inner = twice_inner // 2
    h = x.new_zeros((rows, inner))
    dz = x.new_zeros((rows, twice_inner))
    dx = torch.zeros_like(x)
    ds = torch.zeros_like(scores)
    torch._assert_async(
        (
            (counts >= 0) & (starts >= 0) & (starts <= rows) & (counts <= rows - starts)
        ).all(),
        "Expert rows exceed packed storage",
    )
    torch._assert_async(
        (counts[:-1] <= starts[1:] - starts[:-1]).all(),
        "Expert ranges overlap",
    )
    ends = ((counts + 63) // 64).cumsum(0).int()
    queue = torch.zeros(1, device=x.device, dtype=torch.int32)
    workers = max(
        1, torch.cuda.get_device_properties(x.device).multi_processor_count - 16
    )
    tensors = [x, dy, w1, w2, scores, ends, starts, counts, queue, h, dz, dx, ds]
    with torch.cuda.device(x.device):
        error = _load_kernel().launch(
            [t.data_ptr() for t in tensors],
            experts,
            hidden,
            inner,
            workers,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if error:
        raise RuntimeError(f"Owned backward CUDA launch failed with error {error}")
    return h, dz, dx, ds


def _validate(x, dy, w1, w2, scores, starts, counts):
    if x.ndim != 2 or w1.ndim != 3:
        raise ValueError("Expected packed rows and grouped expert weights")
    rows, hidden = x.shape
    experts, twice_inner, weight_hidden = w1.shape
    if (
        experts < 1
        or rows >= 2**31 - 64
        or hidden < 64
        or hidden % 64
        or twice_inner < 128
        or twice_inner % 128
        or weight_hidden != hidden
        or dy.shape != x.shape
        or w2.shape != (experts, hidden, twice_inner // 2)
        or scores.shape != (rows,)
        or starts.shape != (experts,)
        or counts.shape != (experts,)
        or x.device.type != "cuda"
        or any(t.dtype != torch.bfloat16 for t in (x, dy, w1, w2))
        or scores.dtype != torch.float32
        or starts.dtype != torch.int32
        or counts.dtype != torch.int32
        or any(
            t.device != x.device or not t.is_contiguous()
            for t in (x, dy, w1, w2, scores, starts, counts)
        )
    ):
        raise ValueError("Invalid SM100 owned-backward operands or expert geometry")


@owned_backward.register_fake
def _fake(x, dy, w1, w2, scores, starts, counts):
    _validate(x, dy, w1, w2, scores, starts, counts)
    return (
        x.new_empty((x.shape[0], w1.shape[1] // 2)),
        x.new_empty((x.shape[0], w1.shape[1])),
        torch.empty_like(x),
        torch.empty_like(scores),
    )
