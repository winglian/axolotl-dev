# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compiler boundary for device-ready, aligned expert chunks."""

import torch


def _check_metadata(x, w, out, queue, task):
    if x.ndim != 2 or w.ndim != 3 or out.ndim != 2:
        raise ValueError("Expected x[M,K], w[E,N,K], and out[M,N]")
    if x.shape[1] != w.shape[2] or out.shape != (x.shape[0], w.shape[1]):
        raise ValueError("Incompatible chunk GEMM shapes")
    if any(t.dtype != torch.bfloat16 for t in (x, w, out)):
        raise ValueError("Chunk GEMM requires BF16 inputs, weights, and output")
    if queue.ndim != 2 or queue.shape[1] != 4 or queue.dtype != torch.int32:
        raise ValueError("Queue must be int32[num_tasks, 4]")
    if task < 0 or task >= queue.shape[0]:
        raise ValueError("Task index is outside the queue")
    if x.device.type != "cuda" or any(t.device != x.device for t in (w, out, queue)):
        raise ValueError("All chunk GEMM tensors must be on the same CUDA device")
    if not queue.is_contiguous():
        raise ValueError("Queue must be contiguous")


@torch.library.custom_op("mono_ep::chunk_bf16_out", mutates_args=("out",))
def chunk_bf16_out(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    queue: torch.Tensor,
    task: int,
) -> None:
    """Compute one ready chunk, preserving output rows outside its range.

    Queue rows contain expert, row start, padded row count, and readiness.
    The producer must publish with system-release after writing input data.
    Inputs and descriptors must remain alive and stable until this stream
    finishes. Row start/count must align to DeepGEMM's configured M block.
    Reserve SM capacity for any concurrent GPU producer before launching.
    This low-level operation has no autograd formula; training callers must
    explicitly schedule forward and backward GEMMs.
    """
    _check_metadata(x, w, out, queue, task)
    import deep_gemm

    deep_gemm.bf16_chunk_gemm_nt(x, w, out, queue, task)


@chunk_bf16_out.register_fake
def _chunk_bf16_out_fake(x, w, out, queue, task):
    _check_metadata(x, w, out, queue, task)


def _check_backward_metadata(dy, wt, h, queue, task, z, scores, dz, partial):
    _check_metadata(dy, wt, h, queue, task)
    _check_swiglu_metadata(h, z, scores, dz, partial)


def _check_swiglu_metadata(h, z, scores, dz, partial):
    rows, inner = h.shape
    if inner % 8 or z.shape != (rows, 2 * inner) or dz.shape != z.shape:
        raise ValueError("Invalid SwiGLU gate/up gradient shapes")
    if scores.shape != (rows,) or partial.shape != (rows, inner // 8):
        raise ValueError("Invalid SwiGLU score gradient shapes")
    if z.dtype != torch.bfloat16 or dz.dtype != torch.bfloat16:
        raise ValueError("SwiGLU gate/up tensors must be BF16")
    if scores.dtype != torch.float32 or partial.dtype != torch.float32:
        raise ValueError("SwiGLU scores and partial gradients must be FP32")
    if any(
        t.device != h.device or not t.is_contiguous() for t in (z, scores, dz, partial)
    ):
        raise ValueError("SwiGLU tensors must be contiguous on the GEMM device")


@torch.library.custom_op(
    "mono_ep::chunk_swiglu_backward_out", mutates_args=("h", "dz", "partial")
)
def chunk_swiglu_backward_out(
    dy: torch.Tensor,
    wt: torch.Tensor,
    h: torch.Tensor,
    queue: torch.Tensor,
    task: int,
    z: torch.Tensor,
    scores: torch.Tensor,
    dz: torch.Tensor,
    partial: torch.Tensor,
) -> None:
    """Fuse queued dH GEMM, weighted activation recompute, dZ and score partials.

    wt is the transposed down-projection weight [E,I,H]. h[M,I], dz[M,2I]
    and partial[M,I/8] are output buffers; only the queued row range changes.
    Sum partial over its last dimension to obtain routing-score gradients.
    The readiness and lifetime contract is the same as chunk_bf16_out.
    """
    _check_backward_metadata(dy, wt, h, queue, task, z, scores, dz, partial)
    import deep_gemm

    epilogue = deep_gemm.epilogue.SwiGLUBackward(z, scores, dz, partial)
    deep_gemm.bf16_chunk_gemm_nt(dy, wt, h, queue, task, epilogue=epilogue)


@chunk_swiglu_backward_out.register_fake
def _chunk_swiglu_backward_out_fake(dy, wt, h, queue, task, z, scores, dz, partial):
    _check_backward_metadata(dy, wt, h, queue, task, z, scores, dz, partial)


def _check_fp8_metadata(x, sx, w, sw, out, queue, task, gran_k):
    if x.dtype != torch.float8_e4m3fn or w.dtype != torch.float8_e4m3fn:
        raise ValueError("Queued FP8 GEMM requires E4M3 operands")
    if sx.dtype != torch.int32 or sw.dtype != torch.int32 or gran_k not in (32, 128):
        raise ValueError(
            "Queued FP8 GEMM requires packed UE8M0 scales and K granularity 32/128"
        )
    if x.ndim != 2 or w.ndim != 3 or out.shape != (x.shape[0], w.shape[1]):
        raise ValueError("Expected x[M,K], w[E,N,K], and out[M,N]")
    if x.shape[1] != w.shape[2] or out.dtype != torch.bfloat16:
        raise ValueError("Invalid FP8 GEMM dimensions or output dtype")
    if x.device.type != "cuda" or any(
        t.device != x.device for t in (sx, w, sw, out, queue)
    ):
        raise ValueError("All operands must be on the same CUDA device")
    if queue.ndim != 2 or queue.shape[1] != 4 or queue.dtype != torch.int32:
        raise ValueError("Queue must be int32[num_tasks, 4]")
    if not queue.is_contiguous() or task < 0 or task >= queue.shape[0]:
        raise ValueError("Invalid queue layout or task index")


@torch.library.custom_op("mono_ep::chunk_fp8_out", mutates_args=("out",))
def chunk_fp8_out(
    x: torch.Tensor,
    sx: torch.Tensor,
    w: torch.Tensor,
    sw: torch.Tensor,
    out: torch.Tensor,
    queue: torch.Tensor,
    task: int,
    gran_k: int = 32,
) -> None:
    """Compute a ready E4M3 chunk using prepacked UE8M0 scales on SM100.

    The producer must publish readiness after both operand values and scales
    are visible. Scale packing belongs to the producer, before publication.
    Queue alignment, lifetime and SM reservation match chunk_bf16_out.
    This primitive requires explicitly scheduled backward operations.
    """
    _check_fp8_metadata(x, sx, w, sw, out, queue, task, gran_k)
    import deep_gemm

    deep_gemm.fp8_chunk_gemm_nt((x, sx), (w, sw), out, queue, task, gran_k)


@chunk_fp8_out.register_fake
def _chunk_fp8_out_fake(x, sx, w, sw, out, queue, task, gran_k=32):
    _check_fp8_metadata(x, sx, w, sw, out, queue, task, gran_k)


@torch.library.custom_op(
    "mono_ep::chunk_fp8_swiglu_backward_out", mutates_args=("h", "dz", "partial")
)
def chunk_fp8_swiglu_backward_out(
    dy: torch.Tensor,
    sy: torch.Tensor,
    wt: torch.Tensor,
    sw: torch.Tensor,
    h: torch.Tensor,
    queue: torch.Tensor,
    task: int,
    z: torch.Tensor,
    scores: torch.Tensor,
    dz: torch.Tensor,
    partial: torch.Tensor,
    gran_k: int = 32,
) -> None:
    """Fuse queued E4M3 dH GEMM with BF16 SwiGLU backward and score partials.

    Value/scale readiness follows chunk_fp8_out. Saved gate/up activations and
    scores must also be ready before publication and live through consumption.
    """
    _check_fp8_metadata(dy, sy, wt, sw, h, queue, task, gran_k)
    _check_swiglu_metadata(h, z, scores, dz, partial)
    import deep_gemm

    epilogue = deep_gemm.epilogue.SwiGLUBackward(z, scores, dz, partial)
    deep_gemm.fp8_chunk_gemm_nt(
        (dy, sy), (wt, sw), h, queue, task, gran_k, epilogue=epilogue
    )


@chunk_fp8_swiglu_backward_out.register_fake
def _chunk_fp8_swiglu_backward_out_fake(
    dy, sy, wt, sw, h, queue, task, z, scores, dz, partial, gran_k=32
):
    _check_fp8_metadata(dy, sy, wt, sw, h, queue, task, gran_k)
    _check_swiglu_metadata(h, z, scores, dz, partial)


@torch.library.custom_op("mono_ep::ready_grouped_fp8_out", mutates_args=("out",))
def ready_grouped_fp8_out(
    x: torch.Tensor,
    sx: torch.Tensor,
    w: torch.Tensor,
    sw: torch.Tensor,
    out: torch.Tensor,
    queue: torch.Tensor,
    expert_ids: torch.Tensor,
    incremental: bool = False,
) -> None:
    """Acquire a complete FP8 payload inside one grouped expert GEMM.

    All payload and expert-ID writes must precede publication. The producer must
    publish every queue row; zero-padded output storage belongs to the caller.
    Hopper requires pretransformed FP32 1x128 activation and 128x128 weight
    scales and incremental scheduling. SM100 requires packed UE8M0 scales.
    """
    _ready_grouped_fp8_fake(x, sx, w, sw, out, queue, expert_ids)
    import deep_gemm

    deep_gemm.fp8_ready_grouped_gemm_nt(
        (x, sx), (w, sw), out, expert_ids, queue, incremental=incremental
    )


@ready_grouped_fp8_out.register_fake
def _ready_grouped_fp8_fake(x, sx, w, sw, out, queue, expert_ids, incremental=False):
    if sx.dtype == torch.float32 and sw.dtype == torch.float32:
        if x.dtype != torch.float8_e4m3fn or w.dtype != torch.float8_e4m3fn:
            raise ValueError("Queued FP8 GEMM requires E4M3 operands")
        if x.ndim != 2 or w.ndim != 3 or out.shape != (x.shape[0], w.shape[1]):
            raise ValueError("Expected x[M,K], w[E,N,K], and out[M,N]")
        if x.shape[1] != w.shape[2] or out.dtype != torch.bfloat16:
            raise ValueError("Invalid FP8 GEMM dimensions or output dtype")
        if x.device.type != "cuda" or any(
            t.device != x.device for t in (sx, w, sw, out, queue)
        ):
            raise ValueError("All operands must be on the same CUDA device")
        if queue.ndim != 2 or queue.shape[1] != 4 or queue.dtype != torch.int32:
            raise ValueError("Queue must be int32[num_tasks, 4]")
        if not queue.is_contiguous():
            raise ValueError("Invalid queue layout")
    else:
        _check_fp8_metadata(x, sx, w, sw, out, queue, 0, 32)
    if (
        expert_ids.shape != (x.shape[0],)
        or expert_ids.dtype != torch.int32
        or expert_ids.device != x.device
        or not expert_ids.is_contiguous()
    ):
        raise ValueError("Expected contiguous INT32 packed expert IDs")


@torch.library.custom_op("mono_ep::ready_grouped_bf16_out", mutates_args=("out",))
def ready_grouped_bf16_out(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    queue: torch.Tensor,
    expert_ids: torch.Tensor,
    incremental: bool = False,
) -> None:
    """Acquire every expert descriptor and compute its BF16 packed prefix.

    Publish payload and expert IDs with system-release before each ready flag.
    Every descriptor must be published, even for empty experts. The caller owns
    zero-padded output and reserves SM capacity for concurrent producers.
    """
    _ready_grouped_bf16_fake(x, w, out, queue, expert_ids)
    import deep_gemm

    deep_gemm.bf16_ready_grouped_gemm_nt(
        x, w, out, expert_ids, queue, incremental=incremental
    )


@ready_grouped_bf16_out.register_fake
def _ready_grouped_bf16_fake(x, w, out, queue, expert_ids, incremental=False):
    _check_metadata(x, w, out, queue, 0)
    if (
        expert_ids.shape != (x.shape[0],)
        or expert_ids.dtype != torch.int32
        or expert_ids.device != x.device
        or not expert_ids.is_contiguous()
    ):
        raise ValueError("Expected contiguous INT32 packed expert IDs")


@torch.library.custom_op(
    "mono_ep::ready_grouped_bf16_swiglu_backward_out",
    mutates_args=("h", "dz", "partial"),
)
def ready_grouped_bf16_swiglu_backward_out(
    dy: torch.Tensor,
    wt: torch.Tensor,
    h: torch.Tensor,
    queue: torch.Tensor,
    expert_ids: torch.Tensor,
    z: torch.Tensor,
    scores: torch.Tensor,
    dz: torch.Tensor,
    partial: torch.Tensor,
    incremental: bool = False,
) -> None:
    """Acquire grouped BF16 gradients and fuse weighted SwiGLU derivatives.

    The publication contract also covers saved gate/up activations and scores.
    Callers zero partials before launch; inactive h/dz rows remain untouched.
    Sum partial over its last dimension for routing-score gradients. Routing
    indices are retained from forward.
    """
    _ready_grouped_bf16_swiglu_fake(
        dy, wt, h, queue, expert_ids, z, scores, dz, partial
    )
    import deep_gemm

    epilogue = deep_gemm.epilogue.SwiGLUBackward(z, scores, dz, partial)
    deep_gemm.bf16_ready_grouped_gemm_nt(
        dy, wt, h, expert_ids, queue, epilogue=epilogue, incremental=incremental
    )


@ready_grouped_bf16_swiglu_backward_out.register_fake
def _ready_grouped_bf16_swiglu_fake(
    dy, wt, h, queue, expert_ids, z, scores, dz, partial, incremental=False
):
    _ready_grouped_bf16_fake(dy, wt, h, queue, expert_ids)
    _check_swiglu_metadata(h, z, scores, dz, partial)


@torch.library.custom_op(
    "mono_ep::ready_grouped_bf16_swiglu_forward_out", mutates_args=("z", "h")
)
def ready_grouped_bf16_swiglu_forward_out(
    x: torch.Tensor,
    interleaved_w: torch.Tensor,
    z: torch.Tensor,
    h: torch.Tensor,
    scores: torch.Tensor,
    queue: torch.Tensor,
    expert_ids: torch.Tensor,
    incremental: bool = False,
) -> None:
    """Acquire BF16 inputs and fuse interleaved up projection with weighted SwiGLU.

    Weight rows alternate gate/up. Saved z uses concatenated gate/up columns.
    Inactive output rows remain untouched. Publish scores with the payload before
    readiness; downstream consumers must mask rows outside the published groups.
    """
    _ready_grouped_bf16_swiglu_forward_fake(
        x, interleaved_w, z, h, scores, queue, expert_ids
    )
    import deep_gemm

    deep_gemm.bf16_ready_grouped_gemm_nt(
        x,
        interleaved_w,
        z,
        expert_ids,
        queue,
        epilogue=deep_gemm.epilogue.SwiGLUForward(z, scores, h),
        incremental=incremental,
    )


@ready_grouped_bf16_swiglu_forward_out.register_fake
def _ready_grouped_bf16_swiglu_forward_fake(
    x, interleaved_w, z, h, scores, queue, expert_ids, incremental=False
):
    _ready_grouped_bf16_fake(x, interleaved_w, z, queue, expert_ids)
    if (
        z.shape[1] % 16
        or h.shape != (z.shape[0], z.shape[1] // 2)
        or h.dtype != torch.bfloat16
        or h.device != x.device
        or not h.is_contiguous()
        or scores.shape != (x.shape[0],)
        or scores.dtype != torch.float32
        or scores.device != x.device
        or not scores.is_contiguous()
    ):
        raise ValueError("Expected BF16 activations and contiguous FP32 routing scores")
