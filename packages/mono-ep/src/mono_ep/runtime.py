# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""DeepGEMM persistent forward with retained or recomputed training activations.

Backward can use native SM90/SM100 BF16 dispatch/compute/combine fusion with
grouped weight gradients, or the fixed-capacity NCCL reference path.
Native backward currently requires recomputation and a single host.
"""

from itertools import count

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.autograd.function import once_differentiable

_RUNTIMES: dict[int, "PersistentMoE"] = {}
_RUNTIME_KEYS = count(1)


class PersistentMoE:
    """One reusable symmetric workspace for a fixed expert-parallel geometry."""

    def __init__(
        self,
        group,
        experts,
        max_tokens,
        topk,
        hidden,
        intermediate,
        *,
        fuse_activation=False,
        compute_dtype="bf16",
        retain_activations=False,
        backward_backend="nccl",
        reserved_sms=0,
    ):
        if backward_backend not in ("nccl", "native"):
            raise ValueError("backward_backend must be nccl or native")
        if backward_backend == "native" and (
            compute_dtype not in ("bf16", "blockwise") or retain_activations
        ):
            raise ValueError(
                "Native persistent backward requires recomputing BF16 or Hopper blockwise FP8"
            )
        if compute_dtype == "blockwise" and retain_activations:
            raise ValueError("blockwise persistent FP8 requires recomputation")
        import deep_gemm

        arch = torch.cuda.get_device_capability()[0]
        if arch not in (9, 10):
            raise ValueError("the public DeepGEMM fused forward requires SM90+")
        if experts % dist.get_world_size(group) or min(experts, max_tokens, topk) < 1:
            raise ValueError("invalid expert-parallel geometry")
        if compute_dtype not in ("bf16", "mxfp8", "blockwise"):
            raise ValueError("fused forward supports bf16, mxfp8 or blockwise")
        if compute_dtype == "mxfp8" and arch != 10:
            raise ValueError("mxfp8 fused forward currently requires SM100")
        if compute_dtype == "blockwise" and arch != 9:
            raise ValueError("blockwise persistent FP8 currently requires SM90")
        physical_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        if (
            not isinstance(reserved_sms, int)
            or reserved_sms < 0
            or reserved_sms >= physical_sms - 1
        ):
            raise ValueError(
                "reserved_sms must leave at least two SMs for persistent kernels"
            )
        self.num_sms = physical_sms - reserved_sms
        if arch >= 10 and self.num_sms % 2:
            raise ValueError("SM100 persistent kernels require an even active SM count")
        if deep_gemm.get_num_sms() != self.num_sms:
            raise ValueError(
                "call deep_gemm.set_num_sms(physical_sms - reserved_sms) before constructing PersistentMoE"
            )
        self._fixed_capacity_backward = True
        self._peer_backward = None
        self.retain_activations = retain_activations
        self.compute_dtype = compute_dtype
        self.group = group
        self.experts = experts
        self.max_tokens = max_tokens
        self.topk = topk
        self.hidden = hidden
        self.intermediate = intermediate
        self.fuse_activation = fuse_activation
        self.alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
        self.buffer = deep_gemm.get_symm_buffer_for_mono_ep(
            group,
            experts,
            max_tokens,
            topk,
            hidden,
            intermediate,
            mma_type=(
                "fp8xfp8"
                if compute_dtype == "mxfp8"
                else "fp8xbf16"
                if compute_dtype == "blockwise"
                else "bf16xbf16"
            ),
        )

        if backward_backend == "native":
            from .persistent_backward import NativePersistentBackward

            self._peer_backward = NativePersistentBackward(self)

        self._key = next(_RUNTIME_KEYS)
        _RUNTIMES[self._key] = self

    def close(self):
        """Release the compiler handle after all graphs using it have finished."""
        _RUNTIMES.pop(self._key, None)

    def __call__(self, x, indices, scores, w1, w2):
        local_experts = self.experts // dist.get_world_size(self.group)
        if x.ndim != 2 or x.shape[0] > self.max_tokens or x.shape[1] != self.hidden:
            raise ValueError("input does not match the symmetric workspace")
        if indices.shape != (x.shape[0], self.topk) or scores.shape != indices.shape:
            raise ValueError(
                "expected [tokens, topk] expert indices and routing weights"
            )
        if w1.shape != (local_experts, 2 * self.intermediate, self.hidden):
            raise ValueError("gate/up weights have the wrong shape")
        if w2.shape != (local_experts, self.hidden, self.intermediate):
            raise ValueError("down weights have the wrong shape")
        if (
            x.dtype != torch.bfloat16
            or any(t.dtype not in (torch.bfloat16, torch.float32) for t in (w1, w2))
            or scores.dtype != torch.float32
        ):
            raise ValueError(
                "expected BF16 inputs, BF16/FP32 weights and FP32 routing weights"
            )
        if indices.dtype != torch.int64:
            raise ValueError("expected int64 expert indices")
        if any(
            t.device != x.device or not t.is_contiguous()
            for t in (x, indices, scores, w1, w2)
        ):
            raise ValueError("all tensors must be contiguous on the same CUDA device")
        torch._assert_async(
            ((indices >= -1) & (indices < self.experts)).all(), "invalid expert index"
        )
        if self.retain_activations:
            return _PersistentMoE.apply(x, indices, scores, w1, w2, self)
        return persistent_forward(x, indices, scores, w1, w2, self._key)


def _check_sm_budget(runtime):
    import deep_gemm

    if (
        deep_gemm.get_num_sms() != runtime.num_sms
        or runtime.buffer.num_sms != runtime.num_sms
    ):
        raise RuntimeError(
            "persistent MonoEP SM count changed after workspace allocation"
        )


def _forward_impl(x, indices, scores, w1, w2, runtime):
    import deep_gemm

    _check_sm_budget(runtime)
    buffer = runtime.buffer
    tokens = x.shape[0]
    if runtime.compute_dtype == "blockwise":
        from .paired_weight_quantization import quantize_paired_gate_up_weights
        from .quantization import quantize_fp8
        from .token_quantization import quantize_tokens_into

        quantize_tokens_into(x, buffer.x[:tokens], buffer.x_sf[:tokens], block_cols=128)
        buffer.topk_idx[:tokens].copy_(indices)
        buffer.topk_weights[:tokens].copy_(scores)
        prepared = [
            quantize_paired_gate_up_weights(w1),
            quantize_fp8(w2, block_rows=128, block_cols=128),
        ]
        y = torch.empty_like(x)
        deep_gemm._C.fp8_mono_ep_forward_sm90(
            y,
            prepared[0][0],
            prepared[1][0],
            [p[1] for p in prepared],
            buffer.buffer,
            buffer.handle.buffer_ptrs,
            dist.get_rank(runtime.group),
            buffer.num_max_tokens_per_rank,
            buffer.l1_acts.shape[0],
            runtime.topk,
            None,
            True,
        )
        return y, None
    if runtime.compute_dtype == "mxfp8":
        from .token_quantization import prepare_persistent_mxfp8

        qx, sx = prepare_persistent_mxfp8(x)
        buffer.x[:tokens].copy_(qx)
        buffer.x_sf[:tokens].copy_(sx)
        tw1 = prepare_persistent_mxfp8(w1, weights=True, interleave=True)
        tw2 = prepare_persistent_mxfp8(w2, weights=True)
    else:
        buffer.x[:tokens].copy_(x)
        tw1, tw2 = deep_gemm.transform_weights_for_mono_ep(
            w1.bfloat16(), w2.bfloat16()
        )
    buffer.topk_idx[:tokens].copy_(indices)
    buffer.topk_weights[:tokens].copy_(scores)
    y = torch.empty_like(x)
    forward = (
        deep_gemm.fp8_fp4_mono_ep
        if runtime.compute_dtype == "mxfp8"
        else deep_gemm.bf16_mono_ep
    )
    saved_z = None
    if runtime.retain_activations:
        saved_z = x.new_empty(
            (
                dist.get_world_size(runtime.group),
                buffer.num_max_tokens_per_rank,
                runtime.topk,
                2 * runtime.intermediate,
            )
        )
        forward(y, tw1, tw2, buffer, fast_math=False, saved_z=saved_z)
    else:
        forward(y, tw1, tw2, buffer, fast_math=False)
    return y, saved_z


def _backward_impl(dy, x, indices, scores, w1, w2, runtime, retained=()):
    _check_sm_budget(runtime)
    group = runtime.group
    world = dist.get_world_size(group)
    if runtime._peer_backward is not None:
        dx, ds, gw1, gw2 = runtime._peer_backward.backward(
            dy, x, indices, scores, w1, w2
        )
        return dx, None, ds, gw1.to(w1.dtype), gw2.to(w2.dtype), None

    from .grouped import expert_backward
    from .packing import DeviceExpertLayout, pack_retained_projections, pack_rows
    from .pipeline import _build_send_layout, _compute_send_counts, exchange

    def local_backward(
        grad, inputs, routing, layout, saved_z=None, output_exchange=None
    ):
        if runtime.compute_dtype in ("mxfp8", "blockwise"):
            from .fp8 import expert_backward_fp8

            return expert_backward_fp8(
                grad,
                inputs,
                routing,
                w1,
                w2,
                layout,
                format=runtime.compute_dtype,
                fuse_activation=runtime.fuse_activation,
                output_exchange=output_exchange,
                activations=None
                if saved_z is None
                else (layout.pack(inputs), saved_z, None),
            )
        return expert_backward(
            grad,
            inputs,
            routing,
            w1.bfloat16(),
            w2.bfloat16(),
            layout,
            fuse_activation=runtime.fuse_activation,
            saved_z=saved_z,
            output_exchange=output_exchange,
        )

    if world == 1:
        layout = DeviceExpertLayout.build(indices, w1.shape[0], runtime.alignment)
        packed_z = None
        if retained:
            packed_z = pack_rows(
                retained[0][0, : x.shape[0]].flatten(0, 1),
                layout.row_assignments,
                1,
            )
        dx, ds, gw1, gw2 = local_backward(dy.contiguous(), x, scores, layout, packed_z)
        return dx, None, ds, gw1.to(w1.dtype), gw2.to(w2.dtype), None
    if runtime._fixed_capacity_backward:
        from .fixed_dispatch import fixed_backward

        dx, ds, gw1, gw2 = fixed_backward(
            dy,
            x,
            indices,
            scores,
            w1.shape[0],
            runtime,
            retained,
            local_backward,
        )
        return dx, None, ds, gw1.to(w1.dtype), gw2.to(w2.dtype), None
    dest, membership, counts = _compute_send_counts(indices, world, w1.shape[0])
    recv_counts = funcol.wait_tensor(exchange(counts, [1] * world, [1] * world, group))
    send, recv = torch.stack((counts, recv_counts)).cpu().tolist()
    tokens, _, local_ids, weights = _build_send_layout(
        indices, scores, dest, membership, sum(send), w1.shape[0]
    )
    rx = exchange(x[tokens], recv, send, group)
    rdy = exchange(dy[tokens], recv, send, group)
    rs = exchange(weights, recv, send, group)
    ri = exchange(local_ids, recv, send, group)
    rx, rdy, rs, ri = (funcol.wait_tensor(t) for t in (rx, rdy, rs, ri))
    layout = DeviceExpertLayout.build(ri, w1.shape[0], runtime.alignment)
    packed_z = None
    if retained:
        received_tokens = funcol.wait_tensor(exchange(tokens, recv, send, group))
        source_ranks = torch.repeat_interleave(
            torch.arange(world, device=x.device),
            torch.tensor(recv, device=x.device),
            output_size=sum(recv),
        )
        packed_z = pack_retained_projections(
            retained[0], source_ranks, received_tokens, layout.row_assignments
        )

    def return_gradients(gx, gs):
        return exchange(gx, send, recv, group), exchange(gs, send, recv, group)

    gx, gs, gw1, gw2 = local_backward(rdy, rx, rs, layout, packed_z, return_gradients)
    gx, gs = funcol.wait_tensor(gx), funcol.wait_tensor(gs)
    dx = torch.zeros_like(x, dtype=torch.float32)
    dx.index_add_(0, tokens, gx.float())
    ds = torch.zeros_like(scores)
    ds.index_add_(0, tokens, gs.masked_fill(local_ids < 0, 0))
    return dx.to(x.dtype), None, ds, gw1.to(w1.dtype), gw2.to(w2.dtype), None


class _PersistentMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, scores, w1, w2, runtime):
        y, saved_z = _forward_impl(x, indices, scores, w1, w2, runtime)
        ctx.save_for_backward(
            x, indices, scores, w1, w2, *(() if saved_z is None else (saved_z,))
        )
        ctx.runtime = runtime
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, indices, scores, w1, w2, *retained = ctx.saved_tensors
        return _backward_impl(dy, x, indices, scores, w1, w2, ctx.runtime, retained)


@torch.library.custom_op("mono_ep::persistent_forward", mutates_args=())
def persistent_forward(
    x: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    runtime_key: int,
) -> torch.Tensor:
    runtime = _RUNTIMES[runtime_key]
    if runtime.retain_activations:
        raise ValueError("The recomputing op requires retention disabled")
    return _forward_impl(x, indices, scores, w1, w2, runtime)[0]


@persistent_forward.register_fake
def _persistent_forward_fake(x, indices, scores, w1, w2, runtime_key):
    return torch.empty_like(x)


@torch.library.custom_op("mono_ep::persistent_backward", mutates_args=())
def persistent_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    runtime_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dx, _, ds, dw1, dw2, _ = _backward_impl(
        dy, x, indices, scores, w1, w2, _RUNTIMES[runtime_key]
    )
    return dx, ds, dw1, dw2


@persistent_backward.register_fake
def _persistent_backward_fake(dy, x, indices, scores, w1, w2, runtime_key):
    return tuple(torch.empty_like(t) for t in (x, scores, w1, w2))


def _setup_context(ctx, inputs, output):
    x, indices, scores, w1, w2, runtime_key = inputs
    ctx.save_for_backward(x, indices, scores, w1, w2)
    ctx.runtime_key = runtime_key


def _autograd_backward(ctx, dy):
    dx, ds, dw1, dw2 = persistent_backward(
        dy.contiguous(), *ctx.saved_tensors, ctx.runtime_key
    )
    return dx, None, ds, dw1, dw2, None


persistent_forward.register_autograd(_autograd_backward, setup_context=_setup_context)
