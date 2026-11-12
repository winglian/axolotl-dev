# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""DeepGEMM persistent forward with retained or recomputed training activations.

Backward can use native SM90/SM100 BF16 dispatch/compute/combine fusion with
grouped weight gradients, or the fixed-capacity NCCL reference path.
Native backward currently requires recomputation and a single host.
"""

import os
from itertools import count

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.autograd.function import once_differentiable

from . import workspace_bound

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
            compute_dtype not in ("bf16", "blockwise", "mxfp8") or retain_activations
        ):
            raise ValueError(
                "Native persistent backward requires recomputing BF16, SM100 MXFP8 or Hopper "
                "blockwise FP8"
            )
        if compute_dtype == "blockwise" and retain_activations:
            raise ValueError("blockwise persistent FP8 requires recomputation")
        import deep_gemm

        arch = torch.cuda.get_device_capability()[0]
        if arch not in (9, 10):
            raise ValueError("the public DeepGEMM fused forward requires SM90+")
        if experts % dist.get_world_size(group) or min(experts, max_tokens, topk) < 1:
            raise ValueError("invalid expert-parallel geometry")
        if compute_dtype not in ("bf16", "mxfp8", "blockwise", "nvfp4_lora"):
            raise ValueError(
                "fused forward supports bf16, mxfp8, blockwise or nvfp4_lora"
            )
        if compute_dtype == "mxfp8" and arch != 10:
            raise ValueError("mxfp8 fused forward currently requires SM100")
        if compute_dtype == "nvfp4_lora":
            if arch != 10:
                raise ValueError("nvfp4_lora fused forward requires SM100")
            if hidden % 256 or intermediate % 256:
                raise ValueError(
                    "nvfp4_lora requires hidden and intermediate multiples of 256"
                )
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
        # Stored forward context (MONO_EP_STORED_CONTEXT): one z buffer per runtime and the key of the
        # forward call that last filled it; the native backward consumes it once.
        self._stored = None
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
        # The SM100 BF16 and NVFP4 forwards gate L1 claims on their ring (the same scheduler), so the
        # ring can be sized for a routed-row budget
        self.forward_ring_budget = (
            workspace_bound.forward_ring_budget(max_tokens, topk)
            if arch == 10
            and compute_dtype in ("bf16", "nvfp4_lora")
            and workspace_bound.mode() == "auto"
            else 0
        )
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
                else "nvfp4"
                if compute_dtype == "nvfp4_lora"
                else "bf16xbf16"
            ),
            num_ring_routed_tokens=self.forward_ring_budget,
            num_exact_ring_tokens=workspace_bound.forward_ring_tokens(
                hidden, intermediate, self.forward_ring_budget
            )
            if self.forward_ring_budget
            else 0,
        )

        if backward_backend == "native":
            from .persistent_backward import NativePersistentBackward

            self._peer_backward = NativePersistentBackward(self)

        self._key = next(_RUNTIME_KEYS)
        _RUNTIMES[self._key] = self

    def close(self):
        """Release the compiler handle after all graphs using it have finished."""
        _RUNTIMES.pop(self._key, None)

    def forward_nvfp4(
        self,
        x,
        indices,
        scores,
        w1_packed,
        w1_atoms,
        w2_packed,
        w2_atoms,
        expert_scales,
        record_key=None,
    ):
        """Forward on merged NVFP4 experts, no autograd.

        ``w1_packed`` is ``[E_local, 2I, H / 2]`` uint8 with granularity-8
        interleaved gate/up rows, ``w2_packed`` ``[E_local, H, I / 2]``; the atoms
        are the requant cache's ``scale_atoms`` (``[E_local, N / 128, K / 64, 512]``
        uint8); ``expert_scales`` is ``[E_local, 8]`` fp32 (see ``nvfp4_expert_scales``).
        ``record_key`` (the layer's frozen base tensor) records this call's per-expert
        receive counts for the bounded workspace of the matching backward.
        """
        import deep_gemm

        if self.compute_dtype != "nvfp4_lora":
            raise ValueError("forward_nvfp4 requires compute_dtype='nvfp4_lora'")
        _check_sm_budget(self)
        buffer = self.buffer
        tokens = x.shape[0]
        if x.dtype != torch.bfloat16 or x.shape[1] != self.hidden:
            raise ValueError("nvfp4_lora forward takes BF16 tokens of width hidden")
        if tokens > buffer.num_max_tokens_per_rank:
            raise ValueError("more tokens than the runtime's max_tokens")
        buffer.x[:tokens].copy_(x)
        buffer.topk_idx[:tokens].copy_(indices)
        buffer.topk_weights[:tokens].copy_(scores)
        y = torch.empty_like(x)
        bound = getattr(getattr(self, "_nvfp4_backward", None), "bound", None)
        counts = None
        if bound is not None and record_key is not None:
            counts = torch.zeros(
                self.experts // dist.get_world_size(self.group),
                dtype=torch.int32,
                device=x.device,
            )
        deep_gemm.nvfp4_mono_ep(
            y,
            (
                w1_packed.view(torch.int8),
                w1_atoms.view(torch.int32).view(*w1_atoms.shape[:3], 128),
            ),
            (
                w2_packed.view(torch.int8),
                w2_atoms.view(torch.int32).view(*w2_atoms.shape[:3], 128),
            ),
            expert_scales,
            buffer,
            cumulative_local_expert_recv_stats=counts,
            fast_math=False,
        )
        if counts is not None:
            bound.record(record_key, counts)
        return y

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
        self._record_forward = workspace_bound.will_backward(x, scores, w1, w2)
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


def _forward_impl(x, indices, scores, w1, w2, runtime, record_key=None):
    import deep_gemm

    _check_sm_budget(runtime)
    buffer = runtime.buffer
    tokens = x.shape[0]
    if runtime.compute_dtype == "blockwise":
        from .paired_weight_quantization import quantize_paired_gate_up_weights
        from .quantization import quantize_fp8
        from .sm90_prep import enabled as fused_prep, prepare_tokens
        from .sm90_weight_forms import FORWARD_FORMS, cache_mode, cached_forms
        from .token_quantization import quantize_tokens_into

        if fused_prep():
            prepare_tokens(
                x,
                buffer.x[:tokens],
                buffer.x_sf[:tokens],
                indices,
                buffer.topk_idx[:tokens],
                scores,
                buffer.topk_weights[:tokens],
            )
        else:
            quantize_tokens_into(
                x, buffer.x[:tokens], buffer.x_sf[:tokens], block_cols=128
            )
            buffer.topk_idx[:tokens].copy_(indices)
            buffer.topk_weights[:tokens].copy_(scores)
        mode = cache_mode()
        if mode != "0":
            prepared = list(cached_forms(w1, w2, FORWARD_FORMS, mode))
        else:
            if os.getenv("MONO_EP_SM90_FP8_FUSED_WEIGHT_QUANT", "1") != "0":
                from .block_weight_quantization import quantize_weights_128x128

                down = quantize_weights_128x128(w2)
            else:
                down = quantize_fp8(w2, block_rows=128, block_cols=128)
            prepared = [quantize_paired_gate_up_weights(w1), down]
        y = torch.empty_like(x)
        bound = getattr(runtime._peer_backward, "bound", None)
        counts = None
        if bound is not None and getattr(runtime, "_record_forward", True):
            counts = torch.zeros(
                runtime.experts // dist.get_world_size(runtime.group),
                dtype=torch.int32,
                device=x.device,
            )
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
            counts,
        )
        if counts is not None:
            bound.record(w1 if record_key is None else record_key, counts)
        return y, None
    if runtime.compute_dtype == "mxfp8":
        from .token_quantization import prepare_persistent_mxfp8

        qx, sx = prepare_persistent_mxfp8(x)
        buffer.x[:tokens].copy_(qx)
        buffer.x_sf[:tokens].copy_(sx)
        backward = runtime._peer_backward
        if hasattr(backward, "precision") and backward.precision() == "mxfp8":
            from .weight_cache import mxfp8_forward_weight_forms

            # W1's forward form is the MXFP8 backward's: one cached build serves the step
            tw1, tw2 = mxfp8_forward_weight_forms(w1, w2)
        else:
            tw1 = prepare_persistent_mxfp8(w1, weights=True, interleave=True)
            tw2 = prepare_persistent_mxfp8(w2, weights=True)
    else:
        from .weight_cache import weight_layouts

        buffer.x[:tokens].copy_(x)
        tw1, tw2 = weight_layouts(w1, w2, "w1_interleaved", "w2")
    buffer.topk_idx[:tokens].copy_(indices)
    buffer.topk_weights[:tokens].copy_(scores)
    y = torch.empty_like(x)
    forward = (
        deep_gemm.fp8_fp4_mono_ep
        if runtime.compute_dtype == "mxfp8"
        else deep_gemm.bf16_mono_ep
    )
    saved_z = None
    if _store_context(runtime):
        from .stored_context import context_key, policy

        stored = runtime._stored
        z = stored.prepare(x.device, enforce_memory=policy() == "auto")
        forward(y, tw1, tw2, buffer, fast_math=False, saved_z=z)
        stored.key = context_key(x, indices, scores, w1, w2)
    elif runtime.retain_activations:
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
        bound = getattr(runtime._peer_backward, "bound", None)
        if (
            bound is not None
            and getattr(runtime, "_record_forward", True)
            and not os.getenv("MONO_EP_SM100_FORWARD_PHASE_TRACE")
        ):
            # Per-expert receive counts of this call size the matching backward's pools
            counts = torch.zeros(
                runtime.experts // dist.get_world_size(runtime.group),
                dtype=torch.int32,
                device=x.device,
            )
            forward(
                y,
                tw1,
                tw2,
                buffer,
                fast_math=False,
                cumulative_local_expert_recv_stats=counts,
            )
            bound.record(w1 if record_key is None else record_key, counts)
        else:
            forward(y, tw1, tw2, buffer, fast_math=False)
    return y, saved_z


def _store_context(runtime):
    """Whether this forward stores z for the next backward (``stored_context.policy``)."""
    from . import stored_context

    if runtime._stored is not None:
        runtime._stored.key = None
    if (
        runtime.compute_dtype != "bf16"
        or runtime.retain_activations
        or torch.cuda.get_device_capability()[0] != 10
        or runtime._peer_backward is None
        or not hasattr(runtime._peer_backward, "stored_context_usable")
    ):
        return False
    policy = stored_context.policy()
    if policy == "off":
        return False
    if policy != "always" and torch._C._current_graph_task_id() == -1:
        return False
    if not runtime._peer_backward.stored_context_usable():
        return False
    if policy == "auto" and not stored_context.backward_rule(
        runtime._peer_backward.precision()
    ):
        return False
    if runtime._stored is None:
        runtime._stored = stored_context.StoredContext(
            runtime, dist.get_world_size(runtime.group)
        )
    return policy != "auto" or runtime._stored.decide_auto()


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
