# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Integration surface for training frameworks.

One :class:`MonoEPRuntime` per process owns the symmetric workspace for every
MoE layer. Everything except the runtime constructor and ``forward`` is pure Python
so a launcher can call it before CUDA or ``torch.distributed`` initialization.
"""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .runtime import PersistentMoE

DEFAULT_RESERVED_SMS = 16
PRECISIONS = ("auto", "bf16", "mxfp8", "fp8", "fp8_bwd")
MODES = ("auto", "fused", "dispatch")
FORWARD_OP = "mono_ep::persistent_forward"
BACKWARD_OP = "mono_ep::persistent_backward"
DISPATCH_OP = "mono_ep::peer_dispatch"
COMBINE_OP = "mono_ep::peer_combine"
DISPATCH_BACKWARD_OP = "mono_ep::peer_dispatch_backward"
COMBINE_BACKWARD_OP = "mono_ep::peer_combine_backward"
HIDDEN_MULTIPLE = 256
INTERMEDIATE_MULTIPLE = 128

_COMPUTE_DTYPES = {"bf16": "bf16", "mxfp8": "mxfp8", "fp8": "blockwise"}
_NATIVE_BACKWARD = {
    (9, "bf16"): "bf16_mono_ep_backward_sm90",
    (10, "bf16"): "bf16_mono_ep_backward",
    (9, "fp8"): "fp8_mono_ep_backward_sm90",
}


def resolve_precision(precision: str, capability: tuple[int, int]) -> str:
    """``auto`` is BF16 on SM90 and MXFP8 on SM100; explicit values pass through."""
    if precision != "auto":
        return precision
    return "mxfp8" if capability[0] == 10 else "bf16"


def unsupported_reason(
    *,
    capability: tuple[int, int],
    hidden: int,
    intermediate: int,
    has_bias: bool,
    activation: str,
    precision: str,
    activation_clamp: float | None = None,
) -> str | None:
    """Why the fused kernels cannot run this expert geometry, or ``None``."""
    major, minor = capability
    if major not in (9, 10):
        return (
            f"compute capability sm_{major}{minor} is unsupported; the fused kernels "
            "need SM90 (WGMMA) or SM100 (tcgen05/TMEM)"
        )
    if precision not in PRECISIONS:
        return f"unknown precision {precision!r}; expected one of {PRECISIONS}"
    resolved = resolve_precision(precision, capability)
    if resolved == "fp8_bwd":
        return "precision fp8_bwd (BF16 forward, blockwise FP8 backward) is not implemented yet"
    if resolved == "mxfp8" and major != 10:
        return "precision mxfp8 requires SM100"
    if resolved == "fp8" and major != 9:
        return "precision fp8 (blockwise E4M3) requires SM90"
    if hidden % HIDDEN_MULTIPLE:
        return f"hidden size {hidden} is not a multiple of {HIDDEN_MULTIPLE}"
    if intermediate % INTERMEDIATE_MULTIPLE:
        return (
            f"expert intermediate size {intermediate} is not a multiple of "
            f"{INTERMEDIATE_MULTIPLE}"
        )
    if has_bias:
        return "expert biases are unsupported"
    if activation != "silu":
        return f"activation {activation!r} is unsupported; only SwiGLU (SiLU gate) is fused"
    if activation_clamp is not None:
        return "clamped SwiGLU is not implemented in the fused backward yet"
    return None


def dispatch_unsupported_reason(
    *, capability: tuple[int, int], single_host: bool = True
) -> str | None:
    """Why the peer dispatch/combine ops cannot run here, or ``None``.

    They need torch symmetric memory (peer access inside one node) and the Triton
    gather/scatter kernels; the local experts kernel decides everything else.
    """
    major, minor = capability
    if major < 8:
        return (
            f"compute capability sm_{major}{minor} is unsupported; peer dispatch needs "
            "SM80 or newer"
        )
    if not single_host:
        return "the expert-parallel group spans more than one host; symmetric memory is node-local"
    return None


def _max_ctas(reserved_sms: int, dp_replicate: int) -> int:
    # HSDP adds a second concurrent collective stream (replicate all-reduce).
    return max(1, reserved_sms // (3 if dp_replicate > 1 else 2))


def nccl_env(
    *, reserved_sms: int = DEFAULT_RESERVED_SMS, dp_replicate: int = 1
) -> dict[str, str]:
    """Environment that must be exported before NCCL communicators are created.

    FSDP collectives otherwise launch more CTAs than the reserved SMs hold and wait
    forever behind the spinning persistent grid.
    """
    return {"NCCL_MAX_CTAS": str(_max_ctas(reserved_sms, dp_replicate))}


def check_nccl_env(
    *, reserved_sms: int = DEFAULT_RESERVED_SMS, dp_replicate: int = 1
) -> None:
    """Raise if ``NCCL_MAX_CTAS`` is unset, malformed or larger than the reservation allows."""
    limit = _max_ctas(reserved_sms, dp_replicate)
    value = os.environ.get("NCCL_MAX_CTAS")
    if value is None:
        raise RuntimeError(
            f"NCCL_MAX_CTAS is unset; export NCCL_MAX_CTAS<={limit} before the process "
            "group is created so NCCL kernels fit in the SMs reserved next to the "
            "persistent MoE kernels"
        )
    try:
        ctas = int(value)
    except ValueError:
        raise RuntimeError(f"NCCL_MAX_CTAS={value!r} is not an integer") from None
    if ctas < 1 or ctas > limit:
        raise RuntimeError(
            f"NCCL_MAX_CTAS={ctas} must be between 1 and {limit} with {reserved_sms} "
            f"reserved SMs (dp_replicate={dp_replicate}); larger values can deadlock "
            "FSDP collectives against the persistent MoE kernels"
        )


def _single_host(group) -> bool:
    import torch.distributed as dist

    if dist.get_world_size(group) == 1:
        return True
    hosts: list[str | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    return len(set(hosts)) == 1


class MonoEPDispatchRuntime:
    """Peer-memory dispatch and combine around a caller-supplied local experts kernel.

    Wraps :class:`mono_ep.peer_dispatch.PeerDispatch`: ``dispatch`` and ``combine`` go
    through the ``mono_ep::peer_dispatch`` / ``peer_combine`` custom ops (with their own
    backward ops), move only routed rows over NVLink symmetric memory and return the
    DeepEP row layout (``world * max_tokens_per_rank`` rows, local expert ids with
    ``-1`` for remote slots, zero weights there). Build one per process and share it
    across layers; no DeepGEMM, SM reservation or ``NCCL_MAX_CTAS`` is involved.
    """

    dispatch_op = DISPATCH_OP
    combine_op = COMBINE_OP
    dispatch_backward_op = DISPATCH_BACKWARD_OP
    combine_backward_op = COMBINE_BACKWARD_OP

    def __init__(self, *, group, max_tokens_per_rank: int, top_k: int, hidden: int):
        from .peer_dispatch import PeerDispatch

        reason = dispatch_unsupported_reason(
            capability=torch.cuda.get_device_capability(),
            single_host=_single_host(group),
        )
        if reason is not None:
            raise ValueError(f"mono_ep: {reason}")
        self._dispatch: PeerDispatch | None = PeerDispatch(
            capacity=max_tokens_per_rank, top_k=top_k, hidden=hidden, group=group
        )
        self.max_tokens_per_rank = max_tokens_per_rank
        self.world_size = self._dispatch.world
        self.receive_rows = self._dispatch.rows

    def _live(self):
        if self._dispatch is None:
            raise RuntimeError("mono_ep dispatch runtime used after destroy()")
        return self._dispatch

    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_local_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x`` BF16 ``[T, H]``, ``topk_idx`` int64 global ids ``[T, K]`` (``-1`` =
        dropped), ``topk_weights`` FP32 ``[T, K]`` -> ``(recv_x, recv_idx, recv_w)`` over
        ``world * max_tokens_per_rank`` rows."""
        from .peer_dispatch import peer_dispatch

        return peer_dispatch(
            x.contiguous(),
            topk_idx.contiguous(),
            topk_weights.contiguous(),
            num_local_experts,
            self._live().key,
        )

    def combine(
        self,
        local_out: torch.Tensor,
        recv_idx: torch.Tensor,
        topk_idx: torch.Tensor,
        num_local_experts: int,
    ) -> torch.Tensor:
        """Sum the per-rank partial rows ``local_out`` (the local kernel's output for
        ``dispatch``'s rows) onto this rank's ``topk_idx.shape[0]`` source tokens."""
        from .peer_dispatch import peer_combine

        routed = (recv_idx >= 0).any(dim=-1)
        return peer_combine(
            local_out.contiguous(),
            routed,
            topk_idx.contiguous(),
            num_local_experts,
            self._live().key,
        )

    def destroy(self) -> None:
        """Release the symmetric buffers; call only after every graph using them has finished."""
        if self._dispatch is None:
            return
        self._dispatch.destroy()
        self._dispatch = None


class MonoEPRuntime:
    """Fused dispatch + SwiGLU experts + combine for one expert-parallel group.

    Wraps :class:`mono_ep.runtime.PersistentMoE`: ``forward`` goes through the
    ``mono_ep::persistent_forward`` / ``persistent_backward`` custom ops, which
    retain only ``x``, routing and weights and recompute everything else in
    backward. Build one per process and share it across layers. The DeepGEMM SM
    budget is set to ``physical - reserved_sms`` for the runtime's lifetime (every
    launch checks it) and restored by :meth:`destroy`.
    """

    forward_op = FORWARD_OP
    backward_op = BACKWARD_OP

    def __init__(
        self,
        *,
        group,
        num_experts: int,
        max_tokens_per_rank: int,
        top_k: int,
        hidden: int,
        intermediate: int,
        precision: str = "auto",
        reserved_sms: int = DEFAULT_RESERVED_SMS,
        activation_clamp: float | None = None,
    ):
        import deep_gemm

        from .runtime import PersistentMoE

        capability = torch.cuda.get_device_capability()
        resolved = resolve_precision(precision, capability)
        reason = unsupported_reason(
            capability=capability,
            hidden=hidden,
            intermediate=intermediate,
            has_bias=False,
            activation="silu",
            precision=resolved,
            activation_clamp=activation_clamp,
        )
        if reason is not None:
            raise ValueError(f"mono_ep: {reason}")
        native = _NATIVE_BACKWARD.get((capability[0], resolved))
        backward_backend = (
            "native"
            if native is not None
            and hasattr(deep_gemm._C, native)
            and hidden % 256 == 0
            and intermediate % 256 == 0
            and _single_host(group)
            else "nccl"
        )
        physical = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self._previous_num_sms = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(physical - reserved_sms)
        self._moe: PersistentMoE | None
        try:
            self._moe = PersistentMoE(
                group,
                num_experts,
                max_tokens_per_rank,
                top_k,
                hidden,
                intermediate,
                compute_dtype=_COMPUTE_DTYPES[resolved],
                retain_activations=False,
                backward_backend=backward_backend,
                reserved_sms=reserved_sms,
            )
        except BaseException:
            deep_gemm.set_num_sms(self._previous_num_sms)
            raise
        self.precision = resolved
        self.backward_backend = backward_backend
        self.reserved_sms = reserved_sms
        self.max_tokens_per_rank = self._moe.buffer.num_max_tokens_per_rank

    @staticmethod
    def token_alignment() -> int:
        """Granularity the symmetric buffer rounds ``max_tokens_per_rank`` up to."""
        import deep_gemm

        return int(deep_gemm._C.get_token_alignment_for_mono_ep())

    def forward(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> torch.Tensor:
        """``x`` BF16 ``[T, H]``; ``topk_idx`` int64 global expert ids ``[T, K]`` with
        ``-1`` for dropped slots; ``topk_weights`` FP32 ``[T, K]`` applied at combine;
        ``w1`` ``[E_local, 2I, H]`` (gate rows then up rows) and ``w2``
        ``[E_local, H, I]`` in BF16 or FP32. Weight gradients come back in the
        master dtype."""
        if self._moe is None:
            raise RuntimeError("mono_ep runtime used after destroy()")
        return self._moe(
            x.contiguous(),
            topk_idx.contiguous(),
            topk_weights.contiguous(),
            w1.contiguous(),
            w2.contiguous(),
        )

    __call__ = forward

    def destroy(self) -> None:
        """Release the workspace; call only after every graph using it has finished."""
        if self._moe is None:
            return
        import deep_gemm

        self._moe.close()
        for buffer in (
            self._moe.buffer,
            getattr(self._moe._peer_backward, "buffer", None),
        ):
            if buffer is not None:
                buffer.destroy()
        self._moe = None
        deep_gemm.set_num_sms(self._previous_num_sms)
