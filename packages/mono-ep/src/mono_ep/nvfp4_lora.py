# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""NVFP4 LoRA expert path on the SM100 persistent kernels (``precision="nvfp4_lora"``).

The base experts stay packed NVFP4 and frozen; only the LoRA factors train.
The requant kernel builds the merged forms the kernels consume (``w1`` with
interleaved gate/up rows and ``w2`` for the forward; ``w1``, and the transposed
``w1t`` / ``w2t``, for the backward GEMMs) from the base bytes and the current
adapters. ``MONO_EP_NVFP4_MERGED_CACHE=layer`` (default) builds a layer's forms
right before its forward and again before its backward and keeps the forms of
the last few layers only; ``all`` builds every form of every layer once per
optimizer step and holds them across the step. The forward runs block-scaled
UMMA on them; the backward returns ``dx``, ``dscore`` and the adapter gradients
from rank-``r`` grouped GEMMs on the BF16 pools. No base-weight gradients and
no fp32 masters of the base.
"""

import functools
import logging
import os
import socket
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.autograd.function import once_differentiable

from . import workspace_bound
from .lora import (  # noqa: F401
    MERGED_CACHE_LAYERS,
    MERGED_CACHE_MODES,
    POOL_BLOCK,
    MergedWeightCache,
    attach_post_step_hook,
    lora_weight_grads,
    merged_cache_mode,
    padded_offsets,
)
from .persistent_backward import owned_backward_ring_tokens

LOG = logging.getLogger("mono_ep")
OPTIONS_ENV = "MONO_EP_SM100_NVFP4_BACKWARD_OPTIONS"
MERGED_CACHE_ENV = "MONO_EP_NVFP4_MERGED_CACHE"


REQUANT_BACKENDS = ("triton", "cuda", "cuda_fused")


def requant_modules():
    """The vendored packed requant kernels and references (``mono_ep.nvfp4_requant``)."""
    from . import nvfp4_requant as pkg

    return types.SimpleNamespace(
        requant=pkg.nvfp4_requant,
        triton_requant=pkg.triton_nvfp4_requant,
        cuda_requant=pkg.cuda_nvfp4_requant,
    )


DEFAULT_REQUANT_BACKEND = "cuda_fused"


def requant_backend(backend=None):
    """``cuda_fused`` (default: SM100 kernel; the transposed backward forms are
    quantized from the merged tile, so no transposed copy of the base is stored),
    ``cuda`` (SM100 kernel, the Triton kernel's bytes: the transposed forms come
    from a stored transposed base, a second packed copy per layer) or ``triton``
    (the vendored Triton kernel, same bytes as ``cuda``); ``MONO_EP_NVFP4_REQUANT``
    when not given."""
    backend = backend or os.environ.get(
        "MONO_EP_NVFP4_REQUANT", DEFAULT_REQUANT_BACKEND
    )
    if backend not in REQUANT_BACKENDS:
        raise ValueError(f"requant backend must be one of {REQUANT_BACKENDS}")
    return backend


def interleave_perm(n2, device):
    """Row permutation to the kernels' granularity-8 gate/up interleave: new row
    ``16j + 8h + i`` holds gate (``h=0``) / up (``h=1``) output column ``8j + i``."""
    from deep_gemm.mega import _interleave_weights

    return _interleave_weights(torch.arange(n2, device=device).view(n2, 1)).view(-1)


def _atoms(result):
    atoms = result.scale_atoms
    return atoms.view(torch.int32).view(*atoms.shape[:3], 128)


@dataclass
class MergedNvfp4Weights:
    """One step's merged experts in the kernel layouts (``u8`` packed FP4 and
    ``int32`` UTCCP scale-atom views) with the per-expert scale tables. A
    ``layer`` cache fills the forward forms (``w1``, ``w2``) and the backward
    forms (``w1``, ``w1t``, ``w2t``) separately, so a form may be ``None``."""

    w1: torch.Tensor
    w1_atoms: torch.Tensor
    w2: torch.Tensor
    w2_atoms: torch.Tensor
    w1t: torch.Tensor
    w1t_atoms: torch.Tensor
    w2t: torch.Tensor
    w2t_atoms: torch.Tensor
    scales_forward: torch.Tensor
    scales_backward: torch.Tensor
    version: int

    @property
    def has_forward(self):
        return self.w1 is not None and self.w2 is not None

    @property
    def has_backward(self):
        return self.w1 is not None and self.w1t is not None and self.w2t is not None

    def nbytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.w1,
                self.w1_atoms,
                self.w2,
                self.w2_atoms,
                self.w1t,
                self.w1t_atoms,
                self.w2t,
                self.w2t_atoms,
            )
            if t is not None
        )


class Nvfp4LoraExperts:
    """The frozen packed NVFP4 base of one MoE layer (``q1 [E, 2I, H/2]`` u8 in
    concat gate/up order, ``s1 [E, 2I, H/16]`` e4m3, ``pts1 [E]`` fp32; ``q2
    [E, H, I/2]``, ``s2``, ``pts2``), its interleaved and transposed forms, and
    the requant of the merged weights for given LoRA factors.

    ``input_scale`` is the checkpoint's ``(pts_x, pts_h)`` pair for the static
    activation quantization of the forward (``(1, 1)`` for weight-only exports):
    floats shared by every expert, or ``[E]`` tensors with one scale per expert
    (the kernels read the scale table per local expert). The dy and dz operands
    of the backward are scaled per row by the kernel. ``backend`` picks the
    requant kernel (:func:`requant_backend`).
    """

    def __init__(
        self,
        q1,
        s1,
        pts1,
        q2,
        s2,
        pts2,
        *,
        scaling,
        input_scale=(1.0, 1.0),
        backend=None,
    ):
        mods = requant_modules()
        self.requant, self.triton_requant = mods.requant, mods.triton_requant
        self.cuda_requant = mods.cuda_requant
        self.backend = requant_backend(backend)
        self.scaling = float(scaling)
        self.experts, n1, k1 = q1.shape
        self.hidden, self.inter = 2 * k1, n1 // 2
        device = q1.device
        self.pts1 = pts1.reshape(self.experts).float().contiguous()
        self.pts2 = pts2.reshape(self.experts).float().contiguous()
        self.perm = interleave_perm(2 * self.inter, device)
        self.q1 = q1.view(torch.uint8).contiguous()
        self.s1 = s1.view(torch.float8_e4m3fn).contiguous()
        self.q2 = q2.view(torch.uint8).contiguous()
        self.s2 = s2.view(torch.float8_e4m3fn).contiguous()
        # Transposed bases for the backward's K-major operands (dequant -> transpose -> quantize, same scales);
        # the fused requant builds those forms from the merged tile and needs none
        if self.backend != "cuda_fused":
            self.q1t, self.s1t = self._transposed(self.q1, self.s1, self.pts1)
            self.q2t, self.s2t = self._transposed(self.q2, self.s2, self.pts2)
        pts_x, pts_h = (self._per_expert(v, device) for v in input_scale)
        self.pts_x, self.pts_h = pts_x, pts_h
        fwd = torch.zeros(self.experts, 8, device=device, dtype=torch.float32)
        fwd[:, 0] = 1.0 / (6.0 * pts_x)
        fwd[:, 1] = 1.0 / pts_x
        fwd[:, 2] = pts_x * self.pts1
        fwd[:, 3] = pts_x * self.pts1
        fwd[:, 4] = 1.0 / (6.0 * pts_h)
        fwd[:, 5] = 1.0 / pts_h
        fwd[:, 6] = pts_h * self.pts2
        bwd = torch.zeros_like(fwd)
        bwd[:, 0] = 1.0 / (6.0 * pts_x)
        bwd[:, 1] = 1.0 / pts_x
        bwd[:, 2] = pts_x * self.pts1
        bwd[:, 3] = self.pts2
        bwd[:, 4] = self.pts1
        self.scales_forward, self.scales_backward = fwd, bwd

    @functools.cached_property
    def q1i(self):
        """``q1`` in interleaved rows, for the Triton requant only (the CUDA kernels interleave on store)."""
        return self.q1.index_select(1, self.perm).contiguous()

    @functools.cached_property
    def s1i(self):
        return self.s1.index_select(1, self.perm).contiguous()

    def _per_expert(self, value, device):
        """An ``[E]`` fp32 scale table from a float or a per-expert tensor."""
        if isinstance(value, torch.Tensor):
            value = value.detach().to(device=device, dtype=torch.float32).reshape(-1)
            if value.numel() == 1:
                value = value.expand(self.experts)
            if value.numel() != self.experts:
                raise ValueError(
                    f"input_scale needs one value per expert ({self.experts}), "
                    f"got {value.numel()}"
                )
            return value.contiguous()
        return torch.full(
            (self.experts,), float(value), device=device, dtype=torch.float32
        )

    def _transposed(self, q, s, pts):
        """``(packed [E, K, N/2], scale [E, K, N/16])`` of the transposed base, built
        one expert at a time: the fp32 round trip of a whole Beam layer would be GBs."""
        e, n, k2 = q.shape
        k = 2 * k2
        packed = torch.empty(e, k, n // 2, device=q.device, dtype=torch.uint8)
        scale = torch.empty(e, k, n // 16, device=q.device, dtype=torch.float8_e4m3fn)
        for i in range(e):
            dq = self.requant.dequantize_nvfp4_packed_ref(
                q[i], s[i], pts[i].expand(n), torch.float32
            )
            packed_i, scale_i = self.requant.quantize_nvfp4_packed_ref(
                dq.t().contiguous(), pts[i].expand(k)
            )
            packed[i].copy_(packed_i)
            scale[i].copy_(scale_i.view(torch.float8_e4m3fn))
            del dq, packed_i, scale_i
        return packed, scale

    def dequantized(self, result, pts):
        """``[E, N, K]`` fp32 of a requant result (the oracle's view of the kernel's bytes)."""
        e, n, k2 = result.qdata.shape
        return self.requant.dequantize_nvfp4_packed_ref(
            result.qdata.view(e * n, k2),
            result.scale.view(e * n, k2 // 8),
            pts.repeat_interleave(n),
            torch.float32,
        ).view(e, n, 2 * k2)

    def _requant(self, q, s, pts, lora_a, lora_b):
        if self.backend != "triton":
            return self.cuda_requant.requant_natural(
                q,
                s,
                pts,
                lora_a.to(torch.bfloat16),
                lora_b.to(torch.bfloat16),
                self.scaling,
                pts,
            )
        e, r = lora_a.shape[:2]
        a = lora_a.reshape(e * r, lora_a.shape[2]).to(torch.bfloat16).contiguous()
        b = (
            lora_b.permute(1, 2, 0)
            .reshape(lora_b.shape[1], r * e)
            .to(torch.bfloat16)
            .contiguous()
        )
        return self.triton_requant.requant_lora_packed(
            q, s, pts, a, b, self.scaling, pts
        )

    def requant_all(self, a1, b1, a2, b2):
        """Requant results for ``w1`` (interleaved rows), ``w2``, ``w1t``, ``w2t``
        from adapters ``a1 [E, r, H]``, ``b1 [E, 2I, r]``, ``a2 [E, r, I]``, ``b2 [E, H, r]``."""
        a1, b1, a2, b2 = (t.detach() for t in (a1, b1, a2, b2))
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        if self.backend == "cuda":
            # each weight and its transposed base in one pass: the delta is computed once
            pair = self.cuda_requant.requant_paired
            r1, r1t = pair(
                self.q1,
                self.s1,
                self.q1t,
                self.s1t,
                self.pts1,
                bf(a1),
                bf(b1),
                self.scaling,
                interleave=8,
            )
            r2, r2t = pair(
                self.q2,
                self.s2,
                self.q2t,
                self.s2t,
                self.pts2,
                bf(a2),
                bf(b2),
                self.scaling,
            )
            return r1, r2, r1t, r2t
        if self.backend == "cuda_fused":
            fused = self.cuda_requant.requant_fused
            r1, r1t = fused(
                self.q1, self.s1, self.pts1, bf(a1), bf(b1), self.scaling, interleave=8
            )
            r2, r2t = fused(self.q2, self.s2, self.pts2, bf(a2), bf(b2), self.scaling)
            return r1, r2, r1t, r2t
        r1 = self._requant(
            self.q1i, self.s1i, self.pts1, a1, b1.index_select(1, self.perm)
        )
        r2 = self._requant(self.q2, self.s2, self.pts2, a2, b2)
        # W^T = base^T + scaling * A^T @ B^T: the LoRA "A" of the transposed form is B^T
        r1t = self._requant(
            self.q1t, self.s1t, self.pts1, b1.transpose(1, 2), a1.transpose(1, 2)
        )
        r2t = self._requant(
            self.q2t, self.s2t, self.pts2, b2.transpose(1, 2), a2.transpose(1, 2)
        )
        return r1, r2, r1t, r2t

    def requant_forward(self, a1, b1, a2, b2):
        """The forward's forms only: ``(w1 interleaved, w2)``, one pass over each base."""
        a1, b1, a2, b2 = (t.detach() for t in (a1, b1, a2, b2))
        if self.backend == "triton":
            r1 = self._requant(
                self.q1i, self.s1i, self.pts1, a1, b1.index_select(1, self.perm)
            )
            return r1, self._requant(self.q2, self.s2, self.pts2, a2, b2)
        natural = self.cuda_requant.requant_natural
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        r1 = natural(
            self.q1,
            self.s1,
            self.pts1,
            bf(a1),
            bf(b1),
            self.scaling,
            self.pts1,
            interleave=8,
        )
        r2 = natural(
            self.q2, self.s2, self.pts2, bf(a2), bf(b2), self.scaling, self.pts2
        )
        return r1, r2

    def requant_backward(self, a1, b1, a2, b2):
        """The backward's forms: ``(w1 interleaved, w1t, w2t)``. The fused kernel
        writes each transposed form next to its normal form from one merged
        tile; the other backends build all four forms (``w2`` is dropped)."""
        r1, r2, r1t, r2t = self.requant_all(a1, b1, a2, b2)
        del r2
        return r1, r1t, r2t

    def transposed_base_bytes(self):
        """Bytes of the stored transposed base (``cuda`` / ``triton`` backends; 0 for ``cuda_fused``)."""
        return sum(
            t.numel() * t.element_size()
            for name in ("q1t", "s1t", "q2t", "s2t")
            for t in (getattr(self, name, None),)
            if t is not None
        )

    def merged(self, a1, b1, a2, b2, version=0):
        r1, r2, r1t, r2t = self.requant_all(a1, b1, a2, b2)
        return MergedNvfp4Weights(
            r1.qdata,
            _atoms(r1),
            r2.qdata,
            _atoms(r2),
            r1t.qdata,
            _atoms(r1t),
            r2t.qdata,
            _atoms(r2t),
            self.scales_forward,
            self.scales_backward,
            version,
        )

    def merged_forward(self, a1, b1, a2, b2, version=0):
        r1, r2 = self.requant_forward(a1, b1, a2, b2)
        return MergedNvfp4Weights(
            r1.qdata,
            _atoms(r1),
            r2.qdata,
            _atoms(r2),
            None,
            None,
            None,
            None,
            self.scales_forward,
            self.scales_backward,
            version,
        )

    def merged_backward(self, a1, b1, a2, b2, version=0):
        r1, r1t, r2t = self.requant_backward(a1, b1, a2, b2)
        return MergedNvfp4Weights(
            r1.qdata,
            _atoms(r1),
            None,
            None,
            r1t.qdata,
            _atoms(r1t),
            r2t.qdata,
            _atoms(r2t),
            self.scales_forward,
            self.scales_backward,
            version,
        )


class RequantCache(MergedWeightCache):
    """The merged NVFP4 forms of every layer (keyed by its :class:`Nvfp4LoraExperts`):
    ``w1`` / ``w2`` forward, ``w1`` / ``w1t`` / ``w2t`` backward; the mode comes from
    ``MONO_EP_NVFP4_MERGED_CACHE``."""

    env = MERGED_CACHE_ENV


class Nvfp4PersistentBackward:
    """Native SM100 NVFP4 backward: own BF16 symmetric buffer for the dy exchange,
    BF16 pools (x, dy, h, dz), packed FP4 pools with scale words (x, dy, dz) and
    the per-row dy/dz scales. ``options`` (default: the launcher's) pins the
    kernel options bitmask for this instance."""

    def __init__(self, runtime, options=None):
        import deep_gemm

        self.options = options

        if runtime.compute_dtype != "nvfp4_lora":
            raise ValueError(
                "Nvfp4PersistentBackward requires compute_dtype='nvfp4_lora'"
            )
        if torch.cuda.get_device_capability()[0] != 10 or not hasattr(
            deep_gemm._C, "nvfp4_mono_ep_backward"
        ):
            raise RuntimeError(
                "Rebuild vendored DeepGEMM with the SM100 NVFP4 backward"
            )
        if deep_gemm.get_mk_alignment_for_contiguous_layout() != 128:
            raise ValueError(
                "Native backward requires 128-row grouped-gradient alignment"
            )
        self.runtime = runtime
        self.num_sms = runtime.num_sms
        self.group = runtime.group
        self.rank, self.world = (
            dist.get_rank(self.group),
            dist.get_world_size(self.group),
        )
        hosts = [None] * self.world
        dist.all_gather_object(hosts, socket.gethostname(), group=self.group)
        if len(set(hosts)) != 1:
            raise ValueError("Native persistent backward currently requires one host")
        floor = owned_backward_ring_tokens(
            runtime.num_sms, "MONO_EP_NVFP4_BACKWARD_RING_WAVES", default_waves=1
        )
        self.bounded = workspace_bound.mode() == "auto"
        self.buffer = deep_gemm.get_symm_buffer_for_mono_ep(
            self.group,
            runtime.experts,
            runtime.max_tokens,
            runtime.topk,
            runtime.hidden,
            runtime.intermediate,
            mma_type="bf16xbf16",
            num_min_ring_tokens=floor,
            # the owned schedule keeps one block live per cluster, so the floor is all it needs
            num_exact_ring_tokens=floor if self.bounded else 0,
        )
        self.handles = []
        capacity = self.buffer.num_max_tokens_per_rank
        hidden = runtime.hidden
        self.dy, self.dy_pointers = self._allocate((capacity, hidden), torch.bfloat16)
        self.ds_partials, self.ds_pointers = self._allocate(
            (capacity, runtime.topk, 2), torch.float32
        )
        self.experts = runtime.experts // self.world
        rows = self.world * capacity * runtime.topk + self.experts * POOL_BLOCK
        rows = (rows + POOL_BLOCK - 1) // POOL_BLOCK * POOL_BLOCK
        self.worst_rows = rows
        self.bound = None
        if self.bounded:
            self.bound = workspace_bound.WorkspaceBound(
                capacity, runtime.topk, self.experts, rows
            )
            rows = self.bound.rows
            if runtime.forward_ring_budget:
                self.bound.forward_budget = (
                    runtime.forward_ring_budget + self.experts * POOL_BLOCK
                )
        device = self.dy.device
        self.starts = torch.empty(self.experts, device=device, dtype=torch.int32)
        self.counts = torch.empty_like(self.starts)
        self.finished = None
        self._build_pools(rows)

    def _build_pools(self, rows):
        """BF16 x/dy/h/dz pools, packed FP4 x/dy/dz pools with ``[K / 64][rows]`` scale words,
        and the per-row dy/dz scales, zero-initialised, for ``rows`` routed rows."""
        hidden, inter = self.runtime.hidden, self.runtime.intermediate
        device = self.dy.device
        self.rows = rows
        zeros = lambda *shape, dtype: torch.zeros(*shape, device=device, dtype=dtype)  # noqa: E731
        self.pools = [
            zeros(rows, w, dtype=torch.bfloat16)
            for w in (hidden, hidden, inter, 2 * inter)
        ]
        self.fp4_pools = [
            zeros(rows, w // 2, dtype=torch.uint8) for w in (hidden, hidden, 2 * inter)
        ]
        self.sf_pools = [
            zeros(w // 64, rows, dtype=torch.int32) for w in (hidden, hidden, 2 * inter)
        ]
        self.row_scales = [zeros(rows, dtype=torch.float32) for _ in range(2)]

    def pool_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (*self.pools, *self.fp4_pools, *self.sf_pools, *self.row_scales)
        )

    def _fit_pools(self, key):
        """Bounded workspace: grow every pool before a backward whose routed rows exceed them."""
        if self.bound is None:
            return
        if key is None:
            self.bound.unrecorded += 1
            LOG.warning(
                "mono_ep bounded workspace: an NVFP4 backward without a layer key; sizing "
                "its pools for the worst case (%d rows)",
                self.worst_rows,
            )
            rows = self.bound.target(self.worst_rows)
        else:
            rows = self.bound.target(self.bound.required_rows(key))
        if rows is None:
            return
        del self.pools, self.fp4_pools, self.sf_pools, self.row_scales
        self._build_pools(rows)

    def _allocate(self, shape, dtype):
        allocator = torch if self.world == 1 else symm_mem
        local = allocator.empty(*shape, dtype=dtype, device="cuda")
        if self.world == 1:
            self.handles.append(local)
            return local, [local.data_ptr()]
        handle = symm_mem.rendezvous(local, group=self.group)
        self.handles.append((local, handle))
        return local, [
            handle.get_buffer(r, shape, dtype).data_ptr() for r in range(self.world)
        ]

    def backward(self, dy, x, indices, scores, merged, key=None):
        """``key`` is the layer's frozen base tensor (``Nvfp4LoraExperts.q1``) whose
        forward recorded this rank's receive counts; without it a bounded backward
        sizes its pools for the worst case."""
        import deep_gemm

        if (
            deep_gemm.get_num_sms() != self.num_sms
            or self.runtime.buffer.num_sms != self.num_sms
            or self.buffer.num_sms != self.num_sms
        ):
            raise RuntimeError(
                "persistent backward SM count changed after workspace allocation"
            )
        if x.dtype != torch.bfloat16 or dy.dtype != torch.bfloat16:
            raise ValueError("Native backward requires BF16 activations and gradients")
        stream = torch.cuda.current_stream(x.device)
        if self.finished is not None:
            stream.wait_event(self.finished)
        self._fit_pools(key)
        tokens = x.shape[0]
        self.buffer.x[:tokens].copy_(x)
        self.dy[:tokens].copy_(dy)
        self.buffer.topk_idx[:tokens].copy_(indices)
        self.buffer.topk_weights[:tokens].copy_(scores)
        dx = torch.empty_like(x)
        ds = torch.empty_like(scores, dtype=torch.float32)
        # The launcher reads the options from the environment at every call
        saved = os.environ.get(OPTIONS_ENV)
        if self.options is not None:
            os.environ[OPTIONS_ENV] = str(self.options)
        try:
            self._launch(dx, ds, merged)
        finally:
            if self.options is not None:
                if saved is None:
                    os.environ.pop(OPTIONS_ENV)
                else:
                    os.environ[OPTIONS_ENV] = saved
        self.finished = torch.cuda.Event()
        self.finished.record(stream)
        return dx, ds

    def _launch(self, dx, ds, merged):
        import deep_gemm

        deep_gemm._C.nvfp4_mono_ep_backward(
            dx,
            ds,
            merged.w1.view(torch.int8),
            merged.w1_atoms,
            merged.w2t.view(torch.int8),
            merged.w2t_atoms,
            merged.w1t.view(torch.int8),
            merged.w1t_atoms,
            merged.scales_backward,
            self.pools,
            [p.view(torch.int8) for p in self.fp4_pools],
            self.sf_pools,
            self.row_scales,
            self.starts,
            self.counts,
            self.buffer.buffer,
            self.buffer.handle.buffer_ptrs,
            self.dy,
            self.dy_pointers,
            self.ds_partials,
            self.ds_pointers,
            self.rank,
            self.buffer.num_max_tokens_per_rank,
            self.buffer.l1_acts.shape[0],
            self.runtime.topk,
        )


def backward_for(runtime):
    backward = getattr(runtime, "_nvfp4_backward", None)
    if backward is None:
        backward = Nvfp4PersistentBackward(runtime)
        runtime._nvfp4_backward = backward
    return backward


class _Nvfp4LoraMoE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, indices, scores, a1, b1, a2, b2, runtime, experts, cache, record
    ):
        merged = cache.forward(experts, (a1, b1, a2, b2))
        y = runtime.forward_nvfp4(
            x,
            indices,
            scores,
            merged.w1,
            merged.w1_atoms,
            merged.w2,
            merged.w2_atoms,
            merged.scales_forward,
            record_key=experts.q1 if record else None,
        )
        ctx.save_for_backward(x, indices, scores, a1, b1, a2, b2)
        # the merged forms are fetched again in the backward (a layer cache rebuilds evicted ones)
        ctx.runtime, ctx.experts, ctx.cache = runtime, experts, cache
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, indices, scores, a1, b1, a2, b2 = ctx.saved_tensors
        backward = backward_for(ctx.runtime)
        merged = ctx.cache.backward(ctx.experts, (a1, b1, a2, b2))
        dx, ds = backward.backward(
            dy.contiguous(), x, indices, scores, merged, key=ctx.experts.q1
        )
        da1, db1, da2, db2 = lora_weight_grads(
            backward.pools,
            backward.starts,
            backward.counts,
            a1,
            b1,
            a2,
            b2,
            ctx.experts.scaling,
        )
        return (
            dx,
            None,
            ds.to(scores.dtype),
            da1.to(a1.dtype),
            db1.to(b1.dtype),
            da2.to(a2.dtype),
            db2.to(b2.dtype),
            None,
            None,
            None,
            None,
        )


def nvfp4_lora_moe(x, indices, scores, a1, b1, a2, b2, runtime, experts, cache):
    """Routed MoE on frozen packed NVFP4 experts plus LoRA factors, differentiable
    in ``x``, ``scores`` and the factors; the merged weights come from ``cache``."""
    if runtime.compute_dtype != "nvfp4_lora":
        raise ValueError(
            "nvfp4_lora_moe requires a PersistentMoE with compute_dtype='nvfp4_lora'"
        )
    record = workspace_bound.will_backward(x, scores, a1, b1, a2, b2)
    if record:
        # the backward's bounded pools are sized from counts this forward records, so its
        # workspace (and the record keyed on the layer's base) must exist before the first forward
        backward_for(runtime)
    return _Nvfp4LoraMoE.apply(
        x, indices, scores, a1, b1, a2, b2, runtime, experts, cache, record
    )
