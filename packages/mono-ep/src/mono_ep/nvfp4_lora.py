# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""NVFP4 LoRA expert path on the SM100 persistent kernels (``precision="nvfp4_lora"``).

The base experts stay packed NVFP4 and frozen; only the LoRA factors train.
Per optimizer step the requant kernel rebuilds the merged forms the kernels
consume (``w1`` with interleaved gate/up rows, ``w2``, and the transposed
``w1t`` / ``w2t`` for the backward GEMMs). The forward runs block-scaled UMMA
on them; the backward returns ``dx``, ``dscore`` and the adapter gradients
from rank-``r`` grouped GEMMs on the BF16 pools. No base-weight gradients and
no fp32 masters of the base.
"""

import os
import socket
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.autograd.function import once_differentiable

from .persistent_backward import owned_backward_ring_tokens

POOL_BLOCK = 128


def requant_modules():
    """The vendored packed requant kernel and references (``mono_ep.nvfp4_requant``)."""
    from . import nvfp4_requant as pkg

    return types.SimpleNamespace(
        requant=pkg.nvfp4_requant, triton_requant=pkg.triton_nvfp4_requant
    )


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
    ``int32`` UTCCP scale-atom views) with the per-expert scale tables."""

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


class Nvfp4LoraExperts:
    """The frozen packed NVFP4 base of one MoE layer (``q1 [E, 2I, H/2]`` u8 in
    concat gate/up order, ``s1 [E, 2I, H/16]`` e4m3, ``pts1 [E]`` fp32; ``q2
    [E, H, I/2]``, ``s2``, ``pts2``), its interleaved and transposed forms, and
    the requant of the merged weights for given LoRA factors.

    ``input_scale`` is the checkpoint's ``(pts_x, pts_h)`` pair for the static
    activation quantization of the forward (``(1, 1)`` for weight-only exports).
    The dy and dz operands of the backward are scaled per row by the kernel.
    """

    def __init__(self, q1, s1, pts1, q2, s2, pts2, *, scaling, input_scale=(1.0, 1.0)):
        mods = requant_modules()
        self.requant, self.triton_requant = mods.requant, mods.triton_requant
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
        self.q1i = self.q1.index_select(1, self.perm).contiguous()
        self.s1i = self.s1.index_select(1, self.perm).contiguous()
        # Transposed bases for the backward's K-major operands (dequant -> transpose -> quantize, same scales)
        self.q1t, self.s1t = self._transposed(self.q1, self.s1, self.pts1)
        self.q2t, self.s2t = self._transposed(self.q2, self.s2, self.pts2)
        pts_x, pts_h = (float(v) for v in input_scale)
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

    def _transposed(self, q, s, pts):
        e, n, k2 = q.shape
        k = 2 * k2
        dq = self.requant.dequantize_nvfp4_packed_ref(
            q.view(e * n, k2),
            s.view(e * n, k2 // 8),
            pts.repeat_interleave(n),
            torch.float32,
        ).view(e, n, k)
        packed, scale = self.requant.quantize_nvfp4_packed_ref(
            dq.transpose(1, 2).reshape(e * k, n), pts.repeat_interleave(k)
        )
        return packed.view(e, k, n // 2).contiguous(), scale.view(
            e, k, n // 16
        ).contiguous()

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


class RequantCache:
    """Merged weights per layer, rebuilt lazily after every invalidation
    (``attach_post_step_hook`` invalidates after ``optimizer.step``)."""

    def __init__(self):
        self.version = 0
        self.entries = {}
        self.builds = 0

    def invalidate(self):
        self.version += 1
        self.entries.clear()

    def get(self, experts, adapters):
        entry = self.entries.get(id(experts))
        if entry is None or entry.version != self.version:
            full = [_full_tensor(t) for t in adapters]
            entry = experts.merged(*full, version=self.version)
            self.entries[id(experts)] = entry
            self.builds += 1
        return entry


def _full_tensor(t):
    t = t.detach()
    if hasattr(t, "full_tensor"):
        t = t.full_tensor()
    return t


def attach_post_step_hook(optimizer, cache):
    """Invalidate the merged weights after each ``optimizer.step`` so the next
    forward requantizes from the updated adapters."""
    return optimizer.register_step_post_hook(lambda *_: cache.invalidate())


class Nvfp4PersistentBackward:
    """Native SM100 NVFP4 backward: own BF16 symmetric buffer for the dy exchange,
    BF16 pools (x, dy, h, dz), packed FP4 pools with scale words (x, dy, dz) and
    the per-row dy/dz scales."""

    def __init__(self, runtime):
        import deep_gemm

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
        self.buffer = deep_gemm.get_symm_buffer_for_mono_ep(
            self.group,
            runtime.experts,
            runtime.max_tokens,
            runtime.topk,
            runtime.hidden,
            runtime.intermediate,
            mma_type="bf16xbf16",
            num_min_ring_tokens=owned_backward_ring_tokens(
                runtime.num_sms, "MONO_EP_NVFP4_BACKWARD_RING_WAVES", default_waves=1
            ),
        )
        self.handles = []
        capacity = self.buffer.num_max_tokens_per_rank
        hidden, inter = runtime.hidden, runtime.intermediate
        self.dy, self.dy_pointers = self._allocate((capacity, hidden), torch.bfloat16)
        self.ds_partials, self.ds_pointers = self._allocate(
            (capacity, runtime.topk, 2), torch.float32
        )
        self.experts = runtime.experts // self.world
        rows = self.world * capacity * runtime.topk + self.experts * POOL_BLOCK
        rows = (rows + POOL_BLOCK - 1) // POOL_BLOCK * POOL_BLOCK
        self.rows = rows
        device = self.dy.device
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
        self.starts = torch.empty(self.experts, device=device, dtype=torch.int32)
        self.counts = torch.empty_like(self.starts)
        self.finished = None

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

    def backward(self, dy, x, indices, scores, merged):
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
        tokens = x.shape[0]
        self.buffer.x[:tokens].copy_(x)
        self.dy[:tokens].copy_(dy)
        self.buffer.topk_idx[:tokens].copy_(indices)
        self.buffer.topk_weights[:tokens].copy_(scores)
        dx = torch.empty_like(x)
        ds = torch.empty_like(scores, dtype=torch.float32)
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
        self.finished = torch.cuda.Event()
        self.finished.record(stream)
        return dx, ds


def padded_offsets(starts, counts):
    """Cumulative ends of the block-padded expert ranges in the pools (int32)."""
    last = starts[-1:] + (counts[-1:] + POOL_BLOCK - 1) // POOL_BLOCK * POOL_BLOCK
    return torch.cat([starts[1:], last]).to(torch.int32)


def lora_weight_grads(pools, starts, counts, a1, b1, a2, b2, scaling):
    """``dA, dB`` of both projections from the BF16 pools (fp32 outputs):

    ``W1: P = dz @ B1_e, dA1_e = s * Pᵀ @ x; Q = x @ A1_eᵀ, dB1_e = s * dzᵀ @ Q``
    (``W2`` likewise with ``dy`` and ``h``). The padded pool rows hold zero
    ``h`` / ``dz`` and finite stale ``x`` / ``dy``, so they contribute nothing.
    """
    x_pool, dy_pool, h_pool, dz_pool = pools
    offs = padded_offsets(starts, counts)
    if os.environ.get("MONO_EP_NVFP4_LORA_WGRAD") == "loop":
        return _lora_weight_grads_loop(pools, starts, counts, a1, b1, a2, b2, scaling)
    bf = lambda t: t.detach().to(torch.bfloat16).contiguous()  # noqa: E731

    def rows_times(a, b):
        return torch._grouped_mm(a, b, offs=offs)

    def contract_rows(a, b):
        # fp32 accumulation, BF16 store (the output dtype must match the inputs)
        return torch._grouped_mm(a.t(), b, offs=offs).float()

    p1 = rows_times(dz_pool, bf(b1))
    q1 = rows_times(x_pool, bf(a1.transpose(1, 2)))
    p2 = rows_times(dy_pool, bf(b2))
    q2 = rows_times(h_pool, bf(a2.transpose(1, 2)))
    da1 = scaling * contract_rows(p1, x_pool)
    db1 = scaling * contract_rows(dz_pool, q1)
    da2 = scaling * contract_rows(p2, h_pool)
    db2 = scaling * contract_rows(dy_pool, q2)
    return da1, db1, da2, db2


def _lora_weight_grads_loop(pools, starts, counts, a1, b1, a2, b2, scaling):
    x_pool, dy_pool, h_pool, dz_pool = pools
    starts_l, counts_l = starts.tolist(), counts.tolist()
    da1, db1 = (
        torch.zeros_like(a1, dtype=torch.float32),
        torch.zeros_like(b1, dtype=torch.float32),
    )
    da2, db2 = (
        torch.zeros_like(a2, dtype=torch.float32),
        torch.zeros_like(b2, dtype=torch.float32),
    )
    for e, (s, c) in enumerate(zip(starts_l, counts_l, strict=True)):
        if c == 0:
            continue
        x, dy, h, dz = (
            p[s : s + c].float() for p in (x_pool, dy_pool, h_pool, dz_pool)
        )
        da1[e] = scaling * (dz @ b1[e].float()).t() @ x
        db1[e] = scaling * dz.t() @ (x @ a1[e].float().t())
        da2[e] = scaling * (dy @ b2[e].float()).t() @ h
        db2[e] = scaling * dy.t() @ (h @ a2[e].float().t())
    return da1, db1, da2, db2


def backward_for(runtime):
    backward = getattr(runtime, "_nvfp4_backward", None)
    if backward is None:
        backward = Nvfp4PersistentBackward(runtime)
        runtime._nvfp4_backward = backward
    return backward


class _Nvfp4LoraMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, scores, a1, b1, a2, b2, runtime, experts, merged):
        y = runtime.forward_nvfp4(
            x,
            indices,
            scores,
            merged.w1,
            merged.w1_atoms,
            merged.w2,
            merged.w2_atoms,
            merged.scales_forward,
        )
        ctx.save_for_backward(x, indices, scores, a1, b1, a2, b2)
        ctx.runtime, ctx.experts, ctx.merged = runtime, experts, merged
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, indices, scores, a1, b1, a2, b2 = ctx.saved_tensors
        backward = backward_for(ctx.runtime)
        dx, ds = backward.backward(dy.contiguous(), x, indices, scores, ctx.merged)
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
        )


def nvfp4_lora_moe(x, indices, scores, a1, b1, a2, b2, runtime, experts, cache):
    """Routed MoE on frozen packed NVFP4 experts plus LoRA factors, differentiable
    in ``x``, ``scores`` and the factors; the merged weights come from ``cache``."""
    if runtime.compute_dtype != "nvfp4_lora":
        raise ValueError(
            "nvfp4_lora_moe requires a PersistentMoE with compute_dtype='nvfp4_lora'"
        )
    merged = cache.get(experts, (a1, b1, a2, b2))
    return _Nvfp4LoraMoE.apply(
        x, indices, scores, a1, b1, a2, b2, runtime, experts, merged
    )
