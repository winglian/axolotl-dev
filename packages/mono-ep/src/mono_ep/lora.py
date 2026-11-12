# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""LoRA on the fused expert kernels: frozen base experts, trainable per-expert factors.

The kernels run on the merged weights ``W + scaling * B_e @ A_e``, built from the
frozen base and the live adapters and held by a :class:`MergedWeightCache` that an
optimizer post-step hook invalidates, so every optimizer step merges once more and
gradient-accumulation micro-batches reuse the merge. The merged weights are not
autograd inputs (straight-through): the backward returns ``dx``, ``dscore`` and
the adapter gradients from rank-``r`` grouped GEMMs on the backward's BF16 pools;
the full ``dW`` is never formed. ``nvfp4_lora`` requantizes packed NVFP4 bases
(:mod:`mono_ep.nvfp4_lora`).
"""

import os
from collections import OrderedDict
from dataclasses import dataclass
from itertools import count

import torch

from . import workspace_bound

POOL_BLOCK = 128
_LAYER_KEYS = count(1)
MERGED_CACHE_MODES = ("layer", "all")
# Layers whose merged forms a ``layer`` cache keeps: the checkpoint recompute and the backward of
# one layer are adjacent, a second slot covers a layer called twice in a row
MERGED_CACHE_LAYERS = 2


WGRAD_MODES = ("grouped", "triton", "loop")


def wgrad_mode(default="grouped"):
    """``MONO_EP_LORA_WGRAD`` (or the older ``MONO_EP_NVFP4_LORA_WGRAD``): ``grouped``
    (``torch._grouped_mm``), ``triton`` (the two-pass kernels of :mod:`mono_ep.lora_grads`) or
    ``loop`` (per-expert reference); ``default`` when unset."""
    value = os.environ.get(
        "MONO_EP_LORA_WGRAD", os.environ.get("MONO_EP_NVFP4_LORA_WGRAD", default)
    )
    if value not in WGRAD_MODES:
        raise ValueError(f"MONO_EP_LORA_WGRAD must be one of {WGRAD_MODES}")
    return value


def merged_cache_mode(mode=None, env="MONO_EP_LORA_MERGED_CACHE"):
    """``layer`` (default: forms built just in time, the last ``MERGED_CACHE_LAYERS``
    layers kept) or ``all`` (every layer's forms held across the optimizer step);
    ``env`` when not given."""
    mode = mode or os.environ.get(env, "layer")
    if mode not in MERGED_CACHE_MODES:
        raise ValueError(f"{env} must be one of {MERGED_CACHE_MODES}")
    return mode


class MergedWeightCache:
    """Merged weights per layer (keyed by the layer's experts object),
    invalidated by ``attach_post_step_hook`` after every ``optimizer.step``.

    The experts object builds the forms (``merged``, ``merged_forward``,
    ``merged_backward`` from the full adapters; entries expose ``has_forward``,
    ``has_backward``, ``nbytes()`` and ``version``). ``mode="all"``: a layer's
    forms are built at its first use after an invalidation and held until the
    next one. ``mode="layer"`` (default, see :func:`merged_cache_mode`):
    :meth:`forward` builds the forward forms right before a forward,
    :meth:`backward` the backward forms right before a backward, and only the
    ``capacity`` most recently used layers stay resident, so the persistent
    footprint is a few layers instead of the model.
    Under activation checkpointing the recompute forward rebuilds the forward
    forms (counted in ``forward_builds``); every build is bitwise reproducible
    from the same adapters, so a rebuilt form equals the evicted one.
    """

    env = "MONO_EP_LORA_MERGED_CACHE"

    def __init__(self, mode=None, capacity=MERGED_CACHE_LAYERS):
        self.mode = merged_cache_mode(mode, self.env)
        self.capacity = int(capacity) if self.mode == "layer" else None
        self.version = 0
        self.entries = OrderedDict()
        self.builds = 0
        self.forward_builds = 0
        self.backward_builds = 0
        self.hits = 0
        self.evictions = 0
        self.max_entries = 0
        self.max_resident_bytes = 0

    def invalidate(self):
        self.version += 1
        self.entries.clear()

    def resident_bytes(self):
        return sum(e.nbytes() for e in self.entries.values())

    def stats(self):
        return dict(
            mode=self.mode,
            version=self.version,
            builds=self.builds,
            forward_builds=self.forward_builds,
            backward_builds=self.backward_builds,
            hits=self.hits,
            evictions=self.evictions,
            entries=len(self.entries),
            max_entries=self.max_entries,
            resident_bytes=self.resident_bytes(),
            max_resident_bytes=self.max_resident_bytes,
        )

    def _current(self, experts):
        entry = self.entries.get(id(experts))
        if entry is not None and entry.version != self.version:
            del self.entries[id(experts)]
            entry = None
        return entry

    def _store(self, experts, entry):
        self.entries[id(experts)] = entry
        self.entries.move_to_end(id(experts))
        self.builds += 1
        if self.capacity is not None:
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
                self.evictions += 1
        self.max_entries = max(self.max_entries, len(self.entries))
        self.max_resident_bytes = max(self.max_resident_bytes, self.resident_bytes())
        return entry

    def get(self, experts, adapters):
        """Every form of a layer (built if any is missing)."""
        entry = self._current(experts)
        if entry is not None and entry.has_forward and entry.has_backward:
            self.hits += 1
            self.entries.move_to_end(id(experts))
            return entry
        full = [_full_tensor(t) for t in adapters]
        return self._store(experts, experts.merged(*full, version=self.version))

    def forward(self, experts, adapters):
        """The forward forms of a layer."""
        if self.mode == "all":
            return self.get(experts, adapters)
        entry = self._current(experts)
        if entry is not None and entry.has_forward:
            self.hits += 1
            self.entries.move_to_end(id(experts))
            return entry
        full = [_full_tensor(t) for t in adapters]
        self.forward_builds += 1
        return self._store(experts, experts.merged_forward(*full, version=self.version))

    def backward(self, experts, adapters):
        """The backward forms of a layer."""
        if self.mode == "all":
            return self.get(experts, adapters)
        entry = self._current(experts)
        if entry is not None and entry.has_backward:
            self.hits += 1
            self.entries.move_to_end(id(experts))
            return entry
        full = [_full_tensor(t) for t in adapters]
        self.backward_builds += 1
        return self._store(
            experts, experts.merged_backward(*full, version=self.version)
        )


def _full_tensor(t):
    t = t.detach()
    if hasattr(t, "full_tensor"):
        t = t.full_tensor()
    return t


def attach_post_step_hook(optimizer, cache):
    """Invalidate the merged weights after each ``optimizer.step`` so the next
    forward merges the updated adapters."""
    return optimizer.register_step_post_hook(lambda *_: cache.invalidate())


def padded_offsets(starts, counts):
    """Cumulative ends of the block-padded expert ranges in the pools (int32)."""
    last = starts[-1:] + (counts[-1:] + POOL_BLOCK - 1) // POOL_BLOCK * POOL_BLOCK
    return torch.cat([starts[1:], last]).to(torch.int32)


def lora_weight_grads(
    pools, starts, counts, a1, b1, a2, b2, scaling, default="grouped"
):
    """``dA, dB`` of both projections from the BF16 pools (fp32 outputs):

    ``W1: P = dz @ B1_e, dA1_e = s * Pᵀ @ x; Q = x @ A1_eᵀ, dB1_e = s * dzᵀ @ Q``
    (``W2`` likewise with ``dy`` and ``h``). The padded pool rows hold zero
    ``h`` / ``dz`` and finite stale ``x`` / ``dy``, so they contribute nothing.
    """
    x_pool, dy_pool, h_pool, dz_pool = pools
    mode = wgrad_mode(default)
    if mode == "loop":
        return _lora_weight_grads_loop(pools, starts, counts, a1, b1, a2, b2, scaling)
    if mode == "triton":
        from .lora_grads import lora_weight_grads_triton

        return lora_weight_grads_triton(pools, starts, counts, a1, b1, a2, b2, scaling)
    offs = padded_offsets(starts, counts)
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


def project_weight_grads(dw1, dw2, a1, b1, a2, b2, scaling):
    """``dA = s Bᵀ dW``, ``dB = s dW Aᵀ`` from full fp32 weight gradients (the route for
    backwards that keep no BF16 pools)."""
    f = lambda t: t.detach().float()  # noqa: E731
    return (
        scaling * f(b1).transpose(1, 2) @ dw1,
        scaling * dw1 @ f(a1).transpose(1, 2),
        scaling * f(b2).transpose(1, 2) @ dw2,
        scaling * dw2 @ f(a2).transpose(1, 2),
    )


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


@dataclass
class MergedWeights:
    """One step's merged BF16 experts ``w1 [E, 2I, H]`` (gate rows then up rows) and
    ``w2 [E, H, I]``. The kernel layouts and quantized forms derive from them through
    :mod:`mono_ep.weight_cache`, keyed by these tensors' storage, so a rebuilt merge
    never reuses a previous step's forms."""

    w1: torch.Tensor
    w2: torch.Tensor
    version: int
    has_forward = True
    has_backward = True

    def tensors(self):
        return [self.w1, self.w2]

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors())


@dataclass
class MergedMxfp8Forms:
    """One step's merged experts as the four SM100 MXFP8 forms the fused kernels read
    (``sm100_mx_w1``, ``sm100_mx_w1t``, ``sm100_mx_w2``, ``sm100_mx_w2t``), built by the
    CUDA forms builder with the merge fused in; the merged BF16 weight never exists."""

    forms: dict
    version: int
    has_forward = True
    has_backward = True

    def tensors(self):
        """Every form's values (as bytes) and scale words, in a fixed order."""
        return [
            t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t
            for name in sorted(self.forms)
            for t in self.forms[name]
        ]

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors())


def mxfp8_forms_mode():
    """``MONO_EP_LORA_MXFP8_FORMS``: ``fused`` (default: the merge runs inside the CUDA MXFP8
    forms builder) or ``merge`` (merge to BF16, then the full fine-tune's forms)."""
    value = os.environ.get("MONO_EP_LORA_MXFP8_FORMS", "fused")
    if value not in ("fused", "merge"):
        raise ValueError("MONO_EP_LORA_MXFP8_FORMS must be fused or merge")
    return value


def mxfp8_pools_mode():
    """``MONO_EP_LORA_MXFP8_POOLS``: ``fp8`` (default: the MXFP8 backward writes no BF16 x / dy / dz
    copies and the Triton adapter-gradient kernels read the E4M3 pools, dz after its E4M3 rounding)
    or ``bf16`` (it also writes BF16 copies, read by ``MONO_EP_LORA_WGRAD``'s route)."""
    value = os.environ.get("MONO_EP_LORA_MXFP8_POOLS", "fp8")
    if value not in ("bf16", "fp8"):
        raise ValueError("MONO_EP_LORA_MXFP8_POOLS must be bf16 or fp8")
    return value


def _fused_mxfp8_forms(runtime, w1):
    """Whether this call's merged weights are built as MXFP8 forms with the merge fused in."""
    native = runtime._peer_backward
    if runtime.compute_dtype != "mxfp8" or w1.dtype != torch.bfloat16:
        return False
    if native.precision() != "mxfp8" or mxfp8_forms_mode() != "fused":
        return False
    from .mx_weight_forms import cuda_forms_available

    return cuda_forms_available(w1.device)


def merge(w, lora_b, lora_a, scaling):
    """``bf16(w + scaling * B_e @ A_e)`` per expert, one rounding of the fp32 result;
    the factors enter in BF16, as the NVFP4 requant takes them."""
    bf = torch.bfloat16
    return torch.baddbmm(
        w.detach().to(bf), lora_b.to(bf), lora_a.to(bf), alpha=float(scaling)
    )


class LoraExperts:
    """One MoE layer's frozen base experts and its LoRA scaling: the key of its
    :class:`MergedWeightCache` entry. The base is bound only around a cache access
    (:meth:`bind`), since FSDP2 hands the layer a regathered unsharded tensor each forward
    and backward."""

    def __init__(self, *, scaling):
        self.scaling = float(scaling)
        self.w1 = self.w2 = None
        self.mxfp8_forms = False
        self.key = next(_LAYER_KEYS)

    def bind(self, w1, w2, mxfp8_forms=False):
        self.w1, self.w2, self.mxfp8_forms = w1, w2, mxfp8_forms
        return self

    def merged(self, a1, b1, a2, b2, version=0):
        if self.w1 is None:
            raise RuntimeError("LoraExperts.merged before bind(w1, w2)")
        if self.mxfp8_forms:
            from .mx_weight_forms import cuda_mx_lora_forms

            return MergedMxfp8Forms(
                cuda_mx_lora_forms(
                    self.w1.detach(), self.w2.detach(), a1, b1, a2, b2, self.scaling
                ),
                version,
            )
        return MergedWeights(
            merge(self.w1, b1, a1, self.scaling),
            merge(self.w2, b2, a2, self.scaling),
            version,
        )

    merged_forward = merged
    merged_backward = merged


LORA_COMPUTE_DTYPES = ("bf16", "mxfp8", "blockwise")
# Adapter-gradient route of bf16_lora / mxfp8_lora by architecture: the Triton kernels won the paired
# 8xB200 runs at every shape; on SM90 they are unvalidated, so torch._grouped_mm stays
TRITON_WGRAD_DEFAULT = {9: "grouped", 10: "triton"}


FORWARD_OP = "mono_ep::lora_forward"
BACKWARD_OP = "mono_ep::lora_backward"
# (experts, cache) per LoraExperts key, so the custom ops take only tensors and ints
_LAYERS: dict[int, tuple] = {}


@torch.library.custom_op(FORWARD_OP, mutates_args=())
def lora_forward(
    x: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
    runtime_key: int,
    layer_key: int,
) -> torch.Tensor:
    from .runtime import _RUNTIMES, _forward_impl

    runtime = _RUNTIMES[runtime_key]
    experts, cache = _LAYERS[layer_key]
    merged = cache.forward(
        experts.bind(w1, w2, _fused_mxfp8_forms(runtime, w1)), (a1, b1, a2, b2)
    )
    experts.bind(None, None)
    if isinstance(merged, MergedMxfp8Forms):
        f = merged.forms
        return _forward_impl(
            x,
            indices,
            scores,
            w1,
            w2,
            runtime,
            record_key=w1,
            mx_forms=(f["sm100_mx_w1"], f["sm100_mx_w2"]),
        )[0]
    return _forward_impl(
        x, indices, scores, merged.w1, merged.w2, runtime, record_key=w1
    )[0]


@lora_forward.register_fake
def _lora_forward_fake(
    x, indices, scores, w1, w2, a1, b1, a2, b2, runtime_key, layer_key
):
    return torch.empty_like(x)


@torch.library.custom_op(BACKWARD_OP, mutates_args=())
def lora_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
    runtime_key: int,
    layer_key: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    from .runtime import _RUNTIMES

    native = _RUNTIMES[runtime_key]._peer_backward
    experts, cache = _LAYERS[layer_key]
    adapters = (a1, b1, a2, b2)
    merged = cache.backward(
        experts.bind(w1, w2, _fused_mxfp8_forms(_RUNTIMES[runtime_key], w1)), adapters
    )
    experts.bind(None, None)
    mxfp8 = native.arch == 10 and not native.blockwise and native.precision() == "mxfp8"
    if mxfp8:
        fp8 = mxfp8_pools_mode() == "fp8"
        if fp8 and wgrad_mode("triton") != "triton":
            raise ValueError(
                f"MONO_EP_LORA_WGRAD={wgrad_mode()} needs the BF16 pools; mxfp8_lora reads the "
                "E4M3 pools with the Triton kernels by default, set MONO_EP_LORA_MXFP8_POOLS=bf16 "
                "for the grouped or loop route"
            )
        mx_forms = None
        if isinstance(merged, MergedMxfp8Forms):
            f = merged.forms
            mx_forms = (f["sm100_mx_w1"], f["sm100_mx_w2t"], f["sm100_mx_w1t"])
        dx, ds, _, _ = native.backward(
            dy,
            x,
            indices,
            scores,
            *((w1, w2) if mx_forms is not None else (merged.w1, merged.w2)),
            record_key=w1,
            weight_grads=False,
            mx_forms=mx_forms,
            fp8_pools=fp8,
        )
        if fp8:
            from .lora_grads import lora_weight_grads_triton

            grads = lora_weight_grads_triton(
                native.pools[:4],
                native.starts,
                native.counts,
                *adapters,
                experts.scaling,
                fp8_pools=native.mxfp8_pools(),
            )
        else:
            grads = lora_weight_grads(
                native.pools[:4],
                native.starts,
                native.counts,
                *adapters,
                experts.scaling,
                default=TRITON_WGRAD_DEFAULT[native.arch],
            )
    elif native.blockwise:
        # the blockwise FP8 backward keeps FP8 operands only: project its full dW
        dx, ds, dw1, dw2 = native.backward(
            dy, x, indices, scores, merged.w1, merged.w2, record_key=w1
        )
        grads = project_weight_grads(dw1, dw2, *adapters, experts.scaling)
    else:
        dx, ds, _, _ = native.backward(
            dy,
            x,
            indices,
            scores,
            merged.w1,
            merged.w2,
            record_key=w1,
            weight_grads=False,
        )
        grads = lora_weight_grads(
            native.pools[:4],
            native.starts,
            native.counts,
            *adapters,
            experts.scaling,
            default=TRITON_WGRAD_DEFAULT[native.arch],
        )
    return (
        dx,
        ds,
        *(g.to(t.dtype) for g, t in zip(grads, adapters, strict=True)),
    )


@lora_backward.register_fake
def _lora_backward_fake(dy, x, indices, scores, w1, w2, a1, b1, a2, b2, *keys):
    return tuple(torch.empty_like(t) for t in (x, scores, a1, b1, a2, b2))


def _setup_context(ctx, inputs, output):
    *tensors, runtime_key, layer_key = inputs
    ctx.save_for_backward(*tensors)
    ctx.keys = (runtime_key, layer_key)


def _autograd_backward(ctx, dy):
    dx, ds, *grads = lora_backward(dy.contiguous(), *ctx.saved_tensors, *ctx.keys)
    return (dx, None, ds, None, None, *grads, None, None)


lora_forward.register_autograd(_autograd_backward, setup_context=_setup_context)


def lora_moe(x, indices, scores, w1, w2, a1, b1, a2, b2, runtime, experts, cache):
    """Routed MoE on frozen base experts ``w1 [E_local, 2I, H]``, ``w2 [E_local, H, I]``
    plus adapters ``A1 [E_local, r, H]``, ``B1 [E_local, 2I, r]``, ``A2 [E_local, r, I]``,
    ``B2 [E_local, H, r]``, differentiable in ``x``, ``scores`` and the factors (the base
    gets no gradient). ``runtime`` is a BF16, SM100 MXFP8 or SM90 blockwise FP8
    :class:`PersistentMoE` with the native backward; ``experts`` the layer's :class:`LoraExperts`; ``cache`` the
    :class:`MergedWeightCache` an optimizer post-step hook invalidates. Runs as the
    ``mono_ep::lora_forward`` custom op, so selective activation checkpointing can save
    its output like ``mono_ep::persistent_forward``."""
    if runtime.compute_dtype not in LORA_COMPUTE_DTYPES:
        raise ValueError(
            f"lora_moe supports compute dtypes {LORA_COMPUTE_DTYPES}, not "
            f"{runtime.compute_dtype!r} (nvfp4_lora has its own path)"
        )
    if runtime._peer_backward is None or not hasattr(
        runtime._peer_backward, "precision"
    ):
        raise ValueError("lora_moe needs the native persistent backward")
    _LAYERS[experts.key] = (experts, cache)
    runtime._record_forward = workspace_bound.will_backward(x, scores, a1, b1, a2, b2)
    return lora_forward(
        x, indices, scores, w1, w2, a1, b1, a2, b2, runtime._key, experts.key
    )
