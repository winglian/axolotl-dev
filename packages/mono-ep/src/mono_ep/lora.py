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

import torch

POOL_BLOCK = 128
MERGED_CACHE_MODES = ("layer", "all")
# Layers whose merged forms a ``layer`` cache keeps: the checkpoint recompute and the backward of
# one layer are adjacent, a second slot covers a layer called twice in a row
MERGED_CACHE_LAYERS = 2


def wgrad_mode():
    """``MONO_EP_LORA_WGRAD=loop`` (or the older ``MONO_EP_NVFP4_LORA_WGRAD``): per-expert
    adapter-gradient fallback instead of the grouped GEMMs."""
    return os.environ.get(
        "MONO_EP_LORA_WGRAD", os.environ.get("MONO_EP_NVFP4_LORA_WGRAD", "grouped")
    )


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


def lora_weight_grads(pools, starts, counts, a1, b1, a2, b2, scaling):
    """``dA, dB`` of both projections from the BF16 pools (fp32 outputs):

    ``W1: P = dz @ B1_e, dA1_e = s * Pᵀ @ x; Q = x @ A1_eᵀ, dB1_e = s * dzᵀ @ Q``
    (``W2`` likewise with ``dy`` and ``h``). The padded pool rows hold zero
    ``h`` / ``dz`` and finite stale ``x`` / ``dy``, so they contribute nothing.
    """
    x_pool, dy_pool, h_pool, dz_pool = pools
    offs = padded_offsets(starts, counts)
    if wgrad_mode() == "loop":
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
