# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""BF16 kernel layouts of the expert weights, reused until the weights change.

Contract. An entry belongs to one weight *storage* (the ``UntypedStorage``
object, its data pointer, the tensor's offset/shape/stride/dtype and its
autograd version) within one cache epoch. It is reused by every call in that
epoch that passes a tensor over the same storage, including fresh views
(``DTensor.to_local()``, saved-tensor unpacks), so forward, checkpoint
recompute and backward of one layer share one build.

The epoch ends, and every entry is dropped, when

* :func:`invalidate_weight_caches` is called. Callers whose weights are
  rewritten without a version bump MUST call it after every such write. FSDP2
  is the main case: it all-gathers each step into the same persistent unsharded
  storage under ``_unsafe_preserve_version_counter``, so the unsharded
  parameter keeps its object, storage and version while its values change
  (and the caching allocator usually hands it the same address). The Axolotl
  seam calls it from an optimizer post-step hook; so must any loader that
  writes into sharded parameters (checkpoint resume, weight sync).
* :func:`set_weight_cache_epoch` is called with a different value (for a
  caller that already counts optimizer steps).
* any ``torch.optim`` optimizer finishes ``step()``: a global post-step hook,
  on by default, calls :func:`invalidate_weight_caches` as a safety net
  (``MONO_EP_WEIGHT_CACHE_OPTIMIZER_HOOK=0`` disables it;
  :func:`register_optimizer_invalidation` attaches the same hook to a single
  optimizer). Optimizers that do not go through ``torch.optim.Optimizer.step``
  are not covered by it.

In-place updates through the tensor itself (``copy_``/``add_`` under
``no_grad``, ``out=`` ops) also change the key through the version counter.

Key selection (``MONO_EP_WEIGHT_CACHE_KEY``, default ``auto``): until a
caller supplies an epoch the key also holds the data pointer (``storage``), a
defensive extra that misses whenever FSDP2 re-gathers into a different address,
which happens for most calls within one step on B200 (1 hit in 9 calls for a
3-layer stack). Once :func:`set_weight_cache_epoch` has been called, the caller
has taken on the contract and the pointer is dropped (``storage_epoch``), so
forward, recompute and backward of a layer share one build. Both can be forced.
``MONO_EP_WEIGHT_CACHE_KEY=object`` restores the round-1 key (tensor
object identity + version + data pointer), which is NOT safe under FSDP2
without an explicit invalidation per step. ``MONO_EP_WEIGHT_CACHE=0``
disables caching (layouts rebuilt per call in fused passes) and ``legacy``
restores the original per-call op sequence. ``MONO_EP_WEIGHT_CACHE_ENTRIES``
bounds the number of (w1, w2) pairs kept (default 2, about 300 MB per pair for
Qwen3-30B-A3B at EP8; under FSDP2 each entry is an extra unsharded copy of one
layer's experts). All layouts are bitwise identical to computing them per call.
Layouts are built by Inductor-compiled cast/permute kernels (one tiled pass per
layout; 0.49 ms less per backward call with a weight update on one B200 than
the eager copies); ``MONO_EP_WEIGHT_CACHE_COMPILED=0`` restores the eager
builders.

The SM90 FP8 quantized weights (values + scales, :mod:`mono_ep.sm90_weight_forms`) live in
the same entries through :func:`cached_weight_forms`, so one key, epoch and invalidation covers
the BF16 layouts and the FP8 quantizations alike.
"""

import os
import weakref
from collections import OrderedDict

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

_ENTRIES: OrderedDict = OrderedDict()
_GENERATION = 0
_EPOCH = None
_HOOK_ENABLED = os.getenv("MONO_EP_WEIGHT_CACHE_OPTIMIZER_HOOK", "1") != "0"
_STATS = dict(hits=0, builds=0, layouts_built=0, invalidations=0, compile_fallbacks=0)


def invalidate_weight_caches():
    """End the cache epoch: drop every cached layout.

    Call after every optimizer step or other write that bypasses version
    counters (FSDP2 unsharded storage, ``.data`` mutation, raw/NCCL writes,
    ``set_``, checkpoint loads into sharded parameters).
    """
    global _GENERATION
    _GENERATION += 1
    _STATS["invalidations"] += 1
    _ENTRIES.clear()


def set_weight_cache_epoch(epoch):
    """Bind cached layouts to a caller-supplied epoch (e.g. the optimizer step)."""
    global _EPOCH
    if epoch != _EPOCH:
        _EPOCH = epoch
        _ENTRIES.clear()


def register_optimizer_invalidation(optimizer):
    """Invalidate after every ``optimizer.step()``; returns the hook handle."""
    return optimizer.register_step_post_hook(lambda *_: invalidate_weight_caches())


def set_optimizer_hook_enabled(enabled):
    """Turn the global ``torch.optim`` post-step invalidation hook on or off."""
    global _HOOK_ENABLED
    _HOOK_ENABLED = bool(enabled)


def cache_stats():
    """Counters: calls served from cache, calls that built, layouts built, invalidations."""
    return dict(_STATS)


def _global_optimizer_hook(*_):
    if _HOOK_ENABLED:
        invalidate_weight_caches()


register_optimizer_step_post_hook(_global_optimizer_hook)


def _mode():
    value = os.getenv("MONO_EP_WEIGHT_CACHE", "1")
    if value not in ("0", "1", "legacy"):
        raise ValueError("MONO_EP_WEIGHT_CACHE must be 0, 1 or legacy")
    return value


def _legacy(w1, w2, name):
    from deep_gemm.mega import _interleave_weights

    if name == "w1_interleaved":
        return _interleave_weights(w1.bfloat16())
    if name == "w2":
        return w2.bfloat16()
    return (w1 if name == "w1t" else w2).bfloat16().transpose(1, 2).contiguous()


def _capacity():
    value = int(os.getenv("MONO_EP_WEIGHT_CACHE_ENTRIES", "2"))
    if value < 1:
        raise ValueError("MONO_EP_WEIGHT_CACHE_ENTRIES must be positive")
    return value


def _key_mode():
    value = os.getenv("MONO_EP_WEIGHT_CACHE_KEY", "auto")
    if value not in ("auto", "storage", "storage_epoch", "object"):
        raise ValueError(
            "MONO_EP_WEIGHT_CACHE_KEY must be auto, storage, storage_epoch or object"
        )
    if value == "auto":
        return "storage" if _EPOCH is None else "storage_epoch"
    return value


def _tensor_key(t):
    return (t.data_ptr(), t._version, tuple(t.shape), t.stride(), t.dtype, t.device)


def _storage_key(t, with_pointer):
    return (
        t.untyped_storage()._cdata,
        t.storage_offset(),
        t.data_ptr() if with_pointer else None,
        t._version,
        tuple(t.shape),
        t.stride(),
        t.dtype,
        t.device,
    )


def _interleave_bf16(w1):
    """``_interleave_weights(w1.bfloat16())`` in one cast-and-permute pass."""
    g, n, k = w1.shape
    half, gran = n // 2, 8
    out = torch.empty((g, n, k), device=w1.device, dtype=torch.bfloat16)
    view = out.view(g, half // gran, 2, gran, k)
    view[:, :, 0].copy_(w1[:, :half].reshape(g, half // gran, gran, k))
    view[:, :, 1].copy_(w1[:, half:].reshape(g, half // gran, gran, k))
    return out


def _transposed_bf16(w):
    """``w.bfloat16().transpose(1, 2).contiguous()`` in one pass."""
    g, n, k = w.shape
    out = torch.empty((g, k, n), device=w.device, dtype=torch.bfloat16)
    return out.copy_(w.transpose(1, 2))


_EAGER_BUILDERS = {
    "w1_interleaved": lambda w1, w2: _interleave_bf16(w1),
    "w2": lambda w1, w2: w2 if w2.dtype == torch.bfloat16 else w2.bfloat16(),
    "w1t": lambda w1, w2: _transposed_bf16(w1),
    "w2t": lambda w1, w2: _transposed_bf16(w2),
}
_COMPILED: dict = {}


def _compiled_interleave(w1):
    g, n, k = w1.shape
    half = n // 2
    return (
        torch.stack(
            (
                w1[:, :half].reshape(g, half // 8, 8, k),
                w1[:, half:].reshape(g, half // 8, 8, k),
            ),
            2,
        )
        .reshape(g, n, k)
        .to(torch.bfloat16)
    )


def _compiled_transpose(w):
    return w.transpose(1, 2).to(torch.bfloat16).contiguous()


def _compiled(name, w1, w2):
    """Inductor-generated cast/permute kernels; ``MONO_EP_WEIGHT_CACHE_COMPILED=0`` restores eager copies."""
    if name == "w2" or _COMPILED.get(name, True) is False:
        return _EAGER_BUILDERS[name](w1, w2)
    function = _COMPILED.get(name)
    if function is None:
        function = torch.compile(
            _compiled_interleave if name == "w1_interleaved" else _compiled_transpose,
            dynamic=False,
        )
        _COMPILED[name] = function
    try:
        return function(w1 if name in ("w1_interleaved", "w1t") else w2)
    except Exception:  # noqa: BLE001
        # Fall back to the bitwise-identical eager copies if Inductor is unavailable here
        _COMPILED[name] = False
        _STATS["compile_fallbacks"] += 1
        return _EAGER_BUILDERS[name](w1, w2)


def _build(name, w1, w2):
    if os.getenv("MONO_EP_WEIGHT_CACHE_COMPILED", "1") == "1":
        return _compiled(name, w1, w2)
    return _EAGER_BUILDERS[name](w1, w2)


_BUILDERS = {
    name: (lambda w1, w2, name=name: _build(name, w1, w2)) for name in _EAGER_BUILDERS
}


def weight_layouts(w1, w2, *names):
    """Return the named BF16 layouts of ``(w1, w2)``, building only what is missing."""
    mode = _mode()
    if mode == "legacy":
        return tuple(_legacy(w1, w2, name) for name in names)
    if mode == "0":
        return tuple(_BUILDERS[name](w1, w2) for name in names)
    key_mode = _key_mode()
    if key_mode == "object":
        entry = _object_entry(w1, w2)
    else:
        entry = _storage_entry(w1, w2, key_mode == "storage")
    layouts = entry["layouts"]
    missing = [name for name in names if name not in layouts]
    _STATS["builds" if missing else "hits"] += 1
    _STATS["layouts_built"] += len(set(missing))
    for name in missing:
        if name not in layouts:
            layouts[name] = _BUILDERS[name](w1, w2)
    return tuple(layouts[name] for name in names)


def cached_weight_forms(w1, w2, names, build):
    """Named derived forms of ``(w1, w2)`` under the same key and epoch contract.

    ``build(missing)`` returns a dict with at least every missing name; extra names it returns
    are kept, so one fused pass can fill the forms later calls of the same step ask for.
    """
    key_mode = _key_mode()
    if key_mode == "object":
        entry = _object_entry(w1, w2)
    else:
        entry = _storage_entry(w1, w2, key_mode == "storage")
    layouts = entry["layouts"]
    missing = [name for name in names if name not in layouts]
    _STATS["builds" if missing else "hits"] += 1
    if missing:
        built = build(missing)
        _STATS["layouts_built"] += len(built)
        for name, value in built.items():
            layouts.setdefault(name, value)
    return tuple(layouts[name] for name in names)


def _insert(key, entry, owns):
    for stale in [k for k, v in _ENTRIES.items() if owns(v)]:
        del _ENTRIES[stale]
    _ENTRIES[key] = entry
    while len(_ENTRIES) > _capacity():
        _ENTRIES.popitem(last=False)
    return entry


def _object_entry(w1, w2):
    key = ("object", _tensor_key(w1), _tensor_key(w2), _GENERATION, _EPOCH)
    entry = _ENTRIES.get(key)
    if entry is not None and (entry["w1"]() is not w1 or entry["w2"]() is not w2):
        entry = None
    if entry is not None:
        _ENTRIES.move_to_end(key)
        return entry
    return _insert(
        key,
        dict(w1=weakref.ref(w1), w2=weakref.ref(w2), layouts={}),
        lambda v: v.get("w1") is not None and (v["w1"]() is w1 or v["w2"]() is w2),
    )


def _storage_entry(w1, w2, with_pointer):
    s1, s2 = w1.untyped_storage(), w2.untyped_storage()
    key = (
        "storage",
        _storage_key(w1, with_pointer),
        _storage_key(w2, with_pointer),
        _GENERATION,
        _EPOCH,
    )
    entry = _ENTRIES.get(key)
    if entry is not None and (entry["s1"]() is not s1 or entry["s2"]() is not s2):
        entry = None
    if entry is not None:
        _ENTRIES.move_to_end(key)
        return entry
    return _insert(
        key,
        dict(s1=weakref.ref(s1), s2=weakref.ref(s2), layouts={}),
        lambda v: v.get("s1") is not None and (v["s1"]() is s1 or v["s2"]() is s2),
    )
