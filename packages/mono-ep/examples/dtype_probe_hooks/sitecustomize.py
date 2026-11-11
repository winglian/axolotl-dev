# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Prints, once per rank, the dtype and tensor type of the expert weights where the mono-ep runtime receives them and
where the SM100 MXFP8 weight-form builder reads them (DTYPE_PROBE=1)."""

import importlib.abc
import os
import sys

SEEN: set = set()


def _describe(t):
    return f"{type(t).__name__}:{t.dtype}:{tuple(t.shape)}"


def _once(tag, *tensors):
    if tag in SEEN:
        return
    SEEN.add(tag)
    print(
        f"DTYPE_PROBE rank={os.environ.get('RANK')} {tag} "
        + " ".join(_describe(t) for t in tensors),
        flush=True,
    )


def _wrap_runtime(module):
    cls = module.PersistentMoE
    original = cls.__call__

    def call(self, x, indices, scores, w1, w2):
        _once("runtime_call", w1, w2)
        return original(self, x, indices, scores, w1, w2)

    cls.__call__ = call


def _wrap_forms(module):
    original = module.mxfp8_backward_weight_forms

    def forms(w1, w2, *args, **kwargs):
        _once("mxfp8_form_builder", w1, w2)
        return original(w1, w2, *args, **kwargs)

    module.mxfp8_backward_weight_forms = forms


def _wrap_experts_fn(module):
    original = module._local_expert_weights

    def local(*args, **kwargs):
        out = original(*args, **kwargs)
        experts = args[0] if args else None
        if experts is not None and hasattr(experts, "gate_up_proj"):
            _once("experts_param", experts.gate_up_proj, experts.down_proj)
        _once("experts_fn_out", *[t for t in out if hasattr(t, "dtype")][:2])
        return out

    module._local_expert_weights = local


PATCHES = {
    "mono_ep.runtime": _wrap_runtime,
    "mono_ep.weight_cache": _wrap_forms,
    "axolotl.integrations.expert_parallel.experts_fn": _wrap_experts_fn,
}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            import importlib.util

            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        loader = spec.loader
        exec_module = loader.exec_module

        def _exec(module):
            exec_module(module)
            PATCHES[name](module)

        loader.exec_module = _exec
        return spec


if os.environ.get("DTYPE_PROBE") == "1":
    sys.meta_path.insert(0, _Finder())
