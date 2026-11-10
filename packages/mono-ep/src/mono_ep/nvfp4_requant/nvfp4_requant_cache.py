# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Per-step cache of requantized expert weights, keyed on (adapter version, step).

The merged weight changes exactly when the optimizer steps (or an adapter is
loaded or swapped), so the packed forms are rebuilt from an optimizer post-step
hook on a side stream and looked up by the forward with the same key. A lookup
with a different key misses rather than serving a stale build: under FSDP2 a
cache keyed on tensor identity or version counters goes stale because the
sharded parameter is re-gathered into a fresh tensor every step.

The base experts are frozen packed NVFP4 with the checkpoint's fixed per-tensor
scales, so a rebuild needs only the live LoRA factors; no rank exchanges
anything before quantizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, NamedTuple, Protocol

import torch


class RequantKey(NamedTuple):
    adapter_version: int
    step: int


class AdapterVersion:
    """Monotonic counter the trainer bumps whenever adapter weights change by a
    route other than an optimizer step (load, swap, merge-unmerge)."""

    def __init__(self, value: int = 0):
        self.value = int(value)

    def bump(self) -> int:
        self.value += 1
        return self.value


class StreamBackend(Protocol):
    def run(self, fn: Callable[[], Any]) -> tuple[Any, Any]:
        """Run ``fn`` on the side stream after the current stream's work; return
        ``(result, completion_token)``."""

    def wait(self, token: Any) -> None:
        """Make the current stream wait for ``token``."""


class CudaStreams:
    def __init__(self, device: torch.device | int | None = None):
        self.side = torch.cuda.Stream(device)

    def run(self, fn):
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            out = fn()
        event = torch.cuda.Event()
        event.record(self.side)
        return out, event

    def wait(self, event):
        torch.cuda.current_stream().wait_event(event)


class FakeStreams:
    """Records the stream protocol instead of touching CUDA (CPU tests)."""

    def __init__(self):
        self.runs: list[Any] = []
        self.waits: list[Any] = []

    def run(self, fn):
        out = fn()
        token = object()
        self.runs.append(token)
        return out, token

    def wait(self, token):
        self.waits.append(token)


@dataclass
class _Entry:
    key: RequantKey
    result: Any
    token: Any


@dataclass
class RequantCache:
    """One entry per layer; a refresh with the entry's own key is a no-op."""

    streams: StreamBackend | None = None
    builds: int = 0
    hits: int = 0
    misses: int = 0
    _entries: dict[Hashable, _Entry] = field(default_factory=dict)

    def _backend(self) -> StreamBackend:
        if self.streams is None:
            self.streams = CudaStreams() if torch.cuda.is_available() else FakeStreams()
        return self.streams

    def refresh(self, layer: Hashable, key: RequantKey, build: Callable[[], Any]):
        entry = self._entries.get(layer)
        if entry is not None and entry.key == key:
            return entry.result
        result, token = self._backend().run(build)
        self._entries[layer] = _Entry(key, result, token)
        self.builds += 1
        return result

    def get(self, layer: Hashable, key: RequantKey):
        entry = self._entries.get(layer)
        if entry is None or entry.key != key:
            self.misses += 1
            return None
        self._backend().wait(entry.token)
        self.hits += 1
        return entry.result

    def key_of(self, layer: Hashable) -> RequantKey | None:
        entry = self._entries.get(layer)
        return None if entry is None else entry.key

    def invalidate(self, layer: Hashable | None = None) -> None:
        if layer is None:
            self._entries.clear()
        else:
            self._entries.pop(layer, None)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class LayerRequant:
    """What the post-step hook rebuilds for one layer: ``build()`` returns the
    packed forms (a :class:`~nvfp4_requant.RequantResult` or anything else)."""

    layer: Hashable
    build: Callable[[], Any]


@dataclass
class StepState:
    step: int = 0
    # True once an optimizer post-step hook bumps ``step``; without one the
    # forward cannot key a cache on the optimizer step and falls back to the
    # identity of the factors it was handed
    attached: bool = False


# Process-wide optimizer-step epoch and adapter version the in-forward requant
# cache keys on (one trainer, one optimizer per process).
GLOBAL_STEP = StepState()
GLOBAL_ADAPTER_VERSION = AdapterVersion()


def attach_step_counter(
    optimizer: torch.optim.Optimizer, state: StepState | None = None
):
    """Bump ``state.step`` after every ``optimizer.step()`` (no rebuilds: the
    forward rebuilds lazily on a key miss from the live factors it receives,
    which is also what makes it correct under FSDP2 where the parameters the
    hook could reach are the sharded ones). Returns the hook handle."""
    state = state or GLOBAL_STEP
    state.attached = True

    def hook(_opt, _args, _kwargs):
        state.step += 1

    return optimizer.register_step_post_hook(hook)


def current_key(state: StepState | None = None) -> RequantKey:
    state = state or GLOBAL_STEP
    return RequantKey(GLOBAL_ADAPTER_VERSION.value, state.step)


def attach_post_step_hook(
    optimizer: torch.optim.Optimizer,
    cache: RequantCache,
    layers: list[LayerRequant],
    adapter_version: AdapterVersion,
    state: StepState | None = None,
):
    """Rebuild every layer after each ``optimizer.step()`` under
    ``RequantKey(adapter_version.value, step)``. Returns ``(handle, state)``;
    the forward looks entries up with ``RequantKey(adapter_version.value,
    state.step)``."""
    state = state or StepState()

    def hook(_opt, _args, _kwargs):
        state.step += 1
        key = RequantKey(adapter_version.value, state.step)
        for spec in layers:
            cache.refresh(spec.layer, key, spec.build)

    return optimizer.register_step_post_hook(hook), state
