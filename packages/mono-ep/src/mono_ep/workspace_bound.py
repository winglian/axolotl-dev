# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Bounded SM100 backward pools (``MONO_EP_WORKSPACE_BOUND``).

The persistent backward's x / dy / h / dz pools (and the MXFP8 E4M3 pools) need one row per
routed row this rank receives, padded to 128 per local expert: ``sum_e ceil(count_e / 128) * 128``.
Sizing them for every token of every rank (``ranks x max tokens x top-k``) is what made the
workspace 19.5 / 38.1 / 44.0 GB per rank at the Qwen3-30B-A3B / 235B-A22B / Beam EP8 shapes.

``auto`` (default): the forward reports its per-expert receive counts (an int32 tensor the
kernel already fills), the padded row total is copied to pinned memory without a sync, and the
backward of that layer sizes the pools to it before launching; the backward sees the same
routing, so the count is exact and the kernel never overflows (it traps past ``pool_rows`` as a
host-bug guard). Records are keyed by the layer's expert weight storage (``w1``'s StorageImpl),
which is per layer, never recomputed and kept by FSDP2 across regathers; activations are not
usable keys because a checkpoint recompute rebuilds the MoE input and the routing as new
tensors. Each key holds a stack: a forward pushes, a checkpoint-recompute forward does not (the
original forward's record serves, so the backward never waits on a just-launched forward), the
backward pops. Repeated calls of one layer and gradient-accumulation micro-batches nest
correctly. Only a forward that can reach a backward pushes (grad mode on and an input or weight
requiring grad), so eval, ``no_grad`` / ``inference_mode`` and generation forwards leave no
record; a recompute forward pushes only when its layer has no record (reentrant checkpointing
runs the original forward under ``no_grad``). Records of forwards whose backward never ran (a
skipped or failed micro-batch) are dropped after every ``optimizer.step()``, and a stack deeper
than ``MAX_DEPTH`` drops its oldest entry with a warning. Pools start at 1.25 x tokens x top-k + 128 per local expert and only grow (to 1.1 x a
larger need, logged). A backward with no record is sized for the worst case and logged every
time (it means a forward was not recorded, e.g. a custom autograd path).

The SM100 BF16 forward's symmetric ring is sized the same way: for ``HEADROOM x tokens x top-k`` routed
rows instead of every token of every rank (wide experts: the live-block bound for that budget; narrow
experts, where slot reuse costs forward time: one slot per budgeted row; never above the worst case). Its scheduler gates L1 claims on the ring (an L1 block is
claimed only after every L2 task one ring generation back is claimed), so a step whose routing needs
more live blocks runs gated and slower instead of overflowing; ``forward_over_budget`` counts the
recorded forwards whose padded rows exceeded the budget. The owned backward's ring is its one-wave
floor.

``max``: the worst-case pools and rings, as before.
"""

import logging
import math
import os
import weakref
from collections import OrderedDict

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

LOG = logging.getLogger("mono_ep")
HEADROOM = 1.25
GROWTH = 1.1
MAX_RECORDS = 4096


def mode():
    value = os.getenv("MONO_EP_WORKSPACE_BOUND", "auto")
    if value not in ("auto", "max"):
        raise ValueError("MONO_EP_WORKSPACE_BOUND must be auto or max")
    return value


def _round_up(value, multiple=128):
    return (value + multiple - 1) // multiple * multiple


MAX_DEPTH = 64
_LIVE: "weakref.WeakSet[WorkspaceBound]" = weakref.WeakSet()


def forward_ring_budget(max_tokens, topk):
    override = os.getenv("MONO_EP_FORWARD_RING_ROWS")
    if override:
        return int(override)
    return _round_up(math.ceil(HEADROOM * max_tokens * topk))


# Narrow experts (Qwen3-30B-A3B class) are dispatch-bound: reusing forward ring slots inside one launch costs
# 4-8 % of the forward there, so their ring holds every budgeted row; wide experts hide the reuse waits
WIDE_EXPERT_HIDDEN_X_INTERMEDIATE = 4096 * 1536


def forward_ring_tokens(hidden, intermediate, budget):
    """Exact forward ring tokens (0 = the live-block bound for ``budget`` routed rows), capped at the worst-case
    ring. ``MONO_EP_FORWARD_RING_TOKENS`` overrides."""
    override = os.getenv("MONO_EP_FORWARD_RING_TOKENS")
    if override is not None:
        return int(override)
    if hidden * intermediate >= WIDE_EXPERT_HIDDEN_X_INTERMEDIATE:
        return 0
    return budget


def layer_key(w1):
    return w1.untyped_storage()._cdata


def will_backward(*tensors):
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def clear_all():
    for bound in list(_LIVE):
        bound.clear()


register_optimizer_step_post_hook(lambda *_: clear_all())


class WorkspaceBound:
    def __init__(self, max_tokens, topk, local_experts, worst_rows):
        self.worst_rows = worst_rows
        self.rows = min(
            worst_rows,
            _round_up(math.ceil(HEADROOM * max_tokens * topk)) + local_experts * 128,
        )
        self.records = OrderedDict()
        self.grows = 0
        self.unrecorded = 0
        self.found = 0
        self.high_water = 0
        self.dropped = 0
        self.forward_budget = 0
        self.forward_over_budget = 0
        _LIVE.add(self)

    def clear(self):
        self.dropped += sum(len(stack) for stack in self.records.values())
        self.records.clear()

    def record(self, w1, counts):
        """After a forward that can reach a backward: the padded row total, pushed under the
        layer's key (a recompute forward only when the layer has no record)."""
        if torch._C._current_graph_task_id() != -1 and self.records.get(layer_key(w1)):
            return
        rows = ((counts + 127) // 128).sum() * 128
        host = torch.empty((), dtype=rows.dtype, pin_memory=True)
        host.copy_(rows, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        stack = self.records.setdefault(layer_key(w1), [])
        stack.append((host, event))
        if len(stack) > MAX_DEPTH:
            del stack[0]
            self.dropped += 1
            LOG.warning(
                "mono_ep bounded workspace: more than %d forwards of one layer without a "
                "backward; dropping the oldest record",
                MAX_DEPTH,
            )
        self.records.move_to_end(layer_key(w1))
        while len(self.records) > MAX_RECORDS:
            self.records.popitem(last=False)

    def required_rows(self, w1):
        """The exact pool rows this layer's backward needs (its latest forward), or the worst case."""
        stack = self.records.get(layer_key(w1))
        if not stack:
            self.unrecorded += 1
            LOG.warning(
                "mono_ep bounded workspace: a backward without a recorded forward; sizing its "
                "pools for the worst case (%d rows, %d such backwards so far)",
                self.worst_rows,
                self.unrecorded,
            )
            return self.worst_rows
        record = stack.pop()
        self.found += 1
        host, event = record
        event.synchronize()
        rows = int(host.item())
        self.high_water = max(self.high_water, rows)
        if self.forward_budget and rows > self.forward_budget:
            self.forward_over_budget += 1
            if self.forward_over_budget == 1:
                LOG.info(
                    "mono_ep bounded workspace: %d routed pool rows exceed the forward ring budget of %d; "
                    "the forward ran ring-gated (slower, still exact)",
                    rows,
                    self.forward_budget,
                )
        return rows

    def target(self, needed):
        """Rows to allocate so that ``needed`` fits, or ``None`` if the current pools do."""
        if needed <= self.rows:
            return None
        rows = min(self.worst_rows, max(needed, _round_up(math.ceil(GROWTH * needed))))
        self.grows += 1
        LOG.info(
            "mono_ep bounded workspace: %d routed pool rows exceed %d; growing the pools to %d",
            needed,
            self.rows,
            rows,
        )
        self.rows = rows
        return rows
