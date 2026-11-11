# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Stored forward context for the SM100 persistent backward (option bit 4096).

A storing forward writes z (BF16 gate/up pre-activations) by its own pool row into one
runtime-owned buffer (``layout::StoredZ``: an int32 header, a source -> pool-row index,
then ``[capacity, 2I]`` BF16 rows), and the next backward whose inputs match skips its
gate/up recompute GEMMs and reads z instead. The buffer serves one layer at a time, so
it pays when the storing forward runs right before that layer's backward: the
activation-checkpoint recompute.

``MONO_EP_STORED_CONTEXT``:

* ``auto`` (default): store in checkpoint-recompute forwards when the shape rule says the
  backward is GEMM-bound (``hidden * intermediate >= MIN_HIDDEN_X_INTERMEDIATE``) and the
  buffer fits in free device memory with headroom; otherwise recompute, with one log line
  saying why.
* ``recompute``: store in checkpoint-recompute forwards, no shape or memory check.
* ``always``: store in every forward (only the last forward before a backward is used).
* ``off`` (or ``0``): never store.

Capacity is in forward pool rows: an initial bound of 1.25 x tokens x top-k plus one
block per local expert, raised from the routed-row high-water mark each backward reports
(read one step late, without a host sync). A forward whose rows exceed the capacity sets
the header flag and that layer's backward recomputes z, so a short buffer costs time,
never correctness.
"""

import logging
import math
import os

import torch

LOG = logging.getLogger("mono_ep")

HEADER_INTS = 4
# Per-row gate/up recompute work (hidden x intermediate) above which skipping it pays on the
# SM100 backward. 8xB200 backward with stored z vs recompute: 2048 x 768 and 2048 x 1536 slower,
# 3072 x 1024 and 4096 x 768 neutral, 4096 x 1536, 5120 x 2560, 6144 x 2048, 7168 x 2048 faster.
MIN_HIDDEN_X_INTERMEDIATE = 4096 * 1536
HEADROOM_BYTES = 2 << 30
HEADROOM_FRACTION = 0.05


def policy():
    value = os.getenv("MONO_EP_STORED_CONTEXT", "auto")
    if value == "0":
        value = "off"
    if value not in ("auto", "recompute", "always", "off"):
        raise ValueError(
            "MONO_EP_STORED_CONTEXT must be auto, recompute, always or off"
        )
    return value


def meta_bytes(ranks, max_tokens, topk):
    return (HEADER_INTS * 4 + ranks * max_tokens * topk * 4 + 255) // 256 * 256


def shape_rule(hidden, intermediate):
    return hidden * intermediate >= MIN_HIDDEN_X_INTERMEDIATE


class StoredContext:
    """The z buffer, its capacity and the auto decision for one runtime."""

    def __init__(self, runtime, ranks):
        self.runtime = runtime
        self.ranks = ranks
        self.max_tokens = runtime.buffer.num_max_tokens_per_rank
        self.local_experts = runtime.experts // ranks
        self.row_bytes = 2 * runtime.intermediate * 2
        self.meta = meta_bytes(ranks, self.max_tokens, runtime.topk)
        override = os.getenv("MONO_EP_STORED_CONTEXT_ROWS")
        self.capacity = (
            int(override)
            if override
            else _round_up(math.ceil(1.25 * runtime.max_tokens * runtime.topk), 128)
            + self.local_experts * 128
        )
        self.fixed_capacity = override is not None
        self.buffer = None
        self.header = None
        self.key = None
        self.auto_ok = None
        self.high_water = 0
        self._pending = None
        self.hits = self.misses = self.grows = 0

    def nbytes(self, capacity=None):
        return self.meta + (capacity or self.capacity) * self.row_bytes

    def _fits(self, nbytes):
        free, total = torch.cuda.mem_get_info()
        free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        need = nbytes + max(HEADROOM_BYTES, int(HEADROOM_FRACTION * total))
        return free >= need, free

    def decide_auto(self):
        if self.auto_ok is not None:
            return self.auto_ok
        runtime = self.runtime
        if not shape_rule(runtime.hidden, runtime.intermediate):
            self.auto_ok = False
            reason = (
                f"hidden x intermediate {runtime.hidden} x {runtime.intermediate} below "
                f"4096 x 1536 (the backward is not GEMM-bound enough)"
            )
        else:
            ok, free = self._fits(self.nbytes())
            self.auto_ok = ok
            reason = (
                f"buffer {self.nbytes() / 2**30:.2f} GiB, free {free / 2**30:.1f} GiB"
            )
        LOG.info(
            "mono_ep stored context (auto): %s, %s",
            "storing z in checkpoint recomputes" if self.auto_ok else "recomputing z",
            reason,
        )
        return self.auto_ok

    def _poll_high_water(self, enforce_memory):
        pending = self._pending
        if pending is not None and pending[1].query():
            self.high_water = max(self.high_water, int(pending[0].item()))
            self._pending = None
        if self.fixed_capacity or self.high_water <= self.capacity:
            return
        capacity = _round_up(math.ceil(self.high_water * 1.1), 128)
        if enforce_memory:
            ok, free = self._fits(self.nbytes(capacity) - self.nbytes())
            if not ok:
                LOG.info(
                    "mono_ep stored context: %d routed rows exceed the %d-row buffer and growing "
                    "to %.2f GiB does not fit (free %.1f GiB); those layers recompute z",
                    self.high_water,
                    self.capacity,
                    self.nbytes(capacity) / 2**30,
                    free / 2**30,
                )
                self.fixed_capacity = True
                return
        self.capacity = capacity
        self.buffer = None
        self.grows += 1

    def prepare(self, device, enforce_memory):
        """The buffer for a storing forward, header reset (flag 0, capacity)."""
        self._poll_high_water(enforce_memory)
        if self.buffer is None:
            self.buffer = torch.empty(self.nbytes(), dtype=torch.uint8, device=device)
            self.header = torch.tensor(
                [0, self.capacity, 0, 0], dtype=torch.int32, device=device
            )
        self.buffer[: HEADER_INTS * 4].view(torch.int32).copy_(self.header)
        return self.buffer

    def record_rows(self, counts):
        """After a backward: the pool rows its routing needs, copied to the host without a sync."""
        if self._pending is not None:
            return
        rows = ((counts + 127) // 128).sum() * 128
        host = torch.empty((), dtype=rows.dtype, pin_memory=True)
        host.copy_(rows, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._pending = (host, event)


def context_key(x, indices, scores, w1, w2):
    return tuple(
        (t.data_ptr(), t.shape, t._version) for t in (x, indices, scores, w1, w2)
    )


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple
