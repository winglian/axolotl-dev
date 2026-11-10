# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Peer-memory token dispatch and combine in the DeepEP row layout.

Autograd op pairs over torch symmetric memory that any local experts kernel can sit
between. Every rank receives a fixed ``world * capacity`` rows: row ``s * capacity + t`` is
token ``t`` of source rank ``s`` when one of its top-k experts is local, otherwise a zero
row with all ``-1`` ids. The layout is a pure function of the published routing, so there
is no host count and no sort; the opt-in compact mode trades one host count per dispatch
for a ``[routed, hidden]`` receive side.

Transfers are the bandwidth-shaped streams of ``peer_stream.py`` (``transfer="streamed"``);
``transfer="legacy"`` keeps the fused runtime's per-expert ``pack_peer_rows`` /
``return_peer_rows`` for comparison.
"""

from itertools import count

import torch

_RUNTIMES: dict[int, "PeerDispatch"] = {}
_RUNTIME_KEYS = count(1)
BARRIER_TIMEOUT_MS = 600_000
TRANSFERS = ("streamed", "legacy")

DISPATCH_OP = "mono_ep::peer_dispatch"
COMBINE_OP = "mono_ep::peer_combine"
DISPATCH_BACKWARD_OP = "mono_ep::peer_dispatch_backward"
COMBINE_BACKWARD_OP = "mono_ep::peer_combine_backward"
DISPATCH_PUBLISH_OP = "mono_ep::peer_dispatch_publish"
COUNT_TO_HOST_OP = "mono_ep::peer_count_to_host"
DISPATCH_PULL_OP = "mono_ep::peer_dispatch_pull"

# Buffer families: ops that write this rank's ``x``/``idx``/``w`` and read the peers'
# (dispatch, combine backward) and ops that write the peers' ``y``/``gw`` and read this
# rank's (combine, dispatch backward).
X_FAMILY, Y_FAMILY = "x", "y"


def local_routing(all_idx, rank, num_local_experts):
    """``(local_ids [W*cap, K], routed [W*cap])`` from every rank's global ids ``[W, cap, K]``."""
    lower = rank * num_local_experts
    mine = (all_idx >= lower) & (all_idx < lower + num_local_experts)
    local = torch.where(mine, all_idx - lower, torch.full_like(all_idx, -1))
    local = local.reshape(-1, all_idx.shape[-1])
    return local, (local >= 0).any(dim=-1)


def producer_mask(topk_idx, num_local_experts, world):
    """``[W, T]`` bool: token ``t`` has at least one expert on rank ``p``."""
    dest = torch.div(topk_idx, num_local_experts, rounding_mode="floor")
    ranks = torch.arange(world, device=topk_idx.device, dtype=topk_idx.dtype)
    hit = (dest.unsqueeze(0) == ranks.view(-1, 1, 1)) & (topk_idx >= 0).unsqueeze(0)
    return hit.any(dim=-1).contiguous()


def gather_rows(sources, assignments):
    """Rows ``assignments`` of ``cat(sources)``; ``-1`` gives a zero row (legacy pull)."""
    if sources[0].is_cuda:
        from .peer_packing import pack_peer_rows

        return pack_peer_rows(list(sources), assignments, 1)
    valid = assignments >= 0
    rows = torch.cat(list(sources))[assignments.clamp_min(0)]
    return rows.masked_fill(~valid.unsqueeze(-1), 0)


def scatter_rows(payload, assignments, destinations, score_sink, scores):
    """Write ``payload[i]`` to ``destinations[a // slots][a % slots]`` for ``a = assignments[i] >= 0``
    (legacy push)."""
    if payload.is_cuda:
        from .peer_packing import return_peer_rows

        return_peer_rows(
            payload,
            scores,
            assignments,
            list(destinations),
            [score_sink] * len(destinations),
        )
        return
    slots = destinations[0].shape[0]
    valid = assignments >= 0
    peer = torch.div(assignments, slots, rounding_mode="floor")
    for rank, destination in enumerate(destinations):
        pick = valid & (peer == rank)
        destination[assignments[pick] % slots] = payload[pick].to(destination.dtype)


class SymmetricWorkspace:
    """One symmetric allocation per named buffer; ``views[name][r]`` is rank ``r``'s copy."""

    def __init__(self, group, device, specs):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.handles = []
        self.views = {}
        for name, (shape, dtype) in specs.items():
            local = symm_mem.empty(*shape, dtype=dtype, device=device)
            handle = symm_mem.rendezvous(local, group=group)
            self.handles.append(handle)
            self.views[name] = [
                handle.get_buffer(r, tuple(shape), dtype) for r in range(self.world)
            ]

    def barrier(self):
        # system-scope release/acquire on the signal pad; orders this stream's earlier
        # kernels (local copies and peer stores) before any peer's later reads
        self.handles[0].barrier(channel=0, timeout_ms=BARRIER_TIMEOUT_MS)

    def destroy(self):
        if self.views and self.views["x"][0].is_cuda:
            from .peer_stream import forget_peer_deltas

            for views in self.views.values():
                forget_peer_deltas(views)
        self.views.clear()
        self.handles.clear()


class LocalWorkspace:
    """Simulation workspace: ``world`` ranks (threads) sharing plain tensors, for CPU tests."""

    def __init__(self, world, rank, shared, barrier=None):
        self.rank, self.world, self.views = rank, world, shared
        self._barrier = barrier

    @staticmethod
    def create(world, device, specs, barrier=None):
        shared = {
            name: [
                torch.zeros(*shape, dtype=dtype, device=device) for _ in range(world)
            ]
            for name, (shape, dtype) in specs.items()
        }
        return [LocalWorkspace(world, rank, shared, barrier) for rank in range(world)]

    def barrier(self):
        if self._barrier is not None:
            self._barrier.wait()

    def destroy(self):
        self.views = {}


class PeerDispatch:
    """Fixed-capacity DeepEP-layout dispatch/combine for one expert-parallel group.

    ``capacity`` is the most tokens one rank dispatches per call. Every op is one write
    phase, one barrier and one read phase: dispatch and combine backward publish into this
    rank's ``x``/``idx``/``w`` and pull from the peers'; combine and dispatch backward push
    into the peers' ``y``/``gw`` slots and reduce this rank's. One op's barrier therefore
    orders the previous op's reads of the *other* family's buffers ahead of this op's
    writes, so a leading barrier is only needed when two ops of the same family run back
    to back (``_enter``); every rank must run the same op sequence.
    """

    def __init__(
        self,
        *,
        capacity,
        top_k,
        hidden,
        workspace=None,
        group=None,
        device=None,
        transfer="streamed",
        compact=False,
    ):
        if min(capacity, top_k, hidden) < 1:
            raise ValueError("capacity, top_k and hidden must be positive")
        if transfer not in TRANSFERS:
            raise ValueError(f"transfer must be one of {TRANSFERS}, got {transfer!r}")
        if workspace is None:
            if group is None:
                raise ValueError("a process group is required without a workspace")
            device = device or torch.device("cuda", torch.cuda.current_device())
            import torch.distributed as dist

            world = dist.get_world_size(group)
            workspace = SymmetricWorkspace(
                group, device, self.buffer_specs(world, capacity, top_k, hidden)
            )
        self.workspace = workspace
        self.rank, self.world = workspace.rank, workspace.world
        self.capacity, self.top_k, self.hidden = capacity, top_k, hidden
        self.transfer, self.compact = transfer, compact
        self.rows = self.world * capacity
        device = workspace.views["x"][0].device
        self.device = device
        self.streamed = transfer == "streamed" and device.type == "cuda"
        rows = torch.arange(self.rows, device=device)
        self.pull_base = rows
        # legacy push target of receive row (s, t): slot (rank, t) in source s's return buffer
        self.push_base = (
            torch.div(rows, capacity, rounding_mode="floor") * self.rows
            + self.rank * capacity
            + rows % capacity
        )
        self.score_sink = torch.zeros(self.rows, device=device, dtype=torch.float32)
        self.zero_scores = torch.zeros(self.rows, device=device, dtype=torch.float32)
        self._last_family = None
        self.extra_barriers = 0
        self._key = next(_RUNTIME_KEYS)
        _RUNTIMES[self._key] = self

    @staticmethod
    def buffer_specs(world, capacity, top_k, hidden):
        rows = world * capacity
        return {
            "x": ((capacity, hidden), torch.bfloat16),
            "idx": ((capacity, top_k), torch.int64),
            "w": ((capacity, top_k), torch.float32),
            "y": ((rows, hidden), torch.bfloat16),
            "gw": ((rows, top_k), torch.float32),
        }

    @property
    def key(self):
        return self._key

    def destroy(self):
        _RUNTIMES.pop(self._key, None)
        self.workspace.destroy()

    def _own(self, name):
        return self.workspace.views[name][self.rank]

    def _check_tokens(self, tokens):
        if tokens > self.capacity:
            raise ValueError(
                f"{tokens} tokens exceed the peer dispatch capacity of {self.capacity} per rank"
            )

    def _enter(self, family):
        """Protect the buffers ``family`` is about to overwrite.

        The previous op's barrier already orders every peer's reads of the other family;
        only a same-family predecessor (e.g. two combine backwards in a row when no
        dispatch backward runs between them) can still have readers in flight.
        """
        if self._last_family == family:
            self.workspace.barrier()
            self.extra_barriers += 1
        self._last_family = family

    # --- phases (also driven directly by the CPU simulation tests) ---

    def publish(self, x=None, topk_idx=None, topk_weights=None):
        if topk_idx is not None:
            tokens = topk_idx.shape[0]
            self._check_tokens(tokens)
            idx = self._own("idx")
            idx[:tokens].copy_(topk_idx)
            idx[tokens:].fill_(-1)
        if x is not None:
            self._own("x")[: x.shape[0]].copy_(x)
        if topk_weights is not None:
            self._own("w")[: topk_weights.shape[0]].copy_(topk_weights)

    def collect_routing(self, num_local_experts):
        views = self.workspace.views
        all_idx = torch.stack(views["idx"])
        local, routed = local_routing(all_idx, self.rank, num_local_experts)
        weights = torch.stack(views["w"]).reshape(self.rows, self.top_k)
        weights = weights.masked_fill(local < 0, 0)
        return local, weights, routed

    def pull(self, routed):
        """Fixed layout: ``[world * capacity, hidden]`` with zero rows where not routed."""
        if self.streamed:
            from .peer_stream import pull_rows

            return pull_rows(self.workspace.views["x"], routed, self.rank)
        assignments = torch.where(
            routed, self.pull_base, torch.full_like(self.pull_base, -1)
        )
        return gather_rows(self.workspace.views["x"], assignments)

    def pull_compact(self, routed, prefix, count):
        """``([count, hidden], slots [count])``: the routed rows in receive-row order."""
        if self.streamed:
            from .peer_stream import pull_rows

            return pull_rows(
                self.workspace.views["x"], routed, self.rank, prefix=prefix, count=count
            )
        slots = routed.nonzero().squeeze(1)
        if slots.numel() != count:
            raise RuntimeError("routed count does not match the published routing")
        return gather_rows(self.workspace.views["x"], slots), slots

    def push(self, payload, rows, name):
        """``rows``: the routed mask (fixed layout, ``payload`` over every receive row) or
        the compact ``slots`` (``payload[i]`` is receive row ``slots[i]``)."""
        destinations = self.workspace.views[name]
        payload = payload.to(destinations[0].dtype).contiguous()
        if self.streamed:
            from .peer_stream import push_rows

            push_rows(payload, destinations, rows, self.rank)
            return
        if rows.dtype == torch.bool:
            assignments = torch.where(
                rows, self.push_base, torch.full_like(self.push_base, -1)
            )
        else:
            assignments = self.push_base[rows]
        scatter_rows(
            payload, assignments, destinations, self.score_sink, self.zero_scores
        )

    def collect(self, name, topk_idx, num_local_experts, dtype):
        """FP32 sum of the producer slots that hold a partial of each token, rounded once
        to ``dtype``."""
        tokens = topk_idx.shape[0]
        mask = producer_mask(topk_idx, num_local_experts, self.world)
        if self.streamed:
            from .peer_stream import reduce_slots

            return reduce_slots(self._own(name), mask, self.world, dtype)
        buffer = self._own(name).view(self.world, self.capacity, -1)[:, :tokens]
        total = torch.zeros(
            (tokens, buffer.shape[-1]), device=buffer.device, dtype=torch.float32
        )
        for rank in range(self.world):
            total += buffer[rank].float().masked_fill(~mask[rank].unsqueeze(-1), 0)
        return total.to(dtype)

    # --- op bodies ---

    def dispatch(self, x, topk_idx, topk_weights, num_local_experts):
        self._enter(X_FAMILY)
        self.publish(x, topk_idx, topk_weights)
        self.workspace.barrier()  # publish: peers read idx/w/x after this
        local, weights, routed = self.collect_routing(num_local_experts)
        return self.pull(routed), local, weights

    def dispatch_publish(self, x, topk_idx, topk_weights, num_local_experts):
        """Compact mode, first half: publish, barrier, routing and the routed count."""
        self._enter(X_FAMILY)
        self.publish(x, topk_idx, topk_weights)
        self.workspace.barrier()
        local, weights, routed = self.collect_routing(num_local_experts)
        prefix = torch.cumsum(routed, 0, dtype=torch.int64)
        return local, weights, routed, prefix, prefix[-1:].clone()

    def dispatch_pull(self, local, weights, routed, prefix, count):
        """Compact mode, second half: no barrier, reads the peers' ``x`` published by
        ``dispatch_publish`` (the next op's barrier orders these reads before any
        re-publish)."""
        recv_x, slots = self.pull_compact(routed, prefix, count)
        return (
            recv_x,
            local.index_select(0, slots),
            weights.index_select(0, slots),
            slots,
        )

    def combine(self, local_out, rows, topk_idx, num_local_experts):
        self._enter(Y_FAMILY)
        self.push(local_out, rows, "y")
        self.workspace.barrier()  # every producer's partial rows have landed in own y
        return self.collect("y", topk_idx, num_local_experts, local_out.dtype)

    def dispatch_backward(
        self, grad_recv_x, grad_recv_w, rows, topk_idx, num_local_experts
    ):
        self._enter(Y_FAMILY)
        self.push(grad_recv_x, rows, "y")
        if grad_recv_w is not None:
            self.push(grad_recv_w, rows, "gw")
        self.workspace.barrier()  # all pushes into own y/gw complete
        grad_x = self.collect("y", topk_idx, num_local_experts, grad_recv_x.dtype)
        grad_w = None
        if grad_recv_w is not None:
            grad_w = self.collect("gw", topk_idx, num_local_experts, grad_recv_w.dtype)
        return grad_x, grad_w

    def combine_backward(self, grad_y, rows):
        self._check_tokens(grad_y.shape[0])
        self._enter(X_FAMILY)
        self.publish(x=grad_y.to(torch.bfloat16))
        self.workspace.barrier()  # publish
        if rows.dtype == torch.bool:
            return self.pull(rows).to(grad_y.dtype)
        slots = rows
        if self.streamed:
            from .peer_stream import pull_rows

            routed = torch.zeros(self.rows, dtype=torch.bool, device=self.device)
            routed[slots] = True
            prefix = torch.cumsum(routed, 0, dtype=torch.int64)
            recv, _ = pull_rows(
                self.workspace.views["x"],
                routed,
                self.rank,
                prefix=prefix,
                count=slots.shape[0],
            )
            return recv.to(grad_y.dtype)
        return gather_rows(self.workspace.views["x"], slots).to(grad_y.dtype)


def _runtime(key):
    runtime = _RUNTIMES.get(key)
    if runtime is None:
        raise RuntimeError("mono_ep peer dispatch runtime used after destroy()")
    return runtime


def _check_dispatch_inputs(x, topk_idx, topk_weights, runtime):
    if (
        x.ndim != 2
        or x.shape[1] != runtime.hidden
        or topk_idx.shape != (x.shape[0], runtime.top_k)
        or topk_weights.shape != topk_idx.shape
    ):
        raise ValueError(
            "expected x [T, hidden], topk_idx and topk_weights [T, top_k] matching the runtime"
        )
    if (
        x.dtype != torch.bfloat16
        or topk_idx.dtype != torch.int64
        or (topk_weights.dtype != torch.float32)
    ):
        raise ValueError("expected BF16 x, int64 topk_idx and FP32 topk_weights")
    runtime._check_tokens(x.shape[0])


# --- fixed layout ---


@torch.library.custom_op(DISPATCH_OP, mutates_args=())
def peer_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
    runtime_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    runtime = _runtime(runtime_key)
    _check_dispatch_inputs(x, topk_idx, topk_weights, runtime)
    return runtime.dispatch(x, topk_idx, topk_weights, num_local_experts)


@peer_dispatch.register_fake
def _peer_dispatch_fake(x, topk_idx, topk_weights, num_local_experts, runtime_key):
    runtime = _runtime(runtime_key)
    _check_dispatch_inputs(x, topk_idx, topk_weights, runtime)
    rows = runtime.rows
    return (
        x.new_empty((rows, x.shape[1])),
        topk_idx.new_empty((rows, topk_idx.shape[1])),
        topk_weights.new_empty((rows, topk_weights.shape[1])),
    )


@torch.library.custom_op(DISPATCH_BACKWARD_OP, mutates_args=())
def peer_dispatch_backward(
    grad_recv_x: torch.Tensor,
    grad_recv_w: torch.Tensor | None,
    topk_idx: torch.Tensor,
    rows: torch.Tensor,
    num_local_experts: int,
    runtime_key: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rows``: routed mask (fixed layout) or compact slots (int64)."""
    grad_x, grad_w = _runtime(runtime_key).dispatch_backward(
        grad_recv_x, grad_recv_w, rows, topk_idx, num_local_experts
    )
    # custom ops return tensors only; an absent weight gradient is an empty placeholder
    return grad_x, grad_x.new_empty((0,)) if grad_w is None else grad_w


@peer_dispatch_backward.register_fake
def _peer_dispatch_backward_fake(
    grad_recv_x, grad_recv_w, topk_idx, rows, num_local_experts, runtime_key
):
    tokens = topk_idx.shape[0]
    grad_w = (
        grad_recv_x.new_empty((0,))
        if grad_recv_w is None
        else grad_recv_w.new_empty(topk_idx.shape)
    )
    return grad_recv_x.new_empty((tokens, grad_recv_x.shape[1])), grad_w


def _dispatch_setup_context(ctx, inputs, output):
    _x, topk_idx, _w, num_local_experts, runtime_key = inputs
    ctx.save_for_backward(topk_idx, (output[1] >= 0).any(dim=-1))
    ctx.num_local_experts = num_local_experts
    ctx.runtime_key = runtime_key


def _dispatch_autograd_backward(ctx, grad_recv_x, _grad_recv_idx, grad_recv_w):
    topk_idx, rows = ctx.saved_tensors
    grad_x, grad_w = peer_dispatch_backward(
        grad_recv_x.contiguous(),
        None if grad_recv_w is None else grad_recv_w.contiguous(),
        topk_idx,
        rows,
        ctx.num_local_experts,
        ctx.runtime_key,
    )
    return grad_x, None, None if grad_recv_w is None else grad_w, None, None


peer_dispatch.register_autograd(
    _dispatch_autograd_backward, setup_context=_dispatch_setup_context
)


@torch.library.custom_op(COMBINE_OP, mutates_args=())
def peer_combine(
    local_out: torch.Tensor,
    rows: torch.Tensor,
    topk_idx: torch.Tensor,
    num_local_experts: int,
    runtime_key: int,
) -> torch.Tensor:
    """``rows``: routed mask over the runtime's receive rows (fixed layout) or the
    compact slots of ``local_out``'s rows (int64)."""
    runtime = _runtime(runtime_key)
    if rows.dtype == torch.bool:
        expected = runtime.rows
    else:
        expected = rows.shape[0]
    if local_out.shape != (expected, runtime.hidden) or rows.ndim != 1:
        raise ValueError("local_out and rows must cover the runtime's receive rows")
    return runtime.combine(local_out, rows, topk_idx, num_local_experts)


@peer_combine.register_fake
def _peer_combine_fake(local_out, rows, topk_idx, num_local_experts, runtime_key):
    return local_out.new_empty((topk_idx.shape[0], local_out.shape[1]))


@torch.library.custom_op(COMBINE_BACKWARD_OP, mutates_args=())
def peer_combine_backward(
    grad_y: torch.Tensor, rows: torch.Tensor, runtime_key: int
) -> torch.Tensor:
    return _runtime(runtime_key).combine_backward(grad_y, rows)


@peer_combine_backward.register_fake
def _peer_combine_backward_fake(grad_y, rows, runtime_key):
    rows_out = _runtime(runtime_key).rows if rows.dtype == torch.bool else rows.shape[0]
    return grad_y.new_empty((rows_out, grad_y.shape[1]))


def _combine_setup_context(ctx, inputs, output):
    _local_out, rows, _topk_idx, _num_local_experts, runtime_key = inputs
    ctx.save_for_backward(rows)
    ctx.runtime_key = runtime_key


def _combine_autograd_backward(ctx, grad_y):
    (rows,) = ctx.saved_tensors
    return (
        peer_combine_backward(grad_y.contiguous(), rows, ctx.runtime_key),
        None,
        None,
        None,
        None,
    )


peer_combine.register_autograd(
    _combine_autograd_backward, setup_context=_combine_setup_context
)


# --- compact receive rows ---


@torch.library.custom_op(DISPATCH_PUBLISH_OP, mutates_args=())
def peer_dispatch_publish(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
    runtime_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Publish and barrier; returns ``(local_ids, weights, routed, prefix, count)`` over the
    fixed receive rows. Not differentiable: ``peer_dispatch_pull`` carries the gradients."""
    runtime = _runtime(runtime_key)
    _check_dispatch_inputs(x, topk_idx, topk_weights, runtime)
    return runtime.dispatch_publish(x, topk_idx, topk_weights, num_local_experts)


@peer_dispatch_publish.register_fake
def _peer_dispatch_publish_fake(
    x, topk_idx, topk_weights, num_local_experts, runtime_key
):
    runtime = _runtime(runtime_key)
    _check_dispatch_inputs(x, topk_idx, topk_weights, runtime)
    rows = runtime.rows
    return (
        topk_idx.new_empty((rows, topk_idx.shape[1])),
        topk_weights.new_empty((rows, topk_weights.shape[1])),
        topk_idx.new_empty((rows,), dtype=torch.bool),
        topk_idx.new_empty((rows,)),
        topk_idx.new_empty((1,)),
    )


@torch.library.custom_op(COUNT_TO_HOST_OP, mutates_args=())
def peer_count_to_host(count: torch.Tensor) -> torch.Tensor:
    """The one device-to-host copy of compact mode (a stream sync per dispatch); its own
    op so a selective-checkpointing policy can save it instead of re-syncing."""
    return count.to("cpu", copy=True)


@peer_count_to_host.register_fake
def _peer_count_to_host_fake(count):
    return count.new_empty(count.shape, device="cpu")


@torch.library.custom_op(DISPATCH_PULL_OP, mutates_args=())
def peer_dispatch_pull(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    routed: torch.Tensor,
    prefix: torch.Tensor,
    count: int,
    runtime_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(recv_x [count, H], recv_idx, recv_w [count, K], slots [count])``; ``x`` and
    ``topk_weights`` are the values ``peer_dispatch_publish`` already published and only
    connect the autograd graph."""
    runtime = _runtime(runtime_key)
    if count < 0 or count > runtime.rows:
        raise ValueError("compact count outside the receive layout")
    return runtime.dispatch_pull(local_ids, weights, routed, prefix, count)


@peer_dispatch_pull.register_fake
def _peer_dispatch_pull_fake(
    x, topk_weights, local_ids, weights, routed, prefix, count, runtime_key
):
    return (
        x.new_empty((count, x.shape[1])),
        local_ids.new_empty((count, local_ids.shape[1])),
        weights.new_empty((count, weights.shape[1])),
        local_ids.new_empty((count,)),
    )


def dispatch_compact(x, topk_idx, topk_weights, num_local_experts, runtime_key):
    """Compact-mode dispatch: publish, one host count, pull. Returns the four pull outputs
    plus the count as a Python int."""
    local_ids, weights, routed, prefix, count = peer_dispatch_publish(
        x.detach(), topk_idx, topk_weights.detach(), num_local_experts, runtime_key
    )
    routed_count = int(peer_count_to_host(count).item())
    return _CompactPull.apply(
        x,
        topk_weights,
        local_ids,
        weights,
        routed,
        prefix,
        routed_count,
        topk_idx,
        num_local_experts,
        runtime_key,
    )


class _CompactPull(torch.autograd.Function):
    """Autograd around ``peer_dispatch_pull``: the backward is ``peer_dispatch_backward``
    with the compact slots, exactly the fixed layout's."""

    @staticmethod
    def forward(
        ctx,
        x,
        topk_weights,
        local_ids,
        weights,
        routed,
        prefix,
        count,
        topk_idx,
        num_local_experts,
        runtime_key,
    ):
        recv_x, recv_idx, recv_w, slots = peer_dispatch_pull(
            x, topk_weights, local_ids, weights, routed, prefix, count, runtime_key
        )
        ctx.save_for_backward(topk_idx, slots)
        ctx.num_local_experts = num_local_experts
        ctx.runtime_key = runtime_key
        ctx.mark_non_differentiable(recv_idx, slots)
        return recv_x, recv_idx, recv_w, slots

    @staticmethod
    def backward(ctx, grad_recv_x, _grad_recv_idx, grad_recv_w, _grad_slots):
        topk_idx, slots = ctx.saved_tensors
        needs_w = ctx.needs_input_grad[1]
        grad_x, grad_w = peer_dispatch_backward(
            grad_recv_x.contiguous(),
            grad_recv_w.contiguous() if needs_w and grad_recv_w is not None else None,
            topk_idx,
            slots,
            ctx.num_local_experts,
            ctx.runtime_key,
        )
        return (
            grad_x,
            grad_w if needs_w and grad_recv_w is not None else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
