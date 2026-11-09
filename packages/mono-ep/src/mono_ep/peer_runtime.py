# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Internal fixed-capacity autograd schedule for single-node SM100 peer kernels."""

import socket
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.autograd.function import once_differentiable

from .activation import swiglu_fp8
from .chunk import ready_grouped_fp8_out
from .combine import (
    combine_wait_copy,
    combine_wait_reduce,
    grouped_fp8_scatter,
)
from .packing import DeviceExpertLayout
from .peer import (
    peer_gather_bf16_publish,
    peer_gather_quantize_publish,
    peer_packed_bf16_publish,
    peer_packed_quantize_publish,
)
from .quantization import quantize_fp8
from .scheduled import fp8_packed_backward_scatter
from .synchronization import peer_epoch_barrier


def _packed_forward(
    x,
    sx,
    w1,
    s1,
    w2,
    s2,
    ready,
    scores,
    z,
    ids,
    tokens,
    accumulation,
    complete,
    producer,
    assign=False,
    incremental=False,
):
    ready_grouped_fp8_out(x, sx, w1, s1, z, ready, ids, incremental)
    row_end = (ready[:, 1] + ready[:, 2]).max()
    qh, sh = swiglu_fp8(z, scores, row_end)
    grouped_fp8_scatter(
        qh,
        sh,
        w2,
        s2,
        ids,
        tokens,
        accumulation,
        complete,
        producer,
        assign,
        row_end,
    )


def _packed_backward(
    dy,
    sy,
    x,
    sx,
    w1,
    s1,
    w2,
    s2,
    ready,
    scores,
    z,
    ids,
    ends,
    tokens,
    slots,
    dx_acc,
    dx_flags,
    ds_acc,
    ds_flags,
    producer,
    assign=False,
    incremental=False,
):
    return fp8_packed_backward_scatter(
        dy,
        sy,
        x,
        sx,
        w1,
        s1,
        w2,
        s2,
        ready,
        scores,
        z,
        ids,
        ends,
        tokens,
        slots,
        dx_acc,
        ds_acc,
        dx_flags,
        ds_flags,
        producer,
        assign,
        incremental,
    )


class _PeerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, scores, w1, w2, runtime):
        output, saved = runtime.forward(x, indices, scores, w1, w2)
        ctx.runtime = runtime
        ctx.save_for_backward(w1, w2, indices, *saved)
        return output.to(x.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        w1, w2, indices, *saved = ctx.saved_tensors
        dx, ds, dw1, dw2 = ctx.runtime.backward(dy.contiguous(), indices, w1, w2, saved)
        return dx.bfloat16(), None, ds, dw1.to(w1.dtype), dw2.to(w2.dtype), None


class PeerSchedule:
    """BF16/FP8 experts with fixed token/top-k capacity, same-node SM100 only.

    Routing indices are global expert IDs (rank * experts + local expert),
    or -1 for masked assignments; each rank owns the same number of experts.
    All ranks must construct and call in the same order. This internal schedule
    retains invocation-owned activations for backward and permits multiple
    outstanding forwards. It currently synchronizes at invocation boundaries;
    only local callbacks compile. It is not the default MonoEP scheduler.
    compute_lanes shares compute streams across sources and divides the SM
    budget between lanes; omission retains one lane per source.
    """

    def __init__(
        self,
        *,
        group,
        tokens,
        topk,
        hidden,
        inner,
        experts=1,
        compile_local=False,
        slot_combine=False,
        gpu_barrier=False,
        skip_padding=False,
        dtype="fp8",
        compute_lanes=None,
        incremental_ready=False,
    ):
        import deep_gemm

        if dtype not in ("bf16", "fp8"):
            raise ValueError("PeerSchedule dtype must be bf16 or fp8")
        self.incremental_ready = incremental_ready
        self.dtype = dtype
        self.group, self.rank, self.world = (
            group,
            dist.get_rank(group),
            dist.get_world_size(group),
        )
        if compute_lanes is None:
            compute_lanes = self.world
        if (
            not isinstance(compute_lanes, int)
            or isinstance(compute_lanes, bool)
            or not 1 <= compute_lanes <= self.world
        ):
            raise ValueError("compute_lanes must be between one and the group size")
        self.compute_lanes = compute_lanes
        self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device)[0] != 10:
            raise ValueError("PeerSchedule requires SM100")
        if dist.get_backend(group) != "nccl":
            raise ValueError("PeerSchedule requires NCCL initialization")
        if (
            min(tokens, topk, hidden, inner, experts) <= 0
            or hidden % 128
            or inner % 128
        ):
            raise ValueError(
                "Expected positive capacity and 128-aligned expert dimensions"
            )
        hosts = [None] * self.world
        dist.all_gather_object(hosts, socket.gethostname(), group=group)
        if len(set(hosts)) != 1:
            raise ValueError("PeerSchedule only supports a single host")
        self.tokens, self.topk, self.hidden, self.inner = tokens, topk, hidden, inner
        self.experts = experts
        self.slot_combine = slot_combine
        self.gpu_barrier = gpu_barrier
        self.skip_padding = skip_padding
        self.barrier_epoch = 0
        self.reuse_event = None
        combine_rows = tokens * topk if slot_combine else tokens
        combine_dtype = torch.bfloat16 if slot_combine else torch.float32
        self.rows = ((tokens * topk + experts * 127 + 127) // 128) * 128
        self.sms = (
            torch.cuda.get_device_properties(self.device).multi_processor_count - 16
        ) // self.compute_lanes
        self.sms -= self.sms % 2
        if self.sms < 2:
            raise ValueError("Too many peer consumers to reserve producer SM capacity")
        self.deep_gemm = deep_gemm
        self.handles = []
        scale_shape = (
            (self.world, 1, 1)
            if dtype == "bf16"
            else (self.world, hidden // 128, self.rows)
        )
        self.fwd = self._buffers(
            [
                (
                    (self.world, self.rows, hidden),
                    torch.bfloat16 if dtype == "bf16" else torch.uint8,
                ),
                (scale_shape, torch.int32),
                ((self.world, self.rows), torch.float32),
                ((self.world, experts, 4), torch.int32),
                ((combine_rows, hidden), combine_dtype),
                ((self.world, 4), torch.int32),
            ]
        )
        self.bwd = self._buffers(
            [
                (
                    (self.world, self.rows, hidden),
                    torch.bfloat16 if dtype == "bf16" else torch.uint8,
                ),
                (scale_shape, torch.int32),
                ((self.world, experts, 4), torch.int32),
                ((combine_rows, hidden), combine_dtype),
                ((tokens * topk, 1), torch.float32),
                ((self.world, 4), torch.int32),
                ((self.world, 4), torch.int32),
            ]
        )
        self.metadata = self._buffers(
            [
                ((self.world, self.rows), torch.int32),
                ((self.world, self.rows), torch.int32),
                ((self.world, self.rows), torch.int32),
                ((self.world, experts), torch.int32),
            ]
        )
        self.sender = torch.cuda.Stream(device=self.device)
        compute_streams = [
            torch.cuda.Stream(device=self.device) for _ in range(self.compute_lanes)
        ]
        self.computers = [
            compute_streams[i % self.compute_lanes] for i in range(self.world)
        ]
        self.consumers = [torch.cuda.Stream(device=self.device) for _ in range(2)]

        def compile_fn(fn):
            return (
                torch.compile(fn, fullgraph=True, dynamic=False)
                if compile_local
                else fn
            )

        # Distinct Python call sites avoid a shared custom-op wrapper cache limit.
        self.dispatch = compile_fn(lambda *args: peer_gather_quantize_publish(*args))
        self.dispatch_packed = compile_fn(
            lambda *args: peer_packed_quantize_publish(*args)
        )
        self.finish = compile_fn(lambda *args: combine_wait_copy(*args))
        self.finish_slots = compile_fn(lambda *args: combine_wait_reduce(*args))
        self.compute_forward = compile_fn(
            partial(_packed_forward, incremental=incremental_ready)
        )
        self.compute_backward = compile_fn(
            partial(_packed_backward, incremental=incremental_ready)
        )
        if dtype == "bf16":
            from .bf16 import _peer_backward, _peer_forward

            self.compute_forward = compile_fn(
                partial(_peer_forward, incremental=incremental_ready)
            )
            self.compute_backward = compile_fn(
                partial(_peer_backward, incremental=incremental_ready)
            )
            self.dispatch = compile_fn(
                lambda dy, indices, destination, _scales, queue, _task, _gran, tasks, skip: (
                    peer_gather_bf16_publish(
                        dy, indices, destination, queue, tasks, skip
                    )
                )
            )
            self.dispatch_packed = compile_fn(
                lambda x, slots, ids, starts, sizes, ends, scores, destination, _scales, queue, out_ids, out_tokens, out_slots, out_ends, out_scores, skip: (
                    peer_packed_bf16_publish(
                        x,
                        slots,
                        ids,
                        starts,
                        sizes,
                        ends,
                        scores,
                        destination,
                        queue,
                        out_ids,
                        out_tokens,
                        out_slots,
                        out_ends,
                        out_scores,
                        skip,
                    )
                )
            )
        self.warm_forward = self.warm_backward = False
        self.weight_dtypes = None
        if gpu_barrier:
            self.epoch_flags = [
                parts[0] for parts in self._buffers([((self.world,), torch.int32)])
            ]
            self.epoch_flags[self.rank].zero_()
            self.epoch_pointers = torch.tensor(
                [t.data_ptr() for t in self.epoch_flags],
                device=self.device,
                dtype=torch.int64,
            )
            dist.barrier(group=self.group, async_op=True).wait()

    def _values(self, tensor):
        return tensor if self.dtype == "bf16" else tensor.view(torch.float8_e4m3fn)

    def _finish_tokens(self, source, destination, queue):
        if self.slot_combine:
            self.finish_slots(source, destination, queue, self.topk)
        else:
            self.finish(source, destination, queue)

    def _buffers(self, specs):
        tensors = [
            symm_mem.empty(*shape, device=self.device, dtype=dtype)
            for shape, dtype in specs
        ]
        handles = [symm_mem.rendezvous(t, group=self.group) for t in tensors]
        self.handles.extend(handles)
        return [
            tensors
            if rank == self.rank
            else [
                h.get_buffer(rank, tuple(t.shape), t.dtype)
                for h, t in zip(handles, tensors, strict=True)
            ]
            for rank in range(self.world)
        ]

    def _barrier(self):
        current = torch.cuda.current_stream(self.device)
        for stream in (self.sender, *self.computers, *self.consumers):
            current.wait_stream(stream)
        if self.gpu_barrier and self.warm_forward and self.warm_backward:
            self.barrier_epoch += 1
            peer_epoch_barrier(
                self.epoch_flags, self.epoch_pointers, self.rank, self.barrier_epoch
            )
        else:
            dist.barrier(group=self.group, async_op=True).wait()

    def _maps(self, indices):
        return [
            DeviceExpertLayout.build(
                torch.where(
                    (indices >= rank * self.experts)
                    & (indices < (rank + 1) * self.experts),
                    indices - rank * self.experts,
                    -1,
                ),
                self.experts,
                128,
            )
            for rank in range(self.world)
        ]

    def _wait_reuse(self):
        if self.reuse_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self.reuse_event)

    def _reset(self, buffers):
        self._wait_reuse()
        if self.skip_padding:
            buffers[self.rank][0].zero_()
            buffers[self.rank][1].fill_(0x69696969)
        # The barrier also orders receiver initialization before peer stores.
        payload_count = 3 if buffers is self.fwd else 2
        for tensor in buffers[self.rank][payload_count:]:
            tensor.zero_()
        self._barrier()

    def _execute(self, send, calculate, consumers, warm):
        previous_sms = self.deep_gemm.get_num_sms()
        self.deep_gemm.set_num_sms(self.sms)
        try:
            gradients = []
            if not warm:
                send()
                self._barrier()
                gradients = [calculate(rank) for rank in range(self.world)]
                self._barrier()
                for consume in consumers:
                    consume()
            else:
                current = torch.cuda.current_stream(self.device)
                for stream in (self.sender, *self.computers, *self.consumers):
                    stream.wait_stream(current)
                with torch.cuda.stream(self.sender):
                    send()
                for rank, stream in enumerate(self.computers):
                    with torch.cuda.stream(stream):
                        gradients.append(calculate(rank))
                for stream, consume in zip(self.consumers, consumers, strict=False):
                    with torch.cuda.stream(stream):
                        consume()
            self._barrier()
            current = torch.cuda.current_stream(self.device)
            for result in gradients:
                if result is not None:
                    for tensor in result:
                        tensor.record_stream(current)
            return gradients
        finally:
            self.deep_gemm.set_num_sms(previous_sms)

    def __call__(self, x, indices, scores, w1, w2):
        tensors = (x, indices, scores, w1, w2)
        if any(t.device != self.device or not t.is_contiguous() for t in tensors):
            raise ValueError("PeerSchedule requires contiguous tensors on its device")
        if (
            x.shape != (self.tokens, self.hidden)
            or x.dtype != torch.bfloat16
            or indices.shape != (self.tokens, self.topk)
            or indices.dtype not in (torch.int32, torch.int64)
            or scores.shape != indices.shape
            or scores.dtype != torch.float32
            or w1.shape != (self.experts, 2 * self.inner, self.hidden)
            or w2.shape != (self.experts, self.hidden, self.inner)
            or any(w.dtype not in (torch.bfloat16, torch.float32) for w in (w1, w2))
        ):
            raise ValueError(
                "Inputs do not match the fixed peer schedule capacity/dtypes"
            )
        if self.weight_dtypes != (w1.dtype, w2.dtype):
            self.warm_forward = self.warm_backward = False
            self.weight_dtypes = (w1.dtype, w2.dtype)
        return _PeerFunction.apply(x, indices, scores, w1, w2, self)

    def _prepare_bf16_forward_weights(self, w1):
        return w1

    def _allocate_preactivations(self):
        allocate = torch.empty if self.dtype == "bf16" else torch.zeros
        return allocate(
            self.world,
            self.rows,
            2 * self.inner,
            device=self.device,
            dtype=torch.bfloat16,
        )

    def forward(self, x, indices, scores, w1, w2):
        values, scales, received_scores, ready, accumulation, complete = self.fwd[
            self.rank
        ]
        maps = self._maps(indices)
        if self.dtype == "bf16":
            q1, q2 = w1.bfloat16(), w2.bfloat16()
            q1 = self._prepare_bf16_forward_weights(q1)
            s1 = s2 = None
        else:
            q1, s1 = quantize_fp8(w1)
            s1 = self.deep_gemm.transform_sf_into_required_layout(
                s1, 2 * self.inner, self.hidden, (1, 32), num_groups=self.experts
            )
            q2, s2 = quantize_fp8(w2)
        z = self._allocate_preactivations()
        output = torch.empty(
            (self.tokens, self.hidden), device=self.device, dtype=torch.float32
        )
        self._reset(self.fwd)

        def send():
            for rank, (v, s, rs, r, _, _) in enumerate(self.fwd):
                layout = maps[rank]
                self.dispatch_packed(
                    x,
                    layout.row_assignments,
                    layout.expert_ids,
                    layout.starts,
                    layout.padded_sizes,
                    layout.ends,
                    scores,
                    self._values(v[self.rank]),
                    s[self.rank],
                    r[self.rank],
                    *(t[self.rank] for t in self.metadata[rank]),
                    rs[self.rank],
                    self.skip_padding,
                )

        def calculate(sender):
            self.compute_forward(
                self._values(values[sender]),
                scales[sender].T,
                q1,
                s1,
                q2,
                s2,
                ready[sender],
                received_scores[sender],
                z[sender],
                self.metadata[self.rank][0][sender],
                self.metadata[self.rank][2 if self.slot_combine else 1][sender],
                self.fwd[sender][4],
                self.fwd[sender][5],
                self.rank,
                self.slot_combine,
            )

        self._execute(
            send,
            calculate,
            [lambda: self._finish_tokens(accumulation, output, complete)],
            self.warm_forward,
        )
        self.warm_forward = True
        saved = (
            values.clone(),
            scales.clone(),
            received_scores.clone(),
            z,
            *(t.clone() for t in self.metadata[self.rank]),
            *self._save_dispatch(maps),
        )
        self.reuse_event = torch.cuda.current_stream(self.device).record_event()
        return output, saved

    @staticmethod
    def _save_dispatch(maps):
        return tuple(
            t
            for layout in maps
            for t in (layout.token_ids(), layout.chunk_descriptors())
        )

    def _prepare_bf16_backward_weights(self, w1, w2):
        return dict(
            w1t=w1.transpose(1, 2).contiguous(),
            w2t=w2.transpose(1, 2).contiguous(),
        )

    def backward(self, dy, indices, w1, w2, saved):
        values, scales, scores, z, expert_ids, token_ids, score_ids, ends, *dispatch = (
            saved
        )
        incoming, incoming_scales, ready, dx_acc, ds_acc, dx_flags, ds_flags = self.bwd[
            self.rank
        ]
        prepared_weights = {}
        if self.dtype == "bf16":
            q1, q2 = w1.bfloat16(), w2.bfloat16()
            s1 = s2 = None
            prepared_weights = self._prepare_bf16_backward_weights(q1, q2)
        else:
            q1, s1 = quantize_fp8(w1.transpose(1, 2).contiguous())
            q2, s2 = quantize_fp8(w2.transpose(1, 2).contiguous())
            s2 = self.deep_gemm.transform_sf_into_required_layout(
                s2, self.inner, self.hidden, (1, 32), num_groups=self.experts
            )
        dx = torch.empty(
            (self.tokens, self.hidden), device=self.device, dtype=torch.float32
        )
        ds = torch.empty_like(ds_acc)
        self._reset(self.bwd)

        def send():
            for rank, (v, s, r, *_rest) in enumerate(self.bwd):
                r[self.rank].copy_(dispatch[2 * rank + 1])
                self.dispatch(
                    dy,
                    dispatch[2 * rank],
                    self._values(v[self.rank]),
                    s[self.rank],
                    r[self.rank],
                    0,
                    32,
                    self.experts,
                    self.skip_padding,
                )

        def calculate(sender):
            return self.compute_backward(
                self._values(incoming[sender]),
                incoming_scales[sender].T,
                self._values(values[sender]),
                scales[sender].T,
                q1,
                s1,
                q2,
                s2,
                ready[sender],
                scores[sender],
                z[sender],
                expert_ids[sender],
                ends[sender],
                token_ids[sender],
                score_ids[sender],
                self.bwd[sender][3],
                self.bwd[sender][5],
                self.bwd[sender][4],
                self.bwd[sender][6],
                self.rank,
                self.slot_combine,
                **prepared_weights,
            )

        grads = self._execute(
            send,
            calculate,
            [
                lambda: self._finish_tokens(dx_acc, dx, dx_flags),
                lambda: self.finish(ds_acc, ds, ds_flags),
            ],
            self.warm_backward,
        )
        self.warm_backward = True
        result = (
            dx,
            ds.reshape(self.tokens, self.topk),
            sum(g[0] for g in grads),
            sum(g[1] for g in grads),
        )
        self.reuse_event = torch.cuda.current_stream(self.device).record_event()
        return result
