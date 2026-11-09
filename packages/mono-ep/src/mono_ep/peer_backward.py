# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Experimental backward with GPU-count routing and direct BF16 peer payloads."""

import socket

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .grouped import expert_backward_packed
from .packing import DeviceExpertLayout
from .peer_packing import pack_peer_rows, pack_peer_rows_ready_out, return_peer_rows
from .synchronization import peer_epoch_barrier


class PeerBackward:
    """Reusable single-node workspace; ranks must execute backwards in order.

    Inputs remain invocation-owned until backward. Only backward scratch is shared.
    This stages compact peer payloads around grouped GEMMs, not a persistent fused
    backward kernel. Python epochs currently preclude CUDA graph replay.
    """

    def __init__(
        self,
        runtime,
        *,
        ready_gather=False,
        fused_peer_return=False,
        owned_blocks=False,
    ):
        if runtime.compute_dtype not in ("bf16", "mxfp8") or runtime.retain_activations:
            raise ValueError("Peer backward requires recomputing BF16 or MXFP8")
        if owned_blocks and (
            runtime.compute_dtype != "bf16" or ready_gather or fused_peer_return
        ):
            raise ValueError(
                "Owned blocks require BF16 staged gathering and separate return"
            )
        self.owned_blocks = owned_blocks
        if owned_blocks:
            from .owned_backward import _load_kernel

            _load_kernel()
        self.ready_gather = (
            ready_gather and runtime.compute_dtype == "bf16" and runtime.fuse_activation
        )
        if fused_peer_return and runtime.compute_dtype != "bf16":
            raise ValueError("Fused peer return currently requires BF16")
        self.fused_peer_return = fused_peer_return
        self.runtime = runtime
        self.group = runtime.group
        self.rank = dist.get_rank(self.group)
        self.world = dist.get_world_size(self.group)
        hosts = [None] * self.world
        dist.all_gather_object(hosts, socket.gethostname(), group=self.group)
        if len(set(hosts)) != 1:
            raise ValueError("Peer backward requires a single host")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.handles = []
        capacity, hidden, topk = runtime.max_tokens, runtime.hidden, runtime.topk
        self.x = self._allocate((capacity, hidden), torch.bfloat16)
        self.dy = self._allocate((capacity, hidden), torch.bfloat16)
        self.ids = self._allocate((capacity, topk), torch.int64)
        self.scores = self._allocate((capacity * topk, 1), torch.float32)
        self.dx = self._allocate((capacity * topk, hidden), torch.bfloat16)
        self.ds = self._allocate((capacity * topk,), torch.float32)
        self.dx_pointers = (
            torch.tensor(
                [t.data_ptr() for t in self.dx], device=self.device, dtype=torch.int64
            )
            if fused_peer_return
            else None
        )
        self.flags = self._allocate((self.world,), torch.int32)
        self.pointers = torch.tensor(
            [t.data_ptr() for t in self.flags], device=self.device, dtype=torch.int64
        )
        self.flags[self.rank].zero_()
        dist.barrier(group=self.group)
        self.epoch = 0
        self.finished = None
        self.communication_stream = torch.cuda.Stream(device=self.device)
        self.gather_stream = torch.cuda.Stream(device=self.device)

    def _allocate(self, shape, dtype):
        local = symm_mem.empty(*shape, dtype=dtype, device=self.device)
        handle = symm_mem.rendezvous(local, group=self.group)
        self.handles.append((local, handle))
        return [handle.get_buffer(r, shape, dtype) for r in range(self.world)]

    def _barrier(self):
        self.epoch += 1
        peer_epoch_barrier(self.flags, self.pointers, self.rank, self.epoch)

    def backward(self, dy, x, indices, scores, w1, w2):
        runtime = self.runtime
        stream = torch.cuda.current_stream(self.device)
        if self.finished is not None:
            stream.wait_event(self.finished)
        tokens = x.shape[0]
        self.x[self.rank][:tokens].copy_(x)
        self.dy[self.rank][:tokens].copy_(dy)
        self.ids[self.rank][:tokens].copy_(indices)
        self.ids[self.rank][tokens:].fill_(-1)
        self.scores[self.rank][: scores.numel()].copy_(scores.reshape(-1, 1))
        self.dx[self.rank].zero_()
        self.ds[self.rank].zero_()
        self._barrier()

        all_ids = torch.cat(self.ids)
        experts = w1.shape[0]
        local_ids = torch.where(
            (all_ids >= self.rank * experts) & (all_ids < (self.rank + 1) * experts),
            all_ids - self.rank * experts,
            -1,
        )
        layout = DeviceExpertLayout.build(local_ids, experts, runtime.alignment)
        assignments = layout.row_assignments
        row_end = layout.starts[-1] + layout.padded_sizes[-1]
        self.communication_stream.wait_stream(stream)
        with torch.cuda.stream(self.communication_stream):
            packed_x = pack_peer_rows(self.x, assignments, runtime.topk, row_end)
            packed_scores = pack_peer_rows(
                self.scores, assignments, 1, row_end
            ).flatten()
            assignments.record_stream(self.communication_stream)
            row_end.record_stream(self.communication_stream)
        packed_x.record_stream(stream)
        packed_scores.record_stream(stream)
        precomputed_dh = None
        if self.ready_gather and runtime.fuse_activation:
            import deep_gemm

            from .chunk import ready_grouped_bf16_out

            packed_dy = dy.new_zeros((assignments.numel(), runtime.hidden))
            queue = layout.chunk_descriptors(chunks=8, alignment=runtime.alignment)
            counters = torch.zeros(
                queue.shape[0], device=self.device, dtype=torch.int32
            )
            precomputed_dh = dy.new_zeros((assignments.numel(), runtime.intermediate))
            wt = w2.bfloat16().transpose(1, 2).contiguous()
            self.gather_stream.wait_stream(stream)
            with torch.cuda.stream(self.gather_stream):
                pack_peer_rows_ready_out(
                    self.dy,
                    assignments,
                    runtime.topk,
                    packed_dy,
                    queue,
                    counters,
                    workers=8,
                )
                for tensor in (packed_dy, assignments, queue, counters):
                    tensor.record_stream(self.gather_stream)
            previous_sms = deep_gemm.get_num_sms()
            sms = torch.cuda.get_device_properties(self.device).multi_processor_count
            deep_gemm.set_num_sms(max(2, min(32, sms - 16) // 2 * 2))
            try:
                ready_grouped_bf16_out(
                    packed_dy,
                    wt,
                    precomputed_dh,
                    queue,
                    layout.expert_ids,
                    incremental=True,
                )
            finally:
                deep_gemm.set_num_sms(previous_sms)
            stream.wait_stream(self.gather_stream)
        else:
            packed_dy = pack_peer_rows(self.dy, assignments, runtime.topk, row_end)

        def wait_inputs():
            stream.wait_stream(self.communication_stream)

        def send_gradients(dx, ds):
            self.communication_stream.wait_stream(stream)
            with torch.cuda.stream(self.communication_stream):
                return_peer_rows(dx, ds, assignments, self.dx, self.ds, row_end)
                # The allocator must retain producer storage until peer writes finish.
                for tensor in (dx, ds, assignments, row_end):
                    if tensor is not None:
                        tensor.record_stream(self.communication_stream)
            return dx, ds

        def send_dx_from_gemm(dz, weights, expert_layout):
            from .peer_scatter import peer_gemm_out

            peer_gemm_out(
                dz,
                weights.transpose(1, 2).contiguous(),
                expert_layout.starts + expert_layout.padded_sizes,
                assignments.int(),
                self.dx,
                self.dx_pointers,
            )
            return None

        compute = expert_backward_packed
        options = {
            "input_ready": wait_inputs,
            "precomputed_dh": precomputed_dh,
            "owned_blocks": self.owned_blocks,
        }
        if self.fused_peer_return:
            options["output_gemm"] = send_dx_from_gemm
        if runtime.compute_dtype == "mxfp8":
            from .fp8 import expert_backward_fp8_packed

            compute = expert_backward_fp8_packed
            options = {"format": "mxfp8"}
            wait_inputs()
        _, _, dw1, dw2 = compute(
            packed_dy,
            packed_x,
            packed_scores,
            w1.bfloat16() if runtime.compute_dtype == "bf16" else w1,
            w2.bfloat16() if runtime.compute_dtype == "bf16" else w2,
            layout,
            fuse_activation=runtime.fuse_activation,
            output_exchange=send_gradients,
            **options,
        )
        stream.wait_stream(self.communication_stream)
        self._barrier()
        dx = self.dx[self.rank].reshape(
            runtime.max_tokens, runtime.topk, runtime.hidden
        )
        dx = dx[:tokens].sum(1, dtype=torch.float32).to(x.dtype)
        ds = self.ds[self.rank][: scores.numel()].reshape_as(scores).clone()
        self.finished = torch.cuda.Event()
        self.finished.record(stream)
        return dx, ds, dw1, dw2
