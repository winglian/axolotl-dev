# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Host-visible BF16/FP8 layer forward/backward timing, including routing and sync."""

import argparse
import contextlib
import functools
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import triton
from mono_ep import MonoEP
from mono_ep.packing import DeviceExpertLayout, _permutation
from mono_ep.peer_runtime import PeerSchedule


def cuda_timeline(trace):
    """Preserve kernel intervals; overlapping stream times are not additive."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        trace.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    kernels = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "kernel"
    ]
    if not kernels:
        raise RuntimeError("CUDA profile contains no kernel intervals")
    origin = min(event["ts"] for event in kernels)
    intervals = sorted(
        [
            event["ts"] - origin,
            event["dur"],
            event.get("args", {}).get("stream"),
            event["name"],
        ]
        for event in kernels
    )
    busy = 0.0
    end = 0.0
    for start, duration, _, _ in intervals:
        finish = start + duration
        busy += max(0.0, finish - max(start, end))
        end = max(end, finish)
    grouped = {}
    for _, duration, _, name in intervals:
        grouped.setdefault(name, []).append(duration)
    return dict(
        interval_columns=["start_us", "duration_us", "stream", "kernel"],
        intervals=intervals,
        kernel_busy_union_us=busy,
        first_to_last_kernel_us=end,
        longest_kernel_groups=sorted(
            [
                dict(
                    name=name,
                    count=len(durations),
                    median_us=statistics.median(durations),
                    max_us=max(durations),
                    sum_us=sum(durations),
                )
                for name, durations in grouped.items()
            ],
            key=lambda row: row["sum_us"],
            reverse=True,
        )[:30],
    )


def separate_backward_scatter(
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
):
    assert not assign
    from mono_ep.combine import peer_backward_scatter_publish
    from mono_ep.scheduled import fp8_packed_backward

    dx, ds, dw1, dw2 = fp8_packed_backward(
        dy, sy, x, sx, w1, s1, w2, s2, ready, scores, z, ids, ends
    )
    peer_backward_scatter_publish(
        dx, tokens, ds, slots, dx_acc, ds_acc, dx_flags, ds_flags, producer
    )
    return dw1, dw2


class BF16ForwardSchedule(PeerSchedule):
    fused_forward = True
    initialize_h = False
    initialize_z = False
    fused_scores = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from mono_ep.bf16 import _peer_backward, _peer_forward, _peer_forward_fused

        self.score_backward = {
            (fused, incremental): torch.compile(
                functools.partial(
                    _peer_backward,
                    fused_scores=fused,
                    incremental=incremental,
                ),
                fullgraph=True,
            )
            for fused in (False, True)
            for incremental in (False, True)
        }
        self.separate_compute = {
            incremental: torch.compile(
                functools.partial(_peer_forward, incremental=incremental),
                fullgraph=True,
            )
            for incremental in (False, True)
        }
        self.fused_compute = {
            (initialize, incremental): torch.compile(
                functools.partial(
                    _peer_forward_fused,
                    initialize_h=initialize,
                    incremental=incremental,
                ),
                fullgraph=True,
            )
            for initialize in (False, True)
            for incremental in (False, True)
        }

    def _prepare_bf16_backward_weights(self, w1, w2):
        self.compute_backward = self.score_backward[
            self.fused_scores, self.incremental_ready
        ]
        return super()._prepare_bf16_backward_weights(w1, w2)

    def _allocate_preactivations(self):
        z = super()._allocate_preactivations()
        if self.initialize_z:
            z.zero_()
        return z

    def _prepare_bf16_forward_weights(self, w1):
        self.compute_forward = (
            self.fused_compute[self.initialize_h, self.incremental_ready]
            if self.fused_forward
            else self.separate_compute[self.incremental_ready]
        )
        if self.fused_forward:
            return (
                w1.unflatten(1, (2, self.inner))
                .transpose(1, 2)
                .flatten(1, 2)
                .contiguous()
            )
        return w1


class BF16WeightSchedule(PeerSchedule):
    shared_weights = True

    def _prepare_bf16_backward_weights(self, w1, w2):
        if self.shared_weights:
            return super()._prepare_bf16_backward_weights(w1, w2)
        return {}


class DeviceBoundarySchedule(PeerSchedule):
    def _barrier(self):
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        torch.cuda.synchronize(self.device)


class FullResetSchedule(PeerSchedule):
    def _reset(self, buffers):
        self._wait_reuse()
        for tensor in buffers[self.rank]:
            tensor.zero_()
        self._barrier()


class FP32SlotSchedule(PeerSchedule):
    def _buffers(self, specs):
        return super()._buffers(
            [
                (shape, torch.float32 if dtype == torch.bfloat16 else dtype)
                for shape, dtype in specs
            ]
        )


class RebuiltRoutingSchedule(PeerSchedule):
    @staticmethod
    def _save_dispatch(maps):
        return ()

    def backward(self, dy, indices, w1, w2, saved):
        dispatch = PeerSchedule._save_dispatch(self._maps(indices))
        return super().backward(dy, indices, w1, w2, (*saved, *dispatch))


class SeparateScatterSchedule(PeerSchedule):
    def _execute(self, send, calculate, consumers, warm):
        from mono_ep.combine import (
            peer_scatter_add_publish,
            peer_scatter_store_publish,
        )

        if not getattr(self, "separate_scatter", True):
            return super()._execute(send, calculate, consumers, warm)
        original = self.deep_gemm.fp8_grouped_scatter_gemm_nt

        def separate(
            a,
            b,
            scratch,
            expert_ids,
            tokens,
            output,
            queue,
            task,
            assign=False,
            row_end=None,
        ):
            self.deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a, b, scratch, expert_ids, recipe_a=(1, 32), recipe_b=(1, 32)
            )
            scatter = peer_scatter_store_publish if assign else peer_scatter_add_publish
            scatter(scratch, tokens, output, queue, task)

        self.deep_gemm.fp8_grouped_scatter_gemm_nt = separate
        try:
            return super()._execute(send, calculate, consumers, warm)
        finally:
            self.deep_gemm.fp8_grouped_scatter_gemm_nt = original


class BoundedWorkSchedule(PeerSchedule):
    def _execute(self, send, calculate, consumers, warm):
        from mono_ep import column_quantization, peer_runtime

        activation = peer_runtime.swiglu_fp8
        if not getattr(self, "bound_activation", True):
            peer_runtime.swiglu_fp8 = lambda z, scores, row_end=None: activation(
                z, scores
            )
        plain = column_quantization.quantize_columns
        packed = column_quantization.quantize_packed_columns
        scatter = self.deep_gemm.fp8_grouped_scatter_gemm_nt
        if not getattr(self, "bound_columns", True):
            column_quantization.quantize_columns = lambda values, row_end=None: plain(
                values
            )
            column_quantization.quantize_packed_columns = (
                lambda values, scales, row_end=None: packed(values, scales)
            )
        if not getattr(self, "bound_scatter", True):

            def unbounded_scatter(*args, **kwargs):
                kwargs.pop("row_end", None)
                return scatter(*args, **kwargs)

            self.deep_gemm.fp8_grouped_scatter_gemm_nt = unbounded_scatter
        try:
            return super()._execute(send, calculate, consumers, warm)
        finally:
            peer_runtime.swiglu_fp8 = activation
            column_quantization.quantize_columns = plain
            column_quantization.quantize_packed_columns = packed
            self.deep_gemm.fp8_grouped_scatter_gemm_nt = scatter


class SeparateColumnsSchedule(PeerSchedule):
    def _execute(self, send, calculate, consumers, warm):
        if not getattr(self, "separate_columns", False):
            return super()._execute(send, calculate, consumers, warm)
        from mono_ep import column_quantization
        from mono_ep.scheduled import _dequantize_packed

        original = column_quantization.quantize_packed_columns

        def separate(values, scales, row_end=None):
            return column_quantization.quantize_columns(
                _dequantize_packed(values, scales), row_end
            )

        column_quantization.quantize_packed_columns = separate
        try:
            return super()._execute(send, calculate, consumers, warm)
        finally:
            column_quantization.quantize_packed_columns = original


class DispatchBoundarySchedule(PeerSchedule):
    def _execute(self, send, calculate, consumers, warm):
        if warm and getattr(self, "wait_dispatch", False):
            send()
            self._barrier()
            return super()._execute(lambda: None, calculate, consumers, warm)
        return super()._execute(send, calculate, consumers, warm)


class SeparateActivationSchedule(PeerSchedule):
    def _execute(self, send, calculate, consumers, warm):
        if not getattr(self, "separate_activation", False):
            return super()._execute(send, calculate, consumers, warm)
        from mono_ep.activation import swiglu

        original_gemm = self.deep_gemm.fp8_ready_grouped_gemm_nt
        original_epilogue = self.deep_gemm.epilogue.SwiGLUBackward

        def capture(z, scores, dz, partial):
            return (z, scores, dz, partial)

        def separate(a, b, output, ids, ready, **kwargs):
            epilogue = kwargs.pop("epilogue", None)
            if not isinstance(epilogue, tuple):
                if epilogue is not None:
                    kwargs["epilogue"] = epilogue
                return original_gemm(a, b, output, ids, ready, **kwargs)
            z, scores, dz, partial = epilogue
            original_gemm(a, b, output, ids, ready, **kwargs)
            h, dz_result, ds = swiglu(z, scores, output)
            output.copy_(h)
            dz.copy_(dz_result)
            partial.zero_()
            partial[:, 0].copy_(ds)

        self.deep_gemm.epilogue.SwiGLUBackward = capture
        self.deep_gemm.fp8_ready_grouped_gemm_nt = separate
        try:
            return super()._execute(send, calculate, consumers, warm)
        finally:
            self.deep_gemm.epilogue.SwiGLUBackward = original_epilogue
            self.deep_gemm.fp8_ready_grouped_gemm_nt = original_gemm


class SeparateAcquireSchedule(PeerSchedule):
    def _execute(self, send, calculate, consumers, warm):
        from mono_ep.combine import _wait_copy

        original = self.deep_gemm.fp8_ready_grouped_gemm_nt

        def separate(a, b, out, expert_ids, queue, epilogue=None):
            _wait_copy[(1,)](
                a[0], out, queue, queue.shape[0], 0, num_warps=4, debug=True
            )
            self.deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a,
                b,
                out,
                expert_ids,
                recipe_a=(1, 32),
                recipe_b=(1, 32),
                epilogue=epilogue,
            )

        self.deep_gemm.fp8_ready_grouped_gemm_nt = separate
        try:
            return super()._execute(send, calculate, consumers, warm)
        finally:
            self.deep_gemm.fp8_ready_grouped_gemm_nt = original


class UnfusedLayout(DeviceExpertLayout):
    @classmethod
    def build(cls, indices, experts, alignment):
        """Indices must be valid expert IDs or -1; no host-side data reads occur."""
        if indices.ndim != 2 or alignment < 128 or alignment % 128 or experts < 1:
            raise ValueError("invalid expert layout geometry")
        flat = indices.flatten()
        n = flat.numel()
        counts = torch.zeros(experts + 1, dtype=torch.int32, device=indices.device)
        counts.scatter_add_(
            0, flat.long() + 1, torch.ones_like(flat, dtype=torch.int32)
        )
        preceding = counts.cumsum(0, dtype=torch.int32)
        sizes = ((counts[1:] + alignment - 1) // alignment) * alignment
        bases = sizes.cumsum(0, dtype=torch.int32) - sizes
        ends = bases + counts[1:]
        order = flat.argsort(stable=True)
        capacity = triton.cdiv(n + experts * (alignment - 1), alignment) * alignment
        assignment_rows = torch.full((n,), -1, device=indices.device, dtype=torch.int64)
        row_assignments = torch.full(
            (capacity,), -1, device=indices.device, dtype=torch.int64
        )
        expert_ids = torch.full(
            (capacity,), -1, device=indices.device, dtype=torch.int32
        )
        if n:
            _permutation[(triton.cdiv(n, 256),)](
                flat,
                order,
                preceding,
                bases,
                assignment_rows,
                row_assignments,
                expert_ids,
                n,
                256,
            )
        return cls(
            assignment_rows,
            row_assignments,
            expert_ids,
            ends,
            indices.shape[1],
            bases,
            sizes,
        )


class UnfusedLayoutSchedule(PeerSchedule):
    def _maps(self, indices):
        return [
            UnfusedLayout.build(
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


class ConsumerFirstSchedule(PeerSchedule):
    def _execute(self, send, calculate, consumers, warm):
        if not warm:
            return super()._execute(send, calculate, consumers, warm)
        previous_sms = self.deep_gemm.get_num_sms()
        self.deep_gemm.set_num_sms(self.sms)
        try:
            current = torch.cuda.current_stream(self.device)
            for stream in (self.sender, *self.computers, *self.consumers):
                stream.wait_stream(current)
            gradients = []
            for stream, consume in zip(self.consumers, consumers, strict=False):
                with torch.cuda.stream(stream):
                    consume()
            for rank, stream in enumerate(self.computers):
                with torch.cuda.stream(stream):
                    gradients.append(calculate(rank))
            with torch.cuda.stream(self.sender):
                send()
            self._barrier()
            for result in gradients:
                if result is not None:
                    for tensor in result:
                        tensor.record_stream(current)
            return gradients
        finally:
            self.deep_gemm.set_num_sms(previous_sms)


def relative(got, want, tolerance=0.15):
    assert torch.isfinite(got).all()
    value = (
        (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-8)
    ).item()
    assert value < tolerance, value
    return value


def run(
    profile=False,
    compare_order=False,
    compare_layout=False,
    compare_acquire=False,
    compare_scatter=False,
    compare_routing=False,
    compare_sms=False,
    compare_backward_scatter=False,
    compare_slots=False,
    compare_slot_dtype=False,
    compare_reset=False,
    compare_boundary=False,
    compare_epoch=False,
    qwen=False,
    sweep_sms=False,
    qwen_scatter=False,
    qwen_columns=False,
    qwen_dispatch=False,
    qwen_swiglu=False,
    qwen_padding=False,
    qwen_bounds=False,
    routing="random",
    qwen_activation=False,
    compute_dtype="fp8",
    qwen_bf16_weights=False,
    qwen_bf16_forward=False,
    timing_batch_size=1,
    qwen_incremental=False,
):
    if qwen_incremental and not qwen_bf16_forward:
        raise ValueError("--qwen-incremental requires --qwen-bf16-forward")
    if timing_batch_size < 1:
        raise ValueError("timing_batch_size must be positive")
    if qwen_bf16_forward and (
        not qwen or compute_dtype != "bf16" or sweep_sms or qwen_bf16_weights
    ):
        raise ValueError(
            "--qwen-bf16-forward requires BF16 Qwen without other BF16 controls"
        )
    if qwen_bf16_weights and (not qwen or compute_dtype != "bf16" or sweep_sms):
        raise ValueError("--qwen-bf16-weights requires BF16 Qwen without an SM sweep")
    if compute_dtype not in ("bf16", "fp8"):
        raise ValueError(f"Unknown compute dtype: {compute_dtype}")
    if compute_dtype == "bf16" and (
        not qwen
        or any(
            (
                qwen_scatter,
                qwen_columns,
                qwen_dispatch,
                qwen_swiglu,
                qwen_padding,
                qwen_bounds,
                qwen_activation,
            )
        )
    ):
        raise ValueError("BF16 requires --qwen without FP8 comparison controls")
    if routing not in ("random", "unique", "skewed", "empty"):
        raise ValueError(f"Unknown routing pattern: {routing}")
    if qwen_activation and (
        not qwen
        or any(
            (
                qwen_bounds,
                qwen_padding,
                qwen_swiglu,
                qwen_dispatch,
                qwen_columns,
                qwen_scatter,
                sweep_sms,
            )
        )
    ):
        raise ValueError("--qwen-activation requires --qwen without other comparisons")
    if qwen_bounds and (
        not qwen
        or sweep_sms
        or qwen_scatter
        or qwen_columns
        or qwen_dispatch
        or qwen_swiglu
        or qwen_padding
    ):
        raise ValueError("--qwen-bounds requires --qwen without other comparisons")
    if qwen_padding and (
        not qwen or sweep_sms or qwen_scatter or qwen_columns or qwen_dispatch
    ):
        raise ValueError("--qwen-padding only combines with --qwen-swiglu")
    if qwen_swiglu and (
        not qwen or sweep_sms or qwen_scatter or qwen_columns or qwen_dispatch
    ):
        raise ValueError("--qwen-swiglu requires --qwen without other comparisons")
    if qwen_dispatch and (not qwen or sweep_sms or qwen_scatter or qwen_columns):
        raise ValueError("--qwen-dispatch requires --qwen without other comparisons")
    if qwen_columns and (not qwen or sweep_sms or qwen_scatter):
        raise ValueError("--qwen-columns requires --qwen without other comparisons")
    if qwen_scatter and (not qwen or sweep_sms):
        raise ValueError("--qwen-scatter requires --qwen without --sweep-sms")
    if sweep_sms and not qwen:
        raise ValueError("--sweep-sms requires --qwen")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    assert world == (8 if qwen else 2)
    if qwen and any(
        (
            compare_order,
            compare_layout,
            compare_acquire,
            compare_scatter,
            compare_routing,
            compare_sms,
            compare_backward_scatter,
            compare_slots,
            compare_slot_dtype,
            compare_reset,
            compare_boundary,
            compare_epoch,
        )
    ):
        raise ValueError("Qwen mode only compares NCCL and the fused peer schedule")
    if compute_dtype == "bf16":
        from importlib.metadata import version

        from transformers.integrations.sonicmoe import load_sonicmoe_kernel

        from axolotl.integrations.kernels.plugin import (
            _check_sonicmoe_gpu_compat,
            _preload_sonicmoe_kernel,
        )

        _check_sonicmoe_gpu_compat()
        if not _preload_sonicmoe_kernel():
            raise RuntimeError("Could not preload the BF16 SonicMoE baseline")
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        baseline_kernel_file=getattr(
                            getattr(
                                load_sonicmoe_kernel().moe_general_routing_inputs,
                                "__code__",
                                None,
                            ),
                            "co_filename",
                            None,
                        ),
                        baseline_versions={
                            name: version(name)
                            for name in (
                                "transformers",
                                "kernels",
                                "nvidia-cutlass-dsl",
                                "apache-tvm-ffi",
                            )
                        },
                        axolotl_sonic_compatibility=True,
                    )
                ),
                flush=True,
            )
    torch.manual_seed(613 + rank)
    shapes = (
        ((16384, 2048, 768, 16, 8),)
        if qwen
        else (
            (137, 256, 128, 4, 2),
            (4096, 2048, 768, 8, 4),
        )
    )
    for tokens, hidden, inner, experts, topk in shapes:
        peer_class = SeparateScatterSchedule if qwen_scatter else PeerSchedule
        if qwen_columns:
            peer_class = SeparateColumnsSchedule
        if qwen_dispatch:
            peer_class = DispatchBoundarySchedule
        if qwen_swiglu:
            peer_class = SeparateActivationSchedule
        if qwen_bounds or qwen_activation:
            peer_class = BoundedWorkSchedule
        if qwen_bf16_weights:
            peer_class = BF16WeightSchedule
        if qwen_bf16_forward:
            peer_class = BF16ForwardSchedule
        peer = peer_class(
            group=dist.group.WORLD,
            tokens=tokens,
            topk=topk,
            hidden=hidden,
            inner=inner,
            experts=experts,
            compile_local=True,
            slot_combine=qwen,
            gpu_barrier=qwen,
            skip_padding=qwen_bounds or qwen_activation or compute_dtype == "bf16",
            dtype=compute_dtype,
            compute_lanes=1 if qwen_bf16_weights or qwen_bf16_forward else None,
        )
        if compare_sms:
            peer.sms = min(32, peer.sms)
        pipeline = MonoEP(
            dtype=compute_dtype, compile_local=True, retain_activations=True
        )
        x = torch.randn(
            tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        scores = torch.randn(tokens, topk, device="cuda").softmax(-1).requires_grad_()
        ids = torch.randint(
            world * experts, (tokens, topk), device="cuda", dtype=torch.int32
        )
        if routing in ("unique", "skewed"):
            choices = experts if routing == "skewed" else world * experts
            ids = (
                torch.rand(tokens, choices, device="cuda")
                .topk(topk, dim=-1)
                .indices.to(torch.int32)
            )
        elif routing == "empty":
            ids.fill_(-1)
        route_counts = torch.bincount(ids[ids >= 0].long(), minlength=world * experts)
        dist.all_reduce(route_counts)
        routing_counts = route_counts.cpu().tolist()
        w1 = (
            torch.randn(experts, 2 * inner, hidden, device="cuda") / hidden**0.5
        ).requires_grad_()
        w2 = (
            torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
        ).requires_grad_()
        dy = torch.randn_like(x) / tokens
        parameters = (x, scores, w1, w2)

        def nccl(x=x, ids=ids, scores=scores, w1=w1, w2=w2, pipeline=pipeline):
            return pipeline(x, ids, scores, w1, w2, group=dist.group.WORLD)

        def queued(x=x, ids=ids, scores=scores, w1=w1, w2=w2, peer=peer):
            return peer(x, ids, scores, w1, w2)

        operations = {"nccl": nccl, "peer": queued}
        if qwen_bf16_weights:
            operations = {"nccl": nccl}
            warmed_weights = set()
            for shared in (False, True):

                def weights_variant(
                    shared=shared, peer=peer, queued=queued, warmed=warmed_weights
                ):
                    if shared not in warmed:
                        peer.warm_forward = peer.warm_backward = False
                        warmed.add(shared)
                    peer.shared_weights = shared
                    return queued()

                operations[
                    "peer_shared_weights" if shared else "peer_per_source_weights"
                ] = weights_variant
        if qwen_bf16_forward:
            operations = {"nccl": nccl}
            warmed_forward = set()
            forward_variants = (
                (False, False, False, True),
                (False, False, True, True),
                (True, True, False, True),
                (True, False, True, True),
                (True, False, False, True),
                (True, False, False, False),
            )
            variants = [(*v, False) for v in forward_variants]
            if qwen_incremental:
                variants = [
                    (fused, False, False, False, incremental)
                    for fused in (False, True)
                    for incremental in (False, True)
                ]
            for fused, initialize, initialize_z, fused_scores, incremental in variants:

                def forward_variant(
                    fused=fused,
                    initialize=initialize,
                    initialize_z=initialize_z,
                    fused_scores=fused_scores,
                    incremental=incremental,
                    peer=peer,
                    queued=queued,
                    warmed=warmed_forward,
                ):
                    variant = (
                        fused,
                        initialize,
                        initialize_z,
                        fused_scores,
                        incremental,
                    )
                    if variant not in warmed:
                        peer.warm_forward = peer.warm_backward = False
                        warmed.add(variant)
                    peer.incremental_ready = incremental
                    peer.fused_forward = fused
                    peer.initialize_h = initialize
                    peer.initialize_z = initialize_z
                    peer.fused_scores = fused_scores
                    return queued()

                name = "peer_separate_forward"
                if fused:
                    name = (
                        "peer_fused_forward_zeroed"
                        if initialize
                        else "peer_fused_forward"
                    )
                if initialize_z:
                    name += "_preactivation_zeroed"
                if not fused_scores:
                    name += "_separate_scores"
                if incremental:
                    name += "_incremental"
                operations[name] = forward_variant
        if qwen_bounds:
            warmed_variants = set()
            operations = {"nccl": nccl}
            for columns in (False, True):
                for scatter in (False, True):

                    def bounds_variant(
                        columns=columns,
                        scatter=scatter,
                        peer=peer,
                        queued=queued,
                        warmed=warmed_variants,
                    ):
                        key = (columns, scatter)
                        if key not in warmed:
                            peer.warm_forward = peer.warm_backward = False
                            warmed.add(key)
                        peer.bound_columns = columns
                        peer.bound_scatter = scatter
                        return queued()

                    name = (
                        "peer_"
                        + ("bounded" if columns else "full")
                        + "_columns_"
                        + ("bounded" if scatter else "full")
                        + "_scatter"
                    )
                    operations[name] = bounds_variant
        if qwen_activation:
            warmed_variants = set()
            operations = {"nccl": nccl}
            for bound in (False, True):

                def activation_variant(
                    bound=bound, peer=peer, queued=queued, warmed=warmed_variants
                ):
                    if bound not in warmed:
                        peer.warm_forward = peer.warm_backward = False
                        warmed.add(bound)
                    peer.bound_activation = bound
                    return queued()

                operations[
                    "peer_bounded_activation" if bound else "peer_full_activation"
                ] = activation_variant
        if qwen_padding:
            warmed_variants = set()
            operations = {"nccl": nccl}
            for skip in (False, True):

                def padding_variant(
                    skip=skip, peer=peer, queued=queued, warmed=warmed_variants
                ):
                    if skip not in warmed:
                        peer.warm_forward = peer.warm_backward = False
                        warmed.add(skip)
                    peer.skip_padding = skip
                    return queued()

                operations["peer_local_padding" if skip else "peer_remote_padding"] = (
                    padding_variant
                )
        if qwen_scatter:
            warmed_variants = set()
            operations = {"nccl": nccl}
            for separate in (False, True):

                def scatter_variant(
                    separate=separate, peer=peer, queued=queued, warmed=warmed_variants
                ):
                    if separate not in warmed:
                        peer.warm_forward = peer.warm_backward = False
                        warmed.add(separate)
                    peer.separate_scatter = separate
                    return queued()

                operations[
                    "peer_separate_scatter" if separate else "peer_fused_scatter"
                ] = scatter_variant
        if qwen_swiglu:
            warmed_variants = set()
            operations = {"nccl": nccl}
            for separate in (False, True):

                def activation_variant(
                    separate=separate, peer=peer, queued=queued, warmed=warmed_variants
                ):
                    if separate not in warmed:
                        peer.warm_forward = peer.warm_backward = False
                        warmed.add(separate)
                    peer.separate_activation = separate
                    return queued()

                operations[
                    "peer_separate_swiglu" if separate else "peer_fused_swiglu"
                ] = activation_variant
        if qwen_padding and qwen_swiglu:
            warmed_variants = set()
            operations = {"nccl": nccl}
            for skip in (False, True):
                for separate in (False, True):

                    def combined_variant(
                        skip=skip,
                        separate=separate,
                        peer=peer,
                        queued=queued,
                        warmed=warmed_variants,
                    ):
                        key = (skip, separate)
                        if key not in warmed:
                            peer.warm_forward = peer.warm_backward = False
                            warmed.add(key)
                        peer.skip_padding = skip
                        peer.separate_activation = separate
                        return queued()

                    name = (
                        "peer_"
                        + ("local" if skip else "remote")
                        + "_padding_"
                        + ("separate" if separate else "fused")
                        + "_swiglu"
                    )
                    operations[name] = combined_variant
        if qwen_dispatch:
            operations = {"nccl": nccl}
            for wait in (False, True):

                def dispatch_variant(wait=wait, peer=peer, queued=queued):
                    peer.wait_dispatch = wait
                    return queued()

                operations["peer_dispatch_boundary" if wait else "peer_overlap"] = (
                    dispatch_variant
                )
        if qwen_columns:
            warmed_variants = set()
            operations = {"nccl": nccl}
            for separate in (False, True):

                def column_variant(
                    separate=separate, peer=peer, queued=queued, warmed=warmed_variants
                ):
                    if separate not in warmed:
                        peer.warm_forward = peer.warm_backward = False
                        warmed.add(separate)
                    peer.separate_columns = separate
                    return queued()

                operations[
                    "peer_separate_columns" if separate else "peer_fused_columns"
                ] = column_variant
        if sweep_sms:
            operations = {"nccl": nccl}
            warmed_budgets = set()
            compute_streams = peer.computers
            available = (
                torch.cuda.get_device_properties(x.device).multi_processor_count - 16
            )
            for sms in (peer.sms, 32, 64, available - available % 2):

                def with_sms(
                    sms=sms,
                    peer=peer,
                    queued=queued,
                    warmed=warmed_budgets,
                    streams=compute_streams,
                    available=available,
                ):
                    if sms not in warmed:
                        peer.warm_forward = peer.warm_backward = False
                        warmed.add(sms)
                    peer.sms = sms
                    lanes = min(peer.world, available // sms)
                    peer.computers = tuple(
                        streams[i % lanes] for i in range(peer.world)
                    )
                    return queued()

                operations[f"peer_sms_{sms}"] = with_sms
        if compare_order:
            old_peer = ConsumerFirstSchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
            )

            def consumer_first(
                old_peer=old_peer, x=x, ids=ids, scores=scores, w1=w1, w2=w2
            ):
                return old_peer(x, ids, scores, w1, w2)

            operations["peer_consumer_first"] = consumer_first
        if compare_layout:
            old_layout_peer = UnfusedLayoutSchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
            )
            for new, old in zip(
                peer._maps(ids), old_layout_peer._maps(ids), strict=True
            ):
                for field in (
                    "assignment_rows",
                    "row_assignments",
                    "expert_ids",
                    "starts",
                    "ends",
                    "padded_sizes",
                ):
                    torch.testing.assert_close(
                        getattr(new, field), getattr(old, field), atol=0, rtol=0
                    )

            def unfused_layout(
                old_layout_peer=old_layout_peer,
                x=x,
                ids=ids,
                scores=scores,
                w1=w1,
                w2=w2,
            ):
                return old_layout_peer(x, ids, scores, w1, w2)

            operations["peer_unfused_layout"] = unfused_layout
        if compare_acquire:
            separate_peer = SeparateAcquireSchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
            )

            def separate_acquire(
                separate_peer=separate_peer, x=x, ids=ids, scores=scores, w1=w1, w2=w2
            ):
                return separate_peer(x, ids, scores, w1, w2)

            operations["peer_separate_acquire"] = separate_acquire
        if compare_scatter:
            separate_scatter_peer = SeparateScatterSchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
            )

            def separate_scatter(
                separate_scatter_peer=separate_scatter_peer,
                x=x,
                ids=ids,
                scores=scores,
                w1=w1,
                w2=w2,
            ):
                return separate_scatter_peer(x, ids, scores, w1, w2)

            operations["peer_separate_scatter"] = separate_scatter
        if compare_routing:
            rebuilt_peer = RebuiltRoutingSchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
            )

            def rebuilt_routing(
                rebuilt_peer=rebuilt_peer, x=x, ids=ids, scores=scores, w1=w1, w2=w2
            ):
                return rebuilt_peer(x, ids, scores, w1, w2)

            operations["peer_rebuilt_routing"] = rebuilt_routing
        if compare_sms:
            wider_peer = PeerSchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
            )
            available = (
                torch.cuda.get_device_properties(x.device).multi_processor_count - 16
            ) // world
            wider_peer.sms = available - available % 2

            def wider_sms(
                wider_peer=wider_peer, x=x, ids=ids, scores=scores, w1=w1, w2=w2
            ):
                return wider_peer(x, ids, scores, w1, w2)

            operations[f"peer_sms_{wider_peer.sms}"] = wider_sms
        if compare_backward_scatter:
            separate_backward_peer = PeerSchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
            )
            separate_backward_peer.compute_backward = torch.compile(
                separate_backward_scatter, fullgraph=True, dynamic=False
            )

            def separate_backward(
                separate_backward_peer=separate_backward_peer,
                x=x,
                ids=ids,
                scores=scores,
                w1=w1,
                w2=w2,
            ):
                return separate_backward_peer(x, ids, scores, w1, w2)

            operations["peer_separate_backward_scatter"] = separate_backward
        if (
            compare_slots
            or compare_slot_dtype
            or compare_reset
            or compare_boundary
            or compare_epoch
        ):
            slot_peer = PeerSchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
                slot_combine=True,
            )

            def slot_combine(
                slot_peer=slot_peer, x=x, ids=ids, scores=scores, w1=w1, w2=w2
            ):
                return slot_peer(x, ids, scores, w1, w2)

            operations["peer_slots"] = slot_combine
        if compare_slot_dtype:
            fp32_slot_peer = FP32SlotSchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
                slot_combine=True,
            )

            def fp32_slot_combine(
                fp32_slot_peer=fp32_slot_peer, x=x, ids=ids, scores=scores, w1=w1, w2=w2
            ):
                return fp32_slot_peer(x, ids, scores, w1, w2)

            operations["peer_fp32_slots"] = fp32_slot_combine
        if compare_reset:
            full_reset_peer = FullResetSchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
                slot_combine=True,
            )

            def full_reset(
                full_reset_peer=full_reset_peer,
                x=x,
                ids=ids,
                scores=scores,
                w1=w1,
                w2=w2,
            ):
                return full_reset_peer(x, ids, scores, w1, w2)

            operations["peer_slots_full_reset"] = full_reset
        if compare_boundary:
            device_boundary_peer = DeviceBoundarySchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
                slot_combine=True,
            )

            def device_boundary(
                device_boundary_peer=device_boundary_peer,
                x=x,
                ids=ids,
                scores=scores,
                w1=w1,
                w2=w2,
            ):
                return device_boundary_peer(x, ids, scores, w1, w2)

            operations["peer_slots_device_boundary"] = device_boundary
        if compare_epoch:
            epoch_peer = PeerSchedule(
                group=dist.group.WORLD,
                tokens=tokens,
                topk=topk,
                hidden=hidden,
                inner=inner,
                experts=experts,
                compile_local=True,
                slot_combine=True,
                gpu_barrier=True,
            )

            def epoch_boundary(
                epoch_peer=epoch_peer, x=x, ids=ids, scores=scores, w1=w1, w2=w2
            ):
                return epoch_peer(x, ids, scores, w1, w2)

            operations["peer_slots_gpu_barrier"] = epoch_boundary
        reference = None
        unbounded_reference = None
        parity = {}
        peer_variant_parity = {}
        peer_reference_name = (
            "peer_full_activation"
            if qwen_activation
            else "peer_full_columns_full_scatter"
        )
        if qwen_bf16_weights:
            peer_reference_name = "peer_per_source_weights"
        if qwen_bf16_forward:
            peer_reference_name = (
                "peer_separate_forward_separate_scores"
                if qwen_incremental
                else "peer_separate_forward"
            )
        for name, operation in operations.items():
            if rank == 0:
                print(json.dumps(dict(phase="parity", operation=name)), flush=True)
            for p in parameters:
                p.grad = None
            output = operation()
            output.backward(dy)
            torch.cuda.synchronize()
            actual = [output.detach(), *(p.grad.detach() for p in parameters)]
            if (
                qwen_bounds or qwen_activation or qwen_bf16_weights or qwen_bf16_forward
            ) and name == peer_reference_name:
                unbounded_reference = [t.clone() for t in actual]
            if (
                (
                    qwen_bounds
                    or qwen_activation
                    or qwen_bf16_weights
                    or qwen_bf16_forward
                )
                and name.startswith("peer_")
                and name != peer_reference_name
            ):
                if qwen_bf16_forward:
                    peer_variant_parity[name] = [
                        relative(got, want, 0.001)
                        for got, want in zip(actual, unbounded_reference, strict=True)
                    ]
                else:
                    for got, want in zip(actual, unbounded_reference, strict=True):
                        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
            if reference is None:
                reference = [t.clone() for t in actual]
            else:
                parity[name] = [
                    relative(a, b, 0.02 if compute_dtype == "bf16" else 0.15)
                    for a, b in zip(actual, reference, strict=True)
                ]
            dist.barrier()
        warmup_iterations, measured_iterations = 3, 10

        def measure(
            batch_size,
            operations=operations,
            parameters=parameters,
            dy=dy,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
        ):
            samples = {name: [] for name in operations}
            for trial in range(warmup_iterations + measured_iterations):
                order = list(operations) if trial % 2 else list(reversed(operations))
                for name in order:
                    for p in parameters:
                        p.grad = None
                    torch.cuda.synchronize()
                    dist.barrier()
                    torch.cuda.synchronize()
                    if rank == 0 and trial == 0:
                        print(
                            json.dumps(
                                dict(
                                    phase="warm_timing",
                                    operation=name,
                                    timing_batch_size=batch_size,
                                )
                            ),
                            flush=True,
                        )
                    start = time.perf_counter()
                    for iteration in range(batch_size):
                        if iteration:
                            for p in parameters:
                                p.grad = None
                        output = operations[name]()
                        output.backward(dy)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000 / batch_size
                    assert torch.isfinite(output).all()
                    for p in parameters:
                        assert p.grad is not None and torch.isfinite(p.grad).all()
                    if trial >= warmup_iterations:
                        samples[name].append(elapsed)
                    dist.barrier()
            return samples

        samples = measure(1)
        batched_samples = measure(timing_batch_size) if timing_batch_size > 1 else None
        print(
            json.dumps(
                dict(
                    rank=rank,
                    world_size=world,
                    gpu=torch.cuda.get_device_name(),
                    torch_version=torch.__version__,
                    cuda_version=torch.version.cuda,
                    compute="bf16" if compute_dtype == "bf16" else "mxfp8",
                    nccl_experts="sonicmoe_hub"
                    if compute_dtype == "bf16"
                    else "deepgemm",
                    peer_experts="deepgemm",
                    routing=routing,
                    global_expert_assignments=routing_counts,
                    retain_activations=True,
                    local_fullgraph=True,
                    whole_layer_fullgraph=False,
                    local_scale_buffer_bytes=sum(
                        buffers[rank][1].numel() * buffers[rank][1].element_size()
                        for buffers in (peer.fwd, peer.bwd)
                    ),
                    peer_slots=qwen,
                    peer_gpu_barrier=qwen,
                    tokens=tokens,
                    hidden=hidden,
                    inner=inner,
                    experts_per_rank=experts,
                    topk=topk,
                    peer_compute_sms_per_source=(
                        {
                            name: int(name.rsplit("_", 1)[1])
                            for name in operations
                            if name.startswith("peer_sms_")
                        }
                        if sweep_sms
                        else peer.sms
                    ),
                    parity_relative_l2=parity,
                    peer_variant_relative_l2=peer_variant_parity,
                    warmup_iterations=warmup_iterations,
                    measured_iterations=measured_iterations,
                    timing_scope="synchronized_host_forward_backward",
                    alternating_variant_order=True,
                    median_forward_backward_ms={
                        k: statistics.median(v) for k, v in samples.items()
                    },
                    samples_ms=samples,
                    timing_batch_size=timing_batch_size,
                    qwen_incremental=qwen_incremental,
                    batched_samples_ms_per_iteration=batched_samples,
                    batched_timing_scope="host_forward_backward_batch_with_grad_reset"
                    if batched_samples is not None
                    else None,
                )
            ),
            flush=True,
        )
        if profile:
            for name, operation in operations.items():
                dist.barrier()
                context = (
                    torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    )
                    if rank == 0
                    else contextlib.nullcontext()
                )
                with context as trace:
                    for _ in range(3):
                        for p in parameters:
                            p.grad = None
                        with torch.profiler.record_function(f"{name}_forward"):
                            output = operation()
                        with torch.profiler.record_function(f"{name}_backward"):
                            output.backward(dy)
                        torch.cuda.synchronize()
                if rank == 0:
                    events = list(trace.key_averages())

                    def event_data(event):
                        return dict(
                            name=event.key,
                            count=event.count,
                            self_cpu_us=event.self_cpu_time_total,
                            self_device_us=event.self_device_time_total,
                        )

                    print(
                        json.dumps(
                            dict(
                                profile=name,
                                cuda_timeline=cuda_timeline(trace),
                                tokens=tokens,
                                hidden=hidden,
                                iterations=3,
                                top_device=[
                                    event_data(e)
                                    for e in sorted(
                                        events,
                                        key=lambda e: e.self_device_time_total,
                                        reverse=True,
                                    )[:30]
                                ],
                                top_cpu=[
                                    event_data(e)
                                    for e in sorted(
                                        events,
                                        key=lambda e: e.self_cpu_time_total,
                                        reverse=True,
                                    )[:30]
                                ],
                            )
                        ),
                        flush=True,
                    )
                dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-dtype", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument(
        "--routing", choices=("random", "unique", "skewed", "empty"), default="random"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--sweep-sms", action="store_true")
    parser.add_argument("--qwen-scatter", action="store_true")
    parser.add_argument("--qwen-columns", action="store_true")
    parser.add_argument("--qwen-dispatch", action="store_true")
    parser.add_argument("--qwen-swiglu", action="store_true")
    parser.add_argument("--qwen-padding", action="store_true")
    parser.add_argument("--qwen-bounds", action="store_true")
    parser.add_argument("--qwen-activation", action="store_true")
    parser.add_argument("--qwen-bf16-weights", action="store_true")
    parser.add_argument("--qwen-bf16-forward", action="store_true")
    parser.add_argument("--timing-batch-size", type=int, default=1)
    parser.add_argument("--compare-order", action="store_true")
    parser.add_argument("--compare-layout", action="store_true")
    parser.add_argument("--compare-acquire", action="store_true")
    parser.add_argument("--compare-scatter", action="store_true")
    parser.add_argument("--compare-routing", action="store_true")
    parser.add_argument("--compare-sms", action="store_true")
    parser.add_argument("--compare-backward-scatter", action="store_true")
    parser.add_argument("--compare-slots", action="store_true")
    parser.add_argument("--compare-slot-dtype", action="store_true")
    parser.add_argument("--compare-reset", action="store_true")
    parser.add_argument("--compare-boundary", action="store_true")
    parser.add_argument("--compare-epoch", action="store_true")
    parser.add_argument("--qwen-incremental", action="store_true")
    args = parser.parse_args()
    run(
        routing=args.routing,
        compute_dtype=args.compute_dtype,
        qwen_bf16_weights=args.qwen_bf16_weights,
        qwen_bf16_forward=args.qwen_bf16_forward,
        timing_batch_size=args.timing_batch_size,
        qwen_incremental=args.qwen_incremental,
        profile=args.profile,
        qwen=args.qwen,
        sweep_sms=args.sweep_sms,
        qwen_scatter=args.qwen_scatter,
        qwen_columns=args.qwen_columns,
        qwen_dispatch=args.qwen_dispatch,
        qwen_swiglu=args.qwen_swiglu,
        qwen_padding=args.qwen_padding,
        qwen_bounds=args.qwen_bounds,
        qwen_activation=args.qwen_activation,
        compare_order=args.compare_order,
        compare_layout=args.compare_layout,
        compare_acquire=args.compare_acquire,
        compare_scatter=args.compare_scatter,
        compare_routing=args.compare_routing,
        compare_sms=args.compare_sms,
        compare_backward_scatter=args.compare_backward_scatter,
        compare_slots=args.compare_slots,
        compare_slot_dtype=args.compare_slot_dtype,
        compare_reset=args.compare_reset,
        compare_boundary=args.compare_boundary,
        compare_epoch=args.compare_epoch,
    )
