# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Peer autograd, changing routing, outstanding forwards, and AdamW updates."""

import argparse
import json
import os
from functools import partial
from unittest.mock import Mock

import deep_gemm
import torch
import torch.distributed as dist
from mono_ep.peer_runtime import PeerSchedule


class DelayedPeerSchedule(PeerSchedule):
    fused_forward = False

    def _prepare_bf16_forward_weights(self, w1):
        if self.fused_forward:
            return (
                w1.unflatten(1, (2, self.inner))
                .transpose(1, 2)
                .flatten(1, 2)
                .contiguous()
            )
        return w1

    def _reset(self, buffers):
        self._wait_reuse()
        payload_count = 3 if buffers is self.fwd else 2
        for tensor in buffers[self.rank][:payload_count]:
            tensor.fill_(
                255
                if tensor.dtype == torch.uint8
                else -1
                if tensor.dtype == torch.int32
                else float("nan")
            )
        super()._reset(buffers)

    def _check_payload(self, buffers):
        values, scales, *rest = buffers[self.rank]
        if self.dtype == "bf16":
            assert torch.isfinite(values).all()
        else:
            assert (values != 255).all()
            assert (scales != -1).all()
        if buffers is self.fwd:
            assert torch.isfinite(rest[0]).all()

    def forward(self, *args):
        result = super().forward(*args)
        self._check_payload(self.fwd)
        return result

    def backward(self, *args):
        result = super().backward(*args)
        self._check_payload(self.bwd)
        return result

    def _execute(self, send, calculate, consumers, warm):
        def delayed_send():
            if warm:
                torch.cuda._sleep(10_000_000 * (self.rank + 1))
            send()

        return super()._execute(delayed_send, calculate, consumers, warm)


def gather(t):
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, t.detach())
    return torch.cat(parts)


def reference(x, logits, ids, w1, w2):
    scores = logits.softmax(-1)
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        z = x @ w1[expert].bfloat16().T
        gate, up = z.float().chunk(2, -1)
        h = torch.nn.functional.silu(gate) * up
        for slot in range(ids.shape[1]):
            weight = scores[:, slot].masked_fill(ids[:, slot] != expert, 0)
            output = (
                output
                + ((h * weight[:, None]).bfloat16() @ w2[expert].bfloat16().T).float()
            )
    return output.bfloat16()


def error(got, want, tolerance):
    value = (
        (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-8)
    ).item()
    assert torch.isfinite(got).all() and value < tolerance, value
    return value


def run(
    compiled,
    experts=1,
    slots=False,
    gpu_barrier=False,
    skip_padding=False,
    dtype="fp8",
    compute_lanes=None,
    incremental_ready=False,
    fused_forward=False,
):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    assert world >= 2
    default_sms = deep_gemm.get_num_sms()
    torch.manual_seed(471 + rank)
    tokens, hidden, inner, topk = 137, 256, 128, 2
    torch.cuda._sleep(1)
    torch.cuda.synchronize()
    runtime = DelayedPeerSchedule(
        group=dist.group.WORLD,
        tokens=tokens,
        topk=topk,
        hidden=hidden,
        inner=inner,
        compile_local=compiled,
        experts=experts,
        slot_combine=slots,
        gpu_barrier=gpu_barrier,
        skip_padding=skip_padding,
        dtype=dtype,
        compute_lanes=compute_lanes,
        incremental_ready=incremental_ready,
    )
    if fused_forward:
        if dtype != "bf16":
            raise ValueError("Fused peer forward validation requires BF16")
        from mono_ep.bf16 import _peer_forward_fused

        runtime.fused_forward = True
        callback = partial(_peer_forward_fused, incremental=incremental_ready)
        runtime.compute_forward = (
            torch.compile(callback, fullgraph=True) if compiled else callback
        )
    if dtype == "bf16":
        assert runtime.fwd[rank][1].shape == (world, 1, 1)
        assert runtime.bwd[rank][1].shape == (world, 1, 1)
    runtime._maps = Mock(wraps=runtime._maps)
    caller = torch.cuda.Stream()
    weights = [
        torch.nn.Parameter(torch.randn(shape, device="cuda") / scale**0.5)
        for shape, scale in [
            ((experts, 2 * inner, hidden), hidden),
            ((experts, hidden, inner), inner),
        ]
    ]
    reference_weights = [torch.nn.Parameter(w.detach().clone()) for w in weights]
    optimizers = [
        torch.optim.AdamW(ws, lr=1e-3, foreach=False)
        for ws in (weights, reference_weights)
    ]
    for step in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        retained = []
        expected_weight_grads = [torch.zeros_like(w) for w in weights]
        for micro in range(2):
            x = torch.randn(
                tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            logits = torch.randn(tokens, topk, device="cuda", requires_grad=True)
            ids = torch.randint(
                world * experts, (tokens, topk), device="cuda", dtype=torch.int32
            )
            ids[:: (3 + step + micro)] = -1
            if experts > 1:
                ids[ids == world * experts - 1] = -1
                if step == 1 and micro == 0:
                    ids.fill_(rank * experts)
                    ids[::4, 1] = rank * experts + 1
                if step == 2 and micro == 1:
                    ids.fill_(-1)
            dy = torch.randn_like(x) / tokens
            full_x, full_logits = (
                gather(x).requires_grad_(),
                gather(logits).requires_grad_(),
            )
            full_weights = [gather(w).requires_grad_() for w in reference_weights]
            expected = reference(full_x, full_logits, gather(ids), *full_weights)
            ref_grads = torch.autograd.grad(
                expected, (full_x, full_logits, *full_weights), gather(dy)
            )
            scores = logits.softmax(-1)
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                actual = runtime(x, ids, scores, *weights)
            torch.cuda.current_stream().wait_stream(caller)
            assert deep_gemm.get_num_sms() == default_sms
            lo, hi = rank * tokens, (rank + 1) * tokens
            forward_error = error(
                actual, expected[lo:hi], (0.02 if dtype == "bf16" else 0.15)
            )
            retained.append(
                (
                    actual,
                    dy,
                    x,
                    logits,
                    ref_grads[0][lo:hi],
                    ref_grads[1][lo:hi],
                    forward_error,
                )
            )
            for total, grad in zip(expected_weight_grads, ref_grads[2:], strict=True):
                total.add_(grad[rank * experts : (rank + 1) * experts])
        assert runtime._maps.call_count == 2 * (step + 1)
        errors = []
        for actual, dy, x, logits, dx, dlogits, forward_error in reversed(retained):
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                actual.backward(dy)
            torch.cuda.current_stream().wait_stream(caller)
            assert deep_gemm.get_num_sms() == default_sms
            errors.append(
                [
                    forward_error,
                    error(x.grad, dx, (0.02 if dtype == "bf16" else 0.15)),
                    error(logits.grad, dlogits, (0.02 if dtype == "bf16" else 0.15)),
                ]
            )
        assert runtime._maps.call_count == 2 * (step + 1)
        weight_errors = [
            error(w.grad, expected, (0.02 if dtype == "bf16" else 0.15))
            for w, expected in zip(weights, expected_weight_grads, strict=True)
        ]
        for w, grad in zip(reference_weights, expected_weight_grads, strict=True):
            w.grad = grad
        for optimizer in optimizers:
            optimizer.step()
        update_errors = [
            error(w, r, 0.02) for w, r in zip(weights, reference_weights, strict=True)
        ]
        print(
            json.dumps(
                dict(
                    rank=rank,
                    world_size=world,
                    compute_sms_per_source=runtime.sms,
                    compute_lanes=runtime.compute_lanes,
                    incremental_ready=runtime.incremental_ready,
                    fused_forward=runtime.fused_forward,
                    local_scale_buffer_bytes=sum(
                        runtime_buf[rank][1].numel()
                        * runtime_buf[rank][1].element_size()
                        for runtime_buf in (runtime.fwd, runtime.bwd)
                    ),
                    delayed_dispatch=True,
                    poisoned_payloads=True,
                    nondefault_caller_stream=True,
                    gpu_barrier=gpu_barrier,
                    skip_padding=skip_padding,
                    barrier_epoch=runtime.barrier_epoch,
                    slot_combine=slots,
                    step=step,
                    compiled=compiled,
                    compute_dtype=dtype,
                    experts=experts,
                    output_input_router_relative_l2=errors,
                    weight_gradient_relative_l2=weight_errors,
                    updated_weight_relative_l2=update_errors,
                )
            ),
            flush=True,
        )
    if gpu_barrier:
        cases = []
        expected = []
        for index in range(3):
            x = torch.randn(
                tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            scores = torch.randn(tokens, topk, device="cuda", requires_grad=True)
            ids = torch.randint(
                world * experts, (tokens, topk), device="cuda", dtype=torch.int32
            )
            ids[:: 3 + index] = -1
            dy = torch.randn_like(x) / tokens
            cases.append((x, scores, ids, dy))
            output = runtime(x, ids, scores, *weights)
            gradients = torch.autograd.grad(output, (x, scores, *weights), dy)
            expected.append((output.detach().clone(), *gradients))
        torch.cuda.synchronize()
        runtime._check_payload = lambda _: None
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        outputs, actual = [], [None] * len(cases)
        for index, (x, scores, ids, _) in enumerate(cases):
            stream = streams[index % len(streams)]
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                outputs.append(runtime(x, ids, scores, *weights))
        for index in reversed(range(len(cases))):
            x, scores, _, dy = cases[index]
            with torch.cuda.stream(streams[index % len(streams)]):
                gradients = torch.autograd.grad(
                    outputs[index], (x, scores, *weights), dy
                )
                actual[index] = (outputs[index].detach(), *gradients)
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for got, want in zip(actual, expected, strict=True):
            for tensor, reference_tensor in zip(got, want, strict=True):
                torch.testing.assert_close(
                    tensor, reference_tensor, rtol=1e-5, atol=1e-5
                )
        print(
            json.dumps(
                dict(
                    rank=rank,
                    asynchronous_alternating_streams=True,
                    outstanding_forwards=len(cases),
                    barrier_epoch=runtime.barrier_epoch,
                )
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--slots", action="store_true")
    parser.add_argument("--gpu-barrier", action="store_true")
    parser.add_argument("--skip-padding", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument("--compute-lanes", type=int)
    parser.add_argument("--incremental-ready", action="store_true")
    parser.add_argument("--fused-forward", action="store_true")
    args = parser.parse_args()
    run(
        args.compiled,
        args.experts,
        args.slots,
        args.gpu_barrier,
        args.skip_padding,
        args.dtype,
        args.compute_lanes,
        args.incremental_ready,
        args.fused_forward,
    )
