# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate retained persistent forward with FFT gradients and outstanding graphs."""

import argparse
import json
import os
from contextlib import ExitStack, nullcontext
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from mono_ep import PersistentMoE, fp8, grouped, recompute as recompute_ops
from torch.utils.checkpoint import checkpoint


def relative(actual, expected, tolerance=0.02):
    assert torch.isfinite(actual).all()
    error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-8)
    assert error < tolerance, error.item()
    return error.item()


def reference(x, ids, scores, w1, w2):
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        token, slot = torch.where(ids == expert)
        z = x[token] @ w1[expert].bfloat16().T
        gate, up = z.float().chunk(2, -1)
        h = (torch.nn.functional.silu(gate) * up * scores[token, slot, None]).bfloat16()
        output = output.index_add(0, token, (h @ w2[expert].bfloat16().T).float())
    return output.bfloat16()


def run(
    compute_dtype="bf16",
    checkpointed=False,
    recompute=False,
    compiled=False,
    peer_backward=False,
    owned_blocks=False,
    native_backward=False,
    master_dtype="float32",
):
    if master_dtype not in ("float32", "bf16"):
        raise ValueError("master_dtype must be float32 or bf16")
    weight_dtype = torch.float32 if master_dtype == "float32" else torch.bfloat16
    import deep_gemm

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    if world < 2:
        raise ValueError("Distributed validation requires at least two ranks")
    hidden, inner, experts, topk, tokens = 512, 512, 4 * world, 4, 137
    local = experts // world
    torch.manual_seed(912)
    reference_weights = [
        torch.nn.Parameter(
            (torch.randn(shape, device="cuda") / hidden**0.5).to(weight_dtype)
        )
        for shape in ((experts, 2 * inner, hidden), (experts, hidden, inner))
    ]
    weights = [
        torch.nn.Parameter(w.detach()[rank * local : (rank + 1) * local].clone())
        for w in reference_weights
    ]
    optimizer = torch.optim.AdamW(weights, lr=1e-4)
    ref_optimizer = torch.optim.AdamW(reference_weights, lr=1e-4)
    engine = None
    if native_backward:
        from mono_ep import MonoEP

        engine = MonoEP(
            backend="persistent",
            dtype="fp8" if compute_dtype == "blockwise" else "bf16",
            retain_activations=False,
            max_tokens=tokens + 31,
        )
        runtime = engine.prepare_persistent(
            group=dist.group.WORLD,
            device=torch.device("cuda", rank),
            experts=experts,
            topk=topk,
            hidden=hidden,
            intermediate=inner,
        )
    else:
        runtime = PersistentMoE(
            dist.group.WORLD,
            experts,
            tokens + 31,
            topk,
            hidden,
            inner,
            fuse_activation=True,
            retain_activations=not recompute,
            compute_dtype=compute_dtype,
            backward_backend="native" if native_backward else "nccl",
        )
    if peer_backward and not native_backward:
        from mono_ep.peer_backward import PeerBackward

        runtime._peer_backward = PeerBackward(
            runtime,
            ready_gather=not owned_blocks,
            fused_peer_return=compute_dtype == "bf16" and not owned_blocks,
            owned_blocks=owned_blocks,
        )
    native_forward_calls = 0
    native_name = "fp8_fp4_mono_ep" if compute_dtype == "mxfp8" else "bf16_mono_ep"
    native_module = deep_gemm
    if compute_dtype == "blockwise":
        native_module = deep_gemm._C
        native_name = "fp8_mono_ep_forward_sm90"
    original_native_forward = getattr(native_module, native_name)

    def counted_native_forward(*args, **kwargs):
        nonlocal native_forward_calls
        native_forward_calls += 1
        return original_native_forward(*args, **kwargs)

    setattr(native_module, native_name, counted_native_forward)
    operation = torch.compile(runtime, fullgraph=True) if compiled else runtime
    recompute_calls = 0
    fused_recompute_calls = 0
    import mono_ep.owned_backward as owned_ops

    if native_backward:
        original_native_backward = runtime._peer_backward.backward

        def counted_native_backward(*args):
            nonlocal recompute_calls
            recompute_calls += 1
            return original_native_backward(*args)

        runtime._peer_backward.backward = counted_native_backward

    original_owned = owned_ops.owned_backward

    def counted_owned(*args):
        nonlocal recompute_calls
        assert owned_blocks and recompute
        recompute_calls += 1
        return original_owned(*args)

    original_recompute = recompute_ops.grouped_recompute_swiglu

    def counted_recompute(*args):
        nonlocal recompute_calls, fused_recompute_calls
        assert recompute, "Retained backward recomputed W1"
        recompute_calls += 1
        fused_recompute_calls += 1
        return original_recompute(*args)

    tolerance = 0.15 if compute_dtype in ("mxfp8", "blockwise") else 0.02
    original_fp8_gemm = fp8.expert_gemm_fp8

    def fp8_derivative_gemm(x, w, ids, *, format, group_ends=None):
        nonlocal recompute_calls
        if w.shape[1:] == (2 * inner, hidden):
            assert recompute, "Retained FP8 backward recomputed W1"
            recompute_calls += 1
        return original_fp8_gemm(x, w, ids, format=format, group_ends=group_ends)

    original_gemm = grouped._gemm

    def derivative_gemm(x, w, layout, transpose=False):
        nonlocal recompute_calls
        if not transpose:
            assert recompute, "Retained backward recomputed the first projection"
            recompute_calls += 1
        return original_gemm(x, w, layout, transpose=transpose)

    forward_calls = 0

    def forward(*args):
        nonlocal forward_calls
        forward_calls += 1
        with (
            (
                patch.object(
                    torch.Tensor, "cpu", side_effect=AssertionError("Host tensor copy")
                )
                if not compiled
                else nullcontext()
            ),
            (
                patch.object(
                    torch.Tensor,
                    "tolist",
                    side_effect=AssertionError("Host split list"),
                )
                if not compiled
                else nullcontext()
            ),
            (
                patch.object(
                    torch.Tensor, "item", side_effect=AssertionError("Host scalar read")
                )
                if not compiled
                else nullcontext()
            ),
        ):
            return operation(*args)

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        cases = []
        forward_calls = 0
        native_forward_calls = 0
        recompute_calls = 0
        fused_recompute_calls = 0
        for micro in range(2):
            torch.manual_seed(1100 + step * 2 + micro)
            all_x = torch.randn(
                world * tokens, hidden, device="cuda", dtype=torch.bfloat16
            ).requires_grad_()
            all_scores = torch.rand(
                world * tokens, topk, device="cuda"
            ).requires_grad_()
            ids = (
                torch.rand(world * tokens, experts, device="cuda")
                .argsort(-1)[:, :topk]
                .contiguous()
            )
            if step == 1:
                ids = (
                    torch.rand(world * tokens, local, device="cuda")
                    .argsort(-1)[:, :topk]
                    .contiguous()
                    + local
                )
            ids[::7] = -1
            for source_rank in range(world):
                end = (source_rank + 1) * tokens
                missing = (source_rank + step) % 3 * 11
                ids[end - missing : end] = -1
            dy = torch.randn_like(all_x)
            own = slice(rank * tokens, (rank + 1) * tokens - (rank + step) % 3 * 11)
            x = all_x.detach()[own].clone().requires_grad_()
            scores = all_scores.detach()[own].clone().requires_grad_()
            expected = reference(all_x, ids, all_scores, *reference_weights)
            args = (x, ids[own].contiguous(), scores, *weights)
            actual = (
                checkpoint(
                    forward, *args, use_reentrant=False, preserve_rng_state=False
                )
                if checkpointed
                else forward(*args)
            )
            relative(actual, expected[own], tolerance)
            cases.append((actual, expected, dy, x, scores, all_x, all_scores, own))
        errors = []
        for actual, expected, dy, x, scores, all_x, all_scores, own in reversed(cases):
            with ExitStack() as transport_guards:
                if peer_backward and not compiled:
                    for owner, name in (
                        (dist, "all_to_all_single"),
                        (dist, "all_to_all"),
                        (funcol, "all_to_all_single"),
                    ):
                        transport_guards.enter_context(
                            patch.object(
                                owner,
                                name,
                                side_effect=AssertionError(
                                    "Peer backward used an all-to-all"
                                ),
                            )
                        )
                with (
                    (
                        patch.object(
                            torch.Tensor,
                            "cpu",
                            side_effect=AssertionError("Host tensor copy"),
                        )
                        if not compiled
                        else nullcontext()
                    ),
                    (
                        patch.object(
                            torch.Tensor,
                            "tolist",
                            side_effect=AssertionError("Host split list"),
                        )
                        if not compiled
                        else nullcontext()
                    ),
                    (
                        patch.object(
                            torch.Tensor,
                            "item",
                            side_effect=AssertionError("Host scalar read"),
                        )
                        if not compiled
                        else nullcontext()
                    ),
                    patch.object(owned_ops, "owned_backward", counted_owned),
                    patch.object(grouped, "_gemm", derivative_gemm),
                    patch.object(fp8, "expert_gemm_fp8", fp8_derivative_gemm),
                    patch.object(
                        recompute_ops, "grouped_recompute_swiglu", counted_recompute
                    ),
                ):
                    actual.backward(dy[own].contiguous())
            expected.backward(dy)
            errors.append(
                [
                    relative(x.grad, all_x.grad[own], tolerance),
                    relative(scores.grad, all_scores.grad[own], tolerance),
                ]
            )
        assert forward_calls == (4 if checkpointed else 2), forward_calls
        # Non-reentrant checkpointing may stop once saved inputs are available.
        allowed_native_calls = (
            (2, 4) if checkpointed and recompute else (4 if checkpointed else 2,)
        )
        assert native_forward_calls in allowed_native_calls, native_forward_calls
        assert recompute_calls == (2 if recompute else 0), recompute_calls
        gradient_errors = [
            relative(w.grad, r.grad[rank * local : (rank + 1) * local], tolerance)
            for w, r in zip(weights, reference_weights, strict=True)
        ]
        optimizer.step()
        ref_optimizer.step()
        weight_errors = [
            relative(w, r[rank * local : (rank + 1) * local], 0.01)
            for w, r in zip(weights, reference_weights, strict=True)
        ]
        print(
            json.dumps(
                dict(
                    rank=rank,
                    world_size=world,
                    master_dtype=master_dtype,
                    experts=experts,
                    hidden=hidden,
                    intermediate=inner,
                    topk=topk,
                    max_tokens_per_rank=tokens + 31,
                    actual_tokens=tokens - (rank + step) % 3 * 11,
                    compute_dtype=compute_dtype,
                    owned_blocks=owned_blocks,
                    native_backward=native_backward,
                    step=step,
                    outstanding_forwards=2,
                    reverse_backward=True,
                    all_to_all_guarded=peer_backward and not compiled,
                    fused_peer_return=bool(
                        peer_backward and runtime._peer_backward.fused_peer_return
                    ),
                    backward_expert_w1_recompute=recompute,
                    recompute_calls=recompute_calls,
                    fused_recompute_calls=fused_recompute_calls,
                    fullgraph=compiled,
                    checkpointed=checkpointed,
                    forward_calls=forward_calls,
                    native_forward_calls=native_forward_calls,
                    checkpoint_replayed_native_forward=native_forward_calls == 4,
                    ready_gather=bool(
                        peer_backward and runtime._peer_backward.ready_gather
                    ),
                    dispatch="direct_peer" if peer_backward else "fixed_capacity_nccl",
                    host_scalar_and_list_reads_guarded=not compiled,
                    unequal_rank_tokens=True,
                    input_score_errors=errors,
                    weight_gradient_errors=gradient_errors,
                    updated_weight_errors=weight_errors,
                )
            ),
            flush=True,
        )
    engine.close() if engine is not None else runtime.close()
    setattr(native_module, native_name, original_native_forward)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compute-dtype", choices=("bf16", "mxfp8", "blockwise"), default="bf16"
    )
    parser.add_argument(
        "--master-dtype", choices=("float32", "bf16"), default="float32"
    )
    parser.add_argument("--checkpointed", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--peer-backward", action="store_true")
    parser.add_argument("--owned-blocks", action="store_true")
    parser.add_argument("--native-backward", action="store_true")
    args = parser.parse_args()
    if args.native_backward and (
        not args.peer_backward
        or not args.recompute
        or args.compute_dtype not in ("bf16", "blockwise")
        or args.owned_blocks
    ):
        parser.error(
            "Native backward requires isolated recomputing BF16 or blockwise FP8 peer backward"
        )
    if args.owned_blocks and (
        not args.peer_backward or not args.recompute or args.compute_dtype != "bf16"
    ):
        parser.error("Owned blocks require recomputing BF16 peer backward")
    run(
        args.compute_dtype,
        args.checkpointed,
        args.recompute,
        args.compiled,
        args.peer_backward,
        args.owned_blocks,
        args.native_backward,
        args.master_dtype,
    )
