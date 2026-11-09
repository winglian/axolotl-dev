# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Retained FP8 experts under Axolotl checkpoint and activation-offload contexts."""

import argparse
import contextlib
import json
import os

import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint
from mono_ep import MonoEP

from axolotl.core.trainers.mixins.activation_checkpointing import (
    get_lora_act_offloading_ctx_manager,
)
from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin
from axolotl.monkeypatch.selective_checkpointing import (
    build_sac_context_fn,
    sac_checkpoint_kwargs,
)
from axolotl.monkeypatch.selective_checkpointing_offload import (
    SacOffloadEngine,
    build_sac_offload_context_fn,
)
from axolotl.monkeypatch.trainer.trl_offload import patch_trl_offload_dense_layout
from axolotl.utils.dict import DictDefault


class RoutingCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.topk.default:
            self.calls += 1
        return func(*args, **(kwargs or {}))


class Layer(torch.nn.Module):
    def __init__(self, compile_local=False):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(4, 256, 256, device="cuda") / 16)
        self.w2 = torch.nn.Parameter(torch.randn(4, 256, 128, device="cuda") / 128**0.5)
        self.compute = MonoEP(
            dtype="fp8", retain_activations=True, compile_local=compile_local
        )

    def forward(self, x, logits):
        scores, ids = logits.softmax(-1).topk(2)
        return self.compute(
            x, ids.int(), scores, self.w1, self.w2, group=dist.group.WORLD
        )


def run(compile_local=False):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.manual_seed(659 + rank)
    template = Layer().state_dict()
    inputs = [
        (
            torch.randn(137, 256, device="cuda", dtype=torch.bfloat16),
            torch.randn(137, 8, device="cuda"),
            torch.randn(137, 256, device="cuda", dtype=torch.bfloat16) / 137,
        )
        for _ in range(2)
    ]
    ExpertParallelPlugin._register_checkpoint_saves(
        DictDefault(expert_parallel_save_dispatch=False)
    )
    patch_trl_offload_dense_layout()
    reference = []
    for mode in (
        "retained",
        "checkpoint",
        "sac",
        "sac_cpu",
        "all_cpu",
        "checkpoint_all_cpu",
    ):
        layer = Layer(compile_local=compile_local)
        layer.load_state_dict(template)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3, foreach=False)
        engine = SacOffloadEngine(min_offload_bytes=1) if mode == "sac_cpu" else None
        context_fn = build_sac_context_fn(save=[])
        if mode == "sac":
            context_fn = build_sac_context_fn(save=["mono_ep::grouped_fp8"])
        elif mode == "sac_cpu":
            context_fn = build_sac_offload_context_fn(
                save=["mono_ep::grouped_fp8"], engine=engine
            )
        offload = (
            get_lora_act_offloading_ctx_manager(
                layer, min_offload_size=1, warn_if_no_head=False
            )
            if "all_cpu" in mode
            else contextlib.nullcontext()
        )
        errors = []
        routing_calls = []
        for step, (source, routing, dy) in enumerate(inputs):
            optimizer.zero_grad(set_to_none=True)
            x, logits = (
                source.detach().requires_grad_(),
                routing.detach().requires_grad_(),
            )
            kwargs = sac_checkpoint_kwargs(context_fn) if context_fn is not None else {}
            counter = contextlib.nullcontext if compile_local else RoutingCounter
            forward_counter, backward_counter = counter(), counter()
            with offload:
                with forward_counter:
                    if mode in ("retained", "all_cpu"):
                        output = layer(x, logits)
                    else:
                        output = checkpoint(
                            layer, x, logits, use_reentrant=False, **kwargs
                        )
                with backward_counter:
                    output.backward(dy)
            if not compile_local:
                assert forward_counter.calls == 1, (mode, forward_counter.calls)
                assert backward_counter.calls == 0, (mode, backward_counter.calls)
                routing_calls.append(
                    dict(forward=forward_counter.calls, backward=backward_counter.calls)
                )
            actual = [
                output.detach().clone(),
                x.grad.clone(),
                logits.grad.clone(),
                *(p.grad.clone() for p in layer.parameters()),
            ]
            optimizer.step()
            actual.extend(p.detach().clone() for p in layer.parameters())
            torch.cuda.synchronize()
            if mode == "retained":
                reference.append(actual)
            else:
                step_errors = []
                for got, want in zip(actual, reference[step], strict=True):
                    value = (
                        (got.float() - want.float()).norm()
                        / want.float().norm().clamp_min(1e-8)
                    ).item()
                    assert torch.isfinite(got).all() and value < 1e-5, (
                        mode,
                        step,
                        len(step_errors),
                        value,
                    )
                    step_errors.append(value)
                errors.append(step_errors)
            dist.barrier()
        stats = vars(engine.stats) if engine is not None else None
        if stats is not None:
            assert stats["offloaded_bytes"] > 0 and stats["restored_tensors"] > 0, stats
        print(
            json.dumps(
                dict(
                    rank=rank,
                    mode=mode,
                    compile_local=compile_local,
                    relative_l2=errors,
                    offload_stats=stats,
                    routing_calls=routing_calls,
                )
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-local", action="store_true")
    run(compile_local=parser.parse_args().compile_local)
