# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""EP2 x FSDP2 parity of the mono_ep dispatch mode against the torch backend with the
same local experts kernel, plus a per-layer forward+backward timing at a moderate
geometry. Run under ``torch.distributed.run --nproc-per-node=2`` with ``--worker``."""

import json
import os
import sys
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

TOKENS = 64
STEPS = 2
MICRO_BATCHES = 2
BENCH = {
    "hidden": 2048,
    "intermediate": 1024,
    "experts": 64,
    "top_k": 8,
    "tokens": 4096,
}


def _cfg(mode, adapter):
    return SimpleNamespace(
        expert_parallel_backend="mono_ep",
        expert_parallel_mono_ep_mode=mode,
        expert_parallel_mono_ep_precision="auto",
        expert_parallel_fallback_on_unsupported=False,
        expert_parallel_save_dispatch=False,
        adapter=adapter,
        load_in_4bit=False,
        load_in_8bit=False,
        gptq=False,
        model_quantization_config=None,
        sequence_len=TOKENS,
        micro_batch_size=1,
        eval_batch_size=None,
        context_parallel_size=None,
    )


def _inputs(routing, step, micro):
    generator = torch.Generator(device="cuda").manual_seed(
        1000 + dist.get_rank() + 10 * step + 100 * micro
    )
    if routing == "balanced":
        return torch.randint(0, 256, (1, TOKENS), device="cuda", generator=generator)
    # identical tokens give identical router inputs: every slot lands on the same two
    # experts, so one rank receives everything and the other nothing
    return torch.full((1, TOKENS), 7 + dist.get_rank(), device="cuda")


class RoutingStats:
    """Fraction of top-k slots each rank's experts receive, per call."""

    def __init__(self, world):
        self.world = world
        self.counts = torch.zeros(world, device="cuda", dtype=torch.float64)

    def install(self, experts_fn):
        original = experts_fn._ep_forward
        stats = self

        def recording(self_, hidden_states, top_k_index, top_k_weights, **kwargs):
            local = getattr(self_, "num_local_experts", self_.num_experts)
            dest = torch.div(top_k_index, local, rounding_mode="floor")
            stats.counts += torch.bincount(dest.flatten(), minlength=stats.world).to(
                stats.counts.dtype
            )
            return original(self_, hidden_states, top_k_index, top_k_weights, **kwargs)

        experts_fn._ep_forward = recording
        return lambda: setattr(experts_fn, "_ep_forward", original)

    def fractions(self):
        return (self.counts / self.counts.sum().clamp_min(1)).tolist()


def sonicmoe_available():
    """Register axolotl's LoRA-aware SonicMoE forward (the ``use_sonicmoe`` path) when the
    hub kernel can load here."""
    try:
        from transformers.integrations.sonicmoe import is_sonicmoe_loadable

        from axolotl.integrations.kernels.libs.sonicmoe.experts import (
            register_sonicmoe_experts,
        )

        if not is_sonicmoe_loadable():
            is_sonicmoe_loadable(raise_error=True)
        register_sonicmoe_experts()
        return True
    except Exception as error:  # noqa: BLE001
        if dist.get_rank() == 0:
            print(f"sonicmoe unavailable: {error!r}", flush=True)
        return False


def training_checks():
    from mono_ep import DEFAULT_RESERVED_SMS, nccl_env
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from axolotl.integrations.expert_parallel import experts_fn, shard, torch_dispatch
    from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin
    from axolotl.integrations.expert_parallel.shard import shard_expert_weights
    from axolotl.monkeypatch import selective_checkpointing as sac

    # the fused mode's NCCL settings, so the dispatch mode is measured in the same
    # environment a mixed deployment would export
    for key, value in nccl_env(reserved_sms=DEFAULT_RESERVED_SMS).items():
        os.environ.setdefault(key, value)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    world = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world // 2, 2), mesh_dim_names=("dp_shard", "ep"))
    dense_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("world",))
    ep_group = mesh["ep"].get_group()
    torch_dispatch.set_ep_group(ep_group)
    experts_fn.register_all()
    ExpertParallelPlugin._resolve_ep_group = staticmethod(lambda cfg: ep_group)
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    plugin = ExpertParallelPlugin()
    local_kernels = ["grouped_mm"]
    try:
        from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
            register_scattermoe_experts,
        )

        register_scattermoe_experts()
        local_kernels.append("scattermoe")
    except Exception as error:  # noqa: BLE001
        if dist.get_rank() == 0:
            print(f"scattermoe unavailable: {error!r}", flush=True)
    if sonicmoe_available():
        local_kernels.append("sonicmoe")

    def layers_of(model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        return base.model.layers

    def run(backend, local, routing, lora, starting_weights=None):
        torch.manual_seed(123)
        config = Qwen3MoeConfig(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=512,
            moe_intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=8,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            use_cache=False,
            hidden_act="silu",
        )
        config._experts_implementation = "expert_parallel"
        config._attn_implementation = "sdpa"
        model = Qwen3MoeForCausalLM(config).cuda().train()
        sac.clear_registered_saves()
        experts_fn.set_local_implementation(local)
        # auto must settle on dispatch for LoRA by itself; the full fine-tune would pass
        # the fused checks on this geometry, so it forces the mode
        cfg = _cfg("auto" if lora else "dispatch", "lora" if lora else None)
        if backend == "mono_ep":
            assert plugin._resolve_mono_ep_model(cfg, model) == "mono_ep"
            assert cfg.expert_parallel_mono_ep_mode == "dispatch"
        assert shard_expert_weights(model, ep_group) == 2
        if lora:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    target_parameters=["experts.gate_up_proj", "experts.down_proj"],
                ),
            )
            # lora_A and lora_B of both expert parameters in each of the two layers
            assert shard.shard_expert_lora(model, 2) == 8
        experts_fn.set_backend(backend)
        ExpertParallelPlugin._register_checkpoint_saves(cfg, model)
        if backend == "mono_ep":
            plugin._build_mono_ep_dispatch_runtime(cfg)
            runtime = experts_fn.get_mono_ep_runtime()
            assert runtime.receive_rows == 2 * TOKENS, runtime.receive_rows
        kwargs = {"mp_policy": policy, "reshard_after_forward": True}
        ExpertParallelPlugin.fully_shard_experts(model, mesh["dp_shard"], kwargs)
        for layer in layers_of(model):
            fully_shard(layer, mesh=dense_mesh, **kwargs)
        fully_shard(model, mesh=dense_mesh, **kwargs)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3 if lora else 1e-4)
        stats = RoutingStats(2)
        restore = stats.install(experts_fn)
        history, starts = [], []
        for step in range(STEPS):
            if starting_weights is not None:
                # identical pre-step weights keep near-tie top-k routing from diverging
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        parameter.to_local().copy_(starting_weights[step][name])
            starts.append(
                {n: p.to_local().detach().clone() for n, p in model.named_parameters()}
            )
            optimizer.zero_grad(set_to_none=True)
            for micro in range(MICRO_BATCHES):
                tokens = _inputs(routing, step, micro)
                loss = model(input_ids=tokens, labels=tokens).loss / MICRO_BATCHES
                assert torch.isfinite(loss)
                loss.backward()
            grads = {}
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                assert parameter.grad is not None, name
                value = parameter.grad.to_local().detach()
                assert torch.isfinite(value).all(), name
                grads[name] = value.clone()
            optimizer.step()
            history.append(grads)
        weights = {
            n: p.to_local().detach().clone() for n, p in model.named_parameters()
        }
        restore()
        routing_fractions = stats.fractions()
        plugin.post_train_unload(cfg)
        del optimizer, model
        torch.cuda.synchronize()
        return history, weights, starts, routing_fractions

    results = []
    for local in local_kernels:
        for routing in ("balanced", "skewed"):
            for lora in (False, True):
                reference, ref_weights, starting, ref_routing = run(
                    "torch", local, routing, lora
                )
                history, weights, _, routing_fractions = run(
                    "mono_ep", local, routing, lora, starting
                )
                assert all(
                    abs(a - b) < 1e-6
                    for a, b in zip(routing_fractions, ref_routing, strict=True)
                ), (routing_fractions, ref_routing)
                worst, worst_name = 0.0, ""
                for grads, expected in zip(history, reference, strict=True):
                    for name, grad in grads.items():
                        rel = (grad.float() - expected[name].float()).norm() / expected[
                            name
                        ].float().norm().clamp_min(1e-8)
                        if rel.item() > worst:
                            worst, worst_name = rel.item(), name
                for name, weight in weights.items():
                    torch.testing.assert_close(
                        weight, ref_weights[name], atol=2e-4, rtol=0.02, msg=name
                    )
                worst_error = torch.tensor(worst, device="cuda")
                dist.all_reduce(worst_error, op=dist.ReduceOp.MAX)
                assert worst_error.item() < 0.03, (
                    dist.get_rank(),
                    local,
                    routing,
                    lora,
                    worst_name,
                )
                results.append(
                    {
                        "local_kernel": local,
                        "routing": routing,
                        "lora": lora,
                        "mono_ep_mode": "dispatch",
                        "slots_per_ep_rank": routing_fractions,
                        "max_gradient_relative_l2": worst_error.item(),
                        "worst_parameter_rank0": worst_name,
                        "optimizer_steps": STEPS,
                        "world_size": world,
                        "expert_parallel_size": 2,
                    }
                )
                if dist.get_rank() == 0:
                    print(json.dumps(results[-1]), flush=True)
    sac.clear_registered_saves()
    timing = layer_timing(ep_group, local_kernels)
    if dist.get_rank() == 0:
        print("RESULT " + json.dumps({"parity": results, "timing": timing}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def layer_timing(ep_group, local_kernels):
    """Forward+backward of one MoE block per backend and local kernel at a moderate
    geometry."""
    from mono_ep import MonoEPDispatchRuntime
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    from axolotl.integrations.expert_parallel import experts_fn
    from axolotl.integrations.expert_parallel.shard import shard_expert_weights

    config = Qwen3MoeConfig(
        hidden_size=BENCH["hidden"],
        moe_intermediate_size=BENCH["intermediate"],
        num_experts=BENCH["experts"],
        num_experts_per_tok=BENCH["top_k"],
        norm_topk_prob=True,
        hidden_act="silu",
    )
    config._experts_implementation = "expert_parallel"
    torch.manual_seed(7)
    block = Qwen3MoeSparseMoeBlock(config).cuda().to(torch.bfloat16)
    with torch.no_grad():
        block.gate.weight.normal_(std=0.02)
    assert shard_expert_weights(block, ep_group) == 1
    generator = torch.Generator(device="cuda").manual_seed(11 + dist.get_rank())
    x = torch.randn(
        1,
        BENCH["tokens"],
        BENCH["hidden"],
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    grad = torch.randn_like(x)
    runtime = MonoEPDispatchRuntime(
        group=ep_group,
        max_tokens_per_rank=BENCH["tokens"],
        top_k=BENCH["top_k"],
        hidden=BENCH["hidden"],
    )
    experts_fn.set_mono_ep_runtime(runtime)
    experts_fn.set_mono_ep_mode("dispatch")

    def rel(a, b):
        return ((a - b).norm() / b.norm().clamp_min(1e-8)).item()

    iterations = 20
    per_kernel = {}
    for local in local_kernels:
        experts_fn.set_local_implementation(local)
        outputs = {}
        timings = {}
        for backend in ("torch", "mono_ep"):
            experts_fn.set_backend(backend)
            inp = x.detach().clone().requires_grad_(True)

            def step(inp=inp):
                inp.grad = None
                block.zero_grad(set_to_none=True)
                block(inp).backward(grad)

            for _ in range(5):
                step()
            torch.cuda.synchronize()
            dist.barrier(group=ep_group)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                step()
            end.record()
            torch.cuda.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end) / iterations, device="cuda")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=ep_group)
            timings[backend] = elapsed.item()
            step()
            outputs[backend] = (
                block(inp).detach().float(),
                inp.grad.detach().float(),
                block.experts.gate_up_proj.grad.detach().float().clone(),
            )
        per_kernel[local] = {
            "forward_backward_ms": {
                "torch_nccl": timings["torch"],
                "mono_ep_dispatch": timings["mono_ep"],
            },
            "relative_l2_vs_torch": {
                "y": rel(outputs["mono_ep"][0], outputs["torch"][0]),
                "dx": rel(outputs["mono_ep"][1], outputs["torch"][1]),
                "dgate_up": rel(outputs["mono_ep"][2], outputs["torch"][2]),
            },
        }
        if dist.get_rank() == 0:
            print(json.dumps({"timing": local, **per_kernel[local]}), flush=True)
    experts_fn.set_mono_ep_runtime(None)
    runtime.destroy()
    return {
        "geometry": {**BENCH, "dtype": "bf16"},
        "iterations": iterations,
        "local_kernels": per_kernel,
    }


if __name__ == "__main__":
    if "--worker" in sys.argv:
        started = time.time()
        training_checks()
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"worker finished in {time.time() - started:.1f}s", flush=True)
    else:
        raise SystemExit("run under torch.distributed.run with --worker")
