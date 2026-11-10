# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI
"""TRL ``SFTTrainer`` on a Transformers Qwen3-MoE model with the mono-ep fused experts.

Expert parallel across the whole world, dense weights replicated with DDP. This is the
shortest wiring that matches the ``MonoEPRuntime`` API; the parts marked ``untested`` are
written against the Transformers and TRL interfaces but have not been run end to end.
Axolotl's ``expert_parallel_backend: mono_ep`` has been, and also handles FSDP2, checkpoint
gathering and selective activation checkpointing (see docs/trl-integration.md).

    torchrun --nproc_per_node 8 examples/trl_sft_mono_ep.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from mono_ep import MonoEPRuntime, check_nccl_env, nccl_env
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-30B-A3B"
RESERVED_SMS = 16
SEQ_LEN = 4096
MICRO_BATCH = 1

RUNTIME: dict[str, MonoEPRuntime] = {}


def mono_ep_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    # Transformers passes the flattened tokens [T, H], global expert ids [T, K] and the
    # router weights [T, K]; the runtime routes by global id and holds this rank's slice.
    return RUNTIME["rt"].forward(
        hidden_states.to(torch.bfloat16),
        top_k_index,
        top_k_weights.float(),
        self.gate_up_proj,
        self.down_proj,
    )


def main() -> None:
    os.environ.update(nccl_env(reserved_sms=RESERVED_SMS))  # before init_process_group
    dist.init_process_group("nccl")
    check_nccl_env(reserved_sms=RESERVED_SMS)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    ALL_EXPERTS_FUNCTIONS.register("mono_ep", mono_ep_experts_forward)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, experts_implementation="mono_ep"
    ).cuda()
    cfg = model.config
    num_experts, hidden, intermediate, top_k = (
        cfg.num_experts,
        cfg.hidden_size,
        cfg.moe_intermediate_size,
        cfg.num_experts_per_tok,
    )
    assert num_experts % world == 0
    local = num_experts // world
    e0 = rank * local

    # Keep this rank's expert slice as plain parameters.
    expert_params = []
    for layer in model.model.layers:
        experts = layer.mlp.experts
        experts.gate_up_proj = torch.nn.Parameter(
            experts.gate_up_proj.data[e0 : e0 + local].clone()
        )
        experts.down_proj = torch.nn.Parameter(
            experts.down_proj.data[e0 : e0 + local].clone()
        )
        expert_params += [experts.gate_up_proj, experts.down_proj]

    # DDP averages dense gradients over ranks; expert gradients are already the sum over
    # every rank's tokens, so keep them out of DDP and scale them to match. (untested)
    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
        model,
        [n for n, p in model.named_parameters() if any(p is q for q in expert_params)],
    )
    for p in expert_params:
        p.register_hook(lambda g: g / world)

    RUNTIME["rt"] = MonoEPRuntime(
        group=dist.group.WORLD,
        num_experts=num_experts,
        max_tokens_per_rank=SEQ_LEN * MICRO_BATCH,
        top_k=top_k,
        hidden=hidden,
        intermediate=intermediate,
        precision="auto",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=load_dataset("mhenrichsen/alpaca_2k_test", split="train"),
        args=SFTConfig(
            output_dir="./outputs/qwen3-30b-a3b-trl-mono-ep",
            max_length=SEQ_LEN,
            per_device_train_batch_size=MICRO_BATCH,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            bf16=True,
            gradient_checkpointing=True,
            # Reentrant recompute cannot replay the fused op.
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
            save_strategy="no",
            report_to="none",
        ),
    )
    trainer.train()
    RUNTIME["rt"].destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
