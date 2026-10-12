# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Single-GPU training through the cached-metadata balanced dataloader."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    DataCollatorForSeq2Seq,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    TrainingArguments,
)

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.core.training_args_base import AxolotlTrainingMixins
from axolotl.utils.samplers import LabelBalancedRandomSampler
from axolotl.utils.samplers.utils import add_label_metadata

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA"),
]


@dataclass
class SmokeTrainingArguments(AxolotlTrainingMixins, TrainingArguments):
    pass


def test_label_balanced_training(tmp_path, monkeypatch):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    rows = []
    for i in range(64):
        length = 16 + i % 4 * 8
        count = (1, 2, 4, 12)[i % 4]
        ids = [2 + i % 29] * length
        rows.append(
            dict(
                input_ids=ids,
                attention_mask=[1] * length,
                labels=[-100] * (length - count) + ids[-count:],
            )
        )
    dataset = Dataset.from_list(rows).map(add_label_metadata, batched=True)

    def no_scan(*args, **kwargs):
        raise AssertionError("Sampler must use cached label counts")

    monkeypatch.setattr(Dataset, "iter", no_scan)
    samplers = []
    get_sampler = AxolotlTrainer._get_train_sampler

    def capture_sampler(self, *args, **kwargs):
        sampler = get_sampler(self, *args, **kwargs)
        assert isinstance(sampler, LabelBalancedRandomSampler)
        samplers.append(sampler)
        return sampler

    monkeypatch.setattr(AxolotlTrainer, "_get_train_sampler", capture_sampler)
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config)
    initial = model.lm_head.weight.detach().clone()
    seen = []

    def inspect_batch(module, args, kwargs):
        assert kwargs["input_ids"].is_cuda
        assert not any(key.startswith("_axolotl_") for key in kwargs)
        seen.append(int((kwargs["labels"][..., 1:] != -100).sum()))

    hook = model.register_forward_pre_hook(inspect_batch, with_kwargs=True)
    trainer = AxolotlTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        args=SmokeTrainingArguments(
            output_dir=str(tmp_path),
            max_steps=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            balance_labels=True,
            label_balance_window_optim_steps=2,
            data_seed=7,
            learning_rate=1e-3,
            optim="adamw_torch",
            save_strategy="no",
            logging_steps=1,
            report_to="none",
            disable_tqdm=True,
        ),
    )
    assert trainer.args.world_size == 1
    assert trainer.args.n_gpu == 1
    try:
        result = trainer.train()
    finally:
        hook.remove()
    assert result.global_step == 2
    assert np.isfinite(result.training_loss) and result.training_loss > 0
    assert len(seen) == 4 and all(count > 0 for count in seen)
    assert not torch.equal(initial, model.lm_head.weight.detach().cpu())
    assert len(samplers) == 1
    sampler = samplers[0]
    assert sampler.batches_per_optimizer_step == 2
    np.testing.assert_array_equal(
        sampler.label_counts, [sum(x != -100 for x in r["labels"][1:]) for r in rows]
    )
    metrics = sampler.label_metrics
    assert (
        metrics["after"]["total_label_count"] == metrics["before"]["total_label_count"]
    )
    assert (
        metrics["after"]["std_global_update_labels"]
        < metrics["before"]["std_global_update_labels"]
    )
