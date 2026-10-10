# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Cached supervision metadata and configuration contracts for balanced sampling."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset, load_from_disk

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.dict import DictDefault
from axolotl.utils.samplers.utils import (
    LABEL_METADATA_COLUMNS,
    add_label_metadata,
    get_dataset_label_counts,
)
from axolotl.utils.schemas.config import AxolotlInputConfig


@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("column", ["labels", "shift_labels"])
def test_cached_counts_survive_disk_and_avoid_row_scan(
    tmp_path, monkeypatch, shift, column
):
    data = Dataset.from_dict({column: [[1, -100, 2], [-100, 3], [], [-100]]})
    expected = get_dataset_label_counts(data, shift_labels=shift)
    cached = data.map(add_label_metadata, batched=True)
    cached.save_to_disk(str(tmp_path / "data"))
    restored = load_from_disk(str(tmp_path / "data"))

    def no_scan(*args, **kwargs):
        raise AssertionError("Cached counts should not scan token rows")

    monkeypatch.setattr(Dataset, "iter", no_scan)
    actual = get_dataset_label_counts(restored, shift_labels=shift)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)


def test_both_label_representations_cache_independently():
    data = Dataset.from_dict(
        {"labels": [[1, 2, -100]], "shift_labels": [[2, -100, -100]]}
    )
    cached = data.map(add_label_metadata, batched=True)
    assert get_dataset_label_counts(cached)[0].tolist() == [1]
    raw = cached.remove_columns("shift_labels")
    counts, starts = get_dataset_label_counts(raw)
    assert counts.tolist() == [2]
    assert starts.tolist() == [1]


@pytest.mark.parametrize(
    "balance, split, expected",
    [(True, "train", True), (False, "train", False), (True, "test", False)],
)
def test_preprocessing_saves_metadata_only_for_balanced_training(
    monkeypatch, balance, split, expected
):
    from axolotl.utils.data import sft

    data = Dataset.from_dict({"input_ids": [[1, 2, 3]], "labels": [[-100, 2, 3]]})
    monkeypatch.setattr(sft, "datasets_with_name_generator", lambda configs: configs)
    monkeypatch.setattr(
        sft, "_load_and_process_single_dataset", lambda **kwargs: (data, None)
    )
    monkeypatch.setattr(sft, "merge_datasets", lambda datasets, cfg: datasets[0])
    monkeypatch.setattr(
        sft, "handle_long_seq_in_dataset", lambda dataset, *args: dataset
    )
    monkeypatch.setattr(sft, "generate_dataset_hash_from_config", lambda *args: "test")
    saved = []
    monkeypatch.setattr(
        sft,
        "save_preprocessed_dataset",
        lambda cfg, dataset, *args: saved.append(dataset),
    )
    cfg = DictDefault(balance_labels=balance, sequence_len=8, dataset_num_proc=None)
    result, _ = sft._load_raw_datasets(
        cfg, [{}], SimpleNamespace(name_or_path="test"), split, None, False
    )
    assert (
        bool(set(LABEL_METADATA_COLUMNS).intersection(result.column_names)) == expected
    )
    assert saved[0].column_names == result.column_names


@pytest.mark.parametrize(
    "option,value,packing",
    [
        ("curriculum_sampling", True, True),
        ("sample_packing_sequentially", True, True),
        ("reward_model", True, True),
        ("process_reward_model", True, True),
        ("diffusion_lm", {}, True),
        ("rl", "dpo", False),
        ("streaming", True, False),
        ("group_by_length", True, False),
    ],
)
def test_validation_names_conflicting_option(option, value, packing):
    config = dict(
        base_model="test",
        learning_rate=1e-5,
        datasets=[{"path": "test", "type": "alpaca"}],
        sample_packing=packing,
        balance_labels=True,
    )
    config[option] = value
    with pytest.raises(ValueError, match=option):
        AxolotlInputConfig(**config)


@pytest.mark.parametrize(
    "tp,cp,deepspeed",
    [(2, 1, False), (1, 2, False), (2, 2, False), (1, 2, True)],
)
def test_parallel_setup_excludes_non_data_ranks(monkeypatch, tp, cp, deepspeed):
    from accelerate.utils import ParallelismConfig

    from axolotl.monkeypatch.accelerate import tp as tp_patch
    from axolotl.utils import trainer as trainer_utils

    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in list(os.environ):
        if key.startswith("PARALLELISM_CONFIG_") or key.startswith("ACCELERATE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tp_patch, "patch_accelerate_prepare_tp", lambda: None)
    monkeypatch.setattr(trainer_utils, "get_world_size", lambda: 8)
    replicas = 8 // tp // cp
    cfg = DictDefault(
        tensor_parallel_size=tp,
        context_parallel_size=cp,
        dp_replicate_size=replicas if deepspeed else 1,
        dp_shard_size=1 if deepspeed else replicas,
        deepspeed="test.json" if deepspeed else None,
    )
    trainer_utils.setup_parallelism_envs(cfg)
    assert os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] == "true"
    parallelism = ParallelismConfig()
    assert parallelism.total_size == 8
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(world_size=8, gradient_accumulation_steps=4)
    trainer.accelerator = SimpleNamespace(parallelism_config=parallelism)
    assert trainer._data_parallel_size() == replicas
    assert trainer._batches_per_optimizer_step() == replicas * 4


def test_expert_parallel_ranks_receive_distinct_batches():
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(world_size=16, gradient_accumulation_steps=4)
    trainer.accelerator = SimpleNamespace(
        parallelism_config=SimpleNamespace(
            dp_replicate_size=1,
            dp_shard_size=2,
            ep_size=4,
            cp_size=2,
            tp_size=1,
        )
    )
    assert trainer._data_parallel_size() == 8
    assert trainer._batches_per_optimizer_step() == 32
