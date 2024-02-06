"""
CLI to run training on a model
"""
import logging
import queue
import threading
import time
import typing
from pathlib import Path
from typing import Tuple

import fire
import wandb
from datasets import Dataset, IterableDataset, concatenate_datasets
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from wandb.apis.public.history import HistoryScan

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    load_rl_datasets,
    print_axolotl_text_art,
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.prompt_strategies.sharegpt import register_chatml_template
from axolotl.train import train

LOG = logging.getLogger("axolotl.cli.train")


def get_samples(max_runs=200, num_steps=1, max_samples=None):
    run_count = 0
    api = wandb.Api(timeout=100)
    runs = api.runs(
        "cortex-t/multi-modality",
        filters={
            "$and": [
                {"config.type": "validator"},
                {"state": "running"},
            ]
        },
        per_page=20,
        order="-created_at",
    )
    page_size = 50

    all_datasets: typing.List[Dataset] = []
    for run_index, run in tqdm(enumerate(runs), total=len(runs)):
        if max_runs and run_count >= max_runs:
            break
        run_count += 1
        buffer: typing.List[typing.Dict[str, typing.Any]] = []
        run_id = run.id

        last_step: int = run.lastHistoryStep
        max_step = last_step + 1
        min_step = max(0, max_step - num_steps) if num_steps is not None else 0
        history_scan = HistoryScan(
            run.client, run, min_step, max_step, page_size=page_size
        )
        while True:
            try:
                sample = next(history_scan)
                for uid in range(256):
                    try:
                        prompt: typing.Optional[str] = sample[f"prompts.{uid}"]
                        response: typing.Optional[str] = sample[f"responses.{uid}"]
                        if isinstance(prompt, str) and isinstance(response, str):
                            prompt = prompt.strip()
                            response = response.strip()
                            if len(prompt) > 0 and len(response) > 0:
                                buffer.append(
                                    {
                                        "conversations": [
                                            {
                                                "from": "human",
                                                "value": prompt,
                                            },
                                            {
                                                "from": "gpt",
                                                "value": response,
                                            },
                                        ],
                                        "run_id": run_id,
                                        "step": sample["_step"],
                                        "uid": uid,
                                        "id": f"{run_id}-{sample['_step']}-{uid}",
                                    }
                                )
                            if max_samples and len(buffer) >= max_samples:
                                raise StopIteration
                    except KeyError:
                        pass
            except StopIteration:
                break
        ds = Dataset.from_list(buffer)
        all_datasets.append(ds)

    return concatenate_datasets(all_datasets)


class ExtendableIterableDataset(IterableDataset):
    def __init__(self, *args, ttl=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.ttl = ttl  # Time-to-live for blocking on the queue

    def add_data(self, data):
        """
        Method to add data to the queue.
        """
        self.data_queue.put(data)

    def stop(self):
        """
        Method to signal that no more data will be added and iteration should stop
        when the queue is empty.
        """
        self.stop_event.set()

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_event.is_set() and self.data_queue.empty():
            # If stop has been signaled and the queue is empty, end the iteration
            raise StopIteration

        try:
            # Attempt to get data from the queue with blocking up to the TTL
            data = self.data_queue.get(timeout=self.ttl)
        except queue.Empty:
            # If no data is available within the TTL, raise StopIteration
            raise StopIteration

        return data


def data_producer():
    seen_keys = set()
    while True:
        ds = get_samples()
        for sample in ds:
            id = sample["id"]
            if id in seen_keys:
                continue
            seen_keys.add(id)
            yield sample
        time.sleep(60.0)


def do_cli(config: Path = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config, **kwargs)
    parser = HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    return do_train(parsed_cfg, parsed_cli_args)


def do_train(cfg, cli_args) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print_axolotl_text_art()
    check_accelerate_default_config()
    check_user_token()
    if cfg.chat_template == "chatml" and cfg.default_system_message:
        LOG.info(
            f"ChatML set. Adding default system message: {cfg.default_system_message}"
        )
        register_chatml_template(cfg.default_system_message)
    else:
        register_chatml_template()

    ds = IterableDataset.from_generator(data_producer)
    # ds = ExtendableIterableDataset(ttl=3600)
    import axolotl.utils.data

    def patched_ds_fetcher(*args, **kwargs):
        return ds

    axolotl.utils.data.get_streaming_dataset = patched_ds_fetcher
    producer_thread = threading.Thread(target=data_producer, args=(ds))
    producer_thread.start()

    if cfg.rl:
        dataset_meta = load_rl_datasets(cfg=cfg, cli_args=cli_args)
    else:
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

    return train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)


if __name__ == "__main__":
    fire.Fire(do_cli)
