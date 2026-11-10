# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Loss curves of the SM100 MXFP8 persistent backward vs the BF16 one through the Axolotl seam.

The swiglu_fast harness ported to 2xB200: random-init Qwen3-MoE (hidden 1024, moe_intermediate
512, 8 layers, 32 experts top-4, ~739M parameters), EP2 under FSDP2, BF16 forward through
``expert_parallel_backend: mono_ep``, 300 steps on a fineweb-edu slice. One allocation runs
bf16/mxfp8 x seeds 1/2 back to back (``MONO_EP_SM100_BACKWARD_PRECISION`` per run) and the
local entrypoint reports per-step gaps (same seed, mxfp8 vs bf16) against the seed gap.

    modal run modal_mxfp8_losscurve.py --out ../benchmarks/mxfp8_losscurve_v1.json
"""

import json
import os
import shutil
import subprocess  # nosec B404
import time
from pathlib import Path

import modal

app = modal.App("axolotl-mono-ep-mxfp8-losscurve")
ROOT = Path(__file__).resolve().parents[1]
image = None
if modal.is_local():
    REPO = ROOT.parents[1]
    from modal_validate import base_image

    ignore = ["**/__pycache__/**", "**/*.pyc"]
    image = (
        base_image.add_local_dir(REPO / "src", "/ax/src", ignore=ignore)
        .add_local_dir(ROOT / "src", "/us/src", ignore=ignore)
        .add_local_dir(Path(__file__).parent / "mxfp8_losscurve_hooks", "/ax/hooks")
        .add_local_dir(
            ROOT / "vendor/deepgemm/deep_gemm/include/deep_gemm",
            "/probe/headers/deep_gemm",
        )
    )

TRAIN_ROWS, EVAL_ROWS = 20000, 200
RUNS = (("bf16", 1), ("mxfp8", 1), ("bf16", 2), ("mxfp8", 2))

CONFIG = """\
base_model: /work/tiny_s{seed}
plugins:
  - axolotl.integrations.expert_parallel.ExpertParallelPlugin
expert_parallel_size: 2
expert_parallel_backend: mono_ep
expert_parallel_mono_ep_precision: bf16
expert_parallel_fallback_on_unsupported: false

datasets:
  - path: /work/data/train.jsonl
    ds_type: json
    type: completion
test_datasets:
  - path: /work/data/eval.jsonl
    ds_type: json
    type: completion
    split: train
dataset_prepared_path: /work/prepared
output_dir: /work/out_{run}

sequence_len: 2048
sample_packing: true
eval_sample_packing: true
micro_batch_size: 4
gradient_accumulation_steps: 2
max_steps: {max_steps}
eval_steps: 50
save_strategy: "no"
logging_steps: 1

optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 3e-4
warmup_steps: 30
weight_decay: 0.0

bf16: true
tf32: false
flash_attention: true
seed: {seed}

fsdp_version: 2
fsdp_config:
  offload_params: false
  cpu_ram_efficient_loading: false
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen3MoeDecoderLayer
  state_dict_type: FULL_STATE_DICT
  reshard_after_forward: true
"""


def _prepare_data():
    from datasets import load_dataset

    data = Path("/work/data")
    data.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
    )
    rows = [
        r["text"] for r, _ in zip(stream, range(TRAIN_ROWS + EVAL_ROWS), strict=False)
    ]
    for name, part in (("train", rows[:TRAIN_ROWS]), ("eval", rows[TRAIN_ROWS:])):
        with open(data / f"{name}.jsonl", "w") as handle:
            for text in part:
                handle.write(json.dumps({"text": text}) + "\n")


def _prepare_model(seed):
    import torch
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

    path = Path(f"/work/tiny_s{seed}")
    if path.exists():
        return None
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=32,
        num_experts_per_tok=4,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        use_cache=False,
        hidden_act="silu",
        torch_dtype="bfloat16",
    )
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return sum(p.numel() for p in model.parameters())


def _parse(line):
    start = line.find("LOSSCURVE_RAW ")
    if start == -1 or "loss" not in line:
        return None
    try:
        record = json.loads(line[start + len("LOSSCURVE_RAW ") :])
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None
    out = {}
    for key, value in record.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def _train(run, precision, seed, max_steps):
    Path("/work/config.yaml").write_text(
        CONFIG.format(seed=seed, max_steps=max_steps, run=run)
    )
    run_env = {
        **os.environ,
        "PYTHONPATH": "/ax/hooks:/ax/src:/us/src",
        "PYTHONUNBUFFERED": "1",
        "LOSSCURVE_COUNT": "1",
        "OMP_NUM_THREADS": "1",
        "WANDB_DISABLED": "true",
        "AXOLOTL_DO_NOT_TRACK": "1",
        "MAX_JOBS": "8",
        "DG_JIT_CACHE_DIR": "/tmp/dg_jit_cache",  # nosec B108
        "MONO_EP_SM100_BACKWARD_PRECISION": precision,
    }
    train_log, eval_log, other, lines = [], [], [], []
    start = time.time()
    with subprocess.Popen(  # nosec B603 B607
        ["axolotl", "train", "/work/config.yaml"],
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/work",
    ) as process:
        for line in process.stdout:
            now = time.time() - start
            print(line, end="", flush=True)
            lines.append(line)
            record = _parse(line)
            if record is None:
                continue
            record["wall_s"] = now
            if "eval_loss" in record:
                eval_log.append(record)
            elif "loss" in record and "train_runtime" not in record:
                train_log.append(record)
            else:
                other.append(record)
        returncode = process.wait()
    keep = [
        line.strip()
        for line in lines
        if "mono_ep" in line
        or "MONO_EP_BACKWARD_CALLS" in line
        or "NCCL_MAX_CTAS" in line
    ]
    return dict(
        run=run,
        precision=precision,
        seed=seed,
        returncode=returncode,
        train=train_log,
        eval=eval_log,
        summary=other,
        seam_lines=keep[:200],
        tail="".join(lines[-300:]) if returncode else "".join(lines[-40:]),
    )


@app.function(image=image, gpu="B200:2", cpu=32, memory=262144, timeout=4 * 3600)
def train_all(max_steps):
    import deep_gemm
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != 2 or not all("B200" in n for n in names):
        raise RuntimeError(f"wrong devices {names}")
    installed = Path(deep_gemm.__file__).parent / "include/deep_gemm"
    shutil.copytree("/probe/headers/deep_gemm", installed, dirs_exist_ok=True)
    _prepare_data()
    params = {seed: _prepare_model(seed) for seed in (1, 2)}
    results = []
    for precision, seed in RUNS:
        run = f"{precision}_s{seed}"
        results.append(_train(run, precision, seed, max_steps))
        if results[-1]["returncode"]:
            break
    return dict(
        devices=names,
        params=params,
        runs=results,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )


def _gap(a, b):
    by_step = {r["_step"]: r["loss"] for r in a["train"]}
    pairs = [
        (r["_step"], abs(r["loss"] - by_step[r["_step"]]))
        for r in b["train"]
        if r["_step"] in by_step
    ]
    if not pairs:
        return None
    gaps = [g for _, g in pairs]
    n = len(gaps)
    third = max(n // 3, 1)
    return dict(
        steps=n,
        mean=sum(gaps) / n,
        max=max(gaps),
        last50=sum(gaps[-50:]) / len(gaps[-50:]),
        first_third=sum(gaps[:third]) / third,
        mid_third=sum(gaps[third : 2 * third]) / max(len(gaps[third : 2 * third]), 1),
        last_third=sum(gaps[2 * third :]) / max(len(gaps[2 * third :]), 1),
    )


@app.local_entrypoint()
def main(out: str, max_steps: int = 300):
    result = train_all.remote(max_steps)
    result["modal_app_id"] = app.app_id
    result["gpu"] = "B200:2"
    runs = {r["run"]: r for r in result["runs"]}
    stats = {}
    for a, b in (
        ("bf16_s1", "bf16_s2"),
        ("bf16_s1", "mxfp8_s1"),
        ("bf16_s2", "mxfp8_s2"),
        ("mxfp8_s1", "mxfp8_s2"),
    ):
        if a in runs and b in runs:
            stats[f"{a}|{b}"] = _gap(runs[a], runs[b])
    result["gap_stats"] = stats
    result["final_eval"] = {
        name: (r["eval"][-1]["eval_loss"] if r["eval"] else None)
        for name, r in runs.items()
    }
    result["final_train_loss"] = {
        name: (r["train"][-1]["loss"] if r["train"] else None)
        for name, r in runs.items()
    }
    result["backward_calls"] = {
        name: [line for line in r["seam_lines"] if "MONO_EP_BACKWARD_CALLS" in line]
        for name, r in runs.items()
    }
    Path(out).write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "devices",
                    "modal_app_id",
                    "gap_stats",
                    "final_eval",
                    "final_train_loss",
                    "backward_calls",
                )
            },
            indent=1,
        )
    )
    failed = [r["run"] for r in result["runs"] if r["returncode"]]
    if failed or len(result["runs"]) != len(RUNS):
        for r in result["runs"]:
            if r["returncode"]:
                print(r["tail"][-8000:])
        raise RuntimeError(f"failed runs: {failed}")
