"""LoRA loss curves through the Axolotl seam on the pod: the fused LoRA precisions against the
dispatch mode around SonicMoE LoRA, same random-init Qwen3-MoE, data and seed
(pod_mxfp8_losscurve.py's harness; EP2 under FSDP2 on two GPUs).

    python pod_lora_losscurve.py --work /scratch/losscurve --out result.json --runs fused_mxfp8 dispatch

Runs: fused_mxfp8 (auto -> mxfp8_lora), fused_bf16 (bf16 -> bf16_lora), dispatch
(expert_parallel_mono_ep_fused_lora: false, SonicMoE LoRA behind mono-ep's peer dispatch)."""

import argparse
import json
from pathlib import Path

import pod_mxfp8_losscurve as base

CONFIG = """\
base_model: /work/tiny_s{seed}
plugins:
  - axolotl.integrations.kernels.KernelsPlugin
  - axolotl.integrations.expert_parallel.ExpertParallelPlugin
use_sonicmoe: true
expert_parallel_size: 2
expert_parallel_backend: mono_ep
expert_parallel_mono_ep_precision: PRECISION
expert_parallel_mono_ep_fused_lora: FUSED
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

adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0
lora_target_modules:
  - q_proj
  - v_proj
lora_target_parameters:
  - experts.gate_up_proj
  - experts.down_proj

optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 1e-3
warmup_steps: 20
weight_decay: 0.0

bf16: true
tf32: false
flash_attention: true
seed: {seed}
activation_checkpointing: true

fsdp_version: 2
fsdp_config:
  offload_params: false
  cpu_ram_efficient_loading: false
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen3MoeDecoderLayer
  state_dict_type: FULL_STATE_DICT
  reshard_after_forward: true
"""
RUNS = {
    "fused_mxfp8": ("auto", "true"),
    "fused_bf16": ("bf16", "true"),
    "dispatch": ("auto", "false"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="lora")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    args = parser.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    base.prepare_data(work)
    base.prepare_model(work, args.seed)
    results = []
    for run in args.runs:
        precision, fused = RUNS[run]
        base.PRECISION_ENV[run] = {}
        template = CONFIG.replace("PRECISION", precision).replace("FUSED", fused)
        results.append(
            base.train(work, template, run, run, args.seed, args.max_steps, args.tag)
        )
        Path(args.out).write_text(json.dumps(dict(runs=results), indent=1))
        if results[-1]["returncode"]:
            print("".join(open(work / f"log_{args.tag}_{run}.txt").readlines()[-80:]))
            break


if __name__ == "__main__":
    main()
