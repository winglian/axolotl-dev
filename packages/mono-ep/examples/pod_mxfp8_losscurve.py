"""Loss curves on the pod: modal_mxfp8_losscurve.py's harness (random-init Qwen3-MoE, EP2 under
FSDP2, 300 steps fineweb-edu) with the runs given on the command line.

    python pod_mxfp8_losscurve.py --work /scratch/losscurve --out result.json --runs bf16:1 mxfp8:1 mxfp8dw:1

Run names: bf16, mxfp8 (MXFP8 backward, BF16 dW), mxfp8dw (plus MONO_EP_SM100_DW_PRECISION=mxfp8).
CUDA_VISIBLE_DEVICES selects the two GPUs. The worktree's env.sh must be sourced first (PYTHONPATH
with the built DeepGEMM site and mono_ep src)."""

import argparse
import json
import os
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TRAIN_ROWS, EVAL_ROWS = 20000, 200
PRECISION_ENV = {
    "bf16": {"MONO_EP_SM100_BACKWARD_PRECISION": "bf16"},
    "mxfp8": {"MONO_EP_SM100_BACKWARD_PRECISION": "mxfp8"},
    "mxfp8dw": {
        "MONO_EP_SM100_BACKWARD_PRECISION": "mxfp8",
        "MONO_EP_SM100_DW_PRECISION": "mxfp8",
    },
}


def load_config_template():
    text = (HERE / "modal_mxfp8_losscurve.py").read_text()
    start = text.index('CONFIG = """\\\n') + len('CONFIG = """\\\n')
    end = text.index('"""', start)
    return text[start:end]


def prepare_data(work):
    from datasets import load_dataset

    data = work / "data"
    if (data / "eval.jsonl").exists():
        return
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


def prepare_model(work, seed):
    import torch
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

    path = work / f"tiny_s{seed}"
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


def parse(line):
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


def train(work, template, run, precision, seed, max_steps, tag):
    config = template.replace("/work/", f"{work}/")
    out_dir = work / f"out_{tag}_{run}"
    config_path = work / f"config_{tag}_{run}.yaml"
    config_path.write_text(
        config.format(seed=seed, max_steps=max_steps, run=f"{tag}_{run}")
    )
    run_env = {
        **os.environ,
        "PYTHONPATH": f"{HERE / 'mxfp8_losscurve_hooks'}:{os.environ.get('PYTHONPATH', '')}",
        "PYTHONUNBUFFERED": "1",
        "LOSSCURVE_COUNT": "1",
        "OMP_NUM_THREADS": "1",
        "WANDB_DISABLED": "true",
        "AXOLOTL_DO_NOT_TRACK": "1",
        "MAX_JOBS": "8",
        **PRECISION_ENV[precision],
    }
    if precision != "mxfp8dw":
        run_env.pop("MONO_EP_SM100_DW_PRECISION", None)
    train_log, eval_log, other, lines = [], [], [], []
    start = time.time()
    with subprocess.Popen(  # nosec B603 B607
        ["axolotl", "train", str(config_path)],
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(work),
    ) as process:
        for line in process.stdout:
            now = time.time() - start
            lines.append(line)
            record = parse(line)
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
        if "mono_ep" in line or "MONO_EP_BACKWARD_CALLS" in line
    ]
    print(
        f"run {run}: rc {returncode}, {len(train_log)} train records, final loss "
        f"{train_log[-1]['loss'] if train_log else None}, eval {eval_log[-1]['eval_loss'] if eval_log else None}, "
        f"{time.time() - start:.0f} s",
        flush=True,
    )
    (work / f"log_{tag}_{run}.txt").write_text("".join(lines))
    return dict(
        run=run,
        precision=precision,
        seed=seed,
        returncode=returncode,
        train=train_log,
        eval=eval_log,
        summary=other,
        seam_lines=keep[:200],
        out_dir=str(out_dir),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--runs", nargs="+", required=True, help="precision:seed")
    args = parser.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    template = load_config_template()
    prepare_data(work)
    results = []
    for spec in args.runs:
        precision, seed = spec.split(":")
        seed = int(seed)
        prepare_model(work, seed)
        results.append(
            train(
                work,
                template,
                f"{precision}_s{seed}",
                precision,
                seed,
                args.max_steps,
                args.tag,
            )
        )
        Path(args.out).write_text(json.dumps(dict(runs=results), indent=1))
        if results[-1]["returncode"]:
            print(
                "".join(
                    open(work / f"log_{args.tag}_{precision}_s{seed}.txt").readlines()[
                        -80:
                    ]
                )
            )
            break


if __name__ == "__main__":
    main()
