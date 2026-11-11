"""Which dtype do Axolotl's expert weights have when mono-ep (and its SM100 MXFP8 weight-form builder) reads them?
Runs the loss-curve harness's random-init Qwen3-MoE for a few steps on 4 GPUs under FSDP2 + expert parallelism with
bf16: true, for EP4 x dp_shard 1 and EP2 x dp_shard 2, full fine-tuning and LoRA (with and without lora_fp32_gradients),
and collects the dtype_probe_hooks lines.

    python pod_dtype_probe.py --work /scratch/dtype_probe --out result.json
"""

import argparse
import json
import os
import subprocess  # nosec B404
from pathlib import Path

from pod_mxfp8_losscurve import load_config_template, prepare_data, prepare_model

HERE = Path(__file__).resolve().parent
LORA = """adapter: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.0
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
"""
CASES = {
    "ep4_full": (4, ""),
    "ep2_full": (2, ""),
    "ep4_lora": (4, LORA),
    "ep2_lora": (2, LORA),
    "ep4_lora_fp32grad": (4, LORA + "lora_fp32_gradients: true\n"),
    "ep2_lora_fp32grad": (2, LORA + "lora_fp32_gradients: true\n"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default=",".join(CASES))
    args = parser.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    prepare_data(work)
    prepare_model(work, 1)
    template = load_config_template().replace("/work/", f"{work}/")
    template = template.replace("eval_steps: 50\n", "")
    start = template.index("test_datasets:")
    end = template.index("dataset_prepared_path:")
    template = template[:start] + template[end:]
    results = {}
    for name in args.cases.split(","):
        ep, extra = CASES[name]
        config = template.format(seed=1, max_steps=3, run=f"dtype_{name}")
        config = config.replace(
            "expert_parallel_size: 2", f"expert_parallel_size: {ep}"
        )
        config += extra
        if ep == 2:
            config += "dp_shard_size: 2\n"
        path = work / f"config_{name}.yaml"
        path.write_text(config)
        env = {
            **os.environ,
            "PYTHONPATH": f"{HERE / 'dtype_probe_hooks'}:{os.environ.get('PYTHONPATH', '')}",
            "DTYPE_PROBE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "WANDB_DISABLED": "true",
            "AXOLOTL_DO_NOT_TRACK": "1",
            "MONO_EP_SM100_BACKWARD_PRECISION": "mxfp8",
        }
        proc = subprocess.run(  # nosec B603 B607
            ["axolotl", "train", str(path)],
            env=env,
            cwd=str(work),
            capture_output=True,
            text=True,
            timeout=2400,
        )
        lines = (proc.stdout + proc.stderr).splitlines()
        (work / f"log_{name}.txt").write_text("\n".join(lines))
        probes = sorted(
            {
                line[line.index("DTYPE_PROBE") :]
                for line in lines
                if "DTYPE_PROBE" in line
            }
        )
        results[name] = dict(
            returncode=proc.returncode,
            probes=probes,
            tail=lines[-30:] if proc.returncode else [],
        )
        print(name, proc.returncode, *probes, sep="\n  ", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
