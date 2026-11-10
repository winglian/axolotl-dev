# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run the pre-fix fused SM90 forms kernel (FP8 tile stored, then ``tl.trans``-ed) against the
per-call quantizers, with launch-option variants, next to the current kernel.

``python probe_prefix_fused_forms.py --src sm90_weight_forms_prefix_src.py.txt [--out r.json]
[--dump DIR]``. The pre-fix source is loaded as ``mono_ep.<variant>`` so its relative
imports resolve; variants rewrite ``num_warps`` / ``enable_fp_fusion`` in the launch.
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

import torch
import triton

SHAPES = [(2, 1536, 2048), (3, 512, 384)]
DTYPES = [torch.float32, torch.bfloat16]
VARIANTS = {
    "prefix_w8": {},
    "prefix_w4": {"num_warps=8": "num_warps=4"},
    "prefix_w8_fpfusion": {"enable_fp_fusion=False": "enable_fp_fusion=True"},
    "prefix_w8_nostore_q2": {
        # drop the untransposed store so q is single-use (the w1t pattern)
        "tl.store(\n            Q2 + expert * HIDDEN * INNER + rows[:, None] * INNER + cols[None, :], q\n        )": "pass",
    },
}


def load_variant(name, src_text, replacements, tmp):
    text = src_text
    for old, new in replacements.items():
        assert old in text, (name, old)
        text = text.replace(old, new)
    path = Path(tmp) / f"{name}.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"mono_ep.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inputs(shape, dtype):
    torch.manual_seed(5)
    experts, twice_inner, hidden = shape
    inner = twice_inner // 2
    w1 = torch.randn(experts, twice_inner, hidden, device="cuda")
    w2 = torch.randn(experts, hidden, inner, device="cuda")
    w1, w2 = w1.to(dtype), w2.to(dtype)
    w1[0, :64] *= 64
    w1[0, inner + 64 : inner + 128] *= 0.01
    w1[-1, :128, :128] = 0
    w1[-1, inner : inner + 128, :128] = 0
    w2[0, :128, :128] *= 300
    return w1, w2


def build_ms(module, w1, w2, iters=20):
    for _ in range(3):
        module.quantize_all_forms(w1, w2)
    times = []
    for _ in range(iters):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        module.quantize_all_forms(w1, w2)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def check(module, reference_module):
    rows = []
    for shape in SHAPES:
        for dtype in DTYPES:
            w1, w2 = inputs(shape, dtype)
            forms = module.quantize_all_forms(w1, w2)
            row = dict(
                shape=list(shape),
                dtype=str(dtype).split(".")[-1],
                forms={},
                build_ms=build_ms(module, w1, w2),
            )
            for name, (values, scales) in forms.items():
                exp_values, exp_scales = reference_module.quantize_form(w1, w2, name)
                bad = values.view(torch.uint8) != exp_values.view(torch.uint8)
                row["forms"][name] = dict(
                    values_mismatch_frac=float(bad.float().mean()),
                    scales_equal=bool(torch.equal(scales, exp_scales)),
                )
            q2 = forms["w2"][0].view(torch.uint8)
            q2t = forms["w2t"][0].view(torch.uint8)
            own = q2t != q2.transpose(1, 2)
            row["w2t_vs_own_w2_transposed_mismatch_frac"] = float(own.float().mean())
            if own.any():
                # where inside a 128x128 block do the wrong bytes sit?
                idx = own.nonzero()
                row["w2t_bad_block_row_hist"] = torch.bincount(
                    idx[:, 1] % 128, minlength=128
                ).tolist()
                row["w2t_bad_block_col_hist"] = torch.bincount(
                    idx[:, 2] % 128, minlength=128
                ).tolist()
                row["w2t_first_bad"] = idx[:8].tolist()
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--dump", default="")
    parser.add_argument(
        "--variant", default="", help="run one variant (used by --dump)"
    )
    args = parser.parse_args()
    import mono_ep.sm90_weight_forms as current

    src_text = Path(args.src).read_text()
    tmp = tempfile.mkdtemp(prefix="prefix_forms_")
    report = dict(
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        triton=triton.__version__,
        variants={},
    )
    names = [args.variant] if args.variant else [*VARIANTS, "current"]
    for name in names:
        module = (
            current
            if name == "current"
            else load_variant(name, src_text, VARIANTS[name], tmp)
        )
        rows = check(module, current)
        report["variants"][name] = rows
        worst = max(
            r["forms"][f]["values_mismatch_frac"] for r in rows for f in r["forms"]
        )
        bad_forms = sorted(
            {
                f
                for r in rows
                for f in r["forms"]
                if r["forms"][f]["values_mismatch_frac"]
            }
        )
        builds = " ".join(
            f"{r['dtype']}{r['shape']}={r['build_ms']:.3f}ms" for r in rows
        )
        print(
            f"{name:<24} worst mismatch {worst:.4f} bad forms {bad_forms} build {builds}"
        )
    if args.dump and not args.variant:
        for name in [*VARIANTS][:1] + ["current"]:
            target = Path(args.dump) / name
            target.mkdir(parents=True, exist_ok=True)
            env = {
                **os.environ,
                "TRITON_KERNEL_DUMP": "1",
                "TRITON_DUMP_DIR": str(target / "dump"),
                "TRITON_CACHE_DIR": str(target / "cache"),
                "TRITON_ALWAYS_COMPILE": "1",
            }
            result = subprocess.run(  # nosec B603
                [sys.executable, __file__, "--src", args.src, "--variant", name],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            (target / "run.log").write_text(result.stdout + "\n" + result.stderr)
            shutil.rmtree(target / "cache", ignore_errors=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
