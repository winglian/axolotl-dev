# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-GPU timing of the fused SM100 MXFP8 backward weight forms (``fused_mx_backward_forms``) at a shape's EP8
local experts from FP32 masters, per ``MONO_EP_SM100_MXFP8_FORMS_CONFIG``, each checked bitwise (values and scale
words) against the original ``4,32,0`` launch. Prints ``FORMS_JSON <payload>``."""

import argparse
import json
import os

import torch
from mok_vs_mono_ep import SHAPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--configs", default="4,32,0;4,32,1;2,32,1;8,32,1;4,64,1;8,64,1;2,64,1"
    )
    parser.add_argument(
        "--bf16-input",
        action="store_true",
        help="also time BF16 expert weights and compare their forms to the FP32 masters' forms",
    )
    args = parser.parse_args()
    from mono_ep.mx_weight_forms import fused_mx_backward_forms

    shape = SHAPES[args.shape]
    hidden, inner, experts = (shape[k] for k in ("hidden", "intermediate", "experts"))
    local = experts // args.world
    torch.manual_seed(0)
    w1 = torch.randn(local, 2 * inner, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(local, hidden, inner, device="cuda") / inner**0.5

    def run():
        return fused_mx_backward_forms(w1, w2)

    os.environ["MONO_EP_SM100_MXFP8_FORMS_CONFIG"] = "4,32,0"
    reference = run()
    results = {}
    for config in args.configs.split(";"):
        os.environ["MONO_EP_SM100_MXFP8_FORMS_CONFIG"] = config
        out = run()
        bitwise = all(
            torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
            and torch.equal(a[1], b[1])
            for a, b in zip(out, reference, strict=True)
        )
        for _ in range(2):
            run()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        for _ in range(args.iters):
            run()
        ev[1].record()
        ev[1].synchronize()
        results[config] = dict(
            ms=ev[0].elapsed_time(ev[1]) / args.iters, bitwise=bitwise
        )
    bytes_moved = (w1.numel() + w2.numel()) * 4 + w1.numel() * 2 + w2.numel()
    bf16 = None
    if args.bf16_input:
        os.environ.pop("MONO_EP_SM100_MXFP8_FORMS_CONFIG", None)
        w1h, w2h = w1.bfloat16(), w2.bfloat16()
        fp32_forms = fused_mx_backward_forms(w1, w2)
        bf16_forms = fused_mx_backward_forms(w1h, w2h)
        for _ in range(2):
            fused_mx_backward_forms(w1h, w2h)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        for _ in range(args.iters):
            fused_mx_backward_forms(w1h, w2h)
        ev[1].record()
        ev[1].synchronize()

        def mismatch(a, b):
            return dict(
                values=(a[0].view(torch.uint8) != b[0].view(torch.uint8))
                .float()
                .mean()
                .item(),
                scale_words=(a[1] != b[1]).float().mean().item(),
            )

        def dequant_t(form, rows):
            # T form [E, C, R] with words [E][R / 128][C] (UTCCP column order), byte (r % 128) // 32
            q, words = form
            words = words.transpose(1, 2)
            e_, c_, r_ = q.shape
            col = torch.arange(c_, device=q.device)
            perm = (col // 128) * 128 + (col % 32) * 4 + (col % 128) // 32
            r = torch.arange(r_, device=q.device)
            w = words[:, r // 128][:, :, perm]  # [E, R, C]
            exponent = (w >> (((r % 128) // 32) * 8)[None, :, None]) & 255
            scale = torch.exp2(exponent.float() - 127)
            return q.float() * scale.transpose(1, 2)

        master = w2.transpose(1, 2)
        d32 = dequant_t(fp32_forms[1], w2.shape[1])
        d16 = dequant_t(bf16_forms[1], w2.shape[1])
        rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731
        bf16 = dict(
            ms=ev[0].elapsed_time(ev[1]) / args.iters,
            bytes_gb=((w1.numel() + w2.numel()) * 2 + w1.numel() * 2 + w2.numel())
            / 1e9,
            mismatch_vs_fp32_input={
                name: mismatch(a, b)
                for name, a, b in zip(
                    ("w1", "w2t", "w1t"), bf16_forms, fp32_forms, strict=True
                )
            },
            w2t_rel_error_vs_master=dict(
                fp32_input=rel(d32, master), bf16_input=rel(d16, master)
            ),
            w2t_bf16_vs_fp32_forms=dict(
                max_abs=(d16 - d32).abs().max().item(),
                mean_abs=(d16 - d32).abs().mean().item(),
                mean_abs_weight=master.abs().mean().item(),
            ),
        )
    print(
        "FORMS_JSON "
        + json.dumps(
            dict(
                shape=args.shape,
                local_experts=local,
                configs=results,
                bytes_gb=bytes_moved / 1e9,
                bf16_input=bf16,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
