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
    print(
        "FORMS_JSON "
        + json.dumps(
            dict(
                shape=args.shape,
                local_experts=local,
                configs=results,
                bytes_gb=bytes_moved / 1e9,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
