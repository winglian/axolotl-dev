# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-GPU timing of the MXFP8 dW requant (``requant_k_blocks``) at a shape's EP8 pool sizes: the four pools
(x, dy E4M3 + scales; dz E4M3 + scales; h BF16) with balanced per-expert row counts, over a worst-case-sized pool
(as the backward allocates it) and over a pool trimmed to the routed rows. Splits the requant's time into useful work
and early-exit programs. Prints ``RQ_JSON <payload>``."""

import argparse
import json

import torch
from mok_vs_mono_ep import SHAPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--configs", default="1,4,1;2,4,1;4,4,1;1,8,1;2,8,1;4,8,1")
    args = parser.parse_args()
    from mono_ep.dw import requant_k_blocks

    shape = SHAPES[args.shape]
    tokens, hidden, inner, experts = (
        shape[k] for k in ("tokens", "hidden", "intermediate", "experts")
    )
    topk, local = 8, experts // args.world
    worst = args.world * tokens * topk + local * 128
    per_expert = tokens * topk // local
    counts = torch.full((local,), per_expert, dtype=torch.int32, device="cuda")
    padded = (counts + 127) // 128 * 128
    starts = torch.cumsum(padded, 0, dtype=torch.int32) - padded
    ends = starts + counts
    used = int(padded.sum())
    widths = {"x": hidden, "dy": hidden, "dz": 2 * inner, "h": inner}

    def pools(rows):
        out = {}
        for name, cols in widths.items():
            if name == "h":
                out[name] = (torch.randn(rows, cols, device="cuda").bfloat16(), None)
            else:
                q = (torch.randn(rows, cols, device="cuda") * 50).to(
                    torch.float8_e4m3fn
                )
                sw = torch.full(
                    (cols // 128, rows), 0x7F7F7F7F, dtype=torch.int32, device="cuda"
                )
                out[name] = (q, sw)
        return out

    def time(p):
        res = {}
        for name, (pool, sw) in p.items():
            for _ in range(3):
                requant_k_blocks(pool, starts, ends, sw)
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            ev[0].record()
            for _ in range(args.iters):
                requant_k_blocks(pool, starts, ends, sw)
            ev[1].record()
            ev[1].synchronize()
            res[name] = ev[0].elapsed_time(ev[1]) / args.iters
        res["total"] = sum(res.values())
        return res

    import os

    worst_ms = time(pools(worst))
    trimmed = pools(used)
    os.environ.pop("MONO_EP_SM100_REQUANT_CONFIG", None)
    reference = {
        n: requant_k_blocks(pool, starts, ends, sw) for n, (pool, sw) in trimmed.items()
    }
    configs = {}
    for config in args.configs.split(";"):
        os.environ["MONO_EP_SM100_REQUANT_CONFIG"] = config
        ms = time(trimmed)
        bitwise = all(
            torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
            and torch.equal(a[1], b[1])
            for n, (pool, sw) in trimmed.items()
            for a, b in [(requant_k_blocks(pool, starts, ends, sw), reference[n])]
        )
        configs[config] = dict(ms=ms, bitwise=bitwise)
    os.environ.pop("MONO_EP_SM100_REQUANT_CONFIG", None)
    trimmed_ms = time(trimmed)
    bytes_moved = sum(
        used * c * (2 if n == "h" else 1)
        + used * c
        + used // 128 * c * 4
        + (0 if n == "h" else c // 128 * used * 4)
        for n, c in widths.items()
    )
    print(
        "RQ_JSON "
        + json.dumps(
            dict(
                shape=args.shape,
                worst_rows=worst,
                used_rows=used,
                worst_pool_ms=worst_ms,
                trimmed_pool_ms=trimmed_ms,
                bytes_moved_gb=bytes_moved / 1e9,
                trimmed_gbps=bytes_moved / trimmed_ms["total"] / 1e6,
                configs=configs,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
