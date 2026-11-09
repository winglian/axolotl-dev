# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Summarize paired persistent/NCCL results without reconstructing missing ranks."""

import argparse
import json
from pathlib import Path

SCOPES = (
    ("forward", "Forward", 3.9),
    ("backward", "Backward", 8.2),
    ("recomputed_layer", "F+B", None),
    ("checkpointed_layer", "Checkpointed 2F+B", 15.9),
)


def load_result(path):
    result = json.loads(Path(path).read_text())
    if "returncode" in result:
        if result["returncode"] != 0:
            raise ValueError(f"Cannot report failed run: {path}")
        candidates = []
        for line in result["stdout"].splitlines():
            if line.startswith("{"):
                candidate = json.loads(line)
                if "median_ms" in candidate:
                    candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError(f"Expected one benchmark result: {path}")
        result = candidates[0]
    for scope, _, _ in SCOPES:
        for backend in ("native", "nccl"):
            if f"{scope}/{backend}" not in result["median_ms"]:
                raise ValueError("Expected the default-layout paired native/NCCL run")
    return result


def render(result, baseline=None):
    lines = [
        f"{result['world_size']} × {result['device']}; "
        f"T={result['tokens_per_gpu']}, H={result['hidden']}, "
        f"I={result['intermediate']}, E={result['experts']}, top-k={result['topk']}; "
        f"compute={result['compute_dtype']}, masters={result['master_dtype']}.",
        f"Warmup: {result.get('warmup_iterations', 'unreported')} iterations; "
        f"measured: {result.get('measured_iterations', 'unreported')}. "
        f"Router included: {result.get('router_in_timing', 'unreported')}. "
        f"Transport: {result.get('transport_precision', result.get('transport_dtype', 'unreported'))}.",
        "",
        "All times are milliseconds. Rank aggregates are computed per iteration, "
        "then their medians are taken. Missing historical means remain unavailable.",
        "",
        "| Scope | Native max-rank | Native mean-rank | NCCL max-rank | NCCL mean-rank | Prime reference |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, label, prime in SCOPES:
        values = []
        for backend in ("native", "nccl"):
            key = f"{scope}/{backend}"
            for field in ("median_ms", "mean_rank_median_ms"):
                value = result.get(field, {}).get(key)
                values.append("N/A" if value is None else f"{value:.3f}")
        lines.append(
            f"| {label} | {' | '.join(values)} | {prime if prime is not None else 'N/A'} |"
        )
    lines.extend(
        [
            "",
            "Prime's H200 FP8 figures are an external reference, not a precision-matched "
            "control. The NCCL columns come from the same run. "
            "[Prime Intellect published numbers, prime-rl](https://www.primeintellect.ai/blog/prime-mono-ep).",
        ]
    )
    if baseline is not None:
        geometry = (
            "device",
            "world_size",
            "tokens_per_gpu",
            "hidden",
            "intermediate",
            "experts",
            "topk",
            "compute_dtype",
            "master_dtype",
        )
        if any(result[key] != baseline[key] for key in geometry):
            raise ValueError("Baseline device, geometry or dtypes do not match")
        lines.extend(
            ["", "Relative to the retained baseline from a different allocation:"]
        )
        for scope, label, _ in SCOPES:
            key = f"{scope}/native"
            old, new = baseline["median_ms"][key], result["median_ms"][key]
            lines.append(
                f"- {label}: {old:.3f} → {new:.3f} ms; {100 * (new / old - 1):+.1f}% latency."
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    print(
        render(
            load_result(args.artifact),
            load_result(args.baseline) if args.baseline else None,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
