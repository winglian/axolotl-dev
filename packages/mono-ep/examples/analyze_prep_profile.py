# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Launch-level breakdown of the host-side preparation around the persistent kernels.

Reads a benchmark_ep8_trace.py --profile --profile-cpu gzip JSON (rank 0) and reports, per
variant and scope, every GPU operation before and after the native kernel with its duration and
the idle gap before it, the GPU-busy fraction of the preparation region, and the host time from
the start of the profiled scope to the native kernel's launch call.
"""

import argparse
import gzip
import json

LAUNCHES = (
    "cudaLaunchKernel",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cudaLaunchKernelExC",
    "cudaMemcpyAsync",
    "cudaMemsetAsync",
)


def analyze(entry):
    kernels, host = entry["kernels"], entry["host"]
    native = next(i for i, k in enumerate(kernels) if "mono_ep" in k["name"])
    launches = [h for h in host if h["name"].startswith(LAUNCHES)]
    scope = next(h for h in host if h["name"] == "mono_ep_scope")
    ops = []
    previous_end = None
    for k in kernels:
        gap = None if previous_end is None else k["start_us"] - previous_end
        ops.append(
            dict(
                name=k["name"][:80],
                duration_us=round(k["duration_us"], 1),
                gap_before_us=None if gap is None else round(gap, 1),
            )
        )
        previous_end = k["start_us"] + k["duration_us"]
    prep = kernels[:native]
    post = kernels[native + 1 :]
    span = kernels[native]["start_us"] - kernels[0]["start_us"] if prep else 0.0
    busy = sum(k["duration_us"] for k in prep)
    out = dict(
        gpu_ops_total=len(kernels),
        launch_calls_total=len(launches),
        prep_gpu_ops=len(prep),
        prep_gpu_busy_us=round(busy, 1),
        prep_gpu_span_us=round(span, 1),
        prep_gpu_busy_fraction=round(busy / span, 3) if span else None,
        post_gpu_ops=len(post),
        post_gpu_busy_us=round(sum(k["duration_us"] for k in post), 1),
        native_us=round(kernels[native]["duration_us"], 1),
        ops=ops,
    )
    if len(launches) == len(kernels):
        out["host_scope_start_to_native_launch_us"] = round(
            launches[native]["start_us"] - scope["start_us"], 1
        )
        out["host_launch_call_us"] = [
            round(h["duration_us"], 1) for h in launches[: native + 1]
        ]
    out["host_scope_us"] = round(scope["duration_us"], 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", default="")
    parser.add_argument("--ops", action="store_true")
    args = parser.parse_args()
    report = {}
    for path in args.paths:
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        for key, ranks in data["profile"].items():
            result = analyze(ranks[0])
            report[f"{data['routing']}:{key}"] = result
            print(
                f"{data['routing']:8s} {key:26s} prep ops {result['prep_gpu_ops']:2d} "
                f"busy {result['prep_gpu_busy_us']:7.1f} span {result['prep_gpu_span_us']:7.1f} "
                f"frac {result['prep_gpu_busy_fraction']} host->native "
                f"{result.get('host_scope_start_to_native_launch_us')} post ops "
                f"{result['post_gpu_ops']} post busy {result['post_gpu_busy_us']} "
                f"scope host {result['host_scope_us']} launches {result['launch_calls_total']}"
            )
            if args.ops:
                for op in result["ops"]:
                    print(
                        f"      gap {op['gap_before_us']}  dur {op['duration_us']}  {op['name']}"
                    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
