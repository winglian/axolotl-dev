# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Attribute the EP8-minus-single-rank time from a ``modal_ep_trace_b200.py`` artifact.

Splits each traced native kernel along its own timeline (rank-local
globaltimer, max over CTAs): prologue and pre-pull NVLink barrier, task loop,
local grid sync, combine (ready wait + top-k reduce), dispatch cleanup tail;
adds cross-rank start skew, per-peer pull rates, MMA operand waits and routed
row imbalance, and the call-level parts (native, dW, layouts, other). Every
value is the median over traced iterations of the worst rank unless stated.
"""

import argparse
import json
import statistics
from pathlib import Path


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def kernel_view(result, direction):
    ranks = result["ranks"]
    iters = len(ranks[0]["traces"][direction])
    rows = []
    for i in range(iters):
        per_rank = [r["traces"][direction][i]["dist"] for r in ranks]
        parts = [r["traces"][direction][i]["parts"] for r in ranks]
        traces = [r["traces"][direction][i]["trace"] for r in ranks]
        starts = [d["start_ns"] for d in per_rank]
        ends = [d["end_ns"] for d in per_rank]

        def worst(key, field="max", per_rank=per_rank):
            return max(d[key][field] for d in per_rank if d.get(key))

        def typical(key, field="median", per_rank=per_rank):
            return med([d[key][field] for d in per_rank if d.get(key)])

        t1 = worst("pull_start")
        t2 = worst("epilogue_loop_end")
        t3 = worst("epilogue_grid_sync_exit")
        t4 = worst("combine_start")
        wall = max(d["kernel_ms"] for d in per_rank)
        t5 = worst("combine_end")
        peers_local, peers_remote = [], []
        first_remote, last_remote = [], []
        for rank_idx, d in enumerate(per_rank):
            for p in d["peers"]:
                if p["copy_us_per_row"] is None:
                    continue
                (peers_local if p["peer"] == rank_idx else peers_remote).append(
                    p["copy_us_per_row"]
                )
                if p["peer"] != rank_idx:
                    first_remote.append(p["first_arrival_ms"])
                    last_remote.append(p["last_arrival_ms"])
        rows.append(
            dict(
                kernel_wall_worst_rank=wall,
                global_span=(max(ends) - min(starts)) / 1e6,
                start_skew=(max(starts) - min(starts)) / 1e6,
                end_skew=(max(ends) - min(ends)) / 1e6,
                seg_prologue_and_prepull_barrier=t1,
                seg_task_loop=t2 - t1,
                seg_local_grid_sync=t3 - t2,
                seg_to_combine_start=t4 - t3,
                seg_combine=t5 - t4,
                seg_after_combine=wall - t5,
                prepull_barrier_wait=worst("prepull_barrier_wait_ms"),
                dispatch_prologue=worst("dispatch_prologue_end"),
                pull_end=worst("pull_end"),
                combine_first_ready_wait=worst("combine_first_ready_wait_ms"),
                combine_ready_wait_mean_warp=worst(
                    "combine_ready_wait_mean_warp_ms", "median"
                ),
                combine_span_median_cta=typical("combine_span_ms"),
                tma_a_operand_wait_median_cta=typical("tma_a_operand_wait_ms"),
                mma_full_barrier_wait_median_leader=typical("mma_full_barrier_wait_ms"),
                epilogue_sched_wait_median_cta=typical("epilogue_sched_wait_ms"),
                local_grid_sync_wait_median_cta=typical("local_grid_sync_wait_ms"),
                epilogue_loop_end_spread=worst("epilogue_loop_end")
                - min(d["epilogue_loop_end"]["min"] for d in per_rank),
                copy_us_per_row_local=med(peers_local),
                copy_us_per_row_remote=med(peers_remote),
                first_remote_arrival=min(first_remote) if first_remote else None,
                last_remote_arrival=max(last_remote) if last_remote else None,
                rows_per_rank=[d["rows_dispatched"] for d in per_rank],
                traced_call_whole=max(p["whole"] for p in parts),
                dispatch_busy_mean_warp_median_cta=med(
                    [
                        t["median_over_ctas_ms"]["dispatch_busy_mean_warp"]
                        for t in traces
                    ]
                ),
                dispatch_copy_mean_warp_median_cta=med(
                    [
                        t["median_over_ctas_ms"]["dispatch_copy_mean_warp"]
                        for t in traces
                    ]
                ),
                tma_a_l1_full_spin_median_cta=med(
                    [t["median_over_ctas_ms"].get("tma_a_l1_full_spin") for t in traces]
                ),
                sm_clock_ghz=med([t["sm_clock_ghz_median"] for t in traces]),
            )
        )
    keys = [k for k in rows[0] if k != "rows_per_rank"]
    summary = {k: med([r[k] for r in rows]) for k in keys}
    rows_per_rank = rows[0]["rows_per_rank"]
    mean_rows = sum(rows_per_rank) / len(rows_per_rank)
    summary["rows_max_over_mean"] = max(rows_per_rank) / mean_rows
    summary["rows_per_rank"] = rows_per_rank
    summary["iterations"] = len(rows)
    return summary


def call_view(result):
    out = {}
    for key, value in result["timing_max_rank"].items():
        parts = [r["local_timing"][key] for r in result["ranks"]]
        out[key] = dict(
            whole_max_rank_median=value["max_rank_median_ms"],
            whole_mean_rank_median=value["mean_rank_median_ms"],
            native_max_rank_median=value["native_max_rank_median_ms"],
            native_mean_rank_median=value.get("native_mean_rank_median_ms"),
            stream_native_max_rank_median=value.get("stream_native_max_rank_median_ms"),
            stream_native_mean_rank_median=value.get(
                "stream_native_mean_rank_median_ms"
            ),
            stream_samples=value.get("stream_samples"),
            samples=value["samples"],
            parts_worst_rank_median={
                part: max((p[part] or 0.0) for p in parts)
                for part in ("native", "dw", "layouts", "other")
            },
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--out")
    args = parser.parse_args()
    data = json.loads(Path(args.artifact).read_text())
    runs = {
        f"{r['world']}_{r['result']['routing']}"
        if "kernel_variants" in r["result"]
        else r["world"]: r["result"]
        for r in data["runs"]
        if r.get("result")
    }
    report = dict(
        artifact=args.artifact,
        modal_app_id=data.get("modal_app_id"),
        device=data.get("device"),
    )
    for world, result in runs.items():
        report[f"world{world}"] = dict(
            calls=call_view(result),
            kernels={d: kernel_view(result, d) for d in result["ranks"][0]["traces"]},
            routed_rows_per_rank=result["routed_rows_per_rank"],
        )
    if 8 in runs and 1 in runs:
        delta = {}
        for d in ("fwd", "bwd"):
            ep8, one = report["world8"]["kernels"][d], report["world1"]["kernels"][d]
            delta[d] = {
                k: (ep8[k] - one[k])
                for k in ep8
                if isinstance(ep8.get(k), float) and isinstance(one.get(k), float)
            }
        calls = {}
        for key in report["world8"]["calls"]:
            ep8, one = report["world8"]["calls"][key], report["world1"]["calls"][key]
            calls[key] = dict(
                whole=ep8["whole_max_rank_median"] - one["whole_max_rank_median"],
                native=ep8["native_max_rank_median"] - one["native_max_rank_median"],
                **{
                    f"part_{p}": ep8["parts_worst_rank_median"][p]
                    - one["parts_worst_rank_median"][p]
                    for p in ("native", "dw", "layouts", "other")
                },
            )
        report["ep8_minus_single"] = dict(kernels=delta, calls=calls)
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
