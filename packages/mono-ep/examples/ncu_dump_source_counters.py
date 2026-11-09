# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Dump per-PC Nsight Compute source counters and scalar metrics to JSON."""

import glob
import json
import sys


def main(report_path, out_path):
    candidates = sorted(
        glob.glob("/usr/local/cuda*/nsight-compute*/extras/python")
        + glob.glob("/opt/nvidia/nsight-compute/*/extras/python")
    )
    sys.path[:0] = candidates
    import ncu_report

    context = ncu_report.load_report(report_path)
    action = context.range_by_idx(0).action_by_idx(0)
    scalars, per_pc = {}, {}
    for name in action.metric_names():
        metric = action.metric_by_name(name)
        try:
            instances = metric.num_instances()
        except Exception:
            instances = 0
        if instances and metric.has_correlation_ids():
            ids = metric.correlation_ids()
            values = {}
            for index in range(instances):
                try:
                    value = metric.as_double(index)
                except Exception:  # nosec B112
                    continue
                if value:
                    values[hex(ids.as_uint64(index))] = value
            per_pc[name] = values
            continue
        try:
            value = metric.as_double()
        except Exception:
            try:
                value = metric.as_string()
            except Exception:  # nosec B112
                continue
        scalars[name] = dict(value=value, unit=metric.unit())
    pcs = set()
    for values in per_pc.values():
        pcs.update(values)
    sass, source = {}, {}
    try:
        sass_map = action.sass_by_pc()
        sass = {hex(int(pc)): text for pc, text in sass_map.items()}
    except Exception as error:
        sass = {"error": repr(error)}
    for pc in pcs:
        try:
            info = action.source_info(int(pc, 16))
            if info is not None:
                source[pc] = [info.file_name(), info.line()]
        except Exception:
            pass
    with open(out_path, "w") as handle:
        json.dump(
            dict(
                kernel=action.name(),
                scalars=scalars,
                per_pc=per_pc,
                sass=sass,
                source=source,
            ),
            handle,
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
