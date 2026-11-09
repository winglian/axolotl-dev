# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Aggregate per-PC NCU warp stall samples by source line and warp role."""

import argparse
import collections
import json
import re

KERNEL_FILE = "sm90_fp8_mono_ep_backward.cuh"
ROLES = (
    (0, 345, "setup"),
    (345, 809, "dispatch"),
    (809, 876, "tma_load_a"),
    (876, 911, "tma_load_b"),
    (911, 917, "scheduler"),
    (917, 1508, "math_epilogue"),
    (1508, 10**9, "combine"),
)
FILE_RE = re.compile(r'//## File "([^"]+)", line (\d+)')
INLINED_RE = re.compile(r'inlined at "([^"]+)", line (\d+)')
INST_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);")


def parse_nvdisasm(path):
    chains, opcodes, current = {}, {}, []
    function = None
    for line in open(path):
        if line.lstrip().startswith(".text.") and "mono_ep_backward" in line:
            function = line.strip()
        match = FILE_RE.search(line)
        if match:
            current = [(match.group(1), int(match.group(2)))]
            current += [(f, int(n)) for f, n in INLINED_RE.findall(line)]
            continue
        match = INST_RE.search(line)
        if match and current:
            offset = int(match.group(1), 16)
            chains[offset] = list(current)
            opcodes[offset] = match.group(2).strip()
    return chains, opcodes, function


def role_of(chain):
    for file, line in reversed(chain):
        if file.endswith(KERNEL_FILE):
            for lo, hi, name in ROLES:
                if lo <= line < hi:
                    return name, line
    return "unattributed", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("per_pc")
    parser.add_argument("nvdisasm")
    parser.add_argument("source_sass_csv")
    parser.add_argument("output")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    data = json.load(open(args.per_pc))
    import csv

    with open(args.source_sass_csv) as handle:
        reader = csv.reader(handle)
        next(reader)
        header = next(reader)
        data["sass"] = {
            hex(int(row[0], 16)): row[1].strip()
            for row in reader
            if row and row[0].startswith("0x")
        }
    del header
    chains, opcodes, _ = parse_nvdisasm(args.nvdisasm)
    per_pc = data["per_pc"]
    stall_names = sorted(
        n
        for n in per_pc
        if n.startswith("smsp__pcsamp_warps_issue_stalled_")
        and not n.endswith("_not_issued")
    )
    all_pcs = sorted(int(pc, 16) for pc in data["sass"] if pc.startswith("0x"))
    base = (
        all_pcs[0]
        if all_pcs
        else min(int(pc, 16) for values in per_pc.values() for pc in values)
    )
    mismatches = 0
    checked = 0
    for pc_hex, text in data["sass"].items():
        if not pc_hex.startswith("0x"):
            continue
        offset = int(pc_hex, 16) - base
        if offset in opcodes:
            checked += 1
            a = text.split()[0].lstrip("@!P0123456789T ").split(".")[0]
            b = opcodes[offset].split()
            b = [t for t in b if not t.startswith("@")]
            if not b or a != b[0].split(".")[0]:
                mismatches += 1
    rows = collections.defaultdict(lambda: collections.Counter())
    role_totals = collections.defaultdict(collections.Counter)
    line_info = {}
    for name in stall_names:
        reason = name.removeprefix("smsp__pcsamp_warps_issue_stalled_")
        for pc_hex, value in per_pc[name].items():
            offset = int(pc_hex, 16) - base
            chain = chains.get(offset, [])
            role, kernel_line = role_of(chain)
            inner = chain[0] if chain else ("?", 0)
            key = (inner[0].split("/")[-1], inner[1], role, kernel_line)
            rows[key][reason] += value
            role_totals[role][reason] += value
            line_info.setdefault(key, set()).add(
                opcodes.get(offset, data["sass"].get(pc_hex, "")).split(" ")[0]
            )
    total = sum(sum(c.values()) for c in role_totals.values())

    def summarize(counter):
        s = sum(counter.values())
        return dict(
            samples=s,
            share=s / total if total else 0,
            reasons={
                k: dict(samples=v, share_of_row=v / s)
                for k, v in counter.most_common()
                if v / s >= 0.01
            },
        )

    by_line = sorted(rows.items(), key=lambda kv: -sum(kv[1].values()))
    # Also aggregate by outer kernel line (what the .cuh statement is).
    by_kernel_line = collections.defaultdict(collections.Counter)
    for (_file, _line, role, kline), counter in rows.items():
        by_kernel_line[(role, kline)].update(counter)
    top_kernel = sorted(by_kernel_line.items(), key=lambda kv: -sum(kv[1].values()))
    result = dict(
        total_stall_samples=total,
        pc_base=hex(base),
        sass_alignment=dict(checked=checked, opcode_mismatches=mismatches),
        stall_reasons_total=dict(
            collections.Counter(
                {
                    r: sum(c[r] for c in role_totals.values())
                    for r in {r for c in role_totals.values() for r in c}
                }
            ).most_common()
        ),
        by_role={
            role: summarize(c)
            for role, c in sorted(
                role_totals.items(), key=lambda kv: -sum(kv[1].values())
            )
        },
        top_inner_source_lines=[
            dict(
                file=file,
                line=line,
                role=role,
                kernel_cuh_line=kline,
                opcodes=sorted(line_info[(file, line, role, kline)])[:8],
                **summarize(counter),
            )
            for (file, line, role, kline), counter in by_line[: args.top * 2]
        ],
        top_kernel_cuh_lines=[
            dict(role=role, kernel_cuh_line=kline, **summarize(counter))
            for (role, kline), counter in top_kernel[: args.top * 2]
        ],
    )
    json.dump(result, open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()
