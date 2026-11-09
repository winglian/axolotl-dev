# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Fetch the exact upstream build dependencies recorded with the vendor snapshot."""

import json
import subprocess
from pathlib import Path


def fetch():
    vendor = Path(__file__).resolve().parents[1] / "vendor"
    source = json.loads((vendor / "sources.json").read_text())["deepgemm"]
    for name, dependency in source["dependencies"].items():
        target = vendor / "deepgemm" / "third-party" / name
        if not target.exists():
            target.mkdir(parents=True)
            subprocess.run(["git", "init", str(target)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(target),
                    "fetch",
                    "--depth=1",
                    dependency["url"],
                    dependency["revision"],
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"],
                check=True,
            )
        actual = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(target), "status", "--porcelain"], text=True
        ).strip()
        if actual != dependency["revision"] or dirty:
            raise RuntimeError(f"{target} does not match the pinned clean dependency")


if __name__ == "__main__":
    fetch()
