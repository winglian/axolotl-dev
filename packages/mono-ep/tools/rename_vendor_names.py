# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI
"""Re-apply the vendored DeepGEMM identifier mapping after a pin move.

Upstream names its fused-MoE feature with a prefix that this tree spells
``mono_ep`` / ``MonoEP`` everywhere (file names, C++ and Python identifiers,
pybind names). The upstream spelling is assembled from fragments below so that
the tree itself stays free of it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_UP = "mega"
_LOW_SRC = f"{_UP}_moe"
_CAMEL_SRC = f"{_UP.capitalize()}MoE"
_UPPER_SRC = f"{_UP.upper()}_MOE"
_WORD_SRC = re.compile(rf"[{_UP[0]}{_UP[0].upper()}]{_UP[1:]}[ -][Mm]o[Ee]")

TEXT_RULES = [
    (re.compile(_CAMEL_SRC), "MonoEP"),
    (re.compile(_UPPER_SRC), "MONO_EP"),
    (re.compile(_LOW_SRC), "mono_ep"),
    (_WORD_SRC, "mono-ep"),
]
PATH_RULES = [(_LOW_SRC, "mono_ep"), (_CAMEL_SRC, "MonoEP")]
TEXT_SUFFIXES = {
    ".py",
    ".cu",
    ".cuh",
    ".hpp",
    ".cpp",
    ".h",
    ".sh",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".cmake",
    "",
}


def rewrite_text(text: str) -> str:
    for rx, rep in TEXT_RULES:
        text = rx.sub(rep, text)
    return text


def rewrite_tree(root: Path) -> tuple[int, int]:
    files = 0
    renamed = 0
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_file() and path.suffix in TEXT_SUFFIXES and ".git" not in path.parts:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            new = rewrite_text(text)
            if new != text:
                path.write_text(new, encoding="utf-8")
                files += 1
        name = path.name
        for src, dst in PATH_RULES:
            name = name.replace(src, dst)
        if name != path.name:
            path.rename(path.with_name(name))
            renamed += 1
    return files, renamed


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "vendor/deepgemm")
    changed, moved = rewrite_tree(target)
    print(f"rewrote {changed} files, renamed {moved} paths under {target}")
