# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Counts native persistent backward calls per precision (printed per rank at exit) and prints
Trainer.log dicts unformatted on rank 0."""

import atexit
import importlib.abc
import json
import os
import sys

COUNTS: dict = {}


def _wrap_backward(module):
    backward = module.NativePersistentBackward
    original = backward.backward

    def wrapped(self, *args, **kwargs):
        key = self.precision() if self.arch == 10 else "sm90"
        COUNTS[key] = COUNTS.get(key, 0) + 1
        return original(self, *args, **kwargs)

    backward.backward = wrapped


def _wrap_trainer(module):
    trainer = module.Trainer
    original = trainer.log

    def log(self, logs, *args, **kwargs):
        if os.environ.get("RANK", "0") == "0":
            record = dict(logs, _step=self.state.global_step)
            print(f"LOSSCURVE_RAW {json.dumps(record, default=str)}", flush=True)
        return original(self, logs, *args, **kwargs)

    trainer.log = log


PATCHES = {
    "mono_ep.persistent_backward": _wrap_backward,
    "transformers.trainer": _wrap_trainer,
}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            import importlib.util

            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        loader = spec.loader
        exec_module = loader.exec_module

        def _exec(module):
            exec_module(module)
            PATCHES[name](module)

        loader.exec_module = _exec
        return spec


if os.environ.get("LOSSCURVE_COUNT") == "1":
    sys.meta_path.insert(0, _Finder())

    @atexit.register
    def _report():
        print(
            f"MONO_EP_BACKWARD_CALLS rank={os.environ.get('RANK')} {json.dumps(COUNTS)}",
            flush=True,
        )
