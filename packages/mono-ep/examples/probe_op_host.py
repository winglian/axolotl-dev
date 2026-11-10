# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Host-side attribution of the SM90 persistent custom ops at world 1 (one strict H100).

Times the eager forward/backward op calls after a device synchronize (host time only) and
prints a cProfile breakdown of the calls, for the default and the cached/fused-prep variants.
"""

import cProfile
import io
import json
import os
import pstats
import statistics
import time

import torch
import torch.distributed as dist


def main():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ.setdefault("MONO_EP_SM90_FP8_TRANSPORT", "1")
    os.environ.setdefault("MONO_EP_SM90_FP8_COMBINE", "1")
    assert "H100" in torch.cuda.get_device_name()
    dist.init_process_group(
        "nccl", rank=0, world_size=1, device_id=torch.device("cuda", 0)
    )
    from mono_ep.persistent_backward import NativePersistentBackward
    from mono_ep.runtime import (
        PersistentMoE,
        persistent_backward,
        persistent_forward,
    )

    tokens, hidden, inner, topk, experts = 16384, 2048, 768, 8, 16
    torch.manual_seed(0)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk].contiguous()
    scores = torch.rand(tokens, topk, device="cuda")
    w1 = torch.randn(experts, 2 * inner, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
    dy = torch.randn_like(x)
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        topk,
        hidden,
        inner,
        compute_dtype="blockwise",
    )
    runtime._peer_backward = NativePersistentBackward(runtime)
    key = runtime._key

    def forward():
        persistent_forward(x, ids, scores, w1, w2, key)

    def backward():
        persistent_backward(dy, x, ids, scores, w1, w2, key)

    out = {}
    variants = {
        "base": {},
        "cache_prep": {
            "MONO_EP_SM90_FP8_WEIGHT_CACHE": "1",
            "MONO_EP_SM90_FP8_FUSED_PREP": "1",
        },
    }
    for name, env in variants.items():
        for k in (
            "MONO_EP_SM90_FP8_WEIGHT_CACHE",
            "MONO_EP_SM90_FP8_FUSED_PREP",
        ):
            os.environ.pop(k, None)
        os.environ.update(env)
        for label, fn in (("forward_hit", forward), ("backward_hit", backward)):
            for _ in range(5):
                fn()
            host = []
            for _ in range(30):
                torch.cuda.synchronize()
                t = time.perf_counter()
                fn()
                host.append((time.perf_counter() - t) * 1e6)
            torch.cuda.synchronize()
            profiler = cProfile.Profile()
            for _ in range(20):
                torch.cuda.synchronize()
                profiler.enable()
                fn()
                profiler.disable()
            torch.cuda.synchronize()
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(25)
            out[f"{name}/{label}"] = dict(host_us_median=statistics.median(host))
            print(f"==== {name}/{label} host median {statistics.median(host):.1f} us")
            print(stream.getvalue())
    print("OP_HOST_JSON " + json.dumps(out))
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
