# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Host-side cost of the SM90 weight/token preparation calls on one GPU (no profiler).

Each measured call runs after a device synchronize, so its host time is the Python + launch
cost alone; GPU time comes from CUDA events around the same call.
"""

import json
import statistics
import time

import torch


def measure(fn, iters=50, warmup=5):
    host, gpu = [], []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        t = time.perf_counter()
        a.record()
        fn()
        h = (time.perf_counter() - t) * 1e6
        b.record()
        torch.cuda.synchronize()
        if i >= warmup:
            host.append(h)
            gpu.append(a.elapsed_time(b) * 1e3)
    return dict(host_us=statistics.median(host), gpu_us=statistics.median(gpu))


def main():
    from mono_ep import weight_cache
    from mono_ep.block_weight_quantization import quantize_weights_128x128
    from mono_ep.paired_weight_quantization import quantize_paired_gate_up_weights
    from mono_ep.sm90_prep import prepare_tokens
    from mono_ep.sm90_weight_forms import (
        BACKWARD_FORMS,
        FORWARD_FORMS,
        cached_forms,
        quantize_all_forms,
        quantize_form,
    )
    from mono_ep.token_quantization import quantize_tokens_into

    assert "H100" in torch.cuda.get_device_name()
    w1 = torch.randn(16, 1536, 2048, device="cuda", requires_grad=True)
    w2 = torch.randn(16, 2048, 768, device="cuda", requires_grad=True)
    x = torch.randn(16384, 2048, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, 128, (16384, 8), device="cuda")
    sc = torch.rand(16384, 8, device="cuda")
    qx = torch.empty(17280, 2048, device="cuda", dtype=torch.float8_e4m3fn)
    sx = torch.empty(17280, 16, device="cuda")
    qdy, sdy = torch.empty_like(qx), torch.empty_like(sx)
    idx_out = torch.empty(17280, 8, device="cuda", dtype=torch.int64)
    sc_out = torch.empty(17280, 8, device="cuda")
    t = 16384

    def bump():
        torch.autograd.graph.increment_version(w1)
        torch.autograd.graph.increment_version(w2)

    def miss():
        bump()
        cached_forms(w1, w2, FORWARD_FORMS, "1")

    out = dict(device=torch.cuda.get_device_name(), torch=torch.__version__)
    out["forward_weight_quant_per_call"] = measure(
        lambda: (quantize_weights_128x128(w2), quantize_paired_gate_up_weights(w1))
    )
    out["backward_weight_quant_per_call"] = measure(
        lambda: [quantize_form(w1, w2, n) for n in BACKWARD_FORMS]
    )
    out["all_forms_kernel"] = measure(lambda: quantize_all_forms(w1, w2))
    out["cache_miss_forward"] = measure(miss)
    out["cache_hit_forward"] = measure(lambda: cached_forms(w1, w2, FORWARD_FORMS, "1"))
    out["cache_hit_backward"] = measure(
        lambda: cached_forms(w1, w2, BACKWARD_FORMS, "1")
    )
    out["empty_9"] = measure(
        lambda: [
            torch.empty(16, 1536, 2048, device="cuda", dtype=torch.float8_e4m3fn)
            for _ in range(9)
        ]
    )

    def token_base():
        quantize_tokens_into(x, qx[:t], sx[:t], block_cols=128)
        idx_out[:t].copy_(idx)
        sc_out[:t].copy_(sc)

    def token_base_backward():
        quantize_tokens_into(x, qx[:t], sx[:t], block_cols=128)
        quantize_tokens_into(x, qdy[:t], sdy[:t], block_cols=128)
        idx_out[:t].copy_(idx)
        sc_out[:t].copy_(sc)

    out["token_prep_forward_base"] = measure(token_base)
    out["token_prep_forward_fused"] = measure(
        lambda: prepare_tokens(x, qx[:t], sx[:t], idx, idx_out[:t], sc, sc_out[:t])
    )
    out["token_prep_backward_base"] = measure(token_base_backward)
    out["token_prep_backward_fused"] = measure(
        lambda: prepare_tokens(
            x, qx[:t], sx[:t], idx, idx_out[:t], sc, sc_out[:t], x, qdy[:t], sdy[:t]
        )
    )
    out["cache_stats"] = weight_cache.cache_stats()
    print("PREP_HOST_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
