# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate SM90 GPU-offset grouped BF16 weight gradients."""

import json

import torch


def relative(actual, expected):
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-8))


def run():
    import deep_gemm

    torch.manual_seed(19)
    pool_rows, width_a, width_b = 1152, 128, 192
    cases = (
        ([0, 128, 128, 384], [17, 128, 257, 895]),
        ([0, 0, 512, 640], [0, 511, 513, 640]),
        ([0, 256, 384, 512], [129, 257, 385, 513]),
    )
    records = []
    for iteration, (starts_host, ends_host) in enumerate(cases):
        starts = torch.tensor(starts_host, device="cuda", dtype=torch.int32)
        ends = torch.tensor(ends_host, device="cuda", dtype=torch.int32)
        a = torch.zeros(pool_rows, width_a, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(pool_rows, width_b, device="cuda", dtype=torch.bfloat16)
        reference = []
        for start, end in zip(starts_host, ends_host, strict=True):
            if end > start:
                a[start:end].normal_()
                b[start:end].normal_()
                reference.append(a[start:end].float().T @ b[start:end].float())
            else:
                reference.append(torch.zeros(width_a, width_b, device="cuda"))
        expected = torch.stack(reference)
        actual = torch.zeros_like(expected)
        deep_gemm._C.k_grouped_bf16_gemm_tn_gpu_offsets(
            a, b, actual, starts, ends, pool_rows, actual
        )
        torch.cuda.synchronize()
        error = relative(actual, expected)
        records.append(
            {
                "iteration": iteration,
                "starts": starts_host,
                "ends": ends_host,
                "empty_groups": sum(
                    s == e for s, e in zip(starts_host, ends_host, strict=True)
                ),
                "relative_error": error,
            }
        )
        assert error < 2e-3, records[-1]
    print(json.dumps(records))


if __name__ == "__main__":
    run()
