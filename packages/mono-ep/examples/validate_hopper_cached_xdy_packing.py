# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate cached row-FP8 x/dY against independent column packing."""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from mono_ep.runtime import PersistentMoE
from mono_ep.wgrad_quantization import quantize_transposed_groups


def _decode_row_cache(values, scales):
    return (values.float() * scales.repeat_interleave(128, dim=1)).bfloat16()


def run():
    import deep_gemm

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", rank), timeout=timedelta(seconds=180)
    )
    physical_sms = torch.cuda.get_device_properties(rank).multi_processor_count
    deep_gemm.set_num_sms(physical_sms)
    torch.manual_seed(881 + rank)
    tokens, hidden, inner, experts, topk = 269 - rank * 12, 2048, 768, 128, 8
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        288,
        topk,
        hidden,
        inner,
        compute_dtype="blockwise",
        retain_activations=False,
        backward_backend="native",
    )
    operation = torch.compile(
        runtime, fullgraph=True, options={"triton.cudagraphs": False}
    )
    backend = runtime._peer_backward
    assert backend.fp8_transport
    local_experts = experts // dist.get_world_size()
    records = []
    for master_dtype in (torch.float32, torch.bfloat16):
        for skewed in (False, True):
            for pool in backend.pools[4:8]:
                pool.fill_(float("nan"))
            x = torch.randn(
                tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            ids = (
                torch.rand(tokens, experts, device="cuda")
                .argsort(-1)[:, :topk]
                .contiguous()
            )
            if skewed:
                ids[:] = torch.arange(topk, device="cuda")
            scores = (
                torch.rand(tokens, topk, device="cuda").softmax(-1).requires_grad_()
            )
            w1 = (
                torch.randn(
                    local_experts,
                    2 * inner,
                    hidden,
                    device="cuda",
                    dtype=master_dtype,
                )
                / hidden**0.5
            ).requires_grad_()
            w2 = (
                torch.randn(
                    local_experts,
                    hidden,
                    inner,
                    device="cuda",
                    dtype=master_dtype,
                )
                / inner**0.5
            ).requires_grad_()
            output = operation(x, ids, scores, w1, w2)
            output.backward(torch.randn_like(output))
            padded_ends = backend.starts + backend.padded_counts
            valid_ends = backend.starts + backend.counts
            live = int(padded_ends[-1].item())
            if skewed:
                assert torch.count_nonzero(backend.counts == 0), backend.counts
            for index, source in enumerate(
                (
                    _decode_row_cache(backend.pools[4], backend.pools[6]),
                    _decode_row_cache(backend.pools[5], backend.pools[7]),
                )
            ):
                expected_q, expected_sf = quantize_transposed_groups(
                    source,
                    padded_ends,
                    use_psum_layout=True,
                    scales_mn_major=True,
                    valid_ends=valid_ends,
                )
                assert torch.equal(
                    backend.pools[10 + index][: live * hidden].view(torch.uint8),
                    expected_q[: live * hidden].view(torch.uint8),
                )
                torch.testing.assert_close(
                    backend.pools[14 + index][:, : live // 128],
                    expected_sf[:, : live // 128],
                    rtol=0,
                    atol=0,
                )
            records.append(
                dict(
                    rank=rank,
                    master_dtype=str(master_dtype),
                    skewed=skewed,
                    tokens=tokens,
                    live_rows=live,
                    padded_rows=int((backend.padded_counts - backend.counts).sum()),
                    empty_experts=int((backend.counts == 0).sum()),
                    exact_q_sf=True,
                )
            )
    print(json.dumps(dict(rank=rank, records=records)), flush=True)
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
