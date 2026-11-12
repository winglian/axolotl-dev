# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compare FP8 peer transport with independently decoded BF16 wire operands."""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from mono_ep.runtime import PersistentMoE


def decoded_wire(tensor):
    blocks = tensor.float().reshape(tensor.shape[0], -1, 128)
    scale = torch.exp2(
        torch.ceil(
            torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)
        )
    )
    values = (blocks / scale).to(torch.float8_e4m3fn)
    return (values.float() * scale).reshape_as(tensor).bfloat16()


def run():
    import deep_gemm

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", rank), timeout=timedelta(seconds=180)
    )
    physical = torch.cuda.get_device_properties(rank).multi_processor_count
    deep_gemm.set_num_sms(16)
    wrap = os.environ.get("MONO_EP_TEST_RING_WRAP") == "1"
    tokens = 4096 if wrap else 137
    hidden, inner, experts, topk = 2048, 768, 128, 8
    runtimes = []
    for enabled in ("0", "1"):
        os.environ["MONO_EP_SM90_FP8_TRANSPORT"] = enabled
        os.environ["MONO_EP_SM90_FP8_COMBINE"] = "0"
        runtimes.append(
            PersistentMoE(
                dist.group.WORLD,
                experts,
                tokens,
                topk,
                hidden,
                inner,
                compute_dtype="blockwise",
                retain_activations=False,
                backward_backend="native",
                reserved_sms=physical - 16,
            )
        )
    control, candidate = [runtime._peer_backward for runtime in runtimes]
    assert not control.fp8_transport and candidate.fp8_transport
    assert candidate.buffer.buffer.data_ptr() != runtimes[1].buffer.buffer.data_ptr()
    torch.manual_seed(951 + rank)
    records = []
    for master in (torch.float32, torch.bfloat16):
        w1 = (
            torch.randn(experts // 2, 2 * inner, hidden, device="cuda", dtype=master)
            / hidden**0.5
        )
        w2 = (
            torch.randn(experts // 2, hidden, inner, device="cuda", dtype=master)
            / inner**0.5
        )
        for skewed in (False, True, False):
            x = torch.randn(
                tokens - rank * 11, hidden, device="cuda", dtype=torch.bfloat16
            )
            dy = torch.randn_like(x)
            ids = (
                torch.rand(x.shape[0], experts, device="cuda")
                .argsort(-1)[:, :topk]
                .contiguous()
            )
            if skewed:
                ids[:] = torch.arange(topk, device="cuda")
            scores = torch.rand(x.shape[0], topk, device="cuda").softmax(-1)
            observed = candidate.backward(dy, x, ids, scores, w1, w2)
            expected = control.backward(
                decoded_wire(dy), decoded_wire(x), ids, scores, w1, w2
            )
            errors = {}
            for name, actual, reference in zip(
                ("dx", "dscore", "dw1", "dw2"), observed, expected, strict=True
            ):
                assert torch.isfinite(actual).all(), name
                error = (
                    actual.float() - reference.float()
                ).norm() / reference.float().norm().clamp_min(1e-8)
                errors[name] = error.item()
                assert error < 0.02, (name, error.item())
            torch.testing.assert_close(candidate.counts, control.counts)
            live = int((candidate.starts[-1] + candidate.padded_counts[-1]).item())
            ring = candidate.buffer.l1_acts.shape[0]
            if wrap and skewed and rank == 0:
                assert live > 2 * ring, (live, ring)
            records.append(
                dict(
                    master_dtype=str(master),
                    skewed=skewed,
                    live_rows=live,
                    ring_rows=ring,
                    relative_errors=errors,
                )
            )
    torch.cuda.synchronize()
    print(
        json.dumps(
            dict(
                rank=rank,
                reference="independent row-FP8 decode to BF16 wire",
                records=records,
            )
        ),
        flush=True,
    )
    for runtime in runtimes:
        runtime.close()
    deep_gemm.set_num_sms(physical)
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
