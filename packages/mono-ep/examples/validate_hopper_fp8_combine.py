# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate per-route FP8 combine and replacement of BF16 return storage."""

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
    assert dist.get_world_size() == 2
    physical = torch.cuda.get_device_properties(rank).multi_processor_count
    deep_gemm.set_num_sms(16)
    wrap = os.environ.get("MONO_EP_TEST_RING_WRAP") == "1"
    tokens = int(os.environ.get("MONO_EP_TEST_TOKENS", 4096 if wrap else 137))
    hidden, inner, experts = (
        int(os.environ.get("MONO_EP_TEST_HIDDEN", "2048")),
        int(os.environ.get("MONO_EP_TEST_INTERMEDIATE", "768")),
        128,
    )
    topk = int(os.environ.get("MONO_EP_TEST_TOPK", "8"))
    if topk not in (1, 8) or (wrap and topk != 8):
        raise ValueError("Use top-k 1 or 8; forced ring wrap requires top-k 8")
    runtimes = []
    for enabled in ("0", "1"):
        os.environ["MONO_EP_SM90_FP8_TRANSPORT"] = "1"
        os.environ["MONO_EP_SM90_FP8_COMBINE"] = enabled
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
    assert control.fp8_transport and candidate.fp8_transport
    assert not control.fp8_combine and candidate.fp8_combine

    def combine_views(backend):
        return deep_gemm._C.get_mono_ep_backward_combine_views(
            backend.buffer.buffer,
            2,
            experts,
            backend.buffer.num_max_tokens_per_rank,
            topk,
            hidden,
            inner,
            backend.buffer.l1_acts.shape[0],
            backend.fp8_transport,
            backend.fp8_combine,
        )

    control_values, _ = combine_views(control)
    candidate_values, candidate_scales = combine_views(candidate)
    assert control_values.dtype == torch.bfloat16
    assert candidate_values.dtype == torch.float8_e4m3fn
    assert candidate_scales.dtype == torch.float32
    expected_savings = (
        topk * candidate.buffer.num_max_tokens_per_rank * (hidden - hidden // 128 * 4)
    )
    actual_savings = control.buffer.buffer.nbytes - candidate.buffer.buffer.nbytes
    assert actual_savings == expected_savings, (actual_savings, expected_savings)
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
            ids[::31] = -1
            scores = torch.rand(x.shape[0], topk, device="cuda").softmax(-1)
            candidate_values.fill_(float("nan"))
            candidate_scales.fill_(float("nan"))
            observed = candidate.backward(dy, x, ids, scores, w1, w2)
            missing = (ids < 0).all(-1)
            torch.testing.assert_close(
                observed[0][missing],
                torch.zeros_like(observed[0][missing]),
                rtol=0,
                atol=0,
            )
            expected = list(control.backward(dy, x, ids, scores, w1, w2))
            route_values = control_values[:, : x.shape[0]].clone()
            valid = ids.T >= 0
            route_values.masked_fill_(~valid[..., None], 0)
            decoded = decoded_wire(route_values.reshape(-1, hidden))
            expected[0] = decoded.reshape_as(route_values).float().sum(0).bfloat16()
            assert torch.isfinite(
                candidate_values[:, : x.shape[0]].float()[valid]
            ).all()
            assert torch.isfinite(candidate_scales[:, : x.shape[0]][valid]).all()
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
                    topk=topk,
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
                hidden=hidden,
                intermediate=inner,
                reference="BF16 per-route contributions independently FP8-quantized before sum",
                return_buffer_bytes_saved=actual_savings,
                poisoned_unused_return_capacity=True,
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
