# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Check persistent column-scaled operands against independent packing."""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from mono_ep.hopper_wgrad import weight_grad_fp8_hopper
from mono_ep.runtime import PersistentMoE
from mono_ep.wgrad_quantization import quantize_transposed_groups


def _reference_pools(backend, x, scores, w1, w2):
    import deep_gemm
    from deep_gemm.mega import _interleave_weights
    from mono_ep.quantization import quantize_fp8

    capacity = backend.pools[4].shape[0]
    wrapped = (
        backend.starts[-1] + backend.padded_counts[-1] > backend.pools[0].shape[0]
    ).to(torch.int32)
    dist.all_reduce(wrapped, op=dist.ReduceOp.MAX)
    if not wrapped.item():
        return backend.pools[:4], True
    widths = (w1.shape[2], w1.shape[2], w2.shape[2], w1.shape[1])
    pools = [
        torch.zeros(capacity, width, dtype=torch.bfloat16, device=x.device)
        for width in widths
    ] + [torch.empty_like(pool) for pool in backend.pools[4:10]]
    prepared = [
        quantize_fp8(weight, block_rows=128, block_cols=128)
        for weight in (
            _interleave_weights(w1),
            w2.transpose(1, 2).contiguous(),
            w1.transpose(1, 2).contiguous(),
        )
    ]
    starts, counts = torch.empty_like(backend.starts), torch.empty_like(backend.counts)
    dx, ds = torch.empty_like(x), torch.empty_like(scores)
    deep_gemm._C.fp8_mono_ep_backward_sm90(
        dx,
        ds,
        *(pair[0] for pair in prepared),
        [pair[1] for pair in prepared],
        pools,
        starts,
        counts,
        backend.buffer.buffer,
        backend.buffer.handle.buffer_ptrs,
        backend.dy,
        backend.dy_pointers,
        backend.ds_partials,
        backend.ds_pointers,
        backend.rank,
        backend.buffer.num_max_tokens_per_rank,
        backend.buffer.l1_acts.shape[0],
        backend.runtime.topk,
    )
    torch.testing.assert_close(starts, backend.starts)
    torch.testing.assert_close(counts, backend.counts)
    torch.testing.assert_close(dx, x.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ds, scores.grad, rtol=2e-2, atol=2e-2)
    return pools[:4], False


def run():
    import deep_gemm

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", rank), timeout=timedelta(seconds=120)
    )
    torch.manual_seed(714 + rank)
    ring_wrap = os.environ.get("MONO_EP_TEST_RING_WRAP") == "1"
    physical_sms = torch.cuda.get_device_properties(rank).multi_processor_count
    active_sms = 16 if ring_wrap else physical_sms
    deep_gemm.set_num_sms(active_sms)
    tokens = 4096 if ring_wrap else 137
    hidden, inner, experts, topk = (1024 if ring_wrap else 512), 512, 8, 4
    if os.environ.get("MONO_EP_TEST_QWEN_SHAPE") == "1":
        hidden, inner, experts, topk = 2048, 768, 128, 8
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens if ring_wrap else tokens + 31,
        topk,
        hidden,
        inner,
        compute_dtype="blockwise",
        retain_activations=False,
        backward_backend="native",
        reserved_sms=physical_sms - active_sms,
    )
    operation = torch.compile(
        runtime, fullgraph=True, options={"triton.cudagraphs": False}
    )
    local = experts // dist.get_world_size()
    records = []
    for master_dtype in (torch.float32, torch.bfloat16):
        for skewed in (False, True, False):
            x = torch.randn(
                tokens - rank * 11,
                hidden,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            ids = (
                torch.rand(x.shape[0], experts, device="cuda")
                .argsort(-1)[:, :topk]
                .contiguous()
            )
            if skewed:
                ids[:] = torch.arange(topk, device="cuda")
            scores = (
                torch.rand(x.shape[0], topk, device="cuda").softmax(-1).requires_grad_()
            )
            w1 = (
                torch.randn(local, 2 * inner, hidden, device="cuda", dtype=master_dtype)
                / hidden**0.5
            ).requires_grad_()
            w2 = (
                torch.randn(local, hidden, inner, device="cuda", dtype=master_dtype)
                / inner**0.5
            ).requires_grad_()
            output = operation(x, ids, scores, w1, w2)
            output.backward(torch.randn_like(output))
            backend = runtime._peer_backward
            reference_pools, exact = _reference_pools(backend, x, scores, w1, w2)
            padded_ends = backend.starts + backend.padded_counts
            valid_ends = backend.starts + backend.counts
            live = int(padded_ends[-1].item())
            torch.testing.assert_close(
                backend.starts,
                backend.padded_counts.cumsum(0, dtype=torch.int32)
                - backend.padded_counts,
            )
            assert live <= backend.pools[4].shape[0]
            if ring_wrap and skewed and rank == 0:
                assert live > 2 * backend.buffer.l1_acts.shape[0], (
                    live,
                    backend.buffer.l1_acts.shape[0],
                )
            torch.testing.assert_close(
                backend.padded_counts, ((backend.counts + 127) // 128) * 128
            )
            if exact:
                for index, source in enumerate(reference_pools):
                    expected_q, expected_sf = quantize_transposed_groups(
                        source,
                        padded_ends,
                        use_psum_layout=True,
                        scales_mn_major=True,
                        valid_ends=valid_ends,
                    )
                    width = source.shape[1]
                    torch.testing.assert_close(
                        backend.pools[10 + index][: live * width].float(),
                        expected_q[: live * width].float(),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        backend.pools[14 + index][:, : live // 128],
                        expected_sf[:, : live // 128],
                        rtol=0,
                        atol=0,
                    )
            reference_dw1 = weight_grad_fp8_hopper(
                reference_pools[3], reference_pools[0], padded_ends, valid_ends
            )
            reference_dw2 = weight_grad_fp8_hopper(
                reference_pools[1], reference_pools[2], padded_ends, valid_ends
            )
            weight_errors = []
            for observed, expected in (
                (w1.grad, reference_dw1),
                (w2.grad, reference_dw2),
            ):
                expected = expected.to(master_dtype)
                error = (
                    observed.float() - expected.float()
                ).norm() / expected.float().norm().clamp_min(1e-8)
                weight_errors.append(error.item())
                if exact:
                    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)
                else:
                    assert error < 0.02, error.item()
            records.append(
                dict(
                    master_dtype=str(master_dtype),
                    hidden=hidden,
                    intermediate=inner,
                    experts=experts,
                    topk=topk,
                    skewed=skewed,
                    live_rows=live,
                    ring_rows=backend.buffer.l1_acts.shape[0],
                    exact_packed_operand_parity=exact,
                    weight_gradient_relative_errors=weight_errors,
                )
            )
    print(
        json.dumps(dict(rank=rank, records=records)),
        flush=True,
    )
    runtime.close()
    deep_gemm.set_num_sms(physical_sms)
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
