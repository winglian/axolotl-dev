# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Isolate fixed-capacity dispatch cost; compact capacity is only a diagnostic."""

import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
from mono_ep.peer import _quantize_publish, peer_gather_quantize_publish


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 2
    tokens, hidden = 16384, 2048
    torch.manual_seed(851 + rank)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    sources = [torch.empty_like(x) for _ in range(2)]
    dist.all_gather(sources, x)
    cases = {}
    handles = []
    for rows in (17408, 133120):
        values = symm_mem.empty(rows, hidden, device="cuda", dtype=torch.uint8)
        scales = symm_mem.empty(hidden // 128, rows, device="cuda", dtype=torch.int32)
        queue = symm_mem.empty(1, 4, device="cuda", dtype=torch.int32)
        mapped = []
        for tensor in (values, scales, queue):
            handle = symm_mem.rendezvous(tensor, group=dist.group.WORLD)
            handles.append(handle)
            mapped.append(handle.get_buffer(1 - rank, tensor.shape, tensor.dtype))
        indices = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
        indices[:tokens] = torch.arange(tokens, device="cuda", dtype=torch.int32)
        cases[rows] = (values, scales, queue, mapped, indices)

    def run(rows, skip_padding=False):
        _, _, _, (q, s, ready), indices = cases[rows]
        if skip_padding:
            counter = torch.zeros((), device=x.device, dtype=torch.int32)
            _quantize_publish[(triton.cdiv(rows, 16), hidden // 128)](
                x,
                q.view(torch.float8_e4m3fn),
                s,
                ready,
                counter,
                0,
                rows,
                hidden,
                32,
                indices,
                True,
                tokens,
                SKIP_PADDING=True,
                num_warps=4,
                enable_fp_fusion=False,
                debug=True,
            )
        else:
            peer_gather_quantize_publish(
                x, indices, q.view(torch.float8_e4m3fn), s, ready, 0
            )

    for rows, (values, scales, queue, _, indices) in cases.items():
        values.fill_(127)
        scales.zero_()
        queue.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        run(rows)
        torch.cuda.synchronize()
        dist.barrier()
        expected = torch.empty_like(values)
        expected_scales = torch.empty_like(scales)
        expected_queue = torch.zeros_like(queue)
        peer_gather_quantize_publish(
            sources[1 - rank],
            indices,
            expected.view(torch.float8_e4m3fn),
            expected_scales,
            expected_queue,
            0,
        )
        torch.testing.assert_close(values, expected, atol=0, rtol=0)
        torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
        assert queue[0, 3].item() == 1
        assert not values[tokens:].any().item()
    torch.testing.assert_close(
        cases[17408][0][:tokens], cases[133120][0][:tokens], atol=0, rtol=0
    )
    torch.testing.assert_close(
        cases[17408][1][:, :tokens], cases[133120][1][:, :tokens], atol=0, rtol=0
    )

    # Receiver initialization must finish globally before any sender writes.
    def clear_payload(rows):
        cases[rows][0].zero_()
        cases[rows][1].fill_(0x69696969)

    rows = 133120
    original_indices = cases[rows][4].clone()
    for pattern in ("prefix", "holes", "empty", "full"):
        indices = cases[rows][4]
        indices.copy_(original_indices)
        if pattern == "holes":
            indices[::7] = -1
        elif pattern == "empty":
            indices.fill_(-1)
        elif pattern == "full":
            indices.copy_(torch.arange(rows, device="cuda", dtype=torch.int32) % tokens)
        values, scales, queue = cases[rows][:3]
        values.fill_(127)
        scales.fill_(-1)
        clear_payload(rows)
        queue.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        run(rows, True)
        torch.cuda.synchronize()
        dist.barrier()
        expected = torch.empty_like(values)
        expected_scales = torch.empty_like(scales)
        peer_gather_quantize_publish(
            sources[1 - rank],
            indices,
            expected.view(torch.float8_e4m3fn),
            expected_scales,
            torch.zeros_like(queue),
            0,
        )
        torch.testing.assert_close(values, expected, atol=0, rtol=0)
        torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
        assert queue[0, 3].item() == 1
    cases[rows][4].copy_(original_indices)
    modes = {
        "compact": (17408, False),
        "capacity": (133120, False),
        "local_clear": (133120, True),
    }
    samples = {name: [] for name in modes}
    clear_samples = {name: [] for name in modes}
    for trial in range(25):
        order = list(modes) if trial % 2 else list(reversed(modes))
        for name in order:
            rows, skip = modes[name]
            cases[rows][2].zero_()
            clear_start = torch.cuda.Event(enable_timing=True)
            clear_end = torch.cuda.Event(enable_timing=True)
            clear_start.record()
            if skip:
                clear_payload(rows)
            clear_end.record()
            torch.cuda.synchronize()
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run(rows, skip)
            end.record()
            end.synchronize()
            clear_ms = clear_start.elapsed_time(clear_end) if skip else 0.0
            elapsed = torch.tensor(
                [start.elapsed_time(end) + clear_ms, clear_ms], device="cuda"
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if trial >= 5:
                samples[name].append(elapsed[0].item())
                clear_samples[name].append(elapsed[1].item())
    if rank == 0:
        print(
            json.dumps(
                dict(
                    benchmark="bidirectional_peer_dispatch_capacity",
                    world_size=2,
                    gpu=torch.cuda.get_device_name(),
                    torch_version=torch.__version__,
                    tokens=tokens,
                    hidden=hidden,
                    exact_payload_parity=True,
                    notes="Gather quantization and publication only. Local-clear timing includes receiver fills; the required pre-send barrier is excluded for all modes. No metadata, experts or consumers. Compact capacity is diagnostic only.",
                    padding_patterns=["prefix", "holes", "empty", "full"],
                    clear_samples_max_rank_ms=clear_samples,
                    payload_bytes={
                        rows: rows * (hidden + hidden // 128 * 4) for rows in cases
                    },
                    median_max_rank_ms={
                        rows: statistics.median(v) for rows, v in samples.items()
                    },
                    samples_max_rank_ms=samples,
                )
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
