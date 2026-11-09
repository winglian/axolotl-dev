# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Same-job peer-combine comparison with identical math and completion protocol."""

import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
from mono_ep.combine import _scatter_publish, _wait_copy, combine_wait_copy


def launch(variant, dx, ids, peer, ds, slots):
    rows, hidden = dx.shape
    tokens = peer[0].shape[0]
    _scatter_publish[(triton.cdiv(rows, 16),)](
        dx,
        ids,
        peer[0],
        peer[2],
        0,
        rows,
        tokens,
        hidden,
        ds,
        slots,
        peer[1],
        peer[3],
        tokens * 2,
        True,
        ROW_TILE=16,
        VECTORIZE=variant == "vectorized",
        num_warps=4,
        debug=True,
    )


def benchmark_copy(rank):
    for rows, hidden in ((1, 4096), (1, 4097), (137, 256), (4096, 2048), (8192, 2048)):
        for dtype in (torch.float32, torch.bfloat16):
            source = torch.randn(rows, hidden, device="cuda")
            output = torch.empty_like(source, dtype=dtype)
            expected = source.to(dtype)
            queue = torch.zeros(2, 4, device="cuda", dtype=torch.int32)
            queue[:, 3] = 1
            graphs = {}
            for variant in ("serial", "selected"):
                if variant == "serial":
                    operation = lambda source=source, output=output, queue=queue: (
                        _wait_copy[(1,)](
                            source,
                            output,
                            queue,
                            2,
                            source.numel(),
                            num_warps=4,
                            debug=True,
                        )
                    )
                else:
                    operation = lambda source=source, output=output, queue=queue: (
                        combine_wait_copy(source, output, queue)
                    )
                operation()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph):
                    start.record()
                    operation()
                    end.record()
                graphs[variant] = graph, start, end
            samples = {"serial": [], "selected": []}
            for trial in range(7):
                for variant in (
                    ("serial", "selected") if trial % 2 == 0 else ("selected", "serial")
                ):
                    output.fill_(7)
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph, start, end = graphs[variant]
                    graph.replay()
                    end.synchronize()
                    samples[variant].append(start.elapsed_time(end))
                    torch.testing.assert_close(output, expected, atol=0, rtol=0)
            medians = {k: statistics.median(v) for k, v in samples.items()}
            print(
                json.dumps(
                    dict(
                        rank=rank,
                        gpu=torch.cuda.get_device_name(),
                        rows=rows,
                        hidden=hidden,
                        output_dtype=str(dtype),
                        benchmark="consumer_copy",
                        median_ms_by_variant=medians,
                        samples_ms_by_variant=samples,
                        speedup=medians["serial"] / medians["selected"],
                        timing_scope="CUDA-graph device events; ready-flag acquire plus copy/cast; producers excluded",
                    )
                ),
                flush=True,
            )
            del graphs, graph


def run():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 2
    for rows, hidden in ((512, 256), (4096, 2048), (8192, 2048)):
        tokens = rows // 2
        buffers = [
            symm_mem.empty(tokens, hidden, device="cuda", dtype=torch.float32),
            symm_mem.empty(tokens * 2, 1, device="cuda", dtype=torch.float32),
            symm_mem.empty(1, 4, device="cuda", dtype=torch.int32),
            symm_mem.empty(1, 4, device="cuda", dtype=torch.int32),
        ]
        handles = [symm_mem.rendezvous(t, group=dist.group.WORLD) for t in buffers]
        peer = [
            h.get_buffer(1 - rank, tuple(t.shape), t.dtype)
            for h, t in zip(handles, buffers, strict=True)
        ]
        torch.manual_seed(912 + rank)
        dx = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        ds = torch.randn(rows, device="cuda")
        ids = torch.randint(tokens, (rows,), device="cuda", dtype=torch.int32)
        slots = torch.randint(tokens * 2, (rows,), device="cuda", dtype=torch.int32)
        ids[::7] = -1
        slots[::5] = -1
        expected = [torch.zeros_like(t) for t in buffers[:2]]
        expected[0].index_add_(0, ids[ids >= 0].long(), dx[ids >= 0].float())
        expected[1][:, 0].index_add_(0, slots[slots >= 0].long(), ds[slots >= 0])
        received = []
        for tensor in expected:
            all_expected = [torch.empty_like(tensor) for _ in range(2)]
            dist.all_gather(all_expected, tensor)
            received.append(all_expected[1 - rank])

        timings = {"row_loop": [], "vectorized": []}
        errors = []
        graphs = {}
        for trial in range(9):
            for tile in (
                ("row_loop", "vectorized")
                if trial % 2 == 0
                else ("vectorized", "row_loop")
            ):
                for tensor in buffers:
                    tensor.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.synchronize()
                if trial == 0:
                    start, end = (
                        torch.cuda.Event(enable_timing=True, external=True),
                        torch.cuda.Event(enable_timing=True, external=True),
                    )
                    start.record()
                    launch(tile, dx, ids, peer, ds, slots)
                    end.record()
                else:
                    if tile not in graphs:
                        graph = torch.cuda.CUDAGraph()
                        start, end = (
                            torch.cuda.Event(enable_timing=True, external=True),
                            torch.cuda.Event(enable_timing=True, external=True),
                        )
                        with torch.cuda.graph(graph):
                            start.record()
                            launch(tile, dx, ids, peer, ds, slots)
                            end.record()
                        graphs[tile] = (graph, start, end)
                    graph, start, end = graphs[tile]
                    graph.replay()
                end.synchronize()
                elapsed = start.elapsed_time(end)
                dist.barrier()
                torch.cuda.synchronize()
                assert buffers[2][0, 2].item() == triton.cdiv(rows, 16)
                assert buffers[2][0, 3].item() == buffers[3][0, 3].item() == 1
                for actual, want in zip(buffers[:2], received, strict=True):
                    error = (
                        (actual - want).norm() / want.norm().clamp_min(1e-8)
                    ).item()
                    assert error < 1e-5, error
                    errors.append(error)
                if trial >= 2:
                    timings[tile].append(elapsed)
        medians = {
            str(tile): statistics.median(values) for tile, values in timings.items()
        }
        print(
            json.dumps(
                dict(
                    rank=rank,
                    gpu=torch.cuda.get_device_name(),
                    rows=rows,
                    hidden=hidden,
                    tokens=tokens,
                    median_ms_by_variant=medians,
                    samples_ms_by_variant=timings,
                    speedup=medians["row_loop"] / medians["vectorized"],
                    max_relative_l2=max(errors),
                    timing_scope="CUDA-graph device-event timing of paired peer scatter and completion; excludes resets, barriers and consumer copy",
                )
            ),
            flush=True,
        )
        dist.barrier()
        del graphs, graph, peer, handles, buffers
    benchmark_copy(rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
