# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""CUDA-graph dispatch timing including metadata, excluding layout construction."""

import json
import os
import statistics
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mono_ep.packing import DeviceExpertLayout
from mono_ep.peer import _launch_packed_dispatch, peer_gather_quantize_publish


def run():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 2
    torch.manual_seed(591 + rank)
    for tokens, hidden, experts, topk in ((137, 256, 4, 2), (4096, 2048, 8, 4)):
        ids = torch.randint(experts, (tokens, topk), device="cuda", dtype=torch.int32)
        ids[::7] = -1
        layout = DeviceExpertLayout.build(ids, experts, 128)
        rows = layout.row_assignments.numel()
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        scores = torch.rand(tokens, topk, device="cuda")
        specs = [
            ((rows, hidden), torch.float8_e4m3fn),
            ((hidden // 128, rows), torch.int32),
            ((experts, 4), torch.int32),
            *((((rows,), torch.int32),) * 3),
            ((experts,), torch.int32),
            ((rows,), torch.float32),
        ]
        local = [
            symm_mem.empty(*shape, device="cuda", dtype=dtype) for shape, dtype in specs
        ]
        handles = [symm_mem.rendezvous(t, group=dist.group.WORLD) for t in local]
        peer = [
            h.get_buffer(1 - rank, tuple(t.shape), t.dtype)
            for h, t in zip(handles, local, strict=True)
        ]

        def fused(tile, fence_pack=16, x=x, layout=layout, scores=scores, peer=peer):
            _launch_packed_dispatch(
                x,
                layout.row_assignments,
                layout.expert_ids,
                layout.starts,
                layout.padded_sizes,
                layout.ends,
                scores,
                *peer,
                tile,
                fence_pack,
            )

        def separate(x=x, layout=layout, scores=scores, peer=peer, experts=experts):
            peer[2].copy_(layout.chunk_descriptors())
            for dest, src in zip(
                peer[3:],
                (
                    layout.expert_ids,
                    layout.token_ids(),
                    layout.row_assignments,
                    layout.ends,
                    layout.scores(scores),
                ),
                strict=True,
            ):
                dest.copy_(src)
            peer_gather_quantize_publish(
                x, layout.token_ids(), peer[0], peer[1], peer[2], 0, 32, experts
            )

        graphs = {}
        for name, operation in (
            ("separate", separate),
            ("fused_16_old_fence", partial(fused, 16, 1)),
            *((f"fused_{tile}", partial(fused, tile)) for tile in (16, 32, 64, 128)),
        ):
            operation()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = [
                torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)
            ]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                start.record()
                operation()
                end.record()
            graphs[name] = graph, start, end
        torch.cuda.synchronize()
        dist.barrier()
        expected = [
            t.view(torch.uint8).clone() if t.dtype == torch.float8_e4m3fn else t.clone()
            for t in local
        ]
        timings = {name: [] for name in graphs}
        for trial in range(25):
            for name in list(graphs) if trial % 2 else list(reversed(graphs)):
                for t in local:
                    t.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                graph, start, end = graphs[name]
                graph.replay()
                end.synchronize()
                dist.barrier()
                torch.cuda.synchronize()
                for got, want in zip(local, expected, strict=True):
                    if got.dtype == torch.float8_e4m3fn:
                        got = got.view(torch.uint8)
                    torch.testing.assert_close(got, want, atol=0, rtol=0)
                if trial >= 5:
                    timings[name].append(start.elapsed_time(end))
        print(
            json.dumps(
                dict(
                    rank=rank,
                    tokens=tokens,
                    hidden=hidden,
                    experts=experts,
                    topk=topk,
                    capacity=rows,
                    median_ms={
                        name: statistics.median(v) for name, v in timings.items()
                    },
                    samples_ms=timings,
                )
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
