# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate receive-side packing through real symmetric peer mappings."""

import json
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mono_ep.packing import DeviceExpertLayout
from mono_ep.peer_packing import pack_peer_rows, pack_peer_rows_ready_out
from mono_ep.synchronization import peer_epoch_barrier


def validate_ready_consumer(group, device, world, rank, flags, pointers, epoch):
    import deep_gemm
    from mono_ep.chunk import ready_grouped_bf16_out
    from mono_ep.ops import grouped_bf16

    capacity, width, experts, topk = 4096, 512, 4, 4
    source = symm_mem.empty(capacity, width, dtype=torch.bfloat16, device=device)
    handle = symm_mem.rendezvous(source, group=group)
    sources = [handle.get_buffer(r, source.shape, source.dtype) for r in range(world)]
    torch.manual_seed(319 + rank)
    source.normal_(std=0.1)
    epoch += 1
    peer_epoch_barrier(flags, pointers, rank, epoch)
    weights = (
        torch.randn(experts, 512, width, device=device, dtype=torch.bfloat16) * 0.02
    )
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    main = torch.cuda.current_stream(device)
    previous_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(
        max(
            2,
            (torch.cuda.get_device_properties(device).multi_processor_count - 16)
            // 2
            * 2,
        )
    )
    try:
        for empty in (False, True):
            ids = (
                torch.arange(world * capacity * topk, device=device).reshape(-1, topk)
                % experts
            )
            ids[::11] = -1
            if empty:
                ids.fill_(-1)
            layout = DeviceExpertLayout.build(ids, experts, 128)
            queue = layout.chunk_descriptors()
            counters = torch.zeros(experts, device=device, dtype=torch.int32)
            packed = torch.zeros(
                (layout.row_assignments.numel(), width),
                device=device,
                dtype=torch.bfloat16,
            )
            out = torch.zeros_like(packed)
            expected_x = pack_peer_rows(sources, layout.row_assignments, topk)
            expected = grouped_bf16(expected_x, weights, layout.expert_ids)
            queue[:, 3] = 1
            ready_grouped_bf16_out(
                packed, weights, out, queue, layout.expert_ids, incremental=True
            )
            pack_peer_rows_ready_out(
                sources, layout.row_assignments, topk, packed, queue, counters
            )
            torch.cuda.synchronize()
            packed.fill_(float("nan"))
            out.zero_()
            queue[:, 3].zero_()
            counters.zero_()
            producer.wait_stream(main)
            consumer.wait_stream(main)
            with torch.cuda.stream(consumer):
                ready_grouped_bf16_out(
                    packed, weights, out, queue, layout.expert_ids, incremental=True
                )
            with torch.cuda.stream(producer):
                pack_peer_rows_ready_out(
                    sources, layout.row_assignments, topk, packed, queue, counters
                )
            main.wait_stream(producer)
            main.wait_stream(consumer)
            torch.testing.assert_close(out, expected, atol=0.005, rtol=0.005)
            torch.testing.assert_close(queue[:, 3], torch.ones_like(queue[:, 3]))
    finally:
        torch.cuda.synchronize()
        deep_gemm.set_num_sms(previous_sms)
    epoch += 1
    peer_epoch_barrier(flags, pointers, rank, epoch)
    return epoch


def validate_peer_scatter(group, device, world, rank, flags, flag_pointers, epoch):
    from mono_ep.ops import grouped_bf16
    from mono_ep.peer_scatter import peer_gemm_out

    capacity, width, experts, topk = 137, 512, 4, 4
    slots = capacity * topk
    output = symm_mem.empty(slots, width, device=device, dtype=torch.bfloat16)
    handle = symm_mem.rendezvous(output, group=group)
    outputs = [handle.get_buffer(r, output.shape, output.dtype) for r in range(world)]
    pointers = torch.tensor(
        [t.data_ptr() for t in outputs], device=device, dtype=torch.int64
    )
    torch.manual_seed(830 + rank)
    weights = (
        torch.randn(experts, width, width, device=device, dtype=torch.bfloat16) * 0.02
    )
    for compiled in (False, True):
        if compiled:
            torch.compiler.reset()
        operation = (
            torch.compile(peer_gemm_out, fullgraph=True) if compiled else peer_gemm_out
        )
        for step in range(3):
            ids = torch.arange(world * slots, device=device).reshape(-1, topk) % (
                world * experts
            )
            if step == 1:
                ids.remainder_(experts)
            ids[::5] = -1
            local = torch.where(
                (ids >= rank * experts) & (ids < (rank + 1) * experts),
                ids - rank * experts,
                -1,
            )
            layout = DeviceExpertLayout.build(local, experts, 128)
            x = (
                torch.randn(
                    world * capacity, width, device=device, dtype=torch.bfloat16
                )
                * 0.1
            )
            packed = layout.pack(x)
            reference = grouped_bf16(packed, weights, layout.expert_ids)
            assignments = layout.row_assignments.int()
            valid = assignments >= 0
            expected = torch.zeros(world * slots, width, device=device)
            expected[assignments[valid].long()] = reference[valid].float()
            dist.all_reduce(expected, group=group)
            expected[ids.flatten() < 0] = -19
            output.fill_(-19)
            epoch += 1
            peer_epoch_barrier(flags, flag_pointers, rank, epoch)
            queue = operation(
                packed,
                weights,
                layout.starts + layout.padded_sizes,
                assignments,
                outputs,
                pointers,
            )
            epoch += 1
            peer_epoch_barrier(flags, flag_pointers, rank, epoch)
            torch.testing.assert_close(
                output,
                expected[rank * slots : (rank + 1) * slots].bfloat16(),
                atol=0.005,
                rtol=0.005,
            )
            assert queue[0, 3].item() == 1
    return epoch


def run():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    group = dist.group.WORLD
    world = dist.get_world_size(group)
    capacity, width, topk, local_experts = 137, 513, 4, 4
    flags = symm_mem.empty(world, dtype=torch.int32, device=device)
    handle = symm_mem.rendezvous(flags, group=group)
    flag_views = [handle.get_buffer(r, flags.shape, flags.dtype) for r in range(world)]
    pointers = torch.tensor([t.data_ptr() for t in flag_views], device=device)
    flags.zero_()
    dist.barrier(group=group)
    epoch = 0
    results = []
    for dtype in (torch.bfloat16, torch.float32):
        source = symm_mem.empty(capacity, width, dtype=dtype, device=device)
        source_handle = symm_mem.rendezvous(source, group=group)
        sources = [
            source_handle.get_buffer(r, source.shape, dtype) for r in range(world)
        ]
        row = torch.arange(capacity, device=device)[:, None]
        col = torch.arange(width, device=device)[None, :]
        for compiled in (False, True):
            operation = (
                torch.compile(pack_peer_rows, fullgraph=True)
                if compiled
                else pack_peer_rows
            )
            for step in range(3):
                source.copy_(((row + col + rank * 7 + step * 3) % 32).to(dtype))
                source[capacity - rank * 11 - step * 3 :] = float("nan")
                epoch += 1
                peer_epoch_barrier(flag_views, pointers, rank, epoch)
                ids = torch.arange(world * capacity * topk, device=device).reshape(
                    world * capacity, topk
                ) % (world * local_experts)
                for peer in range(world):
                    end = (peer + 1) * capacity
                    missing = peer * 11 + step * 3
                    ids[end - missing : end] = -1
                ids[::7] = -1
                if step == 1:
                    ids[:] = -1
                local_ids = torch.where(
                    (ids >= rank * local_experts) & (ids < (rank + 1) * local_experts),
                    ids - rank * local_experts,
                    -1,
                )
                layout = DeviceExpertLayout.build(local_ids, local_experts, 128)
                actual = operation(sources, layout.row_assignments, topk)
                assignments = layout.row_assignments
                token = assignments // topk
                expected = (
                    token[:, None] % capacity
                    + col
                    + (token[:, None] // capacity) * 7
                    + step * 3
                ) % 32
                expected = torch.where(assignments[:, None] >= 0, expected, 0).to(dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                ready_out = torch.zeros_like(actual)
                queue = layout.chunk_descriptors()
                counters = torch.zeros(local_experts, device=device, dtype=torch.int32)
                ready_operation = (
                    torch.compile(pack_peer_rows_ready_out, fullgraph=True)
                    if compiled
                    else pack_peer_rows_ready_out
                )
                ready_operation(sources, assignments, topk, ready_out, queue, counters)
                torch.testing.assert_close(ready_out, expected, rtol=0, atol=0)
                torch.testing.assert_close(queue[:, 3], torch.ones_like(queue[:, 3]))
                torch.testing.assert_close(counters, torch.full_like(counters, 32))
                empty = operation(sources, assignments[:0], topk)
                assert empty.shape == (0, width)
                epoch += 1
                peer_epoch_barrier(flag_views, pointers, rank, epoch)
                results.append(dict(dtype=str(dtype), compiled=compiled, step=step))
        torch.cuda.synchronize()
    epoch = validate_ready_consumer(
        group, device, world, rank, flag_views, pointers, epoch
    )
    epoch = validate_peer_scatter(
        group, device, world, rank, flag_views, pointers, epoch
    )
    gathered = [None] * world
    dist.all_gather_object(
        gathered,
        dict(
            rank=rank, cases=results, concurrent_ready_consumer=True, peer_scatter=True
        ),
        group=group,
    )
    if rank == 0:
        print(
            json.dumps(
                dict(peer_packing=gathered, exact=True, gpu_epoch_barriers=True)
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
