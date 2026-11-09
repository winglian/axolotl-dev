# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate GPU-side peer combine completion with concurrent producers."""

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mono_ep.combine import combine_wait_copy, peer_backward_scatter_publish


def scatter(*args):
    peer_backward_scatter_publish(*args)


def consume(*args):
    combine_wait_copy(*args)


def run(compiled):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 2
    tokens, hidden = 137, 2048
    accumulation = symm_mem.empty(tokens, hidden, device="cuda", dtype=torch.float32)
    queue = symm_mem.empty(2, 4, device="cuda", dtype=torch.int32)
    ah = symm_mem.rendezvous(accumulation, group=dist.group.WORLD)
    qh = symm_mem.rendezvous(queue, group=dist.group.WORLD)
    peer_accumulation = ah.get_buffer(1 - rank, (tokens, hidden), torch.float32)
    peer_queue = qh.get_buffer(1 - rank, (2, 4), torch.int32)
    score_acc = symm_mem.empty(tokens * 2, 1, device="cuda", dtype=torch.float32)
    score_queue = symm_mem.empty(2, 4, device="cuda", dtype=torch.int32)
    sh = symm_mem.rendezvous(score_acc, group=dist.group.WORLD)
    sqh = symm_mem.rendezvous(score_queue, group=dist.group.WORLD)
    peer_score_acc = sh.get_buffer(1 - rank, (tokens * 2, 1), torch.float32)
    peer_score_queue = sqh.get_buffer(1 - rank, (2, 4), torch.int32)
    score_stream = torch.cuda.Stream()
    consumer_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    cases = (
        (torch.bfloat16, torch.float32, "normal"),
        (torch.float32, torch.float32, "normal"),
        (torch.bfloat16, torch.bfloat16, "normal"),
        (torch.bfloat16, torch.float32, "empty"),
        (torch.bfloat16, torch.float32, "masked"),
        (torch.float32, torch.float32, "large"),
    )
    for source_dtype, output_dtype, kind in cases:
        if compiled:
            torch.compiler.reset()
        producer = (
            torch.compile(scatter, fullgraph=True, dynamic=False)
            if compiled
            else scatter
        )
        consumer = (
            torch.compile(consume, fullgraph=True, dynamic=False)
            if compiled
            else consume
        )
        torch.manual_seed(831 + rank)
        rows = 0 if kind == "empty" else 8193 if kind == "large" else 513
        source = torch.randn(rows, hidden, device="cuda", dtype=source_dtype)
        indices = torch.randint(
            0, tokens - 5, (source.shape[0],), device="cuda", dtype=torch.int32
        )
        indices[::7] = -1
        if kind == "masked":
            indices.fill_(-1)
        scores = torch.randn(source.shape[0], device="cuda")
        score_ids = torch.randint(
            0, tokens * 2 - 5, (source.shape[0],), device="cuda", dtype=torch.int32
        )
        score_ids[::5] = -1
        if kind == "masked":
            score_ids.fill_(-1)
        expected_scores = torch.zeros_like(score_acc)
        score_valid = score_ids >= 0
        expected_scores[:, 0].index_add_(
            0, score_ids[score_valid].long(), scores[score_valid]
        )
        dist.all_reduce(expected_scores)
        score_output = torch.empty_like(score_acc)
        expected = torch.zeros_like(accumulation)
        valid = indices >= 0
        expected.index_add_(0, indices[valid].long(), source[valid].float())
        dist.all_reduce(expected)
        output = torch.empty((tokens, hidden), device="cuda", dtype=output_dtype)
        accumulation.zero_()
        queue.zero_()
        score_acc.zero_()
        score_queue.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        producer(
            source,
            indices,
            scores,
            score_ids,
            accumulation,
            score_acc,
            queue,
            score_queue,
            rank,
        )
        producer(
            source,
            indices,
            scores,
            score_ids,
            peer_accumulation,
            peer_score_acc,
            peer_queue,
            peer_score_queue,
            rank,
        )
        torch.cuda.synchronize()
        dist.barrier()
        consumer(accumulation, output, queue)
        consumer(score_acc, score_output, score_queue)
        torch.cuda._sleep(1)
        torch.cuda.synchronize()
        dist.barrier()
        errors = []
        score_errors = []
        for _ in range(3):
            accumulation.zero_()
            queue.zero_()
            score_acc.zero_()
            score_queue.zero_()
            score_output.fill_(7)
            output.fill_(7)
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()
            with torch.cuda.stream(consumer_stream):
                consumer(accumulation, output, queue)
            with torch.cuda.stream(score_stream):
                consumer(score_acc, score_output, score_queue)
            with torch.cuda.stream(producer_stream):
                torch.cuda._sleep(10_000_000 * (rank + 1))
                producer(
                    source,
                    indices,
                    scores,
                    score_ids,
                    peer_accumulation,
                    peer_score_acc,
                    peer_queue,
                    peer_score_queue,
                    rank,
                )
                producer(
                    source,
                    indices,
                    scores,
                    score_ids,
                    accumulation,
                    score_acc,
                    queue,
                    score_queue,
                    rank,
                )
            torch.cuda.synchronize()
            error = (
                output.float() - expected.to(output_dtype).float()
            ).norm() / expected.float().norm().clamp_min(1e-8)
            assert torch.isfinite(error) and error < (
                0.004 if output_dtype == torch.bfloat16 else 1e-5
            ), error
            score_error = (
                score_output - expected_scores
            ).norm() / expected_scores.norm().clamp_min(1e-8)
            assert torch.isfinite(score_error) and score_error < 1e-5, score_error
            assert (score_output[-5:] == 0).all()
            score_errors.append(score_error.item())
            assert (output[-5:] == 0).all()
            assert (queue[:, 2] == max(1, (rows + 15) // 16)).all()
            assert (queue[:, 3] == 1).all() and (score_queue[:, 3] == 1).all()
            if kind in ("empty", "masked"):
                assert (output == 0).all()
            errors.append(error.item())
            dist.barrier()
        print(
            json.dumps(
                dict(
                    rank=rank,
                    compiled=compiled,
                    source_dtype=str(source_dtype),
                    output_dtype=str(output_dtype),
                    case=kind,
                    rows=rows,
                    relative_l2=errors,
                    score_relative_l2=score_errors,
                )
            ),
            flush=True,
        )
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    run(parser.parse_args().compiled)
