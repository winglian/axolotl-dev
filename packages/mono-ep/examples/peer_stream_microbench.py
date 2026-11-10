# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Two-GPU microbench of the dispatch mode's transfers at the EP8 pattern.

Rank ``r`` talks to rank ``1 - r``, but the buffers are laid out as if there were eight
source ranks: the peer's ``x`` buffer is eight ``[cap, H]`` source slices and the pull
gathers the receive layout ``[8 * cap, H]`` from them (512 MB at the fused-benchmark
contract, 16384 x 2048 BF16 per source); the push returns the same rows into eight
``[cap, H]`` destination slices of the peer. Pull, push, the combine reduction and the
barriers are timed separately, for the legacy per-expert row kernels
(``pack_peer_rows`` / ``return_peer_rows``, the dispatch mode before this change) and the
streamed kernels (``peer_stream.py``) over a small grid of tile sizes and programs per SM,
against NCCL ``all_to_all_single`` of the same bytes and the plain Triton row copy of the
peer-gather microbench. Run under ``torch.distributed.run --nproc-per-node=2`` with
``--worker``.
"""

import argparse
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

WORLD_EMULATED, CAP, HIDDEN, TOPK, LOCAL_EXPERTS = 8, 16384, 2048, 8, 16
ROWS = WORLD_EMULATED * CAP
ROW_BYTES = HIDDEN * 2
WARMUP, MEASURED = 3, 15
NVLINK_ONE_WAY_GBPS = {"H100": 450.0, "B200": 900.0}
TILE_BYTES_GRID = (16384, 32768, 65536)
PROGRAMS_PER_SM_GRID = (2, 4, 8)


@triton.jit
def _copy_rows_i64(SRC, DST, rows, WIDTH: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    """The peer-gather microbench's plain row copy: int64 view, 16 B vectors, 8 rows per
    tile, grid-stride."""
    cols = tl.arange(0, WIDTH)
    for start in range(
        tl.program_id(0) * BLOCK_ROWS, rows, tl.num_programs(0) * BLOCK_ROWS
    ):
        r = start + tl.arange(0, BLOCK_ROWS)
        offs = r[:, None].to(tl.int64) * WIDTH + cols[None, :]
        mask = (r < rows)[:, None]
        tl.store(DST + offs, tl.load(SRC + offs, mask=mask), mask=mask)


def triton_copy_rows(src, dst, programs):
    src64 = src.view(torch.int64)
    dst64 = dst.view(torch.int64)
    _copy_rows_i64[(programs,)](
        src64, dst64, src64.shape[0], src64.shape[1], 8, num_warps=8
    )


def _timed(fn, group, participate=True, warmup=WARMUP, measured=MEASURED):
    samples = []
    for i in range(warmup + measured):
        dist.barrier(group)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if participate:
            fn()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            samples.append(start.elapsed_time(end))
    per_rank = [None] * dist.get_world_size(group)
    dist.all_gather_object(per_rank, (participate, samples), group=group)
    medians = {r: statistics.median(s) for r, (p, s) in enumerate(per_rank) if p}
    return {
        "median_ms": max(medians.values()),
        "per_rank_median_ms": [medians.get(r) for r in range(len(per_rank))],
        "iterations": measured,
    }


def _with_bandwidth(entry, peer_bytes, nominal):
    gbps = peer_bytes / (entry["median_ms"] / 1000) / 1e9
    entry["peer_bytes_per_active_rank"] = peer_bytes
    entry["gbps_per_active_rank"] = gbps
    entry["fraction_of_nominal_nvlink_one_way"] = gbps / nominal
    return entry


def _routing(kind, device, generator):
    """Receive-row routed mask ``[ROWS]`` and the token routing it comes from."""
    if kind == "all":
        return torch.ones(ROWS, dtype=torch.bool, device=device)
    # the contract's balanced top-8 over 128 experts: P(token hits this rank) = 1-(7/8)^8
    idx = torch.rand(
        ROWS, WORLD_EMULATED * LOCAL_EXPERTS, device=device, generator=generator
    ).argsort(-1)[:, :TOPK]
    return ((idx // LOCAL_EXPERTS) == 0).any(-1).contiguous()


def run(gpu):
    import mono_ep.peer_stream as peer_stream
    from mono_ep.peer_dispatch import SymmetricWorkspace, producer_mask
    from mono_ep.peer_packing import pack_peer_rows, return_peer_rows
    from mono_ep.peer_stream import pull_rows, push_rows, reduce_slots

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    group = dist.group.WORLD
    world = dist.get_world_size()
    assert world == 2, "this microbench is written for two ranks"
    peer = 1 - rank
    device = torch.device("cuda", rank)
    nominal = NVLINK_ONE_WAY_GBPS[gpu]
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    payload = {
        "benchmark": "peer_stream_microbench",
        "device": torch.cuda.get_device_name(),
        "sms": sms,
        "world_size": world,
        "emulated_world": WORLD_EMULATED,
        "capacity": CAP,
        "hidden": HIDDEN,
        "receive_rows": ROWS,
        "receive_bytes": ROWS * ROW_BYTES,
        "timing": "CUDA events per rank; dist.barrier + synchronize around each iteration; median over measured iterations; median_ms is the max over active ranks",
        "nominal_nvlink_one_way_gbps": nominal,
        "versions": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "python": sys.version.split()[0],
        },
        "stages_completed": [],
        "pull": {},
        "push": {},
        "reduce": {},
        "barrier": {},
        "parity": {},
    }

    def snapshot(stage):
        payload["stages_completed"].append(stage)
        if rank == 0:
            print("RESULT " + json.dumps(payload), flush=True)

    generator = torch.Generator(device=device).manual_seed(100 + rank)
    workspace = SymmetricWorkspace(
        group,
        device,
        {
            "x": ((ROWS, HIDDEN), torch.bfloat16),
            "y": ((ROWS, HIDDEN), torch.bfloat16),
        },
    )
    x_views, y_views = workspace.views["x"], workspace.views["y"]
    x_views[rank].copy_(torch.randn(ROWS, HIDDEN, device=device, generator=generator))
    torch.cuda.synchronize()
    workspace.barrier()
    # eight emulated sources, all on the peer (every row crosses NVLink, as the 5.2 copies)
    peer_sources = [
        x_views[peer][s * CAP : (s + 1) * CAP] for s in range(WORLD_EMULATED)
    ]
    # four on the peer, four local: half the bytes cross
    mixed_sources = [
        (x_views[peer] if s % 2 == 0 else x_views[rank])[s * CAP : (s + 1) * CAP]
        for s in range(WORLD_EMULATED)
    ]
    peer_destinations = [
        y_views[peer][s * CAP : (s + 1) * CAP] for s in range(WORLD_EMULATED)
    ]
    receive_index = torch.arange(ROWS, device=device, dtype=torch.int64)
    payload_rows = torch.randn(ROWS, HIDDEN, device=device, generator=generator).to(
        torch.bfloat16
    )
    out = torch.empty(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)

    def configure(tile_bytes, programs_per_sm):
        peer_stream.TILE_BYTES = tile_bytes
        return {"tile_bytes": tile_bytes, "programs_per_sm": programs_per_sm}

    # ---- pull ----
    def _pull_case(routing):
        routed = _routing(routing, device, generator)
        routed_rows = int(routed.sum())
        peer_bytes = routed_rows * ROW_BYTES
        assignments = torch.where(
            routed, receive_index, torch.full_like(receive_index, -1)
        )
        section = payload["pull"][routing] = {
            "routed_rows": routed_rows,
            "routed_fraction": routed_rows / ROWS,
            "peer_bytes": peer_bytes,
            "variants": {},
        }
        variants = section["variants"]
        legacy = pack_peer_rows(peer_sources, assignments, 1)
        for direction, participate in (("both_ways", True), ("one_way", rank == 0)):
            variants[f"legacy_pack_peer_rows/{direction}"] = _with_bandwidth(
                _timed(
                    lambda: pack_peer_rows(peer_sources, assignments, 1),
                    group,
                    participate,
                ),
                peer_bytes,
                nominal,
            )
        for tile_bytes in TILE_BYTES_GRID:
            for programs_per_sm in PROGRAMS_PER_SM_GRID:
                config = configure(tile_bytes, programs_per_sm)
                streamed = pull_rows(
                    peer_sources, routed, 0, out=out, programs_per_sm=programs_per_sm
                )
                equal = bool(torch.equal(streamed, legacy))
                for direction, participate in (
                    ("both_ways", True),
                    ("one_way", rank == 0),
                ):
                    if direction == "one_way" and (
                        tile_bytes != 32768 or programs_per_sm != 4
                    ):
                        continue
                    entry = _with_bandwidth(
                        _timed(
                            lambda p=programs_per_sm: pull_rows(
                                peer_sources, routed, 0, out=out, programs_per_sm=p
                            ),
                            group,
                            participate,
                        ),
                        peer_bytes,
                        nominal,
                    )
                    entry.update(config)
                    entry["bitwise_equal_to_legacy"] = equal
                    variants[
                        f"streamed/tile{tile_bytes // 1024}k/ppsm{programs_per_sm}/{direction}"
                    ] = entry
        config = configure(32768, 4)
        entry = _with_bandwidth(
            _timed(
                lambda: pull_rows(mixed_sources, routed, 0, out=out, programs_per_sm=4),
                group,
            ),
            peer_bytes // 2,
            nominal,
        )
        entry.update(config)
        entry["note"] = (
            "four of the eight sources local; peer_bytes counts the crossing half"
        )
        variants["streamed/tile32k/ppsm4/mixed_sources/both_ways"] = entry
        if routing == "contract":
            prefix = torch.cumsum(routed, 0, dtype=torch.int64)
            compact, slots = pull_rows(
                peer_sources, routed, 0, prefix=prefix, count=routed_rows
            )
            section["compact_bitwise_equal_to_legacy_rows"] = bool(
                torch.equal(compact, legacy[routed])
            ) and bool(torch.equal(slots, routed.nonzero().squeeze(1)))
            entry = _with_bandwidth(
                _timed(
                    lambda: pull_rows(
                        peer_sources, routed, 0, prefix=prefix, count=routed_rows
                    ),
                    group,
                ),
                peer_bytes,
                nominal,
            )
            entry.update(config)
            variants["streamed_compact/tile32k/ppsm4/both_ways"] = entry
        # NCCL and plain copies of the same bytes
        send = payload_rows[:routed_rows]
        recv = out[:routed_rows]
        splits_in = [0, routed_rows] if rank == 0 else [routed_rows, 0]
        splits_out = splits_in

        def a2a():
            dist.all_to_all_single(recv, send, splits_out, splits_in, group=group)

        variants["nccl_all_to_all_single_same_bytes/both_ways"] = _with_bandwidth(
            _timed(a2a, group), peer_bytes, nominal
        )
        for direction, participate in (("both_ways", True), ("one_way", rank == 0)):
            variants[f"triton_plain_row_copy_same_bytes/{direction}"] = _with_bandwidth(
                _timed(
                    lambda: triton_copy_rows(
                        x_views[peer][:routed_rows], out[:routed_rows], sms * 4
                    ),
                    group,
                    participate,
                ),
                peer_bytes,
                nominal,
            )
        variants["torch_copy_same_bytes/both_ways"] = _with_bandwidth(
            _timed(lambda: out[:routed_rows].copy_(x_views[peer][:routed_rows]), group),
            peer_bytes,
            nominal,
        )
        snapshot(f"pull_{routing}")

    for routing in ("all", "contract"):
        _pull_case(routing)

    # ---- push ----
    def _push_case(routing):
        routed = _routing(routing, device, generator)
        routed_rows = int(routed.sum())
        peer_bytes = routed_rows * ROW_BYTES
        # legacy: receive row (s, t) -> destination s, slot t (rank's slot range starts at 0
        # in these [cap, H] destination slices)
        assignments = torch.where(
            routed, receive_index, torch.full_like(receive_index, -1)
        )
        score_sink = torch.zeros(CAP, device=device)
        zero_scores = torch.zeros(ROWS, device=device)
        section = payload["push"][routing] = {
            "routed_rows": routed_rows,
            "peer_bytes": peer_bytes,
            "variants": {},
        }
        variants = section["variants"]

        def legacy_push():
            return_peer_rows(
                payload_rows,
                zero_scores,
                assignments,
                peer_destinations,
                [score_sink] * WORLD_EMULATED,
            )

        y_views[peer].zero_()
        torch.cuda.synchronize()
        workspace.barrier()
        legacy_push()
        torch.cuda.synchronize()
        workspace.barrier()
        legacy_result = y_views[rank].clone()
        workspace.barrier()
        variants["legacy_return_peer_rows/both_ways"] = _with_bandwidth(
            _timed(legacy_push, group), peer_bytes, nominal
        )
        for tile_bytes in TILE_BYTES_GRID:
            for programs_per_sm in PROGRAMS_PER_SM_GRID:
                config = configure(tile_bytes, programs_per_sm)
                y_views[peer].zero_()
                torch.cuda.synchronize()
                workspace.barrier()
                push_rows(
                    payload_rows,
                    peer_destinations,
                    routed,
                    0,
                    capacity=CAP,
                    programs_per_sm=programs_per_sm,
                )
                torch.cuda.synchronize()
                workspace.barrier()
                equal = bool(torch.equal(y_views[rank], legacy_result))
                workspace.barrier()
                entry = _with_bandwidth(
                    _timed(
                        lambda p=programs_per_sm: push_rows(
                            payload_rows,
                            peer_destinations,
                            routed,
                            0,
                            capacity=CAP,
                            programs_per_sm=p,
                        ),
                        group,
                    ),
                    peer_bytes,
                    nominal,
                )
                entry.update(config)
                entry["bitwise_equal_to_legacy"] = equal
                variants[
                    f"streamed/tile{tile_bytes // 1024}k/ppsm{programs_per_sm}/both_ways"
                ] = entry
        if routing == "contract":
            config = configure(32768, 4)
            slots = routed.nonzero().squeeze(1)
            compact_payload = payload_rows[slots].contiguous()
            y_views[peer].zero_()
            torch.cuda.synchronize()
            workspace.barrier()
            push_rows(compact_payload, peer_destinations, slots, 0, capacity=CAP)
            torch.cuda.synchronize()
            workspace.barrier()
            section["compact_bitwise_equal_to_legacy"] = bool(
                torch.equal(y_views[rank], legacy_result)
            )
            workspace.barrier()
            entry = _with_bandwidth(
                _timed(
                    lambda: push_rows(
                        compact_payload, peer_destinations, slots, 0, capacity=CAP
                    ),
                    group,
                ),
                peer_bytes,
                nominal,
            )
            entry.update(config)
            variants["streamed_compact/tile32k/ppsm4/both_ways"] = entry
        snapshot(f"push_{routing}")

    for routing in ("all", "contract"):
        _push_case(routing)
    configure(32768, 4)

    # ---- combine reduction over the own return buffer (local) ----
    tokens = CAP
    torch.manual_seed(7 + rank)
    topk_idx = torch.rand(
        tokens, WORLD_EMULATED * LOCAL_EXPERTS, device=device
    ).argsort(-1)[:, :TOPK]
    mask = producer_mask(topk_idx, LOCAL_EXPERTS, WORLD_EMULATED)
    buffer = y_views[rank]
    buffer.copy_(torch.randn(ROWS, HIDDEN, device=device, generator=generator))

    def legacy_collect(dtype):
        view = buffer.view(WORLD_EMULATED, CAP, HIDDEN)[:, :tokens]
        total = torch.zeros((tokens, HIDDEN), device=device, dtype=torch.float32)
        for p in range(WORLD_EMULATED):
            total += view[p].float().masked_fill(~mask[p].unsqueeze(-1), 0)
        return total.to(dtype)

    reduce = payload["reduce"]
    reduce["tokens"] = tokens
    reduce["bytes_read"] = ROWS * ROW_BYTES
    reduce["producer_mask_fraction"] = float(mask.float().mean())
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        legacy_out = legacy_collect(dtype)
        fused_out = reduce_slots(buffer, mask, WORLD_EMULATED, dtype)
        reduce[f"bitwise_equal/{dtype_name}"] = bool(torch.equal(legacy_out, fused_out))
        reduce[f"legacy_w_pass/{dtype_name}"] = _timed(
            lambda d=dtype: legacy_collect(d), group
        )
        reduce[f"fused_reduce_slots/{dtype_name}"] = _timed(
            lambda d=dtype: reduce_slots(buffer, mask, WORLD_EMULATED, d), group
        )
    reduce["producer_mask_torch"] = _timed(
        lambda: producer_mask(topk_idx, LOCAL_EXPERTS, WORLD_EMULATED), group
    )
    snapshot("reduce")

    # ---- barriers ----
    payload["barrier"]["symmetric_memory_barrier"] = _timed(workspace.barrier, group)
    payload["barrier"]["nccl_dist_barrier"] = _timed(lambda: dist.barrier(group), group)
    payload["barrier"]["note"] = (
        "two ranks; the dispatch mode issues one per op (two per layer forward, two per "
        "layer backward)"
    )
    snapshot("done")
    workspace.destroy()
    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", choices=("H100", "B200"), required=True)
    args = parser.parse_args()
    if not args.worker:
        parser.error("run under torch.distributed.run with --worker")
    run(args.gpu)
