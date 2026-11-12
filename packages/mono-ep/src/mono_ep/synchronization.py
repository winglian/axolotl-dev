# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-node peer barriers with monotonic system-scope epochs."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["EPOCH"])
def _epoch_barrier(
    POINTERS, LOCAL, RANK: tl.constexpr, WORLD: tl.constexpr, EPOCH, BLOCK: tl.constexpr
):
    lane = tl.arange(0, BLOCK)
    address = tl.load(POINTERS + lane, lane < WORLD, 0)
    peer = address.to(tl.pointer_type(tl.int32))
    tl.atomic_xchg(peer + RANK, EPOCH, lane < WORLD, sem="release", scope="sys")
    pointer = LOCAL + tl.minimum(lane, WORLD - 1)
    started = tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;",
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    ready = tl.full((BLOCK,), 0, tl.int32)
    while tl.min(ready, 0) < EPOCH:
        ready = tl.inline_asm_elementwise(
            "ld.acquire.sys.global.u32 $0, [$1];",
            constraints="=r,l",
            args=[pointer],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        now = tl.inline_asm_elementwise(
            "nanosleep.u32 64; mov.u64 $0, %clock64;",
            constraints="=l",
            args=[],
            dtype=tl.uint64,
            is_pure=False,
            pack=1,
        )
        tl.device_assert(now - started < 30_000_000_000, "Peer epoch barrier timeout")


@torch.library.custom_op("mono_ep::peer_epoch_barrier", mutates_args=("flags",))
def peer_epoch_barrier(
    flags: list[torch.Tensor], pointers: torch.Tensor, rank: int, epoch: int
) -> None:
    """Publish this rank's epoch to each peer, then acquire every peer's epoch.

    Every rank must call epochs in the same order on the same process group.
    Flags start at zero and must not be reset or reused by another schedule.
    Each caller stream must join all local work before entering this barrier.
    """
    _epoch_barrier_fake(flags, pointers, rank, epoch)
    _epoch_barrier[(1,)](
        pointers,
        flags[rank],
        rank,
        len(flags),
        epoch,
        triton.next_power_of_2(len(flags)),
        num_warps=4,
        debug=True,
    )


@peer_epoch_barrier.register_fake
def _epoch_barrier_fake(flags, pointers, rank, epoch):
    world = len(flags)
    if not 0 <= rank < world or not 0 < epoch < 2**31:
        raise ValueError("Invalid peer rank or positive INT32 epoch")
    if (
        pointers.shape != (world,)
        or pointers.dtype != torch.int64
        or pointers.device.type != "cuda"
        or not pointers.is_contiguous()
    ):
        raise ValueError("Expected contiguous CUDA INT64 peer-pointer table")
    if any(
        t.shape != (world,)
        or t.dtype != torch.int32
        or t.device != pointers.device
        or not t.is_contiguous()
        for t in flags
    ):
        raise ValueError("Expected contiguous INT32 epoch flags for each peer")
