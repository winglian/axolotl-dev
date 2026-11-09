// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <cuda/atomic>
#include <cstdint>

namespace mono_ep {

struct alignas(16) BackwardSlot {
    unsigned long long ticket;
    unsigned int completed;
    unsigned int padding;
};

struct BackwardTask {
    unsigned int block, phase, tile, slot;
};

struct BackwardExpertTile {
    unsigned int expert, packed_row, valid_rows;
};

// block_ends is the exclusive prefix sum of ceil(tokens_per_expert / block_m).
// Keeping token counts and packed starts separate supports alignment padding.
__device__ inline BackwardExpertTile resolve_backward_tile(
    unsigned int block, const unsigned int* block_ends,
    const unsigned int* packed_starts, const unsigned int* token_counts,
    unsigned int experts, unsigned int block_m) {
    unsigned int begin = 0, end = experts;
    while (begin < end) {
        const auto middle = begin + (end - begin) / 2;
        if (block_ends[middle] <= block)
            begin = middle + 1;
        else
            end = middle;
    }
    if (begin == experts)
        return {experts, 0, 0};
    const auto first_block = begin ? block_ends[begin - 1] : 0;
    const auto row = (block - first_block) * block_m;
    const auto remaining = token_counts[begin] - row;
    return {begin, packed_starts[begin] + row,
            remaining < block_m ? remaining : block_m};
}

// Requires capacity > 0 and 0 < tiles[phase] <= tile_mask; initialize before use.
// One elected thread per CTA calls claim/complete. Tile writers must publish all
// output stores and join the elected thread before complete. dW operands that
// outlive dX must be retained outside this reusable ring by the executor.
struct BackwardSchedule {
    BackwardSlot* slots;
    unsigned int capacity, blocks;
    unsigned int tiles[3];
    static constexpr unsigned int tile_mask = (1u << 30) - 1;
    using Ticket = cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>;
    using Count = cuda::atomic_ref<unsigned int, cuda::thread_scope_device>;

    __device__ static unsigned long long encode(unsigned int block,
                                                unsigned int phase,
                                                unsigned int tile = 0) {
        return (static_cast<unsigned long long>(block) << 32) |
               (static_cast<unsigned long long>(phase) << 30) | tile;
    }

    __device__ void initialize(unsigned int slot) const {
        slots[slot].completed = 0;
        slots[slot].ticket = encode(slot, 0);
    }

    __device__ bool claim(unsigned int& cursor, BackwardTask& task) const {
        for (unsigned int i = 0; i < capacity; ++i) {
            const auto slot = (cursor + i) % capacity;
            auto ticket = Ticket(slots[slot].ticket);
            auto state = ticket.load(cuda::memory_order_acquire);
            const auto block = static_cast<unsigned int>(state >> 32);
            const auto phase = static_cast<unsigned int>((state >> 30) & 3);
            const auto tile = static_cast<unsigned int>(state) & tile_mask;
            if (block >= blocks || phase == 3 || tile >= tiles[phase])
                continue;
            // Generation and phase participate in CAS, preventing ring-reuse ABA.
            if (ticket.compare_exchange_strong(state, state + 1,
                                               cuda::memory_order_acq_rel,
                                               cuda::memory_order_acquire)) {
                task = {block, phase, tile, slot};
                cursor = (slot + 1) % capacity;
                return true;
            }
        }
        return false;
    }

    __device__ void complete(const BackwardTask& task) const {
        auto& slot = slots[task.slot];
        if (Count(slot.completed).fetch_add(1, cuda::memory_order_acq_rel) + 1 !=
            tiles[task.phase])
            return;
        Count(slot.completed).store(0, cuda::memory_order_relaxed);
        const auto next_block = task.phase == 2
            ? (blocks - task.block <= capacity ? blocks : task.block + capacity)
            : task.block;
        const auto next_phase = task.phase == 2 ? 0u : task.phase + 1;
        Ticket(slot.ticket).store(encode(next_block, next_phase),
                                  cuda::memory_order_release);
    }

    __device__ bool finished() const {
        for (unsigned int slot = 0; slot < capacity; ++slot)
            if ((Ticket(slots[slot].ticket).load(cuda::memory_order_acquire) >> 32)
                < blocks)
                return false;
        return true;
    }
};

} // namespace mono_ep
