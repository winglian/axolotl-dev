// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include "backward_schedule.cuh"
#include "sm100_backward_tile.cuh"

namespace mono_ep {

struct alignas(1024) OwnedBackwardStorage {
    Sm100BackwardTile mma;
    float dh[64 * 64], gate_up[64 * 128];
    float ds[64];
    unsigned int block;
    BackwardExpertTile tile;
};

// Packed operands are an interim boundary; ownership spans both math phases.
__global__ void sm100_owned_backward(
    const __nv_bfloat16* x, const __nv_bfloat16* dy,
    const __nv_bfloat16* w1, const __nv_bfloat16* w2, const float* scores,
    const unsigned int* block_ends, const unsigned int* starts,
    const unsigned int* counts, unsigned int* next_block,
    __nv_bfloat16* h, __nv_bfloat16* dz, __nv_bfloat16* dx, float* ds,
    unsigned int experts, unsigned int hidden, unsigned int inner) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    extern __shared__ __align__(1024) unsigned char storage[];
    auto& shared = *reinterpret_cast<OwnedBackwardStorage*>(storage);
    shared.mma.initialize();
    const auto total_blocks = block_ends[experts - 1];
    while (true) {
        if (threadIdx.x == 0) {
            shared.block = atomicAdd(next_block, 1u);
            shared.tile = resolve_backward_tile(shared.block, block_ends, starts,
                                                counts, experts, 64);
        }
        __syncthreads();
        if (shared.block >= total_blocks) break;
        const auto tile = shared.tile;
        if (threadIdx.x < 64) shared.ds[threadIdx.x] = 0;
        __syncthreads();
        for (unsigned int j = 0; j < inner; j += 64) {
            shared.mma.multiply(hidden,
                [&](unsigned int row, unsigned int k) {
                    return row < tile.valid_rows ? dy[(tile.packed_row + row) * hidden + k]
                                                 : __float2bfloat16(0);
                },
                [&](unsigned int col, unsigned int k) {
                    return w2[(tile.expert * hidden + k) * inner + j + col];
                }, shared.dh);
            auto load_x = [&](unsigned int row, unsigned int k) {
                return row < tile.valid_rows ? x[(tile.packed_row + row) * hidden + k]
                                             : __float2bfloat16(0);
            };
            shared.mma.multiply<128>(hidden, load_x,
                [&](unsigned int col, unsigned int k) {
                    const auto projection = j + col / 2 + (col % 2) * inner;
                    return w1[(tile.expert * 2 * inner + projection) * hidden + k];
                }, shared.gate_up);
            if (threadIdx.x < tile.valid_rows) {
                const auto row = threadIdx.x, packed = tile.packed_row + row;
                const float score = scores[packed];
                float partial = 0;
                for (unsigned int col = 0; col < 64; ++col) {
                    const auto offset = row * 64 + col;
                    const float gate = __bfloat162float(__float2bfloat16_rn(shared.gate_up[row * 128 + col * 2]));
                    const float up = __bfloat162float(__float2bfloat16_rn(shared.gate_up[row * 128 + col * 2 + 1]));
                    const float dh = __bfloat162float(__float2bfloat16_rn(shared.dh[offset]));
                    const float sigmoid = 1.0f / (1.0f + expf(-gate));
                    const float silu = gate * sigmoid;
                    h[packed * inner + j + col] = __float2bfloat16_rn(silu * up * score);
                    dz[packed * 2 * inner + j + col] = __float2bfloat16_rn(
                        dh * score * up * (sigmoid + silu * (1.0f - sigmoid)));
                    dz[packed * 2 * inner + inner + j + col] = __float2bfloat16_rn(dh * score * silu);
                    partial += dh * silu * up;
                }
                shared.ds[row] += partial;
            }
            __syncthreads();
        }
        __threadfence();
        __syncthreads();
        for (unsigned int j = 0; j < hidden; j += 64) {
            shared.mma.multiply(2 * inner,
                [&](unsigned int row, unsigned int k) {
                    return row < tile.valid_rows ? dz[(tile.packed_row + row) * 2 * inner + k]
                                                 : __float2bfloat16(0);
                },
                [&](unsigned int col, unsigned int k) {
                    return w1[(tile.expert * 2 * inner + k) * hidden + j + col];
                }, shared.gate_up);
            for (unsigned int i = threadIdx.x; i < tile.valid_rows * 64; i += blockDim.x)
                dx[(tile.packed_row + i / 64) * hidden + j + i % 64] =
                    __float2bfloat16_rn(shared.gate_up[i]);
            __syncthreads();
        }
        if (threadIdx.x < tile.valid_rows)
            ds[tile.packed_row + threadIdx.x] = shared.ds[threadIdx.x];
        __syncthreads();
    }
    shared.mma.release();
#endif
}

} // namespace mono_ep
