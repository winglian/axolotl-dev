// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <cuda_bf16.h>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm::epilogue {

struct MoEBackwardAtom {
    nv_bfloat162 h[2];
    nv_bfloat162 gate[2];
    nv_bfloat162 up[2];
    float2 dscore;
};

// Matches the persistent forward's 16dp256b TMEM atom: two rows per lane,
// with gate/up pairs (0,2), (1,3), (4,6), (5,7).
CUTLASS_DEVICE MoEBackwardAtom moe_backward_atom(
    const uint32_t (&z)[8], const float (&dh)[4], const float2 scores) {
    MoEBackwardAtom result;
    result.dscore = {0.0f, 0.0f};
    #pragma unroll
    for (uint32_t col = 0; col < 2; ++col) {
        float h[2], dg[2], du[2];
        #pragma unroll
        for (uint32_t row = 0; row < 2; ++row) {
            const float gate = __bfloat162float(__float2bfloat16_rn(
                __uint_as_float(z[col * 4 + row])));
            const float up = __bfloat162float(__float2bfloat16_rn(
                __uint_as_float(z[col * 4 + row + 2])));
            const float grad = __bfloat162float(__float2bfloat16_rn(dh[col * 2 + row]));
            const float score = row == 0 ? scores.x : scores.y;
            const float sigmoid = 1.0f / (1.0f + expf(-gate));
            const float silu = gate * sigmoid;
            h[row] = silu * up * score;
            dg[row] = grad * score * up * (sigmoid + silu * (1.0f - sigmoid));
            du[row] = grad * score * silu;
            const float ds = grad * silu * up;
            if (row == 0) result.dscore.x += ds;
            else result.dscore.y += ds;
        }
        result.h[col] = __float22bfloat162_rn(make_float2(h[0], h[1]));
        result.gate[col] = __float22bfloat162_rn(make_float2(dg[0], dg[1]));
        result.up[col] = __float22bfloat162_rn(make_float2(du[0], du[1]));
    }
    return result;
}

template <uint32_t BlockM>
struct alignas(128) MoEBackwardTileEpilogue {
    // BF16 dH stays CTA-local across the two gate/up recompute tiles.
    nv_bfloat16 dh[BlockM * 128];

    CUTLASS_DEVICE void retain_dh_atom(const uint32_t (&values)[8],
                                      uint32_t atom_row, uint32_t warp,
                                      uint32_t lane, uint32_t valid_m) {
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            const auto row = atom_row + (lane % 4) * 2 + i % 2;
            const auto col = warp * 32 + lane / 4 + (i / 2) * 8;
            dh[row * 128 + col] = row < valid_m
                ? __float2bfloat16_rn(__uint_as_float(values[i]))
                : __float2bfloat16_rn(0.0f);
        }
    }

    CUTLASS_DEVICE MoEBackwardAtom recompute_atom(
        const uint32_t (&z)[8], uint32_t atom_row, uint32_t warp,
        uint32_t lane, uint32_t half, const float2 scores) const {
        float gradients[4];
        #pragma unroll
        for (uint32_t col = 0; col < 2; ++col) {
            #pragma unroll
            for (uint32_t row = 0; row < 2; ++row) {
                const auto m = atom_row + (lane % 4) * 2 + row;
                const auto n = half * 64 + warp * 16 + lane / 4 + col * 8;
                gradients[col * 2 + row] = __bfloat162float(dh[m * 128 + n]);
            }
        }
        return moe_backward_atom(z, gradients, scores);
    }
};

} // namespace deep_gemm::epilogue
