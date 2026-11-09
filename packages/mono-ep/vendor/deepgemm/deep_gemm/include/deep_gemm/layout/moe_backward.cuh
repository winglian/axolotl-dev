// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once
#include <cuda_bf16.h>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm::layout {
struct MoEBackwardBuffers {
    SymBuffer<> dy;
    SymBuffer<> dscore_partials;
    __nv_bfloat16 *x_pool, *dy_pool, *h_pool, *dz_pool;
    float* dscore;
    uint32_t *expert_starts, *expert_counts;
    uint8_t *qx_pool = nullptr, *qdy_pool = nullptr;
    float *sx_pool = nullptr, *sdy_pool = nullptr;
    uint8_t* qdz_pool = nullptr;
    float* sdz_pool = nullptr;
    uint8_t *qx_col = nullptr, *qdy_col = nullptr, *qh_col = nullptr, *qdz_col = nullptr;
    float *sx_col = nullptr, *sdy_col = nullptr, *sh_col = nullptr, *sdz_col = nullptr;
    uint32_t* expert_padded_counts = nullptr;
    SymBuffer<> dy_sf;
};
} // namespace deep_gemm::layout
