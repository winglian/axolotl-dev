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

// NVFP4 backward: `[K / 64][pool rows]` UE4M3 scale words for the packed `qx/qdy/qdz` pools and the
// per-row dy/dz scales. A separate type so the kernels taking `MoEBackwardBuffers` keep their
// parameter layout.
struct MoEBackwardNvfp4Buffers : MoEBackwardBuffers {
    uint32_t *nv_sx = nullptr, *nv_sdy = nullptr, *nv_sdz = nullptr;
    float *nv_rdy = nullptr, *nv_rdz = nullptr;
    uint32_t nv_pool_rows = 0;
};
} // namespace deep_gemm::layout
