// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <cstdint>

namespace mono_ep_requant {

// Launch arguments of the SM100 packed requant (nvfp4_requant_launch.cu).
struct Args {
    const uint8_t* q;      // u8 [E, N, K/2] packed base
    const uint8_t* s;      // u8 [E, N, K/16] e4m3 base block scales
    const float* dq_pts;   // fp32 [E, P] base per-tensor scales (ones if none)
    const void* a;         // [E, R, K] LoRA A, bf16 or fp32
    const void* b;         // [E, N, R] LoRA B, same dtype
    const float* fq_pts;   // fp32 [E, P] output per-tensor scales (ones if none)
    uint8_t* oq;           // u8 [E_local, N, K/2]
    uint8_t* os;           // u8 [E_local, N, K/16]
    uint8_t* osfa;         // u8 [E_local, RM, RK, 512] or null
    uint8_t* oq_t;         // fused: u8 [E_local, K, N/2]
    uint8_t* os_t;         // fused: u8 [E_local, K, N/16]
    uint8_t* osfa_t;       // fused: u8 [E_local, RM_T, RK_T, 512] or null
    float scaling;
    int e0, e_local, n, k, r, r_pad, p, rows_per_proj;
    int rm, rk, rm_t, rk_t;
    int interleave;        // 0: concat rows; g > 0: gate/up interleave of granularity g
    int round_bf16, modelopt, reuse, ab_fp32;
    int fused, ffma2;
    int mode;              // 3: requant; 0: copy; 1: no delta; 2: no quantizer (bound probes)
    int k_splits;
};

}  // namespace mono_ep_requant
