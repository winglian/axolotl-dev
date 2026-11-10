// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>

namespace deep_gemm {

// Round a non-negative FP32 scale to UE4M3 (RNE, saturating at 448); returns the byte and the decoded value
CUTLASS_DEVICE uint32_t nvfp4_encode_e4m3_scale(const float& scale, float& decoded) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(0.0f), "f"(scale));
    uint32_t half2_bits;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(half2_bits) : "h"(packed));
    decoded = __low2float(*reinterpret_cast<const __half2*>(&half2_bits));
    return packed & 0xffu;
}

// Two FP32 values to one packed E2M1 byte (RNE, saturating at 6): the even element takes the low nibble
CUTLASS_DEVICE uint32_t nvfp4_pack_pair(const float& even, const float& odd) {
    uint16_t out;
    asm("{\n\t.reg .b8 byte;\n\tcvt.rn.satfinite.e2m1x2.f32 byte, %1, %2;\n\tcvt.u16.u8 %0, byte;\n\t}"
        : "=h"(out) : "f"(odd), "f"(even));
    return out;
}

// Quantize 16 BF16 values (two 16-byte vectors) into one NVFP4 block with torchao numerics:
// scale = UE4M3(amax / (6 * pts)), code = E2M1(x * (1 / pts) / scale). Returns the scale byte.
CUTLASS_DEVICE uint32_t nvfp4_quantize_bf16_block(const uint4& lo, const uint4& hi,
                                                   const float& sf_scale, const float& inv_pts,
                                                   uint32_t& word_0, uint32_t& word_1) {
    const auto p = reinterpret_cast<const nv_bfloat162*>(&lo);
    const auto q = reinterpret_cast<const nv_bfloat162*>(&hi);
    nv_bfloat162 amax2 = __hmax2(__habs2(p[0]), __habs2(p[1]));
    amax2 = __hmax2(amax2, __hmax2(__habs2(p[2]), __habs2(p[3])));
    amax2 = __hmax2(amax2, __hmax2(__habs2(q[0]), __habs2(q[1])));
    amax2 = __hmax2(amax2, __hmax2(__habs2(q[2]), __habs2(q[3])));
    const float amax = fmaxf(__low2float(amax2), __high2float(amax2));
    float decoded;
    const uint32_t sf_byte = nvfp4_encode_e4m3_scale(amax * sf_scale, decoded);
    const float r = decoded > 0.0f ? inv_pts / decoded : 0.0f;
    uint32_t bytes[8];
    #pragma unroll
    for (uint32_t j = 0; j < 4; ++ j) {
        const auto f = __bfloat1622float2(p[j]);
        bytes[j] = nvfp4_pack_pair(f.x * r, f.y * r);
    }
    #pragma unroll
    for (uint32_t j = 0; j < 4; ++ j) {
        const auto f = __bfloat1622float2(q[j]);
        bytes[4 + j] = nvfp4_pack_pair(f.x * r, f.y * r);
    }
    word_0 = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    word_1 = bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    return sf_byte;
}

CUTLASS_DEVICE void nvfp4_fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

// Per-row dynamic NVFP4 scaling: the row's amax lands on the largest UE4M3 scale (448) times the largest
// E2M1 code (6). Returns the per-row "per-tensor" scale and the two quantizer constants.
CUTLASS_DEVICE float nvfp4_row_scale(const float& row_amax, float& sf_scale, float& inv_pts) {
    const bool nonzero = row_amax > 0.0f;
    sf_scale = nonzero ? 448.0f / row_amax : 0.0f;
    inv_pts = nonzero ? 2688.0f / row_amax : 0.0f;
    return nonzero ? row_amax * (1.0f / 2688.0f) : 0.0f;
}

} // namespace deep_gemm
