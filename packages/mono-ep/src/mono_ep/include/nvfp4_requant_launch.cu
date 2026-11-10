// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI

// SM100 packed requant of merged NVFP4 LoRA experts:
// Q(round(round(DQ(base)) + scaling * B_e @ A_e)) on fixed per-tensor scales.
//
// Byte-identical to the Triton kernel in nvfp4_requant/triton_nvfp4_requant.py
// with delta_kernel="ieee": the rank-r delta is a sequential fp32 FMA chain in
// rank order from +0 (Triton's FMA dot, zero-padded to R_PAD), every other op
// is an explicit _rn intrinsic (the file is also built with --fmad=false), and
// E2M1 / E4M3 conversions use the SM100 cvt instructions, whose RNE and
// saturation match the Triton kernel's software encoders on finite inputs.
//
// CTA tile: 64 rows x 128 K, looping over K. A thread owns 2 rows x 16 K
// (one block: 8 code bytes and one scale byte per row); the 8 lanes of a row
// group cover 64 contiguous code bytes. A is staged in shared memory as fp32
// with the 8 chunk lanes' float4s adjacent, so each LDS.128 is one wavefront,
// double buffered: the next tile's A and codes are loaded into registers while
// the current tile computes, so each K tile costs one barrier. B rows sit in
// shared memory with a padded stride. The delta uses FFMA2 on column pairs
// (still one rounding per FMA).
//
// PAIR also emits the transposed form from the same tile, quantized in blocks
// of 16 rows per column and stored transposed. Paired (exact): the transposed
// base merged with the transposed delta, the bytes of a separate requant of
// the transposed form, with the rank-r delta computed once for both. Fused
// (numerics change): the merged tile itself, so no transposed base exists.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "nvfp4_requant_args.h"

namespace mono_ep_requant {

constexpr int kThreads = 256;
constexpr int kTileM = 64;
constexpr int kTileK = 128;
constexpr int kChunk = 16;
constexpr int kChunkLanes = kTileK / kChunk;  // 8
constexpr int kFused = 1, kPaired = 2;  // PAIR: transposed form from the merged tile / from a transposed base
constexpr float kOneSixth = 0.16666667163372040f;  // fp32(1/6), torchao's reciprocal multiply

struct SharedLayout {
    static constexpr int kASize(int r_pad) { return 2 * r_pad * kTileK; }  // double buffered
    static constexpr int kBStride(int r_pad) { return r_pad + 1; }
};

__device__ __forceinline__ float bf16_round(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// 8 floats -> 8 E2M1 codes, element 0 in the low nibble of byte 0.
__device__ __forceinline__ uint32_t cvt_e2m1x8(const float* x) {
    uint32_t out;
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}"
        : "=r"(out)
        : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]),
          "f"(x[7]));
    return out;
}

// 8 E2M1 codes (one word) -> 8 exact floats; code 8 decodes to -0.0.
__device__ __forceinline__ void decode_e2m1x8(uint32_t w, float* v) {
    uint32_t h0, h1, h2, h3;
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}"
        : "=r"(h0), "=r"(h1), "=r"(h2), "=r"(h3)
        : "r"(w));
    const uint32_t h[4] = {h0, h1, h2, h3};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h[i]));
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

// Two E4M3 bytes (low byte first) -> two exact floats.
__device__ __forceinline__ float2 decode_e4m3x2(uint32_t bits16) {
    uint32_t h;
    const uint16_t b = static_cast<uint16_t>(bits16);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(b));
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// Two floats -> two E4M3 bytes (RNE, saturating), lo in the low byte.
__device__ __forceinline__ uint32_t cvt_e4m3x2(float lo, float hi) {
    uint16_t out;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(out) : "f"(hi), "f"(lo));
    return out;
}

__device__ __forceinline__ void ffma2(float2& acc, float2 a, float b) {
    acc = __ffma2_rn(make_float2(b, b), a, acc);
}

__device__ __forceinline__ int64_t dest_row(int n, int half_n, int granularity) {
    if (granularity == 0) return n;
    if (n < half_n) return static_cast<int64_t>(n / granularity) * 2 * granularity + n % granularity;
    const int m = n - half_n;
    return static_cast<int64_t>(m / granularity) * 2 * granularity + granularity + m % granularity;
}

// Byte offset of SF (row, col) inside one expert's 128x4 atom grid.
__device__ __forceinline__ int64_t atom_offset(int64_t row, int col, int rk) {
    return ((row / 128) * rk + col / 4) * 512 + (row % 32) * 16 + ((row / 32) % 4) * 4 + col % 4;
}

// Block scale of a 16-value block from its amax; returns (bits, decoded value).
__device__ __forceinline__ void block_scale(float amax, float fq, bool modelopt, uint32_t& bits,
                                            float& sdec) {
    float s;
    if (modelopt) {
        float bs = __fdiv_rn(amax, __fmul_rn(6.0f, fq));
        bs = bs == 0.0f ? 1.0f : bs;
        s = fminf(fmaxf(bs, 0.001953125f), 448.0f);
    } else {
        const float bs = __fdiv_rn(__fmul_rn(amax, kOneSixth), fq);
        s = fminf(fmaxf(bs, 0.015625f), 448.0f);
    }
    bits = cvt_e4m3x2(s, 0.0f) & 0xFF;
    sdec = decode_e4m3x2(bits).x;
}

// Encode 16 merged values of one block given its scale.
__device__ __forceinline__ void encode_block(const float* w, float sdec, float fq, float inv_fq,
                                             bool modelopt, uint32_t& c0, uint32_t& c1) {
    float q[16];
    if (modelopt) {
        const float total = __fmul_rn(sdec, fq);
#pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = __fdiv_rn(w[i], total);
    } else {
        const float r = __fdiv_rn(inv_fq, sdec);
#pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = __fmul_rn(w[i], r);
    }
    c0 = cvt_e2m1x8(q);
    c1 = cvt_e2m1x8(q + 8);
    if (modelopt) {
        // ModelOpt's sign is q < 0: a -0.0 quotient keeps code 0, not 8
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (q[i] == 0.0f) c0 &= ~(0xFu << (4 * i));
            if (q[i + 8] == 0.0f) c1 &= ~(0xFu << (4 * i));
        }
    }
}

// Merges one 16-value block: w = round(round(v * sc) + scaling * delta), returns amax.
template <bool ROUND_BF16>
__device__ __forceinline__ float merge_block(uint32_t c0, uint32_t c1, float sc, float scaling,
                                             const float2* delta, float* w) {
    decode_e2m1x8(c0, w);
    decode_e2m1x8(c1, w + 8);
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        // scalar _rn ops: the f32x2 multiply / add forms got contracted into an FMA here
        float2 x = make_float2(__fmul_rn(w[2 * j], sc), __fmul_rn(w[2 * j + 1], sc));
        if constexpr (ROUND_BF16) x = __bfloat1622float2(__floats2bfloat162_rn(x.x, x.y));
        x.x = __fadd_rn(x.x, __fmul_rn(delta[j].x, scaling));
        x.y = __fadd_rn(x.y, __fmul_rn(delta[j].y, scaling));
        if constexpr (ROUND_BF16) x = __bfloat1622float2(__floats2bfloat162_rn(x.x, x.y));
        w[2 * j] = x.x;
        w[2 * j + 1] = x.y;
        amax = fmaxf(amax, fmaxf(fabsf(x.x), fabsf(x.y)));
    }
    return amax;
}

// Quantizes one merged block to (8 code bytes, scale bits), fresh or on the reused base scale.
__device__ __forceinline__ void quantize_block(const float* w, float amax, float fq, float inv_fq,
                                               bool modelopt, bool reuse, float s_base,
                                               uint32_t s_base_bits, uint32_t& c0, uint32_t& c1,
                                               uint32_t& bits) {
    float sdec;
    block_scale(amax, fq, modelopt, bits, sdec);
    if (reuse && !(amax > __fmul_rn(6.0f, __fmul_rn(fq, s_base)))) {
        sdec = s_base;
        bits = s_base_bits;
    }
    encode_block(w, sdec, fq, inv_fq, modelopt, c0, c1);
}

template <int R_PAD, bool AB_FP32, bool ROUND_BF16, int MODE, int PAIR, bool FFMA2>
__global__ void __launch_bounds__(kThreads, 2) requant_kernel(const Args args) {
    constexpr int kBStride = SharedLayout::kBStride(R_PAD);
    constexpr int kVec = AB_FP32 ? 4 : 8;  // A elements per 16-byte load
    constexpr int kStash = R_PAD * kTileK / kVec / kThreads;
    constexpr int kABuf = R_PAD * kTileK;
    extern __shared__ __align__(16) float smem[];
    float* a_s = smem;                                    // 2 x [R_PAD][4 qd][8 lanes][4]
    float* b_s = a_s + SharedLayout::kASize(R_PAD);       // [64][R_PAD + 1]
    float* w_s = b_s + kTileM * kBStride;                 // PAIR: [64][128], 16-byte xor swizzle

    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int chunk = lane % kChunkLanes, group = lane / kChunkLanes;
    const int local_e = blockIdx.z;
    const int e = args.e0 + local_e;
    const int n0 = blockIdx.x * kTileM;
    const int N = args.n, K = args.k, R = args.r;
    const int k2 = K / 2, sf_k = K / 16;

    int row_local[2];
    row_local[0] = warp * 4 + group;
    row_local[1] = row_local[0] + 32;

    for (int i = tid; i < kTileM * R_PAD; i += kThreads) {
        const int rl = i / R_PAD, rr = i % R_PAD;
        const int n = n0 + rl;
        float v = 0.0f;
        if (n < N && rr < R) {
            const int64_t off = (static_cast<int64_t>(e) * N + n) * R + rr;
            if constexpr (AB_FP32) v = static_cast<const float*>(args.b)[off];
            else v = __bfloat162float(static_cast<const __nv_bfloat16*>(args.b)[off]);
        }
        b_s[rl * kBStride + rr] = v;
    }

    float dq[2], fq[2], inv_fq[2];
    int64_t src_row[2], dst_row[2];
    bool row_ok[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int n = n0 + row_local[i];
        row_ok[i] = n < N;
        const int proj = min(n / args.rows_per_proj, args.p - 1);
        dq[i] = row_ok[i] ? args.dq_pts[e * args.p + proj] : 1.0f;
        fq[i] = row_ok[i] ? args.fq_pts[e * args.p + proj] : 1.0f;
        inv_fq[i] = __fdiv_rn(1.0f, fq[i]);
        src_row[i] = static_cast<int64_t>(e) * N + n;
        dst_row[i] = static_cast<int64_t>(local_e) * N + dest_row(n, N / 2, args.interleave);
    }
    const int64_t atom_base = static_cast<int64_t>(local_e) * args.rm * args.rk * 512;

    const int k_tiles = (K + kTileK - 1) / kTileK;
    const int tiles_per_split = (k_tiles + gridDim.y - 1) / gridDim.y;
    const int kt_begin = blockIdx.y * tiles_per_split;
    const int kt_end = min(k_tiles, kt_begin + tiles_per_split);

    // Raw 16-byte A vectors of one K tile, held in registers until the buffer they go to is free.
    auto load_a = [&](int k0, uint4 (&stash)[kStash]) {
#pragma unroll
        for (int s = 0; s < kStash; ++s) {
            const int i = tid + s * kThreads;
            const int rr = i / (kTileK / kVec), c = (i % (kTileK / kVec)) * kVec;
            stash[s] = make_uint4(0, 0, 0, 0);
            if (rr < R && k0 + c < K) {
                const int64_t off = (static_cast<int64_t>(e) * R + rr) * K + k0 + c;
                const char* base = static_cast<const char*>(args.a) + off * (AB_FP32 ? 4 : 2);
                stash[s] = __ldg(reinterpret_cast<const uint4*>(base));
            }
        }
    };
    auto store_a = [&](float* buf, const uint4 (&stash)[kStash]) {
#pragma unroll
        for (int s = 0; s < kStash; ++s) {
            const int i = tid + s * kThreads;
            const int rr = i / (kTileK / kVec), c = (i % (kTileK / kVec)) * kVec;
            float v[kVec];
            if constexpr (AB_FP32) {
                const float4 f = *reinterpret_cast<const float4*>(&stash[s]);
                v[0] = f.x, v[1] = f.y, v[2] = f.z, v[3] = f.w;
            } else {
                const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&stash[s]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(p[j]);
                    v[2 * j] = f.x, v[2 * j + 1] = f.y;
                }
            }
            const int lane_c = c / kChunk, j = c % kChunk;
#pragma unroll
            for (int h = 0; h < kVec / 4; ++h)
                *reinterpret_cast<float4*>(buf + ((rr * 4 + j / 4 + h) * kChunkLanes + lane_c) * 4) =
                    make_float4(v[4 * h], v[4 * h + 1], v[4 * h + 2], v[4 * h + 3]);
        }
    };
    auto load_codes = [&](int k0, uint2 (&codes)[2], uint32_t (&sbits)[2]) {
        const int col = k0 + chunk * kChunk;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            codes[i] = make_uint2(0, 0);
            sbits[i] = 0;
            if (row_ok[i] && col < K) {
                codes[i] = __ldg(reinterpret_cast<const uint2*>(args.q + src_row[i] * k2 + col / 2));
                sbits[i] = __ldg(args.s + src_row[i] * sf_k + col / 16);
            }
        }
    };

    uint2 codes[2];
    uint32_t sbits[2];
    uint4 stash[kStash];
    if (kt_begin < kt_end) {
        load_codes(kt_begin * kTileK, codes, sbits);
        if constexpr (MODE >= 2) {
            load_a(kt_begin * kTileK, stash);
            store_a(a_s, stash);
        }
    }
    __syncthreads();

    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int k0 = kt * kTileK;
        const int col = k0 + chunk * kChunk;
        const bool col_ok = col < K;
        const bool has_next = kt + 1 < kt_end;
        const float* a_cur = a_s + ((kt - kt_begin) & 1) * kABuf;
        // paired: this tile's transposed base (column k0 + tid % 128, 32 rows of N), loaded before the delta
        const int half = tid / kTileK;
        uint4 tcodes = make_uint4(0, 0, 0, 0);
        uint32_t tsbits = 0;
        if constexpr (PAIR == kPaired) {
            const int kcol = k0 + tid % kTileK;
            if (kcol < K) {
                const int64_t trow = static_cast<int64_t>(e) * K + kcol;
                const int nb = n0 + half * 32;
                tcodes = __ldg(reinterpret_cast<const uint4*>(args.qt + trow * (N / 2) + nb / 2));
                tsbits = __ldg(reinterpret_cast<const uint16_t*>(args.st + trow * (N / 16) + nb / 16));
            }
        }
        uint2 next_codes[2];
        uint32_t next_sbits[2];
        if (has_next) {
            load_codes(k0 + kTileK, next_codes, next_sbits);
            if constexpr (MODE >= 2) load_a(k0 + kTileK, stash);
        }

        // rank-r delta: acc = fma(b_r, a_r, acc) for r = 0 .. R_PAD - 1, from +0
        float2 acc[2][kChunk / 2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < kChunk / 2; ++j) acc[i][j] = make_float2(0.0f, 0.0f);
        if constexpr (MODE >= 2) {
#pragma unroll
            for (int rr = 0; rr < R_PAD; ++rr) {
                const float bs[2] = {b_s[row_local[0] * kBStride + rr], b_s[row_local[1] * kBStride + rr]};
#pragma unroll
                for (int qd = 0; qd < 4; ++qd) {
                    const float4 a4 = *reinterpret_cast<const float4*>(a_cur + ((rr * 4 + qd) * kChunkLanes + chunk) * 4);
                    const float2 lo = make_float2(a4.x, a4.y), hi = make_float2(a4.z, a4.w);
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        if constexpr (FFMA2) {
                            ffma2(acc[i][qd * 2], lo, bs[i]);
                            ffma2(acc[i][qd * 2 + 1], hi, bs[i]);
                        } else {
                            acc[i][qd * 2].x = __fmaf_rn(bs[i], lo.x, acc[i][qd * 2].x);
                            acc[i][qd * 2].y = __fmaf_rn(bs[i], lo.y, acc[i][qd * 2].y);
                            acc[i][qd * 2 + 1].x = __fmaf_rn(bs[i], hi.x, acc[i][qd * 2 + 1].x);
                            acc[i][qd * 2 + 1].y = __fmaf_rn(bs[i], hi.y, acc[i][qd * 2 + 1].y);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint2 out = codes[i];
            uint32_t out_s = sbits[i];
            if constexpr (MODE != 0) {
                const float s_base = decode_e4m3x2(sbits[i]).x;
                float w[16];
                const float amax = merge_block<ROUND_BF16>(codes[i].x, codes[i].y, __fmul_rn(s_base, dq[i]),
                                                           args.scaling, acc[i], w);
                if constexpr (PAIR != 0) {
                    // fused: the merged tile; paired: the raw delta, for the transposed form
#pragma unroll
                    for (int j4 = 0; j4 < 4; ++j4) {
                        const int col4 = chunk * 4 + j4;
                        const int phys = col4 ^ ((col4 >> 3) & 3);
                        const float4 v = PAIR == kFused
                            ? make_float4(w[4 * j4], w[4 * j4 + 1], w[4 * j4 + 2], w[4 * j4 + 3])
                            : make_float4(acc[i][2 * j4].x, acc[i][2 * j4].y, acc[i][2 * j4 + 1].x, acc[i][2 * j4 + 1].y);
                        *reinterpret_cast<float4*>(w_s + row_local[i] * kTileK + phys * 4) = v;
                    }
                }
                if constexpr (MODE == 2) {
                    // decode + delta only: keep the values live, skip the quantizer
                    uint32_t h = __float_as_uint(amax);
#pragma unroll
                    for (int j = 0; j < 16; ++j) h ^= __float_as_uint(w[j]) >> (j & 7);
                    out = make_uint2(h, h >> 3);
                    out_s = h & 0xFF;
                } else {
                    quantize_block(w, amax, fq[i], inv_fq[i], args.modelopt, args.reuse, s_base, sbits[i],
                                   out.x, out.y, out_s);
                }
            }
            if (row_ok[i] && col_ok) {
                *reinterpret_cast<uint2*>(args.oq + dst_row[i] * k2 + col / 2) = out;
                args.os[dst_row[i] * sf_k + col / 16] = static_cast<uint8_t>(out_s);
                if (args.osfa != nullptr) {
                    const int64_t drow = dst_row[i] - static_cast<int64_t>(local_e) * N;
                    args.osfa[atom_base + atom_offset(drow, col / 16, args.rk)] = static_cast<uint8_t>(out_s);
                }
            }
        }

        if constexpr (PAIR != 0) {
            // Transposed form: column k0 + tid % 128, rows (tid / 128) * 32 .. + 32 = 2 blocks along N,
            // on the expert's one per-tensor scale. Fused: fresh scales of the merged tile. Paired: the
            // transposed base merged with the transposed delta, exactly as a separate requant of it.
            __syncthreads();
            const int c = tid % kTileK;
            const int kcol = k0 + c;
            if (kcol < K) {
                const float fqt = args.fq_pts[e * args.p];
                const float inv_fqt = __fdiv_rn(1.0f, fqt);
                const int col4 = c / 4;
                const int phys = col4 ^ ((col4 >> 3) & 3);
                const uint32_t tw[4] = {tcodes.x, tcodes.y, tcodes.z, tcodes.w};
                uint32_t res[4];
                uint32_t sw = 0;
#pragma unroll
                for (int blk = 0; blk < 2; ++blk) {
                    float w[16];
                    float amax = 0.0f;
                    float s_base = 0.0f;
                    uint32_t s_bits = 0;
                    if constexpr (PAIR == kFused) {
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            w[j] = w_s[(half * 32 + blk * 16 + j) * kTileK + phys * 4 + c % 4];
                            amax = fmaxf(amax, fabsf(w[j]));
                        }
                    } else {
                        float2 d[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            d[j] = make_float2(w_s[(half * 32 + blk * 16 + 2 * j) * kTileK + phys * 4 + c % 4],
                                               w_s[(half * 32 + blk * 16 + 2 * j + 1) * kTileK + phys * 4 + c % 4]);
                        s_bits = (tsbits >> (8 * blk)) & 0xFF;
                        s_base = decode_e4m3x2(s_bits).x;
                        amax = merge_block<ROUND_BF16>(tw[2 * blk], tw[2 * blk + 1],
                                                       __fmul_rn(s_base, args.dq_pts[e * args.p]), args.scaling, d, w);
                    }
                    uint32_t bits;
                    quantize_block(w, amax, fqt, inv_fqt, PAIR == kPaired && args.modelopt,
                                   PAIR == kPaired && args.reuse, s_base, s_bits, res[2 * blk], res[2 * blk + 1], bits);
                    sw |= bits << (8 * blk);
                }
                const int64_t trow = static_cast<int64_t>(local_e) * K + kcol;
                const int nb = n0 + half * 32;
                *reinterpret_cast<uint4*>(args.oq_t + trow * (N / 2) + nb / 2) = make_uint4(res[0], res[1], res[2], res[3]);
                *reinterpret_cast<uint16_t*>(args.os_t + trow * (N / 16) + nb / 16) = static_cast<uint16_t>(sw);
                if (args.osfa_t != nullptr) {
                    const int64_t tbase = static_cast<int64_t>(local_e) * args.rm_t * args.rk_t * 512;
                    *reinterpret_cast<uint16_t*>(args.osfa_t + tbase + atom_offset(kcol, nb / 16, args.rk_t)) =
                        static_cast<uint16_t>(sw);
                }
            }
        }

        if (has_next) {
            if constexpr (MODE >= 2) store_a(a_s + ((kt + 1 - kt_begin) & 1) * kABuf, stash);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                codes[i] = next_codes[i];
                sbits[i] = next_sbits[i];
            }
        }
        __syncthreads();  // next A buffer written; this tile's readers of the other buffer and w_s done
    }
}

template <int R_PAD, bool AB_FP32, bool ROUND_BF16, int MODE, int PAIR, bool FFMA2>
int launch_typed(const Args& args, cudaStream_t stream) {
    auto kernel = requant_kernel<R_PAD, AB_FP32, ROUND_BF16, MODE, PAIR, FFMA2>;
    const size_t smem = sizeof(float) *
        (SharedLayout::kASize(R_PAD) + kTileM * SharedLayout::kBStride(R_PAD) + (PAIR ? kTileM * kTileK : 0));
    auto error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (error != cudaSuccess) return static_cast<int>(error);
    const dim3 grid((args.n + kTileM - 1) / kTileM, args.k_splits, args.e_local);
    kernel<<<grid, kThreads, smem, stream>>>(args);
    return static_cast<int>(cudaGetLastError());
}

template <int R_PAD, bool AB_FP32>
int launch_rank(const Args& args, cudaStream_t stream) {
    if (args.pair == kFused) {
        if (!args.round_bf16) return -2;
        return args.ffma2 ? launch_typed<R_PAD, AB_FP32, true, 3, kFused, true>(args, stream)
                          : launch_typed<R_PAD, AB_FP32, true, 3, kFused, false>(args, stream);
    }
    if (args.pair == kPaired) {
        if (!args.round_bf16) return launch_typed<R_PAD, AB_FP32, false, 3, kPaired, true>(args, stream);
        return args.ffma2 ? launch_typed<R_PAD, AB_FP32, true, 3, kPaired, true>(args, stream)
                          : launch_typed<R_PAD, AB_FP32, true, 3, kPaired, false>(args, stream);
    }
    if (!args.round_bf16) return launch_typed<R_PAD, AB_FP32, false, 3, 0, false>(args, stream);
    if (args.mode == 3)
        return args.ffma2 ? launch_typed<R_PAD, AB_FP32, true, 3, 0, true>(args, stream)
                          : launch_typed<R_PAD, AB_FP32, true, 3, 0, false>(args, stream);
    if constexpr (R_PAD == 16 && !AB_FP32) {
        switch (args.mode) {
            case 0: return launch_typed<16, false, true, 0, 0, false>(args, stream);
            case 1: return launch_typed<16, false, true, 1, 0, false>(args, stream);
            case 2: return launch_typed<16, false, true, 2, 0, false>(args, stream);
        }
    }
    return -3;
}

// Conversion probes for exhaustive tests of the cvt paths against the software encoders.
__global__ void convert_probe_kernel(int kind, const uint32_t* in, uint32_t* out, int64_t n) {
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const uint32_t x = in[i];
        uint32_t y = 0;
        if (kind == 0) {  // fp32 -> e2m1 code (low nibble)
            float v[8] = {__uint_as_float(x), 0, 0, 0, 0, 0, 0, 0};
            y = cvt_e2m1x8(v) & 0xF;
        } else if (kind == 1) {  // fp32 -> e4m3 bits
            y = cvt_e4m3x2(__uint_as_float(x), 0.0f) & 0xFF;
        } else if (kind == 2) {  // e2m1 byte -> low/high nibble values as fp32 (low nibble returned)
            float v[8];
            decode_e2m1x8(x, v);
            y = __float_as_uint(v[0]);
        } else if (kind == 3) {  // e2m1 byte -> high nibble value
            float v[8];
            decode_e2m1x8(x, v);
            y = __float_as_uint(v[1]);
        } else if (kind == 4) {  // e4m3 byte (low) -> fp32
            y = __float_as_uint(decode_e4m3x2(x).x);
        } else {  // e4m3 byte (high) -> fp32
            y = __float_as_uint(decode_e4m3x2(x).y);
        }
        out[i] = y;
    }
}

}  // namespace mono_ep_requant

int nvfp4_requant_probe(int kind, uintptr_t in, uintptr_t out, int64_t n, uintptr_t stream_ptr) {
    mono_ep_requant::convert_probe_kernel<<<1184, 256, 0, reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
        kind, reinterpret_cast<const uint32_t*>(in), reinterpret_cast<uint32_t*>(out), n);
    return static_cast<int>(cudaGetLastError());
}

int nvfp4_requant_launch(const mono_ep_requant::Args& args, uintptr_t stream_ptr) {
    using namespace mono_ep_requant;
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    switch (args.r_pad) {
        case 16: return args.ab_fp32 ? launch_rank<16, true>(args, stream) : launch_rank<16, false>(args, stream);
        case 32: return args.ab_fp32 ? launch_rank<32, true>(args, stream) : launch_rank<32, false>(args, stream);
        case 64: return args.ab_fp32 ? launch_rank<64, true>(args, stream) : launch_rank<64, false>(args, stream);
    }
    return -1;
}
