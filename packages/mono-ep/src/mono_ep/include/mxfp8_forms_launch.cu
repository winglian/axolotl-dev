// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI

// SM100 one-read builder of the MXFP8 backward weight forms (CUDA port of
// mx_weight_forms.py's Triton kernel, bitwise identical to it), structured
// like Mixture-of-Kittens' csrc/mxfp8.cuh quantize_kernel.
//
// One 128 x 128 tile of W ([E, ROWS, COLS], BF16 or FP32) per 256-thread
// CTA in about 50 KB of shared memory (BF16), so four CTAs share an SM and
// their load / quantize / store phases interleave. The tile arrives by TMA
// (128-byte swizzled boxes of 128 rows); the transposed pass runs first
// with every thread owning a column pair and one 32-row block; the row pass
// then pulls two 32-column blocks of a row into registers, the CTA syncs
// and the E4M3 row form is written in place over the first 16 KB of the
// input tile. Both E4M3 tiles leave through TMA tensor stores (the
// interleaved W1 row form through a 4-D map whose box is the tile's 16
// eight-row groups) and the scale words through bulk copies of their
// contiguous UTCCP spans. Staging with the same 128-byte swizzle as the
// loads keeps the shared-memory traffic of both passes conflict-free, and
// the 32-element amax of a block is register-local in both.
//
// Numerics are the Triton kernel's: amax over the 32 values (taken in BF16
// for BF16 input, which is exact), floor 1e-4, exponent =
// ceil(log2(amax / 448)) read from the IEEE-RN quotient's bits, clamped to
// [1, 254]; values are multiplied by the exact power-of-two inverse in FP32
// and converted with cvt.rn.satfinite.e4m3x2.f32 (the Triton clamp to
// +-448 followed by its RNE conversion). The file is built with --fmad=false.
//
// Output contract (see mx_weight_forms.py): row form QR [E, ROWS, COLS]
// with the optional gate / up interleave of 8-row groups and scale words
// SR [E][COLS / 128][ROWS] in the UTCCP row order; transposed form
// QT [E, COLS, ROWS] with words ST [E][ROWS / 128][COLS]. Each word packs
// the four UE8M0 exponents of one 128-wide K block, low byte first.
//
// LoRA instantiation (BF16 W only): the tile is first merged in shared memory with
// W + scaling * B_e @ A_e (B [E, ROWS, R], A [E, R, COLS], BF16): the rank-R product on the
// tensor cores (FP32 accumulate) and the merge one fmaf rounded to BF16 once, so the forms
// quantize the merged weight without it ever reaching global memory.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace mono_ep_mxfp8_forms {

constexpr int kTile = 128;
constexpr int kBlock = 32;               // MX block: one scale per 32 values
constexpr int kBlocks = kTile / kBlock;  // 4 per tile row / column
constexpr int kThreads = 256;
constexpr int kSub = kTile * 128;        // one 128-row x 128-byte swizzled box
constexpr int kWordBytes = kTile * 4;    // scale words of one 128-row / 128-column block

template <typename T>
struct Layout {
    static constexpr int kRowBytes = kTile * sizeof(T);  // 256 / 512
    static constexpr int kSubs = kRowBytes / 128;        // boxes per tile: 2 / 4
    static constexpr int kInner = 128 / sizeof(T);       // columns per box: 64 / 32
    static constexpr int kTileBytes = kTile * kRowBytes;
    static constexpr int kOutT = kTileBytes;               // transposed E4M3 tile (the row form overlays box 0)
    static constexpr int kWordsR = kOutT + kTile * kTile;  // 2 x 128 row words (two output blocks)
    static constexpr int kWordsT = kWordsR + 2 * kWordBytes;
    static constexpr int kBarrier = kWordsT + kWordBytes;
    static constexpr int kSmemBytes = kBarrier + 16;
    // LoRA factors after the barrier, BF16 zero-padded to the MMA K step: B [128 rows][Rp], A^T [128 cols][Rp]
    static constexpr int kLoraOffset = (kSmemBytes + 127) / 128 * 128;
    static constexpr int lora_bytes(int rank) {
        return (rank <= 16 ? 16 : rank <= 32 ? 32 : rank <= 64 ? 64 : 128) * kTile * 4;
    }
    static constexpr int kMinBlocks = sizeof(T) == 2 ? 4 : 2;
};

extern __shared__ __align__(1024) uint8_t smem_raw[];

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Byte offset of 16-byte chunk `chunk` of `row` inside a 128-byte-row box (TMA SWIZZLE_128B pattern).
__device__ __forceinline__ uint32_t swizzled(int row, int chunk) {
    return static_cast<uint32_t>(row * 128 + ((chunk ^ (row & 7)) << 4));
}

// ---- async proxy -------------------------------------------------------------------------------

__device__ __forceinline__ void mbarrier_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint32_t bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n"
        "}" ::"r"(bar),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
        "[%4];" ::"r"(dst),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
        : "memory");
}

__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, int c0, int c1, uint32_t src) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(
                     reinterpret_cast<uint64_t>(map)),
                 "r"(c0), "r"(c1), "r"(src)
                 : "memory");
}

__device__ __forceinline__ void tma_store_4d(const CUtensorMap* map, int c0, int c1, int c2, int c3, uint32_t src) {
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%1, %2, %3, %4}], [%5];" ::"l"(
                     reinterpret_cast<uint64_t>(map)),
                 "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(src)
                 : "memory");
}

__device__ __forceinline__ void bulk_store(void* gmem, uint32_t smem, int bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(gmem), "r"(smem), "r"(bytes)
                 : "memory");
}

// Generic-proxy shared-memory writes -> visible to the bulk-copy (async) proxy.
__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }

__device__ __forceinline__ void bulk_wait_read_all() {
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// ---- shared memory -----------------------------------------------------------------------------

__device__ __forceinline__ void sts128(uint32_t addr, uint4 v) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ void sts8(uint32_t addr, uint32_t byte) {
    asm volatile("st.shared.b8 [%0], %1;" ::"r"(addr), "r"(byte) : "memory");
}

__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
    uint32_t v;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
    return v;
}

__device__ __forceinline__ uint2 lds64(uint32_t addr) {
    uint2 v;
    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(addr) : "memory");
    return v;
}

__device__ __forceinline__ uint4 lds128(uint32_t addr) {
    uint4 v;
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "r"(addr)
                 : "memory");
    return v;
}

// 32 E4M3 bytes of one block into a swizzled 128-byte staging row.
__device__ __forceinline__ void stage_block(uint32_t box, int row, int kb, const uint32_t (&q)[8]) {
    sts128(box + swizzled(row, 2 * kb), make_uint4(q[0], q[1], q[2], q[3]));
    sts128(box + swizzled(row, 2 * kb + 1), make_uint4(q[4], q[5], q[6], q[7]));
}

// Row / column c's word sits at UTCCP slot (c % 32) * 4 + c / 32 of its 128-row / 128-column block.
__device__ __forceinline__ int utccp_slot(int index) { return (index % 32) * 4 + index / 32; }

// ---- numerics ----------------------------------------------------------------------------------

__device__ __forceinline__ void unpack_bf16x2(uint32_t w, float& lo, float& hi) {
    lo = __uint_as_float(w << 16);
    hi = __uint_as_float(w & 0xFFFF0000u);
}

// Triton's _ue8m0_exponent: ceil(log2(max(amax, 1e-4) / 448)) in biased form, clamped to [1, 254].
__device__ __forceinline__ uint32_t ue8m0_exponent(float amax) {
    const uint32_t bits = __float_as_uint(__fdiv_rn(fmaxf(amax, 1.0e-4f), 448.0f));
    const uint32_t exponent = ((bits >> 23) & 255u) + ((bits & 0x7FFFFFu) != 0u ? 1u : 0u);
    return min(max(exponent, 1u), 254u);
}

__device__ __forceinline__ float inverse_scale(uint32_t exponent) {
    return __uint_as_float((254u - exponent) << 23);
}

// cvt.rn.satfinite: RNE with saturation to +-448, which is what the Triton kernel's clamp and
// .to(float8e4m3fn) produce; element `lo` lands in the low byte.
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
    uint16_t out;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(out) : "f"(hi), "f"(lo));
    return out;
}

// |max| of a BF16 pair register against a running BF16 pair maximum (exact: the BF16 and FP32 maxima agree).
__device__ __forceinline__ __nv_bfloat162 absmax2(__nv_bfloat162 acc, uint32_t packed) {
    __nv_bfloat162 v;
    memcpy(&v, &packed, 4);
    return __hmax2(acc, __habs2(v));
}

// 4 packed BF16 pairs (8 values) scaled and converted: 2 E4M3 words, value order kept.
__device__ __forceinline__ void quantize_bf16x8(const uint4 raw, float inv, uint32_t& q0, uint32_t& q1) {
    float v0, v1, v2, v3, v4, v5, v6, v7;
    unpack_bf16x2(raw.x, v0, v1);
    unpack_bf16x2(raw.y, v2, v3);
    unpack_bf16x2(raw.z, v4, v5);
    unpack_bf16x2(raw.w, v6, v7);
    q0 = e4m3x2(v0 * inv, v1 * inv) | (e4m3x2(v2 * inv, v3 * inv) << 16);
    q1 = e4m3x2(v4 * inv, v5 * inv) | (e4m3x2(v6 * inv, v7 * inv) << 16);
}

__device__ __forceinline__ uint32_t quantize_f32x4(const uint4 raw, float inv) {
    return e4m3x2(__uint_as_float(raw.x) * inv, __uint_as_float(raw.y) * inv)
         | (e4m3x2(__uint_as_float(raw.z) * inv, __uint_as_float(raw.w) * inv) << 16);
}

__device__ __forceinline__ float absmax_f32x4(float amax, const uint4 raw) {
    amax = fmaxf(amax, fabsf(__uint_as_float(raw.x)));
    amax = fmaxf(amax, fabsf(__uint_as_float(raw.y)));
    amax = fmaxf(amax, fabsf(__uint_as_float(raw.z)));
    return fmaxf(amax, fabsf(__uint_as_float(raw.w)));
}

// ---- passes ------------------------------------------------------------------------------------

// Transposed form: thread (p, kb) owns tile columns 2p, 2p + 1 for row block kb, staged at rows 2p, 2p + 1
// of the transposed tile.
template <typename T>
__device__ __forceinline__ void transposed_pass(uint32_t smem, int t) {
    using L = Layout<T>;
    const int p = t & 63, kb = t >> 6;
    const int byte = 2 * p * static_cast<int>(sizeof(T));
    const int sub = byte / 128, chunk = (byte % 128) / 16, offset = byte % 16;
    const uint32_t in = smem + sub * kSub;
    uint32_t qa[8], qb[8];
    uint32_t ea, eb;
    if constexpr (sizeof(T) == 2) {
        uint32_t raw[kBlock];
        __nv_bfloat162 amax = __float2bfloat162_rn(0.0f);
#pragma unroll
        for (int i = 0; i < kBlock; ++i) {
            raw[i] = lds32(in + swizzled(kb * kBlock + i, chunk) + offset);
            amax = absmax2(amax, raw[i]);
        }
        ea = ue8m0_exponent(__low2float(amax));
        eb = ue8m0_exponent(__high2float(amax));
        const float ia = inverse_scale(ea), ib = inverse_scale(eb);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float a0, b0, a1, b1, a2, b2, a3, b3;
            unpack_bf16x2(raw[4 * i], a0, b0);
            unpack_bf16x2(raw[4 * i + 1], a1, b1);
            unpack_bf16x2(raw[4 * i + 2], a2, b2);
            unpack_bf16x2(raw[4 * i + 3], a3, b3);
            qa[i] = e4m3x2(a0 * ia, a1 * ia) | (e4m3x2(a2 * ia, a3 * ia) << 16);
            qb[i] = e4m3x2(b0 * ib, b1 * ib) | (e4m3x2(b2 * ib, b3 * ib) << 16);
        }
    } else {
        float a[kBlock], b[kBlock];
        float amax_a = 0.0f, amax_b = 0.0f;
#pragma unroll
        for (int i = 0; i < kBlock; ++i) {
            const uint2 raw = lds64(in + swizzled(kb * kBlock + i, chunk) + offset);
            a[i] = __uint_as_float(raw.x);
            b[i] = __uint_as_float(raw.y);
            amax_a = fmaxf(amax_a, fabsf(a[i]));
            amax_b = fmaxf(amax_b, fabsf(b[i]));
        }
        ea = ue8m0_exponent(amax_a);
        eb = ue8m0_exponent(amax_b);
        const float ia = inverse_scale(ea), ib = inverse_scale(eb);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qa[i] = e4m3x2(a[4 * i] * ia, a[4 * i + 1] * ia) | (e4m3x2(a[4 * i + 2] * ia, a[4 * i + 3] * ia) << 16);
            qb[i] = e4m3x2(b[4 * i] * ib, b[4 * i + 1] * ib) | (e4m3x2(b[4 * i + 2] * ib, b[4 * i + 3] * ib) << 16);
        }
    }
    const uint32_t out = smem + L::kOutT;
    stage_block(out, 2 * p, kb, qa);
    stage_block(out, 2 * p + 1, kb, qb);
    const uint32_t words = smem + L::kWordsT;
    sts8(words + utccp_slot(2 * p) * 4 + kb, ea);
    sts8(words + utccp_slot(2 * p + 1) * 4 + kb, eb);
}

// Row form: thread (r, h) owns tile row r, 32-column blocks 2h and 2h + 1. The values are read before the CTA
// barrier; the E4M3 blocks are written after it, in place over box 0 of the input tile.
template <typename T>
struct RowPass {
    using L = Layout<T>;
    static constexpr int kChunks = kBlock * sizeof(T) / 16;  // 16-byte chunks per block: 4 / 8
    uint4 raw[2][kChunks];
    int r, h;

    __device__ __forceinline__ void read(uint32_t smem, int t) {
        r = t & 127;
        h = t >> 7;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int c = 0; c < kChunks; ++c) {
                const int chunk = (2 * h + j) * kChunks + c;  // within the row; box = chunk / 8
                raw[j][c] = lds128(smem + (chunk / 8) * kSub + swizzled(r, chunk % 8));
            }
        }
    }

    // `word` is the staged word of this row; byte kb of it is this block's exponent.
    __device__ __forceinline__ void write(uint32_t smem, uint32_t word) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int kb = 2 * h + j;
            uint32_t q[8];
            uint32_t exponent;
            if constexpr (sizeof(T) == 2) {
                __nv_bfloat162 amax = __float2bfloat162_rn(0.0f);
#pragma unroll
                for (int c = 0; c < kChunks; ++c) {
                    amax = absmax2(amax, raw[j][c].x);
                    amax = absmax2(amax, raw[j][c].y);
                    amax = absmax2(amax, raw[j][c].z);
                    amax = absmax2(amax, raw[j][c].w);
                }
                exponent = ue8m0_exponent(fmaxf(__low2float(amax), __high2float(amax)));
                const float inv = inverse_scale(exponent);
#pragma unroll
                for (int c = 0; c < kChunks; ++c) quantize_bf16x8(raw[j][c], inv, q[2 * c], q[2 * c + 1]);
            } else {
                float amax = 0.0f;
#pragma unroll
                for (int c = 0; c < kChunks; ++c) amax = absmax_f32x4(amax, raw[j][c]);
                exponent = ue8m0_exponent(amax);
                const float inv = inverse_scale(exponent);
#pragma unroll
                for (int c = 0; c < kChunks; ++c) q[c] = quantize_f32x4(raw[j][c], inv);
            }
            stage_block(smem, r, kb, q);
            sts8(word + kb, exponent);
        }
    }
};

// ---- kernel ------------------------------------------------------------------------------------

struct Maps {
    CUtensorMap in;     // W as [E * ROWS, COLS], 128-byte inner boxes of 128 rows
    CUtensorMap out_r;  // QR as [E * ROWS, COLS] or, interleaved, [E * ROWS / 16, 2, 8, COLS]
    CUtensorMap out_t;  // QT as [E * COLS, ROWS]
};

struct Lora {
    const __nv_bfloat16* b;  // [E, ROWS, R]
    const __nv_bfloat16* a;  // [E, R, COLS]
    int rank;
    float scaling;
};

// Rank padded to a power-of-two number of MMA K steps (1, 2, 4, 8); the factors are staged zero-padded to it.
__device__ __host__ __forceinline__ int lora_steps(int rank) { return rank <= 16 ? 1 : rank <= 32 ? 2 : rank <= 64 ? 4 : 8; }
__device__ __host__ __forceinline__ int padded_rank(int rank) { return lora_steps(rank) * 16; }

__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Warp w merges tile rows 16w .. 16w + 15 in place: delta = B_rows @ A_tile on the tensor cores (BF16 in,
// FP32 accumulate, K = the padded rank), then W + scaling * delta with one fmaf, rounded to BF16. The staged
// factors are B as [128 rows][Rp] and A transposed as [128 columns][Rp], K contiguous in both.
template <int kSteps>
__device__ __forceinline__ void merge_tile(uint32_t smem, const uint8_t* factors, float scaling, int t) {
    constexpr int rp = kSteps * 16;
    const uint32_t* bs = reinterpret_cast<const uint32_t*>(factors);
    const uint32_t* at = reinterpret_cast<const uint32_t*>(factors + kTile * rp * 2);
    const int warp = t >> 5, lane = t & 31, g = lane >> 2, q = lane & 3;
    const int row0 = warp * 16 + g, row1 = row0 + 8;
    constexpr int words = rp / 2;  // u32 per staged row
    uint32_t a[kSteps][4];
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
        a[s][0] = bs[row0 * words + s * 8 + q];
        a[s][1] = bs[row1 * words + s * 8 + q];
        a[s][2] = bs[row0 * words + s * 8 + 4 + q];
        a[s][3] = bs[row1 * words + s * 8 + 4 + q];
    }
    for (int nb = 0; nb < kTile / 8; ++nb) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int n = nb * 8 + g;
#pragma unroll
        for (int s = 0; s < kSteps; ++s) {
            mma_bf16_16816(acc, a[s], at[n * words + s * 8 + q], at[n * words + s * 8 + 4 + q]);
        }
        const int col = nb * 8 + 2 * q;
        const uint32_t base = smem + (col / 64) * kSub + ((col % 8) * 2);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t addr = base + swizzled(h ? row1 : row0, (col % 64) / 8);
            float lo, hi;
            unpack_bf16x2(lds32(addr), lo, hi);
            const __nv_bfloat162 merged =
                __floats2bfloat162_rn(fmaf(scaling, acc[2 * h], lo), fmaf(scaling, acc[2 * h + 1], hi));
            uint32_t out;
            memcpy(&out, &merged, 4);
            asm volatile("st.shared.b32 [%0], %1;" ::"r"(addr), "r"(out) : "memory");
        }
    }
}

template <typename T, bool kRowForm, bool kTForm, int kLoraSteps>
__device__ __forceinline__ void forms_body(const Maps& maps, int rows, int cols, int interleave,
                                           int32_t* __restrict__ sr, int32_t* __restrict__ st, int debug,
                                           const Lora& lora) {
    using L = Layout<T>;
    const uint32_t smem = smem_u32(smem_raw);
    const uint32_t bar = smem + L::kBarrier;
    const int cb = blockIdx.x, rb = blockIdx.y;
    const int64_t e = blockIdx.z;
    const int t = threadIdx.x;

    if (t == 0) {
        mbarrier_init(bar, 1);
        fence_async_shared();
        mbarrier_expect_tx(bar, L::kTileBytes);
        if (!(debug & 1)) {
#pragma unroll
            for (int sub = 0; sub < L::kSubs; ++sub) {
                tma_load_2d(smem + sub * kSub, &maps.in, cb * kTile + sub * L::kInner,
                            static_cast<int>(e * rows) + rb * kTile, bar);
            }
        }
    }
    if constexpr (kLoraSteps > 0) {
        // stage the factors of this tile (zero-padded to the MMA K step) while the TMA load is in flight
        const int rp = padded_rank(lora.rank);
        __nv_bfloat16* bs = reinterpret_cast<__nv_bfloat16*>(smem_raw + L::kLoraOffset);
        __nv_bfloat16* at = bs + kTile * rp;
        const __nv_bfloat16* b = lora.b + (e * rows + rb * kTile) * lora.rank;
        const __nv_bfloat16* a = lora.a + e * lora.rank * cols + cb * kTile;
        const __nv_bfloat16 zero = __float2bfloat16_rn(0.0f);
        for (int i = t; i < kTile * rp; i += kThreads) {
            const int x = i / rp, k = i % rp;  // tile row of B, K contiguous in global and shared memory
            bs[i] = k < lora.rank ? b[x * lora.rank + k] : zero;
            const int kt = i / kTile, xt = i % kTile;  // A row kt, columns contiguous in global memory
            at[xt * rp + kt] = kt < lora.rank ? a[kt * static_cast<int64_t>(cols) + xt] : zero;
        }
    }
    __syncthreads();
    if (!(debug & 1)) mbarrier_wait(bar, 0);
    if constexpr (kLoraSteps > 0) {
        merge_tile<kLoraSteps>(smem, smem_raw + L::kLoraOffset, lora.scaling, t);
        __syncthreads();
    }

    // Interleaved W1: the tile's rows land as the even (gate) or odd (up) 8-row groups of two output blocks.
    const int half = rows / 2;
    const int base = rb * kTile;
    const int group = base % half / 8, side = base / half;
    const int base_out = interleave ? group * 16 + side * 8 : base;

    if (!(debug & 2)) {
        if constexpr (kTForm) transposed_pass<T>(smem, t);
        if constexpr (kRowForm) {
            RowPass<T> row;
            row.read(smem, t);
            __syncthreads();  // every read of the input tile is done: box 0 becomes the row-form staging
            int out_row = base + row.r;
            if (interleave) out_row = (out_row % half / 8) * 16 + (out_row / half) * 8 + out_row % 8;
            const int slot = ((out_row >> 7) - (base_out >> 7)) * 128 + utccp_slot(out_row & 127);
            row.write(smem, smem + L::kWordsR + slot * 4);
        }
    }
    fence_async_shared();
    __syncthreads();
    if (debug & 4) return;

    if (t == 0) {
        if constexpr (kTForm) {
            tma_store_2d(&maps.out_t, rb * kTile, static_cast<int>(e * cols) + cb * kTile, smem + L::kOutT);
            bulk_store(st + (e * (rows / kTile) + rb) * cols + cb * kTile, smem + L::kWordsT, kWordBytes);
        }
        if constexpr (kRowForm) {
            int32_t* words = sr + (e * (cols / kTile) + cb) * rows;
            if (interleave) {
                tma_store_4d(&maps.out_r, cb * kTile, 0, side, static_cast<int>(e * (rows / 16)) + group, smem);
                // UTCCP slots (out_row % 32) in {o..o+7, 16+o..16+o+7} per output block: 128 contiguous bytes each.
                const int o = side * 8;
                const int block = base_out >> 7;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int segment = (k & 1) ? 16 + o : o;
                    bulk_store(words + (block + (k >> 1)) * 128 + segment * 4,
                               smem + L::kWordsR + ((k >> 1) * 128 + segment * 4) * 4, 128);
                }
            } else {
                tma_store_2d(&maps.out_r, cb * kTile, static_cast<int>(e * rows) + rb * kTile, smem);
                bulk_store(words + rb * kTile, smem + L::kWordsR, kWordBytes);
            }
        }
        bulk_commit();
        bulk_wait_read_all();
    }
}

template <typename T, bool kRowForm, bool kTForm>
__global__ void __launch_bounds__(kThreads, Layout<T>::kMinBlocks)
    forms_kernel(const __grid_constant__ Maps maps, int rows, int cols, int interleave, int32_t* __restrict__ sr,
                 int32_t* __restrict__ st, int debug) {
    forms_body<T, kRowForm, kTForm, 0>(maps, rows, cols, interleave, sr, st, debug, Lora{});
}

template <typename T, bool kRowForm, bool kTForm, int kLoraSteps>
__global__ void __launch_bounds__(kThreads, 3)
    lora_forms_kernel(const __grid_constant__ Maps maps, int rows, int cols, int interleave,
                      int32_t* __restrict__ sr, int32_t* __restrict__ st, const __grid_constant__ Lora lora) {
    forms_body<T, kRowForm, kTForm, kLoraSteps>(maps, rows, cols, interleave, sr, st, 0, lora);
}

// ---- host --------------------------------------------------------------------------------------

using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                 const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                 CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiled encode_tiled() {
    static EncodeTiled fn = nullptr;
    if (!fn) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult status;
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &status) ==
                cudaSuccess &&
            status == cudaDriverEntryPointSuccess) {
            fn = reinterpret_cast<EncodeTiled>(p);
        }
    }
    return fn;
}

// Tiled map over a row-major tensor; dims / strides / box innermost first, strides in bytes for dims 1..rank-1.
int encode(CUtensorMap* map, CUtensorMapDataType type, int rank, const void* address, const cuuint64_t* dims,
           const cuuint64_t* strides, const cuuint32_t* box, CUtensorMapSwizzle swizzle) {
    EncodeTiled fn = encode_tiled();
    if (!fn) return -3;
    cuuint32_t ones[5] = {1, 1, 1, 1, 1};
    CUresult r = fn(map, type, rank, const_cast<void*>(address), dims, strides, box, ones,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return r == CUDA_SUCCESS ? 0 : -4;
}

template <typename T, bool kRowForm, bool kTForm, int kLoraSteps = 0>
int launch(const void* w, int experts, int rows, int cols, int interleave, void* qr, int32_t* sr, void* qt,
           int32_t* st, int debug, cudaStream_t stream, Lora lora = Lora{nullptr, nullptr, 0, 0.0f}) {
    using L = Layout<T>;
    auto kernel = [] {
        if constexpr (kLoraSteps > 0) return lora_forms_kernel<T, kRowForm, kTForm, kLoraSteps>;
        else return forms_kernel<T, kRowForm, kTForm>;
    }();
    const int smem_bytes = kLoraSteps > 0 ? L::kLoraOffset + L::lora_bytes(lora.rank) : L::kSmemBytes;
    static int configured = 0;  // per instantiation: the largest dynamic shared memory set so far
    if (smem_bytes > configured) {
        cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return static_cast<int>(err);
        configured = smem_bytes;
    }
    Maps maps;
    memset(&maps, 0, sizeof(maps));
    const cuuint64_t e_rows = static_cast<cuuint64_t>(experts) * rows;
    {
        const cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), e_rows};
        const cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols) * sizeof(T)};
        const cuuint32_t box[2] = {static_cast<cuuint32_t>(L::kInner), kTile};
        const int rc = encode(&maps.in, sizeof(T) == 2 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                              2, w, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_128B);
        if (rc) return rc;
    }
    if (kTForm) {
        const cuuint64_t dims[2] = {static_cast<cuuint64_t>(rows), static_cast<cuuint64_t>(experts) * cols};
        const cuuint64_t strides[1] = {static_cast<cuuint64_t>(rows)};
        const cuuint32_t box[2] = {kTile, kTile};
        const int rc = encode(&maps.out_t, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, qt, dims, strides, box,
                              CU_TENSOR_MAP_SWIZZLE_128B);
        if (rc) return rc;
    }
    if (kRowForm) {
        int rc;
        if (interleave) {
            const cuuint64_t dims[4] = {static_cast<cuuint64_t>(cols), 8, 2, e_rows / 16};
            const cuuint64_t strides[3] = {static_cast<cuuint64_t>(cols), 8ull * cols, 16ull * cols};
            const cuuint32_t box[4] = {kTile, 8, 1, 16};
            rc = encode(&maps.out_r, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, qr, dims, strides, box,
                        CU_TENSOR_MAP_SWIZZLE_128B);
        } else {
            const cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), e_rows};
            const cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols)};
            const cuuint32_t box[2] = {kTile, kTile};
            rc = encode(&maps.out_r, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, qr, dims, strides, box,
                        CU_TENSOR_MAP_SWIZZLE_128B);
        }
        if (rc) return rc;
    }
    const dim3 grid(cols / kTile, rows / kTile, experts);
    if constexpr (kLoraSteps > 0) {
        kernel<<<grid, kThreads, smem_bytes, stream>>>(maps, rows, cols, interleave, sr, st, lora);
    } else {
        kernel<<<grid, kThreads, smem_bytes, stream>>>(maps, rows, cols, interleave, sr, st, debug);
    }
    return static_cast<int>(cudaGetLastError());
}

}  // namespace mono_ep_mxfp8_forms

// dtype: 0 = BF16, 1 = FP32. Returns a cudaError_t code; -1 unsupported form selection, -3 no
// cuTensorMapEncodeTiled entry point, -4 tensor-map encode failure. `debug` bits skip the load (1), the
// quantization (2) or the stores (4) for phase timing.
int mxfp8_forms_launch(uintptr_t w, int dtype, int experts, int rows, int cols, int row_form, int interleave,
                       int t_form, uintptr_t qr, uintptr_t sr, uintptr_t qt, uintptr_t st, int debug,
                       uintptr_t stream_ptr) {
    using namespace mono_ep_mxfp8_forms;
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto* wp = reinterpret_cast<const void*>(w);
    auto* qrp = reinterpret_cast<void*>(qr);
    auto* srp = reinterpret_cast<int32_t*>(sr);
    auto* qtp = reinterpret_cast<void*>(qt);
    auto* stp = reinterpret_cast<int32_t*>(st);
    if (experts == 0) return 0;
    const int which = (row_form ? 2 : 0) | (t_form ? 1 : 0);
#define MONO_EP_FORMS_CASE(TYPE, R, TF) \
    return launch<TYPE, R, TF>(wp, experts, rows, cols, interleave, qrp, srp, qtp, stp, debug, stream)
    if (dtype == 0) {
        switch (which) {
            case 3: MONO_EP_FORMS_CASE(__nv_bfloat16, true, true);
            case 2: MONO_EP_FORMS_CASE(__nv_bfloat16, true, false);
            case 1: MONO_EP_FORMS_CASE(__nv_bfloat16, false, true);
        }
    } else {
        switch (which) {
            case 3: MONO_EP_FORMS_CASE(float, true, true);
            case 2: MONO_EP_FORMS_CASE(float, true, false);
            case 1: MONO_EP_FORMS_CASE(float, false, true);
        }
    }
#undef MONO_EP_FORMS_CASE
    return -1;
}

// The forms of the merged BF16 weight W + scaling * B @ A (see the LoRA instantiation above): the row form
// (optionally interleaved) and the transposed form in one launch. Same return codes; -2 for a rank outside 1..128.
int mxfp8_lora_forms_launch(uintptr_t w, uintptr_t b, uintptr_t a, int rank, float scaling, int experts, int rows,
                            int cols, int interleave, uintptr_t qr, uintptr_t sr, uintptr_t qt, uintptr_t st,
                            uintptr_t stream_ptr) {
    using namespace mono_ep_mxfp8_forms;
    if (experts == 0) return 0;
    if (rank < 1 || rank > 128) return -2;
    const Lora lora{reinterpret_cast<const __nv_bfloat16*>(b), reinterpret_cast<const __nv_bfloat16*>(a), rank,
                    scaling};
#define MONO_EP_LORA_FORMS_CASE(STEPS)                                                                        \
    case STEPS:                                                                                               \
        return launch<__nv_bfloat16, true, true, STEPS>(                                                      \
            reinterpret_cast<const void*>(w), experts, rows, cols, interleave, reinterpret_cast<void*>(qr),  \
            reinterpret_cast<int32_t*>(sr), reinterpret_cast<void*>(qt), reinterpret_cast<int32_t*>(st), 0, \
            reinterpret_cast<cudaStream_t>(stream_ptr), lora)
    switch (lora_steps(rank)) {
        MONO_EP_LORA_FORMS_CASE(1);
        MONO_EP_LORA_FORMS_CASE(2);
        MONO_EP_LORA_FORMS_CASE(4);
        MONO_EP_LORA_FORMS_CASE(8);
    }
#undef MONO_EP_LORA_FORMS_CASE
    return -1;
}
