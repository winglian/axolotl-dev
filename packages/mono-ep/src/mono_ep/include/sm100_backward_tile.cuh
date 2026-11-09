// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <cuda_bf16.h>
#include <cutlass/arch/barrier.h>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace mono_ep {

struct alignas(1024) Sm100BackwardTile {
    __nv_bfloat16 a[2][64 * 64];
    __nv_bfloat16 b[2][128 * 64];
    cutlass::arch::ClusterTransactionBarrier complete;
    unsigned int tmem;
    unsigned int parity;

    __device__ void initialize() {
        if (threadIdx.x == 0) {
            complete.init(1);
            parity = 0;
            cutlass::arch::fence_barrier_init();
        }
        if (threadIdx.x < 32)
            cute::TMEM::Allocator1Sm().allocate(128, &tmem);
        __syncthreads();
    }

    __device__ void release() {
        __syncthreads();
        if (threadIdx.x < 32)
            cute::TMEM::Allocator1Sm().free(tmem, 128);
    }

    // The alternate buffer is filled while UMMA consumes the current buffer.
    template <unsigned int Columns = 64, class LoadA, class LoadB>
    __device__ void multiply(unsigned int k_size, LoadA load_a, LoadB load_b,
                             float* output) {
        static_assert(Columns == 64 || Columns == 128);
        using namespace deep_gemm;
        constexpr auto major = cute::UMMA::Major::K;
        auto instruction = cute::UMMA::make_instr_desc<
            cutlass::bfloat16_t, cutlass::bfloat16_t, float, 64, Columns, major, major>();
        auto stage = [&](unsigned int buffer, unsigned int k) {
            for (unsigned int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
                const auto row = i / 64, col = i % 64;
                const auto swizzled = i ^ ((row & 7) << 3);
                a[buffer][swizzled] = load_a(row, k + col);
            }
            for (unsigned int i = threadIdx.x; i < Columns * 64; i += blockDim.x) {
                const auto row = i / 64, col = i % 64;
                b[buffer][i ^ ((row & 7) << 3)] = load_b(row, k + col);
            }
        };
        stage(0, 0);
        for (unsigned int k = 0; k < k_size; k += 64) {
            const auto buffer = (k / 64) & 1;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            ptx::tcgen05_after_thread_sync();
            if (threadIdx.x == 0) {
                auto da = mma::sm100::make_umma_desc<major, 64, 64, 128>(a[buffer], 0, 0);
                auto db = mma::sm100::make_umma_desc<major, Columns, 64, 128>(b[buffer], 0, 0);
                for (unsigned int inner = 0; inner < 64; inner += 16) {
                    auto aa = da, bb = db;
                    aa.lo = mma::sm100::advance_umma_desc_lo<major, 64, 128, __nv_bfloat16>(da.lo, 0, inner);
                    bb.lo = mma::sm100::advance_umma_desc_lo<major, Columns, 128, __nv_bfloat16>(db.lo, 0, inner);
                    ptx::SM100_MMA_F16BF16_SS::fma(aa, bb, tmem, k || inner,
                        static_cast<uint64_t>(static_cast<uint32_t>(instruction)) << 32);
                }
            }
            // CUTLASS elects the issuer internally and requires a converged warp.
            if (threadIdx.x < 32)
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&complete));
            if (k + 64 < k_size) stage(buffer ^ 1, k + 64);
            const auto wait_start = clock64();
            while (!complete.try_wait(parity)) {
                if (clock64() - wait_start > 10000000000ull) {
                    printf("Owned backward UMMA timeout: CTA=%u K=%u parity=%u\n",
                           blockIdx.x, k, parity);
                    asm volatile("trap;");
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) parity ^= 1;
            __syncthreads();
        }
        ptx::tcgen05_after_thread_sync();
        if (threadIdx.x < 128) {
            for (unsigned int col = 0; col < Columns; col += 4) {
                uint32_t v[4];
                cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem + col, v[0], v[1], v[2], v[3]);
                cutlass::arch::fence_view_async_tmem_load();
                // M64 UMMA occupies the first 16 datapaths of each warp.
                if (threadIdx.x % 32 < 16) {
                    const auto row = (threadIdx.x / 32) * 16 + threadIdx.x % 16;
                    for (unsigned int i = 0; i < 4; ++i)
                        output[row * Columns + col + i] = __uint_as_float(v[i]);
                }
            }
        }
        __syncthreads();
    }
};

} // namespace mono_ep
