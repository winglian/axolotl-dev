// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#include <cuda.h>
#include <cuda_runtime.h>
#include <deep_gemm/impls/sm90_fp8_mono_ep_backward.cuh>
#include <stdexcept>
#include <vector>

static cute::TmaDescriptor descriptor(uintptr_t pointer, int rank,
    std::vector<uint64_t> shape, std::vector<uint64_t> strides,
    std::vector<uint32_t> box, bool fp8 = false) {
    cute::TmaDescriptor result;
    std::vector<uint32_t> elements(rank, 1);
    const auto error = cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap*>(&result),
        fp8 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, reinterpret_cast<void*>(pointer),
        shape.data(), strides.data(), box.data(), elements.data(),
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (error != CUDA_SUCCESS) throw std::runtime_error("TMA descriptor error " + std::to_string(error));
    return result;
}

int launch(std::vector<uintptr_t> p, std::vector<int64_t> sym,
    std::vector<int64_t> dy, std::vector<int64_t> ds,
    int rank, int tokens, int rows, uintptr_t stream) {
    constexpr int H = 2048, I = 768, E = 8;
    auto acts = [&](uintptr_t ptr, uint64_t k) {
        return descriptor(ptr, 2, {k, static_cast<uint64_t>(rows)}, {k * 2}, {64, 128});
    };
    auto cached_acts = [&](uintptr_t ptr, uint64_t k) {
        return descriptor(ptr, 2, {k, static_cast<uint64_t>(rows)}, {k}, {128, 128}, true);
    };
    auto weights = [&](uintptr_t ptr, uint64_t n, uint64_t k) {
        return descriptor(ptr, 4, {k, n / 2, 2, E}, {k, n / 2 * k, n * k}, {128, 128, 1, 1}, true);
    };
    const auto x = @CACHED@ ? cached_acts(p[14], H) : acts(p[2], H);
    const auto gy = @CACHED@ ? cached_acts(p[15], H) : acts(p[3], H);
    const auto dz = @CACHE_DZ@ ? cached_acts(p[18], 2 * I) : acts(p[5], 2 * I);
    const auto w1 = weights(p[6], 2 * I, H), w2t = weights(p[7], I, H), w1t = weights(p[8], H, 2 * I);
    deep_gemm::layout::MoEBackwardBuffers buffers{
        deep_gemm::layout::SymBuffer<>(dy, rank), deep_gemm::layout::SymBuffer<>(ds, rank),
        reinterpret_cast<__nv_bfloat16*>(p[2]), reinterpret_cast<__nv_bfloat16*>(p[3]),
        reinterpret_cast<__nv_bfloat16*>(p[4]), reinterpret_cast<__nv_bfloat16*>(p[5]),
        reinterpret_cast<float*>(p[1]), reinterpret_cast<uint32_t*>(p[9]), reinterpret_cast<uint32_t*>(p[10])};
    if constexpr (@CACHED@) {
        buffers.qx_pool = reinterpret_cast<uint8_t*>(p[14]);
        buffers.qdy_pool = reinterpret_cast<uint8_t*>(p[15]);
        buffers.sx_pool = reinterpret_cast<float*>(p[16]);
        buffers.sdy_pool = reinterpret_cast<float*>(p[17]);
    }
    if constexpr (@CACHE_DZ@) {
        buffers.qdz_pool = reinterpret_cast<uint8_t*>(p[18]);
        buffers.sdz_pool = reinterpret_cast<float*>(p[19]);
    }
    auto kernel = &deep_gemm::sm90_fp8_mono_ep_backward_impl<
        @MAX_T@, H, I, 16, 0, 4, 128, 128, 128, 32,
        @RING@, 2, 4096, @DISPATCH@, @NON_EPILOGUE@, 256, 132, 2, true, @CACHED@, @CACHE_DZ@, @COMPACT@>;
    int device, smem;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    auto error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (error != cudaSuccess) return error;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {1, 1, 1};
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(132);
    config.blockDim = dim3(@THREADS@);
    config.dynamicSmemBytes = smem;
    config.stream = reinterpret_cast<cudaStream_t>(stream);
    config.attrs = &attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel,
        reinterpret_cast<void*>(p[0]), static_cast<int*>(nullptr), static_cast<uint32_t>(tokens),
        deep_gemm::layout::SymBuffer<2>(sym, rank),
        x, w1, x, dz, w1t, x, w1, x, dz, w1t,
        static_cast<const uint8_t*>(nullptr), static_cast<__nv_bfloat16*>(nullptr), gy, w2t, buffers,
        reinterpret_cast<const float*>(p[11]), reinterpret_cast<const float*>(p[12]), reinterpret_cast<const float*>(p[13]));
}
