// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#include <cuda.h>
#include <cuda_runtime.h>
#include <deep_gemm/impls/sm100_bf16_mono_ep_backward.cuh>
#include <stdexcept>
#include <vector>

static cute::TmaDescriptor descriptor(uintptr_t pointer, int rank,
    std::vector<uint64_t> shape, std::vector<uint64_t> strides,
    std::vector<uint32_t> box) {
    cute::TmaDescriptor result;
    std::vector<uint32_t> elements(rank, 1);
    const auto error = cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap*>(&result),
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, reinterpret_cast<void*>(pointer),
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
        return descriptor(ptr, 2, {k, static_cast<uint64_t>(rows)}, {k * 2}, {64, 64});
    };
    auto weights = [&](uintptr_t ptr, uint64_t n, uint64_t k) {
        return descriptor(ptr, 4, {k, n / 2, 2, E}, {k * 2, n / 2 * k * 2, n * k * 2}, {64, 128, 1, 1});
    };
    const auto x = acts(p[2], H), gy = acts(p[3], H), dz = acts(p[5], 2 * I);
    const auto w1 = weights(p[6], 2 * I, H), w2t = weights(p[7], I, H), w1t = weights(p[8], H, 2 * I);
    deep_gemm::layout::MoEBackwardBuffers buffers{};
    buffers.dy = deep_gemm::layout::SymBuffer<>(dy, rank);
    buffers.dscore_partials = deep_gemm::layout::SymBuffer<>(ds, rank);
    buffers.x_pool = reinterpret_cast<__nv_bfloat16*>(p[2]);
    buffers.dy_pool = reinterpret_cast<__nv_bfloat16*>(p[3]);
    buffers.h_pool = reinterpret_cast<__nv_bfloat16*>(p[4]);
    buffers.dz_pool = reinterpret_cast<__nv_bfloat16*>(p[5]);
    buffers.dscore = reinterpret_cast<float*>(p[1]);
    buffers.expert_starts = reinterpret_cast<uint32_t*>(p[9]);
    buffers.expert_counts = reinterpret_cast<uint32_t*>(p[10]);
    auto kernel = &deep_gemm::sm100_bf16_mono_ep_backward_impl<
        @MAX_T@, H, I, 16, 0, 4, 128, 128, 64, 32,
        @RING@, @STAGES@, @PULL@, 128, 128, 256, 128, 2, true, false, 0u, @OPTIONS@u>;
    int device, smem;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    auto error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (error != cudaSuccess) return error;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {2, 1, 1};
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(128);
    config.blockDim = dim3(512);
    config.dynamicSmemBytes = smem;
    config.stream = reinterpret_cast<cudaStream_t>(stream);
    config.attrs = &attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel,
        reinterpret_cast<void*>(p[0]), static_cast<int*>(nullptr), static_cast<uint32_t>(tokens),
        deep_gemm::layout::SymBuffer<2>(sym, rank),
        x, w1, x, dz, w1t, x, w1, x, dz, w1t,
        static_cast<const uint8_t*>(nullptr), static_cast<__nv_bfloat16*>(nullptr), gy, w2t, buffers);
}
