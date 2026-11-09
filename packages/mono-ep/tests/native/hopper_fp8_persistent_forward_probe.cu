// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#include <cuda.h>
#include <cuda_runtime.h>
#include <deep_gemm/impls/sm90_fp8_mono_ep_forward.cuh>
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
    int rank, int tokens, int rows, uintptr_t stream) {
    constexpr int H = 2048, I = 768, E = 8;
    const auto buffer = deep_gemm::layout::MonoEPBuffer(
        reinterpret_cast<void*>(sym[rank]), H, I, 2, 16, @MAX_T@, 4, @RING@, 0,
        false, 0, true);
    auto bf16_acts = [&](uintptr_t ptr, uint64_t k) {
        return descriptor(ptr, 2, {k, static_cast<uint64_t>(rows)}, {k * 2}, {64, 128});
    };
    auto fp8_acts = [&](uintptr_t ptr, uint64_t k) {
        return descriptor(ptr, 2, {k, static_cast<uint64_t>(rows)}, {k}, {128, 128}, true);
    };
    auto weights = [&](uintptr_t ptr, uint64_t n, uint64_t k) {
        return descriptor(ptr, 4, {k, n / 2, 2, E}, {k, n / 2 * k, n * k}, {128, 128, 1, 1}, true);
    };
    const auto x = fp8_acts(reinterpret_cast<uintptr_t>(buffer.l1_token_buffer.base), H);
    const auto h = bf16_acts(reinterpret_cast<uintptr_t>(buffer.l2_token_buffer.base), I);
    const auto h_cached = fp8_acts(reinterpret_cast<uintptr_t>(buffer.l2_cached_token_buffer.base), I);
    const auto h_out = descriptor(reinterpret_cast<uintptr_t>(buffer.l2_token_buffer.base),
        2, {I, static_cast<uint64_t>(rows)}, {I * 2}, {64, 32});
    const auto w1 = weights(p[1], 2 * I, H), w2 = weights(p[2], H, I);
    auto kernel = &deep_gemm::sm90_fp8_mono_ep_forward_impl<
        @MAX_T@, H, I, 16, 0, 4, 128, 128, 128, 32,
        @RING@, 2, 2048, @DISPATCH@, @NON_EPILOGUE@, 256, 132, 2,
        0x7f7fffffu, false, @COMPACT@>;
    int device, smem;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    auto error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (error != cudaSuccess) throw std::runtime_error(std::string("cudaFuncSetAttribute: ") + cudaGetErrorString(error));
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
    error = cudaLaunchKernelEx(&config, kernel,
        reinterpret_cast<void*>(p[0]), static_cast<int*>(nullptr), static_cast<uint32_t>(tokens),
        deep_gemm::layout::SymBuffer<2>(sym, rank),
        x, w1, h_out, h, h_cached, w2, x, w1, h_out, h, w2,
        static_cast<const uint8_t*>(nullptr), reinterpret_cast<__nv_bfloat16*>(p[5]),
        reinterpret_cast<const float*>(p[3]), reinterpret_cast<const float*>(p[4]));
    if (error != cudaSuccess) throw std::runtime_error(std::string("cudaLaunchKernelEx: ") + cudaGetErrorString(error));
    return error;
}
