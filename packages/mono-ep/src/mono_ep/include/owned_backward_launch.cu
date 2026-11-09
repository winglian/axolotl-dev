// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI

#include <sm100_owned_backward.cuh>
#include <vector>
int launch(std::vector<uintptr_t> p, int experts, int hidden, int inner,
           int workers, uintptr_t stream_ptr) {
    using namespace mono_ep;
    auto error = cudaFuncSetAttribute(sm100_owned_backward,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(OwnedBackwardStorage));
    if (error != cudaSuccess) return static_cast<int>(error);
    sm100_owned_backward<<<workers, 128, sizeof(OwnedBackwardStorage),
        reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
        reinterpret_cast<__nv_bfloat16*>(p[0]), reinterpret_cast<__nv_bfloat16*>(p[1]),
        reinterpret_cast<__nv_bfloat16*>(p[2]), reinterpret_cast<__nv_bfloat16*>(p[3]),
        reinterpret_cast<float*>(p[4]), reinterpret_cast<unsigned int*>(p[5]),
        reinterpret_cast<unsigned int*>(p[6]), reinterpret_cast<unsigned int*>(p[7]),
        reinterpret_cast<unsigned int*>(p[8]), reinterpret_cast<__nv_bfloat16*>(p[9]),
        reinterpret_cast<__nv_bfloat16*>(p[10]), reinterpret_cast<__nv_bfloat16*>(p[11]),
        reinterpret_cast<float*>(p[12]), experts, hidden, inner);
    return static_cast<int>(cudaGetLastError());
}
