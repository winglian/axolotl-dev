// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
//
// Chunking of the combine stage of the mono-ep persistent kernels: each epilogue warp reduces one
// token's top-k rows chunk by chunk through three shared-memory slots (two TMA load stages and one
// store stage) and a per-lane float register buffer.
#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>

namespace deep_gemm::mono_ep_combine {

// The smallest divisor of the hidden size whose chunk fits the reusable shared memory, keeps the
// per-lane float buffer within `kNumMaxRegistersForBuffer`, and is one 16-byte `uint4` per lane
// per iteration (so a multiple of 512 bytes); 0 when no divisor does.
template <uint32_t kHidden, uint32_t kNumHiddenBytes,
          uint32_t kNumChunkSlots, uint32_t kNumEpilogueWarps,
          uint32_t kNumReusableSmemBytes, uint32_t kNumMaxRegistersForBuffer>
CUTLASS_HOST_DEVICE constexpr uint32_t get_num_chunks() {
    for (uint32_t num_chunks = 1; num_chunks <= kHidden; ++ num_chunks) {
        if (kHidden % num_chunks != 0 or kNumHiddenBytes % num_chunks != 0)
            continue;
        const uint32_t num_chunk_bytes = kNumHiddenBytes / num_chunks;
        const bool fits_smem = kNumChunkSlots * kNumEpilogueWarps * num_chunk_bytes <= kNumReusableSmemBytes;
        const bool fits_registers = kHidden / num_chunks <= 32 * kNumMaxRegistersForBuffer;
        const bool one_uint4_per_lane = num_chunk_bytes % (16 * 32) == 0;
        if (fits_smem and fits_registers and one_uint4_per_lane)
            return num_chunks;
    }
    return 0;
}

} // namespace deep_gemm::mono_ep_combine
