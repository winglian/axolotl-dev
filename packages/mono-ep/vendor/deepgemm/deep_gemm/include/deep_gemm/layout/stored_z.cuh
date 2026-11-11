// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once
#include <cstdint>

namespace deep_gemm::layout {

// Stored forward context (SM100 BF16 / MXFP8 backward, option bit 4096), one byte buffer:
//   int32 header [overflow flag, capacity rows, 0, 0]
//   int32 index  [ranks * max tokens * top-k]: source (rank, token, slot) -> forward pool row
//   BF16 z       [capacity rows, 2I], 256-byte aligned: gate then up of each forward pool row
// The forward writes z for pool rows below the capacity and sets the flag for any row past it; a backward that
// sees the flag recomputes z as without stored context.
static constexpr uint32_t kStoredZHeaderInts = 4;

__host__ __device__ constexpr uint64_t stored_z_meta_bytes(uint64_t ranks, uint64_t max_tokens, uint64_t topk) {
    return (kStoredZHeaderInts * 4 + ranks * max_tokens * topk * 4 + 255) / 256 * 256;
}

}  // namespace deep_gemm::layout
