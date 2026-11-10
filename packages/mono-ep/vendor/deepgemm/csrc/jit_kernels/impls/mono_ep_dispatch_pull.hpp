// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include "../../runtime/runtime.hpp"

namespace deep_gemm::mono_ep_pull {

// Host mirrors of the SM90 MonoEP `SharedStorage` layouts (member order and alignment as in the
// kernels), used only to size the dispatch pull. A mismatch moves the pull decision, not the kernel: an
// overestimate caps a pull that would have fit, an underestimate leaves the kernel's static assert to fire.
constexpr int kBlock = 128, kBarrierBytes = 8, kTaskInfoBytes = 32, kNumScheduleStages = 2;
constexpr int kMaxPullBytes = 4096, kCappedPullBytes = 2048;

inline int align_to(const int bytes, const int alignment) {
    return (bytes + alignment - 1) / alignment * alignment;
}

inline int common_tail_bytes(int bytes, const int stages, const int dispatch_warps, const int epilogue_warps) {
    bytes = align_to(bytes, 1024) + kBlock * kBlock;  // quantized_a
    bytes += kBlock * 4;                              // activation_scales
    bytes = align_to(bytes, 16) + kNumScheduleStages * kTaskInfoBytes;
    // dispatch, full, empty, combine (2 per epilogue warp), task-info full and empty barriers
    return bytes + (dispatch_warps + 2 * stages + 2 * epilogue_warps + 2 * kNumScheduleStages) * kBarrierBytes;
}

// sm90_fp8_mono_ep_forward.cuh: 226,304 B at hidden 2048 with 5 FP8 stages and a 2 KB pull.
inline int forward_smem_bytes(const int hidden, const int experts, const int pull, const int stages,
                              const bool fp8_only_stages, const int dispatch_warps, const int epilogue_warps) {
    const int pull_buffer = std::max(pull, align_to(hidden / 32, 16));
    int bytes = experts * 4;                                                        // expert_token_count
    bytes = align_to(bytes, 1024) + dispatch_warps * pull_buffer;                   // dispatch_send_buffer
    bytes = align_to(bytes, 1024) + kBlock * kBlock * 2;                            // smem_d (BF16 raw tile)
    bytes = align_to(bytes, 1024) + stages * kBlock * kBlock * (fp8_only_stages ? 1 : 2);  // smem_a
    bytes = align_to(bytes, 1024) + stages * kBlock * kBlock;                       // smem_b
    // cached_activation_scales sits between quantized_a and activation_scales; its size keeps 16-B alignment
    bytes = common_tail_bytes(bytes, stages, dispatch_warps, epilogue_warps) + stages * kBlock * 4;
    return align_to(bytes, 1024);
}

// sm90_fp8_mono_ep_backward.cuh: 226,304 B at hidden 2048 with 3 FP8 stages, the vector x/dy pack and
// a 2 KB pull; the BF16-transport layout (2 BF16 stages, no x/dy stage) is 200,704 B with a 4 KB pull.
inline int backward_smem_bytes(const int experts, const int experts_per_rank, const int pull, const int stages,
                               const bool fp8_only_stages, const bool vector_xdy_pack, const bool dispatch_xdy_pack,
                               const int mainloop_mode, const int dispatch_warps, const int epilogue_warps) {
    int bytes = experts * 4 + experts_per_rank * 4;                   // expert_token_count, expert_padded_counts
    bytes = align_to(bytes, 8) + kBarrierBytes;                       // packed_metadata_barrier
    bytes = align_to(bytes, 1024) + dispatch_warps * pull;            // dispatch_send_buffer
    bytes = align_to(bytes, 1024) + kBlock * kBlock * 2;              // smem_d (BF16 raw tile)
    bytes = align_to(bytes, 128) + kBlock * kBlock * 2;               // backward_epilogue (BF16 dH tile)
    bytes += 2 * kBlock * 4;                                          // dscore
    bytes = align_to(bytes, 1024) + stages * kBlock * kBlock * (fp8_only_stages ? 1 : 2);  // smem_a
    bytes = align_to(bytes, 1024) + stages * kBlock * kBlock;         // smem_b
    bytes = common_tail_bytes(bytes, stages, dispatch_warps, epilogue_warps);
    const int queue = dispatch_xdy_pack ? 16 : 1;
    bytes += 2 * queue * kBarrierBytes;                               // xdy_queue_full, xdy_queue_empty
    bytes = align_to(bytes, 16) + queue * 16;                         // xdy_queue
    const int pack_buffer = align_to(2 * 64 * 64 + 64 + 129 * 4, 128);
    bytes = align_to(bytes, 128) + (dispatch_xdy_pack ? dispatch_warps * pack_buffer : 16);  // xdy_pack_buffer
    bytes = align_to(bytes, 128) + (vector_xdy_pack ? 2 * kBlock * 128 + 2 * kBlock * 4 : 16);  // xdy_stage
    bytes += (mainloop_mode & 1) ? stages * kBlock * 4 : 4;           // cached_activation_scales
    return align_to(bytes, 1024);
}

// Round-4 rule: the largest power of two up to 4 KB that divides the row.
inline int default_pull_bytes(const int row_bytes) {
    int pull = kMaxPullBytes;
    while (row_bytes % pull != 0) pull /= 2;
    return pull;
}

// `MONO_EP_SM90_FP8_DISPATCH_PULL_BYTES` when set; otherwise the round-4 pull, halved to 2 KB (two pulls
// per row) when `fits(pull)` says the layout would exceed the SM90 capacity with the round-4 value.
template <typename Fits>
inline int select_pull_bytes(const int row_bytes, const Fits& fits) {
    const int default_pull = default_pull_bytes(row_bytes);
    const char* selector = std::getenv("MONO_EP_SM90_FP8_DISPATCH_PULL_BYTES");
    if (selector != nullptr and *selector != '\0') {
        const int pull = std::atoi(selector);
        DG_HOST_ASSERT(pull >= 16 and pull <= kMaxPullBytes and pull % 16 == 0 and row_bytes % pull == 0);
        return pull;
    }
    int pull = default_pull;
    while (pull > kCappedPullBytes and not fits(pull))
        pull /= 2;
    return pull;
}

} // namespace deep_gemm::mono_ep_pull
