// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <cstdint>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mono_ep.cuh>
#include <deep_gemm/layout/moe_backward.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/sm100_moe_backward.cuh>
#include <deep_gemm/scheduler/mono_ep.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t kNumRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    bool kOwnMBlocks = true,
    bool kCacheActivations = false,
    bool kCacheDz = false,
    bool kCompactNonMath = false,
    bool kTransportFP8 = false,
    bool kFP8Combine = false,
    bool kPhaseTrace = false,
    uint32_t kDiagAblate = 0,
    bool kWarpReduceDscore = false,
    bool kVectorXdyPack = false,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden * 2,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_mono_ep_backward_impl(void* y,
                         int* cumulative_local_expert_recv_stats,
                         const uint32_t num_tokens,
                         const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                         const uint8_t* sm_locality_domains, __nv_bfloat16* saved_z,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_dy,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_dh_weights,
                         const __grid_constant__ layout::MoEBackwardBuffers backward,
                         const float* w1_scales, const float* w2t_scales, const float* w1t_scales) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(not kHasShared and kOwnMBlocks, "Backward requires owned routed blocks");
    DG_STATIC_ASSERT(not kCacheDz or kCacheActivations, "dZ cache requires cached activation staging");
    DG_STATIC_ASSERT(not kVectorXdyPack or (kTransportFP8 and kCacheActivations),
                     "Vectorized x/dy pack reads the cached FP8 pools");
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr uint32_t kNumLocalityDomains = 1;

    // Template checks
    DG_STATIC_ASSERT(kCompactNonMath ? kNumDispatchThreads == 32 : kNumDispatchThreads % 128 == 0,
                     "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kCompactNonMath ? kNumNonEpilogueThreads == 96 : kNumNonEpilogueThreads == 128,
                     "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    // Thread indices
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_dy);
        cute::prefetch_tma_descriptor(&tensor_map_dh_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
    }

    // Workspaces and Buffer
    const auto buffer = layout::MonoEPBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, 0,
        /*with_sf=*/ false,
        kNumSharedExperts,
        kTransportFP8,
        kFP8Combine
    );
    DG_STATIC_ASSERT(not kFP8Combine or (kTransportFP8 and kNumSharedExperts == 0),
                     "FP8 combine requires routed-only FP8 transport");
    const auto workspace = buffer.workspace;

    using L2KBlockDependency = sched::L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>;

    // Data types
    using a_dtype_t = cutlass::bfloat16_t;
    using b_dtype_t = __nv_fp8_e4m3;
    using d_dtype_t = cutlass::bfloat16_t;

    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M == 128, "SM90 backward requires block M 128");
    DG_STATIC_ASSERT(BLOCK_N == 128, "SM90 backward requires block N 128");
    DG_STATIC_ASSERT(BLOCK_K == 128, "SM90 FP8 backward requires block K 128");
    DG_STATIC_ASSERT(kNumEpilogueThreads == 256, "SM90 backward requires 256 math threads");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N * sizeof(nv_bfloat16) % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Scheduler configs
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kNumScheduleConsumerThreads = 64 + kNumEpilogueThreads;

    // Shared memory sizes
    // NOTES: BF16 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;

    // Assign shared memory
    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        uint32_t expert_padded_counts[kNumExpertsPerRank];
        Barrier packed_metadata_barrier;
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment) d_dtype_t l1[kNumEpilogueWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) d_dtype_t l2[kNumEpilogueWarpgroups][STORE_BLOCK_M * BLOCK_N];
            alignas(kSharedMemoryAlignment) d_dtype_t raw[BLOCK_M * BLOCK_N];
        } smem_d;
        epilogue::MoEBackwardTileEpilogue<BLOCK_M> backward_epilogue;
        float dscore[BLOCK_M];
        alignas(kSharedMemoryAlignment) a_dtype_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K];
        alignas(kSharedMemoryAlignment) b_dtype_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K];
        alignas(kSharedMemoryAlignment) __nv_fp8_e4m3 quantized_a[BLOCK_M * BLOCK_K];
        float activation_scales[BLOCK_M];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
        alignas(128) uint8_t xdy_stage[kVectorXdyPack ? 2 * BLOCK_M * 128 + 2 * BLOCK_M * sizeof(float) : 16];
    };
    DG_STATIC_ASSERT(sizeof(SharedStorage) <= 232448, "SM90 backward exceeds Hopper shared memory capacity");
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);

    // Send buffers
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    __syncthreads();

    // Initialization
    if (warp_idx == 0) {
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            shared_storage.expert_token_count[i] = 0;
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            shared_storage.packed_metadata_barrier.init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                shared_storage.full_barriers[i].init(2);
                shared_storage.empty_barriers[i].init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                shared_storage.combine_barriers[i].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
                shared_storage.task_info_full_barriers[i].init(1);
                shared_storage.task_info_empty_barriers[i].init(kNumScheduleConsumerThreads);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Diagnostic per-CTA phase trace (kPhaseTrace only); the pointer rides in the unused saved_z argument.
    constexpr uint32_t kTraceHeader = 64, kTraceMBlockSlots = 64, kTraceMBlockFields = 16;
    constexpr uint32_t kTraceCTAStride = kTraceHeader + kTraceMBlockSlots * kTraceMBlockFields;
    unsigned long long* const phase_trace = kPhaseTrace
        ? reinterpret_cast<unsigned long long*>(saved_z) + static_cast<uint64_t>(sm_idx) * kTraceCTAStride : nullptr;
    const auto trace_clock = [&]() -> uint64_t {
        if constexpr (kPhaseTrace) return clock64(); else return 0;
    };
    const auto trace_add = [&](const uint32_t& slot, const uint64_t& value) {
        if constexpr (kPhaseTrace) atomicAdd(phase_trace + slot, static_cast<unsigned long long>(value));
    };
    const auto trace_set = [&](const uint32_t& slot, const uint64_t& value) {
        if constexpr (kPhaseTrace) phase_trace[slot] = value;
    };
    const auto trace_globaltimer = [&]() -> uint64_t {
        uint64_t t = 0;
        if constexpr (kPhaseTrace) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        return t;
    };
    if constexpr (kPhaseTrace) {
        if (thread_idx == 0) {
            trace_set(0, trace_globaltimer());
            trace_set(1, trace_clock());
        }
    }

    // Task scheduler
    auto scheduler = sched::MonoEPScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks, kNumLocalityDomains,
        kNumSharedExperts, 1>(
            workspace,
            shared_storage.task_info_full_barriers,
            shared_storage.task_info_empty_barriers,
            shared_storage.task_infos
    );

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 2;

    // Adjust registers
    // NOTES: more experts per rank will cost more schedulers' registers
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
    constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 48 : 96;
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 88;
    constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 208 : 160;
    if constexpr (kCompactNonMath) {
        DG_STATIC_ASSERT(kUseMoreEpilogueRegisters,
                         "Compact non-math WG requires at most 64 experts per rank");
        DG_STATIC_ASSERT(kNumRanks <= 32, "Compact non-math WG requires at most 32 ranks");
        DG_STATIC_ASSERT(88 * 128 + kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                         "Too many compact-WG registers");
    } else {
        DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                         kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                         kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                         "Too many registers");
    }

    // setmaxnreg must execute uniformly across each warpgroup.
    if constexpr (kCompactNonMath) {
        if (warp_idx < 4)
            cutlass::arch::warpgroup_reg_dealloc<88>();
        else
            cutlass::arch::warpgroup_reg_alloc<208>();
    } else {
        if (warp_idx < kNumDispatchWarps)
            cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
        else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps)
            cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        else
            cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
    }

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            // TODO: figure out better unrolling
            // Now, `unroll` is better than `unroll 8`
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[i * kNumTopk + lane_idx]);
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count experts' tokens
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offset (~6.5 us)
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices (~2 us with 512 tokens)
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // Grid sync
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Write expert count
        if (sm_idx == 0) {
            // Push this launch's grid index to every peer for tagging combine readiness; +1 differs from a zeroed workspace
            DG_STATIC_ASSERT(kNumRanks <= kNumDispatchThreads, "Insufficient threads for the grid index push");
            if (thread_idx < kNumRanks)
                *sym_buffer.map(workspace.get_peer_grid_idx_ptr(sym_buffer.rank_idx), thread_idx) = ptx::get_grid_idx() + 1;
            __syncwarp();

            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Barrier before pulling
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Pull token data and SF from remote ranks into local L1 buffer
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = &shared_storage.dispatch_barriers[warp_idx];

        // Per-rank counts for current expert (re-loaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        // Wait token data arrival
        scheduler.fetch_expert_recv_count();

        // Publish the physical grouped-K layout before any block becomes consumable.
        if (warp_idx == 0) {
            for (uint32_t i = 0; i < kNumExpertsPerRank; ++i) {
                const uint32_t count = scheduler.get_num_tokens(i);
                const uint32_t start = scheduler.get_pool_block_offset(i) * BLOCK_M;
                if (lane_idx == 0) {
                    const uint32_t padded = math::align(count, BLOCK_M);
                    shared_storage.expert_padded_counts[i] = padded;
                    if (sm_idx == 0) {
                        backward.expert_starts[i] = start;
                        backward.expert_counts[i] = count;
                        if (backward.expert_padded_counts != nullptr)
                            backward.expert_padded_counts[i] = padded;
                    }
                }
            }
            if (lane_idx == 0) {
                __threadfence_block();
                shared_storage.packed_metadata_barrier.arrive();
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        const uint64_t trace_pull_start = trace_clock();
        uint64_t trace_ring_wait = 0, trace_load_wait = 0, trace_store_wait = 0, trace_rows = 0;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            // Advance expert until within the range
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;

                // Update pool block offset for the new expert
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

                // Move start and end to the next expert
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            // Finish all tokens
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // Load per-rank counts when expert changes
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    // TODO: this is not coalesced
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection via iterative min-peeling
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                // Compute active count and min across all ranks
                // NOTES: reduce within each lane first, then warp-reduce once
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                // Hit in the current round
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Move into the next round
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            // Read source token-topk index (written by remote dispatch via NVLink)
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // Hidden bytes are divided into chunks
            constexpr uint32_t kHiddenBytes = kHidden * (kTransportFP8 ? 1 : sizeof(nv_bfloat16));
            constexpr uint32_t kNumChunks = kHiddenBytes / kNumBytesPerPull;
            DG_STATIC_ASSERT(kHiddenBytes % kNumBytesPerPull == 0, "Invalid hidden");

            // TMA load token from remote rank and store into local
            const auto pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;

            // Wait for ring buffer slot to be available (previous consumer must have finished all N blocks)
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            const uint64_t trace_ring0 = trace_clock();
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
            }
            trace_ring_wait += trace_clock() - trace_ring0;
            ++ trace_rows;

            const auto src_x = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            const auto src_dy = backward.dy.map(
                backward.dy.get_base_ptr<uint8_t*>() +
                    static_cast<uint64_t>(src_token_idx) * kHidden * (kTransportFP8 ? 1 : sizeof(nv_bfloat16)),
                current_rank_in_expert_idx);
            const auto weight = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                current_rank_in_expert_idx);
            for (uint32_t operand = 0; operand < 2; ++operand) {
                const void* src = operand == 0 ? src_x : src_dy;
                for (uint32_t i = 0; i < kNumChunks; ++i) {
                    const uint64_t trace_load0 = trace_clock();
                    if (cute::elect_one_sync()) {
                        ptx::tma_load_1d(pull_buffer.get_base_ptr(),
                            static_cast<const uint8_t*>(src) + i * kNumBytesPerPull,
                            pull_mbarrier, kNumBytesPerPull);
                        ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                        ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                    }
                    __syncwarp();
                    const uint64_t trace_load1 = trace_clock();
                    trace_load_wait += trace_load1 - trace_load0;
                    if constexpr (kTransportFP8) {
                        auto* q = reinterpret_cast<__nv_fp8_e4m3*>(
                            operand == 0 ? backward.qx_pool : backward.qdy_pool);
                        if (cute::elect_one_sync()) {
                            ptx::tma_store_1d(
                                q + static_cast<uint64_t>(pool_token_idx) * kHidden +
                                    i * kNumBytesPerPull,
                                pull_buffer.get_base_ptr(), kNumBytesPerPull);
                            cute::tma_store_arrive();
                            ptx::tma_store_wait<0>();
                        }
                        if constexpr (kPhaseTrace) {
                            __syncwarp();
                            trace_store_wait += trace_clock() - trace_load1;
                        }
                    } else if constexpr (kCacheActivations) {
                        DG_STATIC_ASSERT(kNumBytesPerPull % 256 == 0, "FP8 cache requires full K128 groups");
                        auto* q = reinterpret_cast<__nv_fp8_e4m3*>(operand == 0 ? backward.qx_pool : backward.qdy_pool);
                        auto* sf = operand == 0 ? backward.sx_pool : backward.sdy_pool;
                        const auto* pulled = reinterpret_cast<const __nv_bfloat16*>(pull_buffer.get_base_ptr());
                        for (uint32_t group = 0; group < kNumBytesPerPull / 256; ++group) {
                            float values[4];
                            float amax = 1.0e-4f;
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++j) {
                                values[j] = __bfloat162float(pulled[group * 128 + lane_idx + j * 32]);
                                amax = fmaxf(amax, fabsf(values[j]));
                            }
                            #pragma unroll
                            for (uint32_t delta = 16; delta > 0; delta /= 2)
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
                            const uint32_t bits = __float_as_uint(amax / 448.0f);
                            const uint32_t exponent = ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
                            const float inv = exponent == 255u ? 0.0f : __uint_as_float((254u - exponent) << 23);
                            const uint32_t k_offset = i * kNumBytesPerPull / 2 + group * 128;
                            if (lane_idx == 0)
                                sf[static_cast<uint64_t>(pool_token_idx) * (kHidden / 128) + k_offset / 128] = __uint_as_float(exponent << 23);
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++j)
                                q[static_cast<uint64_t>(pool_token_idx) * kHidden + k_offset + lane_idx + j * 32] = __nv_fp8_e4m3(values[j] * inv);
                        }
                    }
                    __syncwarp();
                    if constexpr (not kTransportFP8) {
                        const uint32_t scratch_token_idx = backward.qx_col == nullptr
                            ? pool_token_idx : pool_token_idx % kNumRingTokens;
                        auto* dst = (operand == 0 ? backward.x_pool : backward.dy_pool) +
                            static_cast<uint64_t>(scratch_token_idx) * kHidden;
                        if (cute::elect_one_sync()) {
                            ptx::tma_store_1d(math::advance_ptr(dst, i * kNumBytesPerPull),
                                pull_buffer.get_base_ptr(), kNumBytesPerPull);
                            cute::tma_store_arrive();
                            ptx::tma_store_wait<0>();
                        }
                    }
                    __syncwarp();
                }
                if constexpr (kTransportFP8) {
                    const auto src_sf = operand == 0
                        ? sym_buffer.map(buffer.input_sf_buffer.get_data_buffer(src_token_idx)
                                             .template get_base_ptr<float>(), current_rank_in_expert_idx)
                        : backward.dy_sf.map(
                              backward.dy_sf.get_base_ptr<float*>() +
                                  static_cast<uint64_t>(src_token_idx) * (kHidden / 128),
                              current_rank_in_expert_idx);
                    auto* dst_sf = operand == 0 ? backward.sx_pool : backward.sdy_pool;
                    for (uint32_t group = lane_idx; group < kHidden / 128; group += 32)
                        dst_sf[static_cast<uint64_t>(pool_token_idx) * (kHidden / 128) + group] =
                            src_sf[group];
                    __syncwarp();
                }
            }
            if constexpr (kCacheActivations) {
                asm volatile("fence.proxy.async.global;" ::: "memory");
                __threadfence();
            }
            __syncwarp();

            // Store weights and metadata
            if (cute::elect_one_sync()) {
                *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = weight;

                // Write source metadata for combine write-back (logical pool token)
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Wait for token TMA store to complete

                const bool is_last_token = (token_idx == expert_end_idx - 1);
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks), 
                    is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u
                );
            }
            __syncwarp();
        }

        if constexpr (kPhaseTrace) {
            if (lane_idx == 0) {
                const uint64_t trace_pull_end = trace_clock();
                trace_set(40 + warp_idx * 4, trace_pull_end - trace_pull_start);
                trace_set(41 + warp_idx * 4, trace_ring_wait);
                trace_set(42 + warp_idx * 4, trace_load_wait);
                trace_set(43 + warp_idx * 4, trace_store_wait);
                trace_set(56 + warp_idx, trace_pull_end);
                trace_add(60, trace_rows);
                if (warp_idx == 0) trace_set(2, trace_pull_start);
            }
        }

        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send count and schedule task counters
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if (warp_idx == 0 and cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumDeviceLocalityDomains; ++ i) {
                    workspace.get_l1_task_count_ptr()[i] = workspace.get_l2_task_count_ptr()[i] = 0;
                    workspace.get_shared_l1_task_count_ptr()[i] = workspace.get_shared_l2_task_count_ptr()[i] = 0;
                }
            }
            __syncwarp();
            for (uint32_t i = thread_idx; i < workspace.num_shared_l2_pool_blocks; i += kNumDispatchThreads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            __syncwarp();
        } else {
            // Other SMs: clean blocks
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read expert token count before clearing
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                if constexpr (kCompactNonMath) {
                    if (cute::elect_one_sync()) {
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                        if (cumulative_local_expert_recv_stats != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_stats + i,
                                         static_cast<int>(num_recv_tokens));
                    }
                    __syncwarp();
                } else {
                    DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                    if (warp_idx == 0) {
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                    } else if (warp_idx == 1) {
                        if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_stats + i,
                                         static_cast<int>(num_recv_tokens));
                        __syncwarp();
                    }
                }

                // Clean per-rank token count
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                // Clean L1 and L2 full stuffs and ring buffer counts
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_mask_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                }
                __syncwarp();
            }
        }

        // Wait for all ranks to finish cleaning
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );
    } else if (warp_idx == kNumDispatchWarps) {
        // GEMM TMA load warp for tokens
        task_info_t task_info;
        uint64_t trace_l1_spin = 0, trace_l2_spin = 0, trace_a_empty = 0;
        while (scheduler.get_next_task(task_info)) {
            scheduler.release_task_info();
            const auto tensor_map_a_ptr = task_info.block_phase == sched::BlockPhase::BackwardDh ? &tensor_map_dy :
                task_info.block_phase == sched::BlockPhase::BackwardDx ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto num_k_blocks = math::ceil_div(task_info.shape_k, BLOCK_K);

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;

            const uint64_t trace_spin0 = trace_clock();
            if (task_info.block_phase != sched::BlockPhase::BackwardDx) {
                const auto expected = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(workspace.get_l1_full_count_ptr(ring_block_idx)) != expected);
                trace_l1_spin += trace_clock() - trace_spin0;
            } else {
                constexpr auto full = L2KBlockDependency::kFullMask;
                const auto expected = (pool_block_idx / kNumRingBlocks) & 1 ? 0ull : full;
                while (ptx::ld_acq_gpu(workspace.get_l2_full_mask_ptr(ring_block_idx)) != expected);
                trace_l2_spin += trace_clock() - trace_spin0;
            }

            // Issue TMA
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {

                // Wait consumer release
                const uint64_t trace_empty0 = trace_clock();
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
                trace_a_empty += trace_clock() - trace_empty0;

                // Compute token offset from ring block index
                uint32_t m_idx = pool_block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;

                // TMA copy tokens, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    if (kCacheActivations and (kCacheDz or task_info.block_phase != sched::BlockPhase::BackwardDx)) {
                        tma::copy<128, LOAD_BLOCK_M, kSwizzleAMode, __nv_fp8_e4m3>(
                            tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx],
                            reinterpret_cast<__nv_fp8_e4m3*>(shared_storage.smem_a[stage_idx]), k_idx, m_idx);
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(BLOCK_M * BLOCK_K);
                    } else {
                    #pragma unroll
                    for (uint32_t half = 0; half < 2; ++half)
                        tma::copy<64, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx],
                            shared_storage.smem_a[stage_idx] + half * BLOCK_M * 64,
                            k_idx + half * 64, m_idx);
                    shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(shared_storage.smem_a[0]));
                    }
                }
                __syncwarp();
            }
        }
        if constexpr (kPhaseTrace) {
            if (lane_idx == 0) {
                trace_set(30, trace_l1_spin);
                trace_set(31, trace_l2_spin);
                trace_set(32, trace_a_empty);
                trace_set(33, trace_clock());
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // GEMM TMA load warp for weights
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            scheduler.release_task_info();
            const auto tensor_map_b_ptr = task_info.block_phase == sched::BlockPhase::BackwardDh ? &tensor_map_dh_weights :
                task_info.block_phase == sched::BlockPhase::BackwardDx ? &tensor_map_l2_weights : &tensor_map_l1_weights;
            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto n_block_idx = task_info.block_phase == sched::BlockPhase::BackwardDh or
                                     task_info.block_phase == sched::BlockPhase::BackwardDx
                ? task_info.n_cluster_idx
                : 2 * task_info.n_cluster_idx +
                    (task_info.block_phase == sched::BlockPhase::BackwardRecomputeHigh);
            const auto num_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            const uint32_t half_n = shape_n / 2, n_idx = n_block_idx * BLOCK_N, half_idx = n_idx >= half_n;
            const uint32_t half_n_idx = n_idx - half_idx * half_n, expert_idx = task_info.is_shared() ? 0 : task_info.local_expert_idx;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute weight offset
                uint32_t k_idx = k_block_idx * BLOCK_K;

                // TMA copy weights
                if (cute::elect_one_sync()) {
                    tma::copy_nd<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, 4>(
                        tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], 1, k_idx, half_n_idx, half_idx, expert_idx);
                    shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(shared_storage.smem_b[0]));
                }
                __syncwarp();
            }
        }
    } else if (not kCompactNonMath and warp_idx == kNumDispatchWarps + 2) {
    } else if (warp_idx == kNumDispatchWarps + (kCompactNonMath ? 2 : 3)) {
        if constexpr (kOwnMBlocks)
            scheduler.template mainloop_owned<true>();
        else
            scheduler.mainloop(num_tokens, sm_locality_domains);
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // GEMM epilogue warps
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // TODO: support effective block M
        // NOTES:
        //  - 2 warpgroups divide the whole BM into BM / 2
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks, i.e. with `STORE_BLOCK_M` size
        //  - `STORE_BLOCK_M` in further divided into `ATOM_M`
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        shared_storage.packed_metadata_barrier.wait(0);

        // Persistently schedule over owned M blocks. Each math warpgroup owns 64 rows.
        task_info_t task_info;
        const bool trace_leader = epilogue_thread_idx == 0;
        uint32_t trace_mblock = 0;
        uint64_t trace_prev_end = trace_clock();
        if constexpr (kPhaseTrace) {
            if (trace_leader) trace_set(3, trace_prev_end);
        }
        while (scheduler.get_next_task(task_info)) {
            scheduler.release_task_info();
            const uint64_t trace_got = trace_clock();
            uint64_t trace_full_first = 0, trace_full_rest = 0;

            float accum[WGMMA::kNumAccum] = {0};
            float partial[WGMMA::kNumAccum];
            auto a_desc = mma::sm90::make_gmma_desc<
                cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode>(
                    (kCacheActivations and (kCacheDz or task_info.block_phase != sched::BlockPhase::BackwardDx))
                        ? reinterpret_cast<__nv_fp8_e4m3*>(shared_storage.smem_a[0]) : shared_storage.quantized_a, 0, 0);
            auto b_desc = mma::sm90::make_gmma_desc<
                cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode>(
                    shared_storage.smem_b[0], 0, 0);
            const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
            const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);
            const bool dh_task = task_info.block_phase == sched::BlockPhase::BackwardDh;
            const bool dx_task = task_info.block_phase == sched::BlockPhase::BackwardDx;
            const uint32_t weight_n = dh_task or dx_task ? task_info.n_cluster_idx :
                2 * task_info.n_cluster_idx + (task_info.block_phase == sched::BlockPhase::BackwardRecomputeHigh);
            const float* weight_scales = dh_task ? w2t_scales : dx_task ? w1t_scales : w1_scales;
            const uint32_t num_k_blocks = task_info.shape_k / BLOCK_K;
            const uint32_t num_n_blocks = task_info.shape_n / BLOCK_N;
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                const uint64_t trace_fw0 = trace_clock();
                shared_storage.full_barriers[stage_idx].wait(phase);
                if constexpr (kPhaseTrace) {
                    const uint64_t trace_fw = trace_clock() - trace_fw0;
                    if (k_block_idx == 0) trace_full_first = trace_fw; else trace_full_rest += trace_fw;
                }
                if (not kCacheActivations or (dx_task and not kCacheDz)) {
                // Quantization remains inside the owning CTA; transport and dW pools stay BF16.
                for (uint32_t row = epilogue_wg_idx * WG_BLOCK_M + warp_idx_in_wg;
                     row < (epilogue_wg_idx + 1) * WG_BLOCK_M; row += 4) {
                    float values[4];
                    float amax = 1.0e-4f;
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; ++i) {
                        const uint32_t col = lane_idx + i * 32;
                        const uint32_t byte_offset = ((col / 64) * BLOCK_M * 64 + row * 64 + col % 64) * 2;
                        const uint32_t physical = byte_offset ^ ((byte_offset >> 3) & 0x70u);
                        values[i] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                            reinterpret_cast<const uint8_t*>(shared_storage.smem_a[stage_idx]) + physical));
                        amax = fmaxf(amax, fabsf(values[i]));
                    }
                    #pragma unroll
                    for (uint32_t delta = 16; delta > 0; delta /= 2)
                        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
                    const uint32_t bits = __float_as_uint(amax / 448.0f);
                    const uint32_t exponent = ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
                    const float scale = __uint_as_float(exponent << 23);
                    const float scale_inv = exponent == 255u ? 0.0f : __uint_as_float((254u - exponent) << 23);
                    if (lane_idx == 0) shared_storage.activation_scales[row] = scale;
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; ++i) {
                        const uint32_t byte_offset = row * BLOCK_K + lane_idx + i * 32;
                        const uint32_t physical = byte_offset ^ ((byte_offset >> 3) & 0x70u);
                        shared_storage.quantized_a[physical] = __nv_fp8_e4m3(values[i] * scale_inv);
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                ptx::sync_aligned(128, 3 + epilogue_wg_idx);
                }
                const uint32_t row0 = epilogue_wg_idx * WG_BLOCK_M + warp_idx_in_wg * 16 + lane_idx / 4;
                const float sb = weight_scales[(task_info.local_expert_idx * num_n_blocks + weight_n) * num_k_blocks + k_block_idx];
                const bool cached_a = kCacheActivations and (kCacheDz or not dx_task);
                const float* cached_scales = dh_task ? backward.sdy_pool : dx_task ? backward.sdz_pool : backward.sx_pool;
                const uint32_t scale_groups = (dx_task ? 2 * kIntermediateHidden : kHidden) / 128;
                const uint64_t sf_offset = (static_cast<uint64_t>(task_info.pool_block_idx) * BLOCK_M + row0) * scale_groups + k_block_idx;
                const float scale0 = (cached_a ? cached_scales[sf_offset] : shared_storage.activation_scales[row0]) * sb;
                const float scale1 = (cached_a ? cached_scales[sf_offset + 8 * scale_groups] : shared_storage.activation_scales[row0 + 8]) * sb;
                const uint32_t a_base = a_desc_lo + (cached_a ? stage_idx * sizeof(shared_storage.smem_a[0]) / 16 : 0);
                const uint32_t b_base = b_desc_lo + stage_idx * sizeof(shared_storage.smem_b[0]) / 16;
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i) {
                    partial[i] = 0.0f;
                    ptx::warpgroup_fence_operand(partial[i]);
                }
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
                    a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<
                        cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode, __nv_fp8_e4m3>(
                            a_base, epilogue_wg_idx * WG_BLOCK_M, k * WGMMA::K, 0);
                    b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<
                        cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode, __nv_fp8_e4m3>(
                            b_base, 0, k * WGMMA::K, 0);
                    WGMMA::wgmma(a_desc, b_desc, partial, k != 0);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                    ptx::warpgroup_fence_operand(partial[i]);
                ptx::warpgroup_wait<0>();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                    accum[i] += partial[i] * (i % 4 < 2 ? scale0 : scale1);
                if (lane_idx == 0)
                    shared_storage.empty_barriers[stage_idx].arrive();
            }

            const uint64_t trace_ml_end = trace_clock();
            uint64_t trace_mark[6] = {};
            constexpr uint32_t kWGMMAStoreRowsPerWarp = WGMMA::M / 4;
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++i) {
                const uint32_t row = epilogue_wg_idx * WG_BLOCK_M +
                    warp_idx_in_wg * kWGMMAStoreRowsPerWarp + lane_idx / 4;
                const uint32_t col = (lane_idx % 4) * 2 + i * 8;
                shared_storage.smem_d.raw[row * BLOCK_N + col] = d_dtype_t(accum[i * 4]);
                shared_storage.smem_d.raw[row * BLOCK_N + col + 1] = d_dtype_t(accum[i * 4 + 1]);
                shared_storage.smem_d.raw[(row + 8) * BLOCK_N + col] = d_dtype_t(accum[i * 4 + 2]);
                shared_storage.smem_d.raw[(row + 8) * BLOCK_N + col + 1] = d_dtype_t(accum[i * 4 + 3]);
            }
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            trace_mark[0] = trace_clock();

            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;
            const bool is_dh = task_info.block_phase == sched::BlockPhase::BackwardDh;
            const bool is_dx = task_info.block_phase == sched::BlockPhase::BackwardDx;
            const uint32_t half = task_info.block_phase == sched::BlockPhase::BackwardRecomputeHigh;
            const uint32_t n_block_idx = (is_dh or is_dx) ? task_info.n_cluster_idx :
                2 * task_info.n_cluster_idx + half;
            auto pack_columns = [&](const __nv_bfloat16* src, uint8_t* packed, float* scales,
                                    uint32_t width, uint32_t first_col, uint32_t num_cols) {
                if (packed == nullptr)
                    return;
                const uint32_t expert_start = pool_m_idx - task_info.m_block_idx * BLOCK_M;
                const uint32_t padded_count = shared_storage.expert_padded_counts[task_info.local_expert_idx];
                const uint32_t local_row = pool_m_idx - expert_start;
                const uint32_t src_m_idx = ring_m_idx;
                auto* q = reinterpret_cast<__nv_fp8_e4m3*>(packed);
                auto* packed_scratch = reinterpret_cast<d_dtype_t*>(shared_storage.quantized_a);
                for (uint32_t tile = 0; tile < num_cols; tile += 64) {
                    for (uint32_t linear = epilogue_thread_idx;
                         linear < BLOCK_M * 64; linear += kNumEpilogueThreads) {
                        const uint32_t row = linear / 64;
                        const uint32_t p = linear % 64;
                        const uint32_t col = first_col + tile + p;
                        const uint32_t swizzled_p = p ^ (2 * (row & 31));
                        packed_scratch[row * 64 + swizzled_p] = row < valid_m
                            ? d_dtype_t(__bfloat162float(
                                src[static_cast<uint64_t>(src_m_idx + row) * width + col]))
                            : d_dtype_t(0.0f);
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    for (uint32_t p = epilogue_warp_idx; p < 64; p += kNumEpilogueWarps) {
                        d_dtype_t values[4];
                        float amax = 1.0e-4f;
                        #pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            const uint32_t row = lane_idx + j * 32;
                            const uint32_t swizzled_p = p ^ (2 * (row & 31));
                            values[j] = packed_scratch[row * 64 + swizzled_p];
                            amax = fmaxf(amax, fabsf(static_cast<float>(values[j])));
                        }
                        #pragma unroll
                        for (uint32_t delta = 16; delta > 0; delta /= 2)
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
                        const uint32_t bits = __float_as_uint(amax / 448.0f);
                        const uint32_t exponent = ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
                        const float inv = exponent == 255u ? 0.0f :
                            __uint_as_float((254u - exponent) << 23);
                        const uint32_t col = first_col + tile + p;
                        if (lane_idx == 0)
                            scales[static_cast<uint64_t>(pool_block_idx) * width + col] =
                                __uint_as_float(exponent << 23);
                        const uint64_t dst = static_cast<uint64_t>(expert_start) * width +
                            static_cast<uint64_t>(col) * padded_count + local_row + lane_idx;
                        #pragma unroll
                        for (uint32_t j = 0; j < 4; ++j)
                            q[dst + j * 32] =
                                __nv_fp8_e4m3(static_cast<float>(values[j]) * inv);
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                asm volatile("fence.proxy.async.global;" ::: "memory");
                __threadfence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            };
            auto pack_cached_columns = [&](const uint8_t* src, const float* src_scales,
                                           uint8_t* packed, float* scales) {
                const uint32_t expert_start = pool_m_idx - task_info.m_block_idx * BLOCK_M;
                const uint32_t padded_count = shared_storage.expert_padded_counts[task_info.local_expert_idx];
                const uint32_t local_row = pool_m_idx - expert_start;
                const auto* source = reinterpret_cast<const __nv_fp8_e4m3*>(src);
                auto* q = reinterpret_cast<__nv_fp8_e4m3*>(packed);
                auto* packed_scratch = reinterpret_cast<d_dtype_t*>(shared_storage.quantized_a);
                for (uint32_t tile = 0; tile < kHidden; tile += 64) {
                    for (uint32_t linear = epilogue_thread_idx;
                         linear < BLOCK_M * 64; linear += kNumEpilogueThreads) {
                        const uint32_t row = linear / 64;
                        const uint32_t p = linear % 64;
                        float value = 0.0f;
                        if (row < valid_m) {
                            const uint64_t src_row = pool_m_idx + row;
                            const uint32_t col = tile + p;
                            value = static_cast<float>(source[src_row * kHidden + col]) *
                                src_scales[src_row * (kHidden / 128) + col / 128];
                            value = __bfloat162float(__float2bfloat16_rn(value));
                        }
                        const uint32_t swizzled_p = p ^ (2 * (row & 31));
                        packed_scratch[row * 64 + swizzled_p] = d_dtype_t(value);
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    for (uint32_t p = epilogue_warp_idx; p < 64; p += kNumEpilogueWarps) {
                        d_dtype_t values[4];
                        float amax = 1.0e-4f;
                        #pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            const uint32_t row = lane_idx + j * 32;
                            const uint32_t swizzled_p = p ^ (2 * (row & 31));
                            values[j] = packed_scratch[row * 64 + swizzled_p];
                            amax = fmaxf(amax, fabsf(static_cast<float>(values[j])));
                        }
                        #pragma unroll
                        for (uint32_t delta = 16; delta > 0; delta /= 2)
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
                        const uint32_t bits = __float_as_uint(amax / 448.0f);
                        const uint32_t exponent = ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
                        const float inv = exponent == 255u ? 0.0f :
                            __uint_as_float((254u - exponent) << 23);
                        const uint32_t col = tile + p;
                        if (lane_idx == 0)
                            scales[static_cast<uint64_t>(pool_block_idx) * kHidden + col] =
                                __uint_as_float(exponent << 23);
                        const uint64_t dst = static_cast<uint64_t>(expert_start) * kHidden +
                            static_cast<uint64_t>(col) * padded_count + local_row + lane_idx;
                        #pragma unroll
                        for (uint32_t j = 0; j < 4; ++j)
                            q[dst + j * 32] =
                                __nv_fp8_e4m3(static_cast<float>(values[j]) * inv);
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                asm volatile("fence.proxy.async.global;" ::: "memory");
                __threadfence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            };
            // Same numerics as pack_cached_columns (max reductions are order free), staged per
            // 128-column scale group through two cp.async buffers. Stage 1 aliases smem_d.raw, so
            // the caller must have consumed the dh accumulator first.
            auto pack_cached_xdy_vectorized = [&]() {
                constexpr uint32_t kChunk = 128;
                constexpr uint32_t kNumChunks = kHidden / kChunk;
                constexpr uint32_t kChunkBytes = BLOCK_M * kChunk;
                constexpr uint32_t kNumVectors = 2 * kChunkBytes / 16;
                DG_STATIC_ASSERT(sizeof(shared_storage.smem_d) >= 2 * kChunkBytes, "x/dy staging needs 32 KB");
                DG_STATIC_ASSERT(sizeof(shared_storage.quantized_a) >= 2 * BLOCK_M * sizeof(float), "scale staging");
                DG_STATIC_ASSERT(not kVectorXdyPack or
                                 sizeof(shared_storage.xdy_stage) >= 2 * kChunkBytes + 2 * BLOCK_M * sizeof(float),
                                 "x/dy second stage");
                DG_STATIC_ASSERT(kNumEpilogueThreads == 2 * BLOCK_M and BLOCK_M == 128 and kNumChunks >= 2,
                                 "vectorized x/dy pack assumes 8 warps over 128x128 chunks");
                DG_STATIC_ASSERT(kNumVectors % kNumEpilogueThreads == 0, "Invalid staging split");
                const uint32_t expert_start = pool_m_idx - task_info.m_block_idx * BLOCK_M;
                const uint32_t padded_count = shared_storage.expert_padded_counts[task_info.local_expert_idx];
                const uint32_t local_row = pool_m_idx - expert_start;
                // Ternaries rather than pointer arrays keep the selections out of local memory.
                auto stage_data = [&](uint32_t stage) {
                    return stage == 0 ? shared_storage.xdy_stage : reinterpret_cast<uint8_t*>(shared_storage.smem_d.raw);
                };
                auto stage_scales = [&](uint32_t stage) {
                    return stage == 0 ? reinterpret_cast<float*>(shared_storage.xdy_stage + 2 * kChunkBytes)
                                      : reinterpret_cast<float*>(shared_storage.quantized_a);
                };
                const uint32_t col_group = 4 * epilogue_warp_idx + lane_idx / 8;
                const uint32_t row_group = lane_idx % 8;
                auto issue = [&](uint32_t chunk, uint32_t stage) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumVectors / kNumEpilogueThreads; ++i) {
                        constexpr uint32_t kPerTensor = kChunkBytes / 16 / kNumEpilogueThreads;
                        const uint32_t tensor = i / kPerTensor;
                        const uint32_t linear = epilogue_thread_idx + (i % kPerTensor) * kNumEpilogueThreads;
                        const uint32_t row = linear / 8;
                        const uint32_t vec = linear % 8;
                        const bool valid = row < valid_m;
                        const uint8_t* src = (tensor == 0 ? backward.qx_pool : backward.qdy_pool) + (valid
                            ? static_cast<uint64_t>(pool_m_idx + row) * kHidden + chunk * kChunk + vec * 16 : 0);
                        const auto dst = static_cast<uint32_t>(__cvta_generic_to_shared(
                            stage_data(stage) + tensor * kChunkBytes + row * kChunk + ((vec ^ ((row / 16) % 8)) * 16)));
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                                     :: "r"(dst), "l"(src), "r"(valid ? 16u : 0u) : "memory");
                    }
                    const uint32_t tensor = epilogue_thread_idx / BLOCK_M;
                    const uint32_t row = epilogue_thread_idx % BLOCK_M;
                    const bool valid = row < valid_m;
                    const float* src = (tensor == 0 ? backward.sx_pool : backward.sdy_pool) + (valid
                        ? static_cast<uint64_t>(pool_m_idx + row) * (kHidden / 128) + chunk : 0);
                    const auto dst = static_cast<uint32_t>(__cvta_generic_to_shared(
                        stage_scales(stage) + epilogue_thread_idx));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
                                 :: "r"(dst), "l"(src), "r"(valid ? 4u : 0u) : "memory");
                    asm volatile("cp.async.commit_group;" ::: "memory");
                };
                auto dequant2 = [](uint32_t word, uint32_t half, float scale) {
                    const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(
                        static_cast<__nv_fp8x2_storage_t>(word >> (16 * half)), __NV_E4M3);
                    const float2 f = __half22float2(__half2(raw));
                    return __bfloat1622float2(__floats2bfloat162_rn(f.x * scale, f.y * scale));
                };
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                issue(0, 0);
                issue(1, 1);
                for (uint32_t chunk = 0; chunk < kNumChunks; ++chunk) {
                    const uint32_t stage = chunk % 2;
                    if (chunk + 1 < kNumChunks)
                        asm volatile("cp.async.wait_group 1;" ::: "memory");
                    else
                        asm volatile("cp.async.wait_group 0;" ::: "memory");
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    #pragma unroll
                    for (uint32_t tensor = 0; tensor < 2; ++tensor) {
                        const auto* words = reinterpret_cast<const uint32_t*>(stage_data(stage) + tensor * kChunkBytes);
                        const uint32_t word_idx = col_group ^ (4 * row_group);
                        const float* row_scales = stage_scales(stage) + tensor * BLOCK_M + row_group * 16;
                        float amax[4] = {1.0e-4f, 1.0e-4f, 1.0e-4f, 1.0e-4f};
                        #pragma unroll
                        for (uint32_t j = 0; j < 16; ++j) {
                            const uint32_t word = words[(row_group * 16 + j) * (kChunk / 4) + word_idx];
                            const float scale = row_scales[j];
                            #pragma unroll
                            for (uint32_t h = 0; h < 2; ++h) {
                                const float2 v = dequant2(word, h, scale);
                                amax[2 * h] = fmaxf(amax[2 * h], fabsf(v.x));
                                amax[2 * h + 1] = fmaxf(amax[2 * h + 1], fabsf(v.y));
                            }
                        }
                        float inv[4], scale_out[4];
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; ++k) {
                            #pragma unroll
                            for (uint32_t delta = 1; delta < 8; delta *= 2)
                                amax[k] = fmaxf(amax[k], __shfl_xor_sync(0xffffffff, amax[k], delta));
                            const uint32_t bits = __float_as_uint(amax[k] / 448.0f);
                            const uint32_t exponent = ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
                            inv[k] = exponent == 255u ? 0.0f : __uint_as_float((254u - exponent) << 23);
                            scale_out[k] = __uint_as_float(exponent << 23);
                        }
                        const uint32_t col = chunk * kChunk + 4 * col_group;
                        if (row_group == 0)
                            *reinterpret_cast<float4*>((tensor == 0 ? backward.sx_col : backward.sdy_col) +
                                static_cast<uint64_t>(pool_block_idx) * kHidden + col) =
                                make_float4(scale_out[0], scale_out[1], scale_out[2], scale_out[3]);
                        uint32_t out[4][4] = {};
                        #pragma unroll
                        for (uint32_t j = 0; j < 16; ++j) {
                            const uint32_t word = words[(row_group * 16 + j) * (kChunk / 4) + word_idx];
                            const float scale = row_scales[j];
                            #pragma unroll
                            for (uint32_t h = 0; h < 2; ++h) {
                                const float2 v = dequant2(word, h, scale);
                                const uint32_t q = __nv_cvt_float2_to_fp8x2(
                                    make_float2(v.x * inv[2 * h], v.y * inv[2 * h + 1]), __NV_SATFINITE, __NV_E4M3);
                                out[2 * h][j / 4] |= (q & 0xffu) << (8 * (j % 4));
                                out[2 * h + 1][j / 4] |= (q >> 8) << (8 * (j % 4));
                            }
                        }
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; ++k) {
                            const uint64_t dst = static_cast<uint64_t>(expert_start) * kHidden +
                                static_cast<uint64_t>(col + k) * padded_count + local_row + row_group * 16;
                            *reinterpret_cast<uint4*>((tensor == 0 ? backward.qx_col : backward.qdy_col) + dst) =
                                make_uint4(out[k][0], out[k][1], out[k][2], out[k][3]);
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (chunk + 2 < kNumChunks)
                        issue(chunk + 2, stage);
                }
                asm volatile("fence.proxy.async.global;" ::: "memory");
                __threadfence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            };
            if (is_dh) {
                if (task_info.n_cluster_idx == 0) {
                    const auto expected = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(workspace.get_l2_empty_count_ptr(ring_block_idx)) != expected) {}
                    trace_mark[1] = trace_clock();
                    for (uint32_t row = epilogue_thread_idx; row < BLOCK_M; row += kNumEpilogueThreads)
                        shared_storage.dscore[row] = 0.0f;
                    if constexpr ((kDiagAblate & 1u) or kVectorXdyPack) {
                    } else if constexpr (kTransportFP8) {
                        pack_cached_columns(backward.qx_pool, backward.sx_pool,
                                            backward.qx_col, backward.sx_col);
                        pack_cached_columns(backward.qdy_pool, backward.sdy_pool,
                                            backward.qdy_col, backward.sdy_col);
                    } else {
                        pack_columns(backward.x_pool, backward.qx_col, backward.sx_col,
                                     kHidden, 0, kHidden);
                        pack_columns(backward.dy_pool, backward.qdy_col, backward.sdy_col,
                                     kHidden, 0, kHidden);
                    }
                    trace_mark[2] = trace_clock();
                }
                for (uint32_t linear = epilogue_thread_idx; linear < BLOCK_M * BLOCK_N;
                     linear += kNumEpilogueThreads) {
                    const uint32_t row = linear / BLOCK_N;
                    shared_storage.backward_epilogue.dh[linear] = row < valid_m
                        ? __float2bfloat16_rn(static_cast<float>(shared_storage.smem_d.raw[linear]))
                        : __float2bfloat16_rn(0.0f);
                }
                if constexpr (kVectorXdyPack and not (kDiagAblate & 1u)) {
                    if (task_info.n_cluster_idx == 0) {
                        pack_cached_xdy_vectorized();
                        trace_mark[2] = trace_clock();
                    }
                }
            } else if (not is_dx) {
                for (uint32_t linear = epilogue_thread_idx; linear < BLOCK_M * (BLOCK_N / 2);
                     linear += kNumEpilogueThreads) {
                    const uint32_t row = linear / (BLOCK_N / 2);
                    const uint32_t p = linear % (BLOCK_N / 2);
                    const uint32_t gate_col = 16 * (p / 8) + p % 8;
                    const float gate = static_cast<float>(shared_storage.smem_d.raw[row * BLOCK_N + gate_col]);
                    const float up = static_cast<float>(shared_storage.smem_d.raw[row * BLOCK_N + gate_col + 8]);
                    const float dh = __bfloat162float(
                        shared_storage.backward_epilogue.dh[row * BLOCK_N + half * 64 + p]);
                    const float score = row < valid_m
                        ? buffer.l1_topk_weights_buffer.get_base_ptr<float>()[ring_m_idx + row] : 0.0f;
                    const float sigmoid = 1.0f / (1.0f + expf(-gate));
                    const float silu = gate * sigmoid;
                    const float h = silu * up * score;
                    const float dg = dh * score * up * (sigmoid + silu * (1.0f - sigmoid));
                    const float du = dh * score * silu;
                    const uint32_t n = n_block_idx * 64 + p;
                    const uint32_t scratch_m_idx = backward.qh_col == nullptr ? pool_m_idx : ring_m_idx;
                    const uint64_t scratch = static_cast<uint64_t>(scratch_m_idx + row);
                    const bool valid = row < valid_m;
                    backward.h_pool[scratch * kIntermediateHidden + n] = __float2bfloat16_rn(valid ? h : 0.0f);
                    backward.dz_pool[scratch * (2 * kIntermediateHidden) + n] = __float2bfloat16_rn(valid ? dg : 0.0f);
                    backward.dz_pool[scratch * (2 * kIntermediateHidden) + kIntermediateHidden + n] =
                        __float2bfloat16_rn(valid ? du : 0.0f);
                    if constexpr (kWarpReduceDscore) {
                        // A warp covers 32 columns of one row, so reduce before the shared atomic.
                        DG_STATIC_ASSERT((BLOCK_N / 2) % 32 == 0 and kNumEpilogueThreads % (BLOCK_N / 2) == 0,
                                         "dscore warp reduction needs row-uniform warps");
                        float contribution = valid ? dh * silu * up : 0.0f;
                        #pragma unroll
                        for (uint32_t delta = 16; delta > 0; delta /= 2)
                            contribution += __shfl_xor_sync(0xffffffff, contribution, delta);
                        if (valid and lane_idx == 0 and not (kDiagAblate & 2u))
                            atomicAdd(shared_storage.dscore + row, contribution);
                    } else if (valid and not (kDiagAblate & 2u)) {
                        atomicAdd(shared_storage.dscore + row, dh * silu * up);
                    }
                    if (backward.qh_col != nullptr) {
                        shared_storage.smem_d.raw[row * BLOCK_N + gate_col] =
                            d_dtype_t(valid ? h : 0.0f);
                        shared_storage.smem_d.raw[row * BLOCK_N + gate_col + 8] =
                            d_dtype_t(valid ? dg : 0.0f);
                        shared_storage.backward_epilogue.dh[row * BLOCK_N + half * 64 + p] =
                            __float2bfloat16_rn(valid ? du : 0.0f);
                    }
                }
                __threadfence();
                asm volatile("fence.proxy.async.global;" ::: "memory");
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                trace_mark[1] = trace_clock();
                if (backward.qh_col != nullptr and not (kDiagAblate & 4u)) {
                    const uint32_t expert_start = pool_m_idx - task_info.m_block_idx * BLOCK_M;
                    const uint32_t padded_count =
                        shared_storage.expert_padded_counts[task_info.local_expert_idx];
                    const uint32_t local_row = pool_m_idx - expert_start;
                    auto* packed_scratch = reinterpret_cast<d_dtype_t*>(
                        shared_storage.quantized_a);
                    for (uint32_t kind = 0; kind < 3; ++kind) {
                        for (uint32_t linear = epilogue_thread_idx;
                             linear < BLOCK_M * 64; linear += kNumEpilogueThreads) {
                            const uint32_t row = linear / 64;
                            const uint32_t p = linear % 64;
                            const uint32_t gate_col = 16 * (p / 8) + p % 8;
                            const float value = kind < 2 ? static_cast<float>(
                                shared_storage.smem_d.raw[row * BLOCK_N + gate_col + kind * 8]) :
                                __bfloat162float(shared_storage.backward_epilogue.dh[
                                    row * BLOCK_N + half * 64 + p]);
                            const uint32_t swizzled_p = p ^ (2 * (row & 31));
                            packed_scratch[row * 64 + swizzled_p] =
                                d_dtype_t(row < valid_m ? value : 0.0f);
                        }
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        for (uint32_t p = epilogue_warp_idx; p < 64; p += kNumEpilogueWarps) {
                            d_dtype_t values[4];
                            float amax = 1.0e-4f;
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++j) {
                                const uint32_t row = lane_idx + j * 32;
                                const uint32_t swizzled_p = p ^ (2 * (row & 31));
                                values[j] = packed_scratch[row * 64 + swizzled_p];
                                amax = fmaxf(amax, fabsf(static_cast<float>(values[j])));
                            }
                            #pragma unroll
                            for (uint32_t delta = 16; delta > 0; delta /= 2)
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
                            const uint32_t bits = __float_as_uint(amax / 448.0f);
                            const uint32_t exponent = ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
                            const float inv = exponent == 255u ? 0.0f :
                                __uint_as_float((254u - exponent) << 23);
                            const uint32_t width = kind == 0 ? kIntermediateHidden : 2 * kIntermediateHidden;
                            const uint32_t col = n_block_idx * 64 + p +
                                (kind == 2 ? kIntermediateHidden : 0);
                            auto* scales = kind == 0 ? backward.sh_col : backward.sdz_col;
                            auto* q = reinterpret_cast<__nv_fp8_e4m3*>(
                                kind == 0 ? backward.qh_col : backward.qdz_col);
                            if (lane_idx == 0)
                                scales[static_cast<uint64_t>(pool_block_idx) * width + col] =
                                    __uint_as_float(exponent << 23);
                            const uint64_t dst = static_cast<uint64_t>(expert_start) * width +
                                static_cast<uint64_t>(col) * padded_count + local_row + lane_idx;
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++j)
                                q[dst + j * 32] = __nv_fp8_e4m3(static_cast<float>(values[j]) * inv);
                        }
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    }
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    __threadfence();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                trace_mark[2] = trace_clock();
                if constexpr (kCacheDz and not (kDiagAblate & 8u)) {
                    if (half == 1) {
                        for (uint32_t row = epilogue_thread_idx / 32; row < BLOCK_M; row += kNumEpilogueWarps) {
                            for (uint32_t operand = 0; operand < 2; ++operand) {
                                const uint32_t column = operand * kIntermediateHidden + task_info.n_cluster_idx * 128;
                                const uint32_t scratch_m_idx = backward.qh_col == nullptr ? pool_m_idx : ring_m_idx;
                                const uint64_t src_offset = static_cast<uint64_t>(scratch_m_idx + row) * (2 * kIntermediateHidden) + column;
                                const uint64_t offset = static_cast<uint64_t>(pool_m_idx + row) * (2 * kIntermediateHidden) + column;
                                float values[4];
                                float amax = 1.0e-4f;
                                #pragma unroll
                                for (uint32_t j = 0; j < 4; ++j) {
                                    values[j] = __bfloat162float(backward.dz_pool[src_offset + lane_idx + j * 32]);
                                    amax = fmaxf(amax, fabsf(values[j]));
                                }
                                #pragma unroll
                                for (uint32_t delta = 16; delta > 0; delta /= 2)
                                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
                                const uint32_t bits = __float_as_uint(amax / 448.0f);
                                const uint32_t exponent = ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
                                const float inv = exponent == 255u ? 0.0f : __uint_as_float((254u - exponent) << 23);
                                if (lane_idx == 0)
                                    backward.sdz_pool[offset / 128] = __uint_as_float(exponent << 23);
                                auto* q = reinterpret_cast<__nv_fp8_e4m3*>(backward.qdz_pool);
                                #pragma unroll
                                for (uint32_t j = 0; j < 4; ++j)
                                    q[offset + lane_idx + j * 32] = __nv_fp8_e4m3(values[j] * inv);
                            }
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        __threadfence();
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    }
                }
                trace_mark[3] = trace_clock();
                if (epilogue_thread_idx == 0) {
                    L2KBlockDependency::arrive(workspace.get_l2_full_mask_ptr(ring_block_idx), n_block_idx);
                    ptx::red_add(workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                }
                if (task_info.block_phase == sched::BlockPhase::BackwardRecomputeHigh and
                    task_info.n_cluster_idx + 1 == kIntermediateHidden / BLOCK_N) {
                    for (uint32_t row = epilogue_thread_idx; row < valid_m; row += kNumEpilogueThreads) {
                        const auto src = *workspace.get_token_src_metadata_ptr(pool_m_idx + row);
                        auto* dst = backward.dscore_partials.map(
                            backward.dscore_partials.get_base_ptr<float*>(), src.rank_idx);
                        const uint64_t offset = (static_cast<uint64_t>(src.token_idx) * kNumTopk + src.topk_idx) * 2;
                        dst[offset] = shared_storage.dscore[row];
                        dst[offset + 1] = 0.0f;
                    }
                    __threadfence_system();
                }
            } else {
                if constexpr (kDiagAblate & 16u) {
                } else if constexpr (kFP8Combine) {
                    for (uint32_t row = epilogue_warp_idx; row < valid_m; row += kNumEpilogueWarps) {
                        d_dtype_t values[4];
                        float amax = 1.0e-4f;
                        #pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            values[j] = shared_storage.smem_d.raw[
                                row * BLOCK_N + lane_idx + j * 32];
                            amax = fmaxf(amax, fabsf(static_cast<float>(values[j])));
                        }
                        #pragma unroll
                        for (uint32_t delta = 16; delta > 0; delta /= 2)
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
                        const uint32_t bits = __float_as_uint(amax / 448.0f);
                        const uint32_t exponent = ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
                        const float inv = exponent == 255u ? 0.0f :
                            __uint_as_float((254u - exponent) << 23);
                        const auto src = *workspace.get_token_src_metadata_ptr(pool_m_idx + row);
                        auto dst = buffer.combine_token_buffer.get_rank_buffer(src.topk_idx)
                            .get_data_buffer(src.token_idx);
                        auto dst_sf = buffer.combine_sf_buffer.get_rank_buffer(src.topk_idx)
                            .get_data_buffer(src.token_idx);
                        auto* q = sym_buffer.map(dst.template get_base_ptr<__nv_fp8_e4m3>() +
                            n_block_idx * BLOCK_N, src.rank_idx);
                        if (lane_idx == 0)
                            sym_buffer.map(dst_sf.template get_base_ptr<float>() + n_block_idx,
                                           src.rank_idx)[0] = __uint_as_float(exponent << 23);
                        #pragma unroll
                        for (uint32_t j = 0; j < 4; ++j)
                            q[lane_idx + j * 32] = __nv_fp8_e4m3(static_cast<float>(values[j]) * inv);
                    }
                } else {
                    for (uint32_t linear = epilogue_thread_idx; linear < valid_m * BLOCK_N;
                         linear += kNumEpilogueThreads) {
                        const uint32_t row = linear / BLOCK_N;
                        const uint32_t col = linear % BLOCK_N;
                        const auto src = *workspace.get_token_src_metadata_ptr(pool_m_idx + row);
                        auto dst = buffer.combine_token_buffer.get_rank_buffer(src.topk_idx)
                            .get_data_buffer(src.token_idx);
                        sym_buffer.map(dst.template get_base_ptr<d_dtype_t>() + n_block_idx * BLOCK_N + col,
                                       src.rank_idx)[0] = shared_storage.smem_d.raw[row * BLOCK_N + col];
                    }
                }
                __threadfence_system();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_thread_idx == 0)
                    ptx::red_add(workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
            }
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            if constexpr (kPhaseTrace) {
                const uint64_t trace_end = trace_clock();
                const bool trace_wg_leader = epilogue_thread_idx % 128 == 0;
                if (trace_wg_leader and not trace_leader) {
                    trace_add(28, trace_ml_end - trace_got);
                    trace_add(29, trace_full_first + trace_full_rest);
                }
                if (trace_leader) {
                    const uint32_t phase_idx = is_dh ? 0 : is_dx ? 3 : 1 + half;
                    const uint64_t mainloop = trace_ml_end - trace_got;
                    const uint64_t epilogue = trace_end - trace_ml_end;
                    const uint64_t sched_wait = trace_got - trace_prev_end;
                    trace_add(6, sched_wait);
                    trace_add(7, trace_full_first);
                    trace_add(8, trace_full_rest);
                    trace_add(9 + phase_idx, mainloop);
                    trace_add(13 + phase_idx, epilogue);
                    trace_add(17, 1);
                    trace_add(20, trace_mark[0] - trace_ml_end);
                    if (is_dh and task_info.n_cluster_idx == 0) {
                        trace_add(21, trace_mark[1] - trace_mark[0]);
                        trace_add(22, trace_mark[2] - trace_mark[1]);
                        trace_add(23, trace_full_first);
                        trace_add(24, 1);
                    } else if (not is_dh and not is_dx) {
                        trace_add(25, trace_mark[1] - trace_mark[0]);
                        trace_add(26, trace_mark[2] - trace_mark[1]);
                        trace_add(27, trace_mark[3] - trace_mark[2]);
                        trace_add(34, trace_end - trace_mark[3]);
                    } else if (is_dx) {
                        trace_add(35, trace_end - trace_mark[0]);
                    }
                    if (is_dh)
                        trace_add(36, trace_full_first);
                    else if (is_dx)
                        trace_add(37, trace_full_first);
                    else
                        trace_add(38, trace_full_first);
                    // Per M-block record
                    if (is_dh and task_info.n_cluster_idx == 0) {
                        ++ trace_mblock;
                        if (trace_mblock <= kTraceMBlockSlots) {
                            const uint32_t base = kTraceHeader + (trace_mblock - 1) * kTraceMBlockFields;
                            trace_set(base + 0, pool_block_idx + 1);
                            trace_set(base + 1, trace_got);
                            trace_set(base + 3, trace_full_first);
                            trace_set(base + 8, trace_mark[2] - trace_mark[0]);
                            trace_set(base + 9, valid_m);
                        }
                    }
                    if (trace_mblock >= 1 and trace_mblock <= kTraceMBlockSlots) {
                        const uint32_t base = kTraceHeader + (trace_mblock - 1) * kTraceMBlockFields;
                        trace_add(base + 4, mainloop);
                        trace_add(base + 5, epilogue);
                        trace_add(base + 6, sched_wait);
                        trace_add(base + 7, trace_full_first + trace_full_rest);
                        trace_add(base + 10 + phase_idx, mainloop);
                        trace_set(base + 2, trace_end);
                        trace_add(base + 14, epilogue * (is_dx ? 1 : 0));
                        trace_add(base + 15, trace_full_first);
                    }
                }
                trace_prev_end = trace_end;
            }
        }
        if constexpr (kPhaseTrace) {
            if (trace_leader) {
                trace_set(4, trace_clock());
                trace_add(19, trace_mblock);
            }
        }

        // Combine: reduce top-k results and write back
        // NOTES: reuse shared memory from start up to the barriers
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // 3 slots of chunk is needed: 2 load stages and 1 store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // NOTES: either 1 or 2 chunks for simplicity
        // NOTES: Restrict on both smem and register
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= kNumReusableSmemBytes and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= kNumReusableSmemBytes, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u, "Top-k + shared must fit in a single warp");

        // Verify combined shared memory budget at runtime
        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <=
            static_cast<uint32_t>(offsetof(SharedStorage, dispatch_barriers)));

        // Per-warp buffer: 2 stage load buffers + 1 store buffer
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        // Per-warp barriers
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return &shared_storage.combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;

        // Peers' grid indices for tagging their combine readiness; the load overlaps with the grid sync
        DG_STATIC_ASSERT(kNumRanks <= kNumEpilogueThreads, "Insufficient threads for combine readiness");
        uint64_t peer_grid_idx = 0;
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
            peer_grid_idx = *workspace.get_peer_grid_idx_ptr(epilogue_thread_idx);

        // All local L2 writes are done after this grid sync
        comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
            workspace, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // Notify remote ranks; ordered before the cleanup barrier by the dispatch/epilogue sync below
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
            ptx::st_rel_sys(sym_buffer.map(workspace.get_combine_ready_grid_idx_ptr(sym_buffer.rank_idx), epilogue_thread_idx), peer_grid_idx);

        // Barrier with dispatch warps, so that they can do clean workspace
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        const auto grid_idx = ptx::get_grid_idx() + 1;
        if constexpr (kFP8Combine) {
            constexpr uint32_t kNumKBlocks = kHidden / 128;
            for (uint32_t token_k_idx = epilogue_warp_idx * kNumSMs + sm_idx;
                 token_k_idx < num_tokens * kNumKBlocks;
                 token_k_idx += kNumSMs * kNumEpilogueWarps) {
                const uint32_t token_idx = token_k_idx / kNumKBlocks;
                const uint32_t k_block_idx = token_k_idx % kNumKBlocks;
                const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                    static_cast<int>(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[
                        token_idx * kNumTopk + lane_idx]) : -1;
                const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);
                const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
                const auto peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                    is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0);
                comm::wait_until(
                    [&]() { return __all_sync(0xffffffff, not is_routed or
                        ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                    [&]() { printf("DeepGEMM FP8 combine peers timeout: rank=%u, token=%u\n",
                                   sym_buffer.rank_idx, token_idx); });
                if (k_block_idx == 0 and lane_idx < kNumTopk) {
                    const auto offset = static_cast<uint64_t>(token_idx) * kNumTopk + lane_idx;
                    const auto partials = backward.dscore_partials.get_base_ptr<float*>();
                    backward.dscore[offset] = stored_topk_slot_idx >= 0 ?
                        partials[offset * 2] + partials[offset * 2 + 1] : 0.0f;
                }
                float reduced[4] = {};
                uint32_t mask = total_mask;
                while (mask) {
                    const uint32_t slot_idx = __ffs(mask) - 1;
                    mask ^= 1u << slot_idx;
                    const auto q = buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                        .get_data_buffer(token_idx).template get_base_ptr<__nv_fp8_e4m3>() +
                        k_block_idx * 128;
                    const float scale = buffer.combine_sf_buffer.get_rank_buffer(slot_idx)
                        .get_data_buffer(token_idx).template get_base_ptr<float>()[k_block_idx];
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; ++j)
                        reduced[j] += static_cast<float>(q[lane_idx + j * 32]) * scale;
                }
                auto dst = reinterpret_cast<nv_bfloat16*>(y) +
                    static_cast<uint64_t>(token_idx) * kHidden + k_block_idx * 128;
                #pragma unroll
                for (uint32_t j = 0; j < 4; ++j)
                    dst[lane_idx + j * 32] = __float2bfloat16_rn(reduced[j]);
            }
        } else {
        // Iterate over all token chunks, 1 token 1 topk latency: ~3 us
        for (uint32_t token_chunk_idx = epilogue_warp_idx * kNumSMs + sm_idx; token_chunk_idx < num_tokens * kNumChunks; token_chunk_idx += kNumSMs * kNumEpilogueWarps) {
            const uint32_t token_idx = token_chunk_idx / kNumChunks;
            const uint32_t chunk_idx = token_chunk_idx % kNumChunks;

            // Read top-k slot indices: each lane reads one slot, then broadcast via exchange
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token_idx * kNumTopk + lane_idx]) :
                (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            // Wait for the ranks of the selected experts to finish their L2 writes
            const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
            const auto peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0);
            comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                             [&]() { printf("DeepGEMM combine peers timeout: rank=%u, token=%u\n", sym_buffer.rank_idx, token_idx); });

            if (chunk_idx == 0 and lane_idx < kNumTopk) {
                const auto offset = static_cast<uint64_t>(token_idx) * kNumTopk + lane_idx;
                const auto partials = backward.dscore_partials.get_base_ptr<float*>();
                backward.dscore[offset] = stored_topk_slot_idx >= 0
                    ? partials[offset * 2] + partials[offset * 2 + 1] : 0.0f;
            }

            const uint32_t chunk_byte_offset = chunk_idx * kNumChunkBytes;

            // Move mask and load
            uint32_t mask = total_mask;
            const auto move_mask_and_load = [&](const uint32_t& i) {
                if (mask) {
                    // Shared slot goes first, then routed slots in top-k order
                    const uint32_t slot_idx = (mask >> kNumTopk) ? kNumTopk : __ffs(mask) - 1;
                    mask ^= 1 << slot_idx;

                    // Load
                    if (cute::elect_one_sync()) {
                        const auto src_ptr = math::advance_ptr<uint8_t>(
                            buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                .get_data_buffer(token_idx).get_base_ptr(),
                            chunk_byte_offset);
                        ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                        ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                    }
                    __syncwarp();
                    return true;
                }
                return false;
            };

            // Load the first selection
            bool do_reduce = move_mask_and_load(load_stage_idx);

            // Accumulate all top-k contributions for this chunk in float registers
            float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
            while (do_reduce) {
                // Prefetch next top-k into the buffer while current is being accumulated
                do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                // Accumulate
                combine_load_barriers[load_stage_idx]->wait(combine_phase);
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                    const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                }
                cutlass::arch::fence_view_async_shared();
                combine_phase ^= load_stage_idx;
                load_stage_idx ^= 1;
            }

            // Cast
            #pragma unroll
            for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                uint4 casted;
                auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                #pragma unroll
                for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                    casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                // Wait share memory release and write
                if (j == 0) {
                    ptx::tma_store_wait<0>();
                    __syncwarp();
                }
                ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                               casted.x, casted.y, casted.z, casted.w);
            }
            __syncwarp();

            // TMA store the token chunk
            if (cute::elect_one_sync()) {
                cute::tma_store_fence();
                ptx::tma_store_1d(
                    math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                    combine_store_buffer, kNumChunkBytes);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
        }
        if constexpr (kPhaseTrace) {
            if (trace_leader) {
                trace_set(5, trace_clock());
                trace_set(18, trace_globaltimer());
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
