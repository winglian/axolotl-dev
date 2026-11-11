// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mono_ep.cuh>
#include <deep_gemm/layout/moe_backward.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/epilogue/sm100_moe_backward.cuh>
#include <deep_gemm/scheduler/mono_ep.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/common/nvfp4_quant.cuh>

namespace deep_gemm {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    bool kOwnMBlocks = true,
    bool kPhaseTrace = false,
    uint32_t kDiagAblate = 0,
    uint32_t kOptions = 0,
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
sm100_nvfp4_mono_ep_backward_impl(void* y,
                         int* cumulative_local_expert_recv_stats,
                         const uint32_t num_tokens,
                         const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_dy,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_dy_sf,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_dh_weights,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_dh_weights_sf,
                         const float* expert_scales,
                         const uint8_t* sm_locality_domains, __nv_bfloat16* saved_z,
                         const __grid_constant__ layout::MoEBackwardNvfp4Buffers backward) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(not kHasShared and kOwnMBlocks, "Backward requires owned routed blocks");
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // Use one domain when clusters cannot split evenly across device locality domains
    constexpr uint32_t kNumLocalityDomains =
        L1_SHAPE_N % (BLOCK_N * 2 * kNumDeviceLocalityDomains) == 0 and
        L2_SHAPE_N % (BLOCK_N * 2 * kNumDeviceLocalityDomains) == 0 ? kNumDeviceLocalityDomains : 1;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_dy);
        cute::prefetch_tma_descriptor(&tensor_map_dy_sf);
        cute::prefetch_tma_descriptor(&tensor_map_dh_weights);
        cute::prefetch_tma_descriptor(&tensor_map_dh_weights_sf);
    }

    // Workspaces and Buffer
    const auto buffer = layout::MonoEPBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, 0,
        /*with_sf=*/ false,
        kNumSharedExperts
    );
    const auto workspace = buffer.workspace;

    using L2KBlockDependency = sched::L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>;

    // SF configs: UE4M3 per 16 elements, UTCCP atoms of 128 rows x 4 K blocks (one 32-bit word per row)
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == BLOCK_M and BLOCK_M == kNumUTCCPAlignedElems, "Pool rows are 128-row blocks");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");
    DG_STATIC_ASSERT(kHidden % 256 == 0 and kIntermediateHidden % 256 == 0, "NVFP4 K blocks hold 256 elements");

    // UTCCP 4x32 transpose index mapping within each 128-row block (pool rows are block aligned)
    const auto transform_sf_token_idx = [](const uint32_t& pool_token_idx) {
        const uint32_t idx = pool_token_idx % BLOCK_M;
        return pool_token_idx / BLOCK_M * SF_BLOCK_M + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // Data types
    // NOTES: both operands are packed E2M1 (two elements per byte in shared memory); the dx staging stays BF16
    using a_dtype_t = cutlass::float_e2m1_t;
    using b_dtype_t = cutlass::float_e2m1_t;
    using d_dtype_t = cutlass::bfloat16_t;
    using sf_dtype_t = cutlass::float_ue4m3_t;

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major; one `kind::mxf4nvf4` instruction covers K = 64
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_BLOCK_K = 64;
    constexpr uint32_t UMMA_K = 64;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K == 256, "One 128B swizzle atom of packed FP4 per stage");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N * sizeof(nv_bfloat16) % kSwizzleCDMode == 0, "Invalid block N");

    // Opt-in variants (bitmask): 1 staged 16-byte h/dz pool stores, 2 TMEM release before the
    // recompute/dh epilogue math, 4 four TMEM accumulator stages, 8 overlapped dispatch x/dy copies
    // (with kNumBytesPerPull == 2 * hidden bytes, whole x and dy rows travel in one round trip).
    constexpr bool kStagedPoolStores = kOptions & 1u;
    constexpr bool kEarlyTmemRelease = kOptions & 2u;
    constexpr bool kAsyncDispatchCopy = kOptions & 8u;
    // 16: lookahead pull (needs 8 and whole x/dy rows in the pull buffer): the next row's routing, source index and
    // weight are resolved while this row's NVLink load is in flight, and the next load is issued before this row is
    // published. Same rows, pool slots and counts as the default loop.
    constexpr bool kLookaheadPull = kOptions & 16u;
    // L2 residency hints: 32 weight (B) tiles loaded EVICT_LAST; 64 h-pool stores (read only by dW after the
    // kernel) with an L2 evict_first policy (needs 1).
    constexpr uint64_t kWeightCacheHint = static_cast<uint64_t>((kOptions & 32u) ?
        cute::TMA::CacheHintSm100::EVICT_LAST : cute::TMA::CacheHintSm100::EVICT_NORMAL);
    constexpr bool kEvictFirstHStores = kOptions & 64u;
    // 128: A (dy / x / dz pool) tiles loaded EVICT_FIRST on their last use in the owned M-block (Dh and
    // RecomputeHigh at the last inner N cluster, Dx at the last output N cluster); 256: dispatch source-row loads
    // EVICT_NORMAL instead of EVICT_FIRST (each source row is read top-k times).
    constexpr bool kEvictFirstLastA = kOptions & 128u;
    constexpr auto kDispatchLoadHint = (kOptions & 256u) ?
        cute::TMA::CacheHintSm100::EVICT_NORMAL : cute::TMA::CacheHintSm100::EVICT_FIRST;
    DG_STATIC_ASSERT(not kEvictFirstHStores or kStagedPoolStores, "Evict-first h stores need the staged stores");
    // 2048: lean recompute tail. The per-task fence only has to order this CTA's h/dz stores before the completer
    // handshake, so Dh / recompute tasks skip it and each CTA fences once at the block's last recompute tile and
    // adds all its recompute tiles to the L1 empty count there (the ring and completer only read the full count).
    constexpr bool kLeanRecomputeTail = kOptions & 2048u;

    // Dispatch pull geometry: whole x and dy rows in one buffer per warp, or (opt-in, frees pipeline stages at wide
    // hidden) two slots of one x chunk and one dy chunk each, double-buffered. Every chunk holds whole 64-element groups.
    constexpr uint32_t kPullRowBytes = kHidden * sizeof(nv_bfloat16);
    constexpr bool kWholeRowPull = kNumBytesPerPull == 2 * kPullRowBytes;
    constexpr uint32_t kNumPullSlots = kWholeRowPull ? 1 : 2;
    constexpr uint32_t kPullChunkBytes = kWholeRowPull ? kPullRowBytes : kNumBytesPerPull / (2 * kNumPullSlots);
    constexpr uint32_t kNumPullChunks = kPullRowBytes / kPullChunkBytes;
    constexpr uint32_t kNumPullChunkGroups = kPullChunkBytes / 128;
    DG_STATIC_ASSERT(kWholeRowPull or (kNumBytesPerPull % (2 * kNumPullSlots) == 0 and kPullChunkBytes % 128 == 0 and
                                       kPullRowBytes % kPullChunkBytes == 0 and kNumPullChunks >= 2),
                     "Chunked pulls need 128-byte chunks that tile the row");
    DG_STATIC_ASSERT(kNumPullChunkGroups <= 32 or kWholeRowPull, "One chunk is at most one 64-element group per lane");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = (kOptions & 4u) ? 4 : 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Scheduler configs
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumEpilogueThreads;

    // Shared memory sizes
    // NOTES: BF16 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Assign shared memory
    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment) d_dtype_t l1[kNumEpilogueWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) d_dtype_t l2[kNumEpilogueWarpgroups][STORE_BLOCK_M * BLOCK_N];
        } smem_d;
        epilogue::MoEBackwardTileEpilogue<BLOCK_M> backward_epilogue;
        float dscore[BLOCK_M];
        // Packed FP4 tiles: two elements per byte
        alignas(kSharedMemoryAlignment) uint8_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K / 2];
        alignas(kSharedMemoryAlignment) uint8_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K / 2];
        uint32_t smem_sfa[kNumStages][(BLOCK_K / UMMA_BLOCK_K) * SF_BLOCK_M];
        uint32_t smem_sfb[kNumStages][(BLOCK_K / UMMA_BLOCK_K) * SF_BLOCK_N];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps * kNumPullSlots];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
        uint32_t tmem_ptr_in_smem;
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);
    if constexpr (kNumStages > 2) {
        DG_STATIC_ASSERT(sizeof(SharedStorage) <= 232448, "Shared storage exceeds the SM100 opt-in capacity");
        // Deeper opt-in pipelines size shared memory on the host; refuse to run past the allocation.
        uint32_t dynamic_smem_bytes;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dynamic_smem_bytes));
        if (sizeof(SharedStorage) > dynamic_smem_bytes)
            asm volatile("trap;");
    }

    // Send buffers
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        if (cute::elect_one_sync()) {
            // The bytes must be 8 bytes aligned
            ptx::st_shared_bulk(
                shared_storage.expert_token_count,
                math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment)
            );
        }
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps * kNumPullSlots; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at 2 CTAs, A + B
                shared_storage.full_barriers[i].init(2 * 2);
                shared_storage.empty_barriers[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                shared_storage.tmem_full_barriers[i].init(1);
                // Arrive only at the leader CTA
                shared_storage.tmem_empty_barriers[i].init(2 * kNumEpilogueThreads);
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
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, &shared_storage.tmem_ptr_in_smem);
    }
    cute::cluster_sync();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Diagnostic per-CTA phase trace (kPhaseTrace only); the pointer rides in the unused saved_z argument.
    constexpr uint32_t kTraceHeader = 96, kTraceMBlockSlots = 64, kTraceMBlockFields = 8;
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

    // Diagnostic multi-rank extension (kPhaseTrace with kDiagAblate bit 1024; no ablation): 64 slots per CTA after
    // all CTA records, globaltimer stamps (comparable across the ranks of one node) and per-peer pull statistics.
    constexpr bool kDistTrace = kPhaseTrace and (kDiagAblate & 1024u) != 0;
    DG_STATIC_ASSERT(not kDistTrace or kNumRanks <= 8, "Distributed trace supports at most 8 ranks");
    constexpr uint32_t kDistTraceStride = 64;
    unsigned long long* const dist_trace = kDistTrace
        ? phase_trace + static_cast<uint64_t>(kNumSMs - sm_idx) * kTraceCTAStride + static_cast<uint64_t>(sm_idx) * kDistTraceStride
        : nullptr;
    const auto dist_set = [&](const uint32_t& slot, const uint64_t& value) {
        if constexpr (kDistTrace) dist_trace[slot] = value;
    };
    const auto dist_add = [&](const uint32_t& slot, const uint64_t& value) {
        if constexpr (kDistTrace) atomicAdd(dist_trace + slot, static_cast<unsigned long long>(value));
    };
    const auto dist_max = [&](const uint32_t& slot, const uint64_t& value) {
        if constexpr (kDistTrace) atomicMax(dist_trace + slot, static_cast<unsigned long long>(value));
    };

    // Task scheduler
    auto scheduler = sched::MonoEPScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks, kNumLocalityDomains,
        kNumSharedExperts>(
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
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 2;

    // Adjust registers
    // NOTES: more experts per rank will cost more schedulers' registers
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
    constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 48 : 96;
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 88;
    constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 208 : 160;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

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

        if constexpr (kDistTrace) {
            if (thread_idx == 0) dist_set(0, trace_globaltimer());
        }

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

        if constexpr (kDistTrace) {
            if (thread_idx == 0) dist_set(1, trace_globaltimer());
        }

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Pull x and dy rows from remote ranks: BF16 copies go to the pools (TMA store) and NVFP4 copies with
        // per-block UE4M3 scales go to the FP4 pools (x with the expert's static scale, dy with a per-row scale)
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

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        const uint64_t trace_pull_start = trace_clock();
        uint64_t trace_ring_wait = 0, trace_copy = 0, trace_rows = 0;

        constexpr uint32_t kRowBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumRowGroups = kHidden / 64;
        constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
        DG_STATIC_ASSERT(kRowBytes == kPullRowBytes, "Pull geometry mismatch");
        struct PullRow {
            const void* src_x;
            const void* src_dy;
            float weight;
            uint32_t pool_token_idx;
            uint32_t rank;
            uint32_t src_token_idx;
            uint32_t src_topk_idx;
            int local_expert_idx;
            bool is_last_token;
        };
        const auto resolve = [&](const uint32_t& token_idx, PullRow& row) -> bool {
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }
            if (current_expert_idx >= kNumExpertsPerRank)
                return false;
            if (old_expert_idx != current_expert_idx) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }
            uint32_t current_rank_in_expert_idx = 0;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            const uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank = 0;
            while (true) {
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
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
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
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            row.src_token_idx = src_token_topk_idx / kNumTopk;
            row.src_topk_idx = src_token_topk_idx % kNumTopk;
            row.rank = current_rank_in_expert_idx;
            row.pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            // The host sizes bounded pools from the matching forward's counts; past them is a host bug, never a drop
            if (backward.pool_rows != 0 and (row.pool_token_idx / BLOCK_M + 1) * BLOCK_M > backward.pool_rows)
                asm volatile("trap;");
            row.local_expert_idx = current_expert_idx;
            row.is_last_token = token_idx == expert_end_idx - 1;
            row.src_x = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(row.src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            row.src_dy = backward.dy.map(
                backward.dy.get_base_ptr<__nv_bfloat16*>() + static_cast<uint64_t>(row.src_token_idx) * kHidden,
                current_rank_in_expert_idx);
            // Consumed only at publish, so the remote load overlaps the row copy
            row.weight = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx, current_rank_in_expert_idx);
            return true;
        };
        const auto smem_x = pull_buffer.get_base_ptr();
        const auto smem_dy = math::advance_ptr(smem_x, kRowBytes);
        const auto issue = [&](const PullRow& row) {
            if (cute::elect_one_sync()) {
                ptx::tma_load_1d(smem_x, row.src_x, pull_mbarrier, kRowBytes, kDispatchLoadHint);
                ptx::tma_load_1d(smem_dy, row.src_dy, pull_mbarrier, kRowBytes, kDispatchLoadHint);
                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, 2 * kRowBytes);
            }
            __syncwarp();
        };

        // Per-expert x quantization constants: `1 / (6 * pts)` and `1 / pts`
        float sf_scale_x = 0.0f, inv_pts_x = 0.0f;
        int scale_expert_idx = -1;
        const auto load_input_scales = [&](const int& local_expert_idx) {
            if (local_expert_idx != scale_expert_idx) {
                const auto scales = *reinterpret_cast<const float2*>(expert_scales + local_expert_idx * 8);
                sf_scale_x = scales.x, inv_pts_x = scales.y;
                scale_expert_idx = local_expert_idx;
            }
        };
        // Row amax over a BF16 row in shared memory (one 64-element group per lane per round)
        const auto row_amax = [&](const uint4* src) -> float {
            nv_bfloat162 amax2 = __floats2bfloat162_rn(0.0f, 0.0f);
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumRowGroups, 32u); ++ i) {
                const uint32_t g = i * 32 + lane_idx;
                if (kNumRowGroups % 32 == 0 or g < kNumRowGroups) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 8; ++ j) {
                        const uint4 v = ptx::ld_shared(src + g * 8 + j);
                        const auto p = reinterpret_cast<const nv_bfloat162*>(&v);
                        amax2 = __hmax2(amax2, __hmax2(__hmax2(__habs2(p[0]), __habs2(p[1])),
                                                       __hmax2(__habs2(p[2]), __habs2(p[3]))));
                    }
                }
            }
            float amax = fmaxf(__low2float(amax2), __high2float(amax2));
            #pragma unroll
            for (uint32_t delta = 16; delta > 0; delta /= 2)
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
            return amax;
        };
        // Quantize a BF16 row from shared memory into one packed FP4 pool row and its scale words
        const auto quantize_row = [&](const uint4* src, uint8_t* dst_row, uint32_t* sf_pool, const uint32_t& sf_row_idx,
                                      const float& sf_scale, const float& inv_pts) {
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumRowGroups, 32u); ++ i) {
                const uint32_t g = i * 32 + lane_idx;
                if (kNumRowGroups % 32 == 0 or g < kNumRowGroups) {
                    uint32_t sf_word = 0, words[8];
                    #pragma unroll
                    for (uint32_t b = 0; b < 4; ++ b) {
                        const auto lo = ptx::ld_shared(src + g * 8 + b * 2);
                        const auto hi = ptx::ld_shared(src + g * 8 + b * 2 + 1);
                        sf_word |= nvfp4_quantize_bf16_block(lo, hi, sf_scale, inv_pts, words[b * 2], words[b * 2 + 1]) << (b * 8);
                    }
                    const auto dst = reinterpret_cast<uint4*>(dst_row + g * 32);
                    dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
                    dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
                    sf_pool[g * backward.nv_pool_rows + sf_row_idx] = sf_word;
                }
            }
        };

        if constexpr (kWholeRowPull) {
            // The next row's routing is resolved and its load issued while this row is stored, quantized and published
            uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx;
            PullRow cur, next;
            bool has_cur = resolve(token_idx, cur);
            if (has_cur)
                issue(cur);
            while (has_cur) {
                token_idx += kNumGlobalWarps;
                const bool has_next = resolve(token_idx, next);

                // Ring slot wait for this row only (no other row of this warp is unpublished)
                const uint32_t pool_block_idx = cur.pool_token_idx / BLOCK_M;
                const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
                const uint64_t trace_ring0 = trace_clock();
                if (l1_empty_count_target > 0) {
                    const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                    while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
                }
                const uint64_t trace_copy0 = trace_clock();
                trace_ring_wait += trace_copy0 - trace_ring0;
                ++ trace_rows;

                // BF16 rows to the pools (async proxy) while the lanes quantize the same shared memory
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                if (cute::elect_one_sync()) {
                    ptx::tma_store_1d(backward.x_pool + static_cast<uint64_t>(cur.pool_token_idx) * kHidden, smem_x, kRowBytes);
                    ptx::tma_store_1d(backward.dy_pool + static_cast<uint64_t>(cur.pool_token_idx) * kHidden, smem_dy, kRowBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
                load_input_scales(cur.local_expert_idx);
                const uint32_t sf_row_idx = transform_sf_token_idx(cur.pool_token_idx);
                quantize_row(static_cast<const uint4*>(smem_x),
                             backward.qx_pool + static_cast<uint64_t>(cur.pool_token_idx) * (kHidden / 2),
                             backward.nv_sx, sf_row_idx, sf_scale_x, inv_pts_x);
                float sf_scale_dy, inv_pts_dy;
                const float pts_dy = nvfp4_row_scale(row_amax(static_cast<const uint4*>(smem_dy)), sf_scale_dy, inv_pts_dy);
                quantize_row(static_cast<const uint4*>(smem_dy),
                             backward.qdy_pool + static_cast<uint64_t>(cur.pool_token_idx) * (kHidden / 2),
                             backward.nv_sdy, sf_row_idx, sf_scale_dy, inv_pts_dy);
                if (lane_idx == 0)
                    backward.nv_rdy[cur.pool_token_idx] = pts_dy;
                // Generic reads of the pull buffer and the bulk stores must complete before the next load overwrites it
                cutlass::arch::fence_view_async_shared();
                __syncwarp();
                if (cute::elect_one_sync())
                    cute::tma_store_wait<0>();
                __syncwarp();
                if (has_next)
                    issue(next);

                // Publish the row: weight and source metadata, then the full count (release) after the FP4/SF
                // stores are made visible to the TMA loads of the consumer
                if (cute::elect_one_sync()) {
                    *buffer.l1_topk_weights_buffer.get_data_buffer(cur.pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = cur.weight;
                    *workspace.get_token_src_metadata_ptr(cur.pool_token_idx) = {cur.rank, cur.src_token_idx, cur.src_topk_idx};
                }
                nvfp4_fence_proxy_async_global();
                __syncwarp();
                if (cute::elect_one_sync()) {
                    ptx::red_add_rel(
                        workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks),
                        cur.is_last_token ? BLOCK_M - (cur.pool_token_idx % BLOCK_M) : 1u);
                }
                __syncwarp();
                const uint64_t trace_row_copy = trace_clock() - trace_copy0;
                trace_copy += trace_row_copy;
                if constexpr (kDistTrace) {
                    if (lane_idx == 0) {
                        const uint64_t arrival = trace_globaltimer();
                        dist_add(8 + cur.rank, 1);
                        dist_max(16 + cur.rank, ~arrival);
                        dist_max(24 + cur.rank, arrival);
                        dist_add(32 + cur.rank, trace_row_copy);
                    }
                }
                cur = next;
                has_cur = has_next;
            }
        } else {
            // Chunked pull: chunk k+1 (or the next row's first chunk) is in flight in the other slot while chunk k is
            // stored to the BF16 pools and quantized. x takes the expert's static scale per chunk; dy needs the row
            // amax, so its BF16 chunks go to the pool first and the row is quantized from the pool (L2-resident)
            // once its last chunk has landed.
            const auto slot_x = [&](const uint32_t& s) {
                return math::advance_ptr(pull_buffer.get_base_ptr(), s * 2 * kPullChunkBytes);
            };
            const auto slot_dy = [&](const uint32_t& s) {
                return math::advance_ptr(pull_buffer.get_base_ptr(), s * 2 * kPullChunkBytes + kPullChunkBytes);
            };
            uint32_t slot_phase_0 = 0, slot_phase_1 = 0;
            const auto wait_slot = [&](const uint32_t& s) {
                if (s == 0)
                    ptx::mbarrier_wait_and_flip_phase(&shared_storage.dispatch_barriers[warp_idx * 2], slot_phase_0);
                else
                    ptx::mbarrier_wait_and_flip_phase(&shared_storage.dispatch_barriers[warp_idx * 2 + 1], slot_phase_1);
            };
            const auto issue_chunk = [&](const PullRow& row, const uint32_t& chunk_idx, const uint32_t& s) {
                if (cute::elect_one_sync()) {
                    const auto mbarrier = &shared_storage.dispatch_barriers[warp_idx * 2 + s];
                    ptx::tma_load_1d(slot_x(s), static_cast<const uint8_t*>(row.src_x) + chunk_idx * kPullChunkBytes,
                                     mbarrier, kPullChunkBytes, kDispatchLoadHint);
                    ptx::tma_load_1d(slot_dy(s), static_cast<const uint8_t*>(row.src_dy) + chunk_idx * kPullChunkBytes,
                                     mbarrier, kPullChunkBytes, kDispatchLoadHint);
                    ptx::mbarrier_arrive_and_set_tx(mbarrier, 2 * kPullChunkBytes);
                }
                __syncwarp();
            };
            // One 64-element group per lane: quantize this x chunk with the static expert scale
            const auto quantize_x_chunk = [&](const uint4* src, uint8_t* dst_row, const uint32_t& chunk_idx,
                                              const uint32_t& sf_row_idx) {
                const uint32_t g = lane_idx;
                if (kNumPullChunkGroups == 32 or g < kNumPullChunkGroups) {
                    uint32_t sf_word = 0, words[8];
                    #pragma unroll
                    for (uint32_t b = 0; b < 4; ++ b) {
                        const auto lo = ptx::ld_shared(src + g * 8 + b * 2);
                        const auto hi = ptx::ld_shared(src + g * 8 + b * 2 + 1);
                        sf_word |= nvfp4_quantize_bf16_block(lo, hi, sf_scale_x, inv_pts_x, words[b * 2], words[b * 2 + 1]) << (b * 8);
                    }
                    const uint32_t group_idx = chunk_idx * kNumPullChunkGroups + g;
                    const auto dst = reinterpret_cast<uint4*>(dst_row + group_idx * 32);
                    dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
                    dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
                    backward.nv_sx[group_idx * backward.nv_pool_rows + sf_row_idx] = sf_word;
                }
            };
            const auto chunk_amax = [&](const uint4* src, nv_bfloat162& amax2) {
                const uint32_t g = lane_idx;
                if (kNumPullChunkGroups == 32 or g < kNumPullChunkGroups) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 8; ++ j) {
                        const uint4 v = ptx::ld_shared(src + g * 8 + j);
                        const auto p = reinterpret_cast<const nv_bfloat162*>(&v);
                        amax2 = __hmax2(amax2, __hmax2(__hmax2(__habs2(p[0]), __habs2(p[1])),
                                                       __hmax2(__habs2(p[2]), __habs2(p[3]))));
                    }
                }
            };
            // Quantize a BF16 pool row (written by this warp's bulk stores, read back through L2) with a row scale
            const auto quantize_row_from_pool = [&](const __nv_bfloat16* pool_row, uint8_t* dst_row, uint32_t* sf_pool,
                                                    const uint32_t& sf_row_idx, const float& sf_scale, const float& inv_pts) {
                const auto src = reinterpret_cast<const uint4*>(pool_row);
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumRowGroups, 32u); ++ i) {
                    const uint32_t g = i * 32 + lane_idx;
                    if (kNumRowGroups % 32 == 0 or g < kNumRowGroups) {
                        uint32_t sf_word = 0, words[8];
                        #pragma unroll
                        for (uint32_t b = 0; b < 4; ++ b) {
                            const auto lo = __ldcg(src + g * 8 + b * 2);
                            const auto hi = __ldcg(src + g * 8 + b * 2 + 1);
                            sf_word |= nvfp4_quantize_bf16_block(lo, hi, sf_scale, inv_pts, words[b * 2], words[b * 2 + 1]) << (b * 8);
                        }
                        const auto dst = reinterpret_cast<uint4*>(dst_row + g * 32);
                        dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
                        dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
                        sf_pool[g * backward.nv_pool_rows + sf_row_idx] = sf_word;
                    }
                }
            };

            uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx;
            PullRow cur, next;
            bool has_cur = resolve(token_idx, cur);
            uint32_t slot = 0;
            if (has_cur)
                issue_chunk(cur, 0, slot);
            while (has_cur) {
                token_idx += kNumGlobalWarps;
                const bool has_next = resolve(token_idx, next);

                const uint32_t pool_block_idx = cur.pool_token_idx / BLOCK_M;
                const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
                const uint64_t trace_ring0 = trace_clock();
                if (l1_empty_count_target > 0) {
                    const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                    while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
                }
                const uint64_t trace_copy0 = trace_clock();
                trace_ring_wait += trace_copy0 - trace_ring0;
                ++ trace_rows;

                load_input_scales(cur.local_expert_idx);
                const uint32_t sf_row_idx = transform_sf_token_idx(cur.pool_token_idx);
                auto* x_row = backward.x_pool + static_cast<uint64_t>(cur.pool_token_idx) * kHidden;
                auto* dy_row = backward.dy_pool + static_cast<uint64_t>(cur.pool_token_idx) * kHidden;
                auto* qx_row = backward.qx_pool + static_cast<uint64_t>(cur.pool_token_idx) * (kHidden / 2);
                auto* qdy_row = backward.qdy_pool + static_cast<uint64_t>(cur.pool_token_idx) * (kHidden / 2);
                nv_bfloat162 amax2 = __floats2bfloat162_rn(0.0f, 0.0f);
                #pragma unroll 1
                for (uint32_t chunk_idx = 0; chunk_idx < kNumPullChunks; ++ chunk_idx) {
                    // The other slot's previous chunk finished its shared-memory reads at the end of its iteration
                    const uint32_t other = slot ^ 1;
                    if (chunk_idx + 1 < kNumPullChunks)
                        issue_chunk(cur, chunk_idx + 1, other);
                    else if (has_next)
                        issue_chunk(next, 0, other);
                    wait_slot(slot);
                    if (cute::elect_one_sync()) {
                        ptx::tma_store_1d(math::advance_ptr(x_row, chunk_idx * kPullChunkBytes), slot_x(slot), kPullChunkBytes);
                        ptx::tma_store_1d(math::advance_ptr(dy_row, chunk_idx * kPullChunkBytes), slot_dy(slot), kPullChunkBytes);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    quantize_x_chunk(static_cast<const uint4*>(slot_x(slot)), qx_row, chunk_idx, sf_row_idx);
                    chunk_amax(static_cast<const uint4*>(slot_dy(slot)), amax2);
                    // Generic reads and the bulk stores' reads of this slot must complete before it is reloaded
                    cutlass::arch::fence_view_async_shared();
                    __syncwarp();
                    if (cute::elect_one_sync())
                        cute::tma_store_wait<0>();
                    __syncwarp();
                    slot = other;
                }
                float amax = fmaxf(__low2float(amax2), __high2float(amax2));
                #pragma unroll
                for (uint32_t delta = 16; delta > 0; delta /= 2)
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
                float sf_scale_dy, inv_pts_dy;
                const float pts_dy = nvfp4_row_scale(amax, sf_scale_dy, inv_pts_dy);
                // The row's BF16 bulk stores must be complete (not just read) and visible to the generic proxy
                if (cute::elect_one_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                    nvfp4_fence_proxy_async_global();
                }
                __syncwarp();
                nvfp4_fence_proxy_async_global();
                quantize_row_from_pool(dy_row, qdy_row, backward.nv_sdy, sf_row_idx, sf_scale_dy, inv_pts_dy);
                if (lane_idx == 0)
                    backward.nv_rdy[cur.pool_token_idx] = pts_dy;

                if (cute::elect_one_sync()) {
                    *buffer.l1_topk_weights_buffer.get_data_buffer(cur.pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = cur.weight;
                    *workspace.get_token_src_metadata_ptr(cur.pool_token_idx) = {cur.rank, cur.src_token_idx, cur.src_topk_idx};
                }
                nvfp4_fence_proxy_async_global();
                __syncwarp();
                if (cute::elect_one_sync()) {
                    ptx::red_add_rel(
                        workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks),
                        cur.is_last_token ? BLOCK_M - (cur.pool_token_idx % BLOCK_M) : 1u);
                }
                __syncwarp();
                const uint64_t trace_row_copy = trace_clock() - trace_copy0;
                trace_copy += trace_row_copy;
                if constexpr (kDistTrace) {
                    if (lane_idx == 0) {
                        const uint64_t arrival = trace_globaltimer();
                        dist_add(8 + cur.rank, 1);
                        dist_max(16 + cur.rank, ~arrival);
                        dist_max(24 + cur.rank, arrival);
                        dist_add(32 + cur.rank, trace_row_copy);
                    }
                }
                cur = next;
                has_cur = has_next;
            }
        }

        if constexpr (kPhaseTrace) {
            if (lane_idx == 0) {
                const uint64_t trace_pull_end = trace_clock();
                trace_set(46 + warp_idx * 3, trace_pull_end - trace_pull_start);
                trace_set(47 + warp_idx * 3, trace_ring_wait);
                trace_set(48 + warp_idx * 3, trace_copy);
                trace_add(58, trace_rows);
                atomicMax(phase_trace + 59, static_cast<unsigned long long>(trace_pull_end));
                if (warp_idx == 0) trace_set(60, trace_pull_start);
                dist_max(2, trace_globaltimer());
            }
        }

        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        if constexpr (kDistTrace) {
            if (thread_idx == 0) dist_set(3, trace_globaltimer());
        }
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

                if (thread_idx == 0) {
                    backward.expert_starts[i] = expert_pool_block_offset * BLOCK_M;
                    backward.expert_counts[i] = num_recv_tokens;
                }

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
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
        if constexpr (kPhaseTrace) {
            if (thread_idx == 0) trace_set(62, trace_clock());
            if (thread_idx == 0) dist_set(4, trace_globaltimer());
        }
    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens
        task_info_t task_info;
        uint64_t trace_l1_spin = 0, trace_l2_spin = 0, trace_a_empty = 0;
        while (scheduler.get_next_task(task_info)) {
            const bool is_dh = task_info.block_phase == sched::BlockPhase::BackwardDh;
            const bool is_dx = task_info.block_phase == sched::BlockPhase::BackwardDx;
            const auto tensor_map_a_ptr = is_dh ? &tensor_map_dy : is_dx ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = is_dh ? &tensor_map_dy_sf : is_dx ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;
            const auto num_k_blocks = math::ceil_div(task_info.shape_k, BLOCK_K);

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;

            const uint64_t trace_spin0 = trace_clock();
            if (not is_dx) {
                const auto expected = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(workspace.get_l1_full_count_ptr(ring_block_idx)) != expected);
                trace_l1_spin += trace_clock() - trace_spin0;
            } else {
                // The completer releases every mask bit of the M block once the FP4 dz rows are written
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

                // Pool row offsets (K coordinates count FP4 elements)
                uint32_t m_idx = pool_block_idx * BLOCK_M;
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sfa_m_idx = pool_block_idx * SF_BLOCK_M;
                const uint32_t sfa_k_idx = k_block_idx * (BLOCK_K / UMMA_BLOCK_K);

                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += task_info.get_umma_aligned_valid_m() / 2;

                // TMA copy tokens and SFA, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx],
                        reinterpret_cast<a_dtype_t*>(shared_storage.smem_a[stage_idx]), k_idx, m_idx, 2);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        // Multicast
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(shared_storage.smem_a[0]) * 2 + sizeof(shared_storage.smem_sfa[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        }
        if constexpr (kPhaseTrace) {
            if (lane_idx == 0) {
                trace_set(43, trace_l1_spin);
                trace_set(44, trace_l2_spin);
                trace_set(45, trace_a_empty);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const bool is_dh = task_info.block_phase == sched::BlockPhase::BackwardDh;
            const bool is_dx = task_info.block_phase == sched::BlockPhase::BackwardDx;
            const auto tensor_map_b_ptr = is_dh ? &tensor_map_dh_weights : is_dx ? &tensor_map_l2_weights : &tensor_map_l1_weights;
            const auto tensor_map_sfb_ptr = is_dh ? &tensor_map_dh_weights_sf : is_dx ? &tensor_map_l2_weights_sf : &tensor_map_l1_weights_sf;
            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto n_block_idx = task_info.get_n_block_idx(is_leader_cta);
            const auto num_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            const uint32_t half_n = shape_n / 2, n_idx = n_block_idx * BLOCK_N, half_idx = n_idx >= half_n;
            const uint32_t half_n_idx = n_idx - half_idx * half_n, expert_idx = task_info.local_expert_idx;
            // Weight SF atoms are `[expert][n_block][k_group][128 words]`: one TMA box covers the stage's K groups
            const uint32_t sfb_row_idx = expert_idx * (shape_n / kNumUTCCPAlignedElems) + n_block_idx;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute weight offset (FP4 elements)
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sfb_k_idx = k_block_idx * (BLOCK_K / UMMA_BLOCK_K);

                // TMA copy weights and SFB
                if (cute::elect_one_sync()) {
                    tma::copy_nd<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, 4>(
                        tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx],
                        reinterpret_cast<b_dtype_t*>(shared_storage.smem_b[stage_idx]), 2, k_idx, half_n_idx, half_idx, expert_idx);
                    tma::copy_nd<kNumUTCCPAlignedElems, 1, 0, uint32_t, 3>(
                        tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], 2, 0u, sfb_k_idx, sfb_row_idx);
                    if (is_leader_cta) {
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(shared_storage.smem_b[0]) * 2 + sizeof(shared_storage.smem_sfb[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        if (is_leader_cta) {
            // Make instruction descriptor
            // NOTES: always swap A/B
            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                b_dtype_t, a_dtype_t, float, sf_dtype_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K>();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            // NOTES: the swizzle atom spans the whole `BLOCK_K` (256 packed FP4 = 128 bytes), and each K = 64
            //        instruction advances 32 bytes inside it
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(reinterpret_cast<a_dtype_t*>(shared_storage.smem_a[0]), 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(reinterpret_cast<b_dtype_t*>(shared_storage.smem_b[0]), 0, 0);
            const uint32_t a_desc_lo = a_desc.lo;
            const uint32_t b_desc_lo = b_desc.lo;

            // Checks for MMA instructions
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            task_info_t task_info;
            uint64_t trace_mma_prev = trace_clock();
            if constexpr (kPhaseTrace) {
                if (lane_idx == 0) trace_set(32, trace_mma_prev);
                if (lane_idx == 0) dist_set(48, trace_globaltimer());
            }
            while (scheduler.get_next_task(task_info)) {
                const uint64_t trace_mma_task = trace_clock();
                uint64_t trace_mma_full = 0;
                const auto num_k_blocks = task_info.shape_k / BLOCK_K;
                // Dynamic update of UMMA N based on effective M
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, task_info.get_umma_aligned_valid_m());

                // Wait tensor memory empty barrier arrival
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();
                const uint64_t trace_mma_empty_done = trace_clock();

                // Empty barrier arrival
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.empty_barriers[stage_idx]));

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait TMA load completion
                    const uint64_t trace_full0 = trace_clock();
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_after_thread_sync();
                    trace_mma_full += trace_clock() - trace_full0;

                    const uint32_t a_desc_base_lo = a_desc_lo + stage_idx * sizeof(SharedStorage::smem_a[0]) / 16;
                    const uint32_t b_desc_base_lo = b_desc_lo + stage_idx * sizeof(SharedStorage::smem_b[0]) / 16;
                    if (cute::elect_one_sync()) {
                        const auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, 0, 0);
                        #pragma unroll
                        for (uint32_t umma_k_block_idx = 0; umma_k_block_idx < BLOCK_K / UMMA_BLOCK_K; ++ umma_k_block_idx) {
                            // UTCCP copy SFA and SFB to TMEM (one 512-byte atom per 128 rows per K = 64)
                            using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfa[stage_idx] + umma_k_block_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfb[stage_idx] + umma_k_block_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            }

                            // Issue UMMA
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, umma_k_block_idx * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, umma_k_block_idx * UMMA_K);
                            ptx::SM100_MMA_MXF4NVF4_2x1SM_SS::fma(
                                b_desc, a_desc, accum_stage_idx * UMMA_N,
                                k_block_idx > 0 or umma_k_block_idx > 0, runtime_instr_desc,
                                kTmemStartColOfSFB, kTmemStartColOfSFA);
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier object
                    // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
                if constexpr (kPhaseTrace) {
                    const uint64_t trace_mma_end = trace_clock();
                    if (lane_idx == 0) {
                        const uint32_t phase_idx = task_info.block_phase == sched::BlockPhase::BackwardDh ? 0u :
                            task_info.block_phase == sched::BlockPhase::BackwardRecomputeLow ? 1u :
                            task_info.block_phase == sched::BlockPhase::BackwardRecomputeHigh ? 2u : 3u;
                        trace_add(30, trace_mma_empty_done - trace_mma_task);
                        trace_add(31, trace_mma_full);
                        trace_add(34 + phase_idx, trace_mma_end - trace_mma_empty_done);
                        trace_add(38 + phase_idx, trace_mma_empty_done - trace_mma_task);
                        trace_add(42, trace_mma_task - trace_mma_prev);
                        trace_add(64 + phase_idx, num_k_blocks);
                    }
                    trace_mma_prev = trace_mma_end;
                }
            }
            if constexpr (kPhaseTrace) {
                if (lane_idx == 0) trace_set(33, trace_clock());
                if (lane_idx == 0) dist_set(49, trace_globaltimer());
            }

            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages].wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // Do mainloop by the leader CTA
        if (is_leader_cta) {
            if constexpr (kOwnMBlocks)
                scheduler.template mainloop_owned<true>();
            else
                scheduler.mainloop(num_tokens, sm_locality_domains);
        }
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&shared_storage.tmem_ptr_in_smem) == 0);

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
        constexpr uint32_t kNumBankGroupBytes = 16u;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks
        uint32_t current_iter_idx = 0;
        task_info_t task_info;
        // Recompute N blocks this CTA finished for the current M block; released on the mask by the completer
        uint64_t completed_n_blocks = 0;
        const bool trace_leader = epilogue_thread_idx == 0;
        uint32_t trace_mblock = 0;
        uint64_t trace_prev_end = trace_clock();
        if constexpr (kPhaseTrace) {
            if (trace_leader) trace_set(4, trace_prev_end);
            if (trace_leader) dist_set(40, trace_globaltimer());
        }
        while (scheduler.get_next_task(task_info)) {
            const uint64_t trace_task = trace_clock();
            // Wait UMMA arrival
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();
            const uint64_t trace_got = trace_clock();
            uint64_t trace_spin = 0, trace_atoms = 0, trace_tail = 0, trace_dscore_partial = 0, trace_dx_tail = 0;

            // Now we can release the task
            scheduler.release_task_info();

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring-buffer offset for reusable data buffers
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;  // Full-pool offset for non-ring metadata
            const uint32_t n_block_idx = task_info.get_n_block_idx(is_leader_cta);
            uint32_t n_idx = n_block_idx * BLOCK_N;

            // Per-expert output folds: `pts_x * pts_w1` (recompute), `pts_w2t` (dH, times the dy row scale),
            // `pts_w1t` (dx, times the dz row scale)
            const auto expert_scale_ptr = expert_scales + task_info.local_expert_idx * 8;
            const float fold_recompute = expert_scale_ptr[2];
            const float pts_w2t = expert_scale_ptr[3];
            const float pts_w1t = expert_scale_ptr[4];

            if (task_info.block_phase != sched::BlockPhase::BackwardDx) {
                const bool is_dh = task_info.block_phase == sched::BlockPhase::BackwardDh;
                if (is_dh and task_info.n_cluster_idx == 0) {
                    const auto expected = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(workspace.get_l2_empty_count_ptr(ring_block_idx)) != expected);
                    for (uint32_t row = epilogue_thread_idx; row < BLOCK_M; row += kNumEpilogueThreads)
                        shared_storage.dscore[row] = 0;
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    trace_spin = trace_clock() - trace_got;
                }
                const uint64_t trace_atom0 = trace_clock();
                constexpr uint32_t kNumAtoms = WG_BLOCK_M / ATOM_M;
                constexpr uint32_t kStageRows = 2 * ATOM_M, kStageStride = 64 + 8;
                DG_STATIC_ASSERT(not kStagedPoolStores or
                                 3 * kStageRows * kStageStride * sizeof(nv_bfloat16) <= sizeof(shared_storage.smem_d.l1[0]),
                                 "h/dz staging does not fit the L1 output staging buffer");
                DG_STATIC_ASSERT(kNumAtoms % 2 == 0, "Staged pool stores pair atoms");
                const auto process_atom = [&](const uint32_t& atom, const uint32_t (&values)[8]) {
                    const auto atom_row = epilogue_wg_idx * WG_BLOCK_M + atom * ATOM_M;
                    if (is_dh) {
                        const auto m0 = atom_row + (lane_idx % 4) * 2;
                        const float rs0 = m0 < valid_m ? backward.nv_rdy[pool_m_idx + m0] * pts_w2t : 0.0f;
                        const float rs1 = m0 + 1 < valid_m ? backward.nv_rdy[pool_m_idx + m0 + 1] * pts_w2t : 0.0f;
                        uint32_t scaled[8];
                        #pragma unroll
                        for (uint32_t i = 0; i < 8; ++ i)
                            scaled[i] = __float_as_uint(__uint_as_float(values[i]) * (i % 2 ? rs1 : rs0));
                        if constexpr (not (kDiagAblate & 1u))
                            shared_storage.backward_epilogue.retain_dh_atom(scaled, atom_row, warp_idx_in_wg, lane_idx, valid_m);
                    } else {
                        const auto row = atom_row + (lane_idx % 4) * 2;
                        const auto half = task_info.block_phase == sched::BlockPhase::BackwardRecomputeHigh ? 1u : 0u;
                        const auto score_ptr = buffer.l1_topk_weights_buffer.get_base_ptr<float>();
                        const float2 scores = {
                            row < valid_m ? score_ptr[ring_m_idx + row] : 0.0f,
                            row + 1 < valid_m ? score_ptr[ring_m_idx + row + 1] : 0.0f};
                        epilogue::MoEBackwardAtom result;
                        if constexpr (kDiagAblate & 8u) {
                            #pragma unroll
                            for (uint32_t col = 0; col < 2; ++col) {
                                result.h[col] = __floats2bfloat162_rn(__uint_as_float(values[col * 4]), __uint_as_float(values[col * 4 + 1]));
                                result.gate[col] = __floats2bfloat162_rn(__uint_as_float(values[col * 4 + 2]), __uint_as_float(values[col * 4 + 3]));
                                result.up[col] = result.h[col];
                            }
                            result.dscore = {__uint_as_float(values[0]), __uint_as_float(values[4])};
                        } else {
                            uint32_t z[8];
                            #pragma unroll
                            for (uint32_t i = 0; i < 8; ++ i)
                                z[i] = __float_as_uint(__uint_as_float(values[i]) * fold_recompute);
                            result = shared_storage.backward_epilogue.recompute_atom(
                                z, atom_row, warp_idx_in_wg, lane_idx, half, scores);
                        }
                        if constexpr (kDiagAblate & 2u) {
                            // Keep the derivative math live without storing it.
                            uint32_t sink = 0;
                            #pragma unroll
                            for (uint32_t col = 0; col < 2; ++col) {
                                sink ^= *reinterpret_cast<const uint32_t*>(&result.h[col]);
                                sink ^= *reinterpret_cast<const uint32_t*>(&result.gate[col]);
                                sink ^= *reinterpret_cast<const uint32_t*>(&result.up[col]);
                            }
                            if (sink == 0x7fc17fc1u and valid_m == 0xffffffffu)
                                *static_cast<uint32_t*>(y) = sink;
                        } else {
                        if constexpr (kStagedPoolStores) {
                            // Stage the WG's 64 h/dg/du columns of this atom in shared memory, then write two atoms
                            // (16 full rows) at a time with 16-byte stores.
                            const auto stage = reinterpret_cast<nv_bfloat16*>(shared_storage.smem_d.l1[epilogue_wg_idx]);
                            #pragma unroll
                            for (uint32_t col = 0; col < 2; ++col) {
                                const auto c = warp_idx_in_wg * 16 + lane_idx / 4 + col * 8;
                                #pragma unroll
                                for (uint32_t v = 0; v < 2; ++v) {
                                    const auto r = (atom % 2) * ATOM_M + (lane_idx % 4) * 2 + v;
                                    const bool valid = row + v < valid_m;
                                    const auto zero = __float2bfloat16_rn(0.0f);
                                    stage[(0 * kStageRows + r) * kStageStride + c] = valid ? (v ? result.h[col].y : result.h[col].x) : zero;
                                    stage[(1 * kStageRows + r) * kStageStride + c] = valid ? (v ? result.gate[col].y : result.gate[col].x) : zero;
                                    stage[(2 * kStageRows + r) * kStageStride + c] = valid ? (v ? result.up[col].y : result.up[col].x) : zero;
                                }
                            }
                            if (atom % 2 == 1) {
                                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                                const uint32_t t = warp_idx_in_wg * 32 + lane_idx;
                                const uint32_t r = t / 8, chunk = t % 8;
                                const auto packed = static_cast<uint64_t>(pool_m_idx + epilogue_wg_idx * WG_BLOCK_M + (atom - 1) * ATOM_M + r);
                                const auto n = n_block_idx * 64 + chunk * 8;
                                #pragma unroll
                                for (uint32_t o = 0; o < 3; ++o) {
                                    const auto value = ptx::ld_shared(reinterpret_cast<float4*>(stage + (o * kStageRows + r) * kStageStride + chunk * 8));
                                    auto* dst = o == 0 ? backward.h_pool + packed * kIntermediateHidden + n :
                                        backward.dz_pool + packed * (2 * kIntermediateHidden) + (o - 1) * kIntermediateHidden + n;
                                    if constexpr (kEvictFirstHStores) {
                                        if (o == 0) {
                                            asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" ::
                                                "l"(dst), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w),
                                                "l"(static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST)) : "memory");
                                        } else {
                                            *reinterpret_cast<float4*>(dst) = value;
                                        }
                                    } else {
                                        *reinterpret_cast<float4*>(dst) = value;
                                    }
                                }
                                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                            }
                        } else {
                        for (uint32_t col = 0; col < 2; ++col) {
                            const auto n = n_block_idx * 64 + warp_idx_in_wg * 16 + lane_idx / 4 + col * 8;
                            const auto h = __bfloat1622float2(result.h[col]);
                            const auto dg = __bfloat1622float2(result.gate[col]);
                            const auto du = __bfloat1622float2(result.up[col]);
                            for (uint32_t v = 0; v < 2; ++v) {
                                const auto packed = static_cast<uint64_t>(pool_m_idx + row + v);
                                const bool valid = row + v < valid_m;
                                backward.h_pool[packed * kIntermediateHidden + n] = __float2bfloat16_rn(valid ? (v ? h.y : h.x) : 0.0f);
                                backward.dz_pool[packed * (2 * kIntermediateHidden) + n] = __float2bfloat16_rn(valid ? (v ? dg.y : dg.x) : 0.0f);
                                backward.dz_pool[packed * (2 * kIntermediateHidden) + kIntermediateHidden + n] = __float2bfloat16_rn(valid ? (v ? du.y : du.x) : 0.0f);
                            }
                        }
                        }
                        }
                        if constexpr (not (kDiagAblate & 4u)) {
                        auto ds = result.dscore;
                        for (uint32_t delta = 4; delta <= 16; delta *= 2) {
                            ds.x += __shfl_xor_sync(0xffffffff, ds.x, delta);
                            ds.y += __shfl_xor_sync(0xffffffff, ds.y, delta);
                        }
                        if (lane_idx < 4) {
                            if (row < valid_m) atomicAdd(shared_storage.dscore + row, ds.x);
                            if (row + 1 < valid_m) atomicAdd(shared_storage.dscore + row + 1, ds.y);
                        }
                        }
                    }
                };
                if constexpr (kEarlyTmemRelease) {
                    // Drain the whole accumulator into registers so the MMA warp can reuse the stage during the math
                    uint32_t atom_values[kNumAtoms][8];
                    #pragma unroll
                    for (uint32_t atom = 0; atom < kNumAtoms; ++atom) {
                        const auto tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + atom * ATOM_M;
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr, atom_values[atom][0], atom_values[atom][1], atom_values[atom][2], atom_values[atom][3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000, atom_values[atom][4], atom_values[atom][5], atom_values[atom][6], atom_values[atom][7]);
                    }
                    cutlass::arch::fence_view_async_tmem_load();
                    ptx::tcgen05_before_thread_sync();
                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                    #pragma unroll
                    for (uint32_t atom = 0; atom < kNumAtoms; ++atom)
                        process_atom(atom, atom_values[atom]);
                } else {
                    for (uint32_t atom = 0; atom < kNumAtoms; ++atom) {
                        const auto tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + atom * ATOM_M;
                        uint32_t values[8];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr, values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000, values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();
                        process_atom(atom, values);
                    }
                }
                const uint64_t trace_atom1 = trace_clock();
                trace_atoms = trace_atom1 - trace_atom0;
                if constexpr (not kEarlyTmemRelease) {
                    ptx::tcgen05_before_thread_sync();
                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                }
                constexpr uint32_t kNumInnerClusters = kIntermediateHidden / (2 * BLOCK_N);
                const bool is_last_recompute = task_info.block_phase == sched::BlockPhase::BackwardRecomputeHigh and
                    task_info.n_cluster_idx + 1 == kNumInnerClusters;
                if (not kLeanRecomputeTail or is_last_recompute) {
                    __threadfence();
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (not is_dh) {
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        if constexpr (kLeanRecomputeTail) {
                            if (is_last_recompute)
                                ptx::red_add(workspace.get_l1_empty_count_ptr(ring_block_idx), 2u * kNumInnerClusters);
                        } else {
                            ptx::red_add(workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                        }
                    }
                    completed_n_blocks ^= 1ull << n_block_idx;
                    if (is_last_recompute) {
                        trace_dscore_partial = trace_clock();
                        for (uint32_t row = epilogue_thread_idx; row < valid_m; row += kNumEpilogueThreads) {
                            const auto src = *workspace.get_token_src_metadata_ptr(pool_m_idx + row);
                            const auto& [rank, token, slot] = src;
                            auto* dst = backward.dscore_partials.map(backward.dscore_partials.get_base_ptr<float*>(), rank);
                            dst[(static_cast<uint64_t>(token) * kNumTopk + slot) * 2 + (is_leader_cta ? 0u : 1u)] = shared_storage.dscore[row];
                        }
                        __threadfence_system();
                        trace_dscore_partial = trace_clock() - trace_dscore_partial;

                        // Completer: once both CTAs finished every recompute tile of this M block, quantize the
                        // BF16 dz rows for the Dx phase (per-row dynamic scale, this CTA's half of the rows),
                        // then release the whole mask
                        const auto expected_empty = (pool_block_idx / kNumRingBlocks + 1) * (L1_SHAPE_N / BLOCK_N);
                        while (ptx::ld_acq(workspace.get_l1_empty_count_ptr(ring_block_idx)) != expected_empty);
                        constexpr uint32_t kDzCols = 2 * kIntermediateHidden, kNumDzGroups = kDzCols / 64;
                        const uint32_t row_begin = is_leader_cta ? 0u : BLOCK_M / 2;
                        for (uint32_t r = row_begin + epilogue_warp_idx; r < row_begin + BLOCK_M / 2 and r < valid_m; r += kNumEpilogueWarps) {
                            const auto pool_row = static_cast<uint64_t>(pool_m_idx + r);
                            const auto src = reinterpret_cast<const uint4*>(backward.dz_pool + pool_row * kDzCols);
                            nv_bfloat162 amax2 = __floats2bfloat162_rn(0.0f, 0.0f);
                            for (uint32_t g = lane_idx; g < kNumDzGroups; g += 32) {
                                #pragma unroll
                                for (uint32_t j = 0; j < 8; ++ j) {
                                    const uint4 v = src[g * 8 + j];
                                    const auto p = reinterpret_cast<const nv_bfloat162*>(&v);
                                    amax2 = __hmax2(amax2, __hmax2(__hmax2(__habs2(p[0]), __habs2(p[1])),
                                                                   __hmax2(__habs2(p[2]), __habs2(p[3]))));
                                }
                            }
                            float amax = fmaxf(__low2float(amax2), __high2float(amax2));
                            #pragma unroll
                            for (uint32_t delta = 16; delta > 0; delta /= 2)
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
                            float sf_scale, inv_pts;
                            const float pts = nvfp4_row_scale(amax, sf_scale, inv_pts);
                            if (lane_idx == 0)
                                backward.nv_rdz[pool_row] = pts;
                            const auto sf_row_idx = transform_sf_token_idx(pool_m_idx + r);
                            auto* dst_row = backward.qdz_pool + pool_row * (kDzCols / 2);
                            for (uint32_t g = lane_idx; g < kNumDzGroups; g += 32) {
                                uint32_t sf_word = 0, words[8];
                                #pragma unroll
                                for (uint32_t b = 0; b < 4; ++ b) {
                                    const uint4 lo = src[g * 8 + b * 2];
                                    const uint4 hi = src[g * 8 + b * 2 + 1];
                                    sf_word |= nvfp4_quantize_bf16_block(lo, hi, sf_scale, inv_pts, words[b * 2], words[b * 2 + 1]) << (b * 8);
                                }
                                const auto dst = reinterpret_cast<uint4*>(dst_row + g * 32);
                                dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
                                dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
                                backward.nv_sdz[g * backward.nv_pool_rows + sf_row_idx] = sf_word;
                            }
                        }
                        __threadfence();
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_warp_idx == 0 and cute::elect_one_sync())
                            ptx::red_xor_rel(workspace.get_l2_full_mask_ptr(ring_block_idx), completed_n_blocks);
                        completed_n_blocks = 0;
                    }
                }
                trace_tail = trace_clock() - trace_atom1;
            } else {
                // Increment L2 empty count for this physical slot (one per N block)
                if (not task_info.is_shared()) {
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_add(
                            workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                    }
                    __syncwarp();
                }

                DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / 8;

                const uint64_t trace_dx0 = trace_clock();
                if constexpr (kDiagAblate & 16u) {
                    ptx::tcgen05_before_thread_sync();
                    shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                } else {
                // L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    // TODO: check performance
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        break;
                    }

                    // Read the source metadata of this warp's rows before the TMEM loads to overlap the latency
                    layout::TokenSrcMetadata cached_src_metadata[kNumRowsPerWarp];
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        if (m_idx_in_block < valid_m)
                            cached_src_metadata[j] = task_info.is_shared() ?
                                layout::TokenSrcMetadata(sym_buffer.rank_idx, pool_m_idx + m_idx_in_block, kNumTopk) :
                                *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                    }
                    __syncwarp();

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                        // Load from TMEM using .16x256b shape to satisfy STSM layout requirements
                        // Start from lane index 0 and 16
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();
                        {
                            const auto m0 = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + (lane_idx % 4) * 2;
                            const float rs0 = m0 < valid_m ? backward.nv_rdz[pool_m_idx + m0] * pts_w1t : 0.0f;
                            const float rs1 = m0 + 1 < valid_m ? backward.nv_rdz[pool_m_idx + m0 + 1] * pts_w1t : 0.0f;
                            #pragma unroll
                            for (uint32_t j = 0; j < ATOM_M; ++ j)
                                values[j] = __float_as_uint(__uint_as_float(values[j]) * (j % 2 ? rs1 : rs0));
                        }

                        // Wait shared memory release from previous NVLink store
                        // NOTES: skip for the first store block since the prior full barrier already ensures completion
                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Signal tensor memory consumed
                        if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }

                        // Store into shared memory
                        // NOTES: each lane provides its own address for stmatrix; 2 warps share a BF16 swizzle atom
                        uint32_t row = lane_idx % 8;
                        uint32_t col = (warp_idx_in_wg % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = shared_storage.smem_d.l2[epilogue_wg_idx] +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M * (kSwizzleCDMode / sizeof(d_dtype_t)) +
                            i * ATOM_M * (kSwizzleCDMode / sizeof(d_dtype_t)) +
                            row * ((kNumBankGroupBytes * 8) / sizeof(d_dtype_t)) +
                            (col ^ row) * (kNumBankGroupBytes / sizeof(d_dtype_t));
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    // Wait shared memory ready
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Write into remote buffers
                    // Each warp writes 2 rows (lane_idx/16 splits the warp into two halves, one per row)
                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;

                        // Skip padding rows beyond the actual token count for this expert
                        if (m_idx_in_block >= valid_m)
                            break;

                        const auto& [dst_rank_idx, dst_token_idx, dst_topk_idx] = cached_src_metadata[j];

                        // Read from shared memory
                        const auto smem_ptr = shared_storage.smem_d.l2[epilogue_wg_idx] +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M * (kSwizzleCDMode / sizeof(d_dtype_t)) +
                            row_in_store * (kSwizzleCDMode / sizeof(d_dtype_t)) +
                            (bank_group_idx ^ row_in_atom) * (kNumBankGroupBytes / sizeof(d_dtype_t));
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

                        // Write into remote
                        const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        const auto dst_ptr = math::advance_ptr<float4>(
                            dst_token.get_base_ptr(),
                            n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
                        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                    }
                }

                }
                const uint64_t trace_dx1 = trace_clock();
                trace_atoms = trace_dx1 - trace_dx0;

                // Ensure the next epilogue safe to use shared memory
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                trace_dx_tail = trace_clock() - trace_dx1;
            }
            if constexpr (kPhaseTrace) {
                const uint64_t trace_end = trace_clock();
                if (trace_leader) {
                    const bool trace_is_dh = task_info.block_phase == sched::BlockPhase::BackwardDh;
                    const bool trace_is_dx = task_info.block_phase == sched::BlockPhase::BackwardDx;
                    const uint32_t phase_idx = trace_is_dh ? 0u :
                        task_info.block_phase == sched::BlockPhase::BackwardRecomputeLow ? 1u :
                        task_info.block_phase == sched::BlockPhase::BackwardRecomputeHigh ? 2u : 3u;
                    const uint64_t busy = trace_end - trace_got, wait = trace_got - trace_task;
                    trace_add(6, wait);
                    trace_add(7 + phase_idx, busy);
                    trace_add(11 + phase_idx, wait);
                    trace_add(15, trace_spin);
                    if (trace_is_dh) trace_add(16, trace_atoms);
                    else if (not trace_is_dx) trace_add(17, trace_atoms);
                    else trace_add(19, trace_atoms);
                    trace_add(18, trace_tail);
                    trace_add(20, trace_dx_tail);
                    trace_add(23, 1);
                    trace_add(25, trace_dscore_partial);
                    trace_add(26, trace_task - trace_prev_end);
                    if (trace_is_dh and task_info.n_cluster_idx == 0) {
                        ++ trace_mblock;
                        if (trace_mblock <= kTraceMBlockSlots) {
                            const uint32_t base = kTraceHeader + (trace_mblock - 1) * kTraceMBlockFields;
                            trace_set(base + 0, pool_block_idx + 1);
                            trace_set(base + 1, trace_task);
                            trace_set(base + 5, trace_spin);
                        }
                    }
                    if (trace_mblock >= 1 and trace_mblock <= kTraceMBlockSlots) {
                        const uint32_t base = kTraceHeader + (trace_mblock - 1) * kTraceMBlockFields;
                        trace_set(base + 2, trace_end);
                        trace_add(base + 3, busy);
                        trace_add(base + 4, wait);
                        if (not trace_is_dh and not trace_is_dx) trace_add(base + 6, trace_atoms);
                        if (trace_is_dx) trace_add(base + 7, trace_atoms);
                    }
                }
                trace_prev_end = trace_end;
            }
        }
        if constexpr (kPhaseTrace) {
            if (trace_leader) {
                trace_set(5, trace_clock());
                trace_add(24, trace_mblock);
                dist_set(41, trace_globaltimer());
            }
        }

        // Deallocate tensor memory
        // NOTES: must be called by the same logical warp ID on both CTAs
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

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

        if constexpr (kDistTrace) {
            if (epilogue_thread_idx == 0) dist_set(42, trace_globaltimer());
        }

        // Notify remote ranks; ordered before the cleanup barrier by the dispatch/epilogue sync below
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
            ptx::st_rel_sys(sym_buffer.map(workspace.get_combine_ready_grid_idx_ptr(sym_buffer.rank_idx), epilogue_thread_idx), peer_grid_idx);

        // Barrier with dispatch warps, so that they can do clean workspace
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Iterate over all token chunks, 1 token 1 topk latency: ~3 us
        const auto grid_idx = ptx::get_grid_idx() + 1;
        const uint64_t trace_combine0 = trace_clock();
        if constexpr (kPhaseTrace) {
            if (trace_leader) trace_set(21, trace_combine0);
            if (trace_leader) dist_set(43, trace_globaltimer());
        }
        [[maybe_unused]] uint64_t dist_ready_wait = 0, dist_first_ready_wait = 0, dist_tokens = 0;
        [[maybe_unused]] bool dist_first_token = true;
        if constexpr (not (kDiagAblate & 32u))
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
            [[maybe_unused]] const uint64_t dist_wait0 = trace_clock();
            comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                             [&]() { printf("DeepGEMM combine peers timeout: rank=%u, token=%u\n", sym_buffer.rank_idx, token_idx); });
            if constexpr (kDistTrace) {
                const uint64_t waited = trace_clock() - dist_wait0;
                dist_ready_wait += waited;
                if (dist_first_token) dist_first_ready_wait = waited;
                dist_first_token = false;
                ++ dist_tokens;
            }

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
        if constexpr (kDistTrace) {
            if (lane_idx == 0) {
                dist_add(44, dist_ready_wait);
                dist_add(47, dist_tokens);
                dist_max(46, dist_first_ready_wait);
            }
            if (trace_leader) dist_set(45, dist_first_ready_wait);
        }
        if constexpr (kPhaseTrace) {
            if (trace_leader) {
                trace_set(22, trace_clock());
                trace_set(2, trace_globaltimer());
                trace_set(3, trace_clock());
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
