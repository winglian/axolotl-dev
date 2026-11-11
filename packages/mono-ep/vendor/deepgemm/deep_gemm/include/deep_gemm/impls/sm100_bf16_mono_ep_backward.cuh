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
sm100_bf16_mono_ep_backward_impl(void* y,
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
                         const __grid_constant__ layout::MoEBackwardBuffers backward) {
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
        if constexpr (not (kOptions & 512u))
            cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
    }

    // 2048: MXFP8 GEMM operands (E4M3 values, UE8M0 scales per 32 K elements; dh, z recompute and dx run as
    // kind::mxf8f6f4). x and dy travel over NVLink as MXFP8; the dispatch warps store the E4M3 pools and their
    // scales and dequantize BF16 copies for the unchanged BF16 weight-gradient GEMMs; dz is quantized in the epilogue.
    constexpr bool kMXFP8 = kOptions & 2048u;

    // Workspaces and Buffer
    const auto buffer = layout::MonoEPBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, 0,
        /*with_sf=*/ false,
        kNumSharedExperts,
        /*fp8_l1_only=*/ kMXFP8
    );
    const auto workspace = buffer.workspace;

    using L2KBlockDependency = sched::L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>;

    // Data types
    using a_dtype_t = cute::conditional_t<kMXFP8, cutlass::float_e4m3_t, cutlass::bfloat16_t>;
    using b_dtype_t = cute::conditional_t<kMXFP8, cutlass::float_e4m3_t, cutlass::bfloat16_t>;
    using d_dtype_t = cutlass::bfloat16_t;

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_BLOCK_K = kMXFP8 ? 128 : 64;
    constexpr uint32_t UMMA_K = kMXFP8 ? 32 : 16;
    DG_STATIC_ASSERT(not kMXFP8 or (BLOCK_K == 128 and BLOCK_M == 128), "MXFP8 backward uses 128 x 128 operand tiles");
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");

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
    // 512: in-kernel weight gradients. Per local expert, 256 x BLOCK_M FP32 tiles of dW1 = dz^T x and dW2 = dy^T h
    // over all of the expert's pool rows (MN-major operands from the row-major pools), scheduled once the
    // expert's h/dz rows are final. Pointers ride in the unused `tensor_map_shared_l2_weights` slot, the h-pool
    // descriptor in `tensor_map_shared_l1_output`.
    constexpr bool kInkernelDw = kOptions & 512u;
    DG_STATIC_ASSERT(not kInkernelDw or (BLOCK_K == 64 and BLOCK_M == 128 and kNumEpilogueWarpgroups == 2),
                     "In-kernel dW expects 64 x 64 pool boxes and two epilogue warpgroups");
    // 1024 (with 512): wide 256 x 256 dW tiles over 32-row stages (one third fewer operand bytes per FLOP); the
    // 64 x 32 pool boxes ride in unused slots (x, dy, h, dz: l1_output, shared_l1_acts, shared_l1_output,
    // shared_l2_acts) and every TMEM accumulator stage is 256 columns wide.
    constexpr bool kDwWide = kInkernelDw and (kOptions & 1024u);
    constexpr uint32_t DW_BLOCK_N = kDwWide ? 2 * BLOCK_M : BLOCK_M;
    constexpr uint32_t DW_BLOCK_K = kDwWide ? 32 : BLOCK_K;
    const auto& dw_params = *reinterpret_cast<const layout::MoEBackwardDwParams*>(&tensor_map_shared_l2_weights);
    const auto is_dw_task = [](const auto& info) {
        return kInkernelDw and (info.block_phase == sched::BlockPhase::BackwardDw1 or
                                info.block_phase == sched::BlockPhase::BackwardDw2);
    };
    // 4096: stored forward context. The forward saved z (BF16 gate/up pre-activations by source row,
    // [ranks, max tokens, topk, 2I]) in `saved_z`; RecomputeLow/High tasks skip their loads and UMMAs and the
    // epilogue reads z from there instead of TMEM. `saved_z` otherwise carries the phase trace.
    constexpr bool kStoredContext = kOptions & 4096u;
    DG_STATIC_ASSERT(not (kStoredContext and (kPhaseTrace or kMXFP8 or kInkernelDw)),
                     "Stored context excludes the phase trace, the MXFP8 backward and in-kernel dW");
    const auto is_stored_recompute = [](const auto& info) {
        return kStoredContext and (info.block_phase == sched::BlockPhase::BackwardRecomputeLow or
                                   info.block_phase == sched::BlockPhase::BackwardRecomputeHigh);
    };
    DG_STATIC_ASSERT(not kMXFP8 or (kStagedPoolStores and not (kOptions & 4u) and not kLookaheadPull),
                     "MXFP8 needs the staged pool stores, two TMEM stages and its own pull loop");
    DG_STATIC_ASSERT(not (kInkernelDw and kMXFP8), "In-kernel dW (BLOCK_K 64) and the MXFP8 backward (BLOCK_K 128) share operand slots");

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
    constexpr uint32_t kTmemStageCols = kDwWide ? 2 * UMMA_N : UMMA_N;
    constexpr uint32_t kNumAccumTmemCols = kTmemStageCols * kNumEpilogueStages;
    // MXFP8: one 128-row UTCCP block of scale factors per operand after the accumulators
    constexpr uint32_t kNumSFTmemCols = kMXFP8 ? 4 : 0;
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFTmemCols;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + 2 * kNumSFTmemCols>();
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
        alignas(kSharedMemoryAlignment) a_dtype_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K];
        alignas(kSharedMemoryAlignment) b_dtype_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
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
    // MXFP8 scale-factor stages (SFA then SFB, 128 packed UE8M0 words each) follow the shared storage
    constexpr uint32_t kNumSFWordsPerStage = 128;
    constexpr uint32_t kSFSmemOffset = math::constexpr_align<uint32_t>(static_cast<uint32_t>(sizeof(SharedStorage)), kSharedMemoryAlignment);
    constexpr uint32_t kNumSmemBytes = kMXFP8 ? kSFSmemOffset + 2 * kNumStages * kNumSFWordsPerStage * sizeof(uint32_t) :
        static_cast<uint32_t>(sizeof(SharedStorage));
    const auto smem_sfa = [&](const uint32_t& stage) {
        return reinterpret_cast<uint32_t*>(smem_buffer + kSFSmemOffset) + stage * kNumSFWordsPerStage;
    };
    const auto smem_sfb = [&](const uint32_t& stage) {
        return reinterpret_cast<uint32_t*>(smem_buffer + kSFSmemOffset) + (kNumStages + stage) * kNumSFWordsPerStage;
    };
    if constexpr (kNumStages > 2) {
        DG_STATIC_ASSERT(kNumSmemBytes <= 232448, "Shared storage exceeds the SM100 opt-in capacity");
        // Deeper opt-in pipelines size shared memory on the host; refuse to run past the allocation.
        uint32_t dynamic_smem_bytes;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dynamic_smem_bytes));
        if (kNumSmemBytes > dynamic_smem_bytes)
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
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
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

    // In-kernel dW k-blocks: the expert's padded pool rows, at least one so that every role consumes the task
    // through the same barriers (a zero-row expert's tile loads one unused box, skips the MMA and stores zeros)
    const auto dw_num_k_blocks = [](const auto& info) -> uint32_t { return cute::max(1u, info.shape_k / DW_BLOCK_K); };

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
    // MXFP8: the dispatch warps also stage and dequantize the pulled rows, so they take registers from the epilogue
    constexpr uint32_t kNumDispatchRegisters = (kUseMoreEpilogueRegisters ? 48 : 96) + (kMXFP8 ? 32 : 0);
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 88;
    constexpr uint32_t kNumEpilogueRegisters = (kUseMoreEpilogueRegisters ? 208 : 160) - (kMXFP8 ? 16 : 0);
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

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        const uint64_t trace_pull_start = trace_clock();
        uint64_t trace_ring_wait = 0, trace_copy = 0, trace_rows = 0;
        // Row-at-a-time pull loops with the next row resolved while this one is in flight
        if constexpr (kMXFP8 or kLookaheadPull) {
        struct PullRow {
            const void* src_x;
            const void* src_dy;
            float weight;
            uint32_t pool_token_idx;
            uint32_t rank;
            uint32_t src_token_idx;
            uint32_t src_topk_idx;
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
            row.is_last_token = token_idx == expert_end_idx - 1;
            row.src_x = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(row.src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            if constexpr (kMXFP8) {
                row.src_dy = backward.dy.map(
                    backward.dy.get_base_ptr<uint8_t*>() + static_cast<uint64_t>(row.src_token_idx) * kHidden,
                    current_rank_in_expert_idx);
            } else {
            row.src_dy = backward.dy.map(
                backward.dy.get_base_ptr<__nv_bfloat16*>() + static_cast<uint64_t>(row.src_token_idx) * kHidden,
                current_rank_in_expert_idx);
            }
            // Consumed only at publish, so the remote load overlaps the row copy
            row.weight = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx, current_rank_in_expert_idx);
            return true;
        };
        if constexpr (kMXFP8) {
            // E4M3 x and dy rows (one NVLink round trip per routed row) plus one packed UE8M0 word per 128 columns;
            // lane l fetches x and dy words l + 32 w. The next row's load is issued as soon as this row's buffer has
            // been read.
            constexpr uint32_t kRowBytes = kHidden;
            constexpr uint32_t kNumSFWords = kHidden / 128;
            constexpr uint32_t kSFWordsPerLane = math::constexpr_ceil_div(kNumSFWords, 32u);
            DG_STATIC_ASSERT(kHidden % 512 == 0, "MXFP8 transport needs hidden % 512 == 0");
            DG_STATIC_ASSERT(kNumBytesPerPull == 2 * kRowBytes, "The MXFP8 pull buffer holds one x and one dy row");
            DG_STATIC_ASSERT(not (kDiagAblate & 64u), "The copy ablation is BF16-only");
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            constexpr uint32_t kSFPoolRows = kNumRanks * kNumMaxTokensPerRank * kNumTopk + kNumExpertsPerRank * 128;
            const auto smem_x = pull_buffer.get_base_ptr();
            const auto smem_dy = math::advance_ptr(smem_x, kRowBytes);
            using sf_words_t = uint32_t[2 * kSFWordsPerLane];
            const auto load_sf = [&](const PullRow& row, sf_words_t& sf_word) {
                const auto* x_sf = sym_buffer.map(
                    buffer.input_sf_buffer.get_data_buffer(row.src_token_idx).template get_base_ptr<uint32_t>(), row.rank);
                const auto* dy_sf = backward.dy_sf.map(
                    backward.dy_sf.template get_base_ptr<uint32_t*>() + static_cast<uint64_t>(row.src_token_idx) * kNumSFWords,
                    row.rank);
                #pragma unroll
                for (uint32_t w = 0; w < kSFWordsPerLane; ++ w) {
                    const uint32_t idx = lane_idx + 32 * w;
                    if (idx < kNumSFWords) {
                        sf_word[w] = x_sf[idx];
                        sf_word[kSFWordsPerLane + w] = dy_sf[idx];
                    }
                }
            };
            const auto issue = [&](const PullRow& row, sf_words_t& sf_word) {
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(smem_x, row.src_x, pull_mbarrier, kRowBytes, kDispatchLoadHint);
                    ptx::tma_load_1d(smem_dy, row.src_dy, pull_mbarrier, kRowBytes, kDispatchLoadHint);
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, 2 * kRowBytes);
                }
                __syncwarp();
                load_sf(row, sf_word);
            };
            uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx;
            PullRow cur, next;
            sf_words_t cur_sf = {}, next_sf = {};
            bool has_cur = resolve(token_idx, cur);
            if (has_cur)
                issue(cur, cur_sf);
            while (has_cur) {
                token_idx += kNumGlobalWarps;
                const bool has_next = resolve(token_idx, next);

                // Ring slot wait for this row only (no other row of this warp is unpublished)
                const uint32_t pool_token_idx = cur.pool_token_idx;
                const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;
                const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
                const uint64_t trace_ring0 = trace_clock();
                if (l1_empty_count_target > 0) {
                    const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                    while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
                }
                const uint64_t trace_copy0 = trace_clock();
                trace_ring_wait += trace_copy0 - trace_ring0;
                ++ trace_rows;

                // Every lane reads the landed rows below
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                if (cute::elect_one_sync()) {
                    ptx::tma_store_1d(backward.qx_pool + static_cast<uint64_t>(pool_token_idx) * kHidden, smem_x, kRowBytes);
                    cute::tma_store_arrive();
                    ptx::tma_store_1d(backward.qdy_pool + static_cast<uint64_t>(pool_token_idx) * kHidden, smem_dy, kRowBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();

                // Scale words into the UTCCP-ordered (MN-major, 4 x 32 transposed) pools the GEMM loaders read; they
                // are written by the generic proxy and read by TMA
                const uint32_t sf_row = (pool_token_idx / 128) * 128 + (pool_token_idx % 32) * 4 + (pool_token_idx % 128) / 32;
                #pragma unroll
                for (uint32_t w = 0; w < kSFWordsPerLane; ++ w) {
                    const uint32_t idx = lane_idx + 32 * w;
                    if (idx < kNumSFWords) {
                        reinterpret_cast<uint32_t*>(backward.sx_pool)[static_cast<uint64_t>(idx) * kSFPoolRows + sf_row] = cur_sf[w];
                        reinterpret_cast<uint32_t*>(backward.sdy_pool)[static_cast<uint64_t>(idx) * kSFPoolRows + sf_row] =
                            cur_sf[kSFWordsPerLane + w];
                    }
                }
                asm volatile("fence.proxy.async.global;" ::: "memory");

                // Per operand: stage this lane's bytes in registers, hand that half of the buffer to the next row's
                // load, then write exact BF16 copies (E4M3 times a power of two) for the BF16 weight-gradient GEMMs
                constexpr uint32_t kNumChunksPerLane = kRowBytes / (16 * 32);
                #pragma unroll 1
                for (uint32_t op = 0; op < 2; ++ op) {
                    const auto src = static_cast<const uint8_t*>(op == 0 ? smem_x : smem_dy);
                    uint4 staged[kNumChunksPerLane];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumChunksPerLane; ++ i)
                        staged[i] = *reinterpret_cast<const uint4*>(src + (i * 32 + lane_idx) * 16);
                    cutlass::arch::fence_view_async_shared();
                    __syncwarp();
                    if (has_next) {
                        if (cute::elect_one_sync()) {
                            if (op == 0) {
                                cute::tma_store_wait<1>();
                                ptx::tma_load_1d(smem_x, next.src_x, pull_mbarrier, kRowBytes, kDispatchLoadHint);
                                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, 2 * kRowBytes);
                            } else {
                                cute::tma_store_wait<0>();
                                ptx::tma_load_1d(smem_dy, next.src_dy, pull_mbarrier, kRowBytes, kDispatchLoadHint);
                            }
                        }
                        __syncwarp();
                    }
                    auto* dst = (op == 0 ? backward.x_pool : backward.dy_pool) + static_cast<uint64_t>(pool_token_idx) * kHidden;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumChunksPerLane; ++ i) {
                        const uint32_t chunk = i * 32 + lane_idx;
                        // Word chunk / 8 = 4 i + lane / 8 sits in register i / 8 of lane (4 i + lane / 8) % 32
                        DG_STATIC_ASSERT(kNumSFWords % 4 == 0, "Scale words of one chunk group share a register");
                        const uint32_t word = __shfl_sync(0xffffffff, op == 0 ? cur_sf[i / 8] : cur_sf[kSFWordsPerLane + i / 8],
                                                          (chunk / 8) % 32);
                        const float scale = __uint_as_float(((word >> (((chunk / 2) % 4) * 8)) & 0xffu) << 23);
                        const auto convert = [&](const uint32_t& packed) {
                            uint32_t out[2];
                            #pragma unroll
                            for (uint32_t j = 0; j < 2; ++ j) {
                                const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(
                                    static_cast<__nv_fp8x2_storage_t>((packed >> (j * 16)) & 0xffffu), __NV_E4M3);
                                const float2 value = __half22float2(*reinterpret_cast<const __half2*>(&raw));
                                const auto bf16 = __float22bfloat162_rn(make_float2(value.x * scale, value.y * scale));
                                out[j] = *reinterpret_cast<const uint32_t*>(&bf16);
                            }
                            return make_uint2(out[0], out[1]);
                        };
                        auto* dst_vec = reinterpret_cast<uint2*>(dst + chunk * 16);
                        dst_vec[0] = convert(staged[i].x);
                        dst_vec[1] = convert(staged[i].y);
                        dst_vec[2] = convert(staged[i].z);
                        dst_vec[3] = convert(staged[i].w);
                    }
                }
                __syncwarp();
                if (has_next)
                    load_sf(next, next_sf);

                // Publish once the E4M3 rows are written (full completion, not only the shared-memory read)
                if (cute::elect_one_sync()) {
                    *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = cur.weight;
                    *workspace.get_token_src_metadata_ptr(pool_token_idx) = {cur.rank, cur.src_token_idx, cur.src_topk_idx};
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(
                        workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks),
                        cur.is_last_token ? BLOCK_M - (pool_token_idx % BLOCK_M) : 1u);
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
                #pragma unroll
                for (uint32_t w = 0; w < 2 * kSFWordsPerLane; ++ w)
                    cur_sf[w] = next_sf[w];
                has_cur = has_next;
            }
        } else {
            constexpr uint32_t kRowBytes = kHidden * sizeof(nv_bfloat16);
            DG_STATIC_ASSERT(kAsyncDispatchCopy and kNumBytesPerPull == 2 * kRowBytes,
                             "The lookahead pull needs whole x and dy rows in one pull buffer");
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const auto smem_x = pull_buffer.get_base_ptr();
            const auto smem_dy = math::advance_ptr(smem_x, kRowBytes);
            const auto issue = [&](const PullRow& row) {
                if (not (kDiagAblate & 64u) and cute::elect_one_sync()) {
                    ptx::tma_load_1d(smem_x, row.src_x, pull_mbarrier, kRowBytes, kDispatchLoadHint);
                    ptx::tma_load_1d(smem_dy, row.src_dy, pull_mbarrier, kRowBytes, kDispatchLoadHint);
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, 2 * kRowBytes);
                }
                __syncwarp();
            };
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

                if (not (kDiagAblate & 64u) and cute::elect_one_sync()) {
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                    ptx::tma_store_1d(backward.x_pool + static_cast<uint64_t>(cur.pool_token_idx) * kHidden, smem_x, kRowBytes);
                    ptx::tma_store_1d(backward.dy_pool + static_cast<uint64_t>(cur.pool_token_idx) * kHidden, smem_dy, kRowBytes);
                    cute::tma_store_arrive();
                    cute::tma_store_wait<0>();
                }
                __syncwarp();
                if (has_next)
                    issue(next);

                if (cute::elect_one_sync()) {
                    *buffer.l1_topk_weights_buffer.get_data_buffer(cur.pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = cur.weight;
                    *workspace.get_token_src_metadata_ptr(cur.pool_token_idx) = {cur.rank, cur.src_token_idx, cur.src_topk_idx};
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
        } else
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
            constexpr uint32_t kHiddenBytes = kHidden * sizeof(nv_bfloat16);
            // A pull buffer of two rows (x and dy together, one NVLink round trip per token) needs the overlapped copy
            DG_STATIC_ASSERT(kNumBytesPerPull <= kHiddenBytes or (kAsyncDispatchCopy and kNumBytesPerPull == 2 * kHiddenBytes),
                             "Pull buffers larger than a row require the overlapped dispatch copy");
            constexpr uint32_t kNumChunks = kNumBytesPerPull <= kHiddenBytes ? kHiddenBytes / kNumBytesPerPull : 1;
            DG_STATIC_ASSERT(kNumBytesPerPull > kHiddenBytes or kHiddenBytes % kNumBytesPerPull == 0, "Invalid hidden");

            // TMA load token from remote rank and store into local
            const auto pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            if (backward.pool_rows != 0 and (pool_token_idx / BLOCK_M + 1) * BLOCK_M > backward.pool_rows)
                asm volatile("trap;");
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;

            // Wait for ring buffer slot to be available (previous consumer must have finished all N blocks)
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            const uint64_t trace_ring0 = trace_clock();
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
            }
            const uint64_t trace_copy0 = trace_clock();
            trace_ring_wait += trace_copy0 - trace_ring0;
            ++ trace_rows;

            const auto src_x = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            const auto src_dy = backward.dy.map(
                backward.dy.get_base_ptr<__nv_bfloat16*>() + static_cast<uint64_t>(src_token_idx) * kHidden,
                current_rank_in_expert_idx);
            const auto weight = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                current_rank_in_expert_idx);
            if constexpr (kAsyncDispatchCopy) {
                if (not (kDiagAblate & 64u) and cute::elect_one_sync()) {
                    // Pull x and dy concurrently in half-buffer pieces; only the last piece waits for store completion
                    constexpr uint32_t kHalfBytes = kNumBytesPerPull / 2;
                    DG_STATIC_ASSERT(kHalfBytes % 16 == 0 and kHiddenBytes % kHalfBytes == 0, "Invalid pull split");
                    auto* dst_x = backward.x_pool + static_cast<uint64_t>(pool_token_idx) * kHidden;
                    auto* dst_dy = backward.dy_pool + static_cast<uint64_t>(pool_token_idx) * kHidden;
                    const auto smem_lo = pull_buffer.get_base_ptr();
                    const auto smem_hi = math::advance_ptr(smem_lo, kHalfBytes);
                    for (uint32_t i = 0; i < kHiddenBytes / kHalfBytes; ++i) {
                        if (i > 0)
                            cute::tma_store_wait<0>();
                        ptx::tma_load_1d(smem_lo, static_cast<const uint8_t*>(src_x) + i * kHalfBytes, pull_mbarrier, kHalfBytes,
                                         kDispatchLoadHint);
                        ptx::tma_load_1d(smem_hi, reinterpret_cast<const uint8_t*>(src_dy) + i * kHalfBytes, pull_mbarrier, kHalfBytes,
                                         kDispatchLoadHint);
                        ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, 2 * kHalfBytes);
                        ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                        ptx::tma_store_1d(math::advance_ptr(dst_x, i * kHalfBytes), smem_lo, kHalfBytes);
                        ptx::tma_store_1d(math::advance_ptr(dst_dy, i * kHalfBytes), smem_hi, kHalfBytes);
                        cute::tma_store_arrive();
                    }
                    ptx::tma_store_wait<0>();
                }
            } else
            if (not (kDiagAblate & 64u) and cute::elect_one_sync()) {
                for (uint32_t operand = 0; operand < 2; ++operand) {
                    const void* src = operand == 0 ? src_x : src_dy;
                    auto* dst = (operand == 0 ? backward.x_pool : backward.dy_pool) +
                        static_cast<uint64_t>(pool_token_idx) * kHidden;
                    for (uint32_t i = 0; i < kNumChunks; ++i) {
                        ptx::tma_load_1d(pull_buffer.get_base_ptr(),
                            static_cast<const uint8_t*>(src) + i * kNumBytesPerPull,
                            pull_mbarrier, kNumBytesPerPull, kDispatchLoadHint);
                        ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                        ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                        ptx::tma_store_1d(math::advance_ptr(dst, i * kNumBytesPerPull),
                            pull_buffer.get_base_ptr(), kNumBytesPerPull);
                        cute::tma_store_arrive();
                        ptx::tma_store_wait<0>();
                    }
                }
            }
            __syncwarp();
            const uint64_t trace_row_copy = trace_clock() - trace_copy0;
            trace_copy += trace_row_copy;
            if constexpr (kDistTrace) {
                if (lane_idx == 0) {
                    const uint64_t arrival = trace_globaltimer();
                    dist_add(8 + current_rank_in_expert_idx, 1);
                    dist_max(16 + current_rank_in_expert_idx, ~arrival);
                    dist_max(24 + current_rank_in_expert_idx, arrival);
                    dist_add(32 + current_rank_in_expert_idx, trace_row_copy);
                }
            }

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
                    // Every task of this rank finished before the epilogue grid sync that precedes this cleanup
                    if constexpr (kInkernelDw)
                        dw_params.ready[i] = dw_params.claims[i] = 0;
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
            if constexpr (kInkernelDw) {
                if (is_dw_task(task_info)) {
                    // dW N side (UMMA B): x (dW1) or h (dW2) columns, MN-major, LOAD_BLOCK_M per CTA
                    const auto desc = task_info.block_phase == sched::BlockPhase::BackwardDw1 ?
                        (kDwWide ? &tensor_map_l1_output : &tensor_map_l1_acts) : &tensor_map_shared_l1_output;
                    const uint32_t n_idx = task_info.n_cluster_idx * DW_BLOCK_N + (is_leader_cta ? 0u : DW_BLOCK_N / 2);
                    const uint32_t row_idx = task_info.pool_block_idx * BLOCK_M;
                    while (ptx::ld_acq(dw_params.ready + task_info.local_expert_idx) < 2 * task_info.shape_k / BLOCK_M);
                    for (uint32_t k_block_idx = 0; k_block_idx < dw_num_k_blocks(task_info); advance_pipeline(k_block_idx)) {
                        shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
                        if (cute::elect_one_sync()) {
                            DG_STATIC_ASSERT(DW_BLOCK_N / 2 * DW_BLOCK_K == LOAD_BLOCK_M * BLOCK_K, "dW N side must fill an A stage");
                            tma::copy<DW_BLOCK_N / 2, DW_BLOCK_K, kSwizzleAMode, a_dtype_t>(
                                desc, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx],
                                n_idx, row_idx + k_block_idx * DW_BLOCK_K, 2);
                            if (is_leader_cta) {
                                shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(shared_storage.smem_a[0]) * 2);
                            } else {
                                shared_storage.full_barriers[stage_idx].arrive(0u);
                            }
                        }
                        __syncwarp();
                    }
                    continue;
                }
            }
            if (is_stored_recompute(task_info)) {
                // No operands: prefetch this task's stored z rows (this CTA's 64 gate and 64 up columns) into L2
                // ahead of its epilogue. The block's source metadata is final once its rows are in the ring.
                const uint32_t pool_block_idx = task_info.pool_block_idx;
                const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
                const auto expected = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(workspace.get_l1_full_count_ptr(ring_block_idx)) != expected);
                const uint32_t n = task_info.get_n_block_idx(is_leader_cta) * 64;
                for (uint32_t row = lane_idx; row < task_info.valid_m; row += 32) {
                    const auto src = *workspace.get_token_src_metadata_ptr(pool_block_idx * BLOCK_M + row);
                    const auto& [rank, token, slot] = src;
                    const auto* z = saved_z + ((static_cast<uint64_t>(rank) * kNumMaxTokensPerRank + token) *
                                               kNumTopk + slot) * (2 * kIntermediateHidden) + n;
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(z));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(z + kIntermediateHidden));
                }
                __syncwarp();
                continue;
            }
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

                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += task_info.get_umma_aligned_valid_m() / 2;

                // TMA copy tokens, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    bool last_a_use = false;
                    if constexpr (kEvictFirstLastA) {
                        const auto phase = task_info.block_phase;
                        last_a_use = phase == sched::BlockPhase::BackwardDx ?
                            task_info.n_cluster_idx == L2_SHAPE_N / (BLOCK_N * 2) - 1 :
                            (phase == sched::BlockPhase::BackwardDh or phase == sched::BlockPhase::BackwardRecomputeHigh) and
                            task_info.n_cluster_idx == kIntermediateHidden / (BLOCK_N * 2) - 1;
                    }
                    if (last_a_use) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t, false,
                                  static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST)>(
                            tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx], k_idx, m_idx, 2);
                    } else {
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx], k_idx, m_idx, 2);
                    }
                    if constexpr (kMXFP8) {
                        // Both CTAs load the whole M-block's scales (the 2-CTA UTCCP reads each CTA's copy)
                        const auto tensor_map_sfa_ptr = task_info.block_phase == sched::BlockPhase::BackwardDh ? &tensor_map_shared_l1_output :
                            task_info.block_phase == sched::BlockPhase::BackwardDx ? &tensor_map_shared_l2_acts : &tensor_map_shared_l1_acts;
                        tma::copy<BLOCK_M, 1, 0>(tensor_map_sfa_ptr, &shared_storage.full_barriers[stage_idx], smem_sfa(stage_idx),
                                                 pool_block_idx * BLOCK_M, k_block_idx * (BLOCK_K / 128), 2);
                    }
                    if (is_leader_cta) {
                        // Multicast
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(
                            sizeof(shared_storage.smem_a[0]) * 2 + (kMXFP8 ? kNumSFWordsPerStage * sizeof(uint32_t) * 2 : 0));
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
            if constexpr (kInkernelDw) {
                if (is_dw_task(task_info)) {
                    // dW M side (UMMA A): dz (dW1) or dy (dW2) columns, MN-major, two 64-column boxes per CTA
                    const auto desc = task_info.block_phase == sched::BlockPhase::BackwardDw1 ?
                        (kDwWide ? &tensor_map_shared_l2_acts : &tensor_map_l2_acts) :
                        (kDwWide ? &tensor_map_shared_l1_acts : &tensor_map_dy);
                    const uint32_t m_idx = task_info.m_block_idx * (BLOCK_N * 2) + (is_leader_cta ? 0u : BLOCK_N);
                    const uint32_t row_idx = task_info.pool_block_idx * BLOCK_M;
                    while (ptx::ld_acq(dw_params.ready + task_info.local_expert_idx) < 2 * task_info.shape_k / BLOCK_M);
                    for (uint32_t k_block_idx = 0; k_block_idx < dw_num_k_blocks(task_info); advance_pipeline(k_block_idx)) {
                        shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
                        if (cute::elect_one_sync()) {
                            tma::copy<LOAD_BLOCK_N, DW_BLOCK_K, kSwizzleBMode, b_dtype_t>(
                                desc, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx],
                                m_idx, row_idx + k_block_idx * DW_BLOCK_K, 2);
                            if (is_leader_cta) {
                                shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(LOAD_BLOCK_N * DW_BLOCK_K * sizeof(b_dtype_t) * 2);
                            } else {
                                shared_storage.full_barriers[stage_idx].arrive(0u);
                            }
                        }
                        __syncwarp();
                    }
                    continue;
                }
            }
            if (is_stored_recompute(task_info))
                continue;
            const auto tensor_map_b_ptr = task_info.block_phase == sched::BlockPhase::BackwardDh ? &tensor_map_dh_weights :
                task_info.block_phase == sched::BlockPhase::BackwardDx ? &tensor_map_l2_weights : &tensor_map_l1_weights;
            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto n_block_idx = task_info.get_n_block_idx(is_leader_cta);
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
                    tma::copy_nd<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, 4, kWeightCacheHint>(
                        tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], 2, k_idx, half_n_idx, half_idx, expert_idx);
                    if constexpr (kMXFP8) {
                        // Weight scales: [expert * K / 128][N] packed words, UTCCP-ordered along N
                        const auto tensor_map_sfb_ptr = task_info.block_phase == sched::BlockPhase::BackwardDh ? &tensor_map_l1_output :
                            task_info.block_phase == sched::BlockPhase::BackwardDx ? &tensor_map_shared_l2_weights : &tensor_map_shared_l1_weights;
                        tma::copy<BLOCK_N, 1, 0>(tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], smem_sfb(stage_idx),
                                                 n_idx, expert_idx * (shape_k / 128) + k_block_idx * (BLOCK_K / 128), 2);
                    }
                    if (is_leader_cta) {
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(
                            sizeof(shared_storage.smem_b[0]) * 2 + (kMXFP8 ? kNumSFWordsPerStage * sizeof(uint32_t) * 2 : 0));
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
            auto instr_desc = [] {
                if constexpr (kMXFP8) {
                    return cute::UMMA::make_instr_desc_block_scaled<b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                                                                    UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
                } else {
                    return cute::UMMA::make_instr_desc<b_dtype_t, a_dtype_t, float, UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
                }
            }();
            [[maybe_unused]] auto sf_desc = [] {
                if constexpr (kMXFP8) return mma::sm100::make_sf_desc(nullptr); else return 0;
            }();

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, UMMA_BLOCK_K, kSwizzleAMode>(shared_storage.smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_BLOCK_K, kSwizzleBMode>(shared_storage.smem_b[0], 0, 0);
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
                if constexpr (kInkernelDw) {
                    if (is_dw_task(task_info)) {
                        const uint64_t trace_dw_task = trace_clock();
                        uint64_t trace_dw_full = 0;
                        const uint32_t num_dw_k_blocks = dw_num_k_blocks(task_info);
                        const bool has_rows = task_info.shape_k > 0;
                        const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                        const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                        shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();
                        const uint64_t trace_dw_empty_done = trace_clock();
                        // Both operands MN-major (K = pool rows), UMMA N = BLOCK_M; built per task to keep the
                        // descriptors out of the main loop's live registers
                        auto dw_a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::MN, DW_BLOCK_N / 2, DW_BLOCK_K, kSwizzleAMode>(
                            shared_storage.smem_a[0], 0, 0);
                        auto dw_b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::MN, LOAD_BLOCK_N, DW_BLOCK_K, kSwizzleBMode>(
                            shared_storage.smem_b[0], 0, 0);
                        const uint32_t dw_a_desc_lo = dw_a_desc.lo;
                        const uint32_t dw_b_desc_lo = dw_b_desc.lo;
                        constexpr auto dw_instr_desc = cute::UMMA::make_instr_desc<b_dtype_t, a_dtype_t, float, UMMA_M, DW_BLOCK_N,
                                                                                   cute::UMMA::Major::MN, cute::UMMA::Major::MN>();
                        for (uint32_t k_block_idx = 0; k_block_idx < num_dw_k_blocks; advance_pipeline(k_block_idx)) {
                            const uint64_t trace_full0 = trace_clock();
                            shared_storage.full_barriers[stage_idx].wait(phase);
                            ptx::tcgen05_after_thread_sync();
                            trace_dw_full += trace_clock() - trace_full0;
                            const uint32_t a_base_lo = dw_a_desc_lo + stage_idx * sizeof(SharedStorage::smem_a[0]) / 16;
                            const uint32_t b_base_lo = dw_b_desc_lo + stage_idx * sizeof(SharedStorage::smem_b[0]) / 16;
                            if (has_rows and cute::elect_one_sync()) {
                                #pragma unroll
                                for (uint32_t k = 0; k < DW_BLOCK_K / UMMA_K; ++ k) {
                                    dw_a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                        cute::UMMA::Major::MN, DW_BLOCK_N / 2, kSwizzleAMode, a_dtype_t>(a_base_lo, 0, k * UMMA_K);
                                    dw_b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                        cute::UMMA::Major::MN, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_base_lo, 0, k * UMMA_K);
                                    ptx::SM100_MMA_F16BF16_2x1SM_SS::fma(
                                        dw_b_desc, dw_a_desc, accum_stage_idx * kTmemStageCols, k_block_idx > 0 or k > 0,
                                        static_cast<uint64_t>(static_cast<uint32_t>(dw_instr_desc)) << 32);
                                }
                            }
                            __syncwarp();
                            constexpr uint16_t kCTAMask = (1 << 2) - 1;
                            cutlass::arch::umma_arrive_multicast_2x1SM(
                                reinterpret_cast<uint64_t*>(&shared_storage.empty_barriers[stage_idx]), kCTAMask);
                            if (k_block_idx == num_dw_k_blocks - 1)
                                cutlass::arch::umma_arrive_multicast_2x1SM(
                                    reinterpret_cast<uint64_t*>(&shared_storage.tmem_full_barriers[accum_stage_idx]), kCTAMask);
                            __syncwarp();
                        }
                        if constexpr (kPhaseTrace) {
                            const uint64_t trace_dw_end = trace_clock();
                            if (lane_idx == 0) {
                                trace_add(30, trace_dw_empty_done - trace_dw_task);
                                trace_add(71, trace_dw_end - trace_dw_empty_done);
                                trace_add(72, trace_dw_full);
                                trace_add(73, has_rows ? num_dw_k_blocks : 0);
                                trace_add(42, trace_dw_task - trace_mma_prev);
                            }
                            trace_mma_prev = trace_dw_end;
                        }
                        continue;
                    }
                }
                if (is_stored_recompute(task_info))
                    continue;
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
                        #pragma unroll
                        for (uint32_t umma_k_block_idx = 0; umma_k_block_idx < BLOCK_K / UMMA_BLOCK_K; ++ umma_k_block_idx) {
                            if constexpr (kMXFP8) {
                                // UTCCP copy SFA and SFB to TMEM
                                using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_sfa(stage_idx));
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA);
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_sfb(stage_idx));
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB);
                            }
                            // Issue UMMA
                            #pragma unroll
                            for (uint32_t k = 0; k < UMMA_BLOCK_K / UMMA_K; ++ k) {
                                a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_M, k * UMMA_K);
                                b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_N, k * UMMA_K);
                                if constexpr (kMXFP8) {
                                    ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                        b_desc, a_desc, accum_stage_idx * UMMA_N,
                                        k_block_idx > 0 or umma_k_block_idx > 0 or k > 0,
                                        mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k),
                                        kTmemStartColOfSFB, kTmemStartColOfSFA);
                                } else {
                                ptx::SM100_MMA_F16BF16_2x1SM_SS::fma(
                                    b_desc, a_desc, accum_stage_idx * kTmemStageCols,
                                    k_block_idx > 0 or umma_k_block_idx > 0 or k > 0,
                                    static_cast<uint64_t>(static_cast<uint32_t>(instr_desc)) << 32);
                                }
                            }
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
            if constexpr (kOwnMBlocks and kInkernelDw)
                scheduler.template mainloop_owned_backward_dw<DW_BLOCK_N>(dw_params.ready, dw_params.claims, dw_params.per_block);
            else if constexpr (kOwnMBlocks)
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
        const bool trace_leader = epilogue_thread_idx == 0;
        uint32_t trace_mblock = 0;
        uint64_t trace_prev_end = trace_clock();
        if constexpr (kPhaseTrace) {
            if (trace_leader) trace_set(4, trace_prev_end);
            if (trace_leader) dist_set(40, trace_globaltimer());
        }
        while (scheduler.get_next_task(task_info)) {
            const uint64_t trace_task = trace_clock();
            // Wait UMMA arrival (stored-context recompute tasks have no UMMA and no accumulator stage)
            const bool stored_z = is_stored_recompute(task_info);
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            if (not stored_z) {
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
                ptx::tcgen05_after_thread_sync();
            }
            const uint64_t trace_got = trace_clock();
            uint64_t trace_spin = 0, trace_atoms = 0, trace_tail = 0, trace_dscore_partial = 0, trace_dx_tail = 0;

            // Now we can release the task
            scheduler.release_task_info();

            if constexpr (kInkernelDw) {
                if (is_dw_task(task_info)) {
                    // TMEM lane = dW row of this CTA's 128-row half, column = dW column; each warpgroup stores 64
                    // columns, each thread one row segment, straight from registers
                    const bool is_dw1 = task_info.block_phase == sched::BlockPhase::BackwardDw1;
                    const uint32_t num_cols = is_dw1 ? kHidden : kIntermediateHidden;
                    const uint32_t num_rows = is_dw1 ? 2 * kIntermediateHidden : kHidden;
                    constexpr uint32_t kWGCols = DW_BLOCK_N / kNumEpilogueWarpgroups;
                    const uint32_t row = task_info.m_block_idx * (BLOCK_N * 2) + (is_leader_cta ? 0u : BLOCK_N) +
                                         warp_idx_in_wg * 32 + lane_idx;
                    const uint32_t col = task_info.n_cluster_idx * DW_BLOCK_N + epilogue_wg_idx * kWGCols;
                    float* dst = (is_dw1 ? dw_params.dw1 : dw_params.dw2) +
                        (static_cast<uint64_t>(task_info.local_expert_idx) * num_rows + row) * num_cols + col;
                    const bool has_rows = task_info.shape_k > 0;
                    constexpr uint32_t kChunk = 16;
                    #pragma unroll 1
                    for (uint32_t chunk = 0; chunk < kWGCols / kChunk; ++ chunk) {
                        uint32_t v[kChunk];
                        if (has_rows) {
                            cute::SM100_TMEM_LOAD_32dp32b16x::copy(
                                accum_stage_idx * kTmemStageCols + epilogue_wg_idx * kWGCols + chunk * kChunk,
                                v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
                            cutlass::arch::fence_view_async_tmem_load();
                        } else {
                            #pragma unroll
                            for (uint32_t j = 0; j < kChunk; ++ j)
                                v[j] = 0;
                        }
                        if (chunk == kWGCols / kChunk - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }
                        #pragma unroll
                        for (uint32_t j = 0; j < kChunk / 4; ++ j) {
                            *reinterpret_cast<float4*>(dst + chunk * kChunk + j * 4) = make_float4(
                                __uint_as_float(v[j * 4]), __uint_as_float(v[j * 4 + 1]),
                                __uint_as_float(v[j * 4 + 2]), __uint_as_float(v[j * 4 + 3]));
                        }
                    }
                    if constexpr (kPhaseTrace) {
                        const uint64_t trace_end = trace_clock();
                        if (trace_leader) {
                            trace_add(68, 1);
                            trace_add(69, trace_end - trace_got);
                            trace_add(70, trace_got - trace_task);
                            trace_add(26, trace_task - trace_prev_end);
                            trace_set(75, trace_end);
                        }
                        trace_prev_end = trace_end;
                    }
                    continue;
                }
            }

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
                        if constexpr (not (kDiagAblate & 1u))
                            shared_storage.backward_epilogue.retain_dh_atom(values, atom_row, warp_idx_in_wg, lane_idx, valid_m);
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
                            result = shared_storage.backward_epilogue.recompute_atom(
                                values, atom_row, warp_idx_in_wg, lane_idx, half, scores);
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
                                    if constexpr (kMXFP8) {
                                        if (o > 0) {
                                            // E4M3 dz for the dX GEMM: one UE8M0 scale per 32 columns (4 adjacent lanes)
                                            const auto bf16 = reinterpret_cast<const nv_bfloat162*>(&value);
                                            float2 f[4];
                                            float amax = 0.0f;
                                            #pragma unroll
                                            for (uint32_t j = 0; j < 4; ++ j) {
                                                f[j] = __bfloat1622float2(bf16[j]);
                                                amax = fmaxf(amax, fmaxf(fabsf(f[j].x), fabsf(f[j].y)));
                                            }
                                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
                                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
                                            const uint32_t sf_exp = math::get_ue8m0_sf_exp(amax);
                                            const float sf_inv = math::get_ue8m0_sf_inv<float>(sf_exp);
                                            uint32_t q[2];
                                            #pragma unroll
                                            for (uint32_t j = 0; j < 2; ++ j) {
                                                const uint32_t lo = __nv_cvt_float2_to_fp8x2(
                                                    make_float2(f[2 * j].x * sf_inv, f[2 * j].y * sf_inv), __NV_SATFINITE, __NV_E4M3);
                                                const uint32_t hi = __nv_cvt_float2_to_fp8x2(
                                                    make_float2(f[2 * j + 1].x * sf_inv, f[2 * j + 1].y * sf_inv), __NV_SATFINITE, __NV_E4M3);
                                                q[j] = lo | (hi << 16);
                                            }
                                            const uint32_t col = (o - 1) * kIntermediateHidden + n;
                                            *reinterpret_cast<uint2*>(backward.qdz_pool + packed * (2 * kIntermediateHidden) + col) = make_uint2(q[0], q[1]);
                                            if (chunk % 4 == 0) {
                                                constexpr uint32_t kSFPoolRows = kNumRanks * kNumMaxTokensPerRank * kNumTopk + kNumExpertsPerRank * 128;
                                                const auto p = static_cast<uint32_t>(packed);
                                                const uint32_t sf_row = (p / 128) * 128 + (p % 32) * 4 + (p % 128) / 32;
                                                const uint32_t group = col / 32;
                                                reinterpret_cast<uint8_t*>(backward.sdz_pool)[
                                                    (static_cast<uint64_t>(group / 4) * kSFPoolRows + sf_row) * 4 + group % 4] = static_cast<uint8_t>(sf_exp);
                                            }
                                        }
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
                if (stored_z) {
                    // The atom's z in the TMEM register layout: lane rows (lane % 4) * 2 + {0, 1}, columns
                    // warp * 16 + lane / 4 + {0, 8} of this CTA's 64 intermediate columns; gate at
                    // values[col * 4 + row], up at values[col * 4 + 2 + row]
                    const auto load_z = [&](const uint32_t& atom, uint32_t (&values)[8]) {
                        const auto atom_row = epilogue_wg_idx * WG_BLOCK_M + atom * ATOM_M;
                        const auto n = n_block_idx * 64 + warp_idx_in_wg * 16 + lane_idx / 4;
                        #pragma unroll
                        for (uint32_t r = 0; r < 2; ++r) {
                            const auto row = atom_row + (lane_idx % 4) * 2 + r;
                            nv_bfloat16 g[2] = {}, u[2] = {};
                            if (row < valid_m) {
                                const auto src = *workspace.get_token_src_metadata_ptr(pool_m_idx + row);
                                const auto& [rank, token, slot] = src;
                                const auto* z = saved_z + ((static_cast<uint64_t>(rank) * kNumMaxTokensPerRank + token) *
                                                           kNumTopk + slot) * (2 * kIntermediateHidden);
                                #pragma unroll
                                for (uint32_t col = 0; col < 2; ++col) {
                                    g[col] = z[n + col * 8];
                                    u[col] = z[kIntermediateHidden + n + col * 8];
                                }
                            }
                            #pragma unroll
                            for (uint32_t col = 0; col < 2; ++col) {
                                values[col * 4 + r] = __float_as_uint(__bfloat162float(g[col]));
                                values[col * 4 + 2 + r] = __float_as_uint(__bfloat162float(u[col]));
                            }
                        }
                    };
                    uint32_t atom_values[kNumAtoms][8];
                    #pragma unroll
                    for (uint32_t atom = 0; atom < kNumAtoms; ++atom)
                        load_z(atom, atom_values[atom]);
                    #pragma unroll
                    for (uint32_t atom = 0; atom < kNumAtoms; ++atom)
                        process_atom(atom, atom_values[atom]);
                } else if constexpr (kEarlyTmemRelease) {
                    // Drain the whole accumulator into registers so the MMA warp can reuse the stage during the math
                    uint32_t atom_values[kNumAtoms][8];
                    #pragma unroll
                    for (uint32_t atom = 0; atom < kNumAtoms; ++atom) {
                        const auto tmem_addr = accum_stage_idx * kTmemStageCols + epilogue_wg_idx * WG_BLOCK_M + atom * ATOM_M;
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
                        const auto tmem_addr = accum_stage_idx * kTmemStageCols + epilogue_wg_idx * WG_BLOCK_M + atom * ATOM_M;
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
                    if (not stored_z) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                    }
                }
                [[maybe_unused]] const bool is_last_recompute = task_info.block_phase == sched::BlockPhase::BackwardRecomputeHigh and
                    task_info.n_cluster_idx + 1 == kIntermediateHidden / (2 * BLOCK_N);
                // The weight-gradient GEMMs reduce over each expert's padded pool rows: zero the x/dy tail of the
                // expert's last block (stale, possibly non-finite rows from earlier launches) before its h/dz are
                // published, so 0 * stale never reaches dW; CTA c clears column half c
                if (task_info.block_phase == sched::BlockPhase::BackwardRecomputeHigh and
                    task_info.n_cluster_idx + 1 == kIntermediateHidden / (2 * BLOCK_N) and valid_m < BLOCK_M) {
                    constexpr uint32_t kHalfChunks = kHidden / 2 / 8;
                    const uint32_t col_offset = is_leader_cta ? 0u : kHidden / 2;
                    for (uint32_t i = epilogue_thread_idx; i < (BLOCK_M - valid_m) * kHalfChunks; i += kNumEpilogueThreads) {
                        const auto offset = static_cast<uint64_t>(pool_m_idx + valid_m + i / kHalfChunks) * kHidden +
                                            col_offset + (i % kHalfChunks) * 8;
                        *reinterpret_cast<float4*>(backward.x_pool + offset) = make_float4(0, 0, 0, 0);
                        *reinterpret_cast<float4*>(backward.dy_pool + offset) = make_float4(0, 0, 0, 0);
                    }
                }
                __threadfence();
                asm volatile("fence.proxy.async.global;" ::: "memory");
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (not is_dh) {
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        L2KBlockDependency::arrive(workspace.get_l2_full_mask_ptr(ring_block_idx), n_block_idx);
                        ptx::red_add(workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                        // This CTA's h/dz rows of the block are final (every recompute epilogue fenced before its sync)
                        if constexpr (kInkernelDw) {
                            if (is_last_recompute)
                                ptx::red_add_rel(dw_params.ready + task_info.local_expert_idx, 1u);
                        }
                    }
                    if (task_info.block_phase == sched::BlockPhase::BackwardRecomputeHigh and
                        task_info.n_cluster_idx + 1 == kIntermediateHidden / (2 * BLOCK_N)) {
                        trace_dscore_partial = trace_clock();
                        for (uint32_t row = epilogue_thread_idx; row < valid_m; row += kNumEpilogueThreads) {
                            const auto src = *workspace.get_token_src_metadata_ptr(pool_m_idx + row);
                            const auto& [rank, token, slot] = src;
                            auto* dst = backward.dscore_partials.map(backward.dscore_partials.get_base_ptr<float*>(), rank);
                            dst[(static_cast<uint64_t>(token) * kNumTopk + slot) * 2 + (is_leader_cta ? 0u : 1u)] = shared_storage.dscore[row];
                        }
                        __threadfence_system();
                        trace_dscore_partial = trace_clock() - trace_dscore_partial;
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
                        uint32_t tmem_addr = accum_stage_idx * kTmemStageCols + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

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
                    if constexpr (kInkernelDw)
                        trace_set(74, trace_end);
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
