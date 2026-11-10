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
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mono_ep.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/impls/mono_ep_activations.cuh>

namespace deep_gemm {

#ifndef DG_MonoEP_RELAXED_SPIN_HELPERS
#define DG_MonoEP_RELAXED_SPIN_HELPERS
// Relaxed polls compile to LDG.STRONG without the per-iteration CCTL.IVALL of ld.acquire.
__device__ __forceinline__ uint64_t mono_ep_ld_relaxed_gpu_u64(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr) : "memory");
    return ret;
}

__device__ __forceinline__ uint64_t mono_ep_ld_relaxed_sys_u64(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr) : "memory");
    return ret;
}

__device__ __forceinline__ bool mono_ep_mbarrier_test(const void* barrier, const uint32_t parity) {
    uint32_t ready;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(ready) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(barrier))), "r"(parity) : "memory");
    return ready != 0;
}
#endif

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
    uint32_t kActivationClampBits,
    bool kFastMath,
    bool kCompactNonMath = false,
    bool kOriginalW1Layout = false,
    bool kPhaseTrace = false,
    uint32_t kDiagAblate = 0,
    bool kVectorScatter = false,
    bool kDeferCombineFence = false,
    bool kVectorSwiGLU = false,
    bool kVectorHQuant = false,
    bool kSwizzleAccumSmem = false,
    bool kFP8OnlyStages = false,
    uint32_t kMainloopMode = 0,
    bool kRelaxedSpins = false,
    bool kDistTrace = false,
    uint32_t kActivation = 0,
    uint32_t kActivationAlphaBits = mono_ep_act::kDefaultAlphaBits,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
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
sm90_fp8_mono_ep_forward_impl(void* y,
                         int* cumulative_local_expert_recv_stats,
                         const uint32_t num_tokens,
                         const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l2_cached_acts,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                         const uint8_t* sm_locality_domains, __nv_bfloat16* saved_z,
                         const float* w1_scales, const float* w2_scales) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(not kHasShared, "FP8 persistent shared experts are not implemented");
    const float kActivationClamp = __uint_as_float(kActivationClampBits);
    const float kActivationAlpha = __uint_as_float(kActivationAlphaBits);
    // kFastMath selects the ex2.approx + rcp.approx sigmoid of the SwiGLU default.
    constexpr auto kGatedActivation = kFastMath and kActivation == 0
        ? mono_ep_act::Activation::SwiGLUFast : static_cast<mono_ep_act::Activation>(kActivation);
    DG_STATIC_ASSERT(kActivation < mono_ep_act::kNumActivations, "Invalid activation");
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
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_cached_acts);
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
        /*fp8_l1_only=*/ true
    );
    const auto workspace = buffer.workspace;

    using L2KBlockDependency = sched::L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>;

    // Data types
    using a_dtype_t = cutlass::bfloat16_t;
    using b_dtype_t = __nv_fp8_e4m3;
    using d_dtype_t = cutlass::bfloat16_t;

    // MMA configs
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M == 128, "SM90 persistent BF16 requires block M 128");
    DG_STATIC_ASSERT(BLOCK_N == 128, "SM90 persistent BF16 requires block N 128");
    DG_STATIC_ASSERT(BLOCK_K == 128, "SM90 persistent FP8 requires block K 128");
    DG_STATIC_ASSERT(not kFP8OnlyStages or not kHasShared, "FP8-only A stages require routed-only experts");
    DG_STATIC_ASSERT(kNumEpilogueThreads == 256, "SM90 persistent BF16 requires 256 math threads");

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
    constexpr uint32_t kScaleRowBytes = math::constexpr_align(kHidden / 32, 16u);
    constexpr uint32_t kPullBufferBytes = kNumBytesPerPull > kScaleRowBytes ? kNumBytesPerPull : kScaleRowBytes;

    // Assign shared memory
    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kPullBufferBytes];
        union {
            alignas(kSharedMemoryAlignment) d_dtype_t l1[kNumEpilogueWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) d_dtype_t l2[kNumEpilogueWarpgroups][STORE_BLOCK_M * BLOCK_N];
            alignas(kSharedMemoryAlignment) d_dtype_t raw[BLOCK_M * BLOCK_N];
        } smem_d;
        // FP8-only stages drop the BF16 A width, which only shared-expert phases load.
        union AStage {
            a_dtype_t bf16[kFP8OnlyStages ? LOAD_BLOCK_M * BLOCK_K / 2 : LOAD_BLOCK_M * BLOCK_K];
            __nv_fp8_e4m3 fp8[LOAD_BLOCK_M * BLOCK_K];
        };
        alignas(kSharedMemoryAlignment) AStage smem_a[kNumStages];
        alignas(kSharedMemoryAlignment) b_dtype_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K];
        alignas(kSharedMemoryAlignment) __nv_fp8_e4m3 quantized_a[BLOCK_M * BLOCK_K];
        float cached_activation_scales[kNumStages][BLOCK_M];
        float activation_scales[BLOCK_M];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
    };
    DG_STATIC_ASSERT(sizeof(SharedStorage) <= 232448, "SM90 persistent BF16 exceeds Hopper shared memory capacity");
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);

    // Send buffers
    constexpr auto pull_layout = layout::Data(kPullBufferBytes);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    __syncthreads();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
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

    // Diagnostic per-CTA phase trace (kPhaseTrace only); the pointer rides in the unused recv-stats argument.
    constexpr uint32_t kTraceHeader = 64, kTraceTaskSlots = 64, kTraceTaskFields = 16;
    constexpr uint32_t kTraceCTAStride = kTraceHeader + kTraceTaskSlots * kTraceTaskFields;
    int* const recv_stats = kPhaseTrace ? nullptr : cumulative_local_expert_recv_stats;
    unsigned long long* const phase_trace = kPhaseTrace
        ? reinterpret_cast<unsigned long long*>(cumulative_local_expert_recv_stats) +
          static_cast<uint64_t>(sm_idx) * kTraceCTAStride : nullptr;
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
    // Diagnostic multi-rank extension (kDistTrace only): 64 slots per CTA after all CTA records,
    // globaltimer stamps (comparable across ranks of one node) and per-peer pull statistics.
    DG_STATIC_ASSERT(not kDistTrace or (kPhaseTrace and kNumRanks <= 8), "Distributed trace needs the phase trace and <= 8 ranks");
    constexpr uint32_t kDistTraceStride = 64;
    unsigned long long* const dist_trace = kDistTrace
        ? reinterpret_cast<unsigned long long*>(cumulative_local_expert_recv_stats) +
          static_cast<uint64_t>(kNumSMs) * kTraceCTAStride + static_cast<uint64_t>(sm_idx) * kDistTraceStride : nullptr;
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
        DG_STATIC_ASSERT(kUseMoreEpilogueRegisters, "Compact non-math WG requires at most 64 experts per rank");
        DG_STATIC_ASSERT(88 * 128 + kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                         "Too many compact-WG registers");
    } else {
        DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                         kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                         kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                         "Too many registers");
    }

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

#ifndef DG_MonoEP_FORWARD_ROLE_REGISTERS
#define DG_MonoEP_FORWARD_ROLE_REGISTERS 0
#endif
    // ptxas drops every setmaxnreg (C7507) when a CFG path leads from a dealloc to the
    // math role; the role-scoped layout places each directive inside its warpgroup's branch.
    DG_STATIC_ASSERT(not DG_MonoEP_FORWARD_ROLE_REGISTERS or not kCompactNonMath,
                     "Role-scoped registers require the 512-thread layout");

    // Register reconfiguration must execute convergently at one instruction per warpgroup.
#if !DG_MonoEP_FORWARD_ROLE_REGISTERS
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
#endif

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
#if DG_MonoEP_FORWARD_ROLE_REGISTERS
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
#endif
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
        if constexpr (kDistTrace) {
            if (thread_idx == 0) dist_set(0, trace_globaltimer());
        }

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
            if (thread_idx == 0) dist_set(1, trace_globaltimer());
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
            if (thread_idx == 0) dist_set(2, trace_globaltimer());
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
        if constexpr (kDistTrace) {
            if (thread_idx == 0) dist_set(3, trace_globaltimer());
        }

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        const uint64_t trace_pull_start = trace_clock();
        uint64_t trace_ring_wait = 0, trace_pull_wait = 0, trace_meta_wait = 0, trace_rows = 0;
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
            constexpr uint32_t kHiddenBytes = kHidden * sizeof(__nv_fp8_e4m3);
            constexpr uint32_t kScaleBytes = kScaleRowBytes;
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
                if constexpr (kRelaxedSpins) {
                    // Relaxed polls with nanosleep backoff; the counter is monotonic within a launch, so
                    // one acquire re-read orders the ring writes after the consumer's release.
                    uint32_t backoff_ns = 128;
                    while (ptx::ld_acq(empty_ptr) < l1_empty_count_target) {
                        while (ptx::ld_relaxed_gpu(empty_ptr) < l1_empty_count_target) {
                            __nanosleep(backoff_ns);
                            backoff_ns = cute::min(backoff_ns * 2, 4096u);
                        }
                    }
                } else {
                    while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
                }
            }
            const uint64_t trace_ring1 = trace_clock();
            trace_ring_wait += trace_ring1 - trace_ring0;
            ++ trace_rows;

            const auto src_base_ptr = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            const auto dst_base_ptr = buffer.l1_token_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).get_base_ptr();
            const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                ptx::tma_store_1d(
                    math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
                    pull_buffer.get_base_ptr(), kNumBytesPerPull
                );
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            };
            // Load the weight first, so that its remote latency overlaps with the token pull below
            const auto weight = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                current_rank_in_expert_idx);
            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumChunks; ++ i) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
                        pull_mbarrier, kNumBytesPerPull
                    );
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                    i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                }
            }
            __syncwarp();
            const uint64_t trace_pull1 = trace_clock();
            trace_pull_wait += trace_pull1 - trace_ring1;
            if constexpr (kDistTrace) {
                if (lane_idx == 0) {
                    const uint64_t arrival = trace_globaltimer();
                    dist_add(16 + current_rank_in_expert_idx, trace_pull1 - trace_ring1);
                    dist_add(24 + current_rank_in_expert_idx, 1);
                    dist_max(32 + current_rank_in_expert_idx, ~arrival);
                    dist_max(40 + current_rank_in_expert_idx, arrival);
                }
            }

            // Store weights and metadata
            if (cute::elect_one_sync()) {
                *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = weight;

                // Write source metadata for combine write-back (logical pool token)
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Wait for token TMA store to complete
                issue_and_wait_pull_store(kNumChunks - 1);

                const auto src_sf_ptr = sym_buffer.map(
                    buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
                const auto dst_sf_ptr = buffer.l1_sf_buffer.get_data_buffer(
                    pool_token_idx % kNumRingTokens).get_base_ptr();
                ptx::tma_load_1d(pull_buffer.get_base_ptr(), src_sf_ptr, pull_mbarrier, kScaleBytes);
                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kScaleBytes);
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                ptx::tma_store_1d(dst_sf_ptr, pull_buffer.get_base_ptr(), kScaleBytes);
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
                const bool is_last_token = (token_idx == expert_end_idx - 1);
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks), 
                    is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u
                );
            }
            __syncwarp();
            trace_meta_wait += trace_clock() - trace_pull1;
        }

        if constexpr (kPhaseTrace) {
            if (lane_idx == 0 and warp_idx < 4) {
                const uint64_t trace_pull_end = trace_clock();
                trace_set(40 + warp_idx * 4, trace_pull_end - trace_pull_start);
                trace_set(41 + warp_idx * 4, trace_ring_wait);
                trace_set(42 + warp_idx * 4, trace_pull_wait);
                trace_set(43 + warp_idx * 4, trace_meta_wait);
                trace_set(56 + warp_idx, trace_pull_end);
                if (warp_idx == 0) trace_set(2, trace_pull_start);
            }
            if (lane_idx == 0) dist_max(4, trace_globaltimer());
            if (lane_idx == 0) trace_add(60, trace_rows);
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
                        if (recv_stats != nullptr)
                            ptx::red_add(recv_stats + i, static_cast<int>(num_recv_tokens));
                    }
                    __syncwarp();
                } else {
                    DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                    if (warp_idx == 0) {
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                    } else if (warp_idx == 1) {
                        if (cute::elect_one_sync() and recv_stats != nullptr)
                            ptx::red_add(recv_stats + i, static_cast<int>(num_recv_tokens));
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

        if constexpr (kDistTrace) {
            if (thread_idx == 0) dist_set(6, trace_globaltimer());
        }
        // Wait for all ranks to finish cleaning
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );
        if constexpr (kDistTrace) {
            if (thread_idx == 0) dist_set(5, trace_globaltimer());
        }
        if constexpr (kPhaseTrace) {
            if (thread_idx == 0) trace_set(62, trace_clock());
        }
#if DG_MonoEP_FORWARD_ROLE_REGISTERS
    } else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        if (warp_idx == kNumDispatchWarps) {
#else
    } else if (warp_idx == kNumDispatchWarps) {
#endif
        // GEMM TMA load warp for tokens
        task_info_t task_info;
        uint64_t trace_l1_spin = 0, trace_l2_dep = 0, trace_a_empty = 0;
        while (scheduler.get_next_task(task_info)) {
            scheduler.release_task_info();
            const auto tensor_map_a_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_cached_acts :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts;
            const auto num_k_blocks = math::ceil_div(task_info.shape_k, BLOCK_K);

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;

            // Wait the token arrival
            const uint64_t trace_spin0 = trace_clock();
            if (task_info.block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_full_count_ptr(block_idx);
                const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                if constexpr (kRelaxedSpins) {
                    while (ptx::ld_acq(ptr) != num_expected_tokens)
                        while (ptx::ld_relaxed_gpu(ptr) != num_expected_tokens);
                } else {
                    while (ptx::ld_acq(ptr) != num_expected_tokens);
                }
                trace_l1_spin += trace_clock() - trace_spin0;
            } else if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                const auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = SHARED_L2_SHAPE_K / (BLOCK_N / 2);
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            }

            L2KBlockDependency l2_k_block_dependency(workspace.get_l2_full_mask_ptr(ring_block_idx), pool_block_idx / kNumRingBlocks);

            // Issue TMA
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                const uint64_t trace_dep0 = trace_clock();
                if (task_info.block_phase == sched::BlockPhase::Linear2) {
                    if constexpr (kRelaxedSpins) {
                        // L2KBlockDependency::wait with relaxed re-reads; the K block's bits flip once per
                        // generation, so the single acquire re-read (read-read coherence) sees them fed too.
                        const auto k_block_mask = ((1ull << L2KBlockDependency::kNumL1BlockNsPerL2KBlock) - 1)
                            << (k_block_idx * L2KBlockDependency::kNumL1BlockNsPerL2KBlock);
                        if (l2_k_block_dependency.pending_mask & k_block_mask) {
                            while ((mono_ep_ld_relaxed_gpu_u64(l2_k_block_dependency.mask_ptr) ^
                                    l2_k_block_dependency.expected_mask) & k_block_mask);
                            l2_k_block_dependency.pending_mask =
                                ptx::ld_acq_gpu(l2_k_block_dependency.mask_ptr) ^ l2_k_block_dependency.expected_mask;
                        }
                    } else {
                        l2_k_block_dependency.wait(k_block_idx);
                    }
                }
                const uint64_t trace_empty0 = trace_clock();
                trace_l2_dep += trace_empty0 - trace_dep0;

                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
                trace_a_empty += trace_clock() - trace_empty0;

                // Compute token offset from ring block index
                uint32_t m_idx = block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;

                if (task_info.block_phase == sched::BlockPhase::Linear1 or
                    task_info.block_phase == sched::BlockPhase::Linear2) {
                    const bool is_l1 = task_info.block_phase == sched::BlockPhase::Linear1;
                    const uint32_t scale_row_stride = is_l1 ?
                        math::constexpr_align(kHidden / 32, 16u) / sizeof(float) :
                        math::constexpr_align(kIntermediateHidden / 32, 16u) / sizeof(float);
                    const auto sf = is_l1 ? buffer.l1_sf_buffer.get_base_ptr<float>() :
                                           buffer.l2_cached_sf_buffer.get_base_ptr<float>();
                    for (uint32_t row = lane_idx; row < BLOCK_M; row += 32)
                        shared_storage.cached_activation_scales[stage_idx][row] =
                            sf[(m_idx + row) * scale_row_stride + k_block_idx];
                }
                __syncwarp();

                // TMA copy tokens, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    if (task_info.block_phase == sched::BlockPhase::Linear1 or
                        task_info.block_phase == sched::BlockPhase::Linear2) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, __nv_fp8_e4m3>(
                            tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx],
                            shared_storage.smem_a[stage_idx].fp8, k_idx, m_idx);
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(
                            sizeof(shared_storage.smem_a[0].fp8));
                    } else {
                        #pragma unroll
                        for (uint32_t half = 0; half < 2; ++half)
                            tma::copy<64, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx],
                                shared_storage.smem_a[stage_idx].bf16 + half * BLOCK_M * 64,
                                k_idx + half * 64, m_idx);
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(
                            sizeof(shared_storage.smem_a[0].bf16));
                    }
                }
                __syncwarp();
            }
        }
        if constexpr (kPhaseTrace) {
            if (lane_idx == 0) {
                trace_set(30, trace_l1_spin);
                trace_set(31, trace_l2_dep);
                trace_set(32, trace_a_empty);
                trace_set(33, trace_clock());
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // GEMM TMA load warp for weights
        task_info_t task_info;
        uint64_t trace_b_empty = 0;
        while (scheduler.get_next_task(task_info)) {
            scheduler.release_task_info();
            const auto tensor_map_b_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights;
            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto n_block_idx = task_info.n_cluster_idx;
            const auto num_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            const uint32_t half_n = shape_n / 2, n_idx = n_block_idx * BLOCK_N, half_idx = n_idx >= half_n;
            const uint32_t half_n_idx = n_idx - half_idx * half_n, expert_idx = task_info.is_shared() ? 0 : task_info.local_expert_idx;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                const uint64_t trace_empty0 = trace_clock();
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
                trace_b_empty += trace_clock() - trace_empty0;

                // Compute weight offset
                uint32_t k_idx = k_block_idx * BLOCK_K;

                // TMA copy weights
                if (cute::elect_one_sync()) {
                    if constexpr (kOriginalW1Layout) {
                        if (task_info.block_phase == sched::BlockPhase::Linear1) {
                            const uint32_t pair_n_idx = n_block_idx * 64;
                            #pragma unroll
                            for (uint32_t half = 0; half < 2; ++half)
                                tma::copy_nd<BLOCK_K, 64, kSwizzleBMode, b_dtype_t, 4>(
                                    tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx],
                                    shared_storage.smem_b[stage_idx] + half * 64 * BLOCK_K,
                                    1, k_idx, pair_n_idx, half, expert_idx);
                        } else {
                            tma::copy_nd<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, 4>(
                                tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx],
                                shared_storage.smem_b[stage_idx], 1, k_idx,
                                half_n_idx, half_idx, expert_idx);
                        }
                    } else {
                        tma::copy_nd<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, 4>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx],
                            shared_storage.smem_b[stage_idx], 1, k_idx,
                            half_n_idx, half_idx, expert_idx);
                    }
                    shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(shared_storage.smem_b[0]));
                }
                __syncwarp();
            }
        }
        if constexpr (kPhaseTrace) {
            if (lane_idx == 0) trace_set(34, trace_b_empty);
        }
    } else if (not kCompactNonMath and warp_idx == kNumDispatchWarps + 2) {
    } else if (warp_idx == kNumDispatchWarps + (kCompactNonMath ? 2 : 3)) {
        scheduler.mainloop(num_tokens, sm_locality_domains);
#if DG_MonoEP_FORWARD_ROLE_REGISTERS
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
#else
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
#endif
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
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks
        task_info_t task_info;
        const bool trace_leader = epilogue_thread_idx == 0;
        uint32_t trace_task = 0;
        uint64_t trace_prev_end = trace_clock();
        if constexpr (kPhaseTrace) {
            if (trace_leader) trace_set(3, trace_prev_end);
        }
        [[maybe_unused]] bool dist_first_task = true;
        while (scheduler.get_next_task(task_info)) {
            // Copy task metadata before releasing the scheduler slot.
            scheduler.release_task_info();
            const uint64_t trace_got = trace_clock();
            uint64_t trace_full_first = 0, trace_full_rest = 0;
            if constexpr (kDistTrace) {
                if (trace_leader and dist_first_task) dist_set(48, trace_globaltimer());
                dist_first_task = false;
            }

            float accum[WGMMA::kNumAccum] = {0};
            float partial[WGMMA::kNumAccum];
            auto a_desc = mma::sm90::make_gmma_desc<
                cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode>(
                    shared_storage.quantized_a, 0, 0);
            auto cached_a_desc = mma::sm90::make_gmma_desc<
                cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode>(
                    shared_storage.smem_a[0].fp8, 0, 0);
            auto b_desc = mma::sm90::make_gmma_desc<
                cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode>(
                    shared_storage.smem_b[0], 0, 0);
            const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
            const uint32_t cached_a_desc_lo = __shfl_sync(0xffffffff, cached_a_desc.reg32_[0], 0);
            const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);
            const uint32_t weight_n = task_info.n_cluster_idx;
            const float* weight_scales = task_info.block_phase == sched::BlockPhase::Linear1 ? w1_scales : w2_scales;
            const uint32_t num_k_blocks = task_info.shape_k / BLOCK_K;
            const uint32_t num_n_blocks = task_info.shape_n / BLOCK_N;
            // Weight scales of one task are contiguous over K: one load per lane, broadcast per K-block.
            const float* task_weight_scales =
                weight_scales + (task_info.local_expert_idx * num_n_blocks + weight_n) * num_k_blocks;
            DG_STATIC_ASSERT(not (kMainloopMode & 2u) or (L1_SHAPE_K / BLOCK_K <= 32 and L2_SHAPE_K / BLOCK_K <= 32),
                             "Weight-scale prefetch holds one K-block per lane");
            const float lane_weight_scale = (kMainloopMode & 2u) and lane_idx < num_k_blocks ?
                __ldg(task_weight_scales + lane_idx) : 0.0f;
            const auto weight_scale = [&](const uint32_t& k_block_idx) {
                if constexpr ((kMainloopMode & 2u) != 0)
                    return __shfl_sync(0xffffffff, lane_weight_scale, k_block_idx);
                else
                    return task_weight_scales[k_block_idx];
            };
            const bool routed = ((kMainloopMode & 1u) and not kHasShared) or
                task_info.block_phase == sched::BlockPhase::Linear1 or
                task_info.block_phase == sched::BlockPhase::Linear2;
            if constexpr ((kMainloopMode & 4u) != 0) {
                // Two partial fragments: K-block k+1 is issued before K-block k is promoted.
                DG_STATIC_ASSERT(not kHasShared and (kMainloopMode & 1u),
                                 "Ping-pong mainloop supports routed experts only");
                DG_STATIC_ASSERT(L1_SHAPE_K % (2 * BLOCK_K) == 0 and L2_SHAPE_K % (2 * BLOCK_K) == 0,
                                 "Ping-pong mainloop needs an even number of K-blocks");
                float partial_odd[WGMMA::kNumAccum];
                const uint32_t row0 = epilogue_wg_idx * WG_BLOCK_M + warp_idx_in_wg * 16 + lane_idx / 4;
                const auto wait_full = [&](const uint32_t& k_block_idx) {
                    const uint64_t trace_fw0 = trace_clock();
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    if constexpr (kPhaseTrace) {
                        const uint64_t trace_fw = trace_clock() - trace_fw0;
                        if (k_block_idx == 0) trace_full_first = trace_fw; else trace_full_rest += trace_fw;
                    }
                };
                const auto issue = [&](float (&frag)[WGMMA::kNumAccum]) {
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                        ptx::warpgroup_fence_operand(frag[i]);
                    ptx::warpgroup_arrive();
                    const uint32_t a_base = cached_a_desc_lo + stage_idx * sizeof(shared_storage.smem_a[0]) / 16;
                    const uint32_t b_base = b_desc_lo + stage_idx * sizeof(shared_storage.smem_b[0]) / 16;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
                        a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<
                            cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode, __nv_fp8_e4m3>(
                                a_base, epilogue_wg_idx * WG_BLOCK_M, k * WGMMA::K, 0);
                        b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<
                            cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode, __nv_fp8_e4m3>(
                                b_base, 0, k * WGMMA::K, 0);
                        WGMMA::wgmma(a_desc, b_desc, frag, k != 0);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                        ptx::warpgroup_fence_operand(frag[i]);
                };
                const auto promote = [&](float (&frag)[WGMMA::kNumAccum], const float& scale0, const float& scale1) {
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                        ptx::warpgroup_fence_operand(frag[i]);
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                        accum[i] += frag[i] * (i % 4 < 2 ? scale0 : scale1);
                };
                // Scales of the in-flight K-block are read before its stage can be released.
                float cur_scale0, cur_scale1, next_scale0, next_scale1;
                const auto read_scales = [&](const uint32_t& k_block_idx, float& scale0, float& scale1) {
                    const float sb = weight_scale(k_block_idx);
                    scale0 = shared_storage.cached_activation_scales[stage_idx][row0] * sb;
                    scale1 = shared_storage.cached_activation_scales[stage_idx][row0 + 8] * sb;
                };
                uint32_t k_block_idx = 0, prev_stage_idx = stage_idx;
                wait_full(k_block_idx);
                read_scales(k_block_idx, cur_scale0, cur_scale1);
                issue(partial);
                advance_pipeline(k_block_idx);
                while (true) {
                    wait_full(k_block_idx);
                    read_scales(k_block_idx, next_scale0, next_scale1);
                    issue(partial_odd);
                    ptx::warpgroup_wait<1>();
                    promote(partial, cur_scale0, cur_scale1);
                    if (lane_idx == 0)
                        shared_storage.empty_barriers[prev_stage_idx].arrive();
                    prev_stage_idx = stage_idx;
                    cur_scale0 = next_scale0, cur_scale1 = next_scale1;
                    advance_pipeline(k_block_idx);
                    if (k_block_idx == num_k_blocks) {
                        ptx::warpgroup_wait<0>();
                        promote(partial_odd, cur_scale0, cur_scale1);
                        if (lane_idx == 0)
                            shared_storage.empty_barriers[prev_stage_idx].arrive();
                        break;
                    }
                    wait_full(k_block_idx);
                    read_scales(k_block_idx, next_scale0, next_scale1);
                    issue(partial);
                    ptx::warpgroup_wait<1>();
                    promote(partial_odd, cur_scale0, cur_scale1);
                    if (lane_idx == 0)
                        shared_storage.empty_barriers[prev_stage_idx].arrive();
                    prev_stage_idx = stage_idx;
                    cur_scale0 = next_scale0, cur_scale1 = next_scale1;
                    advance_pipeline(k_block_idx);
                }
            }
            if constexpr ((kMainloopMode & 8u) != 0) {
                // N-split pipeline: each K-block runs as two 64-column WGMMA groups, and one half is
                // promoted while the other half (or the next K-block's first half) is in flight.
                DG_STATIC_ASSERT(not kHasShared and (kMainloopMode & 1u) and not (kMainloopMode & 4u),
                                 "N-split mainloop supports routed experts only");
                using HalfWGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N / 2>::type;
                constexpr uint32_t kHalfAccum = HalfWGMMA::kNumAccum;
                DG_STATIC_ASSERT(2 * kHalfAccum == WGMMA::kNumAccum, "Invalid N-split fragment");
                float half_lo[kHalfAccum], half_hi[kHalfAccum];
                const uint32_t row0 = epilogue_wg_idx * WG_BLOCK_M + warp_idx_in_wg * 16 + lane_idx / 4;
                const auto wait_full = [&](const uint32_t& k_block_idx) {
                    const uint64_t trace_fw0 = trace_clock();
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    if constexpr (kPhaseTrace) {
                        const uint64_t trace_fw = trace_clock() - trace_fw0;
                        if (k_block_idx == 0) trace_full_first = trace_fw; else trace_full_rest += trace_fw;
                    }
                };
                const auto issue = [&](float (&frag)[kHalfAccum], const uint32_t& half) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kHalfAccum; ++i)
                        ptx::warpgroup_fence_operand(frag[i]);
                    ptx::warpgroup_arrive();
                    const uint32_t a_base = cached_a_desc_lo + stage_idx * sizeof(shared_storage.smem_a[0]) / 16;
                    const uint32_t b_base = b_desc_lo + stage_idx * sizeof(shared_storage.smem_b[0]) / 16;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / HalfWGMMA::K; ++k) {
                        a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<
                            cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode, __nv_fp8_e4m3>(
                                a_base, epilogue_wg_idx * WG_BLOCK_M, k * HalfWGMMA::K, 0);
                        b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<
                            cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode, __nv_fp8_e4m3>(
                                b_base, half * (BLOCK_N / 2), k * HalfWGMMA::K, 0);
                        HalfWGMMA::wgmma(a_desc, b_desc, frag, k != 0);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kHalfAccum; ++i)
                        ptx::warpgroup_fence_operand(frag[i]);
                };
                const auto promote = [&](float (&frag)[kHalfAccum], const uint32_t& half,
                                         const float& scale0, const float& scale1) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kHalfAccum; ++i)
                        ptx::warpgroup_fence_operand(frag[i]);
                    #pragma unroll
                    for (uint32_t i = 0; i < kHalfAccum; ++i)
                        accum[half * kHalfAccum + i] += frag[i] * (i % 4 < 2 ? scale0 : scale1);
                };
                float scale0, scale1;
                const auto read_scales = [&](const uint32_t& k_block_idx) {
                    const float sb = weight_scale(k_block_idx);
                    scale0 = shared_storage.cached_activation_scales[stage_idx][row0] * sb;
                    scale1 = shared_storage.cached_activation_scales[stage_idx][row0 + 8] * sb;
                };
                uint32_t k_block_idx = 0;
                wait_full(k_block_idx);
                read_scales(k_block_idx);
                issue(half_lo, 0);
                issue(half_hi, 1);
                while (true) {
                    ptx::warpgroup_wait<1>();
                    promote(half_lo, 0, scale0, scale1);
                    const float hi_scale0 = scale0, hi_scale1 = scale1;
                    const uint32_t done_stage_idx = stage_idx;
                    advance_pipeline(k_block_idx);
                    const bool has_next = k_block_idx < num_k_blocks;
                    if (has_next) {
                        wait_full(k_block_idx);
                        read_scales(k_block_idx);
                        issue(half_lo, 0);
                        ptx::warpgroup_wait<1>();
                    } else {
                        ptx::warpgroup_wait<0>();
                    }
                    promote(half_hi, 1, hi_scale0, hi_scale1);
                    if (lane_idx == 0)
                        shared_storage.empty_barriers[done_stage_idx].arrive();
                    if (not has_next)
                        break;
                    issue(half_hi, 1);
                }
            }
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks and not (kMainloopMode & 12u); advance_pipeline(k_block_idx)) {
                const uint64_t trace_fw0 = trace_clock();
                shared_storage.full_barriers[stage_idx].wait(phase);
                if constexpr (kPhaseTrace) {
                    const uint64_t trace_fw = trace_clock() - trace_fw0;
                    if (k_block_idx == 0) trace_full_first = trace_fw; else trace_full_rest += trace_fw;
                }
                if (not routed) {
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
                            reinterpret_cast<const uint8_t*>(shared_storage.smem_a[stage_idx].bf16) + physical));
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
                }
                // Only the in-kernel quantized A (shared phases) is written by these threads.
                if (not (kMainloopMode & 1u) or not routed) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    ptx::sync_aligned(128, 3 + epilogue_wg_idx);
                }
                const uint32_t row0 = epilogue_wg_idx * WG_BLOCK_M + warp_idx_in_wg * 16 + lane_idx / 4;
                const float sb = weight_scale(k_block_idx);
                const float* activation_scales = routed ?
                    shared_storage.cached_activation_scales[stage_idx] : shared_storage.activation_scales;
                const float scale0 = activation_scales[row0] * sb;
                const float scale1 = activation_scales[row0 + 8] * sb;
                const uint32_t b_base = b_desc_lo + stage_idx * sizeof(shared_storage.smem_b[0]) / 16;
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i) {
                    partial[i] = 0.0f;
                    ptx::warpgroup_fence_operand(partial[i]);
                }
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
                    const uint32_t current_a_desc_lo =
                        (task_info.block_phase == sched::BlockPhase::Linear1 or
                         task_info.block_phase == sched::BlockPhase::Linear2) ?
                        cached_a_desc_lo + stage_idx * sizeof(shared_storage.smem_a[0]) / 16 : a_desc_lo;
                    a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<
                        cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode, __nv_fp8_e4m3>(
                            current_a_desc_lo, epilogue_wg_idx * WG_BLOCK_M, k * WGMMA::K, 0);
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
            uint64_t trace_mark[5] = {};
            constexpr uint32_t kWGMMAStoreRowsPerWarp = WGMMA::M / 4;
            // 16-byte column chunks of a 256-byte row are XOR-swizzled by row % 8 so the 8 rows of a store are bank-disjoint.
            DG_STATIC_ASSERT(not kSwizzleAccumSmem or (kVectorSwiGLU and kVectorScatter),
                             "Swizzled accumulator staging requires the vector SwiGLU and scatter readers");
            const auto smem_d_chunk = [&](const uint32_t& row, const uint32_t& col) {
                return shared_storage.smem_d.raw + row * BLOCK_N +
                    (kSwizzleAccumSmem ? (((col / 8) ^ (row % 8)) * 8 + col % 8) : col);
            };
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4 and kSwizzleAccumSmem and not (kDiagAblate & 8u); ++ i) {
                const uint32_t row = epilogue_wg_idx * WG_BLOCK_M + warp_idx_in_wg * kWGMMAStoreRowsPerWarp + lane_idx / 4;
                const uint32_t col = (lane_idx % 4) * 2 + i * 8;
                *reinterpret_cast<__nv_bfloat162*>(smem_d_chunk(row, col)) =
                    __floats2bfloat162_rn(accum[i * 4 + 0], accum[i * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(smem_d_chunk(row + 8, col)) =
                    __floats2bfloat162_rn(accum[i * 4 + 2], accum[i * 4 + 3]);
            }
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4 and not kSwizzleAccumSmem and not (kDiagAblate & 8u); ++ i) {
                const uint32_t row = epilogue_wg_idx * WG_BLOCK_M + warp_idx_in_wg * kWGMMAStoreRowsPerWarp + lane_idx / 4;
                const uint32_t col = (lane_idx % 4) * 2 + i * 8;
                shared_storage.smem_d.raw[row * BLOCK_N + col] = d_dtype_t(accum[i * 4 + 0]);
                shared_storage.smem_d.raw[row * BLOCK_N + col + 1] = d_dtype_t(accum[i * 4 + 1]);
                shared_storage.smem_d.raw[(row + 8) * BLOCK_N + col] = d_dtype_t(accum[i * 4 + 2]);
                shared_storage.smem_d.raw[(row + 8) * BLOCK_N + col + 1] = d_dtype_t(accum[i * 4 + 3]);
            }
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            trace_mark[0] = trace_clock();

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring-buffer offset for reusable data buffers
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;  // Full-pool offset for non-ring metadata
            const uint32_t n_block_idx = task_info.n_cluster_idx;
            uint32_t n_idx = n_block_idx * BLOCK_N;

            if (task_info.block_phase == sched::BlockPhase::Linear1 or
                task_info.block_phase == sched::BlockPhase::SharedLinear1) {
                if (not task_info.is_shared()) {
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const auto expected = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    if constexpr (kRelaxedSpins) {
                        while (ptx::ld_acq(l2_empty_ptr) != expected)
                            while (ptx::ld_relaxed_gpu(l2_empty_ptr) != expected) {}
                    } else {
                        while (ptx::ld_acq(l2_empty_ptr) != expected) {}
                    }
                }
                trace_mark[1] = trace_clock();
                const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                auto output = task_info.is_shared() ? buffer.shared_l2_token_buffer : buffer.l2_token_buffer;
                if constexpr (kVectorSwiGLU) {
                    // Eight consecutive outputs per thread: 16-byte gate/up reads and 16-byte stores.
                    constexpr uint32_t kVec = sizeof(uint4) / sizeof(d_dtype_t);
                    constexpr uint32_t kVecsPerRow = L1_OUT_BLOCK_N / kVec;
                    DG_STATIC_ASSERT(L1_OUT_BLOCK_N % kVec == 0 and kVec == 8, "Invalid vector SwiGLU");
                    for (uint32_t linear = epilogue_thread_idx; linear < valid_m * kVecsPerRow and not (kDiagAblate & 1u);
                         linear += kNumEpilogueThreads) {
                        const uint32_t row = linear / kVecsPerRow;
                        const uint32_t p0 = (linear % kVecsPerRow) * kVec;
                        const uint32_t gate_col = kOriginalW1Layout ? p0 : 2 * p0;
                        const uint32_t up_col = kOriginalW1Layout ? p0 + 64 : gate_col + 8;
                        const auto gate_vec = *reinterpret_cast<const uint4*>(smem_d_chunk(row, gate_col));
                        const auto up_vec = *reinterpret_cast<const uint4*>(smem_d_chunk(row, up_col));
                        const auto gates = reinterpret_cast<const d_dtype_t*>(&gate_vec);
                        const auto ups = reinterpret_cast<const d_dtype_t*>(&up_vec);
                        const float score = task_info.is_shared() ? 1.0f :
                            *buffer.l1_topk_weights_buffer.get_data_buffer(ring_m_idx + row).template get_base_ptr<float>();
                        uint4 out_vec, z_gate_vec, z_up_vec;
                        auto outs = reinterpret_cast<d_dtype_t*>(&out_vec);
                        auto z_gates = reinterpret_cast<__nv_bfloat16*>(&z_gate_vec);
                        auto z_ups = reinterpret_cast<__nv_bfloat16*>(&z_up_vec);
                        #pragma unroll
                        for (uint32_t j = 0; j < kVec; ++ j) {
                            const float gate = static_cast<float>(gates[j]);
                            const float up = static_cast<float>(ups[j]);
                            const float clamped_gate = fminf(gate, kActivationClamp);
                            const float clamped_up = fmaxf(fminf(up, kActivationClamp), -kActivationClamp);
                            outs[j] = d_dtype_t(mono_ep_act::Gated<kGatedActivation>::forward(
                                clamped_gate, clamped_up, kActivationAlpha, kActivationClamp) * score);
                            z_gates[j] = __float2bfloat16(clamped_gate);
                            z_ups[j] = __float2bfloat16(clamped_up);
                        }
                        *reinterpret_cast<uint4*>(output.get_data_buffer(m_idx + row).template get_base_ptr<d_dtype_t>() +
                                                  out_n_idx + p0) = out_vec;
                        if (saved_z != nullptr and not task_info.is_shared()) {
                            const auto src = *workspace.get_token_src_metadata_ptr(pool_m_idx + row);
                            const uint64_t offset = ((static_cast<uint64_t>(src.rank_idx) * kNumMaxTokensPerRank + src.token_idx) *
                                kNumTopk + src.topk_idx) * (2 * kIntermediateHidden);
                            *reinterpret_cast<uint4*>(saved_z + offset + out_n_idx + p0) = z_gate_vec;
                            *reinterpret_cast<uint4*>(saved_z + offset + kIntermediateHidden + out_n_idx + p0) = z_up_vec;
                        }
                    }
                }
                for (uint32_t linear = epilogue_thread_idx;
                     linear < valid_m * L1_OUT_BLOCK_N and not (kDiagAblate & 1u) and not kVectorSwiGLU;
                     linear += kNumEpilogueThreads) {
                    const uint32_t row = linear / L1_OUT_BLOCK_N;
                    const uint32_t p = linear % L1_OUT_BLOCK_N;
                    const uint32_t gate_col = kOriginalW1Layout ? p : 16 * (p / 8) + p % 8;
                    const uint32_t up_col = kOriginalW1Layout ? p + 64 : gate_col + 8;
                    const float gate = static_cast<float>(shared_storage.smem_d.raw[row * BLOCK_N + gate_col]);
                    const float up = static_cast<float>(shared_storage.smem_d.raw[row * BLOCK_N + up_col]);
                    const float clamped_gate = fminf(gate, kActivationClamp);
                    const float clamped_up = fmaxf(fminf(up, kActivationClamp), -kActivationClamp);
                    const float score = task_info.is_shared() ? 1.0f :
                        *buffer.l1_topk_weights_buffer.get_data_buffer(ring_m_idx + row).template get_base_ptr<float>();
                    output.get_data_buffer(m_idx + row).template get_base_ptr<d_dtype_t>()[out_n_idx + p] =
                        d_dtype_t(mono_ep_act::Gated<kGatedActivation>::forward(
                            clamped_gate, clamped_up, kActivationAlpha, kActivationClamp) * score);
                    if (saved_z != nullptr and not task_info.is_shared()) {
                        const auto src = *workspace.get_token_src_metadata_ptr(pool_m_idx + row);
                        const uint64_t offset = ((static_cast<uint64_t>(src.rank_idx) * kNumMaxTokensPerRank + src.token_idx) *
                            kNumTopk + src.topk_idx) * (2 * kIntermediateHidden);
                        saved_z[offset + out_n_idx + p] = __float2bfloat16(clamped_gate);
                        saved_z[offset + kIntermediateHidden + out_n_idx + p] = __float2bfloat16(clamped_up);
                    }
                }
                __threadfence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                trace_mark[2] = trace_clock();
                trace_mark[3] = trace_mark[4] = trace_mark[2];
                if (not task_info.is_shared() and (n_block_idx & 1u)) {
                    const auto mask_ptr = workspace.get_l2_full_mask_ptr(ring_block_idx);
                    const uint64_t even_bit = 1ull << (n_block_idx - 1);
                    const uint64_t expected_even = (pool_block_idx / kNumRingBlocks) & 1u ? 0ull : even_bit;
                    if constexpr (kRelaxedSpins) {
                        while ((ptx::ld_acq_gpu(mask_ptr) & even_bit) != expected_even)
                            while ((mono_ep_ld_relaxed_gpu_u64(mask_ptr) & even_bit) != expected_even) {}
                    } else {
                        while ((ptx::ld_acq_gpu(mask_ptr) & even_bit) != expected_even) {}
                    }
                    trace_mark[3] = trace_clock();

                    constexpr uint32_t kScaleRowStride =
                        math::constexpr_align(kIntermediateHidden / 32, 16u) / sizeof(float);
                    const uint32_t cache_k_block = n_block_idx / 2;
                    const uint32_t cache_col = cache_k_block * BLOCK_K;
                    const uint32_t math_warp_idx = epilogue_thread_idx / 32;
                    if constexpr (kVectorHQuant and not (kDiagAblate & 2u)) {
                        // A half-warp quantizes one 128-column row: all row loads are issued before any store.
                        constexpr uint32_t kVec = sizeof(uint4) / sizeof(d_dtype_t);
                        constexpr uint32_t kLanesPerRow = BLOCK_K / kVec;
                        constexpr uint32_t kRowsPerPass = kNumEpilogueThreads / kLanesPerRow;
                        constexpr uint32_t kNumPasses = BLOCK_M / kRowsPerPass;
                        DG_STATIC_ASSERT(kLanesPerRow == 16 and BLOCK_M % kRowsPerPass == 0, "Invalid vector h quantization");
                        const uint32_t lane_in_row = epilogue_thread_idx % kLanesPerRow;
                        constexpr uint32_t kBatch = 4;
                        DG_STATIC_ASSERT(kNumPasses % kBatch == 0, "Invalid h quantization batch");
                        #pragma unroll
                        for (uint32_t batch = 0; batch < kNumPasses; batch += kBatch) {
                        uint4 packed[kBatch];
                        #pragma unroll
                        for (uint32_t j = 0; j < kBatch; ++ j) {
                            const uint32_t row = (batch + j) * kRowsPerPass + epilogue_thread_idx / kLanesPerRow;
                            packed[j] = row < valid_m ? *reinterpret_cast<const uint4*>(
                                buffer.l2_token_buffer.get_data_buffer(ring_m_idx + row).
                                    template get_base_ptr<d_dtype_t>() + cache_col + lane_in_row * kVec) :
                                make_uint4(0, 0, 0, 0);
                        }
                        #pragma unroll
                        for (uint32_t j = 0; j < kBatch; ++ j) {
                            const uint32_t row = (batch + j) * kRowsPerPass + epilogue_thread_idx / kLanesPerRow;
                            const auto elems = reinterpret_cast<const d_dtype_t*>(&packed[j]);
                            float values[kVec];
                            float amax = 1.0e-4f;
                            #pragma unroll
                            for (uint32_t i = 0; i < kVec; ++i) {
                                values[i] = static_cast<float>(elems[i]);
                                amax = fmaxf(amax, fabsf(values[i]));
                            }
                            #pragma unroll
                            for (uint32_t delta = kLanesPerRow / 2; delta > 0; delta /= 2)
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
                            const uint32_t bits = __float_as_uint(amax / 448.0f);
                            const uint32_t exponent = ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
                            const float scale = __uint_as_float(exponent << 23);
                            const float scale_inv = exponent == 255u ? 0.0f :
                                __uint_as_float((254u - exponent) << 23);
                            if (lane_in_row == 0)
                                buffer.l2_cached_sf_buffer.get_base_ptr<float>()[
                                    (ring_m_idx + row) * kScaleRowStride + cache_k_block] = scale;
                            uint2 quantized;
                            auto bytes = reinterpret_cast<__nv_fp8_e4m3*>(&quantized);
                            #pragma unroll
                            for (uint32_t i = 0; i < kVec; ++i)
                                bytes[i] = __nv_fp8_e4m3(values[i] * scale_inv);
                            *reinterpret_cast<uint2*>(buffer.l2_cached_token_buffer.get_data_buffer(ring_m_idx + row).
                                template get_base_ptr<__nv_fp8_e4m3>() + cache_col + lane_in_row * kVec) = quantized;
                        }
                        }
                    }
                    for (uint32_t row = math_warp_idx; row < BLOCK_M and not (kDiagAblate & 2u) and not kVectorHQuant;
                         row += kNumEpilogueWarps) {
                        float values[4];
                        float amax = 1.0e-4f;
                        #pragma unroll
                        for (uint32_t i = 0; i < 4; ++i) {
                            const uint32_t col = lane_idx + i * 32;
                            values[i] = row < valid_m ? static_cast<float>(
                                buffer.l2_token_buffer.get_data_buffer(ring_m_idx + row).
                                    template get_base_ptr<d_dtype_t>()[cache_col + col]) : 0.0f;
                            amax = fmaxf(amax, fabsf(values[i]));
                        }
                        #pragma unroll
                        for (uint32_t delta = 16; delta > 0; delta /= 2)
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
                        const uint32_t bits = __float_as_uint(amax / 448.0f);
                        const uint32_t exponent = ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
                        const float scale = __uint_as_float(exponent << 23);
                        const float scale_inv = exponent == 255u ? 0.0f :
                            __uint_as_float((254u - exponent) << 23);
                        if (lane_idx == 0)
                            buffer.l2_cached_sf_buffer.get_base_ptr<float>()[
                                (ring_m_idx + row) * kScaleRowStride + cache_k_block] = scale;
                        #pragma unroll
                        for (uint32_t i = 0; i < 4; ++i) {
                            const uint32_t col = lane_idx + i * 32;
                            buffer.l2_cached_token_buffer.get_data_buffer(ring_m_idx + row).
                                template get_base_ptr<__nv_fp8_e4m3>()[cache_col + col] =
                                __nv_fp8_e4m3(values[i] * scale_inv);
                        }
                    }
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    __threadfence();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    trace_mark[4] = trace_clock();
                }
                if (epilogue_thread_idx == 0) {
                    if (task_info.is_shared())
                        ptx::red_add_rel(workspace.get_shared_l2_full_count_ptr(pool_block_idx), 1u);
                    else {
                        L2KBlockDependency::arrive(workspace.get_l2_full_mask_ptr(ring_block_idx), n_block_idx);
                        ptx::red_add(workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                }
            } else if constexpr (kVectorScatter) {
                // A half-warp moves one BF16 row as 16-byte vectors; metadata is read once per row.
                constexpr uint32_t kChunksPerRow = BLOCK_N * sizeof(d_dtype_t) / sizeof(uint4);
                constexpr uint32_t kRowsPerPass = kNumEpilogueThreads / kChunksPerRow;
                constexpr uint32_t kNumElemsPerChunk = sizeof(uint4) / sizeof(d_dtype_t);
                DG_STATIC_ASSERT(32 % kChunksPerRow == 0 and BLOCK_M % kRowsPerPass == 0, "Invalid vector scatter");
                const uint32_t chunk = epilogue_thread_idx % kChunksPerRow;
                #pragma unroll
                for (uint32_t pass = 0; pass < BLOCK_M / kRowsPerPass and not (kDiagAblate & 36u); ++ pass) {
                    const uint32_t row = pass * kRowsPerPass + epilogue_thread_idx / kChunksPerRow;
                    if (row < valid_m) {
                        const auto src = task_info.is_shared() ?
                            layout::TokenSrcMetadata(sym_buffer.rank_idx, pool_m_idx + row, kNumTopk) :
                            *workspace.get_token_src_metadata_ptr(pool_m_idx + row);
                        auto dst = buffer.combine_token_buffer.get_rank_buffer(src.topk_idx).get_data_buffer(src.token_idx);
                        const auto values = *reinterpret_cast<const uint4*>(smem_d_chunk(row, chunk * kNumElemsPerChunk));
                        *reinterpret_cast<uint4*>(sym_buffer.map(
                            dst.template get_base_ptr<d_dtype_t>() + n_idx + chunk * kNumElemsPerChunk, src.rank_idx)) = values;
                    }
                }
                if constexpr (not (kDiagAblate & 4u) and not kDeferCombineFence)
                    __threadfence_system();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_thread_idx == 0 and not task_info.is_shared())
                    ptx::red_add(workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
            } else {
                for (uint32_t linear = epilogue_thread_idx; linear < valid_m * BLOCK_N and not (kDiagAblate & 36u);
                     linear += kNumEpilogueThreads) {
                    const uint32_t row = linear / BLOCK_N;
                    const uint32_t col = linear % BLOCK_N;
                    const auto src = task_info.is_shared() ?
                        layout::TokenSrcMetadata(sym_buffer.rank_idx, pool_m_idx + row, kNumTopk) :
                        *workspace.get_token_src_metadata_ptr(pool_m_idx + row);
                    auto dst = buffer.combine_token_buffer.get_rank_buffer(src.topk_idx).get_data_buffer(src.token_idx);
                    sym_buffer.map(dst.template get_base_ptr<d_dtype_t>() + n_idx + col, src.rank_idx)[0] =
                        shared_storage.smem_d.raw[row * BLOCK_N + col];
                }
                if constexpr (not (kDiagAblate & 4u) and not kDeferCombineFence)
                    __threadfence_system();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_thread_idx == 0 and not task_info.is_shared())
                    ptx::red_add(workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
            }
            if constexpr (kPhaseTrace) {
                const uint64_t trace_end = trace_clock();
                const bool is_l1 = task_info.block_phase == sched::BlockPhase::Linear1;
                if (epilogue_thread_idx == 128) {
                    trace_add(28, trace_ml_end - trace_got);
                    trace_add(29, trace_full_first + trace_full_rest);
                }
                if (trace_leader) {
                    const uint64_t mainloop = trace_ml_end - trace_got;
                    const uint64_t epilogue = trace_end - trace_ml_end;
                    const uint64_t sched_wait = trace_got - trace_prev_end;
                    trace_add(6, sched_wait);
                    trace_add(7, trace_full_first);
                    trace_add(8, trace_full_rest);
                    trace_add(is_l1 ? 9 : 10, mainloop);
                    trace_add(is_l1 ? 13 : 14, epilogue);
                    trace_add(is_l1 ? 15 : 16, 1);
                    trace_add(17, 1);
                    trace_add(20, trace_mark[0] - trace_ml_end);
                    if (is_l1) {
                        trace_add(21, trace_mark[1] - trace_mark[0]);
                        trace_add(22, trace_mark[2] - trace_mark[1]);
                        trace_add(23, trace_mark[3] - trace_mark[2]);
                        trace_add(24, trace_mark[4] - trace_mark[3]);
                        trace_add(25, trace_end - trace_mark[4]);
                        trace_add(27, n_block_idx & 1u);
                        trace_add(11, trace_full_first);
                    } else {
                        trace_add(26, trace_end - trace_mark[0]);
                        trace_add(12, trace_full_first);
                    }
                    if (trace_task < kTraceTaskSlots) {
                        const uint32_t base = kTraceHeader + trace_task * kTraceTaskFields;
                        trace_set(base + 0, 1 + (is_l1 ? 0 : 1) + (n_block_idx << 8));
                        trace_set(base + 1, trace_got);
                        trace_set(base + 2, trace_end);
                        trace_set(base + 3, pool_block_idx);
                        trace_set(base + 4, mainloop);
                        trace_set(base + 5, epilogue);
                        trace_set(base + 6, sched_wait);
                        trace_set(base + 7, trace_full_first + trace_full_rest);
                        trace_set(base + 8, trace_mark[0] - trace_ml_end);
                        trace_set(base + 9, is_l1 ? trace_mark[2] - trace_mark[1] : trace_end - trace_mark[0]);
                        trace_set(base + 10, is_l1 ? trace_mark[3] - trace_mark[2] : 0);
                        trace_set(base + 11, is_l1 ? trace_mark[4] - trace_mark[3] : 0);
                        trace_set(base + 12, is_l1 ? trace_mark[1] - trace_mark[0] : 0);
                        trace_set(base + 13, valid_m);
                        trace_set(base + 14, trace_full_first);
                    }
                    ++ trace_task;
                }
                trace_prev_end = trace_end;
            }
        }
        if constexpr (kPhaseTrace) {
            if (trace_leader) trace_set(4, trace_clock());
        }
        if constexpr (kDistTrace) {
            if (trace_leader) dist_set(8, trace_globaltimer());
        }

#if DG_MonoEP_FORWARD_ROLE_REGISTERS && defined(DG_MonoEP_FORWARD_SPLIT_COMBINE_REGISTERS)
        // Without a region boundary here ptxas caps the whole math role near 128 registers.
        cutlass::arch::warpgroup_reg_dealloc<DG_MonoEP_FORWARD_SPLIT_COMBINE_REGISTERS>();
#endif
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

        // Publish this thread's combine writes once, before the grid sync that precedes the readiness flag.
        if constexpr (kDeferCombineFence and not (kDiagAblate & 4u))
            __threadfence_system();

        // All local L2 writes are done after this grid sync
        comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
            workspace, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );
        if constexpr (kDistTrace) {
            if (epilogue_thread_idx == 0) dist_set(9, trace_globaltimer());
        }
        [[maybe_unused]] uint64_t dist_first_ready_wait = 0;
        [[maybe_unused]] bool dist_first_token = true;

        // Notify remote ranks; ordered before the cleanup barrier by the dispatch/epilogue sync below
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
            ptx::st_rel_sys(sym_buffer.map(workspace.get_combine_ready_grid_idx_ptr(sym_buffer.rank_idx), epilogue_thread_idx), peer_grid_idx);

        // Barrier with dispatch warps, so that they can do clean workspace
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        const uint64_t trace_tail_start = trace_clock();
        uint64_t trace_peer_wait = 0;
        if constexpr (kPhaseTrace) {
            if (trace_leader) trace_set(35, trace_tail_start);
        }

        // Iterate over all token chunks, 1 token 1 topk latency: ~3 us
        const auto grid_idx = ptx::get_grid_idx() + 1;
        for (uint32_t token_chunk_idx = epilogue_warp_idx * kNumSMs + sm_idx;
             token_chunk_idx < num_tokens * kNumChunks and not (kDiagAblate & 16u);
             token_chunk_idx += kNumSMs * kNumEpilogueWarps) {
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
            const uint64_t trace_peer0 = trace_clock();
            if constexpr (kRelaxedSpins) {
                // Poll relaxed; each routed lane then acquires its peer's flag once (set once per launch).
                // Read-read coherence makes the acquire observe the flag the relaxed poll saw.
                comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or mono_ep_ld_relaxed_sys_u64(peer_ready_ptr) == grid_idx); },
                                 [&]() { printf("DeepGEMM combine peers timeout: rank=%u, token=%u\n", sym_buffer.rank_idx, token_idx); });
                if (is_routed)
                    (void) ptx::ld_acq_sys(peer_ready_ptr);
                __syncwarp();
            } else {
                comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                                 [&]() { printf("DeepGEMM combine peers timeout: rank=%u, token=%u\n", sym_buffer.rank_idx, token_idx); });
            }
            trace_peer_wait += trace_clock() - trace_peer0;
            if constexpr (kDistTrace) {
                if (dist_first_token) dist_first_ready_wait = trace_clock() - trace_peer0;
                dist_first_token = false;
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
        if constexpr (kPhaseTrace) {
            if (lane_idx == 0) trace_add(37, trace_peer_wait);
            if (lane_idx == 0) dist_add(10, trace_peer_wait);
            if (lane_idx == 0) dist_max(11, dist_first_ready_wait);
            if (trace_leader) {
                trace_set(36, trace_clock() - trace_tail_start);
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
