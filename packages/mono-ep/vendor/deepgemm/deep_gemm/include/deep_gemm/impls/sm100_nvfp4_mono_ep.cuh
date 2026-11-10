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
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/epilogue/sm100_moe_backward.cuh>
#include <deep_gemm/scheduler/mono_ep.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// Round a non-negative FP32 scale to UE4M3 (RNE, saturating at 448); returns the byte and the decoded value
CUTLASS_DEVICE uint32_t nvfp4_encode_e4m3_scale(const float& scale, float& decoded) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(0.0f), "f"(scale));
    uint32_t half2_bits;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(half2_bits) : "h"(packed));
    decoded = __low2float(*reinterpret_cast<const __half2*>(&half2_bits));
    return packed & 0xffu;
}

// Two FP32 values to one packed E2M1 byte (RNE, saturating at 6): the even element takes the low nibble
CUTLASS_DEVICE uint32_t nvfp4_pack_pair(const float& even, const float& odd) {
    uint16_t out;
    asm("{\n\t.reg .b8 byte;\n\tcvt.rn.satfinite.e2m1x2.f32 byte, %1, %2;\n\tcvt.u16.u8 %0, byte;\n\t}"
        : "=h"(out) : "f"(odd), "f"(even));
    return out;
}

// Quantize 16 BF16 values (two 16-byte vectors) into one NVFP4 block with torchao numerics:
// scale = UE4M3(amax / (6 * pts)), code = E2M1(x * (1 / pts) / scale). Returns the scale byte.
CUTLASS_DEVICE uint32_t nvfp4_quantize_bf16_block(const uint4& lo, const uint4& hi,
                                                   const float& sf_scale, const float& inv_pts,
                                                   uint32_t& word_0, uint32_t& word_1) {
    const auto p = reinterpret_cast<const nv_bfloat162*>(&lo);
    const auto q = reinterpret_cast<const nv_bfloat162*>(&hi);
    nv_bfloat162 amax2 = __hmax2(__habs2(p[0]), __habs2(p[1]));
    amax2 = __hmax2(amax2, __hmax2(__habs2(p[2]), __habs2(p[3])));
    amax2 = __hmax2(amax2, __hmax2(__habs2(q[0]), __habs2(q[1])));
    amax2 = __hmax2(amax2, __hmax2(__habs2(q[2]), __habs2(q[3])));
    const float amax = fmaxf(__low2float(amax2), __high2float(amax2));
    float decoded;
    const uint32_t sf_byte = nvfp4_encode_e4m3_scale(amax * sf_scale, decoded);
    const float r = decoded > 0.0f ? inv_pts / decoded : 0.0f;
    uint32_t bytes[8];
    #pragma unroll
    for (uint32_t j = 0; j < 4; ++ j) {
        const auto f = __bfloat1622float2(p[j]);
        bytes[j] = nvfp4_pack_pair(f.x * r, f.y * r);
    }
    #pragma unroll
    for (uint32_t j = 0; j < 4; ++ j) {
        const auto f = __bfloat1622float2(q[j]);
        bytes[4 + j] = nvfp4_pack_pair(f.x * r, f.y * r);
    }
    word_0 = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    word_1 = bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    return sf_byte;
}

CUTLASS_DEVICE void nvfp4_fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kOwnMBlocks = false,
    bool kPhaseTrace = false,
    uint32_t kDiagAblate = 0,
    bool kLookaheadPull = false,
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
sm100_nvfp4_mono_ep_impl(void* y,
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
                          const float* expert_scales,
                          const uint8_t* sm_locality_domains) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
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
    DG_STATIC_ASSERT(kNumSharedExperts == 0, "The NVFP4 forward has no shared expert path");

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
    }

    // Workspaces and Buffer
    const auto buffer = layout::MonoEPBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, kNumSFRingTokens,
        /*with_sf=*/ false,
        kNumSharedExperts,
        /*fp8_l1_only=*/ false, /*fp8_combine=*/ false,
        /*nvfp4=*/ true
    );
    const auto workspace = buffer.workspace;

    using L2KBlockDependency = sched::L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>;

    // SF configs: UE4M3 per 16 elements, UTCCP atoms of 128 rows x 4 K blocks (one 32-bit word per row)
    constexpr uint32_t kGranK = 16;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");
    DG_STATIC_ASSERT(kHidden % 256 == 0 and kIntermediateHidden % 256 == 0, "NVFP4 K blocks hold 256 elements");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // Data types
    // NOTES: both operands are packed E2M1 (two elements per byte in shared memory); the L2 staging stays BF16
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

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
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
        // NOTES: the L1 epilogue writes packed FP4 straight to global memory, so only L2 stages BF16 output
        struct {
            alignas(kSharedMemoryAlignment) d_dtype_t l2[kNumEpilogueWarpgroups][STORE_BLOCK_M * BLOCK_N];
        } smem_d;
        // Packed FP4 tiles: two elements per byte
        alignas(kSharedMemoryAlignment) uint8_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K / 2];
        alignas(kSharedMemoryAlignment) uint8_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K / 2];
        uint32_t smem_sfa[kNumStages][(BLOCK_K / UMMA_BLOCK_K) * SF_BLOCK_M];
        uint32_t smem_sfb[kNumStages][(BLOCK_K / UMMA_BLOCK_K) * SF_BLOCK_N];
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

    // Diagnostic per-CTA phase trace (kPhaseTrace only); the pointer rides in the stats argument, which is then unused.
    constexpr uint32_t kTraceCTAStride = 96;
    unsigned long long* const phase_trace = kPhaseTrace
        ? reinterpret_cast<unsigned long long*>(cumulative_local_expert_recv_stats) + static_cast<uint64_t>(sm_idx) * kTraceCTAStride
        : nullptr;
    int* const recv_stats = kPhaseTrace ? nullptr : cumulative_local_expert_recv_stats;
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
    constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 64 : 96;
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 88;
    constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 200 : 160;
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

        // Pull BF16 token rows from remote ranks and quantize them on arrival into the local packed-FP4 L1 ring
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

        // Pull geometry: whole BF16 rows or 128-byte-aligned chunks of them; every 64 elements (four FP4 blocks)
        // become 32 packed bytes and one 32-bit scale word
        constexpr uint32_t kRowBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumChunks = kRowBytes / kNumBytesPerPull;
        constexpr uint32_t kNumGroupsPerChunk = kNumBytesPerPull / 128;
        constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
        DG_STATIC_ASSERT(kNumBytesPerPull <= kRowBytes and kRowBytes % kNumBytesPerPull == 0, "Invalid pull buffer size");
        DG_STATIC_ASSERT(kNumBytesPerPull % 128 == 0, "The pull chunk must hold whole 64-element groups");
        DG_STATIC_ASSERT(not kLookaheadPull or kNumBytesPerPull == kRowBytes, "The lookahead pull moves whole rows");

        struct PullRow {
            const void* src;
            float weight;
            uint32_t pool_token_idx;
            uint32_t rank;
            uint32_t src_token_idx;
            uint32_t src_topk_idx;
            uint32_t token_idx_in_block;
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
            row.token_idx_in_block = token_idx_in_expert % BLOCK_M;
            row.local_expert_idx = current_expert_idx;
            row.is_last_token = token_idx == expert_end_idx - 1;
            row.src = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(row.src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            row.weight = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx, current_rank_in_expert_idx);
            return true;
        };

        // Per-expert gate/up input quantization constants: `1 / (6 * pts)` and `1 / pts`
        float sf_scale_x = 0.0f, inv_pts_x = 0.0f;
        int scale_expert_idx = -1;
        const auto load_input_scales = [&](const int& local_expert_idx) {
            if (local_expert_idx != scale_expert_idx) {
                const auto scales = *reinterpret_cast<const float2*>(expert_scales + local_expert_idx * 8);
                sf_scale_x = scales.x, inv_pts_x = scales.y;
                scale_expert_idx = local_expert_idx;
            }
        };
        const auto wait_ring_slot = [&](const PullRow& row) {
            const uint32_t pool_block_idx = row.pool_token_idx / BLOCK_M;
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
            }
        };
        const auto issue_chunk = [&](const PullRow& row, const uint32_t& chunk_idx) {
            if (cute::elect_one_sync()) {
                ptx::tma_load_1d(pull_buffer.get_base_ptr(), math::advance_ptr(row.src, chunk_idx * kNumBytesPerPull),
                                 pull_mbarrier, kNumBytesPerPull);
                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
            }
            __syncwarp();
        };
        // All lanes wait for the chunk, quantize it (one 64-element group per lane per round) and release the
        // pull buffer for the next asynchronous load
        const auto quantize_chunk = [&](const PullRow& row, const uint32_t& chunk_idx) {
            ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
            const auto src = pull_buffer.get_base_ptr<uint4>();
            const auto dst_row = buffer.l1_token_buffer.get_data_buffer(row.pool_token_idx % kNumRingTokens).template get_base_ptr<uint8_t>();
            const auto sf_base = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
            const uint32_t ring_block_idx = (row.pool_token_idx / BLOCK_M) % kNumRingBlocks;
            const uint32_t sf_token_idx = ring_block_idx * SF_BLOCK_M + transform_sf_token_idx(row.token_idx_in_block);
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumGroupsPerChunk, 32u); ++ i) {
                const uint32_t g = i * 32 + lane_idx;
                if (kNumGroupsPerChunk % 32 == 0 or g < kNumGroupsPerChunk) {
                    uint32_t sf_word = 0, words[8];
                    #pragma unroll
                    for (uint32_t b = 0; b < 4; ++ b) {
                        const auto lo = ptx::ld_shared(src + g * 8 + b * 2);
                        const auto hi = ptx::ld_shared(src + g * 8 + b * 2 + 1);
                        sf_word |= nvfp4_quantize_bf16_block(lo, hi, sf_scale_x, inv_pts_x, words[b * 2], words[b * 2 + 1]) << (b * 8);
                    }
                    const uint32_t group_idx = chunk_idx * kNumGroupsPerChunk + g;
                    const auto dst = reinterpret_cast<uint4*>(dst_row + group_idx * 32);
                    dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
                    dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
                    sf_base[group_idx * kNumSFRingTokens + sf_token_idx] = sf_word;
                }
            }
            // Generic reads of the pull buffer must complete before the next bulk copy overwrites it
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
        };
        // Publish the row: weight and source metadata, then the full count (release) after the FP4/SF stores
        // are made visible to the TMA loads of the consumer
        const auto publish = [&](const PullRow& row) {
            const uint32_t pool_block_idx = row.pool_token_idx / BLOCK_M;
            if (cute::elect_one_sync()) {
                *buffer.l1_topk_weights_buffer.get_data_buffer(row.pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = row.weight;
                *workspace.get_token_src_metadata_ptr(row.pool_token_idx) = {row.rank, row.src_token_idx, row.src_topk_idx};
            }
            nvfp4_fence_proxy_async_global();
            __syncwarp();
            if (cute::elect_one_sync()) {
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks),
                    row.is_last_token ? BLOCK_M - (row.pool_token_idx % BLOCK_M) : 1u);
            }
            __syncwarp();
            if constexpr (kDistTrace) {
                if (lane_idx == 0) {
                    const uint64_t arrival = trace_globaltimer();
                    dist_add(8 + row.rank, 1);
                    dist_max(16 + row.rank, ~arrival);
                    dist_max(24 + row.rank, arrival);
                }
            }
        };

        if constexpr (kLookaheadPull) {
            // One row per buffer: the next row's routing is resolved and its load issued while this row is
            // quantized and published. Same rows, ring slots and counts as the default loop.
            uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx;
            PullRow cur, next;
            bool has_cur = resolve(token_idx, cur);
            if (has_cur)
                issue_chunk(cur, 0);
            while (has_cur) {
                token_idx += kNumGlobalWarps;
                const bool has_next = resolve(token_idx, next);

                const uint64_t trace_ring0 = trace_clock();
                wait_ring_slot(cur);
                const uint64_t trace_copy0 = trace_clock();
                trace_ring_wait += trace_copy0 - trace_ring0;
                ++ trace_rows;

                load_input_scales(cur.local_expert_idx);
                quantize_chunk(cur, 0);
                if (has_next)
                    issue_chunk(next, 0);
                publish(cur);
                trace_copy += trace_clock() - trace_copy0;
                cur = next;
                has_cur = has_next;
            }
        } else {
            for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
                PullRow row;
                if (not resolve(token_idx, row))
                    break;

                const uint64_t trace_ring0 = trace_clock();
                wait_ring_slot(row);
                const uint64_t trace_copy0 = trace_clock();
                trace_ring_wait += trace_copy0 - trace_ring0;
                ++ trace_rows;

                load_input_scales(row.local_expert_idx);
                #pragma unroll 1
                for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++ chunk_idx) {
                    issue_chunk(row, chunk_idx);
                    quantize_chunk(row, chunk_idx);
                }
                publish(row);
                trace_copy += trace_clock() - trace_copy0;
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

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and recv_stats != nullptr)
                        ptx::red_add(recv_stats + i, static_cast<int>(num_recv_tokens));
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
            const auto tensor_map_a_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts : &tensor_map_l2_acts;
            const auto tensor_map_sfa_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts_sf : &tensor_map_l2_acts_sf;
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
                while (ptx::ld_acq(ptr) != num_expected_tokens);
                trace_l1_spin += trace_clock() - trace_spin0;
            }

            L2KBlockDependency l2_k_block_dependency(workspace.get_l2_full_mask_ptr(ring_block_idx), pool_block_idx / kNumRingBlocks);

            // Issue TMA
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                const uint64_t trace_dep0 = trace_clock();
                if (task_info.block_phase == sched::BlockPhase::Linear2)
                    l2_k_block_dependency.wait(k_block_idx);

                // Wait consumer release
                const uint64_t trace_empty0 = trace_clock();
                trace_l2_spin += trace_empty0 - trace_dep0;
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
                trace_a_empty += trace_clock() - trace_empty0;

                // Compute token offset from ring block index (K coordinates count FP4 elements)
                uint32_t m_idx = block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sfa_m_idx = block_idx * SF_BLOCK_M;
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
            const bool is_l1 = task_info.block_phase == sched::BlockPhase::Linear1;
            const auto tensor_map_b_ptr = is_l1 ? &tensor_map_l1_weights : &tensor_map_l2_weights;
            const auto tensor_map_sfb_ptr = is_l1 ? &tensor_map_l1_weights_sf : &tensor_map_l2_weights_sf;
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
                uint32_t k_idx = k_block_idx * BLOCK_K;
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
            // Make instruction descriptor with block scaling
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
                        const uint32_t phase_idx = task_info.block_phase == sched::BlockPhase::Linear1 ? 0u : 1u;
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
                scheduler.mainloop_owned();
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
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
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
            uint64_t trace_spin = 0, trace_compute = 0, trace_store = 0, trace_tail = 0, trace_scatter = 0;

            // Now we can release the task
            scheduler.release_task_info();

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring-buffer offset for reusable data buffers
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;  // Full-pool offset for non-ring metadata
            const uint32_t n_block_idx = task_info.get_n_block_idx(is_leader_cta);
            uint32_t n_idx = n_block_idx * BLOCK_N;

            // Per-expert scale constants: {1 / (6 pts_x), 1 / pts_x, gate fold, up fold} and
            // {1 / (6 pts_h), 1 / pts_h, L2 fold, unused}; the folds carry the activation and weight per-tensor scales
            const auto l1_scales = *reinterpret_cast<const float4*>(expert_scales + task_info.local_expert_idx * 8);
            const auto l2_scales = *reinterpret_cast<const float4*>(expert_scales + task_info.local_expert_idx * 8 + 4);

            if (task_info.block_phase == sched::BlockPhase::Linear1) {
                {
                    // Wait L2 block empty
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const auto num_expected_blocks = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(l2_empty_ptr) != num_expected_blocks);
                }
                trace_spin = trace_clock() - trace_got;

                // L1 epilogue: SwiGLU on granularity 8 interleaved gate/up columns, then NVFP4 quantization of the
                // 16 intermediate columns each warp owns (one scale block per row) and direct packed stores.
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                //   (values[0], values[2]), (values[1], values[3]),
                //   (values[4], values[6]), (values[5], values[7])
                // and each lane holds rows `(lane % 4) * 2 + {0, 1}` of columns `lane / 4 + {0, 8}`.
                DG_STATIC_ASSERT(L1_OUT_BLOCK_N / 4 == kGranK, "Each epilogue warp owns exactly one scale block of columns");
                const uint32_t sf_mn_stride = kNumSFRingTokens * static_cast<uint32_t>(sizeof(uint32_t));
                const auto l2_sf_base = buffer.l2_sf_buffer.get_base_ptr<uint8_t>();
                const uint32_t h_col_idx = n_block_idx * L1_OUT_BLOCK_N + warp_idx_in_wg * kGranK;
                const bool is_even_col_group = ((lane_idx / 4) % 2) == 0;
                const uint32_t byte_slot = lane_idx / 8;

                // TopK weight for this lane
                float stored_cached_weight = 1.0f;

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        break;
                    }

                    const uint64_t trace_c0 = trace_clock();
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // Load weights from global into register cache per 32 tokens
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        if ((j * ATOM_M) % 32 == 0 and
                            (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                            stored_cached_weight = *buffer.l1_topk_weights_buffer
                                .get_data_buffer(ring_m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                .template get_base_ptr<float>();
                        }

                        // Load weights from register cache
                        const float2 weights = {
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // Load from TMEM
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Signal tensor memory consumed on the last atom
                        if (j == WG_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }

                        // Apply the per-tensor folds and SwiGLU in FP32: h = silu(gate) * up * weight
                        auto fp32_values = reinterpret_cast<float*>(values);
                        float2 h[2];
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto gate = __fmul2_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]), {l1_scales.z, l1_scales.z});
                            auto up = __fmul2_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]), {l1_scales.w, l1_scales.w});
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                gate = {fminf(gate.x, kActivationClamp), fminf(gate.y, kActivationClamp)};
                                up = {fminf(fmaxf(up.x, -kActivationClamp), kActivationClamp), fminf(fmaxf(up.y, -kActivationClamp), kActivationClamp)};
                            }
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            h[k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }

                        // Row amax over the warp's 16 columns: lanes sharing `lane % 4` hold the same two rows
                        float2 amax = {fmaxf(fabsf(h[0].x), fabsf(h[1].x)), fmaxf(fabsf(h[0].y), fabsf(h[1].y))};
                        #pragma unroll
                        for (uint32_t offset = 4; offset <= 16; offset <<= 1) {
                            amax.x = fmaxf(amax.x, __shfl_xor_sync(0xffffffff, amax.x, offset));
                            amax.y = fmaxf(amax.y, __shfl_xor_sync(0xffffffff, amax.y, offset));
                        }

                        // UE4M3 scales and the per-row encode factors
                        float2 decoded;
                        const uint2 sf_bytes = {nvfp4_encode_e4m3_scale(amax.x * l2_scales.x, decoded.x),
                                                nvfp4_encode_e4m3_scale(amax.y * l2_scales.x, decoded.y)};
                        const float2 r = {decoded.x > 0.0f ? l2_scales.y / decoded.x : 0.0f,
                                          decoded.y > 0.0f ? l2_scales.y / decoded.y : 0.0f};
                        float2 q[2];
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k)
                            q[k] = __fmul2_rn(h[k], r);

                        // Pack column pairs: the odd column of each pair lives in lane + 4
                        float2 partner[2];
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            partner[k].x = __shfl_down_sync(0xffffffff, q[k].x, 4);
                            partner[k].y = __shfl_down_sync(0xffffffff, q[k].y, 4);
                        }
                        // Row words: byte `lane / 8` holds column pair `k = 0`, byte `4 + lane / 8` column pair `k = 1`
                        uint64_t row_words[2] = {0ull, 0ull};
                        if (is_even_col_group) {
                            #pragma unroll
                            for (uint32_t v = 0; v < 2; ++ v) {
                                const uint32_t byte_0 = nvfp4_pack_pair(v == 0 ? q[0].x : q[0].y, v == 0 ? partner[0].x : partner[0].y);
                                const uint32_t byte_1 = nvfp4_pack_pair(v == 0 ? q[1].x : q[1].y, v == 0 ? partner[1].x : partner[1].y);
                                row_words[v] = (static_cast<uint64_t>(byte_0) << (8 * byte_slot)) |
                                               (static_cast<uint64_t>(byte_1) << (8 * (4 + byte_slot)));
                            }
                        }
                        #pragma unroll
                        for (uint32_t v = 0; v < 2; ++ v) {
                            uint32_t lo = static_cast<uint32_t>(row_words[v]), hi = static_cast<uint32_t>(row_words[v] >> 32);
                            #pragma unroll
                            for (uint32_t offset = 8; offset <= 16; offset <<= 1) {
                                lo |= __shfl_xor_sync(0xffffffff, lo, offset);
                                hi |= __shfl_xor_sync(0xffffffff, hi, offset);
                            }
                            row_words[v] = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
                        }

                        // Lanes 0-3 store the even rows, lanes 8-11 the odd rows: 8 bytes (16 FP4) per row
                        const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                        if (byte_slot < 2 and is_even_col_group) {
                            const uint32_t row_in_block = token_base_idx + (lane_idx % 4) * 2 + byte_slot;
                            if (row_in_block < valid_m) {
                                const auto dst = buffer.l2_token_buffer.get_data_buffer(ring_m_idx + row_in_block).template get_base_ptr<uint8_t>() + h_col_idx / 2;
                                const auto word = row_words[byte_slot];
                                *reinterpret_cast<uint2*>(dst) = make_uint2(static_cast<uint32_t>(word), static_cast<uint32_t>(word >> 32));
                            }
                        }

                        // Store SF bytes (MN-major words of 4 blocks; this warp owns byte `warp_idx_in_wg` of word `n_block_idx`)
                        // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so their transformed indices differ by 4
                        if (lane_idx < 4) {
                            __builtin_assume(token_base_idx < BLOCK_M);
                            const auto sf_token_idx = block_idx * SF_BLOCK_M + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                            const auto sf_addr = n_block_idx * sf_mn_stride + sf_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + warp_idx_in_wg;
                            l2_sf_base[sf_addr] = static_cast<uint8_t>(sf_bytes.x);
                            l2_sf_base[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] = static_cast<uint8_t>(sf_bytes.y);
                        }
                    }
                    trace_compute += trace_clock() - trace_c0;
                }
                const uint64_t trace_t0 = trace_clock();

                // Notify L2 and increment L1 empty count
                // NOTES: the packed FP4 and SF stores are generic; the L2 TMA loads read them through the async proxy
                nvfp4_fence_proxy_async_global();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    L2KBlockDependency::arrive(workspace.get_l2_full_mask_ptr(ring_block_idx), n_block_idx);

                    // Increment L1 empty count for this physical slot (one per N block)
                    ptx::red_add(
                        workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                }
                __syncwarp();
                trace_tail = trace_clock() - trace_t0;
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
                if constexpr (kDiagAblate & 4u) {
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
                        const auto scale_and_pack = [&](const uint32_t& x, const uint32_t& y) {
                            const auto bf16x2 = __floats2bfloat162_rn(__uint_as_float(x) * l2_scales.z, __uint_as_float(y) * l2_scales.z);
                            return *reinterpret_cast<const uint32_t*>(&bf16x2);
                        };
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            scale_and_pack(values[0], values[1]),
                            scale_and_pack(values[2], values[3]),
                            scale_and_pack(values[4], values[5]),
                            scale_and_pack(values[6], values[7]),
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
                trace_scatter = trace_dx1 - trace_dx0;

                // Ensure the next epilogue safe to use shared memory
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                trace_tail = trace_clock() - trace_dx1;
            }
            if constexpr (kPhaseTrace) {
                const uint64_t trace_end = trace_clock();
                if (trace_leader) {
                    const bool trace_is_l1 = task_info.block_phase == sched::BlockPhase::Linear1;
                    const uint32_t phase_idx = trace_is_l1 ? 0u : 1u;
                    const uint64_t busy = trace_end - trace_got, wait = trace_got - trace_task;
                    trace_add(6, wait);
                    trace_add(7 + phase_idx, busy);
                    trace_add(9 + phase_idx, wait);
                    trace_add(11, trace_spin);
                    trace_add(12, trace_compute);
                    trace_add(13, trace_store);
                    trace_add(14 + phase_idx, trace_tail);
                    trace_add(16, trace_scatter);
                    trace_add(17, trace_task - trace_prev_end);
                    trace_add(18 + phase_idx, 1);
                }
                trace_prev_end = trace_end;
            }
        }
        if constexpr (kPhaseTrace) {
            if (trace_leader) trace_set(5, trace_clock());
            if (trace_leader) dist_set(41, trace_globaltimer());
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
        if constexpr (kPhaseTrace) {
            if (trace_leader) trace_set(21, trace_clock());
            if (trace_leader) dist_set(43, trace_globaltimer());
        }
        [[maybe_unused]] uint64_t dist_ready_wait = 0, dist_first_ready_wait = 0, dist_tokens = 0;
        [[maybe_unused]] bool dist_first_token = true;
        if constexpr (not (kDiagAblate & 8u))
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
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm
