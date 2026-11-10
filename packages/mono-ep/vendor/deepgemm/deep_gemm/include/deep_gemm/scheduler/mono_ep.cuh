#pragma once

#include <cutlass/arch/barrier.h>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mono_ep.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

// TODO: add MonoEP namespace
namespace deep_gemm::sched {

// Get minimal L1 warmup waves to ensure no L1 -> L2 deadlock
// constexpr func, no runtime overhead in kernel
CUTLASS_HOST_DEVICE constexpr
int get_num_l1_warmup_waves(
    const int& num_total_m_blocks,
    const int& num_clusters,
    const int& num_l1_n_clusters,
    const int& num_l2_n_clusters) {
    // The first L2 wave may touch multiple M blocks; all their L1 N tasks must be issued first.
    const int num_first_l2_wave_m_blocks = math::constexpr_ceil_div(num_clusters, num_l2_n_clusters);
    const int num_l1_warmup_clusters_for_first_l2_wave = math::constexpr_ceil_div(
        num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters);

    // When each M block has more L1 tasks than L2 tasks, the interleaved schedule
    // leaves `(num_l1_pair_n_blocks - num_l2_pair_n_blocks)` extra L1 tasks behind
    // per M block. To avoid deadlock, no L2 task for an M block should be scheduled
    // before that block's L1 tasks have been issued. The last M block is the
    // bottleneck: it needs its own `num_l1_pair_n_blocks` L1 tasks, and the preceding
    // `num_total_m_blocks - 1` M blocks accumulate
    // `(num_total_m_blocks - 1) * (num_l1_pair_n_blocks - num_l2_pair_n_blocks)`
    // pending L1 tasks, so we issue them during the warmup phase. Add one extra
    // Add one cluster wave to cover partial-wave rounding.
    const int num_interleave_cluster_diff_per_m_block =
        num_l1_n_clusters > num_l2_n_clusters ? num_l1_n_clusters - num_l2_n_clusters : 0;
    const int num_warmup_waves_for_interleave_schedule = math::constexpr_ceil_div(
        num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block,
        num_clusters) + 1;

    // TODO: may delay combine NVLink
    return cute::max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule);
}

// Host-side ring capacity helper for the MonoEP schedule.
template <int kNumCTAsPerCluster = 2>
CUTLASS_HOST_DEVICE constexpr int get_num_max_live_pool_blocks(
    const int& num_total_m_blocks,
    const int& num_sms,
    const int& hidden,
    const int& intermediate_hidden) {
    constexpr int kMonoEPBlockN = 128;
    DG_UNIFIED_ASSERT(kNumCTAsPerCluster == 1 or kNumCTAsPerCluster == 2);
    DG_UNIFIED_ASSERT(num_sms % kNumCTAsPerCluster == 0);
    DG_UNIFIED_ASSERT((intermediate_hidden * 2) % (kNumCTAsPerCluster * kMonoEPBlockN) == 0);
    DG_UNIFIED_ASSERT(hidden % (kNumCTAsPerCluster * kMonoEPBlockN) == 0);
    const int num_clusters = num_sms / kNumCTAsPerCluster;
    const int num_l1_n_clusters = intermediate_hidden * 2 / (kNumCTAsPerCluster * kMonoEPBlockN);
    const int num_l2_n_clusters = hidden / (kNumCTAsPerCluster * kMonoEPBlockN);
    const int num_l1_clusters = num_total_m_blocks * num_l1_n_clusters;
    const int num_l1_waves = math::constexpr_ceil_div(num_l1_clusters, num_clusters);
    const int num_min_l1_warmup_waves = get_num_l1_warmup_waves(
        num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters);
    const int num_l1_warmup_waves = cute::min(num_min_l1_warmup_waves, num_l1_waves);
    const int num_l1_warmup_clusters = cute::min(num_l1_warmup_waves * num_clusters, num_l1_clusters);
    const int num_live_blocks_after_warmup = math::constexpr_ceil_div(num_l1_warmup_clusters, num_l1_n_clusters);

    // Conservative closed-form bound for the L2/L1 alternating tail. If L1 advances faster in
    // M-block space (B1 < B2), live blocks can grow by about total_m_blocks * (B2 - B1) / B2.
    // Add one global-wave margin to cover partial waves / pipeline lag.
    // This is slightly larger than the tight bound, but cheaper and simpler.
    // TODO: refactor here
    const int frontier_growth = num_l2_n_clusters > num_l1_n_clusters ?
        math::constexpr_ceil_div(
            num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters), num_l2_n_clusters) : 0;
    const int wave_margin = math::constexpr_ceil_div(
        num_clusters, cute::min(num_l1_n_clusters, num_l2_n_clusters));
    return cute::min(num_total_m_blocks, num_live_blocks_after_warmup + frontier_growth + wave_margin);
}

// Computation phase for the current block
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,
    Linear2 = 2,
    SharedLinear1 = 3,
    SharedLinear2 = 4,
    BackwardDh = 5,
    BackwardRecomputeLow = 6,
    BackwardRecomputeHigh = 7,
    BackwardDx = 8,
    BackwardDw1 = 9,
    BackwardDw2 = 10
};

template <bool kHasSharedExperts>
struct alignas(16) TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_cluster_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;

    CUTLASS_HOST_DEVICE
    TaskInfo(): TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0) {}

    CUTLASS_HOST_DEVICE
    TaskInfo(const BlockPhase& block_phase,
             const uint32_t& local_expert_idx,
             const uint32_t& m_block_idx,
             const uint32_t& pair_n_block_idx,
             const uint32_t& pool_block_idx,
             const uint32_t& valid_m,
             const uint32_t& shape_n,
             const uint32_t& shape_k):
        block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(m_block_idx),
        n_cluster_idx(pair_n_block_idx),
        pool_block_idx(pool_block_idx),
        valid_m(valid_m), shape_n(shape_n), shape_k(shape_k) {}

    CUTLASS_DEVICE
    uint32_t is_valid() const {
        return (block_phase != BlockPhase::None);
    }

    CUTLASS_DEVICE uint32_t get_n_block_idx(bool is_leader_cta) const {
        if (block_phase == BlockPhase::BackwardRecomputeLow or
            block_phase == BlockPhase::BackwardRecomputeHigh)
            return n_cluster_idx * 4 + (is_leader_cta ? 0u : 2u) +
                (block_phase == BlockPhase::BackwardRecomputeHigh ? 1u : 0u);
        return n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
    }

    CUTLASS_DEVICE
    uint32_t get_umma_aligned_valid_m() const {
        return math::align(valid_m, 16u);
    }

    CUTLASS_DEVICE uint32_t is_shared() const {
        return kHasSharedExperts ? (block_phase == BlockPhase::SharedLinear1 or
                                   block_phase == BlockPhase::SharedLinear2) : false;
    }
};

DG_STATIC_ASSERT(sizeof(sched::TaskInfo<true>) == sizeof(sched::TaskInfo<false>), "Invalid layout");

#if defined(__CUDACC__) or defined(__CLION_IDE__)

// Each finished L1 N block toggles its bit in `l2_full_mask`, so L2 K blocks can start as soon as they are fed
// NOTES: the mask parity per ring generation relies on the ring capacity, see `get_num_max_live_pool_blocks`
template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K>
struct L2KBlockDependency {
    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N / 2);
    static constexpr uint64_t kFullMask = kNumL1BlockNs == 64 ? ~0ull : (1ull << kNumL1BlockNs) - 1;
    DG_STATIC_ASSERT(kNumL1BlockNs <= 64, "Too many L1 N blocks for the L2 readiness mask");
    DG_STATIC_ASSERT(BLOCK_K % (BLOCK_N / 2) == 0, "Invalid L1/L2 shape relationship");

    const uint64_t* mask_ptr;
    uint64_t expected_mask, pending_mask = kFullMask;

    CUTLASS_DEVICE L2KBlockDependency(const uint64_t* mask_ptr, const uint32_t& generation_idx):
        mask_ptr(mask_ptr), expected_mask(generation_idx & 1 ? 0ull : kFullMask) {}

    CUTLASS_DEVICE static void arrive(const uint64_t* mask_ptr, const uint32_t& n_block_idx) {
        ptx::red_xor_rel(mask_ptr, 1ull << n_block_idx);
    }

    // Re-read the mask only when the K block is not fed yet
    CUTLASS_DEVICE void wait(const uint32_t& k_block_idx) {
        const auto k_block_mask = ((1ull << kNumL1BlockNsPerL2KBlock) - 1) << (k_block_idx * kNumL1BlockNsPerL2KBlock);
        while (pending_mask & k_block_mask)
            pending_mask = ptx::ld_acq_gpu(mask_ptr) ^ expected_mask;
    }
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumRingBlocks, uint32_t kNumLocalityDomains,
          uint32_t kNumSharedExperts = 0,
          uint32_t kNumCTAsPerCluster = 2,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1Clusters = kNumL1BlockNs / kNumCTAsPerCluster,
          uint32_t kNumL2Clusters = kNumL2BlockNs / kNumCTAsPerCluster>
struct MonoEPScheduler {
    static constexpr bool kHasShared = kNumSharedExperts > 0;
    static constexpr uint32_t kNumSchedulerClusters = kNumSMs / kNumCTAsPerCluster;
    static constexpr uint32_t SHARED_L1_SHAPE_N = L1_SHAPE_N * kNumSharedExperts;
    static constexpr uint32_t SHARED_L1_SHAPE_K = L1_SHAPE_K;
    static constexpr uint32_t SHARED_L2_SHAPE_N = L2_SHAPE_N;
    static constexpr uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts;
    using task_info_t = TaskInfo<kHasShared>;

    DG_STATIC_ASSERT(kNumCTAsPerCluster == 1 or kNumCTAsPerCluster == 2, "Invalid cluster size");
    DG_STATIC_ASSERT(L1_SHAPE_N % (BLOCK_N * kNumCTAsPerCluster) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % (BLOCK_N * kNumCTAsPerCluster) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_N % (BLOCK_N * kNumCTAsPerCluster) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_N % (BLOCK_N * kNumCTAsPerCluster) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");

    DG_STATIC_ASSERT(kNumSMs % kNumCTAsPerCluster == 0, "SM count must contain whole clusters");
    DG_STATIC_ASSERT(kNumRingBlocks > 0, "Invalid ring buffer config");

    // Workspace
    const layout::Workspace& workspace;

    // Scheduler configs
    static constexpr uint32_t kNumScheduleStages = 2;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    Barrier* task_info_full_barriers = nullptr;
    Barrier* task_info_empty_barriers = nullptr;
    task_info_t* task_infos = nullptr;

    // Pre-cached per-expert token counts.
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count.
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t num_total_m_blocks = 0;

    // Per-scheduler warmup waves; all cluster schedulers together form one global wave.
    static constexpr uint32_t kNumSchedL1WavesDone = 0xffffffffu;
    uint32_t num_sched_l1_waves = 0;

    // Locality domain affinity. When a queue is drained, we try work stealing to ensure load balancing.
    // But when there are few tasks, we disable work stealing to avoid cross-die latency.
    static constexpr uint32_t kNumMinStealWaves = 2;
    DG_STATIC_ASSERT(kNumLocalityDomains == 1 or kNumLocalityDomains == kNumDeviceLocalityDomains, "Invalid locality domain count");
    DG_STATIC_ASSERT(kNumL1Clusters % kNumLocalityDomains == 0 and kNumL2Clusters % kNumLocalityDomains == 0, "Each domain must take whole clusters");
    uint32_t sm_locality_domain_idx = 0;

    CUTLASS_DEVICE explicit MonoEPScheduler(const layout::Workspace& workspace):
        workspace(workspace) {}

    CUTLASS_DEVICE MonoEPScheduler(const layout::Workspace& workspace,
                                    Barrier* task_info_full_barriers,
                                    Barrier* task_info_empty_barriers,
                                    task_info_t* task_infos):
        workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

    CUTLASS_DEVICE void advance_sched_pipeline() {
        DG_STATIC_ASSERT(kNumScheduleStages == 2, "Invalid stages");
        sched_stage_idx ^= 1;
        sched_phase ^= sched_stage_idx == 0;
    }

    CUTLASS_DEVICE bool get_next_task(task_info_t& task_info) {
        task_info_full_barriers[sched_stage_idx].wait(sched_phase);
        task_info = task_infos[sched_stage_idx];
        advance_sched_pipeline();
        return task_info.is_valid();
    }

    CUTLASS_DEVICE void release_task_info() const {
        // Complete metadata reads before the scheduler can overwrite this slot.
        ptx::fence_acq_rel_cta();
        if constexpr (kNumCTAsPerCluster == 1)
            task_info_empty_barriers[sched_stage_idx ^ 1].arrive();
        else
            task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array.
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE uint32_t get_num_total_pool_blocks() const {
        return get_pool_block_offset(kNumExpertsPerRank);
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();

        num_total_m_blocks = get_num_total_pool_blocks();
        const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1Clusters;
        const uint32_t num_total_l1_waves = math::ceil_div(num_total_l1_tasks, kNumSchedulerClusters);
        const uint32_t min_l1_warmup_waves = get_num_l1_warmup_waves(
            num_total_m_blocks, kNumSchedulerClusters, kNumL1Clusters, kNumL2Clusters);
        num_sched_l1_waves = cute::min(min_l1_warmup_waves, num_total_l1_waves);
    }

    CUTLASS_DEVICE task_info_t create_task(const BlockPhase& block_phase,
                                        const uint32_t& m_block_idx,
                                        const uint32_t& n_cluster_idx,
                                        const uint32_t& shape_n,
                                        const uint32_t& shape_k) const {
        const uint32_t lane_idx = ptx::get_lane_idx();

        task_info_t result(block_phase, 0, 0, n_cluster_idx, m_block_idx, 0, shape_n, shape_k);
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            // Reduce whether task fall in the expert
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t inclusive_num_m_blocks = math::warp_inclusive_sum(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset = block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = expert_idx < kNumExpertsPerRank and
                m_block_idx >= lane_pool_block_offset and m_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            // Exchange the expert info
            if (owner_mask) {
                const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
                const uint32_t owner_m_block_idx = m_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m = cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
                result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
                result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
            }
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        return result;
    }

    static CUTLASS_DEVICE uint32_t get_next_local_task_idx(const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx) {
        uint32_t result = 0;
        if (cute::elect_one_sync())
            result = ptx::atomic_add(task_count_ptr + locality_domain_idx, 1u);
        return ptx::exchange(result, 0);
    }

    template <uint32_t kNumClusters>
    static CUTLASS_DEVICE bool get_next_task_idx(const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx, const uint32_t& num_m_blocks,
                                                 uint32_t& m_block_idx, uint32_t& n_cluster_idx) {
        constexpr uint32_t kNumLocalClusters = kNumClusters / kNumLocalityDomains;
        const uint32_t task_idx = get_next_local_task_idx(task_count_ptr, locality_domain_idx);
        if (task_idx >= num_m_blocks * kNumLocalClusters)
            return false;
        m_block_idx = task_idx / kNumLocalClusters;
        n_cluster_idx = locality_domain_idx * kNumLocalClusters + task_idx % kNumLocalClusters;
        return true;
    }

    template <uint32_t kNumClusters>
    CUTLASS_DEVICE bool get_next_task_idx(const uint32_t* task_count_ptr, const uint32_t& num_m_blocks,
                                          uint32_t& m_block_idx, uint32_t& n_cluster_idx) const {
        if (get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx, num_m_blocks, m_block_idx, n_cluster_idx))
            return true;
        return kNumLocalityDomains > 1 and num_m_blocks * kNumClusters >= kNumMinStealWaves * kNumSchedulerClusters and
               get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx ^ 1, num_m_blocks, m_block_idx, n_cluster_idx);
    }

    CUTLASS_DEVICE task_info_t get_next_task() {
        uint32_t m_block_idx, n_cluster_idx;
        while (true) {
            if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) {
                // One local L1 task per scheduler; globally this is one cluster wave.
                -- num_sched_l1_waves;

                // No more L1 tasks
                if (not get_next_task_idx<kNumL1Clusters>(workspace.get_l1_task_count_ptr(), num_total_m_blocks, m_block_idx, n_cluster_idx)) {
                    num_sched_l1_waves = kNumSchedL1WavesDone;
                    continue;
                }

                // Create task
                return create_task(BlockPhase::Linear1, m_block_idx, n_cluster_idx, L1_SHAPE_N, L1_SHAPE_K);
            } else {
                if (not get_next_task_idx<kNumL2Clusters>(workspace.get_l2_task_count_ptr(), num_total_m_blocks, m_block_idx, n_cluster_idx))
                    break;

                // The next task should be L1
                if (num_sched_l1_waves != kNumSchedL1WavesDone)
                    num_sched_l1_waves = 1;

                // Create task
                auto task_info = create_task(BlockPhase::Linear2, m_block_idx, n_cluster_idx, L2_SHAPE_N, L2_SHAPE_K);

                // Wait until all required L1 tasks are fetched from all scheduling queues
                const auto num_required_l1_tasks = (task_info.pool_block_idx + 1) * (kNumL1Clusters / kNumLocalityDomains);
                #pragma unroll
                for (uint32_t i = 0; i < kNumLocalityDomains; ++ i)
                    while (ptx::ld_volatile(workspace.get_l1_task_count_ptr() + i) < num_required_l1_tasks) {}
                return task_info;
            }
        }
        return task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0);
    }

    CUTLASS_DEVICE void publish_task(const task_info_t& task_info, const uint32_t& lane_idx) {
        if constexpr (kNumCTAsPerCluster == 1) {
            if (lane_idx == 0) {
                task_infos[sched_stage_idx] = task_info;
                __threadfence_block();
                task_info_full_barriers[sched_stage_idx].arrive();
            }
        } else {
            if (lane_idx < kNumCTAsPerCluster) {
                task_info_full_barriers[sched_stage_idx].arrive_and_expect_tx(sizeof(task_info_t), lane_idx);
                ptx::st_async_cluster(
                    task_infos + sched_stage_idx, task_info,
                    lane_idx, task_info_full_barriers[sched_stage_idx]
                );
            }
        }
        __syncwarp();
        advance_sched_pipeline();
    }

    template <BlockPhase kBlockPhase, uint32_t kShapeN, uint32_t kShapeK>
    CUTLASS_DEVICE void shared_mainloop(const uint32_t& num_tokens, const uint32_t& lane_idx, const uint32_t* task_count_ptr) {
        constexpr uint32_t kNumNClusters = kShapeN / BLOCK_N / kNumCTAsPerCluster;
        DG_STATIC_ASSERT(kNumNClusters % kNumLocalityDomains == 0, "Each domain must take whole clusters");
        const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
        uint32_t m_block_idx, n_cluster_idx;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);

            // Use dynamic scheduling to reduce tailing across shared L1/L2 tile shapes.
            if (not get_next_task_idx<kNumNClusters>(task_count_ptr, num_m_blocks, m_block_idx, n_cluster_idx))
                break;
            const uint32_t valid_m = cute::min(num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
            publish_task(task_info_t(kBlockPhase, 0, m_block_idx, n_cluster_idx, m_block_idx, valid_m, kShapeN, kShapeK), lane_idx);
        }
    }

    // Tail window: the last claimed M blocks, at most one per ring slot so per-slot counters are unique.
    static constexpr uint32_t kNumTailWindowBlocks = cute::min(kNumSMs, kNumRingBlocks);

    // Claim index -> pool block inside the tail window: full blocks in pool order, then partial
    // blocks by decreasing row count. The window spans at most one ring generation, so the dispatch
    // order (pool order) can never wait on a block that is claimed later.
    CUTLASS_DEVICE uint32_t get_tail_ordered_m_block(const uint32_t& claim_idx, const uint32_t& tail_begin) const {
        DG_STATIC_ASSERT(kNumExpertsPerLane == 1, "Tail ordering keeps one expert per lane");
        if (claim_idx < tail_begin)
            return claim_idx;
        const uint32_t lane_idx = ptx::get_lane_idx();
        const uint32_t count = lane_idx < kNumExpertsPerRank ? stored_num_tokens_per_expert[0] : 0;
        const uint32_t num_blocks = math::ceil_div(count, BLOCK_M);
        const uint32_t last = math::warp_inclusive_sum(num_blocks, lane_idx) - 1;
        const uint32_t rows = count % BLOCK_M;
        const bool partial = num_blocks > 0 and rows != 0 and last >= tail_begin;
        const uint32_t partial_mask = __ballot_sync(0xffffffff, partial);
        const uint32_t num_full = num_total_m_blocks - tail_begin - __popc(partial_mask);
        const uint32_t j = claim_idx - tail_begin;
        if (j < num_full) {
            uint32_t p = tail_begin + j;
            while (true) {
                const uint32_t q = tail_begin + j + __popc(__ballot_sync(0xffffffff, partial and last <= p));
                if (q == p) break;
                p = q;
            }
            return p;
        }
        uint32_t rank = 0;
        for (uint32_t mask = partial_mask; mask; mask &= mask - 1) {
            const uint32_t other = __ffs(mask) - 1;
            const uint32_t other_rows = __shfl_sync(0xffffffff, rows, other);
            const uint32_t other_last = __shfl_sync(0xffffffff, last, other);
            rank += other_rows > rows or (other_rows == rows and other_last < last);
        }
        const uint32_t owner = __ballot_sync(0xffffffff, partial and rank == j - num_full);
        return __shfl_sync(0xffffffff, last, __ffs(owner) - 1);
    }

    // After the M blocks run out, take dX tiles of tail-window blocks whose dZ is complete. Readiness
    // is the per-slot generation count, not the L2 full mask: before the previous generation's
    // recompute starts, the mask already holds this generation's "full" value.
    template <uint32_t kNumDxTiles>
    CUTLASS_DEVICE void steal_tail_dx_tiles(const uint32_t& tail_begin, const uint32_t& lane_idx) {
        uint32_t backoff_ns = 64;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            bool any_pending = false;
            uint32_t chosen = 0xffffffffu;
            for (uint32_t base = tail_begin; base < num_total_m_blocks; base += 32) {
                const uint32_t block = base + lane_idx;
                bool pending = false, ready = false;
                if (block < num_total_m_blocks) {
                    const uint32_t ring = block % kNumRingBlocks;
                    pending = ptx::ld_volatile(workspace.get_tail_dx_claim_count_ptr(ring)) < kNumDxTiles;
                    // Relaxed: only selects the block; the dX A loader's acquire of the L2 full mask orders dZ.
                    ready = pending and ptx::ld_volatile(workspace.get_tail_dz_ready_count_ptr(ring)) > block / kNumRingBlocks;
                }
                any_pending |= __any_sync(0xffffffff, pending);
                const uint32_t ready_mask = __ballot_sync(0xffffffff, ready);
                if (ready_mask) {
                    chosen = base + __ffs(ready_mask) - 1;
                    break;
                }
            }
            if (chosen != 0xffffffffu) {
                const uint32_t n = get_next_local_task_idx(workspace.get_tail_dx_claim_count_ptr(chosen % kNumRingBlocks), 0);
                if (n < kNumDxTiles)
                    publish_task(create_task(BlockPhase::BackwardDx, chosen, n, L2_SHAPE_N, L1_SHAPE_N), lane_idx);
                backoff_ns = 64;
                continue;
            }
            if (not any_pending)
                break;
            __nanosleep(backoff_ns);
            backoff_ns = cute::min(backoff_ns * 2, 1024u);
        }
    }

    // kTailMode bits (backward only): 1 = dX tiles of tail-window blocks are claimed per tile and
    // idle CTAs take them; 2 = partial blocks are claimed last inside the tail window.
    template <bool kBackward = false, uint32_t kTailMode = 0>
    CUTLASS_DEVICE void mainloop_owned() {
        DG_STATIC_ASSERT(not kHasShared, "Owned blocks currently require routed experts only");
        DG_STATIC_ASSERT(kBackward or kTailMode == 0, "Tail scheduling is backward-only");
        const auto lane_idx = ptx::get_lane_idx();
        fetch_expert_recv_count();
        const uint32_t tail_begin = num_total_m_blocks > kNumTailWindowBlocks ? num_total_m_blocks - kNumTailWindowBlocks : 0;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            // Claim only M blocks; all N tiles stay on this CTA cluster.
            const auto claim_idx = get_next_local_task_idx(workspace.get_l1_task_count_ptr(), 0);
            if (claim_idx >= num_total_m_blocks) break;
            const uint32_t m_block_idx = [&]() -> uint32_t {
                if constexpr ((kTailMode & 2u) != 0) return get_tail_ordered_m_block(claim_idx, tail_begin);
                else return claim_idx;
            }();
            auto emit = [&](BlockPhase phase, uint32_t n, uint32_t shape_n, uint32_t shape_k) {
                task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
                publish_task(create_task(phase, m_block_idx, n, shape_n, shape_k), lane_idx);
            };
            if constexpr (kBackward) {
                constexpr uint32_t kInner = L1_SHAPE_N / 2;
                DG_STATIC_ASSERT(kInner % (BLOCK_N * kNumCTAsPerCluster) == 0,
                                 "Backward dH tiles must cover whole CTA clusters");
                for (uint32_t n = 0; n < kInner / (BLOCK_N * kNumCTAsPerCluster); ++n) {
                    emit(BlockPhase::BackwardDh, n, kInner, L1_SHAPE_K);
                    emit(BlockPhase::BackwardRecomputeLow, n, L1_SHAPE_N, L1_SHAPE_K);
                    emit(BlockPhase::BackwardRecomputeHigh, n, L1_SHAPE_N, L1_SHAPE_K);
                }
                if ((kTailMode & 1u) and m_block_idx >= tail_begin) {
                    DG_STATIC_ASSERT(not (kTailMode & 1u) or kNumCTAsPerCluster == 1, "Tail dX claims are per CTA");
                    const auto claim_ptr = workspace.get_tail_dx_claim_count_ptr(m_block_idx % kNumRingBlocks);
                    while (true) {
                        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
                        const uint32_t n = get_next_local_task_idx(claim_ptr, 0);
                        if (n >= kNumL2Clusters) break;
                        publish_task(create_task(BlockPhase::BackwardDx, m_block_idx, n, L2_SHAPE_N, L1_SHAPE_N), lane_idx);
                    }
                } else {
                    for (uint32_t n = 0; n < kNumL2Clusters; ++n)
                        emit(BlockPhase::BackwardDx, n, L2_SHAPE_N, L1_SHAPE_N);
                }
            } else {
                for (uint32_t n = 0; n < kNumL1Clusters; ++n)
                    emit(BlockPhase::Linear1, n, L1_SHAPE_N, L1_SHAPE_K);
                for (uint32_t n = 0; n < kNumL2Clusters; ++n)
                    emit(BlockPhase::Linear2, n, L2_SHAPE_N, L2_SHAPE_K);
            }
        }
        if constexpr (kTailMode & 1u)
            steal_tail_dx_tiles<kNumL2Clusters>(tail_begin, lane_idx);
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
        publish_task(task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0), lane_idx);
    }

    // In-kernel dW (SM100 backward option 512): per local expert, (2I / 256) x (H / kDwBlockN) dW1 = dz^T x tiles,
    // then (H / 256) x (I / kDwBlockN) dW2 = dy^T h tiles, each reducing over all of the expert's padded pool rows.
    static constexpr uint32_t kNumDwTileM = BLOCK_N * kNumCTAsPerCluster;
    template <uint32_t kDwBlockN>
    struct DwTiles {
        static constexpr uint32_t kNumDw1TilesN = L1_SHAPE_K / kDwBlockN;
        static constexpr uint32_t kNumDw2TilesN = L1_SHAPE_N / 2 / kDwBlockN;
        static constexpr uint32_t kNumDw1Tiles = (L1_SHAPE_N / kNumDwTileM) * kNumDw1TilesN;
        static constexpr uint32_t kNumTiles = kNumDw1Tiles + (L1_SHAPE_K / kNumDwTileM) * kNumDw2TilesN;
        DG_STATIC_ASSERT(L1_SHAPE_K % kDwBlockN == 0 and (L1_SHAPE_N / 2) % kDwBlockN == 0, "Invalid dW tile N");
    };

    // Claim one dW tile of an expert whose h/dz rows are final: `ready[e]` counts finished (M block, CTA) halves,
    // so the target is 2 x its M blocks (0 for an expert without rows). The relaxed reads only select; the dW
    // TMA loaders acquire `ready[e]` themselves. `any_pending`: some expert still has unclaimed tiles.
    template <uint32_t kDwBlockN>
    CUTLASS_DEVICE bool publish_ready_dw_tile(const uint32_t* ready, const uint32_t* claims,
                                              const uint32_t& lane_idx, bool& any_pending) {
        using Tiles = DwTiles<kDwBlockN>;
        constexpr uint32_t kNumDwTiles = Tiles::kNumTiles, kNumDw1Tiles = Tiles::kNumDw1Tiles;
        DG_STATIC_ASSERT(kNumExpertsPerLane == 1 and kNumCTAsPerCluster == 2, "In-kernel dW keeps one expert per lane");
        const bool is_expert = lane_idx < kNumExpertsPerRank;
        const uint32_t num_blocks = is_expert ? math::ceil_div(stored_num_tokens_per_expert[0], BLOCK_M) : 0;
        const bool pending = is_expert and ptx::ld_volatile(claims + lane_idx) < kNumDwTiles;
        const bool is_ready = pending and ptx::ld_volatile(ready + lane_idx) >= 2 * num_blocks;
        any_pending = __any_sync(0xffffffff, pending);
        for (uint32_t mask = __ballot_sync(0xffffffff, is_ready); mask; mask &= mask - 1) {
            const uint32_t expert = __ffs(mask) - 1;
            const uint32_t tile = get_next_local_task_idx(claims, expert);
            if (tile >= kNumDwTiles)
                continue;
            const bool is_dw1 = tile < kNumDw1Tiles;
            const uint32_t local = is_dw1 ? tile : tile - kNumDw1Tiles;
            const uint32_t tiles_n = is_dw1 ? Tiles::kNumDw1TilesN : Tiles::kNumDw2TilesN;
            const uint32_t padded_rows = ptx::exchange(num_blocks, expert) * BLOCK_M;
            publish_task(task_info_t(is_dw1 ? BlockPhase::BackwardDw1 : BlockPhase::BackwardDw2, expert,
                                     local / tiles_n, local % tiles_n, get_pool_block_offset(expert),
                                     get_num_tokens(expert), 0, padded_rows), lane_idx);
            return true;
        }
        return false;
    }

    // `mainloop_owned<true>` plus dW tiles: up to `per_block` ready tiles after each M block's tasks (they fill
    // the tensor pipe while the next block's rows are still being pulled), then the remaining tiles as their
    // experts finish.
    template <uint32_t kDwBlockN>
    CUTLASS_DEVICE void mainloop_owned_backward_dw(const uint32_t* ready, const uint32_t* claims, const uint32_t& per_block) {
        DG_STATIC_ASSERT(not kHasShared, "Owned blocks currently require routed experts only");
        constexpr uint32_t kInner = L1_SHAPE_N / 2;
        DG_STATIC_ASSERT(kInner % (BLOCK_N * kNumCTAsPerCluster) == 0, "Backward dH tiles must cover whole CTA clusters");
        const auto lane_idx = ptx::get_lane_idx();
        fetch_expert_recv_count();
        bool any_pending = true;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            const auto m_block_idx = get_next_local_task_idx(workspace.get_l1_task_count_ptr(), 0);
            if (m_block_idx >= num_total_m_blocks) break;
            auto emit = [&](BlockPhase phase, uint32_t n, uint32_t shape_n, uint32_t shape_k) {
                task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
                publish_task(create_task(phase, m_block_idx, n, shape_n, shape_k), lane_idx);
            };
            for (uint32_t n = 0; n < kInner / (BLOCK_N * kNumCTAsPerCluster); ++n) {
                emit(BlockPhase::BackwardDh, n, kInner, L1_SHAPE_K);
                emit(BlockPhase::BackwardRecomputeLow, n, L1_SHAPE_N, L1_SHAPE_K);
                emit(BlockPhase::BackwardRecomputeHigh, n, L1_SHAPE_N, L1_SHAPE_K);
            }
            for (uint32_t n = 0; n < kNumL2Clusters; ++n)
                emit(BlockPhase::BackwardDx, n, L2_SHAPE_N, L1_SHAPE_N);
            for (uint32_t i = 0; i < per_block and any_pending; ++i) {
                task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
                if (not publish_ready_dw_tile<kDwBlockN>(ready, claims, lane_idx, any_pending))
                    break;
            }
        }
        uint32_t backoff_ns = 64;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            if (publish_ready_dw_tile<kDwBlockN>(ready, claims, lane_idx, any_pending)) {
                backoff_ns = 64;
                continue;
            }
            if (not any_pending)
                break;
            __nanosleep(backoff_ns);
            backoff_ns = cute::min(backoff_ns * 2, 1024u);
        }
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
        publish_task(task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0), lane_idx);
    }

    CUTLASS_DEVICE void mainloop(const uint32_t& num_tokens, const uint8_t* sm_locality_domains) {
        const auto lane_idx = ptx::get_lane_idx();
        sm_locality_domain_idx = kNumLocalityDomains == 1 ? 0u : sm_locality_domains[ptx::get_sm_idx()];

        // Shared L2 tasks go before the routed tasks if one wave fits the dispatch gap, else at the tail
        constexpr uint32_t kNumSharedL2Clusters = SHARED_L2_SHAPE_N / BLOCK_N / kNumCTAsPerCluster;
        const bool is_shared_l2_early = math::ceil_div(num_tokens, BLOCK_M) * kNumSharedL2Clusters <= kNumSchedulerClusters;
        if constexpr (kHasShared) {
            // Shared expert L1 tasks do not depend on dispatch.
            shared_mainloop<BlockPhase::SharedLinear1, SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l1_task_count_ptr());
            if (is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Wait dispatch's results
        fetch_expert_recv_count();

        // Generate routed tasks. Keep the original wait -> claim -> publish ordering:
        // `get_next_task()` advances global task counters and must not run before the
        // schedule slot is released by consumers.
        task_info_t task_info;
        do {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            task_info = get_next_task();
            if (task_info.is_valid()) publish_task(task_info, lane_idx);
        } while (task_info.is_valid());

        if constexpr (kHasShared) {
            if (not is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Sentinel.
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
        publish_task(task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0), lane_idx);
    }
};

#endif

} // namespace deep_gemm::sched
