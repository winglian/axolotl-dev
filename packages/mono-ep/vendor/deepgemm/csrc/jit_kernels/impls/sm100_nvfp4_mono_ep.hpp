#pragma once

#include <cstdlib>
#include <format>
#include <string>
#include <torch/python.h>

#include <deep_gemm/layout/mono_ep.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mono_ep.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// Persistent NVFP4 forward: BF16 tokens are quantized on arrival, both GEMMs run `kind::mxf4nvf4`
// block-scaled UMMA on packed E2M1 weights whose UE4M3 scales come pre-packed in UTCCP atoms.
static void sm100_nvfp4_mono_ep(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& expert_scales,
    const torch::Tensor& sm_locality_domains,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    DG_HOST_ASSERT(hidden % 256 == 0 and intermediate_hidden % 256 == 0);

    // Heuristics
    const auto config = get_mono_ep_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens, MmaKind::NVFP4);
    DG_HOST_ASSERT(config.block_k == 256 and config.block_n == 128);
    DG_HOST_ASSERT(config.store_block_m_l1 == config.store_block_m_l2);

    // Make tensormap
    // NOTES: token and weight tensors are packed FP4 (`int8` storage); the TMA coordinates count FP4 elements
    constexpr int kGranK = 16;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode, 0, false,
                                                     /*fp4_unpacked_smem=*/ false);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim, static_cast<int>(l1_acts_sf.stride(-1)));
    const auto tensor_map_l1_weights = make_tma_weights_3d_desc(l1_weights, config.block_k, config.load_block_n,
                                                                config.swizzle_weights_mode, /*fp4_unpacked_smem=*/ false);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode, 0, false,
                                                     /*fp4_unpacked_smem=*/ false);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim, static_cast<int>(l2_acts_sf.stride(-1)));
    const auto tensor_map_l2_weights = make_tma_weights_3d_desc(l2_weights, config.block_k, config.load_block_n,
                                                                config.swizzle_weights_mode, /*fp4_unpacked_smem=*/ false);

    // Weight scales: `[experts * (N / 128), K / 64, 128]` 32-bit words, one UTCCP atom per `(n_block, k_group)`
    const auto make_weights_sf_desc = [&](const torch::Tensor& sf, const int& shape_n, const int& shape_k) {
        const auto num_n_blocks = shape_n / 128, num_k_groups = shape_k / 64;
        DG_HOST_ASSERT(sf.dim() == 4 and sf.is_contiguous() and sf.scalar_type() == torch::kInt);
        DG_HOST_ASSERT(sf.size(0) == num_experts_per_rank and sf.size(1) == num_n_blocks);
        DG_HOST_ASSERT(sf.size(2) == num_k_groups and sf.size(3) == 128);
        return make_tma_3d_desc(sf,
                                128, num_k_groups, num_experts_per_rank * num_n_blocks,
                                128, config.block_k / 64, 1,
                                128, num_k_groups * 128,
                                0);
    };
    const auto tensor_map_l1_weights_sf = make_weights_sf_desc(l1_weights_sf, intermediate_hidden * 2, hidden);
    const auto tensor_map_l2_weights_sf = make_weights_sf_desc(l2_weights_sf, hidden, intermediate_hidden);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = get_num_sms_for_mono_ep();

    // Diagnostic only: a device pointer to a zeroed uint64 phase-trace buffer (carried in the stats argument)
    const char* phase_trace_selector = std::getenv("MONO_EP_SM100_NVFP4_PHASE_TRACE");
    const uint64_t phase_trace_ptr = phase_trace_selector == nullptr ? 0 :
        std::strtoull(phase_trace_selector, nullptr, 10);
    // Kernel variants (bitmask): 1 lookahead dispatch pull, the default whenever the pull buffer holds one row
    const char* options_selector = std::getenv("MONO_EP_SM100_NVFP4_OPTIONS");
    const uint32_t forward_options = options_selector == nullptr ? (config.num_bytes_per_pull == hidden * 2 ? 1u : 0u) :
        static_cast<uint32_t>(std::strtoul(options_selector, nullptr, 10));
    DG_HOST_ASSERT(forward_options < 2);
    const bool lookahead = forward_options & 1u;
    DG_HOST_ASSERT(not lookahead or config.num_bytes_per_pull == hidden * 2);
    const bool diag = phase_trace_ptr != 0;
    const std::string diag_suffix = (diag or lookahead)
        ? std::format(",\n        false, {}, 0u{}", diag ? "true" : "false", lookahead ? ", true" : "") : std::string();
    const std::string diag_name = (diag ? "_diag_t1" : "") + std::string(lookahead ? "_look" : "");
    if (phase_trace_ptr != 0) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats_ptr == nullptr);
        cumulative_local_expert_recv_stats_ptr = reinterpret_cast<int*>(phase_trace_ptr);
    }

    // Compile
    const std::string pull_name = config.num_bytes_per_pull > hidden * 2
        ? std::format("_pull{}", config.num_bytes_per_pull) : std::string();
    const auto kernel = jit->compile("sm100_nvfp4_mono_ep" + diag_name + pull_name, std::format(R"(
#include <deep_gemm/impls/sm100_nvfp4_mono_ep.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_nvfp4_mono_ep_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}{}
    >);
}};
)", num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, 0,
        num_topk,
        config.block_m, config.block_n, config.block_k,
        config.store_block_m_l1,
        config.sf_block_m, config.sf_block_n,
        config.num_ring_tokens,
        config.num_sf_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false", diag_suffix));

    // Launch
    if (mono_ep_prepare_only)
        return;
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
        },
        y.data_ptr(),
        cumulative_local_expert_recv_stats_ptr,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_acts_sf,
        tensor_map_l1_weights,
        tensor_map_l1_weights_sf,
        tensor_map_l2_acts,
        tensor_map_l2_acts_sf,
        tensor_map_l2_weights,
        tensor_map_l2_weights_sf,
        expert_scales.data_ptr<float>(),
        sm_locality_domains.data_ptr<uint8_t>());
}

} // namespace deep_gemm
