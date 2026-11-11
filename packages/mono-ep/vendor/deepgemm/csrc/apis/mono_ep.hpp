#pragma once

#include <functional>
#include <string>
#include <pybind11/functional.h>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/scheduler/mono_ep.cuh>

#include "../runtime/runtime.hpp"
#include "../jit_kernels/impls/sm90_bf16_mono_ep.hpp"
#include "../jit_kernels/impls/sm90_bf16_mono_ep_backward.hpp"
#include "../jit_kernels/impls/sm90_fp8_mono_ep_backward.hpp"
#include "../jit_kernels/impls/sm90_fp8_mono_ep_forward.hpp"
#include "../jit_kernels/impls/sm100_bf16_mono_ep.hpp"
#include "../jit_kernels/impls/sm100_bf16_mono_ep_backward.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mono_ep.hpp"
#include "../jit_kernels/impls/sm100_nvfp4_mono_ep.hpp"
#include "../jit_kernels/impls/sm100_nvfp4_mono_ep_backward.hpp"
#include "locality_domain.hpp"

namespace deep_gemm::mega {

static int get_token_alignment_for_mono_ep() {
    return layout::kLCMCandidateBlockM;
}

static int get_block_m_for_mono_ep(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const std::string& mma_type) {
    DG_HOST_ASSERT(num_tokens >= 0);
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto [cluster_size, block_m, store_block_m_l1, store_block_m_l2, block_k, num_epilogue_threads] =
        get_block_config_for_mono_ep(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens, mma_kind);
    return block_m;
}

static std::tuple<torch::Tensor, torch::Tensor> get_mono_ep_backward_combine_views(
    const torch::Tensor& buffer, const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk, const int& hidden,
    const int& intermediate_hidden, const int& num_ring_tokens,
    const bool& fp8_transport, const bool& fp8_combine) {
    DG_HOST_ASSERT(buffer.is_cuda() and buffer.is_contiguous() and buffer.scalar_type() == torch::kInt8);
    DG_HOST_ASSERT(num_ranks > 0 and num_experts > 0 and num_experts % num_ranks == 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0 and num_topk > 0 and hidden > 0 and hidden % 128 == 0);
    DG_HOST_ASSERT(not fp8_combine or fp8_transport);
    const auto layout = layout::MonoEPBuffer(
        nullptr, hidden, intermediate_hidden, num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk, num_ring_tokens, 0, false, 0,
        fp8_transport, fp8_combine);
    DG_HOST_ASSERT(buffer.nbytes() >= static_cast<size_t>(layout.get_num_bytes()));
    auto values = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(layout.combine_token_buffer.base)),
        {num_topk, num_max_tokens_per_rank, hidden},
        torch::TensorOptions()
            .dtype(fp8_combine ? torch::kFloat8_e4m3fn : torch::kBFloat16)
            .device(buffer.device()));
    auto scales = fp8_combine ? torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(layout.combine_sf_buffer.base)),
        {num_topk, num_max_tokens_per_rank, hidden / 128},
        torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device())) : torch::Tensor();
    return {values, scales};
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_mono_ep(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0, const int& num_min_ring_tokens = 0,
    const int& num_exact_ring_tokens = 0, const int& num_ring_routed_tokens = 0) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(num_min_ring_tokens >= 0 and num_exact_ring_tokens >= 0 and num_ring_routed_tokens >= 0);
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(num_shared_experts >= 0);

    // Ring capacity: worst-case live pool blocks over all candidate BLOCK_M; mirrors the kernel assert.
    // TODO: we temporarily assume the SM count is consistent with the runtime value
    const auto num_sms = get_num_sms_for_mono_ep();
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto num_active_topk = std::min(num_topk, num_experts_per_rank);
    // NOTES: a routed-token budget below the worst case sizes the ring for that budget, and an exact size skips the
    // live-block bound; only the SM100 BF16 forward (ring-gated scheduler) and owned backwards may run on such rings
    const auto num_worst_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;
    const auto num_max_routed_tokens = num_ring_routed_tokens > 0 ?
        std::min(num_ring_routed_tokens, num_worst_routed_tokens) : num_worst_routed_tokens;

    // Shared
    const int shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Iterate all block candidates to get the maximum ring size (for the budget and for the worst case)
    int num_ring_tokens = 0, num_worst_ring_tokens = 0;
    const auto arch_major = jit->device.get_arch_major();
    const auto live_ring_tokens = [&](const int& num_routed_tokens, const int& block_m) {
        const auto num_pool_blocks = ceil_div(num_routed_tokens, block_m) + num_experts_per_rank;
        const auto num_live_pool_blocks = arch_major == 9 ?
            sched::get_num_max_live_pool_blocks<1>(num_pool_blocks, num_sms, hidden, intermediate_hidden) :
            sched::get_num_max_live_pool_blocks<2>(num_pool_blocks, num_sms, hidden, intermediate_hidden);
        return num_live_pool_blocks * block_m;
    };
    for (const auto& block_m: layout::kCandidateBlockM) {
        num_ring_tokens = std::max(num_ring_tokens, live_ring_tokens(num_max_routed_tokens, block_m));
        num_worst_ring_tokens = std::max(num_worst_ring_tokens, live_ring_tokens(num_worst_routed_tokens, block_m));
    }
    // Owned schedules (the backward) hold one live block per CTA cluster; callers raise the floor for them.
    // An exact size never exceeds what the worst case (with that floor) would allocate.
    const int num_ring_ceiling = std::max(num_worst_ring_tokens, num_min_ring_tokens);
    num_ring_tokens = math::align(
        num_exact_ring_tokens > 0 ? std::min(num_exact_ring_tokens, num_ring_ceiling) :
                                    std::max(num_ring_tokens, num_min_ring_tokens),
        layout::kLCMCandidateBlockM);

    // Parse MMA type
    const bool fp8_combine = mma_type == "fp8xbf16_fp8combine";
    const auto mma_kind = parse_mma_kind(fp8_combine ? "fp8xbf16" : mma_type);
    // NVFP4 keeps BF16 inputs and only the ring buffers carry scales
    const bool nvfp4 = mma_kind == MmaKind::NVFP4;
    const auto with_sf = is_mma_with_sf(mma_kind) and not nvfp4;
    const bool fp8_l1_only = mma_type == "fp8xbf16" or fp8_combine;
    DG_HOST_ASSERT(not (fp8_l1_only or nvfp4) or num_shared_experts == 0);
    DG_HOST_ASSERT(not nvfp4 or (hidden % 256 == 0 and intermediate_hidden % 256 == 0));

    // Compute num_sf_ring_tokens (max across all candidate block sizes)
    int num_sf_ring_tokens = 0;
    if (with_sf or nvfp4) {
        for (auto block_m: layout::kCandidateBlockM) {
            num_sf_ring_tokens = std::max(
                num_sf_ring_tokens,
                layout::get_num_sf_ring_tokens(num_ring_tokens, block_m));
        }
    }

    // All buffers
    const auto mega_buffer = layout::MonoEPBuffer(
        nullptr, hidden, intermediate_hidden,
        num_ranks, num_experts, num_max_tokens_per_rank,
        num_topk, num_ring_tokens, num_sf_ring_tokens, with_sf,
        num_shared_experts, fp8_l1_only, fp8_combine, nvfp4
    );

    // Check SF buffer requirements
    if (with_sf) {
        DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(shared_intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(num_sf_ring_tokens % 4 == 0);
    }

    // Slice function: creates tensor views from the raw buffer.
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype((with_sf or fp8_l1_only) ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto x_sf = (with_sf or fp8_l1_only) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            fp8_l1_only ? std::vector<int64_t>{math::align(hidden / 32, 16) / 4, 1} :
                          std::vector<int64_t>{hidden / 128, 1},
            torch::TensorOptions().dtype(fp8_l1_only ? torch::kFloat32 : torch::kInt).device(buffer.device())) : torch::Tensor();
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));

        auto shared_l1_acts = x;
        auto shared_l1_acts_sf = (with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l1_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts = num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_token_buffer.base)),
            {num_max_tokens_per_rank, shared_intermediate_hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts_sf = (with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), shared_intermediate_hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();

        if (nvfp4) {
            // Packed FP4 ring tokens with MN-major 32-bit scale words (four UE4M3 K groups per word)
            auto l1_acts = torch::from_blob(
                math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base)),
                {num_ring_tokens, hidden / 2},
                torch::TensorOptions().dtype(kPackedFP4).device(buffer.device()));
            auto l1_acts_sf = torch::from_blob(
                math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base)),
                {num_sf_ring_tokens, hidden / 64}, {1, num_sf_ring_tokens},
                torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
            auto l2_acts = torch::from_blob(
                math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base)),
                {num_ring_tokens, intermediate_hidden / 2},
                torch::TensorOptions().dtype(kPackedFP4).device(buffer.device()));
            auto l2_acts_sf = torch::from_blob(
                math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base)),
                {num_sf_ring_tokens, intermediate_hidden / 64}, {1, num_sf_ring_tokens},
                torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
            return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                                   shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                                   l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
        }
        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base)),
            {num_ring_tokens, hidden},
            torch::TensorOptions().dtype((with_sf or fp8_l1_only) ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto l1_acts_sf = (with_sf or fp8_l1_only) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base)),
            {fp8_l1_only ? num_ring_tokens : num_sf_ring_tokens, hidden / 128},
            fp8_l1_only ? std::vector<int64_t>{math::align(hidden / 32, 16) / 4, 1} :
                          std::vector<int64_t>{1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(fp8_l1_only ? torch::kFloat32 : torch::kInt).device(buffer.device())) : torch::Tensor();
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base)),
            {num_ring_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto l2_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base)),
            {num_sf_ring_tokens, intermediate_hidden / 128},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                               shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                               l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {mega_buffer.get_num_bytes(), slice_input_buffers};
}

static void fp8_fp4_mono_ep(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const std::optional<torch::Tensor>& saved_z
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(shared_l1_weights_tuple_opt.has_value() == shared_l2_weights_tuple_opt.has_value());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = check_weights_layout_3d(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = check_weights_layout_3d(l2_weights);
    DG_HOST_ASSERT(is_localized(l1_weights) == is_localized(l2_weights));
    const auto weight_dtype = l1_weights.scalar_type();
    DG_HOST_ASSERT(weight_dtype == torch::kFloat8_e4m3fn or weight_dtype == kPackedFP4);
    DG_HOST_ASSERT(l2_weights.scalar_type() == weight_dtype);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);

    if (saved_z.has_value()) {
        DG_HOST_ASSERT(weight_dtype == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(not shared_l1_weights_tuple_opt.has_value() and not activation_clamp_opt.has_value());
        DG_HOST_ASSERT(saved_z->is_cuda() and saved_z->device() == y.device() and saved_z->is_contiguous());
        DG_HOST_ASSERT(saved_z->scalar_type() == torch::kBFloat16);
        const auto [ranks, tokens, slots, width] = get_shape<4>(*saved_z);
        DG_HOST_ASSERT(ranks == sym_buffer_ptrs.size() and tokens == num_max_tokens_per_rank);
        DG_HOST_ASSERT(slots == num_topk and width == 2 * intermediate_hidden);
    }

    // Check weight SF layout for UE8M0 packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1, kGranK = 32;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l1_weights_sf, shared_l2_weights, shared_l2_weights_sf;
    if (shared_l1_weights_tuple_opt.has_value()) {
        std::tie(shared_l1_weights, shared_l1_weights_sf) = shared_l1_weights_tuple_opt.value();
        std::tie(shared_l2_weights, shared_l2_weights_sf) = shared_l2_weights_tuple_opt.value();
        const auto [shared_l1_n, shared_l1_k] = check_weights_layout_2d(shared_l1_weights);
        const auto [shared_l2_n, shared_l2_k] = check_weights_layout_2d(shared_l2_weights);
        DG_HOST_ASSERT(is_localized(shared_l1_weights) == is_localized(l1_weights));
        DG_HOST_ASSERT(is_localized(shared_l2_weights) == is_localized(l2_weights));
        shared_intermediate_hidden = shared_l2_k;
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_n == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_k == hidden);
        DG_HOST_ASSERT(shared_l2_n == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kFloat8_e4m3fn);
        check_sf_layout(shared_l1_weights_sf, shared_intermediate_hidden * 2, hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
        check_sf_layout(shared_l2_weights_sf, hidden, shared_intermediate_hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mono_ep(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        weight_dtype == torch::kFloat8_e4m3fn ? "fp8xfp8" : "fp8xfp4",
        activation, num_shared_experts
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_fp8_fp4_mono_ep(y,
                               l1_acts, l1_acts_sf,
                               l2_acts, l2_acts_sf,
                               shared_l1_acts, shared_l1_acts_sf,
                               shared_l2_acts, shared_l2_acts_sf,
                               l1_weights, l2_weights,
                               l1_weights_sf, l2_weights_sf,
                               shared_l1_weights, shared_l2_weights,
                               shared_l1_weights_sf, shared_l2_weights_sf,
                               locality_domain::get_balanced_sm_locality_domains(),
                               cumulative_local_expert_recv_stats,
                               sym_buffer_ptrs,
                               rank_idx, num_max_tokens_per_rank,
                               num_experts_per_rank,
                               num_shared_experts,
                               num_tokens, num_topk,
                               hidden, intermediate_hidden,
                               activation_clamp, fast_math, saved_z);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void bf16_mono_ep(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor>& shared_l1_weights_opt,
    const std::optional<torch::Tensor>& shared_l2_weights_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const std::optional<torch::Tensor>& saved_z,
    const int& num_ring_tokens
) {
    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(num_ring_tokens >= 0);
    DG_HOST_ASSERT(num_ring_tokens == 0 or jit->device.get_arch_major() >= 10);
    DG_HOST_ASSERT(shared_l1_weights_opt.has_value() == shared_l2_weights_opt.has_value());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = check_weights_layout_3d(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = check_weights_layout_3d(l2_weights);
    DG_HOST_ASSERT(is_localized(l1_weights) == is_localized(l2_weights));
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);

    if (saved_z.has_value()) {
        DG_HOST_ASSERT(not shared_l1_weights_opt.has_value() and not activation_clamp_opt.has_value());
        DG_HOST_ASSERT(saved_z->is_cuda() and saved_z->device() == y.device() and saved_z->is_contiguous());
        DG_HOST_ASSERT(saved_z->scalar_type() == torch::kBFloat16);
        const auto [ranks, tokens, slots, width] = get_shape<4>(*saved_z);
        DG_HOST_ASSERT(ranks == sym_buffer_ptrs.size() and tokens == num_max_tokens_per_rank);
        DG_HOST_ASSERT(slots == num_topk and width == 2 * intermediate_hidden);
    }

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l2_weights;
    if (shared_l1_weights_opt.has_value()) {
        shared_l1_weights = shared_l1_weights_opt.value();
        shared_l2_weights = shared_l2_weights_opt.value();
        const auto [shared_l1_n, shared_l1_k] = check_weights_layout_2d(shared_l1_weights);
        const auto [shared_l2_n, shared_l2_k] = check_weights_layout_2d(shared_l2_weights);
        DG_HOST_ASSERT(is_localized(shared_l1_weights) == is_localized(l1_weights));
        DG_HOST_ASSERT(is_localized(shared_l2_weights) == is_localized(l2_weights));
        shared_intermediate_hidden = shared_l2_k;
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_n == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_k == hidden);
        DG_HOST_ASSERT(shared_l2_n == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kBFloat16);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mono_ep(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "bf16xbf16", activation, num_shared_experts, 0, num_ring_tokens
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, _x_sf, topk_idx, topk_weights,
                shared_l1_acts, _shared_l1_acts_sf, shared_l2_acts, _shared_l2_acts_sf,
                l1_acts, _l1_acts_sf, l2_acts, _l2_acts_sf] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 9) {
        const auto hopper_domains = torch::zeros(
            {jit->device.get_num_sms()}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        sm90_bf16_mono_ep(y,
                           l1_acts, l2_acts,
                           shared_l1_acts, shared_l2_acts,
                           l1_weights, l2_weights,
                           shared_l1_weights, shared_l2_weights,
                           hopper_domains,
                           cumulative_local_expert_recv_stats,
                           sym_buffer_ptrs,
                           rank_idx, num_max_tokens_per_rank,
                           num_experts_per_rank,
                           num_shared_experts,
                           num_tokens, num_topk,
                           hidden, intermediate_hidden,
                           activation_clamp, fast_math, saved_z);
    } else if (arch_major == 10) {
        sm100_bf16_mono_ep(y,
                            l1_acts, l2_acts,
                            shared_l1_acts, shared_l2_acts,
                            l1_weights, l2_weights,
                            shared_l1_weights, shared_l2_weights,
                            locality_domain::get_balanced_sm_locality_domains(),
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_shared_experts,
                            num_tokens, num_topk,
                            hidden, intermediate_hidden,
                            activation_clamp, fast_math, saved_z);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

// Persistent NVFP4 forward on packed E2M1 experts (SM100 only)
// `l1_weights`: `[E, 2I, H / 2]` packed FP4 with granularity-8 interleaved gate/up rows, `l2_weights`: `[E, H, I / 2]`
// `*_weights_sf`: `[E, N / 128, K / 64, 128]` int32 UTCCP atoms of UE4M3 scales
// `expert_scales`: `[E, 8]` fp32 `{1 / (6 pts_x), 1 / pts_x, gate fold, up fold, 1 / (6 pts_h), 1 / pts_h, L2 fold, 0}`
static void nvfp4_mono_ep(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& expert_scales,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 and y.is_contiguous());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    const auto arch_major = jit->device.get_arch_major();
    DG_HOST_ASSERT(l1_weights.dim() == 3 and l2_weights.dim() == 3);
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = check_weights_layout_3d(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = check_weights_layout_3d(l2_weights);
    DG_HOST_ASSERT(l1_weights.scalar_type() == kPackedFP4 and l2_weights.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(hidden % 256 == 0 and intermediate_hidden % 256 == 0);
    DG_HOST_ASSERT(expert_scales.scalar_type() == torch::kFloat and expert_scales.is_contiguous());
    DG_HOST_ASSERT(expert_scales.dim() == 2 and expert_scales.size(0) == num_experts_per_rank and expert_scales.size(1) == 8);

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mono_ep(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "nvfp4", activation, 0
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, _x_sf, topk_idx, topk_weights,
                _shared_l1_acts, _shared_l1_acts_sf, _shared_l2_acts, _shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    if (arch_major == 10) {
        sm100_nvfp4_mono_ep(y,
                             l1_acts, l1_acts_sf,
                             l2_acts, l2_acts_sf,
                             l1_weights, l1_weights_sf,
                             l2_weights, l2_weights_sf,
                             expert_scales,
                             locality_domain::get_balanced_sm_locality_domains(),
                             cumulative_local_expert_recv_stats,
                             sym_buffer_ptrs,
                             rank_idx, num_max_tokens_per_rank,
                             num_experts_per_rank,
                             num_tokens, num_topk,
                             hidden, intermediate_hidden,
                             activation_clamp, fast_math);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void register_apis(pybind11::module_& m) {
    m.def("get_token_alignment_for_mono_ep", &get_token_alignment_for_mono_ep);
    m.def("get_block_m_for_mono_ep", &get_block_m_for_mono_ep);
    m.def("get_mono_ep_backward_combine_views", &get_mono_ep_backward_combine_views);
    m.def("get_symm_buffer_size_for_mono_ep", &get_symm_buffer_size_for_mono_ep);
    m.def("fp8_fp4_mono_ep", &fp8_fp4_mono_ep);
    m.def("bf16_mono_ep", &bf16_mono_ep);
    m.def("nvfp4_mono_ep", &nvfp4_mono_ep);
    m.def("bf16_mono_ep_backward", &bf16_mono_ep_backward);
    m.def("nvfp4_mono_ep_backward", &nvfp4_mono_ep_backward);
    m.def("bf16_mono_ep_backward_dw", &bf16_mono_ep_backward_dw);
    m.def("mxfp8_mono_ep_backward", &mxfp8_mono_ep_backward);
    m.def("bf16_mono_ep_backward_sm90", &bf16_mono_ep_backward_sm90);
    m.def("fp8_mono_ep_backward_sm90", &fp8_mono_ep_backward_sm90,
          py::arg("dx"), py::arg("ds"), py::arg("w1"), py::arg("w2t"), py::arg("w1t"),
          py::arg("scales"), py::arg("pools"), py::arg("starts"), py::arg("counts"),
          py::arg("sym_buffer"), py::arg("sym_ptrs"), py::arg("dy"), py::arg("dy_ptrs"),
          py::arg("ds_partials"), py::arg("ds_ptrs"), py::arg("rank"), py::arg("max_tokens"),
          py::arg("ring_tokens"), py::arg("topk"), py::arg("padded_counts") = py::none(),
          py::arg("dy_sf") = py::none(), py::arg("dy_sf_ptrs") = std::vector<int64_t>{},
          py::arg("fp8_combine") = false);
    m.def("fp8_mono_ep_forward_sm90", &fp8_mono_ep_forward_sm90,
          py::arg("y"), py::arg("w1"), py::arg("w2"), py::arg("scales"),
          py::arg("sym_buffer"), py::arg("sym_ptrs"), py::arg("rank"),
          py::arg("max_tokens"), py::arg("ring_tokens"), py::arg("topk"),
          py::arg("saved_z"), py::arg("original_w1_layout") = false);
}

} // namespace deep_gemm::mega
