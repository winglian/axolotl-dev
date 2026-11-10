// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <cstdlib>
#include <format>
#include <limits>
#include <string>
#include <torch/python.h>
#include <deep_gemm/layout/moe_backward.cuh>
#include "../../runtime/runtime.hpp"
#include "../heuristics/mono_ep.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// Default kernel variant bitmask for the SM100 BF16 persistent backward.
static constexpr uint32_t kDefaultBackwardOptions = 11;

static void bf16_mono_ep_backward(
    const torch::Tensor& dx, const torch::Tensor& ds,
    const torch::Tensor& w1, const torch::Tensor& w2t, const torch::Tensor& w1t,
    const std::vector<torch::Tensor>& pools,
    const torch::Tensor& starts, const torch::Tensor& counts,
    const torch::Tensor& sym_buffer, const std::vector<int64_t>& sym_ptrs,
    const torch::Tensor& dy, const std::vector<int64_t>& dy_ptrs,
    const torch::Tensor& ds_partials, const std::vector<int64_t>& ds_ptrs,
    int rank, int max_tokens, int ring_tokens, int topk) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(dx.is_cuda() and dx.dim() == 2);
    const auto check = [&](const torch::Tensor& t, torch::ScalarType dtype) {
        DG_HOST_ASSERT(t.is_cuda() and t.device() == dx.device() and t.is_contiguous());
        DG_HOST_ASSERT(t.scalar_type() == dtype);
    };
    check(dx, torch::kBFloat16);
    check(ds, torch::kFloat32);
    for (const auto& t : {w1, w2t, w1t, dy}) check(t, torch::kBFloat16);
    check(ds_partials, torch::kFloat32);
    check(starts, torch::kInt32);
    check(counts, torch::kInt32);
    check(sym_buffer, torch::kInt8);
    DG_HOST_ASSERT(w1.dim() == 3 and w2t.dim() == 3 and w1t.dim() == 3);
    const int experts = w1.size(0), hidden = w1.size(2), inner = w1.size(1) / 2;
    const int world = sym_ptrs.size(), tokens = dx.size(0);
    DG_HOST_ASSERT(world > 0 and world <= layout::kNumMaxRanks and rank >= 0 and rank < world);
    DG_HOST_ASSERT(dy_ptrs.size() == world and ds_ptrs.size() == world);
    DG_HOST_ASSERT(max_tokens > 0 and tokens <= max_tokens and ring_tokens > 0 and ring_tokens % 128 == 0);
    DG_HOST_ASSERT(topk > 0 and topk <= 32 and 32 % topk == 0);
    DG_HOST_ASSERT(experts > 0 and hidden > 0 and inner > 0 and hidden % 256 == 0 and inner % 256 == 0);
    DG_HOST_ASSERT(w1.size(1) == 2 * inner and dx.size(1) == hidden);
    DG_HOST_ASSERT(w2t.size(0) == experts and w2t.size(1) == inner and w2t.size(2) == hidden);
    DG_HOST_ASSERT(w1t.size(0) == experts and w1t.size(1) == hidden and w1t.size(2) == 2 * inner);
    DG_HOST_ASSERT(ds.dim() == 2 and ds.size(0) == tokens and ds.size(1) == topk);
    DG_HOST_ASSERT(dy.dim() == 2 and dy.size(0) == max_tokens and dy.size(1) == hidden);
    DG_HOST_ASSERT(ds_partials.dim() == 3 and ds_partials.size(0) == max_tokens and
                   ds_partials.size(1) == topk and ds_partials.size(2) == 2);
    DG_HOST_ASSERT(starts.dim() == 1 and counts.dim() == 1 and starts.numel() == experts and counts.numel() == experts);
    DG_HOST_ASSERT(sym_ptrs[rank] == reinterpret_cast<int64_t>(sym_buffer.data_ptr()));
    DG_HOST_ASSERT(dy_ptrs[rank] == reinterpret_cast<int64_t>(dy.data_ptr()));
    DG_HOST_ASSERT(ds_ptrs[rank] == reinterpret_cast<int64_t>(ds_partials.data_ptr()));
    DG_HOST_ASSERT(pools.size() == 4);
    const int widths[] = {hidden, hidden, inner, inner * 2};
    const int64_t required_rows = static_cast<int64_t>(world) * max_tokens * topk + experts * 128;
    DG_HOST_ASSERT(required_rows < std::numeric_limits<int>::max());
    const int64_t rows = pools[0].size(0);
    DG_HOST_ASSERT(rows >= required_rows and rows < std::numeric_limits<int>::max());
    for (int i = 0; i < 4; ++i) {
        check(pools[i], torch::kBFloat16);
        DG_HOST_ASSERT(pools[i].dim() == 2 and pools[i].size(0) == rows and pools[i].size(1) == widths[i]);
    }
    const auto buffer_layout = layout::MonoEPBuffer(nullptr, hidden, inner, world, experts * world,
        max_tokens, topk, ring_tokens, 0, false, 0);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(buffer_layout.get_num_bytes()));
    const auto acts = [&](const torch::Tensor& t) {
        return make_tma_2d_desc(t, t.size(1), rows, 64, 64, t.stride(0), 128);
    };
    const auto weights = [&](const torch::Tensor& t) {
        const int n = t.size(1), k = t.size(2);
        return make_tma_4d_desc(t, k, n / 2, 2, experts, 64, 128, 1, 1,
            k, static_cast<int64_t>(n / 2) * k, static_cast<int64_t>(n) * k, 128);
    };
    const auto tx = acts(pools[0]), ty = acts(pools[1]), tz = acts(pools[3]);
    const auto tw1 = weights(w1), tw2t = weights(w2t), tw1t = weights(w1t);
    const int sms = get_num_sms_for_mono_ep();
    DG_HOST_ASSERT(sms % 2 == 0);
    int pull = hidden * 2;
    while (pull > 4096) pull /= 2;
    // A dispatch pull buffer of two rows per warp (x and dy rows in one NVLink round trip, needs options bit 8).
    // Default when whole rows fit the one-row buffer and the exactly sized two-row storage keeps the stage count
    // (6 stages at the EP8 Qwen3-30B-A3B shape); MONO_EP_SM100_BACKWARD_PULL_BYTES=<one row> restores one row.
    const int one_row_pull = pull;
    const char* pull_selector = std::getenv("MONO_EP_SM100_BACKWARD_PULL_BYTES");
    const int pull_override = pull_selector == nullptr ? 0 : std::atoi(pull_selector);
    DG_HOST_ASSERT(pull_override == 0 or pull_override == pull or pull_override == hidden * 4);
    if (pull_override != 0) pull = pull_override;
    DG_HOST_ASSERT((hidden * 2 % pull == 0 or pull == hidden * 4) and pull % 16 == 0);
    const auto [stages, forward_smem] = get_pipeline_config_for_mono_ep(
        SM100ArchSpec::smem_capacity, experts * world, hidden, 128, 128, 64, pull,
        32, 32, 0, 0, 32, 4, 8, MmaKind::BF16);
    constexpr int backward_extra = 33 * 1024;
    // Pipeline shape: BLOCK_K 64 (default) or 128, and the TMA/UMMA ring depth. Unset depth takes the deepest ring
    // that fits shared memory (6 stages at BLOCK_K 64); MONO_EP_SM100_BACKWARD_STAGES=2 restores the old default.
    const char* block_k_selector = std::getenv("MONO_EP_SM100_BACKWARD_BLOCK_K");
    const int block_k = block_k_selector == nullptr ? 64 : std::atoi(block_k_selector);
    DG_HOST_ASSERT(block_k == 64 or block_k == 128);
    const int stage_bytes = (64 + 128) * block_k * 2 + 16;
    const int forward_stage_bytes = (64 + 128) * 64 * 2 + 16;
    const int fixed_smem = forward_smem - stages * forward_stage_bytes + backward_extra;
    // The estimate above over-counts the backward's fixed storage by ~1 KB, which costs the two-row pull buffer its
    // sixth stage; two-row pulls size the kernel's SharedStorage exactly instead (the kernel asserts it fits).
    const char* early_options_selector = std::getenv("MONO_EP_SM100_BACKWARD_OPTIONS");
    const bool four_tmem_stages = early_options_selector != nullptr and
        (std::strtoul(early_options_selector, nullptr, 10) & 4u);
    const auto exact_smem_for = [&](const int& n) {
        int offset = math::align(experts * world * 4, 1024) + 4 * pull;   // token counts, 4 dispatch pull buffers
        offset = math::align(offset, 1024) + 16384;                        // epilogue C/D staging
        offset = math::align(offset, 128) + 128 * 128 * 2 + 128 * 4;       // dH tile, dscore
        offset = math::align(offset, 1024) + n * 64 * block_k * 2;         // A stages
        offset = math::align(offset, 1024) + n * 128 * block_k * 2;        // B stages
        const int tmem_stages = four_tmem_stages ? 4 : 2;
        offset = math::align(offset, 16) + 2 * 32;                         // task infos
        offset += (4 + 2 * n + 2 * tmem_stages + 16 + 4) * 8 + 4;          // barriers, TMEM pointer
        return math::align(offset, 1024);
    };
    const auto smem_for = [&](const int& n) {
        return pull > hidden * 2 ? exact_smem_for(n) : math::align(fixed_smem + n * stage_bytes, 1024);
    };
    int max_stages = 2;
    while (max_stages < 8 and smem_for(max_stages + 1) <= SM100ArchSpec::smem_capacity) ++ max_stages;
    const char* default_options_selector = std::getenv("MONO_EP_SM100_BACKWARD_OPTIONS");
    const uint32_t selected_options = default_options_selector == nullptr ? kDefaultBackwardOptions :
        static_cast<uint32_t>(std::strtoul(default_options_selector, nullptr, 10));
    if (pull_override == 0 and one_row_pull == hidden * 2 and (selected_options & 8u)) {
        pull = hidden * 4;
        if (exact_smem_for(max_stages) > SM100ArchSpec::smem_capacity)
            pull = one_row_pull;
    }
    const char* stages_selector = std::getenv("MONO_EP_SM100_BACKWARD_STAGES");
    const int backward_stages = stages_selector == nullptr ? max_stages : std::atoi(stages_selector);
    DG_HOST_ASSERT(backward_stages >= 2 and backward_stages <= 8);
    const int smem = smem_for(backward_stages);
    DG_HOST_ASSERT(smem <= SM100ArchSpec::smem_capacity);
    // Kernel variant bitmask (see kOptions in the kernel); 0 restores the original epilogue/dispatch code.
    const char* options_selector = std::getenv("MONO_EP_SM100_BACKWARD_OPTIONS");
    const uint32_t options = options_selector == nullptr ? kDefaultBackwardOptions :
        static_cast<uint32_t>(std::strtoul(options_selector, nullptr, 10));
    DG_HOST_ASSERT(options < 512);
    DG_HOST_ASSERT(not (options & 64u) or (options & 1u));
    DG_HOST_ASSERT(pull <= hidden * 2 or (options & 8u));
    // 16 (lookahead pull) needs the overlapped copy and whole x/dy rows in the pull buffer
    DG_HOST_ASSERT(not (options & 16u) or ((options & 8u) and pull == hidden * 4));
    // Diagnostic only: a device pointer to a zeroed uint64 phase-trace buffer, and an epilogue ablation bitmask
    // (outputs become invalid). Unset leaves the default instantiation unchanged.
    const char* phase_trace_selector = std::getenv("MONO_EP_SM100_PHASE_TRACE");
    const uint64_t phase_trace_ptr = phase_trace_selector == nullptr ? 0 :
        std::strtoull(phase_trace_selector, nullptr, 10);
    const char* ablate_selector = std::getenv("MONO_EP_SM100_DIAG_ABLATE");
    const uint32_t diag_ablate = ablate_selector == nullptr ? 0 :
        static_cast<uint32_t>(std::strtoul(ablate_selector, nullptr, 10));
    const bool diag = phase_trace_ptr != 0 or diag_ablate != 0;
    const std::string diag_suffix = (diag or options != 0)
        ? std::format(", true, {}, {}u, {}u", phase_trace_ptr != 0 ? "true" : "false", diag_ablate, options) : std::string();
    const std::string diag_name = (diag
        ? std::format("_diag_t{}_a{}", phase_trace_ptr != 0 ? 1 : 0, diag_ablate) : std::string()) +
        (backward_stages != 2 ? std::format("_stages{}", backward_stages) : std::string()) +
        (block_k != 64 ? std::format("_bk{}", block_k) : std::string()) +
        (options != 0 ? std::format("_opt{}", options) : std::string()) +
        (pull > hidden * 2 ? std::format("_pull{}", pull) : std::string());
    const auto kernel = jit->compile("sm100_bf16_mono_ep_backward" + diag_name, std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mono_ep_backward.cuh>
static void instantiate() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm100_bf16_mono_ep_backward_impl<
        {}, {}, {}, {}, 0, {}, 128, 128, {}, 32, {}, {}, {}, 128, 128, 256, {}, {}{}>);
}}
)", max_tokens, hidden, inner, experts * world, topk, block_k, ring_tokens, backward_stages, pull, sms, world, diag_suffix));
    layout::MoEBackwardBuffers buffers{
        layout::SymBuffer<>(dy_ptrs, rank), layout::SymBuffer<>(ds_ptrs, rank),
        reinterpret_cast<__nv_bfloat16*>(pools[0].data_ptr()), reinterpret_cast<__nv_bfloat16*>(pools[1].data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(pools[2].data_ptr()), reinterpret_cast<__nv_bfloat16*>(pools[3].data_ptr()),
        ds.data_ptr<float>(), reinterpret_cast<uint32_t*>(starts.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(counts.data_ptr<int>())};
    jit->launch(kernel, {.num_smem_bytes = smem, .grid_dim = dim3(sms),
        .block_dim = dim3(512), .cluster_dim = dim3(2, 1, 1)},
        dx.data_ptr(), static_cast<int*>(nullptr), tokens, layout::SymBuffer<>(sym_ptrs, rank),
        tx, tw1, tx, tz, tw1t, tx, tw1, tx, tz, tw1t,
        static_cast<const uint8_t*>(nullptr), reinterpret_cast<__nv_bfloat16*>(phase_trace_ptr), ty, tw2t, buffers);
}

} // namespace deep_gemm
