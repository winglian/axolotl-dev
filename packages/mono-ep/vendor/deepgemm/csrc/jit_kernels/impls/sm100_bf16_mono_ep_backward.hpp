// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <format>
#include <limits>
#include <torch/python.h>
#include <deep_gemm/layout/moe_backward.cuh>
#include "../../runtime/runtime.hpp"
#include "../heuristics/mono_ep.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

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
    DG_HOST_ASSERT(hidden * 2 % pull == 0 and pull % 16 == 0);
    const auto [stages, forward_smem] = get_pipeline_config_for_mono_ep(
        SM100ArchSpec::smem_capacity, experts * world, hidden, 128, 128, 64, pull,
        32, 32, 0, 0, 32, 4, 8, MmaKind::BF16);
    constexpr int backward_extra = 33 * 1024;
    constexpr int stage_bytes = (64 + 128) * 64 * 2 + 16;
    const int smem = math::align(forward_smem - (stages - 2) * stage_bytes + backward_extra, 1024);
    DG_HOST_ASSERT(smem <= SM100ArchSpec::smem_capacity);
    const auto kernel = jit->compile("sm100_bf16_mono_ep_backward", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mono_ep_backward.cuh>
static void instantiate() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm100_bf16_mono_ep_backward_impl<
        {}, {}, {}, {}, 0, {}, 128, 128, 64, 32, {}, 2, {}, 128, 128, 256, {}, {}>);
}}
)", max_tokens, hidden, inner, experts * world, topk, ring_tokens, pull, sms, world));
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
        static_cast<const uint8_t*>(nullptr), static_cast<__nv_bfloat16*>(nullptr), ty, tw2t, buffers);
}

} // namespace deep_gemm
