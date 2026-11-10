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

// Kernel variant bitmask for the SM100 NVFP4 persistent backward: 1 staged h/dz pool stores, 2 TMEM release
// before the recompute/dH math. The dispatch always pulls whole x and dy rows with lookahead.
static constexpr uint32_t kDefaultNvfp4BackwardOptions = 3;

// Persistent NVFP4 LoRA backward: BF16 x/dy rows are pulled, stored to the BF16 pools and quantized into packed
// FP4 pools (x with the expert's static scale, dy and dz per row); the three GEMM phases run `kind::mxf4nvf4`
// on the requant cache's merged weights (`w1` interleaved, `w2t`, `w1t`) and their UTCCP scale atoms.
static void nvfp4_mono_ep_backward(
    const torch::Tensor& dx, const torch::Tensor& ds,
    const torch::Tensor& w1, const torch::Tensor& w1_sf,
    const torch::Tensor& w2t, const torch::Tensor& w2t_sf,
    const torch::Tensor& w1t, const torch::Tensor& w1t_sf,
    const torch::Tensor& expert_scales,
    const std::vector<torch::Tensor>& pools,
    const std::vector<torch::Tensor>& fp4_pools,
    const std::vector<torch::Tensor>& sf_pools,
    const std::vector<torch::Tensor>& row_scales,
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
    check(dy, torch::kBFloat16);
    for (const auto& t : {w1, w2t, w1t}) check(t, kPackedFP4);
    for (const auto& t : {w1_sf, w2t_sf, w1t_sf}) check(t, torch::kInt);
    check(expert_scales, torch::kFloat32);
    check(ds_partials, torch::kFloat32);
    check(starts, torch::kInt32);
    check(counts, torch::kInt32);
    check(sym_buffer, torch::kInt8);
    DG_HOST_ASSERT(w1.dim() == 3 and w2t.dim() == 3 and w1t.dim() == 3);
    const int experts = w1.size(0), hidden = w1.size(2) * 2, inner = w1.size(1) / 2;
    const int world = sym_ptrs.size(), tokens = dx.size(0);
    DG_HOST_ASSERT(world > 0 and world <= layout::kNumMaxRanks and rank >= 0 and rank < world);
    DG_HOST_ASSERT(dy_ptrs.size() == world and ds_ptrs.size() == world);
    DG_HOST_ASSERT(max_tokens > 0 and tokens <= max_tokens and ring_tokens > 0 and ring_tokens % 128 == 0);
    DG_HOST_ASSERT(topk > 0 and topk <= 32 and 32 % topk == 0);
    DG_HOST_ASSERT(experts > 0 and hidden > 0 and inner > 0 and hidden % 256 == 0 and inner % 256 == 0);
    DG_HOST_ASSERT(2 * inner <= 64 * 128);  // one readiness mask bit per recompute N block
    DG_HOST_ASSERT(w1.size(1) == 2 * inner and dx.size(1) == hidden);
    DG_HOST_ASSERT(w2t.size(0) == experts and w2t.size(1) == inner and w2t.size(2) == hidden / 2);
    DG_HOST_ASSERT(w1t.size(0) == experts and w1t.size(1) == hidden and w1t.size(2) == inner);
    DG_HOST_ASSERT(expert_scales.dim() == 2 and expert_scales.size(0) == experts and expert_scales.size(1) == 8);
    DG_HOST_ASSERT(ds.dim() == 2 and ds.size(0) == tokens and ds.size(1) == topk);
    DG_HOST_ASSERT(dy.dim() == 2 and dy.size(0) == max_tokens and dy.size(1) == hidden);
    DG_HOST_ASSERT(ds_partials.dim() == 3 and ds_partials.size(0) == max_tokens and
                   ds_partials.size(1) == topk and ds_partials.size(2) == 2);
    DG_HOST_ASSERT(starts.dim() == 1 and counts.dim() == 1 and starts.numel() == experts and counts.numel() == experts);
    DG_HOST_ASSERT(sym_ptrs[rank] == reinterpret_cast<int64_t>(sym_buffer.data_ptr()));
    DG_HOST_ASSERT(dy_ptrs[rank] == reinterpret_cast<int64_t>(dy.data_ptr()));
    DG_HOST_ASSERT(ds_ptrs[rank] == reinterpret_cast<int64_t>(ds_partials.data_ptr()));

    // Pools: BF16 x, dy, h, dz; packed FP4 x, dy, dz; `[K / 64][rows]` scale words; per-row dy and dz scales
    DG_HOST_ASSERT(pools.size() == 4 and fp4_pools.size() == 3 and sf_pools.size() == 3 and row_scales.size() == 2);
    const int widths[] = {hidden, hidden, inner, inner * 2};
    const int64_t required_rows = static_cast<int64_t>(world) * max_tokens * topk + experts * 128;
    DG_HOST_ASSERT(required_rows < std::numeric_limits<int>::max());
    const int64_t rows = pools[0].size(0);
    DG_HOST_ASSERT(rows >= required_rows and rows % 128 == 0 and rows < std::numeric_limits<int>::max());
    for (int i = 0; i < 4; ++i) {
        check(pools[i], torch::kBFloat16);
        DG_HOST_ASSERT(pools[i].dim() == 2 and pools[i].size(0) == rows and pools[i].size(1) == widths[i]);
    }
    const int fp4_widths[] = {hidden, hidden, inner * 2};
    for (int i = 0; i < 3; ++i) {
        check(fp4_pools[i], kPackedFP4);
        DG_HOST_ASSERT(fp4_pools[i].dim() == 2 and fp4_pools[i].size(0) == rows and fp4_pools[i].size(1) == fp4_widths[i] / 2);
        check(sf_pools[i], torch::kInt);
        DG_HOST_ASSERT(sf_pools[i].dim() == 2 and sf_pools[i].size(0) == fp4_widths[i] / 64 and sf_pools[i].size(1) == rows);
    }
    for (int i = 0; i < 2; ++i) {
        check(row_scales[i], torch::kFloat32);
        DG_HOST_ASSERT(row_scales[i].dim() == 1 and row_scales[i].size(0) == rows);
    }
    const auto buffer_layout = layout::MonoEPBuffer(nullptr, hidden, inner, world, experts * world,
        max_tokens, topk, ring_tokens, 0, false, 0);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(buffer_layout.get_num_bytes()));

    // Tensor maps: packed FP4 pools and weights (coordinates count FP4 elements), MN-major scale words
    constexpr int kBlockK = 256, kBlockM = 128, kBlockN = 128, kGranK = 16;
    const auto acts = [&](const torch::Tensor& t, const int& k) {
        return make_tma_2d_desc(t, k, rows, kBlockK, kBlockM / 2, static_cast<int>(t.stride(0)), 128, 0, false,
                                /*fp4_unpacked_smem=*/ false);
    };
    const auto acts_sf = [&](const torch::Tensor& t, const int& k) {
        return make_tma_sf_desc(cute::UMMA::Major::MN, t, rows, k, kBlockM, kGranK, 1, 0, 0, false,
                                kBlockK / (kGranK * 4), static_cast<int>(t.stride(0)));
    };
    const auto weights = [&](const torch::Tensor& t) {
        return make_tma_weights_3d_desc(t, kBlockK, kBlockN, 128, /*fp4_unpacked_smem=*/ false);
    };
    const auto weights_sf = [&](const torch::Tensor& sf, const int& shape_n, const int& shape_k) {
        const auto num_n_blocks = shape_n / 128, num_k_groups = shape_k / 64;
        DG_HOST_ASSERT(sf.dim() == 4 and sf.size(0) == experts and sf.size(1) == num_n_blocks);
        DG_HOST_ASSERT(sf.size(2) == num_k_groups and sf.size(3) == 128);
        return make_tma_3d_desc(sf, 128, num_k_groups, experts * num_n_blocks,
                                128, kBlockK / 64, 1, 128, num_k_groups * 128, 0);
    };
    const auto tx = acts(fp4_pools[0], hidden), ty = acts(fp4_pools[1], hidden), tz = acts(fp4_pools[2], inner * 2);
    const auto tx_sf = acts_sf(sf_pools[0], hidden), ty_sf = acts_sf(sf_pools[1], hidden), tz_sf = acts_sf(sf_pools[2], inner * 2);
    const auto tw1 = weights(w1), tw2t = weights(w2t), tw1t = weights(w1t);
    const auto tw1_sf = weights_sf(w1_sf, inner * 2, hidden);
    const auto tw2t_sf = weights_sf(w2t_sf, inner, hidden);
    const auto tw1t_sf = weights_sf(w1t_sf, hidden, inner * 2);

    const int sms = get_num_sms_for_mono_ep();
    DG_HOST_ASSERT(sms % 2 == 0);
    // Whole x and dy rows per pull (two BF16 rows per dispatch warp)
    const int pull = hidden * 4;
    const char* options_selector = std::getenv("MONO_EP_SM100_NVFP4_BACKWARD_OPTIONS");
    const uint32_t options = options_selector == nullptr ? kDefaultNvfp4BackwardOptions :
        static_cast<uint32_t>(std::strtoul(options_selector, nullptr, 10));
    DG_HOST_ASSERT((options & ~3u) == 0);
    // Exact `SharedStorage` size of the kernel for `n` TMA/UMMA stages (asserted by the kernel at run time)
    const auto exact_smem_for = [&](const int& n) {
        int offset = math::align(experts * world * 4, 1024) + 4 * pull;   // token counts, 4 dispatch pull buffers
        offset = math::align(offset, 1024) + 16384;                        // epilogue C/D staging
        offset = math::align(offset, 128) + 128 * 128 * 2 + 128 * 4;       // dH tile, dscore
        offset = math::align(offset, 1024) + n * 64 * kBlockK / 2;         // A stages (packed FP4)
        offset = math::align(offset, 1024) + n * 128 * kBlockK / 2;        // B stages (packed FP4)
        offset += n * 4 * (128 + 128) * 4;                                 // SFA + SFB words
        offset = math::align(offset, 16) + 2 * 32;                         // task infos
        offset += (4 + 2 * n + 2 * 2 + 16 + 4) * 8 + 4;                    // barriers, TMEM pointer
        return math::align(offset, 1024);
    };
    int max_stages = 2;
    while (max_stages < 8 and exact_smem_for(max_stages + 1) <= SM100ArchSpec::smem_capacity) ++ max_stages;
    const char* stages_selector = std::getenv("MONO_EP_SM100_NVFP4_BACKWARD_STAGES");
    const int stages = stages_selector == nullptr ? max_stages : std::atoi(stages_selector);
    DG_HOST_ASSERT(stages >= 2 and stages <= max_stages);
    const int smem = exact_smem_for(stages);
    DG_HOST_ASSERT(smem <= SM100ArchSpec::smem_capacity);
    // Diagnostic only: a device pointer to a zeroed uint64 phase-trace buffer
    const char* phase_trace_selector = std::getenv("MONO_EP_SM100_PHASE_TRACE");
    const uint64_t phase_trace_ptr = phase_trace_selector == nullptr ? 0 :
        std::strtoull(phase_trace_selector, nullptr, 10);
    const std::string name = std::format("sm100_nvfp4_mono_ep_backward_stages{}_opt{}{}", stages, options,
        phase_trace_ptr != 0 ? "_trace" : "");
    const auto kernel = jit->compile(name, std::format(R"(
#include <deep_gemm/impls/sm100_nvfp4_mono_ep_backward.cuh>
static void instantiate() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm100_nvfp4_mono_ep_backward_impl<
        {}, {}, {}, {}, 0, {}, 128, 128, {}, 32, 128, 128, {}, {}, {}, 128, 128, 256, {}, {}, true, {}, 0u, {}u>);
}}
)", max_tokens, hidden, inner, experts * world, topk, kBlockK, ring_tokens, stages, pull, sms, world,
        phase_trace_ptr != 0 ? "true" : "false", options));
    layout::MoEBackwardBuffers buffers{
        layout::SymBuffer<>(dy_ptrs, rank), layout::SymBuffer<>(ds_ptrs, rank),
        reinterpret_cast<__nv_bfloat16*>(pools[0].data_ptr()), reinterpret_cast<__nv_bfloat16*>(pools[1].data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(pools[2].data_ptr()), reinterpret_cast<__nv_bfloat16*>(pools[3].data_ptr()),
        ds.data_ptr<float>(), reinterpret_cast<uint32_t*>(starts.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(counts.data_ptr<int>())};
    buffers.qx_pool = static_cast<uint8_t*>(fp4_pools[0].data_ptr());
    buffers.qdy_pool = static_cast<uint8_t*>(fp4_pools[1].data_ptr());
    buffers.qdz_pool = static_cast<uint8_t*>(fp4_pools[2].data_ptr());
    buffers.nv_sx = reinterpret_cast<uint32_t*>(sf_pools[0].data_ptr<int>());
    buffers.nv_sdy = reinterpret_cast<uint32_t*>(sf_pools[1].data_ptr<int>());
    buffers.nv_sdz = reinterpret_cast<uint32_t*>(sf_pools[2].data_ptr<int>());
    buffers.nv_rdy = row_scales[0].data_ptr<float>();
    buffers.nv_rdz = row_scales[1].data_ptr<float>();
    buffers.nv_pool_rows = static_cast<uint32_t>(rows);
    jit->launch(kernel, {.num_smem_bytes = smem, .grid_dim = dim3(sms),
        .block_dim = dim3(512), .cluster_dim = dim3(2, 1, 1)},
        dx.data_ptr(), static_cast<int*>(nullptr), tokens, layout::SymBuffer<>(sym_ptrs, rank),
        tx, tx_sf, tw1, tw1_sf, tz, tz_sf, tw1t, tw1t_sf, ty, ty_sf, tw2t, tw2t_sf,
        expert_scales.data_ptr<float>(),
        static_cast<const uint8_t*>(nullptr), reinterpret_cast<__nv_bfloat16*>(phase_trace_ptr), buffers);
}

} // namespace deep_gemm
