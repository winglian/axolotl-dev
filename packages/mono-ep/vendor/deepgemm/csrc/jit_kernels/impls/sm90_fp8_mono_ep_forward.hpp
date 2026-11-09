// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <cstdlib>
#include <format>
#include <string>
#include <string_view>
#include <torch/python.h>
#include <deep_gemm/layout/mono_ep.cuh>
#include <deep_gemm/scheduler/mono_ep.cuh>
#include "../../runtime/runtime.hpp"
#include "../heuristics/mono_ep.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void fp8_mono_ep_forward_sm90(
    const torch::Tensor& y, const torch::Tensor& w1, const torch::Tensor& w2,
    const std::vector<torch::Tensor>& scales,
    const torch::Tensor& sym_buffer, const std::vector<int64_t>& sym_ptrs,
    int rank, int max_tokens, int ring_tokens, int topk,
    const std::optional<torch::Tensor>& saved_z,
    const bool original_w1_layout = false) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 9);
    DG_HOST_ASSERT(y.is_cuda() and y.dim() == 2);
    const auto check = [&](const torch::Tensor& t, torch::ScalarType dtype) {
        DG_HOST_ASSERT(t.is_cuda() and t.device() == y.device() and t.is_contiguous());
        DG_HOST_ASSERT(t.scalar_type() == dtype);
    };
    check(y, torch::kBFloat16);
    check(w1, torch::kFloat8_e4m3fn);
    check(w2, torch::kFloat8_e4m3fn);
    check(sym_buffer, torch::kInt8);
    DG_HOST_ASSERT(w1.dim() == 3 and w2.dim() == 3);
    const int experts = w1.size(0), hidden = w1.size(2), inner = w1.size(1) / 2;
    const int world = sym_ptrs.size(), tokens = y.size(0);
    DG_HOST_ASSERT(world > 0 and world <= layout::kNumMaxRanks and rank >= 0 and rank < world);
    DG_HOST_ASSERT(max_tokens > 0 and tokens <= max_tokens and ring_tokens > 0 and ring_tokens % 128 == 0);
    DG_HOST_ASSERT(topk > 0 and topk <= 32 and 32 % topk == 0);
    DG_HOST_ASSERT(experts > 0 and hidden > 0 and inner > 0 and hidden % 256 == 0 and inner % 256 == 0);
    DG_HOST_ASSERT(w1.size(1) == 2 * inner and y.size(1) == hidden);
    DG_HOST_ASSERT(w2.size(0) == experts and w2.size(1) == hidden and w2.size(2) == inner);
    DG_HOST_ASSERT(scales.size() == 2);
    const torch::Tensor weights_for_scales[] = {w1, w2};
    for (int i = 0; i < 2; ++i) {
        check(scales[i], torch::kFloat32);
        DG_HOST_ASSERT(scales[i].dim() == 3 and scales[i].size(0) == experts and
            scales[i].size(1) == weights_for_scales[i].size(1) / 128 and
            scales[i].size(2) == weights_for_scales[i].size(2) / 128);
    }
    DG_HOST_ASSERT(sym_ptrs[rank] == reinterpret_cast<int64_t>(sym_buffer.data_ptr()));
    if (saved_z.has_value()) {
        check(*saved_z, torch::kBFloat16);
        DG_HOST_ASSERT(saved_z->dim() == 4 and saved_z->size(0) == world and
            saved_z->size(1) == max_tokens and saved_z->size(2) == topk and saved_z->size(3) == 2 * inner);
    }
    const auto buffer_layout = layout::MonoEPBuffer(sym_buffer.data_ptr(), hidden, inner,
        world, experts * world, max_tokens, topk, ring_tokens, 0, false, 0, true);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(buffer_layout.get_num_bytes()));
    const auto options = y.options();
    const auto l1 = torch::from_blob(buffer_layout.l1_token_buffer.base, {ring_tokens, hidden},
        options.dtype(torch::kFloat8_e4m3fn));
    const auto l2 = torch::from_blob(buffer_layout.l2_token_buffer.base, {ring_tokens, inner}, options);
    const auto l2_cached = torch::from_blob(buffer_layout.l2_cached_token_buffer.base, {ring_tokens, inner},
        options.dtype(torch::kFloat8_e4m3fn));
    const auto tx = make_tma_2d_desc(l1, hidden, ring_tokens, 128, 128, hidden, 128);
    const auto th = make_tma_2d_desc(l2, inner, ring_tokens, 64, 128, inner, 128);
    const auto th_cached = make_tma_2d_desc(l2_cached, inner, ring_tokens, 128, 128, inner, 128);
    const auto th_out = make_tma_2d_desc(l2, inner, ring_tokens, 64, 32, inner, 128);
    const auto weights = [&](const torch::Tensor& t, const int load_n) {
        const int n = t.size(1), k = t.size(2);
        return make_tma_4d_desc(t, k, n / 2, 2, experts, 128, load_n, 1, 1,
            k, static_cast<int64_t>(n / 2) * k, static_cast<int64_t>(n) * k, 128);
    };
    const auto tw1 = weights(w1, original_w1_layout ? 64 : 128), tw2 = weights(w2, 128);
    const char* compact_selector = std::getenv("MONO_EP_SM90_FP8_COMPACT_FORWARD");
    DG_HOST_ASSERT(compact_selector == nullptr or std::string_view(compact_selector) == "0" or
        std::string_view(compact_selector) == "1");
    const bool compact_non_math =
        compact_selector != nullptr and std::string_view(compact_selector) == "1";
    const char* timeout_selector = std::getenv("MONO_EP_SM90_FP8_TRAP_TIMEOUTS");
    DG_HOST_ASSERT(timeout_selector == nullptr or std::string_view(timeout_selector) == "0" or
        std::string_view(timeout_selector) == "1");
    const bool trap_timeouts =
        timeout_selector != nullptr and std::string_view(timeout_selector) == "1";
    if (compact_non_math) {
        DG_HOST_ASSERT(world <= 32 and experts <= 64);
    }
    // Diagnostic only: a device pointer to a zeroed uint64 phase-trace buffer (sms x 1088 entries).
    const char* phase_trace_selector = std::getenv("MONO_EP_SM90_FP8_FORWARD_PHASE_TRACE");
    const uint64_t phase_trace_ptr = phase_trace_selector == nullptr ? 0 :
        std::strtoull(phase_trace_selector, nullptr, 10);
    // Diagnostic only: bitmask removing epilogue constructs to time them (outputs become invalid).
    const char* ablate_selector = std::getenv("MONO_EP_SM90_FP8_FORWARD_DIAG_ABLATE");
    const uint32_t diag_ablate = ablate_selector == nullptr ? 0 :
        static_cast<uint32_t>(std::strtoul(ablate_selector, nullptr, 10));
    const char* vector_scatter_selector = std::getenv("MONO_EP_SM90_FP8_FORWARD_VECTOR_SCATTER");
    DG_HOST_ASSERT(vector_scatter_selector == nullptr or std::string_view(vector_scatter_selector) == "0" or
        std::string_view(vector_scatter_selector) == "1");
    const bool vector_scatter =
        vector_scatter_selector != nullptr and std::string_view(vector_scatter_selector) == "1";
    const char* defer_fence_selector = std::getenv("MONO_EP_SM90_FP8_FORWARD_DEFER_COMBINE_FENCE");
    DG_HOST_ASSERT(defer_fence_selector == nullptr or std::string_view(defer_fence_selector) == "0" or
        std::string_view(defer_fence_selector) == "1");
    const bool defer_fence =
        defer_fence_selector != nullptr and std::string_view(defer_fence_selector) == "1";
    const char* vector_swiglu_selector = std::getenv("MONO_EP_SM90_FP8_FORWARD_VECTOR_SWIGLU");
    DG_HOST_ASSERT(vector_swiglu_selector == nullptr or std::string_view(vector_swiglu_selector) == "0" or
        std::string_view(vector_swiglu_selector) == "1");
    const bool vector_swiglu =
        vector_swiglu_selector != nullptr and std::string_view(vector_swiglu_selector) == "1";
    const char* vector_hquant_selector = std::getenv("MONO_EP_SM90_FP8_FORWARD_VECTOR_HQUANT");
    DG_HOST_ASSERT(vector_hquant_selector == nullptr or std::string_view(vector_hquant_selector) == "0" or
        std::string_view(vector_hquant_selector) == "1");
    const bool vector_hquant =
        vector_hquant_selector != nullptr and std::string_view(vector_hquant_selector) == "1";
    const char* swizzle_selector = std::getenv("MONO_EP_SM90_FP8_FORWARD_SWIZZLE_ACCUM_SMEM");
    DG_HOST_ASSERT(swizzle_selector == nullptr or std::string_view(swizzle_selector) == "0" or
        std::string_view(swizzle_selector) == "1");
    const bool swizzle_accum = swizzle_selector != nullptr and std::string_view(swizzle_selector) == "1";
    DG_HOST_ASSERT(not swizzle_accum or (vector_scatter and vector_swiglu));
    const bool diag = phase_trace_ptr != 0 or diag_ablate != 0 or vector_scatter or defer_fence or
        vector_swiglu or vector_hquant or swizzle_accum;
    const std::string diag_suffix = diag
        ? std::format(", {}, {}u, {}, {}, {}, {}, {}", phase_trace_ptr != 0 ? "true" : "false", diag_ablate,
            vector_scatter ? "true" : "false", defer_fence ? "true" : "false",
            vector_swiglu ? "true" : "false", vector_hquant ? "true" : "false",
            swizzle_accum ? "true" : "false") : std::string();
    const int dispatch_threads = compact_non_math ? 32 : 128;
    const int non_epilogue_threads = compact_non_math ? 96 : 128;
    const int block_threads = compact_non_math ? 384 : 512;
    check_ring_capacity_for_mono_ep(
        ring_tokens, 128, world, experts, max_tokens, topk, hidden, inner);
    const int sms = get_num_sms_for_mono_ep();
    int pull = 4096;
    while (hidden % pull != 0) pull /= 2;
    DG_HOST_ASSERT(hidden % pull == 0 and pull % 16 == 0);
    const auto kernel = jit->compile(
        std::format("sm90_fp8_mono_ep_forward_{}_{}_{}{}{}",
            compact_non_math ? "compact384" : "baseline512",
            trap_timeouts ? "trap" : "diagnostic",
            original_w1_layout ? "original_w1" : "interleaved_w1",
            phase_trace_ptr != 0 ? "_phasetrace" : "",
            (diag_ablate != 0 ? std::format("_ablate{}", diag_ablate) : std::string()) +
                (vector_scatter ? "_vecscatter" : "") + (defer_fence ? "_deferfence" : "") +
                (vector_swiglu ? "_vecswiglu" : "") + (vector_hquant ? "_vechquant" : "") +
                (swizzle_accum ? "_swizzleacc" : "")),
        std::format(R"({}
#include <deep_gemm/impls/sm90_fp8_mono_ep_forward.cuh>
static void instantiate() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mono_ep_forward_impl<
        {}, {}, {}, {}, 0, {}, 128, 128, 128, 32, {}, 2, {}, {}, {}, 256, {}, {}, 0x7f7fffffu, false, {}, {}{}>);
}}
)", trap_timeouts ? "#define DG_TRAP_ONLY_TIMEOUTS 1" : "",
            max_tokens, hidden, inner, experts * world, topk, ring_tokens, pull,
            dispatch_threads, non_epilogue_threads, sms, world,
            compact_non_math ? "true" : "false", original_w1_layout ? "true" : "false", diag_suffix));
    jit->launch(kernel, {.num_smem_bytes = SM90ArchSpec::smem_capacity, .grid_dim = dim3(sms),
        .block_dim = dim3(block_threads), .cluster_dim = dim3(1, 1, 1)},
        y.data_ptr(), reinterpret_cast<int*>(phase_trace_ptr), tokens, layout::SymBuffer<>(sym_ptrs, rank),
        tx, tw1, th_out, th, th_cached, tw2, tx, tw1, th_out, th, tw2,
        static_cast<const uint8_t*>(nullptr), saved_z.has_value() ? saved_z->data_ptr() : nullptr,
        scales[0].data_ptr<float>(), scales[1].data_ptr<float>());
}

} // namespace deep_gemm
