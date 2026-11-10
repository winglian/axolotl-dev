// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <cstdio>
#include <cstdlib>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <torch/python.h>
#include <deep_gemm/layout/moe_backward.cuh>
#include <deep_gemm/scheduler/mono_ep.cuh>
#include "../../runtime/runtime.hpp"
#include "../heuristics/mono_ep.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"
#include "mono_ep_activation_selector.hpp"

namespace deep_gemm {

static void fp8_mono_ep_backward_sm90(
    const torch::Tensor& dx, const torch::Tensor& ds,
    const torch::Tensor& w1, const torch::Tensor& w2t, const torch::Tensor& w1t,
    const std::vector<torch::Tensor>& scales,
    const std::vector<torch::Tensor>& pools,
    const torch::Tensor& starts, const torch::Tensor& counts,
    const torch::Tensor& sym_buffer, const std::vector<int64_t>& sym_ptrs,
    const torch::Tensor& dy, const std::vector<int64_t>& dy_ptrs,
    const torch::Tensor& ds_partials, const std::vector<int64_t>& ds_ptrs,
    int rank, int max_tokens, int ring_tokens, int topk,
    const std::optional<torch::Tensor>& padded_counts = std::nullopt,
    const std::optional<torch::Tensor>& dy_sf = std::nullopt,
    const std::vector<int64_t>& dy_sf_ptrs = {},
    const bool fp8_combine = false) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 9);
    DG_HOST_ASSERT(dx.is_cuda() and dx.dim() == 2);
    const auto check = [&](const torch::Tensor& t, torch::ScalarType dtype) {
        DG_HOST_ASSERT(t.is_cuda() and t.device() == dx.device() and t.is_contiguous());
        DG_HOST_ASSERT(t.scalar_type() == dtype);
    };
    check(dx, torch::kBFloat16);
    check(ds, torch::kFloat32);
    const bool transport_fp8 = dy_sf.has_value();
    DG_HOST_ASSERT(not fp8_combine or transport_fp8);
    check(dy, transport_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16);
    for (const auto& t : {w1, w2t, w1t}) check(t, torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(scales.size() == 3);
    for (const auto& t : scales) check(t, torch::kFloat32);
    check(ds_partials, torch::kFloat32);
    check(starts, torch::kInt32);
    check(counts, torch::kInt32);
    check(sym_buffer, torch::kInt8);
    DG_HOST_ASSERT(w1.dim() == 3 and w2t.dim() == 3 and w1t.dim() == 3);
    const int experts = w1.size(0), hidden = w1.size(2), inner = w1.size(1) / 2;
    const int world = sym_ptrs.size(), tokens = dx.size(0);
    DG_HOST_ASSERT(world > 0 and world <= layout::kNumMaxRanks and rank >= 0 and rank < world);
    DG_HOST_ASSERT(dy_ptrs.size() == world and ds_ptrs.size() == world);
    DG_HOST_ASSERT(not transport_fp8 or dy_sf_ptrs.size() == world);
    DG_HOST_ASSERT(max_tokens > 0 and tokens <= max_tokens and ring_tokens > 0 and ring_tokens % 128 == 0);
    DG_HOST_ASSERT(topk > 0 and topk <= 32 and 32 % topk == 0);
    DG_HOST_ASSERT(experts > 0 and hidden > 0 and inner > 0 and hidden % 256 == 0 and inner % 256 == 0);
    DG_HOST_ASSERT(w1.size(1) == 2 * inner and dx.size(1) == hidden);
    DG_HOST_ASSERT(w2t.size(0) == experts and w2t.size(1) == inner and w2t.size(2) == hidden);
    DG_HOST_ASSERT(w1t.size(0) == experts and w1t.size(1) == hidden and w1t.size(2) == 2 * inner);
    const torch::Tensor weights_for_scales[] = {w1, w2t, w1t};
    for (int i = 0; i < 3; ++i) {
        DG_HOST_ASSERT(scales[i].dim() == 3 and scales[i].size(0) == experts and
            scales[i].size(1) == weights_for_scales[i].size(1) / 128 and
            scales[i].size(2) == weights_for_scales[i].size(2) / 128);
    }
    DG_HOST_ASSERT(ds.dim() == 2 and ds.size(0) == tokens and ds.size(1) == topk);
    DG_HOST_ASSERT(dy.dim() == 2 and dy.size(0) == max_tokens and dy.size(1) == hidden);
    DG_HOST_ASSERT(ds_partials.dim() == 3 and ds_partials.size(0) == max_tokens and
                   ds_partials.size(1) == topk and ds_partials.size(2) == 2);
    DG_HOST_ASSERT(starts.dim() == 1 and counts.dim() == 1 and starts.numel() == experts and counts.numel() == experts);
    DG_HOST_ASSERT(sym_ptrs[rank] == reinterpret_cast<int64_t>(sym_buffer.data_ptr()));
    DG_HOST_ASSERT(dy_ptrs[rank] == reinterpret_cast<int64_t>(dy.data_ptr()));
    DG_HOST_ASSERT(ds_ptrs[rank] == reinterpret_cast<int64_t>(ds_partials.data_ptr()));
    DG_HOST_ASSERT(pools.size() == 4 or pools.size() == 8 or pools.size() == 10 or pools.size() == 18);
    const bool cached = pools.size() >= 8;
    const bool cached_dz = pools.size() >= 10;
    const bool packed_wgrad = pools.size() == 18;
    const int widths[] = {hidden, hidden, inner, inner * 2};
    const int64_t required_rows = static_cast<int64_t>(world) * max_tokens * topk + experts * 128;
    DG_HOST_ASSERT(required_rows < std::numeric_limits<int>::max());
    const int64_t rows = packed_wgrad ? pools[4].size(0) : pools[0].size(0);
    const int64_t scratch_rows = packed_wgrad ? ring_tokens : rows;
    DG_HOST_ASSERT(rows >= required_rows and rows < std::numeric_limits<int>::max());
    for (int i = 0; i < 4; ++i) {
        check(pools[i], torch::kBFloat16);
        DG_HOST_ASSERT(pools[i].dim() == 2 and pools[i].size(0) == scratch_rows and pools[i].size(1) == widths[i]);
    }
    if (cached) {
        for (int i = 4; i < 8; ++i) {
            check(pools[i], i < 6 ? torch::kFloat8_e4m3fn : torch::kFloat32);
            DG_HOST_ASSERT(pools[i].dim() == 2 and pools[i].size(0) == rows and
                pools[i].size(1) == (i < 6 ? hidden : hidden / 128));
        }
    }
    if (cached_dz) {
        check(pools[8], torch::kFloat8_e4m3fn);
        check(pools[9], torch::kFloat32);
        for (int i = 8; i < 10; ++i)
            DG_HOST_ASSERT(pools[i].dim() == 2 and pools[i].size(0) == rows and
                pools[i].size(1) == (i == 8 ? inner * 2 : inner * 2 / 128));
    }
    if (packed_wgrad) {
        DG_HOST_ASSERT(padded_counts.has_value());
        check(*padded_counts, torch::kInt32);
        DG_HOST_ASSERT(padded_counts->dim() == 1 and padded_counts->numel() == experts);
        const int packed_widths[] = {hidden, hidden, inner, inner * 2};
        for (int i = 0; i < 4; ++i) {
            check(pools[10 + i], torch::kFloat8_e4m3fn);
            DG_HOST_ASSERT(pools[10 + i].numel() == rows * packed_widths[i]);
            const auto& sf = pools[14 + i];
            DG_HOST_ASSERT(sf.is_cuda() and sf.device() == dx.device() and sf.scalar_type() == torch::kFloat32);
            DG_HOST_ASSERT(sf.dim() == 2 and sf.size(0) == packed_widths[i] and sf.size(1) == rows / 128);
            DG_HOST_ASSERT(sf.stride(0) == 1 and sf.stride(1) == packed_widths[i]);
        }
    } else {
        DG_HOST_ASSERT(not padded_counts.has_value());
    }
    if (transport_fp8) {
        DG_HOST_ASSERT(packed_wgrad);
        check(*dy_sf, torch::kFloat32);
        DG_HOST_ASSERT(dy_sf->dim() == 2 and dy_sf->size(0) == max_tokens and
                       dy_sf->size(1) == hidden / 128);
        DG_HOST_ASSERT(dy_sf_ptrs[rank] == reinterpret_cast<int64_t>(dy_sf->data_ptr()));
    }
    const auto buffer_layout = layout::MonoEPBuffer(nullptr, hidden, inner, world, experts * world,
        max_tokens, topk, ring_tokens, 0, false, 0, transport_fp8, fp8_combine);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(buffer_layout.get_num_bytes()));
    const auto acts = [&](const torch::Tensor& t) {
        return make_tma_2d_desc(t, t.size(1), rows, 64, 128, t.stride(0), 128);
    };
    const auto weights = [&](const torch::Tensor& t) {
        const int n = t.size(1), k = t.size(2);
        return make_tma_4d_desc(t, k, n / 2, 2, experts, 128, 128, 1, 1,
            k, static_cast<int64_t>(n / 2) * k, static_cast<int64_t>(n) * k, 128);
    };
    const auto cache_acts = [&](const torch::Tensor& t) {
        return make_tma_2d_desc(t, t.size(1), rows, 128, 128, t.stride(0), 128);
    };
    const auto tx = cached ? cache_acts(pools[4]) : acts(pools[0]);
    const auto ty = cached ? cache_acts(pools[5]) : acts(pools[1]);
    const auto tz = cached_dz ? cache_acts(pools[8]) : acts(pools[3]);
    const auto tw1 = weights(w1), tw2t = weights(w2t), tw1t = weights(w1t);
    const char* compact_selector = std::getenv("MONO_EP_SM90_FP8_COMPACT_BACKWARD");
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
    const char* phase_trace_selector = std::getenv("MONO_EP_SM90_FP8_PHASE_TRACE");
    const uint64_t phase_trace_ptr = phase_trace_selector == nullptr ? 0 :
        std::strtoull(phase_trace_selector, nullptr, 10);
    // Diagnostic only: bitmask removing epilogue constructs to time them (outputs become invalid).
    const char* ablate_selector = std::getenv("MONO_EP_SM90_FP8_DIAG_ABLATE");
    const uint32_t diag_ablate = ablate_selector == nullptr ? 0 :
        static_cast<uint32_t>(std::strtoul(ablate_selector, nullptr, 10));
    // Integrated epilogue/dispatch selectors: unset picks the default, "0" restores the original code path.
    const auto env_mode = [](const char* name, int fallback, int max_value) {
        const char* value = std::getenv(name);
        const int mode = value == nullptr or *value == '\0' ? fallback : std::atoi(value);
        DG_HOST_ASSERT(mode >= 0 and mode <= max_value);
        return mode;
    };
    // 1 = the dispatch warps pack the x/dy columns from the cached FP8 row pools (replaces the math-warp pack).
    // Off by default: 0.40 ms faster single-rank but 0.32 ms slower checkpointed 2F+B at EP8 (NVLink pulls share
    // the dispatch warps), and it spills with the epilogue overlap.
    const bool dispatch_xdy_pack = env_mode("MONO_EP_SM90_FP8_DISPATCH_XDY_PACK", 0, 1) != 0;
    DG_HOST_ASSERT(not dispatch_xdy_pack or (transport_fp8 and cached and packed_wgrad and not compact_non_math));
    const bool xdy_vector_pack = not dispatch_xdy_pack and
        env_mode("MONO_EP_SM90_FP8_XDY_VECTOR_PACK", transport_fp8 and cached ? 1 : 0, 1) != 0;
    // Dispatch / non-epilogue / epilogue setmaxnreg split used with the dispatch-warp pack (default "64/40/200";
    // "48/40/208" is the split without the pack).
    int pack_registers[3] = {64, 40, 200};
    if (const char* value = std::getenv("MONO_EP_SM90_FP8_DISPATCH_PACK_REGISTERS"); value != nullptr and *value != '\0') {
        DG_HOST_ASSERT(std::sscanf(value, "%d%*[,/]%d%*[,/]%d", &pack_registers[0], &pack_registers[1], &pack_registers[2]) == 3);
        for (const int r : pack_registers)
            DG_HOST_ASSERT(r >= 24 and r <= 256 and r % 8 == 0);
        DG_HOST_ASSERT(pack_registers[0] * 128 + pack_registers[1] * 128 + pack_registers[2] * 256 <= 64512);
    }
    const bool custom_pack_registers = dispatch_xdy_pack and
        (pack_registers[0] != 64 or pack_registers[1] != 40 or pack_registers[2] != 200);
    // 1 = relaxed polls plus one acquire on the A-loader, dh epilogue and combine-readiness spins.
    const bool relaxed_spins = env_mode("MONO_EP_SM90_FP8_RELAXED_SPINS", 0, 1) != 0;
    DG_HOST_ASSERT(not xdy_vector_pack or (transport_fp8 and cached));
    // 1 = vector pack with the original fences, 2 = also drops them (packed columns are read by a later launch).
    const int vector_hdz_pack = env_mode("MONO_EP_SM90_FP8_VECTOR_HDZ_PACK", 2, 2);
    // Load-batch depth of the vector dZ row-cache quantize; 2 spills in the combined build.
    const int vector_dz_quant = env_mode("MONO_EP_SM90_FP8_VECTOR_DZ_QUANT", 1, 2);
    // FP8 dx combine bitmask: 1 vectorized scatter, 2 one system fence per thread, 4 vectorized tail.
    const int dx_combine_env = env_mode("MONO_EP_SM90_FP8_DX_COMBINE", 7, 7);
    const int dx_combine_mode = fp8_combine ? dx_combine_env : 0;
    // 0 = ld.acquire spin on ring-slot release, 1 = relaxed poll + one acquire,
    // 2/3/4 = relaxed poll with nanosleep backoff (32-256 / 64-1024 / 128-4096 ns).
    const int dispatch_spin = env_mode("MONO_EP_SM90_FP8_DISPATCH_SPIN", 4, 4);
    // FP8-sized A stages (cached dZ only); 0 keeps two BF16-sized stages. Three FP8 stages use the
    // same shared memory as two BF16 stages.
    const int fp8_stages = env_mode("MONO_EP_SM90_FP8_BACKWARD_STAGES", cached_dz ? 3 : 0, 8);
    DG_HOST_ASSERT(fp8_stages == 0 or (fp8_stages >= 2 and cached_dz));
    const bool fp8_only_stages = fp8_stages != 0;
    const int num_stages = fp8_only_stages ? fp8_stages : 2;
    // Mainloop bits: 1 stages cached row scales in shared memory from the A loader, 2 loads a task's
    // weight scales once per lane.
    // Off by default: exclusive with the epilogue overlap; without the overlap and the pack, mode 2 saved
    // 0.05 ms of checkpointed 2F+B at EP8, against 0.27 ms for the overlap with the activation loop.
    const int mainloop_mode = env_mode("MONO_EP_SM90_FP8_BACKWARD_MAINLOOP", 0, 3);
    const std::string diag_suffix = std::format(", {}, {}u, {}, {}u, {}u, {}u, {}u, {}, {}u, {}, {}",
        phase_trace_ptr != 0 ? "true" : "false", diag_ablate, xdy_vector_pack ? "true" : "false",
        vector_hdz_pack, vector_dz_quant, dx_combine_mode, dispatch_spin, fp8_only_stages ? "true" : "false",
        mainloop_mode, relaxed_spins ? "true" : "false", dispatch_xdy_pack ? "true" : "false");
    // Diagnostic only: multi-rank extension of the phase trace (sms x 64 more entries after the CTA records).
    const char* dist_trace_selector = std::getenv("MONO_EP_SM90_FP8_DIST_TRACE");
    const bool dist_trace = dist_trace_selector != nullptr and std::string_view(dist_trace_selector) == "1";
    DG_HOST_ASSERT(not dist_trace or (phase_trace_ptr != 0 and world <= 8));
    // Tail scheduling bits: 1 = per-tile dX claims in the last-wave window, taken by idle CTAs;
    // 2 = partial M blocks claimed last inside that window. 0 keeps the original claim order.
    const int tail_mode = env_mode("MONO_EP_SM90_FP8_TAIL_SCHEDULE", 0, 3);
    DG_HOST_ASSERT((tail_mode & 2) == 0 or experts <= 32);
    const auto activation = select_mono_ep_activation(not compact_non_math);
    const bool fast_pair = activation.kind == 1 and activation.limit_bits == 0x7f7fffffu;
    // 0 = the original SwiGLU-backward loop; 1 = score held in a lane and shuffled per row, activation from
    // mono_ep_activations.cuh, dead BF16 h stores skipped on the packed-wgrad path (dx and dscore bitwise
    // equal to 0 for the default SwiGLU); 2 = 1 with two columns per lane (BF16x2 loads and stores; dscore
    // summation order changes). 1 or 2 is required for any activation other than the default SwiGLU.
    // Default on 512-thread launches: 2 for swiglu_fast, 1 otherwise; compact384 keeps 0 for the default SwiGLU.
    const int activation_loop = env_mode("MONO_EP_SM90_FP8_ACTIVATION_LOOP",
        compact_non_math ? (activation.is_default() ? 0 : 1) : (fast_pair ? 2 : 1), 2);
    DG_HOST_ASSERT(activation_loop != 0 or activation.is_default());
    // Rows per dscore reduction chunk in the new loop (0 = the build default, 8 with role registers).
    const int activation_chunk = env_mode("MONO_EP_SM90_FP8_ACTIVATION_CHUNK",
        activation_loop != 0 and not compact_non_math ? 16 : 0, 32);
    DG_HOST_ASSERT(activation_chunk == 0 or activation_chunk == 2 or activation_chunk == 4 or
                   activation_chunk == 8 or activation_chunk == 16 or activation_chunk == 32);
    DG_HOST_ASSERT(activation_chunk == 0 or activation_loop != 0);
    DG_HOST_ASSERT(activation_loop != 2 or activation_chunk <= 16);
    // Issue the next task's k-blocks (same M block) from yield points inside the epilogues:
    // 1 = the measured best site set (dZ quantize after its loads, dX scatter), N > 1 = raw site
    // bitmask of the kernel's yield sites (diagnostic).
    // Default 1 when the caches are on, without the dispatch-warp pack (spills) or a mainloop mode (exclusive).
    const int overlap_mode = env_mode("MONO_EP_SM90_FP8_OVERLAP_EPILOGUE",
        cached and cached_dz and not dispatch_xdy_pack and mainloop_mode == 0 and not compact_non_math ? 1 : 0, 255);
    const int overlap_epilogue = overlap_mode == 1 ? 0x60 : overlap_mode;
    DG_HOST_ASSERT(overlap_epilogue == 0 or (cached and cached_dz));
    DG_HOST_ASSERT(overlap_epilogue == 0 or mainloop_mode == 0);
    // Trailing template arguments after kDistTrace; with every selector off nothing is appended, so the
    // instantiation text is unchanged.
    const std::string activation_args = activation_loop == 0 and overlap_epilogue == 0 and tail_mode == 0 ? std::string() :
        std::format(", {}u, 0x{:08x}u, 0x{:08x}u, {}u, {}u", activation.kind, activation.alpha_bits,
                    activation.limit_bits, activation_loop, activation_chunk) +
        (overlap_epilogue != 0 or tail_mode != 0 ? std::format(", {}u", overlap_epilogue) : std::string()) +
        (tail_mode != 0 ? std::format(", {}u", tail_mode) : std::string());
    const std::string trailing_args = dist_trace or not activation_args.empty() ?
        std::string(dist_trace ? ", true" : ", false") + activation_args : std::string();
    // 2 = role-scoped setmaxnreg plus a split math region before the combine (removes C7507 and the
    // epilogue spills); compact 384-thread launches keep the shared register budget.
    const int role_registers_mode = env_mode("MONO_EP_SM90_FP8_ROLE_REGISTERS", compact_non_math ? 0 : 2, 2);
    const bool role_registers = role_registers_mode != 0;
    const bool split_combine_registers = role_registers_mode == 2;
    DG_HOST_ASSERT(not (role_registers and compact_non_math));
    const int dispatch_threads = compact_non_math ? 32 : 128;
    const int non_epilogue_threads = compact_non_math ? 96 : 128;
    const int block_threads = compact_non_math ? 384 : 512;
    check_ring_capacity_for_mono_ep(
        ring_tokens, 128, world, experts, max_tokens, topk, hidden, inner);
    const int sms = get_num_sms_for_mono_ep();
    int pull = 4096;
    const int hidden_bytes = hidden * (transport_fp8 ? 1 : 2);
    while (hidden_bytes % pull != 0) pull /= 2;
    DG_HOST_ASSERT(hidden_bytes % pull == 0 and pull % 16 == 0);
    const int smem = SM90ArchSpec::smem_capacity;
    const auto kernel = jit->compile(
        std::format("sm90_fp8_mono_ep_backward_{}_{}_{}{}{}{}",
            compact_non_math ? "compact384" : "baseline512",
            trap_timeouts ? "trap" : "diagnostic",
            transport_fp8 ? (fp8_combine ? "fp8transport_fp8combine" : "fp8transport_bf16combine") : "bf16transport",
            phase_trace_ptr != 0 ? (dist_trace ? "_phasetrace_dist" : "_phasetrace") : "",
            (diag_ablate != 0 ? std::format("_ablate{}", diag_ablate) : std::string()) +
                (fp8_only_stages ? std::format("_fp8stages{}", num_stages) : std::string()) +
                (xdy_vector_pack ? "_xdyvec" : "") +
                (vector_hdz_pack ? std::format("_vechdz{}", vector_hdz_pack) : std::string()) +
                (vector_dz_quant ? std::format("_vecdzq{}", vector_dz_quant) : std::string()) +
                (dx_combine_mode != 0 ? std::format("_dxcombine{}", dx_combine_mode) : std::string()) +
                (dispatch_spin != 0 ? std::format("_spin{}", dispatch_spin) : std::string()) +
                (mainloop_mode != 0 ? std::format("_mainloop{}", mainloop_mode) : std::string()) +
                (relaxed_spins ? "_relaxedspins" : "") + (dispatch_xdy_pack ? "_dispatchxdy" : "") +
                (custom_pack_registers ? std::format("_reg{}_{}_{}", pack_registers[0], pack_registers[1],
                    pack_registers[2]) : std::string()) +
                (activation_loop != 0 ? std::format("_actloop{}c{}", activation_loop, activation_chunk) +
                    (activation.is_default() ? std::string("_act0") : activation.tag()) : std::string()) +
                (overlap_epilogue != 0 ? std::format("_overlap{:x}", overlap_epilogue) : std::string()),
            std::string(split_combine_registers ? "_rolereg_split" : role_registers ? "_rolereg" : "") +
                (tail_mode != 0 ? std::format("_tail{}", tail_mode) : std::string())),
        std::format(R"({}{}
#include <deep_gemm/impls/sm90_fp8_mono_ep_backward.cuh>
static void instantiate() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mono_ep_backward_impl<
        {}, {}, {}, {}, 0, {}, 128, 128, 128, 32, {}, {}, {}, {}, {}, 256, {}, {}, true, {}, {}, {}, {}, {}{}>);
}}
)", trap_timeouts ? "#define DG_TRAP_ONLY_TIMEOUTS 1" : "",
            std::string(split_combine_registers ? "\n#define DG_MonoEP_ROLE_SCOPED_REGISTERS 1\n#define DG_MonoEP_SPLIT_COMBINE_REGISTERS 160" :
                role_registers ? "\n#define DG_MonoEP_ROLE_SCOPED_REGISTERS 1" : "") +
                (custom_pack_registers ? std::format(
                    "\n#define DG_MonoEP_PACK_DISPATCH_REGISTERS {}\n#define DG_MonoEP_PACK_NON_EPILOGUE_REGISTERS {}"
                    "\n#define DG_MonoEP_PACK_EPILOGUE_REGISTERS {}", pack_registers[0], pack_registers[1],
                    pack_registers[2]) : std::string()),
            max_tokens, hidden, inner, experts * world, topk, ring_tokens, num_stages, pull,
            dispatch_threads, non_epilogue_threads, sms, world, cached, cached_dz,
            compact_non_math ? "true" : "false", transport_fp8 ? "true" : "false",
            fp8_combine ? "true" : "false", diag_suffix + trailing_args));
    layout::MoEBackwardBuffers buffers{
        layout::SymBuffer<>(dy_ptrs, rank), layout::SymBuffer<>(ds_ptrs, rank),
        reinterpret_cast<__nv_bfloat16*>(pools[0].data_ptr()), reinterpret_cast<__nv_bfloat16*>(pools[1].data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(pools[2].data_ptr()), reinterpret_cast<__nv_bfloat16*>(pools[3].data_ptr()),
        ds.data_ptr<float>(), reinterpret_cast<uint32_t*>(starts.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(counts.data_ptr<int>())};
    if (cached) {
        buffers.qx_pool = reinterpret_cast<uint8_t*>(pools[4].data_ptr());
        buffers.qdy_pool = reinterpret_cast<uint8_t*>(pools[5].data_ptr());
        buffers.sx_pool = pools[6].data_ptr<float>();
        buffers.sdy_pool = pools[7].data_ptr<float>();
    }
    if (cached_dz) {
        buffers.qdz_pool = reinterpret_cast<uint8_t*>(pools[8].data_ptr());
        buffers.sdz_pool = pools[9].data_ptr<float>();
    }
    if (packed_wgrad) {
        buffers.qx_col = reinterpret_cast<uint8_t*>(pools[10].data_ptr());
        buffers.qdy_col = reinterpret_cast<uint8_t*>(pools[11].data_ptr());
        buffers.qh_col = reinterpret_cast<uint8_t*>(pools[12].data_ptr());
        buffers.qdz_col = reinterpret_cast<uint8_t*>(pools[13].data_ptr());
        buffers.sx_col = pools[14].data_ptr<float>();
        buffers.sdy_col = pools[15].data_ptr<float>();
        buffers.sh_col = pools[16].data_ptr<float>();
        buffers.sdz_col = pools[17].data_ptr<float>();
        buffers.expert_padded_counts = reinterpret_cast<uint32_t*>(padded_counts->data_ptr<int>());
    }
    if (transport_fp8)
        buffers.dy_sf = layout::SymBuffer<>(dy_sf_ptrs, rank);
    jit->launch(kernel, {.num_smem_bytes = smem, .grid_dim = dim3(sms),
        .block_dim = dim3(block_threads), .cluster_dim = dim3(1, 1, 1)},
        dx.data_ptr(), static_cast<int*>(nullptr), tokens, layout::SymBuffer<>(sym_ptrs, rank),
        tx, tw1, tx, tz, tw1t, tx, tw1, tx, tz, tw1t,
        static_cast<const uint8_t*>(nullptr), reinterpret_cast<__nv_bfloat16*>(phase_trace_ptr), ty, tw2t, buffers,
        scales[0].data_ptr<float>(), scales[1].data_ptr<float>(), scales[2].data_ptr<float>());
}

} // namespace deep_gemm
