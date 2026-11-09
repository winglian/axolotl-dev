// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Copyright (c) 2026 Axolotl AI
#pragma once

#include "sm90_fp8_gemm_1d1d.hpp"

namespace deep_gemm {

static void k_grouped_fp8_gemm_nt_gpu_sizes(const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb, const torch::Tensor& d,
    const torch::Tensor& grouped_layout, int capacity) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 9);
    DG_HOST_ASSERT(d.is_cuda() and d.dim() == 3 and d.is_contiguous() and d.scalar_type() == torch::kFloat32);
    const int num_groups = d.size(0), m = d.size(1), n = d.size(2);
    DG_HOST_ASSERT(num_groups > 0 and m > 0 and n > 0 and m % 4 == 0 and n % 4 == 0);
    DG_HOST_ASSERT(capacity >= 128 and capacity % 128 == 0);
    const auto check = [&](const torch::Tensor& t, torch::ScalarType dtype) {
        DG_HOST_ASSERT(t.is_cuda() and t.device() == d.device() and t.scalar_type() == dtype);
    };
    check(a, torch::kFloat8_e4m3fn);
    check(b, torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(a.is_contiguous() and b.is_contiguous());
    DG_HOST_ASSERT(a.numel() == static_cast<int64_t>(capacity) * m and b.numel() == static_cast<int64_t>(capacity) * n);
    check(grouped_layout, torch::kInt32);
    DG_HOST_ASSERT(grouped_layout.dim() == 1 and grouped_layout.numel() == num_groups and grouped_layout.is_contiguous());
    const torch::Tensor scale_tensors[] = {sfa, sfb};
    const int widths[] = {m, n};
    for (int i = 0; i < 2; ++i) {
        const auto& sf = scale_tensors[i];
        check(sf, torch::kFloat32);
        DG_HOST_ASSERT(sf.dim() == 2 and sf.size(0) == widths[i] and sf.size(1) == capacity / 128);
        DG_HOST_ASSERT(sf.stride(0) == 1 and sf.stride(1) == widths[i]);
    }
    const auto major_a = cute::UMMA::Major::K, major_b = cute::UMMA::Major::K;
    const int first_k = 128, sum_k = capacity, sum_sf_k = capacity / 128, max_k = capacity;
    const std::string compiled_dims = "mn";
    const auto tensor_map_buffer = torch::empty(
        {runtime->get_num_sms() * 4 * static_cast<int>(sizeof(CUtensorMap))}, d.options().dtype(torch::kByte));

    const auto desc = GemmDesc {
        .gemm_type = GemmType::KGroupedContiguous,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = sum_k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = true,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = m, .expected_n = n, .expected_k = max_k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);

    const auto tensor_map_a_base = make_tma_a_desc(major_a, a, m, first_k,
                                                   config.storage_config.load_block_m,
                                                   config.layout.block_k, first_k, 1,
                                                   config.storage_config.swizzle_a_mode);
    const auto tensor_map_b_base = make_tma_b_desc(major_b, b, n, first_k,
                                                   config.storage_config.load_block_n,
                                                   config.layout.block_k, first_k, 1,
                                                   config.storage_config.swizzle_b_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, sum_sf_k * 128,
                                                 config.layout.block_m, config.layout.block_k, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, sum_sf_k * 128,
                                                 config.layout.block_n, config.layout.block_k, 1, 0);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.storage_config.swizzle_cd_mode);

    // Compile and launch
    SM90FP8Gemm1D1DRuntime::compile_and_launch("sm90_fp8_gemm_gpu_sizes", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .gmem_a_ptr = a.data_ptr(),
        .gmem_b_ptr = b.data_ptr(),
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_buffer = tensor_map_buffer.data_ptr(),
        .tensor_map_a_base = tensor_map_a_base,
        .tensor_map_b_base = tensor_map_b_base,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
        .validate_capacity = true,
    });
}

} // namespace deep_gemm
