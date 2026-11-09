#pragma once

#include "../utils/compatibility.hpp"

#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_wgrad.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"

#include "../jit_kernels/impls/smxx_cublaslt.hpp"

#include "layout.hpp"

namespace deep_gemm::gemm {

static bool early_return(const int& m, const int &n, const int& k,
                         const torch::Tensor& d, const std::optional<torch::Tensor>& c,
                         const std::optional<torch::Tensor>& sfd = std::nullopt) {
    // Do nothing if the problem is empty
    if (m == 0 or n == 0)
        return true;

    // Checks
    // NOTES: quantized D never accumulates
    DG_HOST_ASSERT(not sfd.has_value() or not c.has_value());
    const bool is_cd_same = c.has_value() and c->data_ptr() == d.data_ptr();
    if (is_cd_same)
        DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat or
                   d.scalar_type() == torch::kFloat8_e4m3fn);
    // NOTES: an FP8 D must come with SFD, and vice versa; kernels without SFD support must not receive FP8 D
    DG_HOST_ASSERT((d.scalar_type() == torch::kFloat8_e4m3fn) == sfd.has_value());
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(d.scalar_type() == c.value().scalar_type());
    }

    // No accumulation
    if (k == 0) {
        if (not is_cd_same)
            c.has_value() ? d.copy_(c.value()) : d.zero_();
        // A zero D dequantizes to zero regardless of the SF content
        if (sfd.has_value())
            sfd->zero_();
        return true;
    }

    // With accumulation, do copy before GEMM (assuming the GEMM kernel does not support different C/D)
    if (c.has_value() and not is_cd_same)
        d.copy_(c.value());
    return false;
}

static int check_k_grouped_args(const std::optional<std::vector<int>>& ks_cpu,
                                const torch::Tensor& grouped_layout,
                                const int& num_groups,
                                const bool& use_psum_layout,
                                const int& k_alignment,
                                const int& sum_k_if_ks_cpu_missing = 0) {
    DG_HOST_ASSERT(grouped_layout.is_contiguous());
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(static_cast<int>(grouped_layout.numel()) == num_groups);

    if (ks_cpu.has_value() and not ks_cpu.value().empty()) {
        DG_HOST_ASSERT(static_cast<int>(ks_cpu.value().size()) == num_groups);
        int sum_k = 0;
        for (const auto k: ks_cpu.value()) {
            DG_HOST_ASSERT(k % k_alignment == 0);
            sum_k += k;
        }
        return sum_k;
    }
    DG_HOST_ASSERT(use_psum_layout);
    return sum_k_if_ks_cpu_missing;
}

static void fp8_fp4_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            std::optional<std::tuple<int, int, int>> recipe,
                            std::optional<std::tuple<int, int>> recipe_a,
                            std::optional<std::tuple<int, int>> recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast,
                            const std::optional<float>& alpha,
                            const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    if (fp8_fp4_requires_k_major(a.first, b.first)) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto arch_major = jit->device.get_arch_major();
    const auto [m , k ] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [n , k_] = check_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    // Transform SFA and SFB into compute-required layout
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, std::nullopt, disable_ue8m0_cast);

    // Dispatch into different implements
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        DG_HOST_ASSERT(not alpha.has_value() and "FP8 GEMM alpha requires SM100");
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
        const int gran_n = recipe.has_value() ? std::get<1>(recipe.value()) : std::get<0>(recipe_b.value());
        if (gran_n == 1) {
            sm90_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
        } else {
            const auto major_sfb = get_major_type_ab(sfb);
            sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, major_sfb, compiled_dims);
        }
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, gran_k_a, gran_k_b,
                                major_a, major_b, compiled_dims, resolve_epilogue_class(epilogue_class, c, d, alpha));
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void fp8_fp4_gemm_nn(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast,
                            const std::optional<float>& alpha,
                            const std::shared_ptr<EpilogueClass>& epilogue_class) {
    fp8_fp4_gemm_nt(a, {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, alpha, epilogue_class);
}

static void fp8_fp4_gemm_tn(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast,
                            const std::optional<float>& alpha,
                            const std::shared_ptr<EpilogueClass>& epilogue_class) {
    fp8_fp4_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)},
                    {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, alpha, epilogue_class);
}

static void fp8_fp4_gemm_tt(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast,
                            const std::optional<float>& alpha,
                            const std::shared_ptr<EpilogueClass>& epilogue_class) {
    fp8_fp4_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)}, b,
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, alpha, epilogue_class);
}

static void m_grouped_fp8_fp4_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 std::optional<std::tuple<int, int, int>> recipe,
                                                 std::optional<std::tuple<int, int>> recipe_a,
                                                 std::optional<std::tuple<int, int>> recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 const bool& use_psum_layout,
                                                 const bool& ensure_zero_padding,
                                                 const std::optional<int>& expected_m_for_psum_layout,
                                                 const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_fp4_requires_k_major(a.first, b.first))
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());

    // Type and shape checks
    const auto arch_major = jit->device.get_arch_major();
    const auto [m , k ] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [num_groups, n, k_] = check_grouped_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);

    // Layout checks
    if (use_psum_layout) {
        const auto [num_groups_] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(num_groups == num_groups_);
    } else {
        const auto [m__] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(m == m__);
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Pass PSUM layout so SFA packing skips gap rows
    const std::optional<torch::Tensor> psum_sfa_layout = use_psum_layout ? std::make_optional(grouped_layout) : std::nullopt;
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, num_groups, disable_ue8m0_cast,
        psum_sfa_layout);

    // Dispatch implementation
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        const auto major_sfb = get_major_type_ab(sfb);
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
        sm90_m_grouped_fp8_gemm_contiguous_1d2d(a.first, sfa, b.first, sfb, d, grouped_layout,
                                                num_groups, m, n, k, major_a, major_b, major_sfb,
                                                compiled_dims, use_psum_layout, expected_m_for_psum_layout);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d, grouped_layout,
                                                     num_groups, m, n, k, gran_k_a, gran_k_b, major_a, major_b,
                                                     compiled_dims, use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout,
                                                     resolve_epilogue_class(epilogue_class, std::nullopt, d));
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void m_grouped_fp8_fp4_gemm_nn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 const std::optional<std::tuple<int, int, int>>& recipe,
                                                 const std::optional<std::tuple<int, int>>& recipe_a,
                                                 const std::optional<std::tuple<int, int>>& recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 const bool& use_psum_layout,
                                                 const bool& ensure_zero_padding,
                                                 const std::shared_ptr<EpilogueClass>& epilogue_class) {
    m_grouped_fp8_fp4_gemm_nt_contiguous(a, {b.first.transpose(1, 2), b.second.transpose(1, 2)},
                                         d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast,
                                         use_psum_layout, ensure_zero_padding, std::nullopt, epilogue_class);
}

static void m_grouped_fp8_fp4_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& masked_m,
                                             const int& expected_m,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             std::optional<std::tuple<int, int>> recipe_a,
                                             std::optional<std::tuple<int, int>> recipe_b,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast,
                                             const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    // Type and shape checks
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_groups  , m , k ] = check_grouped_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [num_groups_ , n , k_] = check_grouped_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [num_groups__, m_, n_] = get_shape<3>(d);
    const auto num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Transform scaling factors
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, num_groups, num_groups, disable_ue8m0_cast);

    // Dispatch implementation
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        const auto major_sfb = get_major_type_ab(sfb);
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
        sm90_m_grouped_fp8_gemm_masked_1d2d(a.first, sfa, b.first, sfb, d, masked_m,
                                            num_groups, m, n, k, expected_m, major_a, major_b, major_sfb, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_m_grouped_fp8_fp4_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m,
                                                 num_groups, m, n, k, expected_m, gran_k_a, gran_k_b,
                                                 major_a, major_b, compiled_dims, resolve_epilogue_class(epilogue_class, std::nullopt, d));
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void k_grouped_fp8_gemm_tn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const std::optional<std::vector<int>>& ks_cpu,
                                             const torch::Tensor& grouped_layout,
                                             const std::optional<torch::Tensor>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims,
                                             const bool& use_psum_layout,
                                             const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(std::get<0>(recipe) == 1 and std::get<1>(recipe) == 1);

    // `k_alignment` must be a multiple of `BLOCK_K = 128`
    // All A/B padding must be zero, and the corresponding SF padding must be valid
    const int gran_k = std::get<2>(recipe);
    const int k_alignment = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    DG_HOST_ASSERT(k_alignment % 128 == 0);

    // A/B use MN-major layouts `[sum_k, M]` and `[sum_k, N]`
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto [sum_k_ , m_] = get_shape<2>(a.first);
    const auto [sum_k__, n_] = get_shape<2>(b.first);

    // Without PSUM layout, `grouped_layout[i]` is the padded K size of group i;
    // With PSUM layout, it is the logical K end, and the group starts at `align(previous_end, k_alignment)`
    const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups,
                                           use_psum_layout, k_alignment, static_cast<int>(a.first.size(0)));
    DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);
    // Contiguity checks
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(not c.has_value() or c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, sum_k, d, c))
        return;

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 10) {
        const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks_cpu, grouped_layout, recipe, k_alignment, use_psum_layout);
        const auto sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ks_cpu, grouped_layout, recipe, k_alignment, use_psum_layout);
        sm100_k_grouped_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, grouped_layout, gran_k, k_alignment,
                                      cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims, use_psum_layout,
                                      resolve_epilogue_class(epilogue_class, c, d));
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_fp8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const std::optional<std::vector<int>>& ks_cpu,
                                             const torch::Tensor& grouped_layout,
                                             const std::optional<torch::Tensor>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims,
                                             const bool& use_psum_layout,
                                             const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));
    DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");

    // No psum on FP8 NT
    DG_HOST_ASSERT(not use_psum_layout and ks_cpu.has_value() and not ks_cpu.value().empty());

    // Shape checks
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto sum_mk = a.first.numel();
    const auto sum_nk = b.first.numel();
    const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups,
                                           use_psum_layout, 128);
    DG_HOST_ASSERT(sum_mk == static_cast<int64_t>(sum_k) * m);
    DG_HOST_ASSERT(sum_nk == static_cast<int64_t>(sum_k) * n);

    // Contiguity checks
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, sum_k, d, c))
        return;

    // Transform SF with padding
    const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks_cpu, grouped_layout, recipe, 128, false);
    const auto sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ks_cpu, grouped_layout, recipe, 128, false);

    // Allocate tensormap buffer
    // `4` means the double buffering for both A and B operands (2 * 2)
    const auto num_sms = runtime->get_num_sms();
    const auto tensor_map_buffer = torch::empty({num_sms * 4 * static_cast<int>(sizeof(CUtensorMap))},
                                                a.first.options().dtype(torch::kByte));

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        sm90_k_grouped_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, ks_cpu.value(), grouped_layout, tensor_map_buffer,
                                     cute::UMMA::Major::K, cute::UMMA::Major::K, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_fp4_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const std::optional<std::vector<int>>& ks_cpu,
                                             const torch::Tensor& grouped_layout,
                                             const std::optional<torch::Tensor>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims,
                                             const bool& use_psum_layout,
                                             const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 32));
    DG_HOST_ASSERT(a.first.scalar_type() == kPackedFP4 and b.first.scalar_type() == kPackedFP4);

    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    // `k_alignment` must be a multiple of `BLOCK_K = 256`
    // All A/B padding codes must be zero, and the corresponding SF padding must be valid
    const int k_alignment = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
    DG_HOST_ASSERT(k_alignment % 256 == 0);

    // A/B use K-major logical layouts `[M, sum_k]`/`[N, sum_k]`,
    // backed by packed-byte storage `[M, sum_k / 2]`/`[N, sum_k / 2]`
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto [m_, sum_k_] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [n_, sum_k__] = check_ab_fp8_fp4(b.first, major_b, arch_major);

    // Without PSUM layout, `grouped_layout[i]` is the padded K size of group i;
    // With PSUM layout, it is the logical K end, and the group starts at `align(previous_end, k_alignment)`
    const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups,
                                           use_psum_layout, k_alignment, sum_k_);
    DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);

    // Contiguity checks
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(not c.has_value() or c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, sum_k, d, c))
        return;

    // Transform SF with padding
    const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks_cpu, grouped_layout, recipe, k_alignment, use_psum_layout);
    const auto sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ks_cpu, grouped_layout, recipe, k_alignment, use_psum_layout);

    // Dispatch implementation
    if (arch_major == 10) {
        sm100_k_grouped_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, sum_k, grouped_layout,
                                      k_alignment, major_a, major_b, compiled_dims, use_psum_layout,
                                      resolve_epilogue_class(epilogue_class, c, d));
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}
static void bf16_gemm_nt(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims,
                         const std::optional<float>& alpha,
                         const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a);
    const auto [n , k_] = get_shape<2>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    const auto resolved_epilogue_class = resolve_epilogue_class(epilogue_class, c, d, alpha);
    const auto epilogue_alpha = resolved_epilogue_class->get_alpha();

    // Dispatch into different implements; cuBLASLt only covers a plain store, optionally alpha-scaled
    if (not heuristics_runtime->get_deterministic_algorithms() and runtime->is_cublaslt_available()
        and epilogue_alpha.has_value()) {
        cublaslt_gemm(a, b, d, m, n, k, major_a, major_b, c.has_value(), epilogue_alpha.value());
        return;
    }

    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        DG_HOST_ASSERT(not alpha.has_value() and "BF16 GEMM alpha requires SM100");
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
        sm90_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims, resolved_epilogue_class);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bf16_gemm_nn(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims,
                         const std::optional<float>& alpha,
                         const std::shared_ptr<EpilogueClass>& epilogue_class) {
    bf16_gemm_nt(a, b.transpose(0, 1), d, c, compiled_dims, alpha, epilogue_class);
}

static void bf16_gemm_tn(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims,
                         const std::optional<float>& alpha,
                         const std::shared_ptr<EpilogueClass>& epilogue_class) {
    bf16_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c, compiled_dims, alpha, epilogue_class);
}

static void bf16_gemm_tt(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims,
                         const std::optional<float>& alpha,
                         const std::shared_ptr<EpilogueClass>& epilogue_class) {
    bf16_gemm_nt(a.transpose(0, 1), b, d, c, compiled_dims, alpha, epilogue_class);
}

static void m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout,
                                              const bool& ensure_zero_padding,
                                              const std::optional<int>& expected_m_for_psum_layout,
                                              const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());

    // Type and shape checks
    const auto [m, k] = get_shape<2>(a);
    const auto [num_groups, n, k_] = get_shape<3>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);

    // Layout checks
    if (use_psum_layout) {
        const auto [num_groups_] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(num_groups == num_groups_);
    } else {
        const auto [m__] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(m == m__);
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        const auto ready = std::dynamic_pointer_cast<ReadyEpilogue>(epilogue_class);
        DG_HOST_ASSERT(epilogue_class == nullptr or
                       (ready and ready->is_incremental_sm90_supported() and not use_psum_layout));
        if (ready and ready->is_incremental_sm90_swiglu())
            DG_HOST_ASSERT(d.is_contiguous());
        sm90_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims,
                                            use_psum_layout, expected_m_for_psum_layout,
                                            ready);
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                             num_groups, m, n, k, major_a, major_b, compiled_dims,
                                             use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout,
                                             resolve_epilogue_class(epilogue_class, std::nullopt, d));
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void fp8_chunk_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                              const std::pair<torch::Tensor, torch::Tensor>& b,
                              const torch::Tensor& d, const torch::Tensor& task_queue,
                              const int task_idx, const int gran_k,
                              const std::string& compiled_dims,
                              const std::shared_ptr<EpilogueClass>& epilogue_class) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(a.first.is_cuda() and b.first.device() == a.first.device());
    DG_HOST_ASSERT(d.device() == a.first.device() and task_queue.device() == a.first.device());
    DG_HOST_ASSERT(a.second.device() == a.first.device() and b.second.device() == a.first.device());
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn and b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(a.second.scalar_type() == torch::kInt and b.second.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    DG_HOST_ASSERT(task_queue.is_contiguous() and task_queue.scalar_type() == torch::kInt);
    const auto [tasks, fields] = get_shape<2>(task_queue);
    DG_HOST_ASSERT(fields == 4 and task_idx >= 0 and task_idx < tasks);
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    const auto [m, k] = check_ab_fp8_fp4(a.first, major_a, 10);
    const auto [groups, n, bk] = check_grouped_ab_fp8_fp4(b.first, major_b, 10);
    const auto [dm, dn] = get_shape<2>(d);
    DG_HOST_ASSERT(m == dm and n == dn and k == bk and n > 0 and k > 0 and groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    check_major_type_cd(d);
    // Packed scales require metadata checks only: producer data may not be ready yet.
    const auto sfa = layout::transform_sf_into_required_layout(a.second, m, k,
        std::tuple<int, int>{1, gran_k}, std::nullopt, std::nullopt, false);
    const auto sfb = layout::transform_sf_into_required_layout(b.second, n, k,
        std::tuple<int, int>{1, gran_k}, groups, std::nullopt, false);
    if (m == 0)
        return;
    sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d,
        task_queue, groups, m, n, k, gran_k, gran_k, major_a, major_b,
        compiled_dims, false, true, std::nullopt,
        resolve_epilogue_class(epilogue_class, std::nullopt, d), task_idx);
}

static void fp8_ready_grouped_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                                     const std::pair<torch::Tensor, torch::Tensor>& b,
                                     const torch::Tensor& d, const torch::Tensor& ids,
                                     const torch::Tensor& queue,
                                     const std::shared_ptr<EpilogueClass>& epilogue_class, bool incremental) {
    const auto arch_major = jit->device.get_arch_major();
    DG_HOST_ASSERT(arch_major == 9 or arch_major == 10);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn and b.first.scalar_type() == torch::kFloat8_e4m3fn);
    for (const auto& t: {a.first, a.second, b.first, b.second, ids, queue})
        DG_HOST_ASSERT(t.is_cuda() and t.device() == d.device());
    if (arch_major == 9) {
        DG_HOST_ASSERT(incremental and epilogue_class == nullptr);
        DG_HOST_ASSERT(a.second.scalar_type() == torch::kFloat and b.second.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(queue.dim() == 2 and queue.size(1) == 4 and queue.scalar_type() == torch::kInt);
        const auto major_a = get_major_type_ab(a.first);
        const auto major_b = get_major_type_ab(b.first);
        const auto major_sfb = get_major_type_ab(b.second);
        const auto [m, k] = check_ab_fp8_fp4(a.first, major_a, arch_major);
        const auto [num_groups, n, k_] = check_grouped_ab_fp8_fp4(b.first, major_b, arch_major);
        const auto [m_, n_] = get_shape<2>(d);
        DG_HOST_ASSERT(m == m_ and n == n_ and k == k_ and m == ids.numel());
        DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0 and queue.size(0) > 0);
        DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 and ids.scalar_type() == torch::kInt);
        DG_HOST_ASSERT(d.is_contiguous() and ids.is_contiguous() and queue.is_contiguous());
        check_sf_layout(a.second, m, k, 1, 128, std::nullopt,
                        true, false, torch::kFloat);
        DG_HOST_ASSERT(a.second.stride(-1) ==
                       get_tma_aligned_size(m, a.second.element_size()));
        check_sf_layout(b.second, n, k, 128, 128, num_groups,
                        false, true, torch::kFloat);
        if (m == 0)
            return;
        sm90_m_grouped_fp8_gemm_contiguous_1d2d(
            a.first, a.second, b.first, b.second, d, ids,
            num_groups, m, n, k, major_a, major_b, major_sfb,
            "nk", false, std::nullopt, queue);
        return;
    }
    DG_HOST_ASSERT(a.second.scalar_type() == torch::kInt and b.second.scalar_type() == torch::kInt);
    const auto ready = std::make_shared<ReadyEpilogue>(
        resolve_epilogue_class(epilogue_class, std::nullopt, d), queue, incremental);
    ready->check(std::nullopt, d);
    m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, ids, std::nullopt,
        std::tuple<int, int>{1, 32}, std::tuple<int, int>{1, 32},
        "nk", false, false, true, std::nullopt, ready);
}

static void fp8_grouped_scatter_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                                       const std::pair<torch::Tensor, torch::Tensor>& b,
                                       const torch::Tensor& scratch, const torch::Tensor& ids,
                                       const torch::Tensor& tokens, const torch::Tensor& output,
                                       const torch::Tensor& queue, int task, bool assign,
                                       const std::optional<torch::Tensor>& row_end) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn and b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(a.second.scalar_type() == torch::kInt and b.second.scalar_type() == torch::kInt);
    for (const auto& t: {a.first, a.second, b.first, b.second, ids})
        DG_HOST_ASSERT(t.is_cuda() and t.device() == scratch.device());
    const auto epilogue = std::make_shared<ScatterAddEpilogue>(tokens, output, queue, task, assign, row_end);
    epilogue->check(std::nullopt, scratch);
    m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, scratch, ids, std::nullopt,
        std::tuple<int, int>{1, 32}, std::tuple<int, int>{1, 32},
        "nk", false, false, true, std::nullopt, epilogue);
}

static void bf16_grouped_scatter_gemm_nt(const torch::Tensor& a, const torch::Tensor& b,
                                       const torch::Tensor& scratch, const torch::Tensor& ids,
                                       const torch::Tensor& tokens, const torch::Tensor& output,
                                       const torch::Tensor& queue, int task, bool assign,
                                       const std::optional<torch::Tensor>& row_end) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    for (const auto& t: {a, b, ids})
        DG_HOST_ASSERT(t.is_cuda() and t.device() == scratch.device());
    const auto epilogue = std::make_shared<ScatterAddEpilogue>(tokens, output, queue, task, assign, row_end);
    epilogue->check(std::nullopt, scratch);
    m_grouped_bf16_gemm_nt_contiguous(a, b, scratch, ids, "nk", false, true,
                                     std::nullopt, epilogue);
}

static void bf16_ready_grouped_gemm_nt(const torch::Tensor& a, const torch::Tensor& b,
                                      const torch::Tensor& d, const torch::Tensor& ids,
                                      const torch::Tensor& queue,
                                      const std::shared_ptr<EpilogueClass>& epilogue_class, bool incremental) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10 or
                   (jit->device.get_arch_major() == 9 and incremental and
                    (epilogue_class == nullptr or
                     epilogue_class->get_epilogue_operator_type() == "epilogue::operators::SwiGLUForward" or
                     epilogue_class->get_epilogue_operator_type() == "epilogue::operators::SwiGLUBackward")));
    for (const auto& t: {a, b, ids, queue})
        DG_HOST_ASSERT(t.is_cuda() and t.device() == d.device());
    const auto ready = std::make_shared<ReadyEpilogue>(
        resolve_epilogue_class(epilogue_class, std::nullopt, d), queue, incremental);
    ready->check(std::nullopt, d);
    const auto sm90_swiglu_forward = jit->device.get_arch_major() == 9 and
        epilogue_class != nullptr and
        epilogue_class->get_epilogue_operator_type() == "epilogue::operators::SwiGLUForward";
    const auto gemm_output = sm90_swiglu_forward ? torch::empty_like(d) : d;
    m_grouped_bf16_gemm_nt_contiguous(a, b, gemm_output, ids, "nk", false, true,
                                     std::nullopt, ready);
}

static void bf16_chunk_gemm_nt(const torch::Tensor& a, const torch::Tensor& b,
                               const torch::Tensor& d, const torch::Tensor& task_queue,
                               const int task_idx, const std::string& compiled_dims,
                               const std::shared_ptr<EpilogueClass>& epilogue_class) {
    DG_HOST_ASSERT(a.is_cuda() and b.device() == a.device() and d.device() == a.device());
    DG_HOST_ASSERT(task_queue.device() == a.device() and task_queue.is_contiguous());
    DG_HOST_ASSERT(task_queue.scalar_type() == torch::kInt);
    const auto [tasks, fields] = get_shape<2>(task_queue);
    DG_HOST_ASSERT(fields == 4 and task_idx >= 0 and task_idx < tasks);
    const auto [m, k] = get_shape<2>(a);
    const auto [groups, n, bk] = get_shape<3>(b);
    const auto [dm, dn] = get_shape<2>(d);
    DG_HOST_ASSERT(m == dm and n == dn and k == bk and n > 0 and k > 0 and groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 and b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    check_major_type_cd(d);
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    if (m == 0)
        return;
    sm100_m_grouped_bf16_gemm_contiguous(a, b, d, task_queue, groups, m, n, k,
        major_a, major_b, compiled_dims, false, true, std::nullopt,
        resolve_epilogue_class(epilogue_class, std::nullopt, d), task_idx);
}

static void m_grouped_bf16_gemm_nn_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout,
                                              const bool& ensure_zero_padding,
                                              const std::shared_ptr<EpilogueClass>& epilogue_class) {
    m_grouped_bf16_gemm_nt_contiguous(a, b.transpose(1, 2),
                                      d, grouped_layout, compiled_dims, use_psum_layout, ensure_zero_padding, std::nullopt, epilogue_class);
}

static void m_grouped_bf16_gemm_nt_masked(const torch::Tensor& a, const torch::Tensor& b,
                                          const torch::Tensor& d, const torch::Tensor& masked_m,
                                          const int& expected_m, const std::string& compiled_dims,
                                          const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    // Type and shape checks
    const auto [num_groups, m, k] = get_shape<3>(a);
    const auto [num_groups_, n, k_] = get_shape<3>(b);
    const auto [num_groups__, m_, n_] = get_shape<3>(d);
    const auto num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
        sm90_bf16_m_grouped_gemm_masked(a, b, d, masked_m,
                                        num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_gemm_masked(a, b, d, masked_m,
                                         num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims,
                                         resolve_epilogue_class(epilogue_class, std::nullopt, d));
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_bf16_gemm_tn_contiguous(const torch::Tensor& a,
                                              const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const std::optional<std::vector<int>>& ks_cpu,
                                              const torch::Tensor& grouped_layout,
                                              const std::optional<torch::Tensor>& c,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout,
                                              const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Shape checks
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto [sum_k_ , m_] = get_shape<2>(a);
    const auto [sum_k__, n_] = get_shape<2>(b);

    const auto k_alignment = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
    DG_HOST_ASSERT(k_alignment % 128 == 0);
    const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups,
                                           use_psum_layout, k_alignment, static_cast<int>(a.size(0)));
    DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);

    // Contiguity checks
    DG_HOST_ASSERT(a.is_contiguous());
    DG_HOST_ASSERT(b.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(not c.has_value() or c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, sum_k, d, c))
        return;

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        // No direct output or psum on SM90
        DG_HOST_ASSERT(c.has_value());
        DG_HOST_ASSERT(not use_psum_layout and ks_cpu.has_value() and not ks_cpu.value().empty());
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
        sm90_bf16_k_grouped_gemm(a, b, c, d, m, n, ks_cpu.value(), grouped_layout,
                                 cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else if (arch_major == 10) {
        sm100_bf16_k_grouped_gemm(a, b, c, d, m, n, grouped_layout,
                                  cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims, use_psum_layout,
                                  resolve_epilogue_class(epilogue_class, c, d));
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_bf16_gemm_tn_gpu_offsets(const torch::Tensor& a,
                                                const torch::Tensor& b,
                                                const torch::Tensor& d,
                                                const torch::Tensor& starts,
                                                const torch::Tensor& ends,
                                                const int& expected_k,
                                                const std::optional<torch::Tensor>& c,
                                                const std::string& compiled_dims) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 9);
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto [pool_rows, m_] = get_shape<2>(a);
    const auto [pool_rows_, n_] = get_shape<2>(b);
    DG_HOST_ASSERT(m == m_ and n == n_ and pool_rows == pool_rows_);
    DG_HOST_ASSERT(pool_rows > 0 and expected_k > 0 and expected_k <= pool_rows);
    DG_HOST_ASSERT(c.has_value());
    DG_HOST_ASSERT(a.is_cuda() and b.is_cuda() and d.is_cuda() and c.value().is_cuda());
    DG_HOST_ASSERT(starts.is_cuda() and ends.is_cuda());
    DG_HOST_ASSERT(a.device() == b.device() and a.device() == d.device());
    DG_HOST_ASSERT(a.device() == c.value().device() and a.device() == starts.device());
    DG_HOST_ASSERT(a.device() == ends.device());
    DG_HOST_ASSERT(starts.is_contiguous() and ends.is_contiguous());
    DG_HOST_ASSERT(starts.scalar_type() == torch::kInt and ends.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(starts.numel() == num_groups and ends.numel() == num_groups);
    DG_HOST_ASSERT(a.is_contiguous() and b.is_contiguous() and d.is_contiguous());
    DG_HOST_ASSERT(c.value().is_contiguous());
    DG_HOST_ASSERT(c.value().data_ptr() == d.data_ptr());
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 and b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat and c.value().scalar_type() == torch::kFloat);
    // Partial K tiles read the caller-provided zero padding through the next 64-row boundary.
    sm90_bf16_k_grouped_gemm_gpu_offsets(a, b, c.value(), d, starts, ends, expected_k, compiled_dims);
}
static void cublaslt_gemm_nt(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a);
    const auto [n , k_] = get_shape<2>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    cublaslt_gemm(a, b, d, m, n, k, major_a, major_b, c.has_value());
}

static void cublaslt_gemm_nn(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a, b.transpose(0, 1), d, c);
}

static void cublaslt_gemm_tn(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c);
}

static void cublaslt_gemm_tt(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a.transpose(0, 1), b, d, c);
}

static void cublaslt_nvfp4_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                                   const std::pair<torch::Tensor, torch::Tensor>& b,
                                   const torch::Tensor& d,
                                   const std::optional<torch::Tensor>& c) {
    // Shape must be `[M, K] @ [N, K].T` with both operands packed FP4 and K-major
    DG_HOST_ASSERT(a.first.scalar_type() == kPackedFP4 and b.first.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(a.first.is_contiguous() and b.first.is_contiguous());

    // Type and shape checks
    // NOTES: shapes are in logical FP4 elements (2 elements per packed byte)
    const auto [m , packed_k ] = get_shape<2>(a.first);
    const auto [n , packed_k_] = get_shape<2>(b.first);
    const auto [m_, n_] = get_shape<2>(d);
    const auto k = packed_k * 2;
    DG_HOST_ASSERT(m == m_ and n == n_ and packed_k == packed_k_);
    DG_HOST_ASSERT(k % 32 == 0);

    // Scaling factors must be UE4M3 bytes (NVFP4 recipe, 16-element blocks) in the
    // cuBLASLt tiled layout: `[ceil(mn / 128), ceil(k / 64), 32, 4, 4]` in bytes
    const auto& check_sf = [&](const torch::Tensor& sf, const int& mn) {
        DG_HOST_ASSERT(sf.scalar_type() == torch::kByte or sf.scalar_type() == torch::kChar);
        DG_HOST_ASSERT(sf.is_contiguous());
        DG_HOST_ASSERT(sf.numel() == static_cast<int64_t>(ceil_div(mn, 128)) * ceil_div(k, 64) * 512);
    };
    check_sf(a.second, m);
    check_sf(b.second, n);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    cublaslt_nvfp4_gemm(a.first, a.second, b.first, b.second, d, m, n, k, c.has_value());
}

static auto get_cublaslt_batched_view(const torch::Tensor& tensor) {
    DG_HOST_ASSERT(tensor.dim() == 2 or tensor.dim() == 3);
    return tensor.dim() == 2 ? tensor.unsqueeze(0) : tensor;
}

static void batched_syrk(const torch::Tensor& a, const torch::Tensor& d) {
    // D = A @ A.mT. A and D may be either unbatched 2D tensors or batched 3D tensors.
    const auto a_3d = get_cublaslt_batched_view(a);
    const auto d_3d = get_cublaslt_batched_view(d);
    const auto [num_batches, m, k] = get_shape<3>(a_3d);
    const auto [num_batches_, m_, n_] = get_shape<3>(d_3d);
    const auto major_a = get_major_type_ab<false>(a_3d);
    check_major_type_cd<false>(d_3d);
    DG_HOST_ASSERT(num_batches == num_batches_ and m == m_ and m == n_);
    DG_HOST_ASSERT(a_3d.scalar_type() == d_3d.scalar_type());

    if (num_batches == 0 or early_return(m, m, k, d_3d, std::nullopt))
        return;
    cublaslt_batched_gemm(a_3d, a_3d, d_3d, m, m, k, num_batches, major_a, major_a);
}

static void batched_symm(const torch::Tensor& a, const torch::Tensor& b,
                         const torch::Tensor& d) {
    // D = A @ B. A is symmetric by contract; cuBLASLt executes this as strided-batched GEMM.
    const auto a_3d = get_cublaslt_batched_view(a);
    const auto b_3d = get_cublaslt_batched_view(b);
    const auto d_3d = get_cublaslt_batched_view(d);
    const auto [num_batches, m, m_] = get_shape<3>(a_3d);
    const auto [num_batches_b, m_b, k] = get_shape<3>(b_3d);
    const auto [num_batches_d, m_d, k_d] = get_shape<3>(d_3d);
    const auto major_a = get_major_type_ab<false>(a_3d);
    check_major_type_cd<false>(d_3d);
    DG_HOST_ASSERT(num_batches == num_batches_b and num_batches == num_batches_d);
    DG_HOST_ASSERT(m == m_ and m == m_b and m == m_d and k == k_d);
    DG_HOST_ASSERT(a_3d.scalar_type() == b_3d.scalar_type() and a_3d.scalar_type() == d_3d.scalar_type());

    if (num_batches == 0 or early_return(m, k, m, d_3d, std::nullopt))
        return;

    const auto b_transposed = b_3d.transpose(1, 2);
    const auto major_b = get_major_type_ab<false>(b_transposed);
    cublaslt_batched_gemm(a_3d, b_transposed, d_3d, m, k, m, num_batches, major_a, major_b);
}

static void register_apis(pybind11::module_& m) {

    // FP8 FP4 GEMMs
    m.def("k_grouped_fp8_gemm_nt_gpu_sizes", &k_grouped_fp8_gemm_nt_gpu_sizes);
    m.def("fp8_fp4_gemm_nt", &fp8_fp4_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("fp8_fp4_gemm_nn", &fp8_fp4_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("fp8_fp4_gemm_tn", &fp8_fp4_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("fp8_fp4_gemm_tt", &fp8_fp4_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("m_grouped_fp8_fp4_gemm_nt_contiguous", &m_grouped_fp8_fp4_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = false,
          py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("m_grouped_fp8_fp4_gemm_nn_contiguous", &m_grouped_fp8_fp4_gemm_nn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = false,
          py::arg("ensure_zero_padding") = true,
          py::arg("epilogue") = nullptr);
    m.def("m_grouped_fp8_fp4_gemm_nt_masked", &m_grouped_fp8_fp4_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false,
          py::arg("epilogue") = nullptr);
    m.def("k_grouped_fp8_gemm_tn_contiguous", &k_grouped_fp8_gemm_tn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("ks_cpu"), py::arg("grouped_layout"),
          py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 128),
          py::arg("compiled_dims") = "mn",
          py::arg("use_psum_layout") = false,
          py::arg("epilogue") = nullptr);
    m.def("k_grouped_fp8_gemm_nt_contiguous", &k_grouped_fp8_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("ks_cpu"), py::arg("grouped_layout"),
          py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 128),
          py::arg("compiled_dims") = "mn",
          py::arg("use_psum_layout") = false,
          py::arg("epilogue") = nullptr);
    m.def("k_grouped_fp4_gemm_nt_contiguous", &k_grouped_fp4_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("ks_cpu"), py::arg("grouped_layout"),
          py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 32),
          py::arg("compiled_dims") = "mn",
          py::arg("use_psum_layout") = false,
          py::arg("epilogue") = nullptr);

    // FP4 and FP8 GEMM alias names
    m.attr("fp4_gemm_nt") = m.attr("fp8_fp4_gemm_nt");
    m.attr("fp8_gemm_nt") = m.attr("fp8_fp4_gemm_nt");
    m.attr("fp8_gemm_nn") = m.attr("fp8_fp4_gemm_nn");
    m.attr("fp8_gemm_tn") = m.attr("fp8_fp4_gemm_tn");
    m.attr("fp8_gemm_tt") = m.attr("fp8_fp4_gemm_tt");
    m.attr("m_grouped_fp4_gemm_nt_contiguous") = m.attr("m_grouped_fp8_fp4_gemm_nt_contiguous");
    m.attr("m_grouped_fp8_gemm_nt_contiguous") = m.attr("m_grouped_fp8_fp4_gemm_nt_contiguous");
    m.attr("m_grouped_fp8_gemm_nn_contiguous") = m.attr("m_grouped_fp8_fp4_gemm_nn_contiguous");
    m.attr("m_grouped_fp4_gemm_nt_masked") = m.attr("m_grouped_fp8_fp4_gemm_nt_masked");
    m.attr("m_grouped_fp8_gemm_nt_masked") = m.attr("m_grouped_fp8_fp4_gemm_nt_masked");
    // BF16 GEMMs
    m.def("bf16_gemm_nt", &bf16_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("bf16_gemm_nn", &bf16_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("bf16_gemm_tn", &bf16_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("bf16_gemm_tt", &bf16_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("fp8_grouped_scatter_gemm_nt", &fp8_grouped_scatter_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("scratch"), py::arg("expert_ids"),
          py::arg("token_ids"), py::arg("output"), py::arg("queue"), py::arg("task"),
          py::arg("assign") = false, py::arg("row_end") = std::nullopt);
    m.def("fp8_ready_grouped_gemm_nt", &fp8_ready_grouped_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("expert_ids"),
          py::arg("queue"), py::arg("epilogue") = nullptr, py::arg("incremental") = false);
    m.def("fp8_chunk_gemm_nt", &fp8_chunk_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("task_queue"),
          py::arg("task_idx"), py::arg("gran_k") = 32, py::arg("compiled_dims") = "nk",
          py::arg("epilogue") = nullptr);
    m.def("bf16_grouped_scatter_gemm_nt", &bf16_grouped_scatter_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("scratch"), py::arg("expert_ids"),
          py::arg("token_ids"), py::arg("output"), py::arg("queue"), py::arg("task"),
          py::arg("assign") = false, py::arg("row_end") = std::nullopt);
    m.def("bf16_ready_grouped_gemm_nt", &bf16_ready_grouped_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("expert_ids"),
          py::arg("queue"), py::arg("epilogue") = nullptr, py::arg("incremental") = false);
    m.def("bf16_chunk_gemm_nt", &bf16_chunk_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("task_queue"),
          py::arg("task_idx"), py::arg("compiled_dims") = "nk",
          py::arg("epilogue") = nullptr);
    m.def("m_grouped_bf16_gemm_nt_contiguous", &m_grouped_bf16_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("compiled_dims") = "nk",
          py::arg("use_psum_layout") = false,
          py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("m_grouped_bf16_gemm_nn_contiguous", &m_grouped_bf16_gemm_nn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("compiled_dims") = "nk",
          py::arg("use_psum_layout") = false,
          py::arg("ensure_zero_padding") = true,
          py::arg("epilogue") = nullptr);
    m.def("m_grouped_bf16_gemm_nt_masked", &m_grouped_bf16_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("compiled_dims") = "nk",
          py::arg("epilogue") = nullptr);
    m.def("k_grouped_bf16_gemm_tn_contiguous", &k_grouped_bf16_gemm_tn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("ks_cpu"), py::arg("grouped_layout"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("use_psum_layout") = false,
          py::arg("epilogue") = nullptr);
    m.def("k_grouped_bf16_gemm_tn_gpu_offsets", &k_grouped_bf16_gemm_tn_gpu_offsets,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("starts"), py::arg("ends"),
          py::arg("expected_k"), py::arg("c") = std::nullopt, py::arg("compiled_dims") = "mn");
    // cuBLASLt GEMMs
    m.def("cublaslt_gemm_nt", &cublaslt_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("cublaslt_gemm_nn", &cublaslt_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("cublaslt_gemm_tn", &cublaslt_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("cublaslt_gemm_tt", &cublaslt_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("cublaslt_nvfp4_gemm_nt", &cublaslt_nvfp4_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("batched_syrk", &batched_syrk,
          py::arg("a"), py::arg("d"));
    m.def("batched_symm", &batched_symm,
          py::arg("a"), py::arg("b"), py::arg("d"));
}

} // namespace deep_gemm::gemm
