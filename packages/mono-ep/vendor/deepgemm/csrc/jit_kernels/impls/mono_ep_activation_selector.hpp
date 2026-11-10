// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <string>
#include <string_view>

#include "../../utils/exception.hpp"

namespace deep_gemm {

// Host-side mirror of mono_ep_act::Activation (deep_gemm/impls/mono_ep_activations.cuh).
struct MonoEPActivation {
    uint32_t kind = 0;
    uint32_t alpha_bits = 0x3fd9db23u;  // 1.702f
    uint32_t limit_bits = 0x7f7fffffu;  // FLT_MAX: no clamp
    bool is_default() const { return kind == 0 and limit_bits == 0x7f7fffffu; }
    // Suffix for JIT kernel names; empty for the default SwiGLU.
    std::string tag() const {
        return is_default() ? std::string() : std::format("_act{}_a{:08x}_l{:08x}", kind, alpha_bits, limit_bits);
    }
};

// MONO_EP_SM90_FP8_ACTIVATION = swiglu | swiglu_fast | swiglu_tanh | swiglu_oai | geglu | geglu_tanh |
// reglu | glu; unset selects swiglu_fast when fast_default (512-thread launches), else swiglu. swiglu is the
// bitwise reference (IEEE expf + divide); swiglu_fast passed the 2xH100 validators and a loss-curve check.
// _ACTIVATION_ALPHA sets swiglu_oai's alpha (1.702). _ACTIVATION_LIMIT clamps the gate above and the up half
// on both sides for the SwiGLU forms and swiglu_oai (swiglu_oai defaults to 7).
static MonoEPActivation select_mono_ep_activation(const bool fast_default = false) {
    constexpr std::string_view kNames[] = {
        "swiglu", "swiglu_fast", "swiglu_tanh", "swiglu_oai", "geglu", "geglu_tanh", "reglu", "glu"};
    MonoEPActivation result;
    result.kind = fast_default ? 1 : 0;
    const char* name = std::getenv("MONO_EP_SM90_FP8_ACTIVATION");
    if (name != nullptr and *name != '\0') {
        bool found = false;
        for (uint32_t i = 0; i < sizeof(kNames) / sizeof(kNames[0]); ++i) {
            if (kNames[i] == name) {
                result.kind = i;
                found = true;
            }
        }
        DG_HOST_ASSERT(found and "unknown MONO_EP_SM90_FP8_ACTIVATION");
    }
    const bool oai = result.kind == 3;
    const bool uses_limit = result.kind <= 3;
    const auto parse = [](const char* env) -> const char* {
        const char* value = std::getenv(env);
        return value != nullptr and *value != '\0' ? value : nullptr;
    };
    if (const char* alpha = parse("MONO_EP_SM90_FP8_ACTIVATION_ALPHA")) {
        DG_HOST_ASSERT(oai and "MONO_EP_SM90_FP8_ACTIVATION_ALPHA applies to swiglu_oai only");
        const float value = std::strtof(alpha, nullptr);
        DG_HOST_ASSERT(std::isfinite(value));
        result.alpha_bits = std::bit_cast<uint32_t>(value);
    }
    if (oai)
        result.limit_bits = std::bit_cast<uint32_t>(7.0f);
    if (const char* limit = parse("MONO_EP_SM90_FP8_ACTIVATION_LIMIT")) {
        DG_HOST_ASSERT(uses_limit and "MONO_EP_SM90_FP8_ACTIVATION_LIMIT applies to the SwiGLU forms only");
        const float value = std::strtof(limit, nullptr);
        DG_HOST_ASSERT(value > 0.0f);
        result.limit_bits = std::isfinite(value) ? std::bit_cast<uint32_t>(value) : 0x7f7fffffu;
    }
    return result;
}

} // namespace deep_gemm
