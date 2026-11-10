// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
//
// Gated activations for the mono-ep expert epilogues: out = f(x) * g(y), where x is the gate
// half and y the up half of the first projection. The tanh-form and FFMA-factored gradients follow
// the formulations in Dao-AILab/quack (quack/activation.py); nothing here depends on quack.
#pragma once

#include <cstdint>

namespace deep_gemm::mono_ep_act {

enum class Activation : uint32_t {
    SwiGLU = 0,      // silu(x) * y, IEEE expf + divide (the original epilogue numerics)
    SwiGLUFast = 1,  // silu(x) * y, sigmoid from ex2.approx + rcp.approx
    SwiGLUTanh = 2,  // silu(x) * y, sigmoid = 0.5 * tanh.approx(0.5 x) + 0.5
    SwiGLUOAI = 3,   // x * sigmoid(alpha x) * (y + 1)
    GeGLU = 4,       // gelu_erf(x) * y
    GeGLUTanh = 5,   // gelu_tanh(x) * y, tanh.approx
    ReGLU = 6,       // relu(x) * y
    GLU = 7,         // sigmoid(x) * y
};

constexpr uint32_t kNumActivations = 8;
// FLT_MAX: the "no clamp" limit. Clamps are x <= limit and -limit <= y <= limit.
constexpr uint32_t kNoLimitBits = 0x7f7fffffu;
// 1.702f, the GPT-OSS SwiGLU alpha.
constexpr uint32_t kDefaultAlphaBits = 0x3fd9db23u;

constexpr bool is_swiglu_family(const Activation a) {
    return a == Activation::SwiGLU or a == Activation::SwiGLUFast or a == Activation::SwiGLUTanh;
}

// Whether the activation applies the gate/up clamp (SwiGLU forms and OAI only).
constexpr bool uses_limit(const Activation a) {
    return is_swiglu_family(a) or a == Activation::SwiGLUOAI;
}

__device__ __forceinline__ float tanh_approx(const float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float rcp_approx(const float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float sigmoid_exact(const float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_fast(const float x) { return rcp_approx(1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoid_tanh(const float x) { return fmaf(0.5f, tanh_approx(0.5f * x), 0.5f); }

__device__ __forceinline__ float clamp_gate(const float x, const float limit) { return fminf(x, limit); }
__device__ __forceinline__ float clamp_up(const float y, const float limit) { return fmaxf(fminf(y, limit), -limit); }

// Result of a gated-activation backward. out = f * g; dscore-style terms use (dout_raw * f) * g.
struct Grad {
    float dx, dy, out, f, g;
};

template <Activation kAct>
struct Gated;

// SwiGLU forms. kClamp applies the one-sided gate clamp and two-sided up clamp; their gradients are
// zeroed where clamped, including on the boundary itself, as torch.clamp's backward does.
template <Activation kAct>
struct SwiGLUImpl {
    __device__ __forceinline__ static float sigmoid(const float x) {
        if constexpr (kAct == Activation::SwiGLU)
            return sigmoid_exact(x);
        else if constexpr (kAct == Activation::SwiGLUFast)
            return sigmoid_fast(x);
        else
            return sigmoid_tanh(x);
    }

    // Inputs are already clamped by the caller (the forward stores the clamped z).
    __device__ __forceinline__ static float forward(const float x, const float y, float, float) {
        if constexpr (kAct == Activation::SwiGLUTanh) {
            const float x_half = 0.5f * x;
            return fmaf(x_half, tanh_approx(x_half), x_half) * y;
        } else {
            return x * sigmoid(x) * y;
        }
    }

    template <bool kClamp>
    __device__ __forceinline__ static Grad backward(const float x_raw, const float y_raw, const float dout,
                                                    float, const float limit) {
        const float x = kClamp ? clamp_gate(x_raw, limit) : x_raw;
        const float y = kClamp ? clamp_up(y_raw, limit) : y_raw;
        const float s = sigmoid(x);
        const float silu = x * s;
        Grad r;
        if constexpr (kAct == Activation::SwiGLU) {
            // Same operation order as the original epilogue: ((dout * y) * dsilu), dout * silu, silu * y.
            r.dx = dout * y * (s + silu * (1.0f - s));
            r.dy = dout * silu;
        } else {
            // dsilu * dout = (s - silu * s) * dout + silu * dout: 3 FFMA, no 1 - s.
            const float silu_dout = silu * dout;
            r.dx = fmaf(fmaf(-silu, s, s), dout, silu_dout) * y;
            r.dy = silu_dout;
        }
        r.f = silu;
        r.g = y;
        r.out = silu * y;
        if constexpr (kClamp) {
            r.dx = x_raw < limit ? r.dx : 0.0f;
            r.dy = fabsf(y_raw) < limit ? r.dy : 0.0f;
        }
        return r;
    }
};

template <> struct Gated<Activation::SwiGLU> : SwiGLUImpl<Activation::SwiGLU> {};
template <> struct Gated<Activation::SwiGLUFast> : SwiGLUImpl<Activation::SwiGLUFast> {};
template <> struct Gated<Activation::SwiGLUTanh> : SwiGLUImpl<Activation::SwiGLUTanh> {};

// GPT-OSS: x = clamp(x, max=limit), y = clamp(y, -limit, limit), out = x * sigmoid(alpha x) * (y + 1).
template <>
struct Gated<Activation::SwiGLUOAI> {
    __device__ __forceinline__ static float forward(const float x, const float y, const float alpha, float) {
        return x * sigmoid_exact(alpha * x) * (y + 1.0f);
    }

    template <bool kClamp>
    __device__ __forceinline__ static Grad backward(const float x_raw, const float y_raw, const float dout,
                                                    const float alpha, const float limit) {
        const float x = kClamp ? clamp_gate(x_raw, limit) : x_raw;
        const float y = kClamp ? clamp_up(y_raw, limit) : y_raw;
        const float s = sigmoid_exact(alpha * x);
        const float f = x * s;
        const float g = y + 1.0f;
        // d/dx [x s(ax)] = s + a x s (1 - s) = s + a (f - f s)
        const float df = fmaf(alpha, fmaf(-f, s, f), s);
        Grad r;
        r.dx = dout * g * df;
        r.dy = dout * f;
        r.f = f;
        r.g = g;
        r.out = f * g;
        if constexpr (kClamp) {
            r.dx = x_raw < limit ? r.dx : 0.0f;
            r.dy = fabsf(y_raw) < limit ? r.dy : 0.0f;
        }
        return r;
    }
};

// GELU with erf: 0.5 x (1 + erf(x / sqrt 2)).
template <>
struct Gated<Activation::GeGLU> {
    __device__ __forceinline__ static float forward(const float x, const float y, float, float) {
        return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)) * y;
    }

    template <bool>
    __device__ __forceinline__ static Grad backward(const float x, const float y, const float dout, float, float) {
        const float cdf = 0.5f * (1.0f + erff(x * 0.70710678118654752f));
        const float pdf = 0.39894228040143268f * expf(-0.5f * x * x);
        const float f = x * cdf;
        Grad r;
        r.dx = dout * y * fmaf(x, pdf, cdf);
        r.dy = dout * f;
        r.f = f;
        r.g = y;
        r.out = f * y;
        return r;
    }
};

// GELU tanh approximation: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
template <>
struct Gated<Activation::GeGLUTanh> {
    static constexpr float kC0 = 0.79788456080286536f;              // sqrt(2 / pi)
    static constexpr float kC1 = 0.044715f * 0.79788456080286536f;
    static constexpr float kC1x3 = 3.0f * 0.044715f * 0.79788456080286536f;

    __device__ __forceinline__ static float forward(const float x, const float y, float, float) {
        const float t = tanh_approx(x * fmaf(kC1, x * x, kC0));
        return x * fmaf(0.5f, t, 0.5f) * y;
    }

    template <bool>
    __device__ __forceinline__ static Grad backward(const float x, const float y, const float dout, float, float) {
        const float x_sq = x * x;
        const float t = tanh_approx(x * fmaf(kC1, x_sq, kC0));
        const float half_t_plus_half = fmaf(0.5f, t, 0.5f);
        const float f = x * half_t_plus_half;
        // d gelu = 0.5 (1 + t) + 0.5 x (1 - t^2) (c0 + 3 c1 x^2)
        const float df = fmaf(0.5f * x * fmaf(-t, t, 1.0f), fmaf(kC1x3, x_sq, kC0), half_t_plus_half);
        Grad r;
        r.dx = dout * y * df;
        r.dy = dout * f;
        r.f = f;
        r.g = y;
        r.out = f * y;
        return r;
    }
};

template <>
struct Gated<Activation::ReGLU> {
    __device__ __forceinline__ static float forward(const float x, const float y, float, float) {
        return fmaxf(x, 0.0f) * y;
    }

    template <bool>
    __device__ __forceinline__ static Grad backward(const float x, const float y, const float dout, float, float) {
        const float f = fmaxf(x, 0.0f);
        Grad r;
        r.dx = x > 0.0f ? dout * y : 0.0f;
        r.dy = dout * f;
        r.f = f;
        r.g = y;
        r.out = f * y;
        return r;
    }
};

template <>
struct Gated<Activation::GLU> {
    __device__ __forceinline__ static float forward(const float x, const float y, float, float) {
        return sigmoid_exact(x) * y;
    }

    template <bool>
    __device__ __forceinline__ static Grad backward(const float x, const float y, const float dout, float, float) {
        const float s = sigmoid_exact(x);
        Grad r;
        r.dx = dout * y * fmaf(-s, s, s);
        r.dy = dout * s;
        r.f = s;
        r.g = y;
        r.out = s * y;
        return r;
    }
};

// Entry points. The forward clamps (when the activation uses a limit) before the activation so the
// caller can store the clamped inputs; kClamp = false compiles the clamp out entirely.
template <uint32_t kActivation, bool kClamp>
__device__ __forceinline__ float forward(const float x, const float y, const float alpha, const float limit) {
    constexpr auto kAct = static_cast<Activation>(kActivation);
    constexpr bool kApply = kClamp and uses_limit(kAct);
    return Gated<kAct>::forward(kApply ? clamp_gate(x, limit) : x, kApply ? clamp_up(y, limit) : y, alpha, limit);
}

template <uint32_t kActivation, bool kClamp>
__device__ __forceinline__ Grad backward(const float x, const float y, const float dout,
                                         const float alpha, const float limit) {
    constexpr auto kAct = static_cast<Activation>(kActivation);
    return Gated<kAct>::template backward<kClamp and uses_limit(kAct)>(x, y, dout, alpha, limit);
}

} // namespace deep_gemm::mono_ep_act
