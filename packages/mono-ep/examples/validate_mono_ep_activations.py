# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Elementwise check of deep_gemm/impls/mono_ep_activations.cuh against float64 torch autograd.

Every activation's forward and backward (including the clamp masks, which must match torch.clamp's
backward: the gradient is zero on the boundary too) is compiled into a small CUDA kernel and compared on
fp32 inputs that include exact boundary values. Prints one JSON line.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

ACTIVATIONS = {
    "swiglu": 0,
    "swiglu_fast": 1,
    "swiglu_tanh": 2,
    "swiglu_oai": 3,
    "geglu": 4,
    "geglu_tanh": 5,
    "reglu": 6,
    "glu": 7,
}
USES_LIMIT = {"swiglu", "swiglu_fast", "swiglu_tanh", "swiglu_oai"}
# Relative-norm and max-abs/absmax tolerances. Exact forms use IEEE expf/erff and divides; the fast
# form uses ex2.approx + rcp.approx; the tanh forms use tanh.approx.f32 (max relative error 2^-10.99).
TOLERANCE = {
    "swiglu": 1e-6,
    "swiglu_oai": 1e-6,
    "geglu": 1e-6,
    "reglu": 1e-6,
    "glu": 1e-6,
    "swiglu_fast": 1e-5,
    "swiglu_tanh": 4e-3,
    "geglu_tanh": 4e-3,
}

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <deep_gemm/impls/mono_ep_activations.cuh>

namespace act = deep_gemm::mono_ep_act;

template <uint32_t kAct, bool kClamp>
__global__ void activation_kernel(const float* x, const float* y, const float* dout, float* fwd, float* dx,
                                  float* dy, float* out, float* fg, int64_t n, float alpha, float limit) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    fwd[i] = act::forward<kAct, kClamp>(x[i], y[i], alpha, limit);
    const auto g = act::backward<kAct, kClamp>(x[i], y[i], dout[i], alpha, limit);
    dx[i] = g.dx;
    dy[i] = g.dy;
    out[i] = g.out;
    fg[i] = g.f * g.g;
}

template <uint32_t kAct, bool kClamp>
void launch(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& dout,
            std::vector<torch::Tensor>& r, float alpha, float limit) {
    const int64_t n = x.numel();
    activation_kernel<kAct, kClamp><<<(n + 255) / 256, 256>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), dout.data_ptr<float>(), r[0].data_ptr<float>(),
        r[1].data_ptr<float>(), r[2].data_ptr<float>(), r[3].data_ptr<float>(), r[4].data_ptr<float>(), n,
        alpha, limit);
}

template <uint32_t kAct>
void dispatch_clamp(bool clamp, const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& dout,
                    std::vector<torch::Tensor>& r, float alpha, float limit) {
    if (clamp)
        launch<kAct, true>(x, y, dout, r, alpha, limit);
    else
        launch<kAct, false>(x, y, dout, r, alpha, limit);
}

std::vector<torch::Tensor> run(torch::Tensor x, torch::Tensor y, torch::Tensor dout, int64_t kind, bool clamp,
                               double alpha, double limit) {
    std::vector<torch::Tensor> r;
    for (int i = 0; i < 5; ++i)
        r.push_back(torch::empty_like(x));
    const float a = static_cast<float>(alpha), l = static_cast<float>(limit);
    switch (kind) {
        case 0: dispatch_clamp<0>(clamp, x, y, dout, r, a, l); break;
        case 1: dispatch_clamp<1>(clamp, x, y, dout, r, a, l); break;
        case 2: dispatch_clamp<2>(clamp, x, y, dout, r, a, l); break;
        case 3: dispatch_clamp<3>(clamp, x, y, dout, r, a, l); break;
        case 4: dispatch_clamp<4>(clamp, x, y, dout, r, a, l); break;
        case 5: dispatch_clamp<5>(clamp, x, y, dout, r, a, l); break;
        case 6: dispatch_clamp<6>(clamp, x, y, dout, r, a, l); break;
        case 7: dispatch_clamp<7>(clamp, x, y, dout, r, a, l); break;
        default: TORCH_CHECK(false, "bad activation");
    }
    return r;
}
"""

CPP_SOURCE = (
    "std::vector<torch::Tensor> run(torch::Tensor x, torch::Tensor y, torch::Tensor dout, "
    "int64_t kind, bool clamp, double alpha, double limit);"
)


def reference(name, x, y, dout, clamp, alpha, limit):
    x = x.double().requires_grad_(True)
    y = y.double().requires_grad_(True)
    xc = x.clamp(max=limit) if clamp and name in USES_LIMIT else x
    yc = y.clamp(-limit, limit) if clamp and name in USES_LIMIT else y
    if name.startswith("swiglu") and name != "swiglu_oai":
        out = F.silu(xc) * yc
    elif name == "swiglu_oai":
        out = xc * torch.sigmoid(alpha * xc) * (yc + 1)
    elif name == "geglu":
        out = F.gelu(xc) * yc
    elif name == "geglu_tanh":
        out = F.gelu(xc, approximate="tanh") * yc
    elif name == "reglu":
        out = F.relu(xc) * yc
    else:
        out = torch.sigmoid(xc) * yc
    out.backward(dout.double())
    return out.detach(), x.grad, y.grad


def error(a, b):
    diff = (a.double() - b).abs()
    return dict(
        rel_norm=float(diff.norm() / b.norm().clamp_min(1e-300)),
        max_abs_over_absmax=float(diff.max() / b.abs().max().clamp_min(1e-300)),
        finite=bool(torch.isfinite(a).all()),
    )


def main(include_dir, n):
    device_name = torch.cuda.get_device_name()
    module = load_inline(
        "mono_ep_activations_check",
        CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=["run"],
        extra_include_paths=[include_dir],
        extra_cuda_cflags=["-O3", "-std=c++20", "--gpu-architecture=sm_90a"],
        verbose=False,
    )
    torch.manual_seed(1234)
    results, passed = {}, True
    for name, kind in ACTIVATIONS.items():
        cases = [("noclamp", False, 1.702, 3.4028234663852886e38)]
        if name in USES_LIMIT:
            cases.append(("clamp", True, 1.702, 7.0 if name == "swiglu_oai" else 3.0))
        for case, clamp, alpha, limit in cases:
            x = torch.randn(n, device="cuda") * 4
            y = torch.randn(n, device="cuda") * 4
            dout = torch.randn(n, device="cuda")
            if clamp:
                # Exact boundary values: torch.clamp's gradient is zero at x == limit and |y| == limit.
                x[:64] = limit
                y[64:128] = limit
                y[128:192] = -limit
                # One float32 ulp inside the clamp: gradients pass.
                x[192:256] = limit * (1 - 2**-23)
                y[256:320] = -limit * (1 - 2**-23)
            fwd, dx, dy, out, fg = module.run(x, y, dout, kind, clamp, alpha, limit)
            torch.cuda.synchronize()
            ref_out, ref_dx, ref_dy = reference(name, x, y, dout, clamp, alpha, limit)
            record = dict(
                forward=error(fwd, ref_out),
                backward_out=error(out, ref_out),
                f_times_g=error(fg, ref_out),
                dx=error(dx, ref_dx),
                dy=error(dy, ref_dy),
            )
            if clamp:
                # Masks: where torch's gradient is exactly zero from the clamp, ours must be too.
                gate_masked = x.double() >= limit
                up_masked = y.double().abs() >= limit
                record["clamp_mask"] = dict(
                    gate_masked=int(gate_masked.sum()),
                    up_masked=int(up_masked.sum()),
                    gate_mask_match=bool(
                        (dx[gate_masked] == 0).all()
                        and (ref_dx[gate_masked] == 0).all()
                    ),
                    up_mask_match=bool(
                        (dy[up_masked] == 0).all() and (ref_dy[up_masked] == 0).all()
                    ),
                    boundary_grad_zero=bool(
                        (dx[:64] == 0).all() and (dy[64:192] == 0).all()
                    ),
                )
            tol = TOLERANCE[name]
            ok = all(
                v["finite"] and v["rel_norm"] <= tol and v["max_abs_over_absmax"] <= tol
                for k, v in record.items()
                if k != "clamp_mask"
            )
            if clamp:
                ok = ok and all(
                    record["clamp_mask"][k]
                    for k in ("gate_mask_match", "up_mask_match", "boundary_grad_zero")
                )
            record.update(
                tolerance=tol, passed=ok, alpha=alpha, limit=limit if clamp else None
            )
            results[f"{name}/{case}"] = record
            passed = passed and ok
    print(
        "ACT_JSON "
        + json.dumps(dict(device=device_name, n=n, passed=passed, results=results)),
        flush=True,
    )
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-dir",
        default=str(
            Path(__file__).resolve().parents[1] / "vendor/deepgemm/deep_gemm/include"
        ),
    )
    parser.add_argument("--n", type=int, default=1 << 22)
    args = parser.parse_args()
    raise SystemExit(0 if main(args.include_dir, args.n) else 1)
