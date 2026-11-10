// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#include <pybind11/pybind11.h>

#include <cstdint>

#include "nvfp4_requant_args.h"

int nvfp4_requant_launch(const mono_ep_requant::Args& args, uintptr_t stream_ptr);
int nvfp4_requant_probe(int kind, uintptr_t in, uintptr_t out, int64_t n, uintptr_t stream_ptr);

namespace {

template <typename T>
T* ptr(uintptr_t p) {
    return reinterpret_cast<T*>(p);
}

int launch(pybind11::dict d, uintptr_t stream) {
    auto u = [&](const char* k) { return d[k].cast<uintptr_t>(); };
    auto i = [&](const char* k) { return d[k].cast<int>(); };
    mono_ep_requant::Args a{};
    a.q = ptr<const uint8_t>(u("q"));
    a.s = ptr<const uint8_t>(u("s"));
    a.dq_pts = ptr<const float>(u("dq_pts"));
    a.a = ptr<const void>(u("a"));
    a.b = ptr<const void>(u("b"));
    a.fq_pts = ptr<const float>(u("fq_pts"));
    a.oq = ptr<uint8_t>(u("oq"));
    a.os = ptr<uint8_t>(u("os"));
    a.osfa = ptr<uint8_t>(u("osfa"));
    a.qt = ptr<const uint8_t>(u("qt"));
    a.st = ptr<const uint8_t>(u("st"));
    a.oq_t = ptr<uint8_t>(u("oq_t"));
    a.os_t = ptr<uint8_t>(u("os_t"));
    a.osfa_t = ptr<uint8_t>(u("osfa_t"));
    a.scaling = d["scaling"].cast<float>();
    a.e0 = i("e0");
    a.e_local = i("e_local");
    a.n = i("n");
    a.k = i("k");
    a.r = i("r");
    a.r_pad = i("r_pad");
    a.p = i("p");
    a.rows_per_proj = i("rows_per_proj");
    a.rm = i("rm");
    a.rk = i("rk");
    a.rm_t = i("rm_t");
    a.rk_t = i("rk_t");
    a.interleave = i("interleave");
    a.round_bf16 = i("round_bf16");
    a.modelopt = i("modelopt");
    a.reuse = i("reuse");
    a.ab_fp32 = i("ab_fp32");
    a.pair = i("pair");
    a.ffma2 = i("ffma2");
    a.mode = i("mode");
    a.k_splits = i("k_splits");
    return nvfp4_requant_launch(a, stream);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch);
    m.def("probe", &nvfp4_requant_probe);
}
