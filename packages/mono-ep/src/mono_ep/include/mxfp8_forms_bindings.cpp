// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#include <pybind11/pybind11.h>

#include <cstdint>

int mxfp8_forms_launch(uintptr_t w, int dtype, int experts, int rows, int cols, int row_form, int interleave,
                       int t_form, uintptr_t qr, uintptr_t sr, uintptr_t qt, uintptr_t st, int debug, uintptr_t stream_ptr);

int mxfp8_lora_forms_launch(uintptr_t w, uintptr_t b, uintptr_t a, int rank, float scaling, int experts, int rows,
                            int cols, int interleave, uintptr_t qr, uintptr_t sr, uintptr_t qt, uintptr_t st,
                            uintptr_t stream_ptr);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &mxfp8_forms_launch);
    m.def("lora_launch", &mxfp8_lora_forms_launch);
}
