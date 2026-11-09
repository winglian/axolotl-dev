// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Axolotl AI
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
int launch(std::vector<uintptr_t>, int, int, int, int, uintptr_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
