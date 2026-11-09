#pragma once

#include <memory>

#include <pybind11/pybind11.h>

#include "../jit_kernels/impls/epilogue_class.hpp"

namespace deep_gemm::epilogue_class {

// User-facing epilogue classes for the `epilogue` argument of the GEMM APIs, as `deep_gemm.epilogue.X`
// NOTES: `alpha` and the FP8 `(d, sfd)` output pair are the shorthand for `Alpha` and unclamped `FP8CastWithClamp`
static void register_apis(pybind11::module_& m) {
    const auto submodule = m.def_submodule("epilogue", "GEMM epilogue classes");
    pybind11::class_<EpilogueClass, std::shared_ptr<EpilogueClass>>(submodule, "Epilogue");
    pybind11::class_<SwiGLUForwardEpilogue, EpilogueClass, std::shared_ptr<SwiGLUForwardEpilogue>>(submodule, "SwiGLUForward")
        .def(pybind11::init<const torch::Tensor&, const torch::Tensor&, const torch::Tensor&>(),
             pybind11::arg("z"), pybind11::arg("scores"), pybind11::arg("h"));
    pybind11::class_<SwiGLURecomputeBackwardEpilogue, EpilogueClass, std::shared_ptr<SwiGLURecomputeBackwardEpilogue>>(submodule, "SwiGLURecomputeBackward")
        .def(pybind11::init<const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&>(),
             pybind11::arg("dh"), pybind11::arg("scores"), pybind11::arg("h"), pybind11::arg("dz"), pybind11::arg("ds"));
    pybind11::class_<SwiGLUBackwardEpilogue, EpilogueClass, std::shared_ptr<SwiGLUBackwardEpilogue>>(submodule, "SwiGLUBackward")
        .def(pybind11::init<const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&>(),
             pybind11::arg("z"), pybind11::arg("scores"), pybind11::arg("dz"), pybind11::arg("ds"));
    pybind11::class_<PeerScatterStoreEpilogue, EpilogueClass, std::shared_ptr<PeerScatterStoreEpilogue>>(submodule, "PeerScatterStoreBF16")
        .def(pybind11::init<const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, int>(),
             pybind11::arg("assignments"), pybind11::arg("pointers"), pybind11::arg("queue"), pybind11::arg("rows_per_peer"));
    pybind11::class_<IdentityEpilogue, EpilogueClass, std::shared_ptr<IdentityEpilogue>>(submodule, "Identity")
        .def(pybind11::init<>());
    pybind11::class_<AlphaEpilogue, EpilogueClass, std::shared_ptr<AlphaEpilogue>>(submodule, "Alpha")
        .def(pybind11::init<const float&>(), pybind11::arg("alpha"));
    pybind11::class_<FP8CastWithClampEpilogue, EpilogueClass, std::shared_ptr<FP8CastWithClampEpilogue>>(
        submodule, "FP8CastWithClamp",
        "Cast BF16-rounded GEMM output to FP8, optionally clamping before scaling. "
        "clamped_count is an optional single-element CUDA uint32 tensor; "
        "the kernel writes the total asynchronously when provided.")
        .def(pybind11::init<const torch::Tensor&, const std::optional<float>&, const std::optional<torch::Tensor>&>(),
             pybind11::arg("sfd"), pybind11::arg("clamp_value") = std::nullopt,
             pybind11::arg("clamped_count") = std::nullopt);
    pybind11::class_<BF16StochasticRoundingEpilogue, EpilogueClass,
                     std::shared_ptr<BF16StochasticRoundingEpilogue>>(submodule, "BF16StochasticRounding")
        .def(pybind11::init<>());
}

} // namespace deep_gemm::epilogue_class
