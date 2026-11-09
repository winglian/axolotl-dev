#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include <torch/python.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <deep_gemm/common/types.cuh>

#include "../../runtime/jit.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

// The GEMM epilogue type: selects the device-side `epilogue::operators` implementation,
// carries its runtime state, and validates the GEMM output against it
// NOTES: every SM100 GEMM launches with exactly one epilogue class; they do not compose
class EpilogueClass {
public:
    virtual ~EpilogueClass() = default;

    virtual std::string get_epilogue_operator_type() const = 0;

    virtual EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const { return {}; }

    virtual deep_jit::cuda::CompilerOptions compiler_options() const { return {}; }

    virtual void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const {
        DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    }

    // The BLAS linear-combination coefficient, when this epilogue is a plain store cuBLASLt
    // can compute directly; `nullopt` when cuBLASLt cannot handle it
    virtual std::optional<float> get_alpha() const { return std::nullopt; }

    // The SFD this epilogue quantizes into, if any (an empty problem zeroes it)
    virtual std::optional<torch::Tensor> output_sfd() const { return std::nullopt; }
};

class ReadyEpilogue final: public EpilogueClass {
    std::shared_ptr<EpilogueClass> inner;
    torch::Tensor queue;
    bool incremental;
public:
    ReadyEpilogue(std::shared_ptr<EpilogueClass> inner, const torch::Tensor& queue, bool incremental = false):
        inner(std::move(inner)), queue(queue), incremental(incremental) {}

    bool is_incremental_sm90_supported() const {
        const auto type = inner->get_epilogue_operator_type();
        return incremental and (type == "epilogue::operators::Identity" or
                                type == "epilogue::operators::SwiGLUForward" or
                                type == "epilogue::operators::SwiGLUBackward");
    }

    bool is_incremental_sm90_swiglu_backward() const {
        return incremental and
               inner->get_epilogue_operator_type() == "epilogue::operators::SwiGLUBackward";
    }

    bool is_incremental_sm90_swiglu() const {
        const auto type = inner->get_epilogue_operator_type();
        return incremental and (type == "epilogue::operators::SwiGLUForward" or
                                type == "epilogue::operators::SwiGLUBackward");
    }

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::Ready<" + inner->get_epilogue_operator_type() +
            (incremental ? ", true>" : ", false>");
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        auto args = inner->make_epilogue_operator_args(m, n);
        args.ready_queue = queue.data_ptr<int>();
        args.ready_tasks = static_cast<uint32_t>(queue.size(0));
        return args;
    }

    deep_jit::cuda::CompilerOptions compiler_options() const override {
        return inner->compiler_options();
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        inner->check(c, d);
        DG_HOST_ASSERT(queue.is_cuda() and queue.device() == d.device() and queue.is_contiguous());
        DG_HOST_ASSERT(queue.scalar_type() == torch::kInt and queue.dim() == 2);
        DG_HOST_ASSERT(queue.size(0) > 0 and queue.size(1) == 4);
    }
};

class IdentityEpilogue final: public EpilogueClass {
public:
    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::Identity";
    }

    std::optional<float> get_alpha() const override { return 1.0f; }
};

class AlphaEpilogue final: public EpilogueClass {
    float alpha;

public:
    explicit AlphaEpilogue(const float& alpha): alpha(alpha) {}

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::ScaleByAlpha";
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int&, const int&) const override {
        return {.alpha = alpha};
    }

    std::optional<float> get_alpha() const override { return alpha; }
};

class FP8CastWithClampEpilogue final: public EpilogueClass {
    torch::Tensor sfd;
    std::optional<float> clamp_value;
    std::optional<torch::Tensor> clamped_count;

    static uint32_t* get_clamp_count_ready() {
        const auto stream = c10::cuda::getCurrentCUDAStream();
        static std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> ready_by_stream;
        auto& ready = ready_by_stream[stream];
        if (not ready.defined()) {
            // Initialize each stream once, before graph capture.
            DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None);
            ready = torch::zeros({2}, torch::TensorOptions().device(stream.device()).dtype(torch::kUInt32));
        }
        return static_cast<uint32_t*>(ready.data_ptr());
    }

public:
    explicit FP8CastWithClampEpilogue(const torch::Tensor& sfd,
                                      const std::optional<float>& clamp_value = std::nullopt,
                                      const std::optional<torch::Tensor>& clamped_count = std::nullopt):
        sfd(sfd), clamp_value(clamp_value), clamped_count(clamped_count) {
        DG_HOST_ASSERT(not clamp_value.has_value() or clamp_value.value() > 0.0f);
        if (clamped_count.has_value())
            DG_HOST_ASSERT(clamped_count->scalar_type() == torch::kUInt32 and clamped_count->numel() == 1);
    }

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::CastToFP8WithClamp";
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.sfd = static_cast<uint32_t*>(sfd.data_ptr()),
                .sfd_stride = static_cast<uint32_t>(sfd.stride(-1)),
                .shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n),
                .clamp_value = clamp_value.value_or(0.0f),
                .clamped_count = clamped_count.has_value() ? static_cast<uint32_t*>(clamped_count->data_ptr()) : nullptr,
                .clamp_count_ready = clamped_count.has_value() ? get_clamp_count_ready() : nullptr};
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kFloat8_e4m3fn and
                       "FP8 quantization requires a direct E4M3 output");
    }

    std::optional<torch::Tensor> output_sfd() const override { return sfd; }
};

// Axolotl extension; outputs must be zero-initialized for skipped expert tiles.
class SwiGLUForwardEpilogue final: public EpilogueClass {
    torch::Tensor z, scores, h;
public:
    SwiGLUForwardEpilogue(const torch::Tensor& z, const torch::Tensor& scores, const torch::Tensor& h):
        z(z), scores(scores), h(h) {}

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::SwiGLUForward";
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.dim() == 2 and d.scalar_type() == torch::kBFloat16);
        const auto m = d.size(0), n = d.size(1);
        DG_HOST_ASSERT(n % 16 == 0 and z.sizes() == d.sizes() and z.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(h.dim() == 2 and h.size(0) == m and h.size(1) == n / 2 and h.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(scores.dim() == 1 and scores.size(0) == m and scores.scalar_type() == torch::kFloat);
        for (const auto& t: {z, scores, h})
            DG_HOST_ASSERT(t.is_cuda() and t.device() == d.device() and t.is_contiguous());
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n),
                .swiglu_forward_z = z.data_ptr(), .swiglu_h = h.data_ptr(), .swiglu_scores = scores.data_ptr<float>()};
    }
};

class SwiGLURecomputeBackwardEpilogue final: public EpilogueClass {
    torch::Tensor dh, scores, h, dz, ds;
public:
    SwiGLURecomputeBackwardEpilogue(const torch::Tensor& dh, const torch::Tensor& scores,
                                   const torch::Tensor& h, const torch::Tensor& dz,
                                   const torch::Tensor& ds):
        dh(dh), scores(scores), h(h), dz(dz), ds(ds) {}

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::SwiGLURecomputeBackward";
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.dim() == 2 and d.scalar_type() == torch::kBFloat16);
        const auto m = d.size(0), n = d.size(1);
        DG_HOST_ASSERT(n % 16 == 0 and dz.sizes() == d.sizes() and dz.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(dh.dim() == 2 and dh.size(0) == m and dh.size(1) == n / 2 and dh.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(h.sizes() == dh.sizes() and h.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(scores.dim() == 1 and scores.size(0) == m and scores.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(ds.dim() == 2 and ds.size(0) == m and ds.size(1) == n / 8 and ds.scalar_type() == torch::kFloat);
        for (const auto& t: {dh, scores, h, dz, ds})
            DG_HOST_ASSERT(t.is_cuda() and t.device() == d.device() and t.is_contiguous());
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n),
                .swiglu_h = h.data_ptr(), .swiglu_dh = dh.data_ptr(),
                .swiglu_scores = scores.data_ptr<float>(), .swiglu_dz = dz.data_ptr(),
                .swiglu_ds = ds.data_ptr<float>()};
    }
};

class SwiGLUBackwardEpilogue final: public EpilogueClass {
    torch::Tensor z, scores, dz, ds;
public:
    SwiGLUBackwardEpilogue(const torch::Tensor& z, const torch::Tensor& scores,
                           const torch::Tensor& dz, const torch::Tensor& ds):
        z(z), scores(scores), dz(dz), ds(ds) {}

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::SwiGLUBackward";
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.dim() == 2 and d.scalar_type() == torch::kBFloat16);
        const auto m = d.size(0), n = d.size(1);
        DG_HOST_ASSERT(n % 8 == 0);
        DG_HOST_ASSERT(z.dim() == 2 and z.size(0) == m and z.size(1) == 2 * n);
        DG_HOST_ASSERT(dz.sizes() == z.sizes() and z.scalar_type() == torch::kBFloat16 and dz.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(scores.dim() == 1 and scores.size(0) == m and scores.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(ds.dim() == 2 and ds.size(0) == m and ds.size(1) == n / 8 and ds.scalar_type() == torch::kFloat);
        for (const auto& t: {z, scores, dz, ds})
            DG_HOST_ASSERT(t.is_cuda() and t.device() == d.device() and t.is_contiguous());
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n),
                .swiglu_z = z.data_ptr(), .swiglu_scores = scores.data_ptr<float>(),
                .swiglu_dz = dz.data_ptr(), .swiglu_ds = ds.data_ptr<float>()};
    }
};

class ScatterAddEpilogue final: public EpilogueClass {
    torch::Tensor tokens, output, queue;
    std::optional<torch::Tensor> row_end;
    int task;
    bool assign;
public:
    ScatterAddEpilogue(const torch::Tensor& tokens, const torch::Tensor& output,
                        const torch::Tensor& queue, int task, bool assign = false,
                        std::optional<torch::Tensor> row_end = std::nullopt):
        tokens(tokens), output(output), queue(queue), row_end(row_end), task(task), assign(assign) {}

    std::string get_epilogue_operator_type() const override {
        if (assign and output.scalar_type() == torch::kBFloat16)
            return "epilogue::operators::ScatterStoreBF16";
        return assign ? "epilogue::operators::ScatterStore" : "epilogue::operators::ScatterAdd";
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n),
                .scatter_tokens = tokens.data_ptr<int>(), .scatter_output = output.data_ptr(),
                .scatter_queue = queue.data_ptr<int>(), .scatter_task = static_cast<uint32_t>(task),
                .scatter_rows = static_cast<uint32_t>(output.size(0)),
                .scatter_row_end = row_end.has_value() ? row_end->data_ptr<int>() : nullptr};
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.dim() == 2 and d.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(d.size(0) > 0);
        DG_HOST_ASSERT(tokens.dim() == 1 and tokens.size(0) == d.size(0) and tokens.scalar_type() == torch::kInt);
        DG_HOST_ASSERT(output.dim() == 2 and output.size(1) == d.size(1) and output.size(0) > 0);
        DG_HOST_ASSERT(output.scalar_type() == torch::kFloat or (assign and output.scalar_type() == torch::kBFloat16));
        DG_HOST_ASSERT(queue.dim() == 2 and queue.size(1) == 4 and queue.scalar_type() == torch::kInt);
        DG_HOST_ASSERT(task >= 0 and task < queue.size(0));
        if (row_end.has_value())
            DG_HOST_ASSERT(row_end->is_cuda() and row_end->device() == d.device() and
                           row_end->scalar_type() == torch::kInt and row_end->numel() == 1 and row_end->is_contiguous());
        for (const auto& t: {tokens, output, queue})
            DG_HOST_ASSERT(t.is_cuda() and t.device() == d.device() and t.is_contiguous());
    }
};

class PeerScatterStoreEpilogue final: public EpilogueClass {
    torch::Tensor tokens, pointers, queue;
    int rows_per_peer;
public:
    PeerScatterStoreEpilogue(const torch::Tensor& tokens, const torch::Tensor& pointers,
                            const torch::Tensor& queue, int rows_per_peer):
        tokens(tokens), pointers(pointers), queue(queue), rows_per_peer(rows_per_peer) {}

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::PeerScatterStoreBF16";
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n),
                .scatter_tokens = tokens.data_ptr<int>(), .scatter_output = pointers.data_ptr(),
                .scatter_queue = queue.data_ptr<int>(), .scatter_task = 0,
                .scatter_rows = static_cast<uint32_t>(rows_per_peer * pointers.numel()),
                .scatter_peer_rows = static_cast<uint32_t>(rows_per_peer)};
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.dim() == 2 and d.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(tokens.dim() == 1 and tokens.size(0) == d.size(0) and tokens.scalar_type() == torch::kInt);
        DG_HOST_ASSERT(pointers.dim() == 1 and pointers.numel() > 0 and pointers.scalar_type() == torch::kLong);
        DG_HOST_ASSERT(rows_per_peer > 0 and static_cast<int64_t>(rows_per_peer) * pointers.numel() <= 2147483647);
        DG_HOST_ASSERT(queue.dim() == 2 and queue.size(0) == 1 and queue.size(1) == 4 and queue.scalar_type() == torch::kInt);
        for (const auto& t: {tokens, pointers, queue})
            DG_HOST_ASSERT(t.is_cuda() and t.device() == d.device() and t.is_contiguous());
    }
};

class BF16StochasticRoundingEpilogue final: public EpilogueClass {
public:
    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::StochasticRoundToBF16";
    }

    // `cvt.rs` requires an architecture-specific target (e.g., `sm_100a`), not a family one
    deep_jit::cuda::CompilerOptions compiler_options() const override {
        return {.arch = jit->device.get_arch(false)};
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kBFloat16 and
                       "Stochastic rounding requires a direct BF16 output");
    }
};

// Resolve the epilogue-related API inputs into the single epilogue class of an SM100 GEMM,
// validated against the GEMM output: the user class, the standalone `alpha` (an API argument
// for cuBLASLt signature parity), or the FP8 `(d, sfd)` output pair; all exclusive
static std::shared_ptr<EpilogueClass> resolve_epilogue_class(const std::shared_ptr<EpilogueClass>& epilogue_class,
                                                             const std::optional<torch::Tensor>& c,
                                                             const torch::Tensor& d,
                                                             const std::optional<float>& alpha = std::nullopt,
                                                             const std::optional<torch::Tensor>& sfd = std::nullopt) {
    DG_HOST_ASSERT((epilogue_class != nullptr) + alpha.has_value() + sfd.has_value() <= 1 and
                   "The epilogue class, `alpha` and the FP8 output pair are exclusive");
    std::shared_ptr<EpilogueClass> resolved = epilogue_class;
    if (alpha.has_value())
        resolved = std::make_shared<AlphaEpilogue>(alpha.value());
    if (sfd.has_value())
        resolved = std::make_shared<FP8CastWithClampEpilogue>(sfd.value());
    if (resolved == nullptr)
        resolved = std::make_shared<IdentityEpilogue>();
    resolved->check(c, d);
    return resolved;
}

} // namespace deep_gemm
