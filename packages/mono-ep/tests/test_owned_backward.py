# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Owned M-block backward matches staged BF16 math without a global dH tensor."""

from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]


@pytest.fixture(scope="module")
def owned_probe(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Owned UMMA backward requires SM100")
    deep_gemm = pytest.importorskip("deep_gemm")
    import mono_ep
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        pytest.skip("Owned backward validation requires nvcc")
    directory = tmp_path_factory.mktemp("owned_backward")
    cpp = directory / "binding.cpp"
    cpp.write_text(
        "#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n"
        "#include <cstdint>\n#include <vector>\n"
        "int launch(std::vector<uintptr_t>, int, int, int, int, uintptr_t);\n"
        "int tile(uintptr_t, uintptr_t, uintptr_t, uintptr_t);\n"
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); m.def("tile", &tile); }\n'
    )
    cu = directory / "probe.cu"
    cu.write_text(r"""
#include <sm100_owned_backward.cuh>
#include <vector>
__global__ void tile_probe(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c) {
    __shared__ __align__(1024) unsigned char buffer[sizeof(mono_ep::Sm100BackwardTile)];
    auto& shared = *reinterpret_cast<mono_ep::Sm100BackwardTile*>(buffer);
    shared.initialize();
    shared.multiply(64, [&](unsigned int m, unsigned int k) { return a[m * 64 + k]; },
        [&](unsigned int n, unsigned int k) { return b[n * 64 + k]; }, c);
    shared.release();
}
int tile(uintptr_t a, uintptr_t b, uintptr_t c, uintptr_t stream) {
    tile_probe<<<1, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<__nv_bfloat16*>(a), reinterpret_cast<__nv_bfloat16*>(b),
        reinterpret_cast<float*>(c));
    return static_cast<int>(cudaGetLastError());
}
int launch(std::vector<uintptr_t> p, int experts, int hidden, int inner,
           int workers, uintptr_t stream_ptr) {
    using namespace mono_ep;
    auto error = cudaFuncSetAttribute(sm100_owned_backward,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(OwnedBackwardStorage));
    if (error != cudaSuccess) return static_cast<int>(error);
    sm100_owned_backward<<<workers, 128, sizeof(OwnedBackwardStorage),
        reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
        reinterpret_cast<__nv_bfloat16*>(p[0]), reinterpret_cast<__nv_bfloat16*>(p[1]),
        reinterpret_cast<__nv_bfloat16*>(p[2]), reinterpret_cast<__nv_bfloat16*>(p[3]),
        reinterpret_cast<float*>(p[4]), reinterpret_cast<unsigned int*>(p[5]),
        reinterpret_cast<unsigned int*>(p[6]), reinterpret_cast<unsigned int*>(p[7]),
        reinterpret_cast<unsigned int*>(p[8]), reinterpret_cast<__nv_bfloat16*>(p[9]),
        reinterpret_cast<__nv_bfloat16*>(p[10]), reinterpret_cast<__nv_bfloat16*>(p[11]),
        reinterpret_cast<float*>(p[12]), experts, hidden, inner);
    return static_cast<int>(cudaGetLastError());
}
""")
    return load(
        name="mono_ep_owned_backward_probe",
        sources=[str(cpp), str(cu)],
        extra_include_paths=[
            str(Path(mono_ep.__file__).parent / "include"),
            str(Path(deep_gemm.__file__).parent / "include"),
        ],
        extra_cuda_cflags=[
            "-O2",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        build_directory=str(directory),
    )


def test_owned_tile_layout(owned_probe):
    a = torch.eye(64, device="cuda", dtype=torch.bfloat16)
    b = (
        torch.arange(64, device="cuda")[:, None]
        + torch.arange(64, device="cuda")[None, :] / 64
    ).bfloat16()
    c = torch.full((64, 64), float("nan"), device="cuda")
    assert (
        owned_probe.tile(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        == 0
    )
    expected = a.float() @ b.float().T
    print(
        "tile rows:",
        c[::8, :4].tolist(),
        "expected:",
        expected[::8, :4].tolist(),
        flush=True,
    )
    torch.testing.assert_close(c, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "hidden,inner,counts",
    [
        (256, 128, [0, 1, 127, 65]),
        (512, 256, [257, 0, 0, 0]),
        (2048, 768, [65, 0]),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_owned_backward(owned_probe, hidden, inner, counts, compiled):
    torch.manual_seed(783)
    counts_gpu = torch.tensor(counts, device="cuda", dtype=torch.int32)
    padded = ((counts_gpu + 63) // 64) * 64
    starts = (padded.cumsum(0) - padded).int()
    rows = sum((n + 63) // 64 * 64 for n in counts)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    dy = torch.randn_like(x) * 0.1
    w1 = torch.randn(len(counts), 2 * inner, hidden, device="cuda", dtype=x.dtype) * 0.1
    w2 = torch.randn(len(counts), hidden, inner, device="cuda", dtype=x.dtype) * 0.1
    scores = torch.rand(rows, device="cuda")
    from mono_ep.owned_backward import owned_backward

    operation = (
        torch.compile(owned_backward, fullgraph=True) if compiled else owned_backward
    )
    h, dz, dx, ds = operation(x, dy, w1, w2, scores, starts, counts_gpu)
    torch.cuda.synchronize()
    expected = [torch.zeros_like(t) for t in (h, dz, dx, ds)]
    start = 0
    for expert, count in enumerate(counts):
        sl = slice(start, start + count)
        z = (x[sl].float() @ w1[expert].float().T).bfloat16().float()
        dh = (dy[sl].float() @ w2[expert].float()).bfloat16().float()
        gate, up = z.chunk(2, dim=1)
        sigmoid = gate.sigmoid()
        silu = gate * sigmoid
        score = scores[sl, None]
        expected[0][sl] = (silu * up * score).bfloat16()
        dz_ref = torch.cat(
            (dh * score * up * (sigmoid + silu * (1 - sigmoid)), dh * score * silu),
            dim=1,
        ).bfloat16()
        expected[1][sl] = dz_ref
        expected[2][sl] = (dz_ref.float() @ w1[expert].float()).bfloat16()
        expected[3][sl] = (dh * silu * up).sum(-1)
        start += (count + 63) // 64 * 64
    for actual, reference in zip((h, dz, dx, ds), expected, strict=True):
        relative = (
            actual.float() - reference.float()
        ).norm() / reference.float().norm().clamp_min(1e-12)
        assert relative.item() < 0.015
        torch.testing.assert_close(
            actual[reference == 0], reference[reference == 0], atol=1e-5, rtol=0
        )
