# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Exercise cross-CTA phase publication and repeated ring reuse on device."""

from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]

CUDA_SOURCE = r"""
#include <backward_schedule.cuh>
#include <cuda_runtime.h>
using namespace mono_ep;
__global__ void initialize(BackwardSchedule schedule) {
    for (unsigned int i = threadIdx.x; i < schedule.capacity; i += blockDim.x)
        schedule.initialize(i);
}
__global__ void exercise(BackwardSchedule schedule, int* visits, int* payload,
                         int* errors, int stride) {
    if (threadIdx.x) return;
    unsigned int cursor = blockIdx.x % schedule.capacity;
    const auto started = clock64();
    while (!schedule.finished()) {
        if (atomicAdd(errors, 0)) return;
        if (clock64() - started > 10000000000ull) {
            atomicAdd(errors, 1); return;
        }
        BackwardTask task;
        if (!schedule.claim(cursor, task)) {
            __nanosleep(64); continue;
        }
        if (task.phase) {
            for (unsigned int i = 0; i < schedule.tiles[task.phase - 1]; ++i)
                if (payload[(task.slot * 3 + task.phase - 1) * stride + i]
                    != static_cast<int>(task.block + 1)) atomicAdd(errors, 1);
        }
        // Uneven service time makes task completion order differ from claim order.
        __nanosleep(((task.block * 17 + task.tile * 31 + blockIdx.x) % 13) * 128);
        payload[(task.slot * 3 + task.phase) * stride + task.tile] = task.block + 1;
        atomicAdd(visits + (task.block * 3 + task.phase) * stride + task.tile, 1);
        __threadfence();
        schedule.complete(task);
    }
}
__global__ void resolve_tiles(const unsigned int* ends,
    const unsigned int* starts, const unsigned int* counts, unsigned int* output,
    int experts, int block_m, int capacity) {
    const auto block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= capacity) return;
    const auto tile = resolve_backward_tile(block, ends, starts, counts, experts, block_m);
    output[block * 3] = tile.expert;
    output[block * 3 + 1] = tile.packed_row;
    output[block * 3 + 2] = tile.valid_rows;
}
int resolve(uintptr_t ends, uintptr_t starts, uintptr_t counts, uintptr_t output,
            int experts, int block_m, int capacity, uintptr_t stream_ptr) {
    resolve_tiles<<<(capacity + 127) / 128, 128, 0,
        reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
        reinterpret_cast<unsigned int*>(ends), reinterpret_cast<unsigned int*>(starts),
        reinterpret_cast<unsigned int*>(counts), reinterpret_cast<unsigned int*>(output),
        experts, block_m, capacity);
    return static_cast<int>(cudaGetLastError());
}
int launch(uintptr_t slot_ptr, uintptr_t visits, uintptr_t payload,
           uintptr_t errors, int capacity, int blocks, int a, int b, int c,
           int workers, uintptr_t stream_ptr) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    BackwardSchedule schedule{reinterpret_cast<BackwardSlot*>(slot_ptr),
        static_cast<unsigned int>(capacity), static_cast<unsigned int>(blocks),
        {static_cast<unsigned int>(a), static_cast<unsigned int>(b),
         static_cast<unsigned int>(c)}};
    initialize<<<1, 128, 0, stream>>>(schedule);
    exercise<<<workers, 32, 0, stream>>>(schedule, reinterpret_cast<int*>(visits),
        reinterpret_cast<int*>(payload), reinterpret_cast<int*>(errors),
        max(a, max(b, c)));
    return static_cast<int>(cudaGetLastError());
}
"""


@pytest.fixture(scope="module")
def scheduler_probe(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Scheduler validation requires Hopper or Blackwell")
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        pytest.skip("Scheduler validation requires nvcc")
    import mono_ep

    directory = tmp_path_factory.mktemp("backward_schedule")
    cpp = directory / "binding.cpp"
    cpp.write_text(
        "#include <pybind11/pybind11.h>\n#include <cstdint>\n"
        "int launch(uintptr_t, uintptr_t, uintptr_t, uintptr_t, "
        "int, int, int, int, int, int, uintptr_t);\n"
        "int resolve(uintptr_t, uintptr_t, uintptr_t, uintptr_t, int, int, int, uintptr_t);\n"
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); '
        'm.def("resolve", &resolve); }\n'
    )
    cu = directory / "probe.cu"
    cu.write_text(CUDA_SOURCE)
    return load(
        name="mono_ep_backward_schedule_probe",
        sources=[str(cpp), str(cu)],
        extra_include_paths=[str(Path(mono_ep.__file__).parent / "include")],
        extra_cuda_cflags=["-O2"],
        build_directory=str(directory),
    )


@pytest.mark.parametrize("blocks,capacity", [(0, 1), (1, 7), (137, 1), (257, 7)])
@pytest.mark.parametrize("tiles", [(3, 6, 8), (16, 32, 4), (1, 1, 1)])
def test_dynamic_backward_ring(scheduler_probe, blocks, capacity, tiles):
    stride = max(tiles)
    slots = torch.empty((capacity, 2), device="cuda", dtype=torch.int64)
    visits = torch.zeros((blocks, 3, stride), device="cuda", dtype=torch.int32)
    payload = torch.full((capacity, 3, stride), -1, device="cuda", dtype=torch.int32)
    errors = torch.zeros(1, device="cuda", dtype=torch.int32)
    assert (
        scheduler_probe.launch(
            slots.data_ptr(),
            visits.data_ptr(),
            payload.data_ptr(),
            errors.data_ptr(),
            capacity,
            blocks,
            *tiles,
            96,
            torch.cuda.current_stream().cuda_stream,
        )
        == 0
    )
    assert errors.item() == 0
    expected = torch.zeros_like(visits)
    for phase, count in enumerate(tiles):
        expected[:, phase, :count] = 1
    torch.testing.assert_close(visits, expected, rtol=0, atol=0)
    assert (slots[:, 0] >> 32 >= blocks).all().item()


@pytest.mark.parametrize(
    "counts", [[0, 0, 0], [0, 1, 255, 0, 129, 17], [8193, 0, 0, 0]]
)
def test_backward_expert_mapping(scheduler_probe, counts):
    block_m = 128
    counts_gpu = torch.tensor(counts, device="cuda", dtype=torch.int32)
    sizes = (counts_gpu + block_m - 1) // block_m
    ends = sizes.cumsum(0).int()
    padded = ((counts_gpu + 255) // 256) * 256
    starts = (padded.cumsum(0) - padded).int()
    expected = []
    packed_start = 0
    for expert, count in enumerate(counts):
        for row in range(0, count, block_m):
            expected.append([expert, packed_start + row, min(block_m, count - row)])
        packed_start += ((count + 255) // 256) * 256
    expected.extend([[len(counts), 0, 0]] * 3)
    output = torch.empty((len(expected), 3), device="cuda", dtype=torch.int32)
    assert (
        scheduler_probe.resolve(
            ends.data_ptr(),
            starts.data_ptr(),
            counts_gpu.data_ptr(),
            output.data_ptr(),
            len(counts),
            block_m,
            len(expected),
            torch.cuda.current_stream().cuda_stream,
        )
        == 0
    )
    torch.testing.assert_close(
        output, torch.tensor(expected, device="cuda", dtype=torch.int32), rtol=0, atol=0
    )
