# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Numerically validate native peer dispatch, persistent forward, and combine."""

import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist


def relative(actual, expected):
    return (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-8)
    ).item()


def _run_original_w1_parity(rank):
    import deep_gemm
    from deep_gemm.mega import SymmBuffer, _interleave_weights
    from mono_ep.paired_weight_quantization import (
        quantize_paired_gate_up_weights,
    )
    from mono_ep.quantization import quantize_fp8
    from mono_ep.token_quantization import quantize_tokens

    hidden, inner, experts, topk = 2048, 768, 128, 8
    buffer = SymmBuffer(
        dist.group.WORLD, experts, 288, topk, hidden, inner, mma_type="fp8xbf16"
    )
    capacity = buffer.num_max_tokens_per_rank
    local_experts = experts // dist.get_world_size()
    records = []
    for master_dtype in (torch.bfloat16, torch.float32):
        w1 = (
            torch.randn(
                local_experts,
                2 * inner,
                hidden,
                device="cuda",
                dtype=master_dtype,
            )
            / hidden**0.5
        )
        w2 = (
            torch.randn(
                local_experts,
                hidden,
                inner,
                device="cuda",
                dtype=master_dtype,
            )
            / inner**0.5
        )
        legacy_w1, legacy_s1 = quantize_fp8(
            _interleave_weights(w1), block_rows=128, block_cols=128
        )
        original_w1, original_s1 = quantize_paired_gate_up_weights(w1)
        q2, s2 = quantize_fp8(w2, block_rows=128, block_cols=128)
        torch.testing.assert_close(original_s1, legacy_s1, rtol=0, atol=0)
        for step, skewed in enumerate((False, True, False)):
            tokens = 269 - rank * 12
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
            ids = (
                torch.rand(tokens, experts, device="cuda")
                .argsort(-1)[:, :topk]
                .contiguous()
            )
            if skewed:
                ids[:] = torch.arange(topk, device="cuda")
            ids[::31, -1] = -1
            scores = torch.rand(tokens, topk, device="cuda").softmax(-1)
            qx, sx = quantize_tokens(x, 128)
            buffer.x[:tokens].copy_(qx)
            buffer.x_sf[:tokens].copy_(sx)
            buffer.topk_idx.fill_(-1)
            buffer.topk_idx[:tokens].copy_(ids)
            buffer.topk_weights.zero_()
            buffer.topk_weights[:tokens].copy_(scores)
            outputs = []
            projections = []
            for original_layout, q1, s1 in (
                (False, legacy_w1, legacy_s1),
                (True, original_w1, original_s1),
            ):
                y = torch.empty_like(x)
                saved = torch.full(
                    (2, capacity, topk, 2 * inner),
                    float("nan"),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                dist.barrier()
                deep_gemm._C.fp8_mono_ep_forward_sm90(
                    y,
                    q1,
                    q2,
                    [s1, s2],
                    buffer.buffer,
                    list(buffer.handle.buffer_ptrs),
                    rank,
                    capacity,
                    buffer.l1_acts.shape[0],
                    topk,
                    saved,
                    original_layout,
                )
                torch.cuda.synchronize()
                outputs.append(y)
                projections.append(saved)
            torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
            torch.testing.assert_close(
                torch.isfinite(projections[1]),
                torch.isfinite(projections[0]),
                rtol=0,
                atol=0,
            )
            finite = torch.isfinite(projections[0])
            torch.testing.assert_close(
                projections[1][finite], projections[0][finite], rtol=0, atol=0
            )
            histogram = torch.bincount(ids[ids >= 0], minlength=experts)
            dist.all_reduce(histogram)
            if skewed:
                assert torch.count_nonzero(histogram == 0)
            records.append(
                dict(
                    rank=rank,
                    master_dtype=str(master_dtype),
                    step=step,
                    skewed=skewed,
                    tokens=tokens,
                    empty_experts=int((histogram == 0).sum()),
                    output_exact=True,
                    saved_projection_exact=True,
                )
            )
    print(json.dumps(dict(rank=rank, original_w1_layout=True, records=records)))
    buffer.destroy()


def run():
    import deep_gemm
    from deep_gemm.mega import SymmBuffer, _interleave_weights
    from torch.utils.cpp_extension import load
    from mono_ep.quantization import quantize_fp8

    torch.backends.cuda.matmul.allow_tf32 = False

    def quantized(t, rows=1):
        q, scales = quantize_fp8(t, block_rows=rows, block_cols=128)
        decoded = (
            q.float()
            * scales.repeat_interleave(rows, -2).repeat_interleave(128, -1)[
                ..., : t.shape[-2], : t.shape[-1]
            ]
        )
        return q, scales.contiguous(), decoded

    def gemm(a, w):
        return (quantized(a)[2] @ w.T).bfloat16().float()

    def recompute(a, w):
        z = gemm(a, w)
        return (
            z.reshape(*z.shape[:-1], inner // 8, 2, 8)
            .transpose(-3, -2)
            .reshape(*z.shape[:-1], 2 * inner)
        )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    assert dist.get_world_size() == 2
    assert torch.cuda.get_device_properties(rank).major == 9
    torch.manual_seed(517 + rank)
    if os.environ.get("MONO_EP_ORIGINAL_W1_LAYOUT") == "1":
        _run_original_w1_parity(rank)
        dist.destroy_process_group()
        return
    hidden, inner, experts, topk = 2048, 768, 16, 4
    buffer = SymmBuffer(
        dist.group.WORLD, experts, 256, topk, hidden, inner, mma_type="fp8xbf16"
    )
    capacity = buffer.num_max_tokens_per_rank
    w1 = (
        torch.randn(experts // 2, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(experts // 2, hidden, inner, device="cuda") / inner**0.5
    ).bfloat16()
    all_w1, all_w2 = (
        [torch.empty_like(w1) for _ in range(2)],
        [torch.empty_like(w2) for _ in range(2)],
    )
    dist.all_gather(all_w1, w1)
    dist.all_gather(all_w2, w2)
    all_w1, all_w2 = torch.cat(all_w1), torch.cat(all_w2)
    weights = [
        _interleave_weights(w1),
        w2,
    ]
    prepared = [quantized(w, rows=128) for w in weights]
    weights = [p[0] for p in prepared]
    scales = [p[1] for p in prepared]
    decoded = [p[2] for p in prepared]
    global_decoded = []
    for w in decoded:
        gathered = [torch.empty_like(w) for _ in range(2)]
        dist.all_gather(gathered, w)
        global_decoded.append(torch.cat(gathered))
    bound = os.environ.get("MONO_EP_NATIVE_BINDING") == "1"
    compact = os.environ.get("MONO_EP_COMPACT_FORWARD") == "1"
    trap_timeouts = os.environ.get("MONO_EP_TRAP_TIMEOUTS") == "1"
    if bound and compact:
        raise ValueError("compact forward is available only through the raw probe")
    if bound and trap_timeouts:
        raise ValueError(
            "trap-only timeout handling is available only through the raw probe"
        )
    if not bound:
        build = Path(tempfile.mkdtemp(prefix="hopper_backward_"))
        source = (
            Path("/probe/hopper_fp8_persistent_forward_probe.cu")
            .read_text()
            .replace("@MAX_T@", str(capacity))
            .replace("@RING@", str(buffer.l1_acts.shape[0]))
            .replace("@DISPATCH@", "32" if compact else "128")
            .replace("@NON_EPILOGUE@", "96" if compact else "128")
            .replace("@THREADS@", "384" if compact else "512")
            .replace("@COMPACT@", "true" if compact else "false")
        )
        (build / "probe.cu").write_text(source)
        (build / "binding.cpp").write_text(
            "#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <cstdint>\n#include <vector>\n"
            "int launch(std::vector<uintptr_t>,std::vector<int64_t>,int,int,int,uintptr_t);\n"
            'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("launch",&launch);}\n'
        )
        native = load(
            name=(
                "mono_ep_hopper_fp8_forward_probe_"
                f"{'compact' if compact else 'default'}_"
                f"{'trap' if trap_timeouts else 'diagnostic'}"
            ),
            sources=[str(build / "binding.cpp"), str(build / "probe.cu")],
            extra_include_paths=[
                "/probe/native",
                str(Path(deep_gemm.__file__).parent / "include"),
            ],
            extra_cuda_cflags=[
                "-O2",
                "-std=c++20",
                "--expt-relaxed-constexpr",
                "-gencode=arch=compute_90a,code=sm_90a",
                *(["-DDG_TRAP_ONLY_TIMEOUTS=1"] if trap_timeouts else []),
            ],
            extra_ldflags=["-lcuda"],
            build_directory=str(build),
        )
    for step, skew in enumerate((False, True, False)):
        tokens = (1800 if skew else 97) + rank * 13
        assert tokens <= capacity
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        ids = (
            torch.rand(tokens, topk if skew else experts, device="cuda")
            .argsort(-1)[:, :topk]
            .contiguous()
        )
        ids[::17, -1] = -1
        scores = torch.rand(tokens, topk, device="cuda").softmax(-1)
        qx, sx, _ = quantized(x)
        buffer.x[:tokens].copy_(qx)
        buffer.x_sf[:tokens].copy_(sx)
        buffer.topk_idx.fill_(-1)
        buffer.topk_idx[:tokens].copy_(ids)
        buffer.topk_weights.zero_()
        buffer.topk_weights[:tokens].copy_(scores)
        y = torch.empty_like(x)
        saved = torch.full(
            (2, capacity, topk, 2 * inner),
            float("nan"),
            device="cuda",
            dtype=torch.bfloat16,
        )
        dist.barrier()
        if bound:
            deep_gemm._C.fp8_mono_ep_forward_sm90(
                y,
                *weights,
                scales,
                buffer.buffer,
                list(buffer.handle.buffer_ptrs),
                rank,
                capacity,
                buffer.l1_acts.shape[0],
                topk,
                saved,
            )
        else:
            error = native.launch(
                [
                    t.data_ptr()
                    for t in (
                        y,
                        *weights,
                        *scales,
                        saved,
                    )
                ],
                list(buffer.handle.buffer_ptrs),
                rank,
                tokens,
                buffer.l1_acts.shape[0],
                torch.cuda.current_stream().cuda_stream,
            )
            assert error == 0, error
        torch.cuda.synchronize()
        expected = torch.zeros_like(x, dtype=torch.float32)
        for expert in range(experts):
            rows, slots = (ids == expert).nonzero(as_tuple=True)
            if not rows.numel():
                continue
            z = recompute(x[rows], global_decoded[0][expert])
            gate, up = z.chunk(2, -1)
            h = (
                torch.nn.functional.silu(gate) * up * scores[rows, slots, None]
            ).bfloat16()
            out = gemm(h, global_decoded[1][expert])
            expected.index_add_(0, rows, out)
        output_error = relative(y, expected.bfloat16())
        torch.testing.assert_close(
            torch.isfinite(y), torch.ones_like(y, dtype=torch.bool)
        )
        gathered = []
        width = 1813 if skew else 110
        for tensor, fill in ((x, 0), (ids, -1)):
            padded = torch.nn.functional.pad(
                tensor, (0, 0, 0, width - tokens), value=fill
            )
            peers = [torch.empty_like(padded) for _ in range(2)]
            dist.all_gather(peers, padded)
            gathered.append(torch.stack(peers))
        all_x, all_ids = gathered
        projection_errors = []
        for expert in range(experts // 2):
            source, row, slot = (all_ids == rank * (experts // 2) + expert).nonzero(
                as_tuple=True
            )
            if row.numel():
                z = recompute(all_x[source, row], decoded[0][expert])
                projection_errors.append(relative(saved[source, row, slot], z))
        assert torch.isnan(saved[:, :width][all_ids == -1]).all()
        histogram = torch.bincount(ids[ids >= 0], minlength=experts)
        dist.all_reduce(histogram)
        used_rows = int(
            (((histogram[rank * 8 : (rank + 1) * 8] + 127) // 128) * 128).sum().item()
        )
        if skew and rank == 0:
            assert used_rows > buffer.l1_acts.shape[0]
        print(
            json.dumps(
                dict(
                    rank=rank,
                    step=step,
                    skew=skew,
                    tokens=tokens,
                    compute_dtype="blockwise_fp8",
                    transport_dtype="fp8_input_bf16_combine",
                    activation_cache="input_and_intermediate",
                    compact_non_math_wg=compact,
                    timeout_handler=(
                        "trap_only_no_diagnostic_calls"
                        if trap_timeouts
                        else "diagnostic_calls"
                    ),
                    raw_probe=not bound,
                    output_error=output_error,
                    projection_errors=projection_errors,
                    used_pool_rows=used_rows,
                    ring_tokens=buffer.l1_acts.shape[0],
                )
            ),
            flush=True,
        )
        assert max([output_error, *projection_errors]) < 0.02
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
