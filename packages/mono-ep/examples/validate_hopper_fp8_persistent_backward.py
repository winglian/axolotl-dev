# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Numerically validate native peer dispatch, persistent backward, and combine."""

import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm


def relative(actual, expected):
    return (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-8)
    ).item()


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
    hidden, inner, experts, topk = (
        int(os.environ.get("MONO_EP_TEST_HIDDEN", "2048")),
        int(os.environ.get("MONO_EP_TEST_INTERMEDIATE", "768")),
        16,
        4,
    )
    buffer = SymmBuffer(
        dist.group.WORLD, experts, 256, topk, hidden, inner, mma_type="bf16xbf16"
    )
    capacity = buffer.num_max_tokens_per_rank
    pool_rows = 2 * capacity * topk + (experts // 2) * 128
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
        w2.transpose(1, 2).contiguous(),
        w1.transpose(1, 2).contiguous(),
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
    dy_local = symm.empty(capacity, hidden, dtype=torch.bfloat16, device="cuda")
    ds_local = symm.empty(capacity, topk, 2, dtype=torch.float32, device="cuda")
    dy_handle, ds_handle = (
        symm.rendezvous(dy_local, group=dist.group.WORLD),
        symm.rendezvous(ds_local, group=dist.group.WORLD),
    )
    dy_pointers = [
        dy_handle.get_buffer(r, dy_local.shape, dy_local.dtype).data_ptr()
        for r in range(2)
    ]
    ds_pointers = [
        ds_handle.get_buffer(r, ds_local.shape, ds_local.dtype).data_ptr()
        for r in range(2)
    ]
    pools = [
        torch.zeros(pool_rows, n, device="cuda", dtype=torch.bfloat16)
        for n in (hidden, hidden, inner, 2 * inner)
    ]
    starts, counts = [
        torch.zeros(experts // 2, device="cuda", dtype=torch.int32) for _ in range(2)
    ]
    bound = os.environ.get("MONO_EP_NATIVE_BINDING") == "1"
    cached = os.environ.get("MONO_EP_CACHE_ACTIVATIONS") == "1"
    if cached:
        pools.extend(
            torch.empty(pool_rows, width, device="cuda", dtype=dtype)
            for width, dtype in (
                (hidden, torch.float8_e4m3fn),
                (hidden, torch.float8_e4m3fn),
                (hidden // 128, torch.float32),
                (hidden // 128, torch.float32),
            )
        )
    cache_dz = os.environ.get("MONO_EP_CACHE_DZ") == "1"
    if cache_dz:
        assert cached
        pools.extend(
            torch.empty(pool_rows, width, device="cuda", dtype=dtype)
            for width, dtype in (
                (2 * inner, torch.float8_e4m3fn),
                (2 * inner // 128, torch.float32),
            )
        )
    trap_timeouts = os.environ.get("MONO_EP_TRAP_TIMEOUTS") == "1"
    if trap_timeouts and bound:
        raise ValueError("Trap-only validation requires the raw probe")
    compact = os.environ.get("MONO_EP_COMPACT_BACKWARD") == "1"
    if compact and bound:
        raise ValueError("Compact validation requires the raw probe")
    if not bound:
        build = Path(tempfile.mkdtemp(prefix="hopper_backward_"))
        source = (
            Path("/probe/hopper_fp8_persistent_backward_probe.cu")
            .read_text()
            .replace("@MAX_T@", str(capacity))
            .replace("@RING@", str(buffer.l1_acts.shape[0]))
            .replace("@CACHED@", str(cached).lower())
            .replace("@CACHE_DZ@", str(cache_dz).lower())
            .replace("@COMPACT@", str(compact).lower())
            .replace("@DISPATCH@", "32" if compact else "128")
            .replace("@NON_EPILOGUE@", "96" if compact else "128")
            .replace("@THREADS@", "384" if compact else "512")
        )
        (build / "probe.cu").write_text(source)
        (build / "binding.cpp").write_text(
            "#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <cstdint>\n#include <vector>\n"
            "int launch(std::vector<uintptr_t>,std::vector<int64_t>,std::vector<int64_t>,std::vector<int64_t>,int,int,int,uintptr_t);\n"
            'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("launch",&launch);}\n'
        )
        native = load(
            name="mono_ep_hopper_fp8_backward_probe",
            sources=[str(build / "binding.cpp"), str(build / "probe.cu")],
            extra_include_paths=[
                "/probe/native",
                str(Path(deep_gemm.__file__).parent / "include"),
            ],
            extra_cuda_cflags=[
                "-O2",
                *(["-DDG_TRAP_ONLY_TIMEOUTS=1"] if trap_timeouts else []),
                "-std=c++20",
                "--expt-relaxed-constexpr",
                "-gencode=arch=compute_90a,code=sm_90a",
            ],
            extra_ldflags=["-lcuda"],
            build_directory=str(build),
        )
    for skew in (False, True):
        tokens = (1800 if skew else 97) + rank * 13
        assert tokens <= capacity
        x, dy = [
            torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        ids = (
            torch.rand(tokens, topk if skew else experts, device="cuda")
            .argsort(-1)[:, :topk]
            .contiguous()
        )
        ids[::17, -1] = -1
        scores = torch.rand(tokens, topk, device="cuda").softmax(-1)
        buffer.x[:tokens].copy_(x)
        buffer.topk_idx.fill_(-1)
        buffer.topk_idx[:tokens].copy_(ids)
        buffer.topk_weights.zero_()
        buffer.topk_weights[:tokens].copy_(scores)
        dy_local.zero_()
        dy_local[:tokens].copy_(dy)
        ds_local.fill_(float("nan"))
        for pool in pools:
            pool.zero_()
        dx = torch.empty_like(x)
        ds = torch.empty_like(scores)
        dist.barrier()
        torch.cuda.synchronize()
        if bound:
            deep_gemm._C.fp8_mono_ep_backward_sm90(
                dx,
                ds,
                *weights,
                scales,
                pools,
                starts,
                counts,
                buffer.buffer,
                list(buffer.handle.buffer_ptrs),
                dy_local,
                dy_pointers,
                ds_local,
                ds_pointers,
                rank,
                capacity,
                buffer.l1_acts.shape[0],
                topk,
            )
        else:
            error = native.launch(
                [
                    t.data_ptr()
                    for t in (
                        dx,
                        ds,
                        *pools[:4],
                        *weights,
                        starts,
                        counts,
                        *scales,
                        *pools[4:],
                    )
                ],
                list(buffer.handle.buffer_ptrs),
                dy_pointers,
                ds_pointers,
                rank,
                tokens,
                pool_rows,
                torch.cuda.current_stream().cuda_stream,
            )
            assert error == 0, error
        torch.cuda.synchronize()
        expected_dx = torch.zeros_like(x, dtype=torch.float32)
        expected_ds = torch.zeros_like(scores)
        for expert in range(experts):
            rows, slots = (ids == expert).nonzero(as_tuple=True)
            if not rows.numel():
                continue
            z = recompute(x[rows], global_decoded[0][expert])
            gate, up = z.chunk(2, dim=-1)
            dh = gemm(dy[rows], global_decoded[1][expert])
            sigmoid = gate.sigmoid()
            silu = gate * sigmoid
            weighted_dh = dh * scores[rows, slots, None]
            dz = torch.cat(
                (
                    weighted_dh * up * (sigmoid + silu * (1 - sigmoid)),
                    weighted_dh * silu,
                ),
                -1,
            ).bfloat16()
            partial_dx = gemm(dz, global_decoded[2][expert]).bfloat16()
            expected_dx.index_add_(0, rows, partial_dx.float())
            expected_ds[rows, slots] = (dh * silu * up).sum(-1)
        errors = {
            "dx": relative(dx, expected_dx.bfloat16()),
            "dscore": relative(ds, expected_ds),
        }
        gathered = []
        for tensor, fill in ((x, 0), (dy, 0), (ids, -1), (scores, 0)):
            padded = torch.nn.functional.pad(
                tensor, (0, 0, 0, (1813 if skew else 110) - tokens), value=fill
            )
            peers = [torch.empty_like(padded) for _ in range(2)]
            dist.all_gather(peers, padded)
            gathered.append(torch.cat(peers))
        source_x, source_dy, source_ids, source_scores = gathered
        errors.update(h=0.0, dz=0.0, dw1=0.0, dw2=0.0)
        for expert, (start, count) in enumerate(
            zip(starts.tolist(), counts.tolist(), strict=True)
        ):
            if not count:
                continue
            xp, yp, hp, zp = [pool[start : start + count] for pool in pools[:4]]
            if cached:
                cache_cases = [(xp, pools[4], pools[6]), (yp, pools[5], pools[7])]
                if cache_dz:
                    cache_cases.append((zp, pools[8], pools[9]))
                for operand, qpool, spool in cache_cases:
                    qref, sref = quantize_fp8(
                        operand.contiguous(), block_rows=1, block_cols=128
                    )
                    assert torch.equal(
                        qpool[start : start + count].view(torch.uint8),
                        qref.view(torch.uint8),
                    )
                    torch.testing.assert_close(
                        spool[start : start + count], sref, rtol=0, atol=0
                    )
            if cache_dz:
                padded_end = start + ((count + 127) // 128) * 128
                assert (
                    torch.count_nonzero(pools[8][start + count : padded_end].float())
                    == 0
                )
                assert torch.all(pools[9][start + count : padded_end] == 2.0**-22)
            matches = (xp[:, None, :8] == source_x[None, :, :8]).all(-1)
            assert (matches.sum(-1) == 1).all()
            source_rows = matches.int().argmax(-1)
            torch.testing.assert_close(xp, source_x[source_rows], rtol=0, atol=0)
            torch.testing.assert_close(yp, source_dy[source_rows], rtol=0, atol=0)
            route = source_ids[source_rows] == rank * 8 + expert
            assert (route.sum(-1) == 1).all()
            weight = (source_scores[source_rows] * route).sum(-1, keepdim=True)
            z = recompute(xp, decoded[0][expert])
            gate, up = z.chunk(2, dim=-1)
            dh = gemm(yp, decoded[1][expert])
            sigmoid = gate.sigmoid()
            silu = gate * sigmoid
            expected_h = (silu * up * weight).bfloat16()
            expected_dz = torch.cat(
                (
                    dh * weight * up * (sigmoid + silu * (1 - sigmoid)),
                    dh * weight * silu,
                ),
                -1,
            ).bfloat16()
            checks = {
                "h": (hp, expected_h),
                "dz": (zp, expected_dz),
                "dw1": (zp.float().T @ xp.float(), expected_dz.float().T @ xp.float()),
                "dw2": (yp.float().T @ hp.float(), yp.float().T @ expected_h.float()),
            }
            for name, (actual, expected) in checks.items():
                errors[name] = max(errors[name], relative(actual, expected))
        histogram = torch.bincount(ids[ids >= 0], minlength=experts)
        dist.all_reduce(histogram)
        torch.testing.assert_close(counts.long(), histogram[rank * 8 : (rank + 1) * 8])
        used_pool_rows = int((((counts + 127) // 128) * 128).sum().item())
        if skew and rank == 0:
            assert used_pool_rows > buffer.l1_acts.shape[0], (
                "Ring-wrap case did not wrap"
            )
        print(
            json.dumps(
                dict(
                    rank=rank,
                    hidden=hidden,
                    intermediate=inner,
                    compact_non_math_wg=compact,
                    trap_only_timeouts=trap_timeouts,
                    compute_dtype="blockwise_fp8",
                    transport_dtype="bf16",
                    activation_cache="x_dy_dz"
                    if cache_dz
                    else "x_dy"
                    if cached
                    else "none",
                    reference="quantized_operands_fp32_matmul",
                    skew=skew,
                    tokens=tokens,
                    errors=errors,
                    counts=counts.tolist(),
                    used_pool_rows=used_pool_rows,
                    ring_tokens=buffer.l1_acts.shape[0],
                )
            ),
            flush=True,
        )
        assert max(errors.values()) < 0.02, errors
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
