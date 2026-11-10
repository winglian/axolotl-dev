# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""SM100 CUDA packed requant of merged NVFP4 LoRA experts (``include/nvfp4_requant_launch.cu``).

Same contract and bytes as :func:`triton_nvfp4_requant.requant_lora_packed`
with ``delta_kernel="ieee"`` (qdata, row-major scales, 128x4 SF atoms, gate/up
row interleave, ``scale_mode="reuse"``, ModelOpt numerics), with the E2M1 /
E4M3 conversions on the SM100 ``cvt`` instructions instead of select chains.
The adapters are taken in their natural per-expert layouts ``A [E, r, K]``,
``B [E, N, r]``, so callers holding PEFT-style ``[E, r, K]`` / ``[E, N, r]``
parameters need no rank-major copy.

``requant_fused`` is a numerics change, not a faster copy of the same op: it
quantizes one merged tile twice, along K for the normal form and along N
(stored transposed) for the backward's form, so the transposed form is built
from the merged weight directly instead of from a separately requantized
transposed base.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from .nvfp4_requant import (
    DELTA_KERNELS,
    SF_VEC_SIZE,
    RequantBuffers,
    RequantNumerics,
    RequantResult,
    atom_grid,
)
from .triton_nvfp4_requant import _pts_table

__all__ = [
    "cuda_requant_available",
    "requant_fused",
    "requant_lora_packed_cuda",
    "requant_natural",
]

TILE_M, TILE_K = 64, 256
MODES = {"requant": 3, "copy": 0, "no_delta": 1, "no_quantizer": 2}


@functools.lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    include = Path(__file__).resolve().parents[1] / "include"
    verbose = os.environ.get("MONO_EP_NVFP4_REQUANT_VERBOSE") == "1"
    return load(
        name="mono_ep_nvfp4_requant",
        sources=[
            str(include / "nvfp4_requant_bindings.cpp"),
            str(include / "nvfp4_requant_launch.cu"),
        ],
        extra_include_paths=[str(include)],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--fmad=false",
            "-gencode=arch=compute_100a,code=sm_100a",
            *(["-Xptxas=-v"] if verbose else []),
        ],
        verbose=verbose,
    )


def cuda_requant_available(device=None) -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device)[0] == 10


def _ffma2_default() -> bool:
    return os.environ.get("MONO_EP_NVFP4_REQUANT_FFMA2", "1") != "0"


def _k_splits(n, k, e_local, device) -> int:
    k_tiles = -(-k // TILE_K)
    ctas = -(-n // TILE_M) * e_local
    target = 8 * torch.cuda.get_device_properties(device).multi_processor_count
    return max(1, min(k_tiles, -(-target // ctas)))


def _r_pad(r: int) -> int:
    r_pad = max(16, 1 << (r - 1).bit_length())
    if r_pad > 64:
        raise NotImplementedError(f"rank {r} > 64")
    return r_pad


def _check_inputs(q, s, a, b):
    E, N, k2 = q.shape
    K = 2 * k2
    assert K % 64 == 0, f"K={K} must be a multiple of 64"
    assert s.shape == (E, N, K // SF_VEC_SIZE), tuple(s.shape)
    assert a.dim() == 3 and a.shape[0] == E and a.shape[2] == K, tuple(a.shape)
    r = a.shape[1]
    assert b.shape == (E, N, r), tuple(b.shape)
    assert a.dtype == b.dtype and a.dtype in (torch.bfloat16, torch.float32), (
        a.dtype,
        b.dtype,
    )
    return E, N, K, r


PROBES = (
    "e2m1_encode",
    "e4m3_encode",
    "e2m1_decode_lo",
    "e2m1_decode_hi",
    "e4m3_decode_lo",
    "e4m3_decode_hi",
)


def convert_probe(kind: str, words: torch.Tensor) -> torch.Tensor:
    """Run one of the kernel's conversion primitives on int32 ``words`` (tests)."""
    words = words.contiguous().view(torch.int32)
    out = torch.empty_like(words)
    stream = torch.cuda.current_stream(words.device).cuda_stream
    err = _extension().probe(
        PROBES.index(kind), words.data_ptr(), out.data_ptr(), words.numel(), stream
    )
    if err:
        raise RuntimeError(f"nvfp4 requant probe failed with code {err}")
    return out


def _launch(params, device):
    stream = torch.cuda.current_stream(device).cuda_stream
    err = _extension().launch(params, stream)
    if err:
        raise RuntimeError(f"nvfp4 requant launch failed with code {err}")


def _base_params(q, s, dq_pts, a, b, scaling, fq_pts, numerics, e0, e_local, N, K, r):
    return dict(
        q=q.data_ptr(),
        s=s.data_ptr(),
        dq_pts=dq_pts.data_ptr(),
        a=a.data_ptr(),
        b=b.data_ptr(),
        fq_pts=fq_pts.data_ptr(),
        scaling=float(scaling),
        e0=e0,
        e_local=e_local,
        n=N,
        k=K,
        r=r,
        r_pad=_r_pad(r),
        round_bf16=int(numerics.round_dtype == torch.bfloat16),
        modelopt=int(numerics.modelopt_compat),
        reuse=int(numerics.scale_mode == "reuse"),
        ab_fp32=int(a.dtype == torch.float32),
        oq_t=0,
        os_t=0,
        osfa_t=0,
        rm_t=0,
        rk_t=0,
    )


def _pts_pair(base_pts, out_pts, E, q, numerics):
    dq_pts, p_dq, has_dq = _pts_table(base_pts, E, q)
    fq_pts, p_fq, has_fq = _pts_table(out_pts, E, q)
    if has_dq and has_fq and p_dq != p_fq:
        raise ValueError(
            f"base per-tensor scales have {p_dq} projections, output has {p_fq}"
        )
    p = max(p_dq if has_dq else 1, p_fq if has_fq else 1)
    if numerics.scale_mode == "reuse":
        same = (not has_fq and not has_dq) or (
            has_fq and has_dq and (out_pts is base_pts or torch.equal(fq_pts, dq_pts))
        )
        if not same:
            raise ValueError(
                "scale_mode='reuse' keeps the base block scales, so the output "
                "per-tensor scales must equal the base per-tensor scales"
            )
    # a missing level is exactly a scale of 1.0 in every op the kernel does with it
    ones = torch.ones(E, p, device=q.device, dtype=torch.float32)
    dq = dq_pts.expand(E, p).contiguous() if has_dq else ones
    fq = fq_pts.expand(E, p).contiguous() if has_fq else ones
    return dq, fq, p, has_fq, fq_pts


def requant_natural(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    base_pts: torch.Tensor | None,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    out_pts: torch.Tensor | None,
    e0: int = 0,
    e1: int | None = None,
    *,
    numerics: RequantNumerics | None = None,
    write_atoms: bool = True,
    out: RequantBuffers | None = None,
    interleave: int = 0,
    mode: str = "requant",
    ffma2: bool | None = None,
) -> RequantResult:
    """``Q(round(DQ(base) + scaling * B_e @ A_e))`` for experts ``e0:e1`` with
    ``lora_a [E, r, K]``, ``lora_b [E, N, r]`` (bf16 or fp32, contiguous).

    ``interleave = g`` stores concat ``[gate; up]`` row ``n`` at the gated
    engine's interleaved row (granularity ``g``: 1 is the Triton kernel's
    ``interleave_rows``, 8 is ``deep_gemm.mega._interleave_weights``); the
    per-projection scales still follow the concat rows. ``mode`` other than
    ``"requant"`` runs a bound probe (copy, no delta, no quantizer) whose bytes
    are meaningless.
    """
    numerics = numerics or RequantNumerics()
    if numerics.delta_kernel != DELTA_KERNELS[0]:
        raise NotImplementedError("the CUDA requant implements delta_kernel='ieee'")
    q = qdata.view(torch.uint8).contiguous()
    s = scale.view(torch.uint8).contiguous()
    a, b = lora_a.contiguous(), lora_b.contiguous()
    E, N, K, r = _check_inputs(q, s, a, b)
    e1 = E if e1 is None else e1
    e_local = e1 - e0
    dq, fq, p, has_fq, fq_table = _pts_pair(base_pts, out_pts, E, q, numerics)
    assert N % p == 0, f"N={N} not divisible by {p} projections"
    if interleave:
        assert N % (2 * interleave) == 0, (N, interleave)
    if out is None:
        out = RequantBuffers.allocate(e_local, N, K, q.device, with_atoms=write_atoms)
    else:
        out.check(e_local, N, K)
    atoms = out.atoms if write_atoms else None
    rm, rk = atom_grid(N, K // SF_VEC_SIZE)
    params = _base_params(
        q, s, dq, a, b, scaling, fq, numerics, e0, e_local, N, K, r
    ) | dict(
        oq=out.qdata.data_ptr(),
        os=out.scale.data_ptr(),
        osfa=atoms.data_ptr() if atoms is not None else 0,
        p=p,
        rows_per_proj=N // p,
        rm=rm,
        rk=rk,
        interleave=int(interleave),
        fused=0,
        ffma2=int(_ffma2_default() if ffma2 is None else ffma2),
        mode=MODES[mode],
        k_splits=_k_splits(N, K, e_local, q.device),
    )
    _launch(params, q.device)
    return RequantResult(
        out.qdata,
        out.scale.view(torch.float8_e4m3fn),
        atoms,
        fq_table[e0:e1].clone() if has_fq else None,
        numerics,
        e0,
        row_order="interleaved" if interleave else "concat",
    )


def requant_fused(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    pts: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    e0: int = 0,
    e1: int | None = None,
    *,
    write_atoms: bool = True,
    interleave: int = 0,
    ffma2: bool | None = None,
) -> tuple[RequantResult, RequantResult]:
    """The normal and the transposed packed forms of one merged weight.

    Normal: the bytes of :func:`requant_natural` (default numerics). Transposed:
    ``Q_N(W)`` stored ``[E, K, N/2]``, i.e. the same bf16 merged tile quantized
    in 16-row blocks along N with fresh torchao scales on the same per-tensor
    scale, in natural (concat) row order. One per-tensor scale per expert
    (``pts [E]``); ``N % 64 == 0``.
    """
    numerics = RequantNumerics()
    q = qdata.view(torch.uint8).contiguous()
    s = scale.view(torch.uint8).contiguous()
    a, b = lora_a.contiguous(), lora_b.contiguous()
    E, N, K, r = _check_inputs(q, s, a, b)
    assert N % TILE_M == 0, f"N={N} must be a multiple of {TILE_M}"
    pts = pts.detach().to(torch.float32).reshape(E, -1).contiguous()
    assert pts.shape[1] == 1, "the fused requant takes one per-tensor scale per expert"
    if interleave:
        assert N % (2 * interleave) == 0, (N, interleave)
    e1 = E if e1 is None else e1
    e_local = e1 - e0
    normal = RequantBuffers.allocate(e_local, N, K, q.device, with_atoms=write_atoms)
    trans = RequantBuffers.allocate(e_local, K, N, q.device, with_atoms=write_atoms)
    rm, rk = atom_grid(N, K // SF_VEC_SIZE)
    rm_t, rk_t = atom_grid(K, N // SF_VEC_SIZE)
    params = _base_params(q, s, pts, a, b, scaling, pts, numerics, e0, e_local, N, K, r)
    params.update(
        oq=normal.qdata.data_ptr(),
        os=normal.scale.data_ptr(),
        osfa=normal.atoms.data_ptr() if write_atoms else 0,
        oq_t=trans.qdata.data_ptr(),
        os_t=trans.scale.data_ptr(),
        osfa_t=trans.atoms.data_ptr() if write_atoms else 0,
        p=1,
        rows_per_proj=N,
        rm=rm,
        rk=rk,
        rm_t=rm_t,
        rk_t=rk_t,
        interleave=int(interleave),
        fused=1,
        ffma2=int(_ffma2_default() if ffma2 is None else ffma2),
        mode=MODES["requant"],
        k_splits=_k_splits(N, K, e_local, q.device),
    )
    _launch(params, q.device)
    pts_local = pts[e0:e1].clone()
    return (
        RequantResult(
            normal.qdata,
            normal.scale.view(torch.float8_e4m3fn),
            normal.atoms,
            pts_local,
            numerics,
            e0,
            row_order="interleaved" if interleave else "concat",
        ),
        RequantResult(
            trans.qdata,
            trans.scale.view(torch.float8_e4m3fn),
            trans.atoms,
            pts_local.clone(),
            numerics,
            e0,
        ),
    )


def requant_lora_packed_cuda(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    base_pts: torch.Tensor | None,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
    out_pts: torch.Tensor | None,
    e0: int = 0,
    e1: int | None = None,
    *,
    numerics: RequantNumerics | None = None,
    write_atoms: bool = True,
    out: RequantBuffers | None = None,
    interleave_rows: bool = False,
    **_unused,
) -> RequantResult:
    """Drop-in for ``triton_nvfp4_requant.requant_lora_packed`` (rank-major PEFT
    factors ``A [E*r, K]``, ``B [N, r*E]``); ``block`` / ``num_warps`` are ignored."""
    E, N, k2 = qdata.shape
    r = lora_A.shape[0] // E
    a = lora_A.reshape(E, r, 2 * k2)
    b = lora_B.reshape(N, r, E).permute(2, 0, 1)
    return requant_natural(
        qdata,
        scale,
        base_pts,
        a,
        b,
        scaling,
        out_pts,
        e0,
        e1,
        numerics=numerics,
        write_atoms=write_atoms,
        out=out,
        interleave=1 if interleave_rows else 0,
    )
