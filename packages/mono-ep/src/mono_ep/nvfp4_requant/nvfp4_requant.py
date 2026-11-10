# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Packed NVFP4 requantization of merged expert weights, in the ModelOpt layout.

Per optimizer step the merged expert weight ``W_eff = DQ(W_base) + s * B @ A``
(frozen packed base, trainable LoRA) is re-quantized to NVFP4 on the base's
fixed per-tensor scales and kept in the same three-part form a ModelOpt
checkpoint ships:

- ``qdata``  u8 ``[E, N, K/2]``: E2M1 codes, low nibble first along K
- ``scale``  e4m3 ``[E, N, K/16]``: one block scale per 16 consecutive K values
- ``per_tensor_scale`` fp32 ``[E, P]``: one scalar per (expert, projection),
  where the ``N`` rows of an expert split into ``P`` equal projection segments
  (``P = 2`` for a fused ``[gate; up]`` stack, ``P = 1`` for ``down``). Gate
  and up keep separate scalars; nothing here folds them into the block scales.

The block scales are additionally emitted in the 128x4 swizzled scale-factor
atom layout (``sf_layout.pack_scales_blocked``) per expert, so a block-scaled
consumer can TMA them without a repack.

Two numeric modes, both realized bit for bit by the Triton kernel in
:mod:`triton_nvfp4_requant` and by the pure-torch references here:

- default: torchao ``nvfp4_quantize`` numerics (the merge-identity quantizer
  ``nvfp4_quant.quantize_nvfp4_merge`` delegates to). Scale floor ``2**-6``,
  reciprocal-multiply encode, E2M1 sign taken from the sign bit.
- ``modelopt_compat``: ModelOpt ``NVFP4QTensor.quantize`` numerics. Zero blocks
  get scale 1.0, floor ``2**-9`` (e4m3 subnormals kept), true-division
  encode, E2M1 sign from ``x < 0``. Use it to reproduce shipped ModelOpt bytes
  on blocks the default mode cannot (zero or subnormal-scale blocks).

The CPU references are the oracles for the kernels; ``modelopt_quantize_ref``
replicates ModelOpt itself so tests can fabricate shipped bytes without the
``modelopt`` package.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sf_layout import SF_TILE_COLS, SF_TILE_ROWS, ceil_div, pack_scales_blocked

SF_VEC_SIZE = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
# torchao clamps block scales at the smallest e4m3 normal; ModelOpt at the
# smallest e4m3 subnormal and maps all-zero blocks to 1.0.
E4M3_EPS = 2.0**-6
MODELOPT_SCALE_FLOOR = 2.0**-9
PER_TENSOR_DIVISOR = E2M1_MAX * E4M3_MAX  # 2688: torchao and ModelOpt agree

DELTA_KERNELS = ("ieee", "bf16x3")
# fresh: block scales recomputed from the merged weight (torchao / ModelOpt
# fresh quantization). reuse: keep the base's block scales, recomputing only
# blocks whose merged amax outgrew 6 * scale * pts; on a checkpoint whose
# scales were calibrated rather than derived from the stored codes (QAD
# exports) this is the only mode under which Q(DQ(W)) == W.
SCALE_MODES = ("fresh", "reuse")


@dataclass(frozen=True)
class RequantNumerics:
    """Everything that decides the bytes, so trainer and merge writer can agree.

    ``round_dtype`` is the dtype the dequantized base and the merged weight are
    rounded to before quantization (the loader's ``orig_dtype``; bf16 for
    ModelOpt checkpoints). ``delta_kernel`` names the B @ A implementation:
    ``ieee`` (fp32 CUDA-core dot) or ``bf16x3`` (three bf16 tensor-core dots,
    hi*hi + hi*lo + lo*hi, small terms accumulated first).
    """

    modelopt_compat: bool = False
    delta_kernel: str = "ieee"
    round_dtype: torch.dtype = torch.bfloat16
    scale_mode: str = "fresh"

    def __post_init__(self):
        if self.delta_kernel not in DELTA_KERNELS:
            raise ValueError(
                f"delta_kernel must be one of {DELTA_KERNELS}, got {self.delta_kernel!r}"
            )
        if self.round_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                f"round_dtype must be bf16 or fp32, got {self.round_dtype}"
            )
        if self.scale_mode not in SCALE_MODES:
            raise ValueError(
                f"scale_mode must be one of {SCALE_MODES}, got {self.scale_mode!r}"
            )


@dataclass
class RequantBuffers:
    """Caller-owned output buffers of a requant: the SM100 engine's B operand
    (``qdata``), the row-major scales the BF16 dequant reads, and the swizzled
    SF atoms the engine reads. Allocated once per layer and rewritten in place
    every step; the atom padding is zero from the allocation and never written."""

    qdata: torch.Tensor  # u8 [E, N, K/2]
    scale: torch.Tensor  # u8 [E, N, K/16]
    atoms: torch.Tensor | None  # u8 [E, RM, RK, 512], or None

    @classmethod
    def allocate(cls, e: int, n: int, k: int, device, *, with_atoms: bool = True):
        sf_k = k // SF_VEC_SIZE
        rm, rk = atom_grid(n, sf_k)
        return cls(
            torch.empty(e, n, k // 2, dtype=torch.uint8, device=device),
            torch.empty(e, n, sf_k, dtype=torch.uint8, device=device),
            torch.zeros(e, rm, rk, 512, dtype=torch.uint8, device=device)
            if with_atoms
            else None,
        )

    def check(self, e: int, n: int, k: int) -> None:
        sf_k = k // SF_VEC_SIZE
        rm, rk = atom_grid(n, sf_k)
        for name, t, shape in (
            ("qdata", self.qdata, (e, n, k // 2)),
            ("scale", self.scale, (e, n, sf_k)),
            ("atoms", self.atoms, (e, rm, rk, 512)),
        ):
            if t is None:
                continue
            assert t.dtype == torch.uint8 and t.is_contiguous(), name
            assert tuple(t.shape) == shape, f"{name} {tuple(t.shape)} vs {shape}"


@dataclass
class RequantResult:
    """Packed NVFP4 of ``[E_local, N, K]`` merged experts (experts ``e0:e1``).

    ``row_order`` is ``"concat"`` (the checkpoint's ``[gate; up]`` rows) or
    ``"interleaved"`` (the gated SM100 engine's ``g0, u0, g1, u1, ...`` rows).
    """

    qdata: torch.Tensor  # u8 [E_local, N, K/2]
    scale: torch.Tensor  # float8_e4m3fn [E_local, N, K/16]
    scale_atoms: torch.Tensor | None  # u8 [E_local, RM, RK, 512] swizzled, or None
    per_tensor_scale: torch.Tensor | None  # fp32 [E_local, P], or None (single level)
    numerics: RequantNumerics
    e0: int = 0
    row_order: str = "concat"

    @property
    def num_projections(self) -> int:
        return (
            1 if self.per_tensor_scale is None else int(self.per_tensor_scale.shape[1])
        )


def per_tensor_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """``amax / (6 * 448)``: torchao ``per_tensor_amax_to_scale`` and ModelOpt
    ``get_weights_scaling_factor_2`` are this same fp32 division."""
    return amax.to(torch.float32) / PER_TENSOR_DIVISOR


def expand_pts_rows(pts: torch.Tensor | None, num_rows: int) -> torch.Tensor | None:
    """``[E, P]`` per-(expert, projection) scales -> one fp32 scale per row of
    the flattened ``[E * num_rows, K]`` stack (projection ``p`` owns rows
    ``p * num_rows / P ... (p + 1) * num_rows / P`` of each expert)."""
    if pts is None:
        return None
    pts = pts.to(torch.float32)
    if pts.dim() == 1:
        pts = pts[:, None]
    e, p = pts.shape
    assert num_rows % p == 0, f"N={num_rows} not divisible by P={p} projections"
    return pts.repeat_interleave(num_rows // p, dim=1).reshape(e * num_rows)


def round_e4m3(s: torch.Tensor) -> torch.Tensor:
    """RNE onto e4m3 (subnormals included) and back to fp32, torch semantics."""
    return s.to(torch.float8_e4m3fn).to(torch.float32)


def e4m3_bits(s: torch.Tensor) -> torch.Tensor:
    """Bit pattern (u8) of an fp32 tensor that already holds exact e4m3 values."""
    return s.to(torch.float8_e4m3fn).view(torch.uint8)


def _e2m1_magnitude_index(a: torch.Tensor) -> torch.Tensor:
    """RNE bucket of ``|q|`` onto (0, .5, 1, 1.5, 2, 3, 4, 6): ties go to the even
    code, hence the alternating strict / non-strict boundaries."""
    return (
        (a > 0.25).to(torch.int32)
        + (a >= 0.75).to(torch.int32)
        + (a > 1.25).to(torch.int32)
        + (a >= 1.75).to(torch.int32)
        + (a > 2.5).to(torch.int32)
        + (a >= 3.5).to(torch.int32)
        + (a > 5.0).to(torch.int32)
    )


def encode_e2m1_rne(q: torch.Tensor, *, sign_from_bit: bool) -> torch.Tensor:
    """fp32 ``q`` (already scaled to the E2M1 range) -> 4-bit codes (u8).

    ``sign_from_bit``: torchao keeps the fp32 sign bit (``-0.0`` and negatives
    rounding to zero become code 8); ModelOpt uses ``q < 0`` (code 0 there).
    """
    q = q.float()
    idx = _e2m1_magnitude_index(q.abs())
    if sign_from_bit:
        sign = (q.view(torch.int32) >> 31) & 1
    else:
        sign = (q < 0).to(torch.int32)
    return (idx | (sign << 3)).to(torch.uint8)


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """4-bit codes ``[..., K]`` -> bytes ``[..., K/2]``, low nibble first."""
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)


def code_values(codes: torch.Tensor) -> torch.Tensor:
    """E2M1 code -> fp32 value; code 8 is ``-0.0`` (what the Triton dequant and
    torchao's unpack produce), so sign-of-zero survives a dequant/requant."""
    mags = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=torch.float32,
        device=codes.device,
    )
    mag = mags[(codes & 7).long()]
    neg = (codes & 8) != 0
    return torch.where(neg, -mag, mag)


def quantize_nvfp4_packed_ref(
    x: torch.Tensor,
    pts_rows: torch.Tensor | None,
    *,
    modelopt_compat: bool = False,
    base_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch oracle for the packed requant kernels. ``x [rows, K]`` (any
    float dtype; values are taken as-is in fp32), ``pts_rows`` fp32 ``[rows]``
    or None (single-level scaling). Returns ``(packed u8 [rows, K/2],
    scale e4m3 [rows, K/16])``.

    Default mode is op-for-op torchao 0.18 ``nvfp4_quantize``: block scale
    ``amax * (1/6)``, two-level ``/ pts``, clamp ``[2**-6, 448]``, e4m3 RNE,
    encode by multiplying with ``(1/pts) / scale``, clamp to ``[-6, 6]``, RNE
    onto E2M1 keeping the sign bit.

    ``modelopt_compat`` is op-for-op ModelOpt ``NVFP4QTensor.quantize``: block
    scale ``amax / (6 * pts)``, zero -> 1.0, clamp ``[2**-9, 448]``, e4m3 RNE,
    encode by true division with ``scale * pts``, sign from ``x < 0``.

    ``base_scale`` (e4m3 ``[rows, K/16]``) switches to the reuse grid: a block
    keeps its base scale unless ``amax > 6 * pts * base_scale``, in which case
    it gets the fresh scale of the active mode; encoding is unchanged.
    """
    rows, k = x.shape
    assert k % SF_VEC_SIZE == 0
    xf = x.float().reshape(rows, k // SF_VEC_SIZE, SF_VEC_SIZE)
    amax = xf.abs().amax(dim=-1)
    pts = None if pts_rows is None else pts_rows.float().reshape(rows, 1)
    outgrown = s_base = None
    if base_scale is not None:
        s_base = base_scale.view(torch.uint8).view(torch.float8_e4m3fn).float()
        hp_base = s_base if pts is None else pts * s_base
        outgrown = amax > 6.0 * hp_base
    if modelopt_compat:
        denom = 6.0 * pts if pts is not None else torch.full_like(amax, 6.0)
        bs = amax / denom
        bs = torch.where(bs == 0, torch.ones_like(bs), bs)
        sdec = round_e4m3(bs.clamp(MODELOPT_SCALE_FLOOR, E4M3_MAX))
        if outgrown is not None:
            sdec = torch.where(outgrown, sdec, s_base)
        total = sdec * pts if pts is not None else sdec
        q = xf / total.unsqueeze(-1)
        codes = encode_e2m1_rne(q, sign_from_bit=False)
    else:
        bs = amax * (1.0 / 6.0)
        if pts is not None:
            bs = bs / pts
        sdec = round_e4m3(bs.clamp(E4M3_EPS, E4M3_MAX))
        if outgrown is not None:
            sdec = torch.where(outgrown, sdec, s_base)
        r = (1.0 / pts) / sdec if pts is not None else 1.0 / sdec
        q = (xf * r.unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
        codes = encode_e2m1_rne(q, sign_from_bit=True)
    packed = pack_codes(codes.reshape(rows, k))
    return packed, e4m3_bits(sdec).view(torch.float8_e4m3fn)


def dequantize_nvfp4_packed_ref(
    packed: torch.Tensor,
    scale: torch.Tensor,
    pts_rows: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``out_dtype(code * fp32(pts * scale))``: the loader / trainer dequant
    order (``pts * scale`` first), for ``packed [rows, K/2]``."""
    vals = code_values(unpack_codes(packed))
    hp = scale.view(torch.uint8).view(torch.float8_e4m3fn).float()
    if pts_rows is not None:
        hp = pts_rows.float().reshape(-1, 1) * hp
    return (vals * hp.repeat_interleave(SF_VEC_SIZE, dim=-1)).to(out_dtype)


def modelopt_quantize_ref(
    w: torch.Tensor, per_tensor_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Literal replica of ModelOpt's ``NVFP4QTensor.quantize`` for a 2D weight
    ``[N, K]`` (searchsorted + odd-bound bump encode, zero blocks -> 1.0, clamp
    ``[2**-9, 448]``). Returns ``(weight u8 [N, K/2], weight_scale e4m3
    [N, K/16], weight_scale_2 fp32 scalar)``. Used to fabricate shipped bytes.
    """
    n, k = w.shape
    ws2 = (
        per_tensor_scale_from_amax(w.abs().amax())
        if per_tensor_scale is None
        else per_tensor_scale.float()
    )
    blocks = w.reshape(n, k // SF_VEC_SIZE, SF_VEC_SIZE)
    block_amax = blocks.abs().amax(dim=-1).float()
    bs = block_amax / (E2M1_MAX * ws2)
    bs = torch.where(bs == 0, torch.ones_like(bs), bs)
    ws = bs.clamp(min=MODELOPT_SCALE_FLOOR, max=E4M3_MAX).to(torch.float8_e4m3fn)
    scaled = blocks / (ws.to(torch.float32) * ws2).unsqueeze(-1)
    scaled = scaled.reshape(n, k)
    sign_bit = (scaled < 0).to(torch.uint8)
    a = scaled.abs()
    bounds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=a.dtype, device=a.device
    )
    ordv = torch.searchsorted(bounds, a.contiguous(), out_int32=True).to(torch.uint8)
    odd = torch.any(a.unsqueeze(-1) == bounds[[1, 3, 5]], dim=-1).to(torch.uint8)
    codes = (sign_bit << 3) + ordv + odd
    packed = torch.bitwise_left_shift(codes[..., 1::2], 4) | codes[..., 0::2]
    return packed, ws, ws2


def scale_atoms_from_rowmajor(scale: torch.Tensor) -> torch.Tensor:
    """``[E, N, K/16]`` block scales -> ``[E, RM, RK, 512]`` u8 swizzled atoms
    (zero padded), one atom grid per expert."""
    return pack_scales_blocked(scale.view(torch.uint8)).view(torch.uint8)


def atom_grid(n: int, sf_k: int) -> tuple[int, int]:
    return ceil_div(n, SF_TILE_ROWS), ceil_div(sf_k, SF_TILE_COLS)


def requant_reference(
    base: dict,
    *,
    lora: tuple[torch.Tensor, torch.Tensor, float, int],
    out_pts: torch.Tensor | None,
    e0: int,
    e1: int,
    numerics: RequantNumerics,
    with_atoms: bool = True,
) -> RequantResult:
    """Pure-torch end-to-end reference for the packed requant.

    ``base = {"qdata", "scale", "per_tensor_scale"}`` (``[E, N, K/2]``,
    ``[E, N, K/16]``, fp32 ``[E, P]`` or None) and ``lora = (A, B, scaling, r)``
    with rank-major PEFT factors ``A [E*r, K]``, ``B [N, r*E]``. ``out_pts`` is
    the output per-(expert, projection) scale ``[E, P]`` or None.
    """
    qdata, scale, dq_pts = base["qdata"], base["scale"], base["per_tensor_scale"]
    E, N, k2 = qdata.shape
    K = 2 * k2
    rows = (e1 - e0) * N
    dq_rows = expand_pts_rows(dq_pts[e0:e1] if dq_pts is not None else None, N)
    w = dequantize_nvfp4_packed_ref(
        qdata[e0:e1].reshape(rows, k2),
        scale[e0:e1].reshape(rows, K // SF_VEC_SIZE),
        dq_rows,
        numerics.round_dtype,
    ).reshape(e1 - e0, N, K)
    A, B, scaling, r = lora
    A3 = A.reshape(E, r, K)[e0:e1].float()
    B3 = B.reshape(N, r, E).permute(2, 0, 1)[e0:e1].float()
    w_eff = (w.float() + torch.bmm(B3, A3) * scaling).to(numerics.round_dtype)
    pts_local = None if out_pts is None else out_pts[e0:e1].to(torch.float32)
    packed, sc = quantize_nvfp4_packed_ref(
        w_eff.reshape(rows, K),
        expand_pts_rows(pts_local, N),
        modelopt_compat=numerics.modelopt_compat,
        base_scale=(
            scale[e0:e1].reshape(rows, K // SF_VEC_SIZE)
            if numerics.scale_mode == "reuse"
            else None
        ),
    )
    packed = packed.reshape(e1 - e0, N, K // 2)
    sc = sc.reshape(e1 - e0, N, K // SF_VEC_SIZE)
    atoms = scale_atoms_from_rowmajor(sc) if with_atoms else None
    return RequantResult(packed, sc, atoms, pts_local, numerics, e0)
