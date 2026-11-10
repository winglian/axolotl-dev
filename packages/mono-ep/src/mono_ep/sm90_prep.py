# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""One launch for the SM90 persistent kernels' token-side preparation.

Quantizes the token rows (and, for backward, the output-gradient rows) into the kernels' FP8
buffers with 1x128 power-of-two scales and copies the routing indices and weights into the
symmetric buffer, which otherwise takes one Triton launch per tensor plus two copy kernels.
Values and scales are bitwise equal to ``quantize_tokens_into``.
On by default; ``MONO_EP_SM90_FP8_FUSED_PREP=0`` restores the per-tensor launches.
"""

import os

import torch
import triton
import triton.language as tl

_TILE = 16


@triton.jit
def _quantize_rows(X, Q, S, row, rows, COLS: tl.constexpr, S_STRIDE: tl.constexpr):
    group = tl.program_id(1)
    col = group * 128 + tl.arange(0, 128)
    valid = (row[:, None] < rows) & (col[None, :] < COLS)
    offset = row[:, None] * COLS + col[None, :]
    value = tl.load(X + offset, valid, 0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(value), 1), 1.0e-4)
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (tl.minimum(tl.maximum(exponent, 1), 254) << 23).to(
        tl.float32, bitcast=True
    )
    quantized = tl.minimum(tl.maximum(value / scale[:, None], -448.0), 448.0)
    tl.store(Q + offset, quantized, valid)
    tl.store(S + row * S_STRIDE + group, scale, row < rows)


@triton.jit
def _prepare(
    X,
    QX,
    SX,
    DY,
    QDY,
    SDY,
    IDX,
    IDX_OUT,
    WEIGHTS,
    WEIGHTS_OUT,
    rows,
    COLS: tl.constexpr,
    SX_STRIDE: tl.constexpr,
    SDY_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    TILE: tl.constexpr,
):
    row = (tl.program_id(0) * TILE + tl.arange(0, TILE)).to(tl.int64)
    if tl.program_id(2) == 0:
        _quantize_rows(X, QX, SX, row, rows, COLS, SX_STRIDE)
        if tl.program_id(1) == 0:
            k = tl.arange(0, TOPK_P2)
            mask = (row[:, None] < rows) & (k[None, :] < TOPK)
            offset = row[:, None] * TOPK + k[None, :]
            tl.store(IDX_OUT + offset, tl.load(IDX + offset, mask), mask)
            tl.store(WEIGHTS_OUT + offset, tl.load(WEIGHTS + offset, mask), mask)
    else:
        _quantize_rows(DY, QDY, SDY, row, rows, COLS, SDY_STRIDE)


def enabled():
    value = os.getenv("MONO_EP_SM90_FP8_FUSED_PREP", "1")
    if value not in ("0", "1"):
        raise ValueError("MONO_EP_SM90_FP8_FUSED_PREP must be 0 or 1")
    return value == "1"


def prepare_tokens(
    x, qx, sx, indices, idx_out, scores, scores_out, dy=None, qdy=None, sdy=None
):
    """Quantize ``x`` (and ``dy``) and copy ``indices``/``scores`` in one launch.

    Output views must have the caller's row count; scale views may be row-strided.
    """
    rows, cols = x.shape
    topk = indices.shape[1]
    dense = [x, qx, indices, idx_out, scores, scores_out]
    if dy is not None:
        dense += [dy, qdy]
    strided = [sx] if dy is None else [sx, sdy]
    if (
        cols % 128
        or x.dtype not in (torch.bfloat16, torch.float32)
        or any(t.device != x.device for t in dense + strided)
        or any(not t.is_contiguous() for t in dense)
        or qx.shape != x.shape
        or qx.dtype != torch.float8_e4m3fn
        or sx.shape != (rows, cols // 128)
        or sx.dtype != torch.float32
        or sx.stride(1) != 1
        or indices.shape != (rows, topk)
        or idx_out.shape != indices.shape
        or scores.shape != indices.shape
        or scores_out.shape != indices.shape
        or (
            dy is not None
            and (
                dy.shape != x.shape
                or qdy.shape != x.shape
                or qdy.dtype != torch.float8_e4m3fn
                or sdy.shape != sx.shape
                or sdy.dtype != torch.float32
                or sdy.stride(1) != 1
            )
        )
    ):
        raise ValueError("invalid fused SM90 token preparation operands")
    if rows:
        second = dy is not None
        _prepare[(triton.cdiv(rows, _TILE), cols // 128, 2 if second else 1)](
            x,
            qx,
            sx,
            dy if second else x,
            qdy if second else qx,
            sdy if second else sx,
            indices,
            idx_out,
            scores,
            scores_out,
            rows,
            cols,
            sx.stride(0),
            sdy.stride(0) if second else sx.stride(0),
            topk,
            triton.next_power_of_2(topk),
            _TILE,
            enable_fp_fusion=False,
        )
