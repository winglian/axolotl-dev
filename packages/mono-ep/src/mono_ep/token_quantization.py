# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Single-pass per-token E4M3 quantization with power-of-two scales."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(
    X,
    Q,
    S,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    GROUPS: tl.constexpr,
    GRAN: tl.constexpr,
    TILE: tl.constexpr,
    PACKED: tl.constexpr,
    WEIGHTS: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    SF_ROW_STRIDE: tl.constexpr,
    EXPERT_ROWS: tl.constexpr,
    EXPERT_IDS=None,
    MASK_EXPERTS: tl.constexpr = False,
    TRANSPOSE: tl.constexpr = False,
):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    if ROWS * COLS > 2147483647:
        row = row.to(tl.int64)
    group = tl.program_id(1)
    col = group * GRAN + tl.arange(0, GRAN)
    valid = (row[:, None] < ROWS) & (col[None, :] < COLS)
    load_valid = valid
    if MASK_EXPERTS:
        expert = tl.load(EXPERT_IDS + row, row < ROWS, -1)
        load_valid = valid & (expert[:, None] >= 0)
    offset = row[:, None] * COLS + col[None, :]
    if TRANSPOSE:
        offset = (
            (row[:, None] // EXPERT_ROWS) * EXPERT_ROWS * COLS
            + col[None, :] * EXPERT_ROWS
            + row[:, None] % EXPERT_ROWS
        )
    value = tl.load(X + offset, load_valid, 0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(value), 1), 1.0e-4)
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (tl.minimum(tl.maximum(exponent, 1), 254) << 23).to(
        tl.float32, bitcast=True
    )
    quantized = tl.minimum(tl.maximum(value / scale[:, None], -448.0), 448.0)
    out_row = row
    if INTERLEAVE:
        local_row = row % EXPERT_ROWS
        half = EXPERT_ROWS // 2
        out_row = (
            (row // EXPERT_ROWS) * EXPERT_ROWS
            + (local_row % half // 8) * 16
            + (local_row // half) * 8
            + local_row % 8
        )
    tl.store(Q + out_row[:, None] * COLS + col[None, :], quantized, valid)
    if PACKED:
        if WEIGHTS:
            local_row = out_row % EXPERT_ROWS
            sf_row = local_row // 128 * 128 + local_row % 32 * 4 + local_row % 128 // 32
            offset = (
                (row // EXPERT_ROWS * (GROUPS // 4) + group // 4) * EXPERT_ROWS + sf_row
            ) * 4 + group % 4
        else:
            offset = row * GROUPS + group
        tl.store(
            S.to(tl.pointer_type(tl.uint8)) + offset,
            tl.minimum(tl.maximum(exponent, 1), 254).to(tl.uint8),
            row < ROWS,
        )
    else:
        tl.store(S + row * SF_ROW_STRIDE + group, scale, row < ROWS)


def quantize_tokens(x, block_cols, expert_ids=None):
    """Quantize independent rows of a contiguous BF16/FP32 tensor on CUDA."""
    if expert_ids is not None and (
        x.ndim != 2
        or expert_ids.shape != (x.shape[0],)
        or expert_ids.device != x.device
        or expert_ids.dtype != torch.int32
        or not expert_ids.is_contiguous()
    ):
        raise ValueError("Expected one contiguous INT32 expert ID per matrix row")
    cols = x.shape[-1]
    rows = x.numel() // cols
    groups = triton.cdiv(cols, block_cols)
    values = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((*x.shape[:-1], groups), device=x.device, dtype=torch.float32)
    if rows:
        _quantize[(triton.cdiv(rows, 16), groups)](
            x,
            values,
            scales,
            rows,
            cols,
            groups,
            block_cols,
            16,
            False,
            False,
            False,
            groups,
            x.shape[-2],
            expert_ids,
            expert_ids is not None,
            False,
            enable_fp_fusion=False,
        )
    return values, scales


def quantize_tokens_into(x, q_out, s_out, block_cols):
    """Quantize CUDA matrix rows into caller-owned values and strided scales."""
    if x.ndim != 2 or block_cols not in (32, 128):
        raise ValueError("invalid caller-owned FP8 token quantization buffers")
    groups = triton.cdiv(x.shape[1], block_cols)
    if (
        not x.is_cuda
        or not x.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float32)
        or x.shape[1] == 0
        or q_out.shape != x.shape
        or q_out.device != x.device
        or q_out.dtype != torch.float8_e4m3fn
        or not q_out.is_contiguous()
        or s_out.shape != (x.shape[0], groups)
        or s_out.device != x.device
        or s_out.dtype != torch.float32
        or s_out.stride(1) != 1
        or s_out.stride(0) < groups
    ):
        raise ValueError("invalid caller-owned FP8 token quantization buffers")
    rows, cols = x.shape
    if rows:
        _quantize[(triton.cdiv(rows, 16), groups)](
            x,
            q_out,
            s_out,
            rows,
            cols,
            groups,
            block_cols,
            16,
            False,
            False,
            False,
            s_out.stride(0),
            rows,
            x,
            False,
            False,
            enable_fp_fusion=False,
        )
    return q_out, s_out


def prepare_persistent_mxfp8(x, *, weights=False, interleave=False):
    """Quantize into the fused SM100 kernel's value and packed E8M0 layouts."""
    if (
        x.ndim not in (2, 3)
        or not x.is_cuda
        or not x.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float32)
        or x.shape[-1] % 128
        or x.shape[-1] == 0
        or (weights and x.shape[-2] % 128)
        or (interleave and (not weights or x.shape[-2] % 256))
    ):
        raise ValueError("invalid fused MXFP8 operand geometry or dtype")
    cols = x.shape[-1]
    rows = x.numel() // cols
    values = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    shape = (
        (*x.shape[:-2], cols // 128, x.shape[-2])
        if weights
        else (*x.shape[:-1], cols // 128)
    )
    scales = torch.empty(shape, device=x.device, dtype=torch.int32)
    if weights:
        scales = scales.transpose(-1, -2)
    if rows:
        _quantize[(triton.cdiv(rows, 16), cols // 32)](
            x,
            values,
            scales,
            rows,
            cols,
            cols // 32,
            32,
            16,
            True,
            weights,
            interleave,
            scales.stride(-2) if weights else scales.stride(-1) * scales.shape[-1],
            x.shape[-2],
            x,
            False,
            False,
            enable_fp_fusion=False,
        )
    return values, scales


def quantize_transposed_weights(x, block_cols=32):
    """Quantize a batched transposed view without materializing its source dtype."""
    if (
        x.ndim != 3
        or not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float32)
        or not x.transpose(-1, -2).is_contiguous()
        or block_cols not in (32, 128)
        or x.shape[-1] == 0
    ):
        raise ValueError("Expected transposed contiguous CUDA expert weights")
    experts, expert_rows, cols = x.shape
    rows = experts * expert_rows
    groups = triton.cdiv(cols, block_cols)
    values = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (experts, expert_rows, groups), device=x.device, dtype=torch.float32
    )
    if rows:
        _quantize[(triton.cdiv(rows, 16), groups)](
            x,
            values,
            scales,
            rows,
            cols,
            groups,
            block_cols,
            16,
            False,
            False,
            False,
            groups,
            expert_rows,
            x,
            False,
            True,
            enable_fp_fusion=False,
        )
    return values, scales


def quantize_persistent_mxfp8_into(x, q_out, s_out):
    """Quantize token rows into caller-owned E4M3 values and packed UE8M0 words (``[rows, cols / 128]``)."""
    rows, cols = x.shape if x.ndim == 2 else (-1, -1)
    if (
        x.ndim != 2
        or not x.is_cuda
        or not x.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float32)
        or cols % 128
        or q_out.shape != x.shape
        or q_out.dtype != torch.float8_e4m3fn
        or not q_out.is_contiguous()
        or s_out.shape != (rows, cols // 128)
        or s_out.element_size() != 4
        or s_out.stride(1) != 1
        or s_out.stride(0) != cols // 128
    ):
        raise ValueError("invalid caller-owned MXFP8 token buffers")
    if rows:
        _quantize[(triton.cdiv(rows, 16), cols // 32)](
            x,
            q_out,
            s_out,
            rows,
            cols,
            cols // 32,
            32,
            16,
            True,
            False,
            False,
            cols // 128,
            rows,
            x,
            False,
            False,
            enable_fp_fusion=False,
        )
    return q_out, s_out


def _mx_rows(value):
    """E4M3 values and UE8M0 exponents (int32) of rows along the last dim, bitwise as ``_quantize``."""
    groups = value.float().unflatten(-1, (-1, 32))
    amax = groups.abs().amax(-1).clamp_min(1.0e-4)
    bits = (amax / 448.0).view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()).clamp(1, 254)
    scale = (exponent << 23).view(torch.float32)
    quantized = (groups / scale[..., None]).clamp(-448.0, 448.0)
    return quantized.to(torch.float8_e4m3fn).flatten(-2), exponent


def _mx_words(exponent):
    """Pack four consecutive group exponents into one little-endian int32 word."""
    e = exponent.unflatten(-1, (-1, 4))
    return e[..., 0] | (e[..., 1] << 8) | (e[..., 2] << 16) | (e[..., 3] << 24)


def mxfp8_weight_form_builder():
    """``MONO_EP_SM100_MXFP8_WEIGHT_FORMS``: build the MXFP8 backward weight forms with the SM100
    CUDA tile kernel (``cuda``, default), the Triton one-read tile kernel (``fused``), one Inductor
    graph (``compiled``) or the Triton per-group kernel (``triton``); all four are bitwise
    identical. ``MONO_EP_SM100_MXFP8_COMPILED_QUANT`` (``1`` compiled, ``0`` triton) is honoured
    when the new selector is unset."""
    value = os.getenv("MONO_EP_SM100_MXFP8_WEIGHT_FORMS")
    if value is None:
        legacy = os.getenv("MONO_EP_SM100_MXFP8_COMPILED_QUANT")
        if legacy is None:
            return "cuda"
        if legacy not in ("0", "1"):
            raise ValueError("MONO_EP_SM100_MXFP8_COMPILED_QUANT must be 0 or 1")
        return "compiled" if legacy == "1" else "triton"
    if value not in ("cuda", "fused", "compiled", "triton"):
        raise ValueError(
            "MONO_EP_SM100_MXFP8_WEIGHT_FORMS must be cuda, fused, compiled or triton"
        )
    return value


def compiled_mxfp8_enabled():
    """Whether the Inductor weight-form builder is selected (see :func:`mxfp8_weight_form_builder`)."""
    return mxfp8_weight_form_builder() == "compiled"
