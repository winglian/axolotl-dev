# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""SM100 expert weight gradients from the persistent backward's pools.

The pools hold each local expert's rows from ``starts[e]`` (a multiple of 128) to
``ends[e]``, with the tail of its last 128-row block zeroed. The k-grouped GEMM
writes ``out[e] = a_pool[rows of e]^T b_pool[rows of e]`` straight into an
uninitialized FP32 output; only the slabs of experts with no rows, which it leaves
unwritten, are zeroed afterwards. ``MONO_EP_SM100_DW_PRECISION=mxfp8`` (opt-in) quantizes
the pools to E4M3 with one UE8M0 scale per 32 pool rows and column (from the BF16 pools, or
from the MXFP8 backward's row-wise E4M3 pools and their scales) and runs DeepGEMM's
k-grouped ``recipe=(1, 1, 32)`` GEMM instead.
"""

import os

import torch
import triton
import triton.language as tl


def direct_dw():
    """``MONO_EP_SM100_DW_DIRECT``: ``1`` (default) writes D directly, ``0`` accumulates into zeros."""
    value = os.getenv("MONO_EP_SM100_DW_DIRECT", "1")
    if value not in ("0", "1"):
        raise ValueError("MONO_EP_SM100_DW_DIRECT must be 0 or 1")
    return value == "1"


def dw_precision():
    value = os.getenv("MONO_EP_SM100_DW_PRECISION", "bf16")
    if value not in ("bf16", "mxfp8"):
        raise ValueError("MONO_EP_SM100_DW_PRECISION must be bf16 or mxfp8")
    return value


@triton.jit
def _requant_block(
    block,
    X,
    SW,
    SF_ROWS,
    Q,
    S,
    STARTS,
    ENDS,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
    COLS: tl.constexpr,
    COL_BLOCKS: tl.constexpr,
    FROM_MX: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    # One 128-row block and COL_BLOCKS x 128 columns of a pool: E4M3 values with one UE8M0 scale per
    # 32 rows and column, packed per column into one int32 (rows 32 j .. 32 j + 31 in byte j). Rows
    # outside the owning expert's [start, end) are written as zero. FROM_MX reads the backward's
    # row-wise E4M3 pool and its packed scale words ([COLS / 128][SF_ROWS], UTCCP row order)
    # instead of BF16.
    experts = tl.arange(0, EXPERTS_POW2)
    starts = tl.load(STARTS + experts, experts < NUM_EXPERTS, other=2147483647)
    owner = tl.max(tl.where(starts <= block * 128, experts, -1), 0)
    end = tl.load(ENDS + tl.maximum(owner, 0))
    if FROM_MX:
        shift = (tl.arange(0, 128) // 32) * 8
    for cb in tl.static_range(COL_BLOCKS):
        col_block = tl.program_id(1) * COL_BLOCKS + cb
        col = col_block * 128 + tl.arange(0, 128)
        packed = tl.zeros([128], dtype=tl.int32)
        for j in tl.static_range(4):
            r = j * 32 + tl.arange(0, 32)
            row = block.to(tl.int64) * 128 + r
            valid = (row < end) & (owner >= 0)
            value = tl.load(
                X + row[:, None] * COLS + col[None, :], valid[:, None], other=0.0
            ).to(tl.float32)
            if FROM_MX:
                sf_row = block.to(tl.int64) * 128 + (r % 32) * 4 + r // 32
                word = tl.load(SW + col_block.to(tl.int64) * SF_ROWS + sf_row)
                # Stale words of masked rows may hold 255: keep the scale finite so 0 * scale stays 0
                exponent = tl.minimum((word[:, None] >> shift[None, :]) & 255, 254)
                value = value * (exponent << 23).to(tl.float32, bitcast=True)
            amax = tl.maximum(tl.max(tl.abs(value), 0), 1.0e-4)
            bits = (amax / 448.0).to(tl.int32, bitcast=True)
            exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
            exponent = tl.minimum(tl.maximum(exponent, 1), 254)
            if RECIPROCAL:
                # The scale is a power of two, so multiplying by its exact inverse rounds like the division
                inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
                quantized = tl.minimum(
                    tl.maximum(value * inverse[None, :], -448.0), 448.0
                )
            else:
                scale = (exponent << 23).to(tl.float32, bitcast=True)
                quantized = tl.minimum(
                    tl.maximum(value / scale[None, :], -448.0), 448.0
                )
            tl.store(
                Q + row[:, None] * COLS + col[None, :],
                quantized.to(Q.dtype.element_ty),
            )
            packed += exponent << (j * 8)
        tl.store(S + block.to(tl.int64) * COLS + col, packed)


@triton.jit
def _requant_k_blocks(
    X,
    SW,
    SF_ROWS,
    Q,
    S,
    STARTS,
    ENDS,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
    COLS: tl.constexpr,
    COL_BLOCKS: tl.constexpr,
    FROM_MX: tl.constexpr,
    RECIPROCAL: tl.constexpr = False,
    PERSISTENT: tl.constexpr = False,
):
    # Pools are sized for the worst case; blocks past the last expert's rows are never read. PERSISTENT
    # launches a fixed row grid that strides over the used blocks instead of one program per block.
    last = tl.load(ENDS + NUM_EXPERTS - 1)
    block = tl.program_id(0)
    if PERSISTENT:
        while block * 128 < last:
            _requant_block(
                block,
                X,
                SW,
                SF_ROWS,
                Q,
                S,
                STARTS,
                ENDS,
                NUM_EXPERTS,
                EXPERTS_POW2,
                COLS,
                COL_BLOCKS,
                FROM_MX,
                RECIPROCAL,
            )
            block += tl.num_programs(0)
    elif block * 128 < last:
        _requant_block(
            block,
            X,
            SW,
            SF_ROWS,
            Q,
            S,
            STARTS,
            ENDS,
            NUM_EXPERTS,
            EXPERTS_POW2,
            COLS,
            COL_BLOCKS,
            FROM_MX,
            RECIPROCAL,
        )


def requant_k_blocks(pool, starts, ends, scale_words=None):
    """E4M3 ``[rows, C]`` copy of a pool with ``[rows / 128, C]`` packed UE8M0 scales along rows.

    ``pool`` is BF16, or the backward's row-wise E4M3 pool when ``scale_words`` (its packed
    ``[C / 128, scale rows]`` scale pool, ``scale rows >= rows``) is given; the quantization is then a
    requantization of the exactly dequantized E4M3 values along the row axis. The scale pool may keep
    more rows than the bounded E4M3 pool (the kernel's scale layout is compiled for the worst case);
    only its first ``rows`` rows are read."""
    rows, cols = pool.shape
    if rows % 128 or cols % 128 or not pool.is_contiguous():
        raise ValueError(
            "pools must be contiguous with rows and columns multiples of 128"
        )
    experts = starts.numel()
    scale_rows = rows
    if scale_words is not None:
        scale_rows = scale_words.shape[1]
        if (
            scale_words.dim() != 2
            or scale_words.shape[0] != cols // 128
            or scale_rows < rows
            or not scale_words.is_contiguous()
        ):
            raise ValueError(
                "scale words must be a contiguous [C / 128, >= rows] tensor"
            )
    q = torch.empty(rows, cols, device=pool.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(rows // 128, cols, device=pool.device, dtype=torch.int32)
    col_blocks, num_warps, reciprocal, persistent = _requant_config(cols)
    row_programs = rows // 128
    if persistent:
        sms = torch.cuda.get_device_properties(pool.device).multi_processor_count
        row_programs = min(
            row_programs, max(1, persistent * sms // (cols // (128 * col_blocks)))
        )
    _requant_k_blocks[(row_programs, cols // (128 * col_blocks))](
        pool,
        scale_words if scale_words is not None else pool,
        scale_rows,
        q,
        s,
        starts,
        ends,
        experts,
        triton.next_power_of_2(experts),
        cols,
        col_blocks,
        scale_words is not None,
        reciprocal,
        persistent > 0,
        num_warps=num_warps,
    )
    return q, s


def _requant_config(cols):
    """``MONO_EP_SM100_REQUANT_CONFIG`` = ``col_blocks,num_warps,reciprocal[,persistent]`` (``/`` also separates); default ``1,2,1``
    (bitwise equal to the original ``1,4,0`` and a third faster). ``persistent`` > 0 launches that many programs
    per SM, striding over the used row blocks."""
    value = os.getenv("MONO_EP_SM100_REQUANT_CONFIG", "1,2,1")
    fields = [int(v) for v in value.replace("/", ",").split(",")]
    col_blocks, num_warps, reciprocal = fields[:3]
    persistent = fields[3] if len(fields) > 3 else 0
    if (cols // 128) % col_blocks:
        col_blocks = 1
    return col_blocks, num_warps, bool(reciprocal), persistent


def weight_grad_quantized(a, b, ends, counts, experts):
    """FP32 ``[experts, A cols, B cols]`` weight gradients from ``(E4M3, scales)`` pairs of
    ``requant_k_blocks`` (DeepGEMM k-grouped ``recipe=(1, 1, 32)``, MN-major operands)."""
    import deep_gemm

    out = torch.empty(
        (experts, a[0].shape[1], b[0].shape[1]),
        device=a[0].device,
        dtype=torch.float32,
    )
    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
        a, b, out, None, ends, None, (1, 1, 32), "mn", True
    )
    return zero_empty_experts(out, counts)


@triton.jit
def _zero_empty_experts(OUT, COUNTS, NUMEL: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    if tl.load(COUNTS + expert) != 0:
        return
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + expert.to(tl.int64) * NUMEL + offset, 0.0, offset < NUMEL)


def zero_empty_experts(out, counts):
    """Zero the ``[M, N]`` slabs of experts with no rows."""
    experts, numel = out.shape[0], out[0].numel()
    block = 16384
    _zero_empty_experts[(experts, triton.cdiv(numel, block))](out, counts, numel, block)
    return out


def weight_grad(a_pool, b_pool, ends, counts, experts, starts=None):
    """FP32 ``[experts, A cols, B cols]`` weight gradients, written directly."""
    import deep_gemm

    if dw_precision() == "mxfp8":
        return weight_grad_quantized(
            requant_k_blocks(a_pool, starts, ends),
            requant_k_blocks(b_pool, starts, ends),
            ends,
            counts,
            experts,
        )
    out = torch.empty(
        (experts, a_pool.shape[1], b_pool.shape[1]),
        device=a_pool.device,
        dtype=torch.float32,
    )
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        a_pool, b_pool, out, None, ends, None, use_psum_layout=True
    )
    return zero_empty_experts(out, counts)
