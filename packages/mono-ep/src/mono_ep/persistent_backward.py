# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Native single-host persistent BF16 backward and grouped weight gradients."""

import os
import socket

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from . import workspace_bound
from .dw import direct_dw, weight_grad
from .ops import weight_grad_bf16, weight_grad_bf16_gpu_offsets

OWNED_BLOCK_M = 128


def owned_backward_ring_tokens(
    num_sms, env="MONO_EP_BACKWARD_RING_WAVES", default_waves=1
):
    """Ring floor for the owned backward schedule, in waves of CTA clusters (pairs
    on SM100, single CTAs on SM90): each cluster holds one pool block live through
    all four phases, and the forward's interleaved-schedule sizing leaves
    wide-expert shapes with fewer ring blocks than clusters. ``env`` overrides the
    wave count; 0 keeps that sizing. The floor only ever raises the ring."""
    waves = int(os.environ.get(env, str(default_waves)))
    if waves < 0:
        raise ValueError(f"{env} must be >= 0")
    ctas_per_cluster = 1 if torch.cuda.get_device_capability()[0] == 9 else 2
    return waves * (num_sms // ctas_per_cluster) * OWNED_BLOCK_M


class NativePersistentBackward:
    ready_gather = False
    fused_peer_return = True

    def __init__(self, runtime):
        import deep_gemm

        self.arch = torch.cuda.get_device_capability()[0]
        self.native_name = (
            "bf16_mono_ep_backward_sm90" if self.arch == 9 else "bf16_mono_ep_backward"
        )
        self.blockwise = runtime.compute_dtype == "blockwise"
        transport = os.getenv("MONO_EP_SM90_FP8_TRANSPORT")
        if transport not in (None, "0", "1"):
            raise ValueError("MONO_EP_SM90_FP8_TRANSPORT must be 0 or 1")
        self.fp8_transport = self.blockwise and transport == "1"
        combine = os.getenv("MONO_EP_SM90_FP8_COMBINE")
        if combine not in (None, "0", "1"):
            raise ValueError("MONO_EP_SM90_FP8_COMBINE must be 0 or 1")
        self.fp8_combine = self.blockwise and combine == "1"
        if self.fp8_combine and not self.fp8_transport:
            raise ValueError("SM90 FP8 combine requires SM90 FP8 transport")
        if self.blockwise:
            if self.arch != 9:
                raise ValueError("Native blockwise FP8 requires SM90")
            self.native_name = "fp8_mono_ep_backward_sm90"
        if self.arch not in (9, 10) or not hasattr(deep_gemm._C, self.native_name):
            raise RuntimeError("Rebuild vendored DeepGEMM with native MonoEP backward")
        if (
            runtime.compute_dtype not in ("bf16", "blockwise")
            or runtime.retain_activations
        ):
            raise ValueError(
                "Native persistent backward requires recomputing BF16 or Hopper blockwise FP8"
            )
        if deep_gemm.get_mk_alignment_for_contiguous_layout() != 128:
            raise ValueError(
                "Native backward requires 128-row grouped-gradient alignment"
            )
        if runtime.hidden % 256 or runtime.intermediate % 256:
            raise ValueError("Native backward requires H and I divisible by 256")
        self.runtime = runtime
        self.num_sms = runtime.num_sms
        if deep_gemm.get_num_sms() != self.num_sms:
            raise RuntimeError("persistent backward SM count differs from its runtime")
        self.group = runtime.group
        self.rank, self.world = (
            dist.get_rank(self.group),
            dist.get_world_size(self.group),
        )
        hosts = [None] * self.world
        dist.all_gather_object(hosts, socket.gethostname(), group=self.group)
        if len(set(hosts)) != 1:
            raise ValueError("Native persistent backward currently requires one host")
        floor = owned_backward_ring_tokens(runtime.num_sms)
        self.bounded = workspace_bound.mode() == "auto"
        self.buffer = deep_gemm.get_symm_buffer_for_mono_ep(
            self.group,
            runtime.experts,
            runtime.max_tokens,
            runtime.topk,
            runtime.hidden,
            runtime.intermediate,
            mma_type=(
                "fp8xbf16_fp8combine"
                if self.fp8_combine
                else "fp8xbf16"
                if self.fp8_transport
                else "bf16xbf16"
            ),
            num_min_ring_tokens=floor,
            # the owned schedule keeps one block live per cluster, so the floor is all it needs
            num_exact_ring_tokens=self._bounded_ring(floor) if self.bounded else 0,
        )
        self.handles = []
        capacity = self.buffer.num_max_tokens_per_rank
        self.dy, self.dy_pointers = self._allocate(
            (capacity, runtime.hidden),
            torch.float8_e4m3fn if self.fp8_transport else torch.bfloat16,
        )
        self.dy_sf = None
        self.dy_sf_pointers = None
        if self.fp8_transport:
            self.dy_sf, self.dy_sf_pointers = self._allocate(
                (capacity, runtime.hidden // 128), torch.float32
            )
        self.ds_partials, self.ds_pointers = self._allocate(
            (capacity, runtime.topk, 2), torch.float32
        )
        self.experts = runtime.experts // self.world
        rows = self.world * capacity * runtime.topk + self.experts * 128
        self.worst_rows = rows
        self.bound = None
        if self.bounded:
            self.bound = workspace_bound.WorkspaceBound(
                capacity, runtime.topk, self.experts, rows
            )
            rows = self.bound.rows
            if runtime.forward_ring_budget:
                self.bound.forward_budget = (
                    runtime.forward_ring_budget + self.experts * 128
                )
        self.starts = torch.empty(
            self.experts, device=self.dy.device, dtype=torch.int32
        )
        self.counts = torch.empty_like(self.starts)
        self.padded_counts = torch.empty_like(self.starts) if self.blockwise else None
        self.pools = self._build_pools(rows)
        self.finished = None
        self.context_hits = self.context_misses = 0
        self.dw_state = None
        self._mx = None

    def inkernel_dw(self):
        """SM100 BF16: compute dW1/dW2 inside the persistent backward (``MONO_EP_SM100_INKERNEL_DW``)."""
        value = os.getenv("MONO_EP_SM100_INKERNEL_DW", "0")
        if value not in ("0", "1"):
            raise ValueError("MONO_EP_SM100_INKERNEL_DW must be 0 or 1")
        if value == "0" or self.arch != 10 or self.blockwise:
            return False
        if self.experts > 32:
            raise ValueError("In-kernel dW supports at most 32 local experts")
        return True

    def stored_context_usable(self):
        """The SM100 BF16 backward can read a stored forward z (not with the MXFP8 backward or in-kernel dW)."""
        return (
            self.arch == 10
            and not self.blockwise
            and self.precision() == "bf16"
            and not self.inkernel_dw()
            and os.getenv("MONO_EP_SM100_PHASE_TRACE") is None
            and self.runtime.buffer.num_max_tokens_per_rank
            == self.buffer.num_max_tokens_per_rank
        )

    def _take_context(self, x, indices, scores, w1, w2):
        from .stored_context import context_key

        stored = self.runtime._stored
        if stored is None:
            return None
        key, stored.key = stored.key, None
        if key is None or key != context_key(x, indices, scores, w1, w2):
            if key is not None:
                self.context_misses += 1
            return None
        self.context_hits += 1
        return stored.buffer

    def precision(self):
        """``MONO_EP_SM100_BACKWARD_PRECISION``: ``bf16`` (default) or ``mxfp8`` (SM100 BF16 runtimes)."""
        value = os.getenv("MONO_EP_SM100_BACKWARD_PRECISION", "bf16")
        if value not in ("bf16", "mxfp8"):
            raise ValueError("MONO_EP_SM100_BACKWARD_PRECISION must be bf16 or mxfp8")
        if value == "mxfp8" and (self.arch != 10 or self.blockwise):
            raise ValueError(
                "The MXFP8 persistent backward requires SM100 and a BF16 runtime"
            )
        return value

    def _mx_resources(self):
        """E4M3 transport buffers and pools, allocated on the first MXFP8 call (collective on the EP group)."""
        if self._mx is not None:
            return self._mx
        import types

        import deep_gemm

        runtime = self.runtime
        buffer = deep_gemm.get_symm_buffer_for_mono_ep(
            self.group,
            runtime.experts,
            runtime.max_tokens,
            runtime.topk,
            runtime.hidden,
            runtime.intermediate,
            mma_type="fp8xbf16",
        )
        capacity = buffer.num_max_tokens_per_rank
        if capacity != self.buffer.num_max_tokens_per_rank:
            raise RuntimeError(
                "MXFP8 transport buffer capacity differs from the BF16 one"
            )
        dy, dy_pointers = self._allocate(
            (capacity, runtime.hidden), torch.float8_e4m3fn
        )
        dy_sf, dy_sf_pointers = self._allocate(
            (capacity, runtime.hidden // 128), torch.int32
        )
        # E4M3 pools follow the (possibly bounded) BF16 pools; the scale pools keep the worst-case
        # row count, which the kernel's UTCCP-ordered scale layout is compiled for
        rows = self.pools[0].shape[0]
        device = self.dy.device
        pools = [
            torch.zeros(rows, width, device=device, dtype=torch.float8_e4m3fn)
            for width in (runtime.hidden, runtime.hidden, 2 * runtime.intermediate)
        ] + [
            torch.zeros(width // 128, self.worst_rows, device=device, dtype=torch.int32)
            for width in (runtime.hidden, runtime.hidden, 2 * runtime.intermediate)
        ]
        self._mx = types.SimpleNamespace(
            buffer=buffer,
            dy=dy,
            dy_pointers=dy_pointers,
            dy_sf=dy_sf,
            dy_sf_pointers=dy_sf_pointers,
            pools=pools,
        )
        return self._mx

    def _backward_mxfp8(self, dy, x, indices, scores, w1, w2):
        import deep_gemm

        from .token_quantization import quantize_persistent_mxfp8_into
        from .weight_cache import mxfp8_backward_weight_forms

        mx = self._mx_resources()
        tokens = x.shape[0]
        quantize_persistent_mxfp8_into(
            x.contiguous(), mx.buffer.x[:tokens], mx.buffer.x_sf[:tokens]
        )
        quantize_persistent_mxfp8_into(
            dy.contiguous(), mx.dy[:tokens], mx.dy_sf[:tokens]
        )
        mx.buffer.topk_idx[:tokens].copy_(indices)
        mx.buffer.topk_weights[:tokens].copy_(scores)
        forms = mxfp8_backward_weight_forms(w1, w2)
        dx = torch.empty_like(x)
        ds = torch.empty_like(scores, dtype=torch.float32)
        deep_gemm._C.mxfp8_mono_ep_backward(
            dx,
            ds,
            *[f[0] for f in forms],
            [f[1] for f in forms],
            self.pools[:4] + mx.pools,
            self.starts,
            self.counts,
            mx.buffer.buffer,
            mx.buffer.handle.buffer_ptrs,
            mx.dy,
            mx.dy_pointers,
            mx.dy_sf,
            mx.dy_sf_pointers,
            self.ds_partials,
            self.ds_pointers,
            self.rank,
            mx.buffer.num_max_tokens_per_rank,
            mx.buffer.l1_acts.shape[0],
            self.runtime.topk,
        )
        return dx, ds

    def _bounded_ring(self, floor):
        """Ring tokens under the workspace bound: the owned floor, four waves for the SM90 BF16 backward (one wave
        costs it 3-4 % at Qwen3-30B-A3B on H200, four waves 0.5-2.4 %; blockwise FP8 is flat at one wave)."""
        if (
            self.arch == 9
            and not self.blockwise
            and "MONO_EP_BACKWARD_RING_WAVES" not in os.environ
        ):
            return 4 * floor
        return floor

    def _build_pools(self, rows, scratch=None):
        """The x/dy/h/dz pools for ``rows`` routed rows; blockwise adds ring-sized BF16 scratch, the FP8 row
        caches and the packed column-major wgrad operands with their scales."""
        runtime = self.runtime
        device = self.dy.device
        widths = (
            runtime.hidden,
            runtime.hidden,
            runtime.intermediate,
            2 * runtime.intermediate,
        )
        if scratch is None:
            scratch_rows = self.buffer.l1_acts.shape[0] if self.blockwise else rows
            scratch = [
                torch.zeros(scratch_rows, width, device=device, dtype=torch.bfloat16)
                for width in widths
            ]
        pools = list(scratch)
        if not self.blockwise:
            return pools
        pools.extend(
            torch.empty(rows, width, device=device, dtype=dtype)
            for width, dtype in (
                (runtime.hidden, torch.float8_e4m3fn),
                (runtime.hidden, torch.float8_e4m3fn),
                (runtime.hidden // 128, torch.float32),
                (runtime.hidden // 128, torch.float32),
                (runtime.intermediate * 2, torch.float8_e4m3fn),
                (runtime.intermediate * 2 // 128, torch.float32),
            )
        )
        pools.extend(
            torch.empty(rows * width, device=device, dtype=torch.float8_e4m3fn)
            for width in widths
        )
        pools.extend(
            torch.empty_strided(
                (width, rows // 128), (1, width), device=device, dtype=torch.float32
            )
            for width in widths
        )
        return pools

    def _fit_pools(self, w1):
        """Bounded workspace: grow the pools before a backward whose routed rows exceed them."""
        if self.bound is None:
            return
        rows = self.bound.target(self.bound.required_rows(w1))
        if rows is None:
            return
        # Blockwise keeps its ring-sized BF16 scratch; every row-sized pool is rebuilt
        scratch = self.pools[:4] if self.blockwise else None
        del self.pools
        self.pools = self._build_pools(rows, scratch)
        if self._mx is not None:
            self._mx.pools[:3] = [
                torch.zeros(rows, p.shape[1], device=p.device, dtype=p.dtype)
                for p in self._mx.pools[:3]
            ]

    def _allocate(self, shape, dtype):
        allocator = torch if self.world == 1 else symm_mem
        local = allocator.empty(*shape, dtype=dtype, device="cuda")
        if self.world == 1:
            self.handles.append(local)
            return local, [local.data_ptr()]
        handle = symm_mem.rendezvous(local, group=self.group)
        self.handles.append((local, handle))
        return local, [
            handle.get_buffer(r, shape, dtype).data_ptr() for r in range(self.world)
        ]

    def backward(self, dy, x, indices, scores, w1, w2):
        import deep_gemm
        from deep_gemm.mega import _interleave_weights

        if (
            deep_gemm.get_num_sms() != self.num_sms
            or self.runtime.buffer.num_sms != self.num_sms
            or self.buffer.num_sms != self.num_sms
        ):
            raise RuntimeError(
                "persistent backward SM count changed after workspace allocation"
            )
        if x.dtype != torch.bfloat16 or dy.dtype != torch.bfloat16:
            raise ValueError("Native backward requires BF16 activations and gradients")
        stream = torch.cuda.current_stream(x.device)
        if self.finished is not None:
            stream.wait_event(self.finished)
        self._fit_pools(w1)
        if self.arch == 10 and not self.blockwise and self.precision() == "mxfp8":
            if self.inkernel_dw():
                # Both reuse the kernel's shared-expert operand slots and need different BLOCK_K
                raise ValueError(
                    "MONO_EP_SM100_INKERNEL_DW=1 does not combine with "
                    "MONO_EP_SM100_BACKWARD_PRECISION=mxfp8"
                )
            dx, ds = self._backward_mxfp8(dy, x, indices, scores, w1, w2)
            return self._finish_bf16_sm100(dx, ds, scores, stream)
        tokens = x.shape[0]
        from .sm90_prep import enabled as fused_prep

        fused = self.fp8_transport and fused_prep()
        if fused:
            from .sm90_prep import prepare_tokens

            prepare_tokens(
                x.contiguous(),
                self.buffer.x[:tokens],
                self.buffer.x_sf[:tokens],
                indices,
                self.buffer.topk_idx[:tokens],
                scores,
                self.buffer.topk_weights[:tokens],
                dy.contiguous(),
                self.dy[:tokens],
                self.dy_sf[:tokens],
            )
        elif self.fp8_transport:
            from .token_quantization import quantize_tokens_into

            quantize_tokens_into(
                x.contiguous(),
                self.buffer.x[:tokens],
                self.buffer.x_sf[:tokens],
                block_cols=128,
            )
            quantize_tokens_into(
                dy.contiguous(),
                self.dy[:tokens],
                self.dy_sf[:tokens],
                block_cols=128,
            )
        else:
            self.buffer.x[:tokens].copy_(x)
            self.dy[:tokens].copy_(dy)
        if not fused:
            self.buffer.topk_idx[:tokens].copy_(indices)
            self.buffer.topk_weights[:tokens].copy_(scores)
        scale_args = ()
        cache = "0"
        if self.blockwise:
            from .sm90_weight_forms import cache_mode

            cache = cache_mode()
        if cache != "0":
            from .sm90_weight_forms import BACKWARD_FORMS, cached_forms

            prepared = cached_forms(w1, w2, BACKWARD_FORMS, cache)
            interleaved, w2t, w1t = [p[0] for p in prepared]
            scale_args = ([p[1] for p in prepared],)
        elif self.blockwise:
            from .quantization import quantize_fp8
            from .weight_quantization import quantize_transposed_weights_128x128

            if os.getenv("MONO_EP_SM90_FP8_FUSED_WEIGHT_QUANT", "1") != "0":
                from .block_weight_quantization import quantize_weights_128x128

                gate_up = quantize_weights_128x128(w1, interleave_gate_up=True)
            else:
                gate_up = quantize_fp8(
                    _interleave_weights(w1), block_rows=128, block_cols=128
                )
            prepared = [
                gate_up,
                quantize_transposed_weights_128x128(w2),
                quantize_transposed_weights_128x128(w1),
            ]
            interleaved, w2t, w1t = [p[0] for p in prepared]
            scale_args = ([p[1] for p in prepared],)
        else:
            from .weight_cache import weight_layouts

            interleaved, w2t, w1t = weight_layouts(
                w1, w2, "w1_interleaved", "w2t", "w1t"
            )
        dx = torch.empty_like(x)
        ds = torch.empty_like(scores, dtype=torch.float32)
        inkernel_dw = self.inkernel_dw()
        dw_args = ()
        native_name = self.native_name
        stored_z = (
            self._take_context(x, indices, scores, w1, w2)
            if self.arch == 10 and not self.blockwise
            else None
        )
        if stored_z is not None:
            dw_args = (stored_z,)
            native_name = "bf16_mono_ep_backward_ctx"
        elif inkernel_dw:
            if self.dw_state is None:
                # Readiness and claim counters; the kernel leaves them zeroed for the next launch
                self.dw_state = torch.zeros(
                    2 * self.experts, device=x.device, dtype=torch.int32
                )
            hidden, inner = self.runtime.hidden, self.runtime.intermediate
            dw1 = torch.empty(
                (self.experts, 2 * inner, hidden), device=x.device, dtype=torch.float32
            )
            dw2 = torch.empty(
                (self.experts, hidden, inner), device=x.device, dtype=torch.float32
            )
            dw_args = (dw1, dw2, self.dw_state)
            native_name = "bf16_mono_ep_backward_dw"
        getattr(deep_gemm._C, native_name)(
            dx,
            ds,
            interleaved,
            w2t,
            w1t,
            *scale_args,
            self.pools,
            self.starts,
            self.counts,
            self.buffer.buffer,
            self.buffer.handle.buffer_ptrs,
            self.dy,
            self.dy_pointers,
            self.ds_partials,
            self.ds_pointers,
            self.rank,
            self.buffer.num_max_tokens_per_rank,
            self.buffer.l1_acts.shape[0],
            self.runtime.topk,
            *(
                (
                    self.padded_counts,
                    self.dy_sf,
                    self.dy_sf_pointers,
                    self.fp8_combine,
                )
                if self.fp8_transport
                else (self.padded_counts,)
                if self.blockwise
                else ()
            ),
            *dw_args,
        )
        x_pool, dy_pool, h_pool, dz_pool = self.pools[:4]
        if not (fused and self.blockwise) and not inkernel_dw:
            ends = self.starts + self.counts
        if inkernel_dw and stored_z is None:
            dw1, dw2 = dw_args[:2]
        elif self.blockwise:
            from .hopper_wgrad import weight_grad_fp8_hopper_packed

            qx, qdy, qh, qdz, sx, sdy, sh, sdz = self.pools[10:]
            capacity = self.pools[4].shape[0]
            dw1 = weight_grad_fp8_hopper_packed(
                qdz, sdz, qx, sx, self.padded_counts, capacity
            )
            dw2 = weight_grad_fp8_hopper_packed(
                qdy, sdy, qh, sh, self.padded_counts, capacity
            )
        elif self.arch == 9:
            # The GEMM's K hint may not exceed the (bounded) pool rows
            expected_k = min(
                self.buffer.num_max_tokens_per_rank * self.world * self.runtime.topk,
                x_pool.shape[0],
            )
            dw1 = weight_grad_bf16_gpu_offsets(
                dz_pool, x_pool, self.starts, ends, expected_k
            )
            dw2 = weight_grad_bf16_gpu_offsets(
                dy_pool, h_pool, self.starts, ends, expected_k
            )
        elif direct_dw():
            dw1 = weight_grad(dz_pool, x_pool, ends, self.counts, self.experts)
            dw2 = weight_grad(dy_pool, h_pool, ends, self.counts, self.experts)
        else:
            dw1 = weight_grad_bf16(dz_pool, x_pool, ends, None, self.experts, True)
            dw2 = weight_grad_bf16(dy_pool, h_pool, ends, None, self.experts, True)
        if self.runtime._stored is not None:
            self.runtime._stored.record_rows(self.counts)
        self.finished = torch.cuda.Event()
        self.finished.record(stream)
        return dx, ds.to(scores.dtype), dw1, dw2

    def _finish_bf16_sm100(self, dx, ds, scores, stream):
        """SM100 BF16 weight gradients from the BF16 pools (the MXFP8 backward dequantizes x and dy into them)."""
        x_pool, dy_pool, h_pool, dz_pool = self.pools[:4]
        ends = self.starts + self.counts
        if direct_dw():
            dw1 = weight_grad(dz_pool, x_pool, ends, self.counts, self.experts)
            dw2 = weight_grad(dy_pool, h_pool, ends, self.counts, self.experts)
        else:
            dw1 = weight_grad_bf16(dz_pool, x_pool, ends, None, self.experts, True)
            dw2 = weight_grad_bf16(dy_pool, h_pool, ends, None, self.experts, True)
        if self.runtime._stored is not None:
            self.runtime._stored.record_rows(self.counts)
        self.finished = torch.cuda.Event()
        self.finished.record(stream)
        return dx, ds.to(scores.dtype), dw1, dw2
