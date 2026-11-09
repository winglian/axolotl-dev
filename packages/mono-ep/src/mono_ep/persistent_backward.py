# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Native single-host persistent BF16 backward and grouped weight gradients."""

import os
import socket

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .ops import weight_grad_bf16, weight_grad_bf16_gpu_offsets


class NativePersistentBackward:
    ready_gather = False
    fused_peer_return = True

    def __init__(self, runtime):
        import deep_gemm

        self.arch = torch.cuda.get_device_capability()[0]
        self.native_name = (
            "bf16_mono_ep_backward_sm90"
            if self.arch == 9
            else "bf16_mono_ep_backward"
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
            raise RuntimeError(
                "Rebuild vendored DeepGEMM with native MonoEP backward"
            )
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
        scratch_rows = self.buffer.l1_acts.shape[0] if self.blockwise else rows
        self.pools = [
            torch.zeros(
                scratch_rows, width, device=self.dy.device, dtype=torch.bfloat16
            )
            for width in (
                runtime.hidden,
                runtime.hidden,
                runtime.intermediate,
                2 * runtime.intermediate,
            )
        ]
        if self.blockwise:
            self.pools.extend(
                torch.empty(rows, width, device=self.dy.device, dtype=dtype)
                for width, dtype in (
                    (runtime.hidden, torch.float8_e4m3fn),
                    (runtime.hidden, torch.float8_e4m3fn),
                    (runtime.hidden // 128, torch.float32),
                    (runtime.hidden // 128, torch.float32),
                    (runtime.intermediate * 2, torch.float8_e4m3fn),
                    (runtime.intermediate * 2 // 128, torch.float32),
                )
            )
        self.starts = torch.empty(
            self.experts, device=self.dy.device, dtype=torch.int32
        )
        self.counts = torch.empty_like(self.starts)
        self.padded_counts = None
        if self.blockwise:
            widths = (
                runtime.hidden,
                runtime.hidden,
                runtime.intermediate,
                2 * runtime.intermediate,
            )
            self.pools.extend(
                torch.empty(
                    rows * width, device=self.dy.device, dtype=torch.float8_e4m3fn
                )
                for width in widths
            )
            self.pools.extend(
                torch.empty_strided(
                    (width, rows // 128),
                    (1, width),
                    device=self.dy.device,
                    dtype=torch.float32,
                )
                for width in widths
            )
            self.padded_counts = torch.empty_like(self.starts)
        self.finished = None

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
        tokens = x.shape[0]
        if self.fp8_transport:
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
        self.buffer.topk_idx[:tokens].copy_(indices)
        self.buffer.topk_weights[:tokens].copy_(scores)
        scale_args = ()
        if self.blockwise:
            from .quantization import quantize_fp8
            from .weight_quantization import quantize_transposed_weights_128x128

            prepared = [
                quantize_fp8(_interleave_weights(w1), block_rows=128, block_cols=128),
                quantize_transposed_weights_128x128(w2),
                quantize_transposed_weights_128x128(w1),
            ]
            interleaved, w2t, w1t = [p[0] for p in prepared]
            scale_args = ([p[1] for p in prepared],)
        else:
            w1 = w1.bfloat16()
            w2t = w2.bfloat16().transpose(1, 2).contiguous()
            w1t = w1.transpose(1, 2).contiguous()
            interleaved = _interleave_weights(w1)
        dx = torch.empty_like(x)
        ds = torch.empty_like(scores, dtype=torch.float32)
        getattr(deep_gemm._C, self.native_name)(
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
        )
        x_pool, dy_pool, h_pool, dz_pool = self.pools[:4]
        ends = self.starts + self.counts
        if self.blockwise:
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
            expected_k = (
                self.buffer.num_max_tokens_per_rank * self.world * self.runtime.topk
            )
            dw1 = weight_grad_bf16_gpu_offsets(
                dz_pool, x_pool, self.starts, ends, expected_k
            )
            dw2 = weight_grad_bf16_gpu_offsets(
                dy_pool, h_pool, self.starts, ends, expected_k
            )
        else:
            dw1 = weight_grad_bf16(dz_pool, x_pool, ends, None, self.experts, True)
            dw2 = weight_grad_bf16(dy_pool, h_pool, ends, None, self.experts, True)
        self.finished = torch.cuda.Event()
        self.finished.record(stream)
        return dx, ds.to(scores.dtype), dw1, dw2
