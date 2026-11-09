# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Common training entry point for expert compute and communication scheduling."""

import torch


class MonoEP:
    """SwiGLU expert parallelism with hardware-selected BF16 or FP8 compute.

    Inputs are BF16 activations, global expert indices, FP32 routing scores,
    and local weights in [expert, output, input] layout. Gate/up rows are
    concatenated. The caller owns the process group and parameter sharding.
    NCCL routing remains eager; compile_local compiles expert callbacks only.
    The persistent backend requires a fixed token capacity and recomputation; persistent FP8 currently requires Hopper.
    """

    def __init__(
        self,
        *,
        dtype="bf16",
        compile_local=False,
        retain_activations=True,
        chunks=1,
        backend="pipeline",
        max_tokens=None,
    ):
        if dtype not in ("bf16", "fp8"):
            raise ValueError("MonoEP compute dtype must be bf16 or fp8")
        if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks < 1:
            raise ValueError("MonoEP chunks must be a positive integer")
        if backend not in ("pipeline", "persistent"):
            raise ValueError("MonoEP backend must be pipeline or persistent")
        if backend == "persistent":
            if retain_activations or chunks != 1:
                raise ValueError(
                    "Persistent MonoEP requires recomputation and chunks=1"
                )
            if (
                not isinstance(max_tokens, int)
                or isinstance(max_tokens, bool)
                or max_tokens < 1
            ):
                raise ValueError("Persistent MonoEP requires positive max_tokens")
            if compile_local:
                raise ValueError(
                    "compile_local applies to pipeline callbacks; compile the prepared persistent runtime instead"
                )
        elif max_tokens is not None:
            raise ValueError("max_tokens applies only to the persistent backend")
        self.backend = backend
        self.max_tokens = max_tokens
        self._persistent = {}
        self.dtype = dtype
        self.compile_local = compile_local
        self.retain_activations = retain_activations
        self.chunks = chunks
        self._compute = {}

    def _local_compute(self, device):
        capability = torch.cuda.get_device_capability(device)
        if capability[0] not in (9, 10):
            raise ValueError("MonoEP currently supports SM90 and SM100 GPUs")
        key = (device, capability, self.dtype, self.compile_local)
        if key not in self._compute:
            if self.dtype == "fp8":
                from .fp8_autograd import FP8Experts

                compute = FP8Experts(
                    "blockwise" if capability[0] == 9 else "mxfp8",
                    compile_local=self.compile_local,
                )
            else:
                from .sonic import SonicExperts

                compute = SonicExperts(compile_local=self.compile_local)
            self._compute[key] = compute
        return self._compute[key]

    def prepare_persistent(self, *, group, device, experts, topk, hidden, intermediate):
        """Collectively allocate a reusable runtime before compilation or execution."""
        if self.backend != "persistent":
            raise ValueError("prepare_persistent requires the persistent backend")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Persistent MonoEP requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (group, device, experts, topk, hidden, intermediate)
        if key not in self._persistent:
            from .runtime import PersistentMoE

            with torch.cuda.device(device):
                self._persistent[key] = PersistentMoE(
                    group,
                    experts,
                    self.max_tokens,
                    topk,
                    hidden,
                    intermediate,
                    compute_dtype="blockwise" if self.dtype == "fp8" else "bf16",
                    retain_activations=False,
                    backward_backend="native",
                )
        return self._persistent[key]

    def close(self):
        """Release persistent handles after all graphs using them have finished."""
        for runtime in self._persistent.values():
            runtime.close()
        self._persistent.clear()

    def __call__(self, x, indices, scores, w1, w2, *, group):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            raise ValueError("MonoEP requires CUDA BF16 input activations")
        if self.backend == "persistent":
            import torch.distributed as dist

            if w1.ndim != 3 or w2.ndim != 3 or indices.ndim != 2:
                raise ValueError(
                    "Expected rank-3 expert weights and rank-2 route indices"
                )
            runtime = self.prepare_persistent(
                group=group,
                device=x.device,
                experts=w1.shape[0] * dist.get_world_size(group),
                topk=indices.shape[1],
                hidden=x.shape[-1],
                intermediate=w2.shape[-1],
            )
            return runtime(x, indices, scores, w1, w2)
        from .pipeline import pipelined_moe

        compute = self._local_compute(x.device)
        backward = (
            compute.backward if self.retain_activations or self.dtype == "fp8" else None
        )
        return pipelined_moe(
            x.contiguous(),
            indices,
            scores,
            w1,
            w2,
            group=group,
            local_kernel=compute,
            local_backward=backward,
            local_forward=(
                compute.forward_with_activations if self.retain_activations else None
            ),
            chunks=self.chunks,
            overlap_backward=True,
        )
