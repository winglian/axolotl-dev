# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""SonicMoE local compute adapter for expert-parallel transport layers."""

import torch


def _load_sonic(fp8=False):
    if fp8:
        from sonicmoe.enums import ActivationType

        from ._vendor.sonic_fp8 import moe_general_routing_inputs

        return moe_general_routing_inputs, ActivationType.SWIGLU
    from transformers.integrations.sonicmoe import load_sonicmoe_kernel

    kernel = load_sonicmoe_kernel()
    return kernel.moe_general_routing_inputs, kernel.activation_type_enum.SWIGLU


def sonic_experts(x, indices, scores, w1, w2, *, fp8=False):
    """Evaluate local experts with -1 marking slots owned by another rank.

    Weights use [expert, output, input] with concatenated gate/up rows.
    Transport, collective ordering, and chunk scheduling belong to the caller.
    FP8 uses experimental per-tensor E4M3 compute with BF16 activations.
    """
    moe_general_routing_inputs, activation_type = _load_sonic(fp8)

    options = {"is_inference_mode_enabled": not torch.is_grad_enabled()}
    if fp8:
        w1 = w1.unflatten(1, (2, -1)).transpose(1, 2).flatten(1, 2)
        options["use_fp8"] = True
    if x.dtype != torch.bfloat16 or scores.dtype != torch.float32:
        raise ValueError("SonicMoE adapter requires BF16 inputs and FP32 scores")
    if x.shape[0] == 0:
        # Empty slices keep all gradient edges without reducing entire weights.
        return x + scores.sum() * 0 + w1.flatten()[:0].sum() + w2.flatten()[:0].sum()
    tokens = x.shape[0]
    padding = -tokens % 128
    if padding:
        # The pinned Sonic/Quack weight-gradient path faults on small unaligned batches.
        x = torch.nn.functional.pad(x, (0, 0, 0, padding))
        indices = torch.nn.functional.pad(indices, (0, 0, 0, padding))
        scores = torch.nn.functional.pad(scores, (0, 0, 0, padding))
    token_ids = torch.arange(x.shape[0], device=x.device, dtype=torch.int32)
    token_ids = token_ids.repeat_interleave(indices.shape[1])
    local_ids = indices.clamp_min(0).flatten().int()
    local_scores = scores.masked_fill(indices < 0, 0).flatten()
    return moe_general_routing_inputs(
        x,
        local_scores,
        token_ids,
        local_ids,
        w1.to(x.dtype).permute(1, 2, 0),
        None,
        w2.to(x.dtype).permute(1, 2, 0),
        None,
        w1.shape[0],
        None,
        activation_type,
        concat_layout=not fp8,
        **options,
    )[0][:tokens]


def sonic_fp8_experts(x, indices, scores, w1, w2):
    """Experimental per-tensor FP8 SonicMoE compute; validated on SM100."""
    return sonic_experts(x, indices, scores, w1, w2, fp8=True)


class SonicExperts:
    """Local Sonic compute with optional activation retention for the EP pipeline."""

    def __init__(self, *, fp8=False, compile_local=False):
        _load_sonic(fp8)
        kernel = sonic_fp8_experts if fp8 else sonic_experts
        self.kernel = (
            torch.compile(kernel, fullgraph=True, dynamic=False)
            if compile_local
            else kernel
        )

    def __call__(self, x, indices, scores, w1, w2):
        return self.kernel(x, indices, scores, w1, w2)

    def forward_with_activations(self, x, indices, scores, w1, w2):
        with torch.enable_grad():
            inputs = tuple(t.detach().requires_grad_(True) for t in (x, scores, w1, w2))
            y = self.kernel(inputs[0], indices, inputs[1], inputs[2], inputs[3])
        return y.detach(), (y, *inputs)

    def backward(self, dy, x, indices, scores, w1, w2, *, activations):
        y, *inputs = activations
        return torch.autograd.grad(y, inputs, dy)
