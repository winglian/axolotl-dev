# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("routing", ["masked", "unrouted", "empty"])
def test_sonic_fp8_gradients(compiled, routing, retained):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Validated FP8 Sonic path requires SM100")
    pytest.importorskip("sonicmoe")
    pytest.importorskip("quack")
    from mono_ep.sonic import SonicExperts, sonic_fp8_experts

    torch.manual_seed(73)
    tokens = 0 if routing == "empty" else 37
    x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16).requires_grad_()
    scores, ids = torch.randn(tokens, 8, device="cuda").softmax(-1).topk(4)
    ids[:, 1::2] = -1
    if routing == "unrouted":
        ids[:] = -1
    scores.requires_grad_()
    w1 = (torch.randn(8, 512, 256, device="cuda") / 16).requires_grad_()
    w2 = (torch.randn(8, 256, 256, device="cuda") / 16).requires_grad_()
    inputs = (x, scores, w1, w2)
    expected = (
        x.float() * 0
        + scores.sum() * 0
        + w1.flatten()[:0].sum()
        + w2.flatten()[:0].sum()
    )
    for expert in range(8):
        token, slot = torch.where(ids == expert)
        gate, up = (x[token] @ w1[expert].bfloat16().T).float().chunk(2, -1)
        h = (torch.nn.functional.silu(gate) * up * scores[token, slot, None]).bfloat16()
        expected = expected.index_add(0, token, (h @ w2[expert].bfloat16().T).float())
    dy = torch.randn_like(x)
    reference_grads = torch.autograd.grad(expected.bfloat16(), inputs, dy)
    fn = (
        torch.compile(sonic_fp8_experts, fullgraph=True)
        if compiled
        else sonic_fp8_experts
    )
    if retained:
        compute = SonicExperts(fp8=True, compile_local=compiled)
        actual, state = compute.forward_with_activations(x, ids, scores, w1, w2)
        grads = compute.backward(dy, x, ids, scores, w1, w2, activations=state)
    else:
        actual = fn(x, ids, scores, w1, w2)
        grads = torch.autograd.grad(actual, inputs, dy)
    for value, reference in zip(
        (actual, *grads), (expected, *reference_grads), strict=True
    ):
        error = (value.float() - reference.float()).norm()
        assert error <= 0.15 * reference.float().norm() + 1e-6
    assert torch.count_nonzero(grads[1][ids < 0]) == 0
