# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The smallest SM100 BF16 forward ring (``MONO_EP_FORWARD_RING_ROWS=1024``: 1,920 ring tokens, eight
240-row blocks at the Qwen3-30B-A3B shape on one rank) runs ring-gated and matches the worst-case ring
bitwise (y and dx). It used to hang: an L2 task two ring generations ahead read the parity-only L2
readiness mask as ready. Runs in a subprocess under a timeout so a hang fails instead of stalling."""

import os
import subprocess  # nosec B404
import sys

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]

TIMEOUT_S = 600


def _worker():
    import torch.distributed as dist

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    import deep_gemm
    from mono_ep.runtime import PersistentMoE

    deep_gemm.set_num_sms(torch.cuda.get_device_properties(0).multi_processor_count)
    tokens, hidden, inner, experts, topk = 16384, 2048, 768, 128, 8
    runtimes = {}
    for mode in ("max", "auto"):
        os.environ["MONO_EP_WORKSPACE_BOUND"] = mode
        runtimes[mode] = PersistentMoE(
            dist.group.WORLD,
            experts,
            tokens,
            topk,
            hidden,
            inner,
            fuse_activation=True,
            backward_backend="native",
        )
    assert runtimes["auto"].buffer.l1_acts.shape[0] == 1920
    torch.manual_seed(0)
    w1 = (
        torch.randn(experts, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
    ).requires_grad_()
    for case in ("balanced", "skewed"):
        if case == "skewed":
            logits = torch.randn(tokens, experts, device="cuda")
            logits[:, : experts // 8] += 2
            ids = logits.topk(topk, -1).indices
        else:
            ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk]
        ids = ids.contiguous()
        x = torch.randn(
            tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        scores = torch.rand(tokens, topk, device="cuda")
        scores = (scores / scores.sum(-1, keepdim=True)).requires_grad_()
        dy = torch.randn_like(x)
        outs = {}
        for mode, runtime in runtimes.items():
            x.grad = None
            y = runtime(x, ids, scores, w1, w2)
            y.backward(dy)
            outs[mode] = (y.detach().clone(), x.grad.clone())
        torch.cuda.synchronize()
        assert torch.equal(outs["auto"][0], outs["max"][0]), f"{case}: y differs"
        assert torch.equal(outs["auto"][1], outs["max"][1]), f"{case}: dx differs"
        assert runtimes["auto"]._peer_backward.bound.forward_over_budget > 0
    print("FORWARD_RING_OK", flush=True)
    dist.destroy_process_group()
    os._exit(0)


def test_smallest_forward_ring_completes_and_matches_worst_case_ring():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the ring-gated forward is SM100 only")
    pytest.importorskip("deep_gemm")
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=os.environ.get("MONO_EP_TEST_PORT", "29517"),
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        MONO_EP_FORWARD_RING_ROWS="1024",
    )
    try:
        result = subprocess.run(  # nosec B603
            [sys.executable, __file__],
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"the 1,920-token forward ring did not finish in {TIMEOUT_S} s")
    assert "FORWARD_RING_OK" in result.stdout, (
        result.stdout[-4000:] + result.stderr[-4000:]
    )


if __name__ == "__main__":
    _worker()
