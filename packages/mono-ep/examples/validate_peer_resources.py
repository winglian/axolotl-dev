# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Compile the Qwen peer GEMM specializations with PTXAS resource reporting."""

import argparse
import json

import torch
import torch.distributed as dist
from benchmark_peer_runtime import BF16ForwardSchedule, SeparateActivationSchedule


def main(dtype="fp8"):
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    torch.manual_seed(849)
    tokens, hidden, inner, experts, topk = 16384, 2048, 768, 16, 8
    schedule = BF16ForwardSchedule if dtype == "bf16" else SeparateActivationSchedule
    peer = schedule(
        group=dist.group.WORLD,
        tokens=tokens,
        hidden=hidden,
        inner=inner,
        experts=experts,
        topk=topk,
        slot_combine=True,
        dtype=dtype,
        compile_local=dtype == "bf16",
    )
    peer.sms = 132 if dtype == "bf16" else 16
    x = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    scores = torch.randn(tokens, topk, device="cuda").softmax(-1).requires_grad_()
    ids = torch.randint(experts, (tokens, topk), device="cuda", dtype=torch.int32)
    w1 = (
        torch.randn(experts, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).requires_grad_()
    w2 = (
        torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
    ).requires_grad_()
    for separate in (False, True):
        print(
            json.dumps(
                dict(
                    phase="compile",
                    dtype=dtype,
                    separate_activation=separate,
                    sms=peer.sms,
                )
            ),
            flush=True,
        )
        if dtype == "bf16":
            peer.fused_forward = not separate
        else:
            peer.separate_activation = separate
        peer.warm_forward = peer.warm_backward = False
        for p in (x, scores, w1, w2):
            p.grad = None
        output = peer(x, ids, scores, w1, w2)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in (x, scores, w1, w2)
        )
        print(
            json.dumps(dict(phase="complete", separate_activation=separate)), flush=True
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="fp8")
    main(parser.parse_args().dtype)
