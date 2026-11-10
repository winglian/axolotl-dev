# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Check the SM100 BF16 persistent forward's dispatch-pull variants on two GPUs.

Runs the runtime forward with the base pull (``MONO_EP_SM100_FORWARD_OPTIONS=0``),
with ``MONO_EP_SM100_FORWARD_PULL_BYTES`` set to two rows and with the
lookahead pull (``MONO_EP_SM100_FORWARD_OPTIONS=1``), on balanced and on
skewed routing (all tokens to rank 0's experts, so its ring wraps), and checks
that every variant is bitwise identical to the base and matches an FP32 reference.
"""

import json
import os

import torch
import torch.distributed as dist

HIDDEN, INNER, EXPERTS, TOPK = 2048, 768, 16, 8
PULL_ENV = "MONO_EP_SM100_FORWARD_PULL_BYTES"
OPTIONS_ENV = "MONO_EP_SM100_FORWARD_OPTIONS"
VARIANTS = {
    "base": {OPTIONS_ENV: "0"},
    "two_row": {OPTIONS_ENV: "0", PULL_ENV: str(4 * HIDDEN)},
    "lookahead": {OPTIONS_ENV: "1"},
}


def reference(x, ids, scores, w1, w2):
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        z = (x[rows].float() @ w1[expert].float().T).bfloat16().float()
        gate, up = z.chunk(2, -1)
        h = (torch.nn.functional.silu(gate) * up * scores[rows, slots, None]).bfloat16()
        out.index_add_(0, rows, (h.float() @ w2[expert].float().T).bfloat16().float())
    return out


def main():
    from mono_ep.runtime import PersistentMoE

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    name = torch.cuda.get_device_name()
    assert "B200" in name, name
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    tokens = 4096
    runtime = PersistentMoE(
        dist.group.WORLD,
        EXPERTS,
        tokens,
        TOPK,
        HIDDEN,
        INNER,
        compute_dtype="bf16",
        backward_backend="nccl",
    )
    torch.manual_seed(77)
    w1_all = (
        torch.randn(EXPERTS, 2 * INNER, HIDDEN, device="cuda") / HIDDEN**0.5
    ).bfloat16()
    w2_all = (
        torch.randn(EXPERTS, HIDDEN, INNER, device="cuda") / INNER**0.5
    ).bfloat16()
    local = EXPERTS // world
    w1 = w1_all[rank * local : (rank + 1) * local].contiguous()
    w2 = w2_all[rank * local : (rank + 1) * local].contiguous()
    results = []
    for routing in ("balanced", "skewed"):
        torch.manual_seed(100 + rank)
        count = tokens - 37 * rank
        x = torch.randn(count, HIDDEN, device="cuda", dtype=torch.bfloat16)
        ids = torch.rand(count, EXPERTS, device="cuda").argsort(-1)[:, :TOPK]
        if routing == "skewed":
            ids = torch.rand(count, local, device="cuda").argsort(-1)[:, :TOPK]
        ids = ids.contiguous()
        ids[::19, -1] = -1
        scores = torch.rand(count, TOPK, device="cuda").softmax(-1)
        outputs = {}
        for label, env in VARIANTS.items():
            for key in (PULL_ENV, OPTIONS_ENV):
                os.environ.pop(key, None)
            os.environ.update(env)
            dist.barrier()
            y = runtime(x, ids, scores, w1, w2)
            torch.cuda.synchronize()
            outputs[label] = y.clone()
        for key in (PULL_ENV, OPTIONS_ENV):
            os.environ.pop(key, None)
        expected = reference(x, ids.clamp_min(0), scores * (ids >= 0), w1_all, w2_all)
        rows = int((ids >= 0).sum())
        received = torch.tensor([rows], device="cuda")
        dist.all_reduce(received)
        for label, y in outputs.items():
            rel = (
                (y.float() - expected).norm() / expected.norm().clamp_min(1e-12)
            ).item()
            record = dict(
                rank=rank,
                routing=routing,
                variant=label,
                tokens=count,
                bitwise_equal_to_base=bool(torch.equal(y, outputs["base"])),
                rel_error_vs_fp32_reference=rel,
                finite=bool(torch.isfinite(y.float()).all()),
                ring_tokens=int(runtime.buffer.l1_acts.shape[0]),
                routed_rows_world=int(received),
            )
            results.append(record)
            print("FORWARD_PULL " + json.dumps(record), flush=True)
            assert (
                record["bitwise_equal_to_base"] and record["finite"] and rel < 0.02
            ), record
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
