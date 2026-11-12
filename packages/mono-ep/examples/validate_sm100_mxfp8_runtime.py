# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Validate the SM100 ``precision="mxfp8"`` runtime (MXFP8 forward, fused MXFP8 backward) on 8 GPUs.

At one EP8 layer shape (hidden / intermediate / experts of ``--shape``, ``--tokens`` per rank so the
FP32 reference fits), balanced and skewed routing (every token to rank 0's experts: its rings wrap):

* forward + backward through autograd, against FP32 autograd over all ranks' routed inputs
  (y, dx, dscore, dW1, dW2 relative L2), with and without ``MONO_EP_SM100_DW_PRECISION=mxfp8``;
* a second identical step (dx reproducible, dscore / dW are atomics-ordered);
* dx against the BF16 runtime with ``MONO_EP_SM100_BACKWARD_PRECISION=mxfp8`` on the same inputs
  (the same backward kernel and operands: bitwise);
* the forward's cached MXFP8 weight forms against a direct quantization (bitwise).

Prints ``MXRT_JSON <record>`` per (routing, dW precision) on rank 0.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from mok_vs_mono_ep import SHAPES


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def reference(x, ids, scores, w1, w2, dy, local):
    """FP32 autograd of the layer over all ranks' tokens; weight gradients for the ``local`` experts."""
    x, scores = (t.detach().float().requires_grad_() for t in (x, scores))
    lw1 = w1[local].float().requires_grad_()
    lw2 = w2[local].float().requires_grad_()
    # one index_add over all experts' rows (a per-expert index_add chain keeps an FP32 output per
    # expert alive for autograd); non-local weights are cast per expert and need no gradient
    all_rows, parts = [], []
    for expert in range(w1.shape[0]):
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        if local.start <= expert < local.stop:
            a, b = lw1[expert - local.start], lw2[expert - local.start]
        else:
            a, b = w1[expert].float(), w2[expert].float()
        gate, up = (x[rows] @ a.T).chunk(2, -1)
        h = torch.nn.functional.silu(gate) * up * scores[rows, slots, None]
        all_rows.append(rows)
        parts.append(h @ b.T)
    out = torch.zeros(x.shape[0], w2.shape[1], device=x.device).index_add(
        0, torch.cat(all_rows), torch.cat(parts)
    )
    out.backward(dy.float())
    # a rank whose experts receive no rows (skewed routing) has zero weight gradients
    zeros = torch.zeros_like
    return (
        out.detach(),
        x.grad,
        scores.grad,
        lw1.grad if lw1.grad is not None else zeros(lw1),
        lw2.grad if lw2.grad is not None else zeros(lw2),
    )


def gather(tensor):
    peers = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(peers, tensor.contiguous())
    return torch.cat(peers)


def step(runtime, x, ids, scores, w1, w2, dy):
    xg, sg = x.clone().requires_grad_(), scores.clone().requires_grad_()
    w1g, w2g = w1.clone().requires_grad_(), w2.clone().requires_grad_()
    dist.barrier()
    y = runtime(xg, ids, sg, w1g, w2g)
    y.backward(dy)
    torch.cuda.synchronize()
    return [t.detach().clone() for t in (y, xg.grad, sg.grad, w1g.grad, w2g.grad)]


def main(args):
    import deep_gemm
    from mono_ep import weight_cache
    from mono_ep.runtime import PersistentMoE
    from mono_ep.token_quantization import prepare_persistent_mxfp8

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    name = torch.cuda.get_device_name()
    assert torch.cuda.get_device_capability()[0] == 10, name
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    shape = SHAPES[args.shape]
    hidden, inner, experts = shape["hidden"], shape["intermediate"], shape["experts"]
    topk, tokens, local = 8, args.tokens, experts // world
    deep_gemm.set_num_sms(torch.cuda.get_device_properties(rank).multi_processor_count)
    for key in ("MONO_EP_SM100_BACKWARD_PRECISION", "MONO_EP_SM100_DW_PRECISION"):
        os.environ.pop(key, None)
    runtimes = {
        dtype: PersistentMoE(
            dist.group.WORLD,
            experts,
            tokens,
            topk,
            hidden,
            inner,
            compute_dtype=dtype,
            fuse_activation=True,
            backward_backend="native",
        )
        for dtype in ("mxfp8", "bf16")
    }
    assert runtimes["mxfp8"]._peer_backward.precision() == "mxfp8"
    torch.manual_seed(77)
    w1_all = (
        torch.randn(experts, 2 * inner, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    w2_all = (
        torch.randn(experts, hidden, inner, device="cuda") / inner**0.5
    ).bfloat16()
    w1 = w1_all[rank * local : (rank + 1) * local].contiguous()
    w2 = w2_all[rank * local : (rank + 1) * local].contiguous()
    forms = weight_cache.mxfp8_forward_weight_forms(w1, w2)
    direct = (
        prepare_persistent_mxfp8(w1, weights=True, interleave=True),
        prepare_persistent_mxfp8(w2, weights=True),
    )
    forms_bitwise = all(
        torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
        and torch.equal(a[1], b[1])
        and a[1].stride() == b[1].stride()
        for a, b in zip(forms, direct, strict=True)
    )
    for routing in ("balanced", "skewed"):
        torch.manual_seed(100 + rank)
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :topk]
        if routing == "skewed":
            ids = torch.rand(tokens, local, device="cuda").argsort(-1)[:, :topk]
        ids = ids.contiguous()
        ids[::19, -1] = -1
        scores = torch.rand(tokens, topk, device="cuda").softmax(-1)
        dy = (
            torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
            * hidden**-0.5
        )
        mask = ids >= 0
        local_experts = slice(rank * local, (rank + 1) * local)
        ry, rx, rs, rw1, rw2 = reference(
            gather(x),
            gather(ids).clamp_min(0),
            gather(scores * mask),
            w1_all,
            w2_all,
            gather(dy),
            local_experts,
        )
        sl = slice(rank * tokens, (rank + 1) * tokens)
        expected = dict(
            y=ry[sl],
            dx=rx[sl],
            dscore=rs[sl] * mask,
            dw1=rw1,
            dw2=rw2,
        )
        histogram = torch.bincount(ids[mask], minlength=experts)
        dist.all_reduce(histogram)
        mine = histogram[rank * local : (rank + 1) * local]
        used_pool_rows = int((((mine + 127) // 128) * 128).sum().item())
        ring = int(
            runtimes["mxfp8"]._peer_backward._mx_resources().buffer.l1_acts.shape[0]
        )
        for dw in ("bf16", "mxfp8"):
            os.environ["MONO_EP_SM100_DW_PRECISION"] = dw
            first = step(runtimes["mxfp8"], x, ids, scores, w1, w2, dy)
            second = step(runtimes["mxfp8"], x, ids, scores, w1, w2, dy)
            os.environ["MONO_EP_SM100_BACKWARD_PRECISION"] = "mxfp8"
            bf16_runtime = step(runtimes["bf16"], x, ids, scores, w1, w2, dy)
            os.environ.pop("MONO_EP_SM100_BACKWARD_PRECISION")
            got = dict(zip(("y", "dx", "dscore", "dw1", "dw2"), first, strict=True))
            got["dscore"] = got["dscore"] * mask
            stats = torch.tensor(
                [rel(got[k], expected[k]) for k in expected]
                + [
                    float(not torch.equal(first[1], second[1])),
                    float(not torch.equal(first[1], bf16_runtime[1])),
                    rel(first[3], bf16_runtime[3]),
                    rel(first[3], second[3]),
                    float(not all(torch.isfinite(t.float()).all() for t in first)),
                    float(used_pool_rows > ring),
                ],
                device="cuda",
            )
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            values = stats.tolist()
            record = dict(
                shape=args.shape,
                device=name,
                world=world,
                tokens_per_rank=tokens,
                routing=routing,
                dw_precision=dw,
                rel_l2_vs_fp32_max_rank=dict(zip(expected, values[:5], strict=True)),
                dx_reproducible=values[5] == 0.0,
                dx_bitwise_vs_bf16_runtime_mxfp8_backward=values[6] == 0.0,
                dw1_rel_vs_bf16_runtime=values[7],
                dw1_rel_rerun=values[8],
                finite=values[9] == 0.0,
                ring_wrapped_any_rank=values[10] == 1.0,
                forward_forms_bitwise_vs_direct=forms_bitwise,
            )
            if rank == 0:
                print("MXRT_JSON " + json.dumps(record), flush=True)
            os.environ.pop("MONO_EP_SM100_DW_PRECISION")
    for runtime in runtimes.values():
        runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=("30b", "235b", "beam"), required=True)
    parser.add_argument("--tokens", type=int, default=2048, help="tokens per rank")
    main(parser.parse_args())
