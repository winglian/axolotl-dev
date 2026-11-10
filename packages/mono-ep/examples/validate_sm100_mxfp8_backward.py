# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""FP32-reference validation of the SM100 persistent backward, BF16 vs MXFP8.

Runs under ``torchrun`` on one or two B200s. For balanced and skewed routing it calls
``NativePersistentBackward.backward`` with ``MONO_EP_SM100_BACKWARD_PRECISION`` set to
``bf16`` and ``mxfp8`` and compares dx, dscore, dW1, dW2 and the pooled h / dz rows that feed
dW against one FP32 reference computed from the original BF16 activations and FP32 masters
(BF16 rounding at z, dh, dz, h and the partial dx as the kernel stores them). The MXFP8 x/dy
pool rows are also checked bitwise against the dequantized host quantization of the source
rows, so the transport is exact. Errors: relative L2, relative L1 (sum |d| / sum |ref|, the
metric of mixture-of-kittens' ``check_correctness``) and max |d|.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist

HIDDEN, INNER, TOPK = 2048, 768, 8
PRECISION_ENV = "MONO_EP_SM100_BACKWARD_PRECISION"


def stats(actual, expected):
    a, e = actual.float(), expected.float()
    d = a - e
    return dict(
        rel_l2=float(d.norm() / e.norm().clamp_min(1e-30)),
        rel_l1=float(d.abs().sum() / e.abs().sum().clamp_min(1e-30)),
        max_abs=float(d.abs().max()) if d.numel() else 0.0,
        ref_max_abs=float(e.abs().max()) if e.numel() else 0.0,
    )


def dequantized(rows):
    from mono_ep.token_quantization import prepare_persistent_mxfp8

    q, s = prepare_persistent_mxfp8(rows.contiguous())
    exp = s.view(torch.uint8).view(rows.shape[0], -1).float()
    scale = torch.exp2(exp - 127.0).repeat_interleave(32, dim=1)
    return (q.float() * scale).bfloat16()


def row_hash(rows, probe):
    return rows[:, : probe.shape[0]].double() @ probe


def reference(x, dy, ids, scores, w1, w2, rank, local_experts):
    """FP32 reference; returns dx, dscore, local dW1/dW2 and per (token, slot) h / dz of local experts."""
    w1b, w2b = w1.bfloat16().float(), w2.bfloat16().float()
    dx = torch.zeros_like(x, dtype=torch.float32)
    ds = torch.zeros_like(scores)
    dw1 = torch.zeros(local_experts, *w1.shape[1:], device=x.device)
    dw2 = torch.zeros(local_experts, *w2.shape[1:], device=x.device)
    h_ref, dz_ref = {}, {}
    for expert in range(w1.shape[0]):
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        xr, dyr = x[rows].float(), dy[rows].float()
        z = (xr @ w1b[expert].T).bfloat16().float()
        gate, up = z.chunk(2, dim=-1)
        dh = (dyr @ w2b[expert]).bfloat16().float()
        sigmoid = gate.sigmoid()
        silu = gate * sigmoid
        s = scores[rows, slots, None]
        weighted = dh * s
        dz = torch.cat(
            (weighted * up * (sigmoid + silu * (1 - sigmoid)), weighted * silu), -1
        ).bfloat16()
        dx.index_add_(0, rows, (dz.float() @ w1b[expert]).bfloat16().float())
        ds[rows, slots] = (dh * silu * up).sum(-1)
        h = (silu * up * s).bfloat16()
        local = expert - rank * local_experts
        if 0 <= local < local_experts:
            dw1[local] = dz.float().T @ xr
            dw2[local] = dyr.T @ h.float()
            h_ref[local] = (rows, h)
            dz_ref[local] = (rows, dz)
    return dx, ds, dw1, dw2, h_ref, dz_ref


def run(args):
    from mono_ep.runtime import PersistentMoE

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    device = torch.cuda.get_device_name()
    if "B200" not in device:
        raise RuntimeError(f"expected B200, got {device}")
    experts = 16 * world
    tokens = args.tokens
    runtime = PersistentMoE(
        dist.group.WORLD,
        experts,
        tokens,
        TOPK,
        HIDDEN,
        INNER,
        compute_dtype="bf16",
        retain_activations=False,
        backward_backend="native",
    )
    backend = runtime._peer_backward
    torch.manual_seed(1234)
    w1_all = torch.randn(experts, 2 * INNER, HIDDEN, device="cuda") / HIDDEN**0.5
    w2_all = torch.randn(experts, HIDDEN, INNER, device="cuda") / INNER**0.5
    w1 = w1_all[rank * 16 : (rank + 1) * 16].contiguous()
    w2 = w2_all[rank * 16 : (rank + 1) * 16].contiguous()
    probe = torch.randn(16, device="cuda", dtype=torch.float64)
    records = []
    for routing in args.routings.split(","):
        torch.manual_seed(97 + rank)
        x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x)
        ids = torch.rand(tokens, experts, device="cuda").argsort(-1)[:, :TOPK]
        if routing == "skewed":
            # Every token to the first rank's experts 0..7: wraps rank 0's ring
            ids[:] = torch.arange(TOPK, device="cuda")
        ids = ids.contiguous()
        ids[::17, -1] = -1
        scores = torch.rand(tokens, TOPK, device="cuda").softmax(-1)
        gathered = []
        for tensor in (x, dy, ids, scores):
            peers = [torch.empty_like(tensor) for _ in range(world)]
            dist.all_gather(peers, tensor)
            gathered.append(torch.cat(peers))
        source_x, source_dy, source_ids, source_scores = gathered
        # dx / dscore from this rank's tokens; dW and the pooled h / dz from every rank's tokens
        expected = reference(
            source_x, source_dy, source_ids, source_scores, w1_all, w2_all, rank, 16
        )
        if world > 1:
            local = reference(x, dy, ids, scores, w1_all, w2_all, rank, 16)
            expected = (local[0], local[1], *expected[2:])
        deq_x, deq_dy = dequantized(source_x), dequantized(source_dy)
        for precision in args.precisions.split(","):
            os.environ[PRECISION_ENV] = precision
            outputs = []
            for _ in range(args.repeats):
                dist.barrier()
                result = backend.backward(dy, x, ids, scores, w1, w2)
                torch.cuda.synchronize()
                outputs.append([t.clone() for t in result])
            dx, ds, dw1, dw2 = outputs[0]
            errors = dict(
                dx=stats(dx, expected[0]),
                dscore=stats(ds, expected[1]),
                dw1=stats(dw1, expected[2]),
                dw2=stats(dw2, expected[3]),
            )
            # Pooled rows: map each pool row back to its source token by hashing the BF16 x pool row
            x_pool, dy_pool, h_pool, dz_pool = backend.pools[:4]
            src_x = deq_x if precision == "mxfp8" else source_x
            src_dy = deq_dy if precision == "mxfp8" else source_dy
            keys = row_hash(src_x, probe)
            order = keys.argsort()
            sorted_keys = keys[order]
            h_actual, h_expected, dz_actual, dz_expected = [], [], [], []
            transport_exact = True
            for local, (start, count) in enumerate(
                zip(backend.starts.tolist(), backend.counts.tolist(), strict=True)
            ):
                if not count:
                    continue
                xp = x_pool[start : start + count]
                position = torch.searchsorted(sorted_keys, row_hash(xp, probe))
                position = position.clamp_max(sorted_keys.numel() - 1)
                source_rows = order[position]
                transport_exact &= bool(torch.equal(xp, src_x[source_rows]))
                transport_exact &= bool(
                    torch.equal(dy_pool[start : start + count], src_dy[source_rows])
                )
                rows_ref, h_ref = expected[4][local]
                _, dz_ref = expected[5][local]
                lookup = torch.full(
                    (world * tokens,), -1, device="cuda", dtype=torch.long
                )
                lookup[rows_ref] = torch.arange(rows_ref.numel(), device="cuda")
                index = lookup[source_rows]
                assert bool((index >= 0).all()), "pool row without a routed source"
                h_actual.append(h_pool[start : start + count])
                h_expected.append(h_ref[index])
                dz_actual.append(dz_pool[start : start + count])
                dz_expected.append(dz_ref[index])
            if h_actual:
                errors["h"] = stats(torch.cat(h_actual), torch.cat(h_expected))
                errors["dz"] = stats(torch.cat(dz_actual), torch.cat(dz_expected))
            counts = backend.counts
            used_pool_rows = int((((counts + 127) // 128) * 128).sum())
            ring_tokens = (
                backend._mx.buffer if precision == "mxfp8" else backend.buffer
            ).l1_acts.shape[0]
            record = dict(
                rank=rank,
                world=world,
                device=device,
                routing=routing,
                precision=precision,
                tokens=tokens,
                errors=errors,
                transport_exact=transport_exact,
                dx_reproducible=all(torch.equal(o[0], dx) for o in outputs),
                finite=all(bool(torch.isfinite(t.float()).all()) for t in outputs[0]),
                routed_rows=int(counts.sum()),
                used_pool_rows=used_pool_rows,
                ring_tokens=ring_tokens,
                ring_wrapped=used_pool_rows > ring_tokens,
            )
            records.append(record)
            print("MXVAL " + json.dumps(record), flush=True)
    os.environ.pop(PRECISION_ENV, None)
    runtime.close()
    dist.destroy_process_group()
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--routings", default="balanced,skewed")
    parser.add_argument("--precisions", default="bf16,mxfp8")
    parser.add_argument("--repeats", type=int, default=2)
    run(parser.parse_args())
