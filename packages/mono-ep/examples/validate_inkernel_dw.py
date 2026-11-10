# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Parity of the SM100 in-kernel dW option against the grouped-GEMM dW path.

Runs the native persistent backward with ``MONO_EP_SM100_INKERNEL_DW`` off and
on in one process for several routings. Per case and rank: dx bitwise against
the opt-out build, dscore against it (and opt-out against itself, since the
dscore epilogue accumulates with shared-memory atomics), and dW1/dW2 of both
paths against an FP64 reference computed from the pools the opt-out launch
captured (so the opt-in error is anchored by the opt-out error). Pool row order
inside an expert is not reproducible across launches (dispatch slots come from
atomics), so pools are not compared; dW is a sum over the expert's rows.

Cases: balanced, skewed (one local expert with zero rows on every rank, tails
not a multiple of 128), wrap (every token to the first rank's experts so the
pool wraps the dispatch ring), poisoned (x/dy pools filled with NaN before each
launch, so the padded tails hold non-finite stale rows; both paths zero those
tails before the dW GEMMs, so the opt-out dW must be finite too), repeat (two opt-in launches in a
row: the readiness counters must be back at zero).
"""

import json
import os

import torch
import torch.distributed as dist

HIDDEN, INNER, TOPK, LOCAL = 2048, 768, 8, 16


def _stats(actual, expected):
    diff = (actual.double() - expected).abs()
    return dict(
        max_abs=float(diff.max()) if diff.numel() else 0.0,
        rel_norm=float(diff.norm() / expected.norm().clamp_min(1e-30))
        if diff.numel()
        else 0.0,
    )


def _routing(case, tokens, experts, world, rank, generator):
    if case == "wrap":
        local = torch.arange(LOCAL, device="cuda")
        ids = local[
            torch.rand(tokens, LOCAL, device="cuda", generator=generator).argsort(-1)[
                :, :TOPK
            ]
        ]
    else:
        logits = torch.rand(tokens, experts, device="cuda", generator=generator)
        if case == "skewed":
            # Expert e of every rank gets weight ~exp(-e/4); local expert 3 never gets rows
            bias = torch.arange(experts, device="cuda") % LOCAL
            logits = logits.log() - bias / 4.0
            logits[:, bias == 3] = -float("inf")
        ids = logits.argsort(-1, descending=True)[:, :TOPK]
    scores = torch.rand(tokens, TOPK, device="cuda", generator=generator).softmax(-1)
    return ids.contiguous(), scores


def run():
    from mono_ep.runtime import PersistentMoE

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    device = torch.cuda.get_device_name()
    if "B200" not in device:
        raise RuntimeError(f"Expected B200, got {device}")
    tokens = int(os.environ.get("INKERNEL_DW_TOKENS", "4096"))
    experts = LOCAL * world
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
    generator = torch.Generator(device="cuda").manual_seed(1009 + rank)
    w1 = (
        torch.randn(LOCAL, 2 * INNER, HIDDEN, device="cuda", generator=generator)
        / HIDDEN**0.5
    )
    w2 = (
        torch.randn(LOCAL, HIDDEN, INNER, device="cuda", generator=generator)
        / INNER**0.5
    )
    ring = backend.buffer.l1_acts.shape[0]
    failures = []
    anchors = {"dw1": float("nan"), "dw2": float("nan")}
    cases = os.environ.get(
        "INKERNEL_DW_CASES", "balanced,skewed,wrap,poisoned,repeat"
    ).split(",")
    for case in cases:
        routing = "balanced" if case in ("poisoned", "repeat") else case
        x = torch.randn(
            tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        dy = torch.randn(
            tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        ids, scores = _routing(routing, tokens, experts, world, rank, generator)

        def call(on, case=case, x=x, dy=dy, ids=ids, scores=scores):
            os.environ["MONO_EP_SM100_INKERNEL_DW"] = "1" if on else "0"
            if case == "poisoned":
                backend.pools[0].fill_(float("nan"))
                backend.pools[1].fill_(float("nan"))
            dist.barrier()
            out = [o.clone() for o in backend.backward(dy, x, ids, scores, w1, w2)]
            torch.cuda.synchronize()
            return out

        off = call(False)
        pools_off = [p.clone() for p in backend.pools[:4]]
        starts, counts = backend.starts.tolist(), backend.counts.tolist()
        off_again = call(False)
        on = call(True)
        on_second = call(True) if case == "repeat" else None
        ref1 = torch.zeros(LOCAL, 2 * INNER, HIDDEN, dtype=torch.float64, device="cuda")
        ref2 = torch.zeros(LOCAL, HIDDEN, INNER, dtype=torch.float64, device="cuda")
        for e, (s, c) in enumerate(zip(starts, counts, strict=True)):
            if not c:
                continue
            xp, yp, hp, zp = [p[s : s + c].double() for p in pools_off]
            ref1[e] = zp.T @ xp
            ref2[e] = yp.T @ hp
        padded = [((c + 127) // 128) * 128 for c in counts]
        record = dict(
            case=case,
            rank=rank,
            world=world,
            tokens=tokens,
            counts=counts,
            zero_row_experts=[e for e, c in enumerate(counts) if c == 0],
            tail_experts=[e for e, c in enumerate(counts) if c % 128],
            used_pool_rows=sum(padded),
            ring_tokens=ring,
            wraps_ring=sum(padded) > ring,
            dx_bitwise_on_vs_off=bool(torch.equal(on[0], off[0])),
            dx_bitwise_off_vs_off=bool(torch.equal(off_again[0], off[0])),
            dscore_bitwise_on_vs_off=bool(torch.equal(on[1], off[1])),
            dscore_bitwise_off_vs_off=bool(torch.equal(off_again[1], off[1])),
            dscore_on_vs_off=_stats(on[1], off[1].double()),
            dscore_off_vs_off=_stats(off_again[1], off[1].double()),
            finite_off=all(bool(torch.isfinite(t).all()) for t in off),
            finite_on=all(bool(torch.isfinite(t).all()) for t in on),
            zero_row_dw_zero=all(
                bool((on[2][e] == 0).all()) and bool((on[3][e] == 0).all())
                for e, c in enumerate(counts)
                if c == 0
            ),
            dw1_off_vs_fp64=_stats(off[2], ref1),
            dw1_on_vs_fp64=_stats(on[2], ref1),
            dw1_on_vs_off=_stats(on[2], off[2].double()),
            dw1_bitwise_on_vs_off=bool(torch.equal(on[2], off[2])),
            dw2_off_vs_fp64=_stats(off[3], ref2),
            dw2_on_vs_fp64=_stats(on[3], ref2),
            dw2_on_vs_off=_stats(on[3], off[3].double()),
            dw2_bitwise_on_vs_off=bool(torch.equal(on[3], off[3])),
        )
        if on_second is not None:
            record["repeat_dx_bitwise"] = bool(torch.equal(on_second[0], on[0]))
            record["repeat_dw1_vs_first"] = _stats(on_second[2], on[2].double())
            record["repeat_dw2_vs_first"] = _stats(on_second[3], on[3].double())
        problems = []
        if not record["dx_bitwise_on_vs_off"]:
            problems.append("dx not bitwise")
        if not record["finite_on"]:
            problems.append("non-finite opt-in output")
        if not record["zero_row_dw_zero"]:
            problems.append("zero-row expert dW not zero")
        for name in ("dw1", "dw2"):
            on_err = record[f"{name}_on_vs_fp64"]["rel_norm"]
            off_err = record[f"{name}_off_vs_fp64"]["rel_norm"]
            if case == "poisoned" and off_err != off_err:
                problems.append(
                    f"{name} opt-out error is non-finite with stale pool tails"
                )
            # Anchor on the last finite opt-out error of a case where this rank has rows
            if off_err == off_err and off_err > 0 and case != "poisoned":
                anchors[name] = off_err
            elif sum(counts):
                off_err = anchors[name]
            record[f"{name}_anchor_rel_norm"] = off_err
            if not on_err <= 1.5 * off_err + 1e-8:
                problems.append(
                    f"{name} error {on_err:.3e} outside 1.5x the opt-out envelope {off_err:.3e}"
                )
        if record["dscore_on_vs_off"]["rel_norm"] > 1e-5:
            problems.append("dscore differs beyond atomic-order noise")
        if on_second is not None and not record["repeat_dx_bitwise"]:
            problems.append("repeat dx not bitwise")
        if case == "wrap" and rank == 0 and not record["wraps_ring"]:
            problems.append("wrap case did not wrap the ring")
        record["problems"] = problems
        failures += problems
        print("INKERNEL_DW_JSON " + json.dumps(record), flush=True)
    os.environ.pop("MONO_EP_SM100_INKERNEL_DW", None)
    runtime.close()
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"rank {rank}: {failures}")


if __name__ == "__main__":
    run()
