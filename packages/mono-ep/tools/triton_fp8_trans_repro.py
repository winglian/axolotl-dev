# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Standalone repro for the 8-bit tile transpose miscompile seen in sm90_weight_forms.py.

Each variant loads an FP32 (or BF16) tile, scales it by one per-tile e4m3 scale, casts to
float8_e4m3fn and stores the transposed tile; ``store_first`` also stores the untransposed FP8
tile before the transpose (the pattern of the fused ``w2``/``w2t`` forms). Paths:

* ``fp8_trans``: ``tl.trans`` on the FP8 tile (the original w2t code)
* ``fp8_permute``: ``tl.permute(q, (1, 0))`` on the FP8 tile
* ``fp8_reshape``: ``tl.reshape`` to 3-D, permute, reshape back, on the FP8 tile
* ``fp32_trans``: ``tl.trans`` on the FP32 tile, then scale and cast (the 92c804aad6 workaround)
* ``fp32_reload``: load the tile again through transposed pointers, no transpose in registers
* ``fp8_scatter``: store the untransposed FP8 tile to transposed addresses, no transpose

``int8`` runs a plain int8 tile transpose (no cast) with the same shapes.

Usage: ``python triton_fp8_trans_repro.py [--out results.json] [--dump DIR] [--ptxas-matrix]``.
``--dump`` writes the TTIR/TTGIR/LLIR/PTX of every failing variant (and its passing twin) under
DIR; ``--ptxas-matrix`` recompiles three exemplars under alternative ptxas binaries/options.
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch
import triton
import triton.language as tl

PATHS = (
    "fp8_trans",
    "fp8_permute",
    "fp8_reshape",
    "fp32_trans",
    "fp32_reload",
    "fp8_scatter",
)


@triton.jit
def _e4m3_scale(amax):
    amax = tl.maximum(amax, 1.0e-4)
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, 1), 254)
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def _tile_transpose(
    W,
    Q,
    QT,
    S,
    M: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    STORE_FIRST: tl.constexpr,
    PATH: tl.constexpr,
    SCALED: tl.constexpr,
):
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    rows = (rb * BM + tl.arange(0, BM)).to(tl.int64)
    cols = (cb * BN + tl.arange(0, BN)).to(tl.int64)
    w = tl.load(W + rows[:, None] * N + cols[None, :]).to(tl.float32)
    if SCALED:
        s = _e4m3_scale(tl.max(tl.max(tl.abs(w), 1), 0))
    else:
        s = 1.0
    q = tl.minimum(tl.maximum(w / s, -448.0), 448.0).to(Q.dtype.element_ty)
    if STORE_FIRST:
        tl.store(Q + rows[:, None] * N + cols[None, :], q)
    t_rows = (cb * BN + tl.arange(0, BN)).to(tl.int64)
    t_cols = (rb * BM + tl.arange(0, BM)).to(tl.int64)
    if PATH == 5:
        tl.store(QT + cols[None, :] * M + rows[:, None], q)
    else:
        if PATH == 0:
            qt = tl.trans(q)
        elif PATH == 1:
            qt = tl.permute(q, (1, 0))
        elif PATH == 2:
            qt = tl.reshape(tl.permute(tl.reshape(q, (BM, BN, 1)), (1, 0, 2)), (BN, BM))
        elif PATH == 3:
            wt = tl.trans(w)
            qt = tl.minimum(tl.maximum(wt / s, -448.0), 448.0).to(QT.dtype.element_ty)
        else:
            wt = tl.load(W + t_cols[None, :] * N + t_rows[:, None]).to(tl.float32)
            qt = tl.minimum(tl.maximum(wt / s, -448.0), 448.0).to(QT.dtype.element_ty)
        tl.store(QT + t_rows[:, None] * M + t_cols[None, :], qt)
    if not STORE_FIRST:
        tl.store(Q + rows[:, None] * N + cols[None, :], q)
    tl.store(S + rb * (N // BN) + cb, s)


@triton.jit
def _int8_tile_transpose(
    X,
    Y,
    YT,
    M: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    STORE_FIRST: tl.constexpr,
):
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    rows = (rb * BM + tl.arange(0, BM)).to(tl.int64)
    cols = (cb * BN + tl.arange(0, BN)).to(tl.int64)
    x = tl.load(X + rows[:, None] * N + cols[None, :])
    if STORE_FIRST:
        tl.store(Y + rows[:, None] * N + cols[None, :], x)
    xt = tl.trans(x)
    t_rows = (cb * BN + tl.arange(0, BN)).to(tl.int64)
    t_cols = (rb * BM + tl.arange(0, BM)).to(tl.int64)
    tl.store(YT + t_rows[:, None] * M + t_cols[None, :], xt)
    if not STORE_FIRST:
        tl.store(Y + rows[:, None] * N + cols[None, :], x)


def e4m3_scale_torch(amax):
    amax = amax.clamp_min(1.0e-4)
    bits = (amax / 448.0).view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exponent = exponent.clamp(1, 254)
    return (exponent << 23).view(torch.float32)


def reference(w, bm, bn, scaled):
    m, n = w.shape
    w32 = w.float()
    tiles = w32.view(m // bm, bm, n // bn, bn)
    if scaled:
        amax = tiles.abs().amax(dim=(1, 3))
        s = e4m3_scale_torch(amax)
    else:
        s = torch.ones(m // bm, n // bn, device=w.device)
    scaled_w = (tiles / s[:, None, :, None]).reshape(m, n)
    q = scaled_w.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, s


def displacement_pattern(got, expected, source, max_offsets=6):
    """Where the wrong bytes came from: (drow, dcol) displacements inside the source tile."""
    bad = (got != expected).nonzero()
    if not len(bad):
        return {}
    sample = bad[:: max(1, len(bad) // 512)][:512]
    counts = Counter()
    bm, bn = source.shape[-2], source.shape[-1]
    for i, j in sample.tolist():
        value = got[i, j]
        # expected source element is (j, i); candidates are elements equal to the wrong byte
        matches = (source == value).nonzero()
        for r, c in matches.tolist():
            counts[(r - j, c - i)] += 1
    total = sum(counts.values())
    return {
        "bad_rows_mod_16": Counter((bad[:, 0] % 16).tolist()),
        "bad_cols_mod_16": Counter((bad[:, 1] % 16).tolist()),
        "top_displacements": [
            (k, v / total) for k, v in counts.most_common(max_offsets)
        ],
        "tile_shape": (bm, bn),
    }


def run_variant(
    bm,
    bn,
    num_warps,
    store_first,
    path,
    scaled,
    in_dtype,
    m=512,
    n=1024,
    seed=0,
    diagnose=False,
):
    torch.manual_seed(seed)
    w = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w[:bm, :bn] *= 300.0
    w[bm : 2 * bm, :bn] = 0
    w = w.to(in_dtype)
    q = torch.empty(m, n, device="cuda", dtype=torch.float8_e4m3fn)
    qt = torch.empty(n, m, device="cuda", dtype=torch.float8_e4m3fn)
    s = torch.empty(m // bm, n // bn, device="cuda", dtype=torch.float32)
    kernel = _tile_transpose[(m // bm, n // bn)](
        w,
        q,
        qt,
        s,
        m,
        n,
        bm,
        bn,
        store_first,
        PATHS.index(path),
        scaled,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    torch.cuda.synchronize()
    q_ref, s_ref = reference(w, bm, bn, scaled)
    q_u8, qt_u8 = q.view(torch.uint8), qt.view(torch.uint8)
    qt_vs_q = qt_u8 != q_u8.t()
    result = dict(
        kind="fp8",
        bm=bm,
        bn=bn,
        num_warps=num_warps,
        store_first=store_first,
        path=path,
        scaled=scaled,
        in_dtype=str(in_dtype).split(".")[-1],
        q_ok=bool(torch.equal(q_u8, q_ref.view(torch.uint8))),
        s_ok=bool(torch.equal(s, s_ref)),
        qt_ok=bool(torch.equal(qt_u8, q_ref.view(torch.uint8).t())),
        qt_vs_q_mismatch=int(qt_vs_q.sum()),
        qt_vs_q_mismatch_frac=float(qt_vs_q.float().mean()),
        kernel_name=kernel.name,
        kernel_hash=kernel.hash,
    )
    if diagnose and qt_vs_q.any():
        # third tile along the columns: random data, no zero/outlier block
        got = qt_u8[2 * bn : 3 * bn, :bm].cpu()
        src = q_u8[:bm, 2 * bn : 3 * bn].cpu()
        pattern = displacement_pattern(got, src.t(), src)
        result["pattern"] = {
            k: (sorted(v.items()) if isinstance(v, Counter) else v)
            for k, v in pattern.items()
        }
    return result


def run_int8_variant(
    bm, bn, num_warps, store_first, m=512, n=1024, seed=0, diagnose=False
):
    torch.manual_seed(seed)
    x = torch.randint(-128, 128, (m, n), device="cuda", dtype=torch.int8)
    y = torch.empty_like(x)
    yt = torch.empty(n, m, device="cuda", dtype=torch.int8)
    kernel = _int8_tile_transpose[(m // bm, n // bn)](
        x, y, yt, m, n, bm, bn, store_first, num_warps=num_warps
    )
    torch.cuda.synchronize()
    bad = yt != x.t()
    result = dict(
        kind="int8",
        bm=bm,
        bn=bn,
        num_warps=num_warps,
        store_first=store_first,
        path="int8_trans",
        scaled=False,
        in_dtype="int8",
        q_ok=bool(torch.equal(y, x)),
        s_ok=True,
        qt_ok=bool(not bad.any()),
        qt_vs_q_mismatch=int(bad.sum()),
        qt_vs_q_mismatch_frac=float(bad.float().mean()),
        kernel_name=kernel.name,
        kernel_hash=kernel.hash,
    )
    if diagnose and bad.any():
        got = yt[2 * bn : 3 * bn, :bm].cpu()
        src = x[:bm, 2 * bn : 3 * bn].cpu()
        pattern = displacement_pattern(got, src.t(), src)
        result["pattern"] = {
            k: (sorted(v.items()) if isinstance(v, Counter) else v)
            for k, v in pattern.items()
        }
    return result


def run_spec(spec, diagnose=False):
    spec = dict(spec)
    if spec.get("kind") == "int8":
        return run_int8_variant(
            spec["bm"],
            spec["bn"],
            spec["num_warps"],
            spec["store_first"],
            diagnose=diagnose,
        )
    if isinstance(spec.get("in_dtype"), str):
        spec["in_dtype"] = getattr(torch, spec["in_dtype"])
    return run_variant(
        **{
            k: spec[k]
            for k in (
                "bm",
                "bn",
                "num_warps",
                "store_first",
                "path",
                "scaled",
                "in_dtype",
            )
        },
        diagnose=diagnose,
    )


def variants(quick=False):
    shapes = [(128, 128), (64, 128)] if not quick else [(128, 128)]
    warps = [8, 4] if not quick else [8]
    for (bm, bn), nw, store_first, path in itertools.product(
        shapes, warps, (True, False), PATHS
    ):
        yield dict(
            kind="fp8",
            bm=bm,
            bn=bn,
            num_warps=nw,
            store_first=store_first,
            path=path,
            scaled=True,
            in_dtype="float32",
        )
    if quick:
        return
    for store_first in (True, False):
        # bf16 masters, and the unscaled cast (rules out the per-tile reduction as the trigger)
        yield dict(
            kind="fp8",
            bm=128,
            bn=128,
            num_warps=8,
            store_first=store_first,
            path="fp8_trans",
            scaled=True,
            in_dtype="bfloat16",
        )
        yield dict(
            kind="fp8",
            bm=128,
            bn=128,
            num_warps=8,
            store_first=store_first,
            path="fp8_trans",
            scaled=False,
            in_dtype="float32",
        )
    for (bm, bn), nw, store_first in itertools.product(shapes, warps, (True, False)):
        yield dict(kind="int8", bm=bm, bn=bn, num_warps=nw, store_first=store_first)


EXEMPLARS = [
    dict(
        kind="fp8",
        bm=128,
        bn=128,
        num_warps=8,
        store_first=True,
        path="fp8_trans",
        scaled=True,
        in_dtype="float32",
    ),
    dict(
        kind="fp8",
        bm=128,
        bn=128,
        num_warps=8,
        store_first=False,
        path="fp32_trans",
        scaled=True,
        in_dtype="float32",
    ),
    dict(
        kind="fp8",
        bm=64,
        bn=128,
        num_warps=8,
        store_first=False,
        path="fp8_trans",
        scaled=True,
        in_dtype="float32",
    ),
    dict(kind="int8", bm=128, bn=128, num_warps=8, store_first=True),
]


def label(spec):
    return "{kind}_{bm}x{bn}_w{num_warps}_{path}_{store}_{dtype}".format(
        store="storefirst" if spec["store_first"] else "nostore",
        dtype=spec.get("in_dtype", "int8"),
        **{**dict(path="int8_trans"), **spec},
    )


def run_in_subprocess(spec, env, diagnose=False):
    code = (
        "import json,sys;sys.path.insert(0, {root!r});"
        "from triton_fp8_trans_repro import run_spec;"
        "print(json.dumps(run_spec(json.loads(sys.argv[1]), diagnose={diagnose!r})))"
    ).format(root=str(Path(__file__).resolve().parent), diagnose=diagnose)
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", code, json.dumps(spec)],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1]), result
    except (IndexError, json.JSONDecodeError):
        return None, result


def dump_variant(spec, dump_dir):
    """Recompile one variant with TRITON_KERNEL_DUMP into dump_dir/<label>."""
    target = Path(dump_dir) / label(spec)
    target.mkdir(parents=True, exist_ok=True)
    cache = target / "cache"
    env = dict(
        TRITON_KERNEL_DUMP="1",
        TRITON_DUMP_DIR=str(target / "dump"),
        TRITON_CACHE_DIR=str(cache),
        TRITON_ALWAYS_COMPILE="1",
    )
    parsed, result = run_in_subprocess(spec, env, diagnose=True)
    (target / "run.log").write_text(result.stdout + "\n" + result.stderr)
    shutil.rmtree(cache, ignore_errors=True)
    files = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
    return dict(
        label=label(spec), files=files, returncode=result.returncode, result=parsed
    )


def ptxas_candidates():
    from triton import knobs

    bundled = Path(knobs.nvidia.ptxas.path)
    candidates = {"default": {}}
    blackwell = bundled.with_name("ptxas-blackwell")
    if blackwell.exists():
        candidates["ptxas_blackwell_binary"] = {"TRITON_PTXAS_PATH": str(blackwell)}
    for name, path in (
        ("system_ptxas", shutil.which("ptxas")),
        ("cuda_home_ptxas", "/usr/local/cuda/bin/ptxas"),
    ):
        if path and Path(path).exists() and Path(path).resolve() != bundled.resolve():
            candidates[name] = {"TRITON_PTXAS_PATH": str(path)}
    candidates["opt_level_0"] = {"DISABLE_PTXAS_OPT": "1"}
    candidates["opt_level_1"] = {"PTXAS_OPTIONS": "--opt-level 1"}
    candidates["opt_level_2"] = {"PTXAS_OPTIONS": "--opt-level 2"}
    return candidates


def ptxas_version(env):
    from triton import knobs

    path = env.get("TRITON_PTXAS_PATH", knobs.nvidia.ptxas.path)
    out = subprocess.run(
        [path, "--version"], capture_output=True, text=True, check=False
    )  # nosec B603
    return out.stdout.strip().splitlines()[-2:] if out.stdout else out.stderr[-200:]


def ptxas_matrix(cache_root):
    rows = []
    for name, env in ptxas_candidates().items():
        for spec in EXEMPLARS:
            cache = Path(cache_root) / f"{name}_{label(spec)}"
            parsed, result = run_in_subprocess(
                spec,
                {**env, "TRITON_CACHE_DIR": str(cache), "TRITON_ALWAYS_COMPILE": "1"},
            )
            shutil.rmtree(cache, ignore_errors=True)
            rows.append(
                dict(
                    ptxas=name,
                    env=env,
                    version=ptxas_version(env),
                    variant=label(spec),
                    qt_ok=None if parsed is None else parsed["qt_ok"],
                    mismatch_frac=None
                    if parsed is None
                    else parsed["qt_vs_q_mismatch_frac"],
                    error=None if parsed is not None else result.stderr[-1500:],
                )
            )
            r = rows[-1]
            print(
                f"ptxas={name:<24} {label(spec):<48} "
                f"{'ok' if r['qt_ok'] else ('BAD' if r['qt_ok'] is False else 'ERR')} "
                f"{r['mismatch_frac'] if r['mismatch_frac'] is not None else ''}"
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--dump", default="", help="dump IR of failing variants here")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ptxas-matrix", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    results = [run_spec(spec, diagnose=args.diagnose) for spec in variants(args.quick)]
    info = dict(
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        triton=triton.__version__,
        python=sys.version.split()[0],
        ptxas=ptxas_version({}),
        results=results,
    )
    width = max(len(r["path"]) for r in results)
    for r in results:
        print(
            f"{r['bm']:>3}x{r['bn']:<3} warps={r['num_warps']} "
            f"store_first={int(r['store_first'])} {r['path']:<{width}} "
            f"scaled={int(r['scaled'])} {r['in_dtype']:<8} "
            f"q={'ok' if r['q_ok'] else 'BAD'} s={'ok' if r['s_ok'] else 'BAD'} "
            f"qt={'ok' if r['qt_ok'] else 'BAD'} "
            f"qt!=q.T: {r['qt_vs_q_mismatch_frac']:.3f}"
            + (
                f" pattern={r['pattern']['top_displacements'][:3]}"
                if "pattern" in r
                else ""
            )
        )
    failing = [r for r in results if not r["qt_ok"]]
    print(
        f"triton {triton.__version__} ptxas {info['ptxas']} on {info['device']}: "
        f"{len(failing)} of {len(results)} variants fail"
    )
    if args.dump and failing:
        dumps = []
        for r in failing:
            spec = {
                k: r[k]
                for k in (
                    "kind",
                    "bm",
                    "bn",
                    "num_warps",
                    "store_first",
                    "path",
                    "scaled",
                    "in_dtype",
                )
            }
            dumps.append(dump_variant(spec, args.dump))
            dumps.append(
                dump_variant(
                    {**spec, "store_first": not spec["store_first"]}, args.dump
                )
            )
        info["dumps"] = dumps
    if args.ptxas_matrix:
        info["ptxas_matrix"] = ptxas_matrix(
            Path(args.dump or tempfile.gettempdir()) / "ptxas_cache"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(info, indent=1))
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
