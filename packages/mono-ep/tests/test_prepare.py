# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The first-launch rendezvous (``mono_ep.prepare.launch``) and ``MonoEPRuntime.prepare``.

CPU: the guard's call order against a stand-in ``deep_gemm``. GPU: after ``prepare`` a
training step compiles and loads nothing, counted on fresh DeepGEMM / Triton caches in a
subprocess.
"""

import json
import os
import subprocess  # nosec B404
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
from mono_ep import prepare


class _Recorder:
    def __init__(self, monkeypatch, world=2):
        self.calls = []
        self.flag = False
        native = SimpleNamespace(set_mono_ep_prepare_only=self._set)
        monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace(_C=native))
        monkeypatch.setattr(prepare.dist, "get_world_size", lambda group: world)
        monkeypatch.setattr(
            prepare, "host_barrier", lambda group: self.calls.append("barrier")
        )
        monkeypatch.setattr(
            prepare.torch.cuda, "is_current_stream_capturing", lambda: False
        )

    def _set(self, value):
        self.flag = value

    def kernel(self, *args, **kwargs):
        self.calls.append(("prepare" if self.flag else "launch", args, kwargs))
        return "out"


def test_first_launch_per_direction_compiles_then_meets_the_ranks(monkeypatch):
    recorder = _Recorder(monkeypatch)
    runtime = SimpleNamespace(group=None, launched=set())
    out = prepare.launch(runtime, prepare.FORWARD, recorder.kernel, 1, a=2)
    assert out == "out"
    assert recorder.calls == [
        ("prepare", (1,), {"a": 2}),
        "barrier",
        ("launch", (1,), {"a": 2}),
    ]
    assert not recorder.flag
    recorder.calls.clear()
    prepare.launch(runtime, prepare.FORWARD, recorder.kernel, 3)
    assert recorder.calls == [("launch", (3,), {})]
    recorder.calls.clear()
    prepare.launch(runtime, prepare.BACKWARD, recorder.kernel)
    assert [c if c == "barrier" else c[0] for c in recorder.calls] == [
        "prepare",
        "barrier",
        "launch",
    ]


def test_single_rank_and_unpatched_builds_launch_directly(monkeypatch):
    recorder = _Recorder(monkeypatch, world=1)
    runtime = SimpleNamespace(group=None)
    prepare.launch(runtime, prepare.FORWARD, recorder.kernel)
    assert recorder.calls == [("launch", (), {})]
    assert runtime.launched == {prepare.FORWARD}

    recorder = _Recorder(monkeypatch, world=2)
    monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace(_C=SimpleNamespace()))
    runtime = SimpleNamespace(group=None, launched=set())
    prepare.launch(runtime, prepare.FORWARD, recorder.kernel)
    assert recorder.calls == [("launch", (), {})]


def test_prepare_only_is_cleared_when_the_compile_raises(monkeypatch):
    recorder = _Recorder(monkeypatch)

    def failing(*args):
        assert recorder.flag
        raise RuntimeError("compile failed")

    runtime = SimpleNamespace(group=None, launched=set())
    with pytest.raises(RuntimeError, match="compile failed"):
        prepare.launch(runtime, prepare.FORWARD, failing)
    assert not recorder.flag
    assert "barrier" not in recorder.calls
    assert runtime.launched == set()


_STEP = textwrap.dedent(
    """
    import ctypes, json, os, sys
    import torch, torch.distributed as dist

    precision, store = sys.argv[1], sys.argv[2]
    dist.init_process_group(
        "nccl", store=dist.FileStore(store, 1), rank=0, world_size=1,
        device_id=torch.device("cuda", 0),
    )
    from mono_ep import MonoEPRuntime
    from mono_ep.lora import LoraExperts, MergedWeightCache

    E, H, I, K, T, R = 8, 1024, 512, 4, 256, 8

    def entries():
        dirs = (os.path.join(os.environ["DG_JIT_CACHE_DIR"], "cache"), os.environ["TRITON_CACHE_DIR"])
        return [sorted(os.listdir(d)) if os.path.isdir(d) else [] for d in dirs]

    def flush():
        # DeepGEMM prints its load times through C stdio
        ctypes.CDLL(None).fflush(None)
        sys.stdout.flush()

    runtime = MonoEPRuntime(
        group=dist.group.WORLD, num_experts=E, max_tokens_per_rank=T, top_k=K,
        hidden=H, intermediate=I, precision=precision,
    )
    lora = precision.endswith("_lora")
    kwargs = dict(num_tokens=T, lora_rank=R if lora else None)
    runtime.prepare(**kwargs)
    prepared = entries()
    runtime.prepare(**kwargs)
    flush()
    print("STEP")
    flush()
    gen = torch.Generator("cuda").manual_seed(1)
    x = torch.randn(T, H, device="cuda", generator=gen).bfloat16().requires_grad_()
    ids = torch.rand(T, E, device="cuda", generator=gen).topk(K, dim=-1).indices
    scores = torch.rand(T, K, device="cuda", generator=gen).requires_grad_()

    def factor(*shape):
        return (torch.randn(shape, device="cuda", generator=gen) * 0.01).requires_grad_()

    factors = [factor(E, R, H), factor(E, 2 * I, R), factor(E, R, I), factor(E, H, R)]
    if precision == "nvfp4_lora":
        from mono_ep.nvfp4_lora import Nvfp4LoraExperts, RequantCache

        ones = torch.ones(E, device="cuda")
        experts = Nvfp4LoraExperts(
            torch.randint(0, 256, (E, 2 * I, H // 2), device="cuda", dtype=torch.uint8),
            torch.ones(E, 2 * I, H // 16, device="cuda").to(torch.float8_e4m3fn), ones,
            torch.randint(0, 256, (E, H, I // 2), device="cuda", dtype=torch.uint8),
            torch.ones(E, H, I // 16, device="cuda").to(torch.float8_e4m3fn), ones,
            scaling=2.0,
        )
        y = runtime.forward_lora(x, ids, scores, *factors, experts, RequantCache())
        inputs = [x, scores, *factors]
    else:
        w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) / H**0.5).bfloat16()
        w2 = (torch.randn(E, H, I, device="cuda", generator=gen) / I**0.5).bfloat16()
        if lora:
            y = runtime.forward_lora(
                x, ids, scores, *factors, LoraExperts(scaling=2.0),
                MergedWeightCache(), w1=w1, w2=w2,
            )
            inputs = [x, scores, *factors]
        else:
            w1.requires_grad_()
            w2.requires_grad_()
            y = runtime.forward(x, ids, scores, w1, w2)
            inputs = [x, scores, w1, w2]
    grads = torch.autograd.grad(y.float().square().sum(), inputs)
    torch.cuda.synchronize()
    finite = all(bool(g.isfinite().all()) for g in grads)
    flush()
    print(json.dumps({"prepared": prepared, "after": entries(), "finite": finite}))
    flush()
    runtime.destroy()
    dist.destroy_process_group()
    """
)


@pytest.mark.gpu
@pytest.mark.gpu_blackwell
@pytest.mark.parametrize(
    "precision",
    [
        "mxfp8",
        "mxfp8_lora",
        "fp8",
        "fp8_lora",
        pytest.param("bf16", marks=pytest.mark.slow),
        pytest.param("bf16_lora", marks=pytest.mark.slow),
        pytest.param("nvfp4_lora", marks=pytest.mark.slow),
    ],
)
def test_step_after_prepare_compiles_and_loads_nothing(precision, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("the fused kernels need CUDA")
    deep_gemm = pytest.importorskip("deep_gemm")
    from mono_ep import unsupported_reason

    reason = unsupported_reason(
        capability=torch.cuda.get_device_capability(),
        hidden=1024,
        intermediate=512,
        has_bias=False,
        activation="silu",
        precision=precision,
    )
    if reason is not None:
        pytest.skip(reason)
    if not hasattr(deep_gemm._C, "set_mono_ep_prepare_only"):
        pytest.skip("rebuild vendored DeepGEMM with set_mono_ep_prepare_only")
    env = dict(
        os.environ,
        DG_JIT_CACHE_DIR=str(tmp_path / "dg"),
        TRITON_CACHE_DIR=str(tmp_path / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"),
        DG_JIT_PRINT_LOAD_TIME="1",
        CUDA_VISIBLE_DEVICES=str(torch.cuda.current_device()),
    )
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", _STEP, precision, str(tmp_path / "store")],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    out = result.stdout
    before, after = out.split("STEP\n", 1)
    assert "Load time" in before
    assert "Load time" not in after, after
    report = json.loads(after.strip().splitlines()[-1])
    assert report["prepared"][0], "prepare compiled nothing into the fresh caches"
    assert report["after"] == report["prepared"]
    assert report["finite"]
