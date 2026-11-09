"""CPU tests for the fused ``mono_ep`` expert-parallel backend seam."""

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.expert_parallel import (
    ExpertParallelArgs,
    ExpertParallelPlugin,
    experts_fn,
    plugin as plugin_mod,
)

PACKAGE_API = (
    Path(__file__).resolve().parents[3] / "packages/mono-ep/src/mono_ep/api.py"
)


def _qwen3_model(hidden=512, intermediate=1024, hidden_act="silu", num_experts=4):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    config = Qwen3MoeConfig(
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        num_experts=num_experts,
        num_experts_per_tok=2,
        hidden_act=hidden_act,
    )
    model = torch.nn.Module()
    model.config = config
    model.mlp = Qwen3MoeSparseMoeBlock(config)
    return model


def _cfg(**kw):
    defaults = dict(
        expert_parallel_size=2,
        expert_parallel_backend="mono_ep",
        expert_parallel_mono_ep_precision="auto",
        expert_parallel_fallback_on_unsupported=True,
        expert_parallel_save_dispatch=False,
        expert_parallel_token_capacity=None,
        expert_parallel_dispatch_chunks=1,
        experts_implementation=None,
        adapter=None,
        dp_shard_size=None,
        dp_replicate_size=None,
        context_parallel_size=None,
        sequence_len=1000,
        micro_batch_size=2,
        eval_batch_size=None,
        activation_checkpointing={"enabled": True},
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class _FakeRuntime:
    forward_op = "mono_ep::persistent_forward"
    instances: list["_FakeRuntime"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.destroyed = False
        self.precision = "bf16"
        _FakeRuntime.instances.append(self)

    @staticmethod
    def token_alignment():
        return 128

    def forward(self, x, topk_idx, topk_weights, w1, w2):
        self.calls.append((x, topk_idx, topk_weights, w1, w2))
        return x * 2

    def destroy(self):
        self.destroyed = True


@pytest.fixture
def mono_ep_package(monkeypatch):
    """The real package API (pure Python) with a fake runtime class, installed as
    ``mono_ep`` so the plugin's imports and ``find_spec`` resolve to it."""
    if not PACKAGE_API.exists():
        pytest.skip("packages/mono-ep is not in this checkout")
    spec = importlib.util.spec_from_file_location("_mono_ep_api", PACKAGE_API)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    module = types.ModuleType("mono_ep")
    module.__spec__ = importlib.machinery.ModuleSpec("mono_ep", None)
    for name in (
        "DEFAULT_RESERVED_SMS",
        "check_nccl_env",
        "nccl_env",
        "resolve_precision",
        "unsupported_reason",
    ):
        setattr(module, name, getattr(api, name))
    module.MonoEPRuntime = _FakeRuntime
    monkeypatch.setitem(sys.modules, "mono_ep", module)
    _FakeRuntime.instances = []
    return module


@pytest.fixture
def no_package(monkeypatch):
    monkeypatch.setattr(plugin_mod, "find_spec", lambda name: None)


@pytest.fixture
def cuda_capability(monkeypatch):
    def set_capability(capability):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: capability is not None)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)

    set_capability((9, 0))
    return set_capability


@pytest.fixture(autouse=True)
def _restore_ep_state():
    from axolotl.integrations.expert_parallel.torch_dispatch import set_ep_group
    from axolotl.monkeypatch.selective_checkpointing import clear_registered_saves

    yield
    experts_fn.set_backend(None)
    experts_fn.set_local_implementation("grouped_mm")
    experts_fn.set_mono_ep_runtime(None)
    experts_fn.set_valid_token_mask(None)
    experts_fn.set_token_capacity(None)
    set_ep_group(None)
    clear_registered_saves()


class TestFusedForward:
    @staticmethod
    def _experts():
        experts = _qwen3_model(num_experts=4).mlp.experts
        experts.num_experts_global = 4
        experts.num_local_experts = 2
        return experts

    def test_bypasses_local_implementation(self, monkeypatch):
        runtime = _FakeRuntime()
        experts_fn.set_mono_ep_runtime(runtime)

        def forbidden(*args, **kwargs):
            raise AssertionError("the fused backend must not run the local path")

        monkeypatch.setattr(experts_fn, "_run_local", forbidden)
        monkeypatch.setattr(experts_fn, "resolve_local_implementation", forbidden)
        experts = self._experts()
        x = torch.randn(3, 512)
        ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)
        weights = torch.rand(3, 2, dtype=torch.bfloat16)
        y = experts_fn._ep_forward(experts, x, ids, weights, backend="mono_ep")

        assert y.dtype == torch.float32  # cast back after the BF16 kernel
        (rx, ridx, rw, w1, w2) = runtime.calls[0]
        assert rx.dtype == torch.bfloat16
        assert ridx.dtype == torch.int64 and ridx.tolist() == ids.tolist()
        assert rw.dtype == torch.float32
        assert w1 is experts.gate_up_proj and w2 is experts.down_proj
        torch.testing.assert_close(y, (x.bfloat16() * 2).float())

    def test_padding_sentinel_and_capacity_apply_first(self):
        runtime = _FakeRuntime()
        experts_fn.set_mono_ep_runtime(runtime)
        experts_fn.set_valid_token_mask(torch.tensor([True, False, True, True]))
        experts_fn.set_token_capacity(2)
        ids = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 3]])
        weights = torch.tensor([[0.6, 0.4], [0.9, 0.1], [0.7, 0.3], [0.5, 0.5]])
        experts_fn._ep_forward(
            self._experts(),
            torch.randn(4, 512, dtype=torch.bfloat16),
            ids,
            weights,
            backend="mono_ep",
        )
        _, ridx, rw, _, _ = runtime.calls[0]
        assert ridx[1].tolist() == [-1, -1]  # padding row
        # expert 0 keeps its two highest-weight real tokens; token 3 drops it and its
        # surviving weight is rescaled to the original gate sum
        assert ridx[:, 0].tolist() == [0, -1, 0, -1]
        torch.testing.assert_close(rw[3], torch.tensor([0.0, 1.0]))

    def test_unbuilt_runtime_raises(self):
        with pytest.raises(RuntimeError, match="no runtime built"):
            experts_fn._ep_forward(
                self._experts(),
                torch.randn(1, 512),
                torch.zeros(1, 2, dtype=torch.long),
                torch.ones(1, 2),
                backend="mono_ep",
            )

    def test_set_backend_accepts_mono_ep(self):
        experts_fn.set_backend("mono_ep")
        assert experts_fn.get_backend() == "mono_ep"


class TestArgs:
    def test_backend_and_precision_keys(self):
        args = ExpertParallelArgs(
            expert_parallel_backend="mono_ep",
            expert_parallel_mono_ep_precision="fp8",
        )
        assert args.expert_parallel_backend == "mono_ep"
        assert args.expert_parallel_mono_ep_precision == "fp8"
        assert ExpertParallelArgs().expert_parallel_mono_ep_precision == "auto"
        with pytest.raises(ValueError):
            ExpertParallelArgs(expert_parallel_mono_ep_precision="fp4")


class TestBackendResolution:
    @pytest.mark.parametrize("installed", [True, False])
    def test_auto_never_selects_mono_ep(self, monkeypatch, installed):
        monkeypatch.setattr(
            plugin_mod, "find_spec", lambda name: object() if installed else None
        )
        cfg = _cfg(expert_parallel_backend="auto")
        assert ExpertParallelPlugin._resolve_backend(cfg) == (
            "deep_ep" if installed else "torch"
        )
        assert ExpertParallelPlugin().get_distributed_env(cfg) == {}

    def test_explicit_mono_ep(self, mono_ep_package):
        cfg = _cfg()
        assert ExpertParallelPlugin._resolve_backend(cfg) == "mono_ep"
        assert cfg.expert_parallel_backend == "mono_ep"

    def test_missing_package_falls_back(self, no_package, caplog):
        cfg = _cfg()
        with caplog.at_level("WARNING"):
            assert ExpertParallelPlugin._resolve_backend(cfg) == "torch"
        assert cfg.expert_parallel_backend == "torch"
        assert "not installed" in caplog.text

    def test_missing_package_raises_without_fallback(self, no_package):
        cfg = _cfg(expert_parallel_fallback_on_unsupported=False)
        with pytest.raises(ImportError, match="not installed"):
            ExpertParallelPlugin._resolve_backend(cfg)

    @pytest.mark.parametrize(
        "extra,match",
        [
            ({"adapter": "lora"}, "adapter"),
            ({"adapter": "qlora", "load_in_4bit": True}, "adapter"),
            ({"load_in_8bit": True}, "load_in_8bit"),
            ({"gptq": True}, "gptq"),
            ({"model_quantization_config": "Mxfp4Config"}, "model_quantization_config"),
        ],
    )
    def test_config_reasons(self, mono_ep_package, extra, match):
        cfg = _cfg(**extra)
        assert ExpertParallelPlugin._resolve_backend(cfg) == "torch"
        cfg = _cfg(expert_parallel_fallback_on_unsupported=False, **extra)
        with pytest.raises(ValueError, match=match):
            ExpertParallelPlugin._resolve_backend(cfg)

    def test_lora_with_sonicmoe_lands_on_torch_sonicmoe(
        self, monkeypatch, mono_ep_package
    ):
        registered = []
        monkeypatch.setattr(
            "axolotl.integrations.kernels.libs.sonicmoe.experts.register_sonicmoe_experts",
            lambda: registered.append("sonicmoe"),
        )
        cfg = _cfg(adapter="lora", use_sonicmoe=True)
        ExpertParallelPlugin().pre_model_load(cfg)
        assert cfg.expert_parallel_backend == "torch"
        assert experts_fn.get_backend() == "torch"
        assert experts_fn.get_local_implementation() == "sonicmoe"
        assert registered == ["sonicmoe"]


class TestModelFallback:
    @pytest.mark.parametrize(
        "build,mutate,match",
        [
            ({"hidden": 384}, None, "multiple of 256"),
            ({"intermediate": 1000}, None, "multiple of 128"),
            ({"hidden_act": "gelu"}, None, "SwiGLU"),
            ({}, lambda e: setattr(e, "has_bias", True), "biases"),
            ({}, lambda e: setattr(e, "limit", 7.0), "clamped"),
            ({}, lambda e: setattr(e, "alpha", 1.702), "alpha"),
            ({}, lambda e: setattr(e, "has_gate", False), "not gated"),
            ({}, lambda e: setattr(e, "is_transposed", True), "transposed"),
            ({}, lambda e: setattr(e, "has_post_expert_norm", True), "post-expert"),
        ],
    )
    def test_unsupported_model_falls_back_to_torch(
        self, mono_ep_package, cuda_capability, caplog, build, mutate, match
    ):
        model = _qwen3_model(**build)
        if mutate is not None:
            mutate(model.mlp.experts)
        cfg = _cfg()
        with caplog.at_level("WARNING"):
            ExpertParallelPlugin().post_model_build(cfg, model)
        assert match in caplog.text
        assert cfg.expert_parallel_backend == "torch"
        assert experts_fn.get_backend() == "torch"

        model = _qwen3_model(**build)
        if mutate is not None:
            mutate(model.mlp.experts)
        with pytest.raises(ValueError, match=match):
            ExpertParallelPlugin().post_model_build(
                _cfg(expert_parallel_fallback_on_unsupported=False), model
            )

    @pytest.mark.parametrize(
        "capability,precision,match",
        [
            (None, "auto", "CUDA is unavailable"),
            ((8, 0), "auto", "< SM90"),
            ((12, 0), "auto", "sm_120"),
            ((9, 0), "mxfp8", "requires SM100"),
            ((10, 0), "fp8", "requires SM90"),
            ((9, 0), "fp8_bwd", "not implemented"),
        ],
    )
    def test_hardware_and_precision_reasons(
        self, mono_ep_package, cuda_capability, capability, precision, match
    ):
        cuda_capability(capability)
        cfg = _cfg(expert_parallel_mono_ep_precision=precision)
        geometry, reason = ExpertParallelPlugin._mono_ep_model_reason(
            cfg, _qwen3_model()
        )
        assert geometry is None and match in reason

    @pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
    def test_supported_model_keeps_backend(
        self, mono_ep_package, cuda_capability, capability
    ):
        from axolotl.monkeypatch.selective_checkpointing import has_registered_saves

        cuda_capability(capability)
        cfg = _cfg()
        plugin = ExpertParallelPlugin()
        plugin.post_model_build(cfg, _qwen3_model())
        assert cfg.expert_parallel_backend == "mono_ep"
        assert experts_fn.get_backend() == "mono_ep"
        assert plugin._mono_ep_geometry == plugin_mod.MonoEPGeometry(
            num_experts=4, hidden=512, intermediate=1024, top_k=2, activation_clamp=None
        )
        assert has_registered_saves()


class TestRuntimeLifecycle:
    @staticmethod
    def _built(cfg, monkeypatch):
        group = object()
        monkeypatch.setattr(
            ExpertParallelPlugin, "_resolve_ep_group", staticmethod(lambda cfg: group)
        )
        plugin = ExpertParallelPlugin()
        plugin._mono_ep_geometry = plugin_mod.MonoEPGeometry(
            8, 512, 1024, 2, None
        )
        cfg.activation_checkpointing = {"enabled": False}
        plugin.post_model_load(cfg, torch.nn.Linear(2, 2))
        return plugin, group

    @pytest.mark.parametrize(
        "extra,tokens",
        [
            ({}, 2048),  # 1000 * 2 rounded up to 128
            ({"eval_batch_size": 3}, 3072),
            ({"context_parallel_size": 2}, 1024),
            ({"sequence_len": 4096, "micro_batch_size": 1}, 4096),
        ],
    )
    def test_built_once_and_destroyed(
        self, monkeypatch, mono_ep_package, extra, tokens
    ):
        from torch.utils.checkpoint import CheckpointPolicy

        from axolotl.monkeypatch.selective_checkpointing import registered_save_policy

        cfg = _cfg(**extra)
        plugin, group = self._built(cfg, monkeypatch)
        (runtime,) = _FakeRuntime.instances
        assert runtime.kwargs == {
            "group": group,
            "num_experts": 8,
            "max_tokens_per_rank": tokens,
            "top_k": 2,
            "hidden": 512,
            "intermediate": 1024,
            "precision": "auto",
            "reserved_sms": 16,
            "activation_clamp": None,
        }
        assert experts_fn.get_mono_ep_runtime() is runtime
        op = SimpleNamespace(name=lambda: "mono_ep::persistent_forward")
        assert registered_save_policy(op, {}) == CheckpointPolicy.MUST_SAVE

        plugin.post_model_load(cfg, torch.nn.Linear(2, 2))
        assert len(_FakeRuntime.instances) == 1

        plugin.post_train_unload(cfg)
        assert runtime.destroyed
        with pytest.raises(RuntimeError, match="no runtime built"):
            experts_fn.get_mono_ep_runtime()
        plugin.post_train_unload(cfg)

    def test_torch_fallback_builds_nothing(self, monkeypatch, mono_ep_package):
        self._built(_cfg(expert_parallel_backend="torch"), monkeypatch)
        assert _FakeRuntime.instances == []


class TestDistributedEnv:
    @pytest.mark.parametrize("replicate,ctas", [(None, "8"), (1, "8"), (2, "5")])
    def test_mono_ep_exports_nccl_max_ctas(
        self, mono_ep_package, replicate, ctas
    ):
        env = ExpertParallelPlugin().get_distributed_env(
            _cfg(dp_replicate_size=replicate)
        )
        assert env == {"NCCL_MAX_CTAS": ctas}

    @pytest.mark.parametrize(
        "extra",
        [
            {"expert_parallel_backend": "torch"},
            {"expert_parallel_backend": "deep_ep"},
            {"expert_parallel_backend": "auto"},
            {"expert_parallel_size": 1},
            {"adapter": "lora"},
        ],
    )
    def test_other_backends_export_nothing(self, mono_ep_package, extra):
        assert ExpertParallelPlugin().get_distributed_env(_cfg(**extra)) == {}

    def test_missing_package_exports_nothing(self, no_package):
        assert ExpertParallelPlugin().get_distributed_env(_cfg()) == {}

    @pytest.mark.parametrize("value,ok", [("8", True), ("16", False)])
    def test_pre_model_load_checks_user_value(
        self, monkeypatch, mono_ep_package, value, ok
    ):
        monkeypatch.setenv("NCCL_MAX_CTAS", value)
        cfg = _cfg()
        if ok:
            ExpertParallelPlugin().pre_model_load(cfg)
            assert experts_fn.get_backend() == "mono_ep"
            assert cfg.experts_implementation == "expert_parallel"
        else:
            with pytest.raises(RuntimeError, match="NCCL_MAX_CTAS"):
                ExpertParallelPlugin().pre_model_load(cfg)
