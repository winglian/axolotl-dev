"""CPU tests for the ``mono_ep`` fused LoRA precisions (``bf16_lora`` / ``mxfp8_lora``):
config and model qualification, the dispatch fallback selector, routing of the fused
forward and the merged-weight cache's per-process state."""

# ruff: noqa: F811  (fixtures imported from test_mono_ep_backend are reused by name)

from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.expert_parallel import (
    ExpertParallelPlugin,
    experts_fn,
    fused_lora,
)

from .test_mono_ep_backend import (  # noqa: F401
    _cfg,
    _qwen3_model,
    _restore_ep_state,
    cuda_capability,
    mono_ep_package,
)
from .test_mono_ep_nvfp4_lora import _lora_cfg, _nvfp4_model


def _bf16_lora_cfg(**kw):
    return _lora_cfg(base_model="Qwen/Qwen3-30B-A3B", **kw)


class TestConfigReason:
    @pytest.mark.parametrize("precision", ["auto", "bf16", "mxfp8", "mxfp8_lora"])
    def test_qualifying_config_allows_fused(self, mono_ep_package, precision):
        cfg = _bf16_lora_cfg(expert_parallel_mono_ep_precision=precision)
        assert ExpertParallelPlugin._mono_ep_fused_config_reason(cfg) is None
        assert ExpertParallelPlugin._mono_ep_fused_possible(cfg)

    @pytest.mark.parametrize(
        "extra,match",
        [
            ({"adapter": "qlora"}, "single lora adapter"),
            ({"peft_use_dora": True}, "DoRA"),
            ({"lora_dropout": 0.1}, "lora_dropout: 0"),
            ({"use_sonicmoe": False}, "use_sonicmoe: true"),
            ({"lora_target_parameters": ["experts.gate_up_proj"]}, "down_proj"),
        ],
    )
    def test_unsupported_adapters_keep_the_dispatch_mode(
        self, mono_ep_package, extra, match
    ):
        reason = ExpertParallelPlugin._mono_ep_fused_config_reason(
            _bf16_lora_cfg(expert_parallel_mono_ep_precision="mxfp8", **extra)
        )
        assert "the fused kernels cannot train it" in reason and match in reason
        explicit = ExpertParallelPlugin._mono_ep_fused_config_reason(
            _bf16_lora_cfg(expert_parallel_mono_ep_precision="mxfp8_lora", **extra)
        )
        assert match in explicit

    def test_selector_restores_the_dispatch_fallback(self, mono_ep_package):
        cfg = _bf16_lora_cfg(
            expert_parallel_mono_ep_precision="mxfp8",
            expert_parallel_mono_ep_fused_lora=False,
        )
        reason = ExpertParallelPlugin._mono_ep_fused_config_reason(cfg)
        assert "trains full expert weights only" in reason
        assert not ExpertParallelPlugin._mono_ep_fused_possible(cfg)

    def test_lora_precision_without_adapter_is_refused(self, mono_ep_package):
        reason = ExpertParallelPlugin._mono_ep_fused_config_reason(
            _cfg(expert_parallel_mono_ep_precision="bf16_lora")
        )
        assert "needs adapter: lora" in reason


class TestModelResolution:
    @pytest.mark.parametrize(
        "capability,precision,resolved",
        [
            ((10, 0), "auto", "mxfp8_lora"),
            ((10, 0), "mxfp8", "mxfp8_lora"),
            ((10, 0), "bf16", "bf16_lora"),
            ((9, 0), "auto", "bf16_lora"),
            ((9, 0), "fp8", "fp8_lora"),
        ],
    )
    def test_bf16_experts_resolve_the_lora_precision(
        self, mono_ep_package, cuda_capability, capability, precision, resolved
    ):
        cuda_capability(capability)
        cfg = _bf16_lora_cfg(expert_parallel_mono_ep_precision=precision)
        ExpertParallelPlugin().post_model_build(cfg, _qwen3_model())
        assert cfg.expert_parallel_backend == "mono_ep"
        assert cfg.expert_parallel_mono_ep_mode == "fused"
        assert cfg.expert_parallel_mono_ep_precision == resolved

    @pytest.mark.parametrize(
        "capability,precision,build,match",
        [
            ((9, 0), "mxfp8", _qwen3_model, "requires SM100"),
            ((10, 0), "fp8", _qwen3_model, "requires SM90"),
            ((10, 0), "mxfp8", _nvfp4_model, "packed NVFP4"),
        ],
    )
    def test_unsupported_take_the_dispatch_mode(
        self, mono_ep_package, cuda_capability, capability, precision, build, match
    ):
        cuda_capability(capability)
        cfg = _bf16_lora_cfg(expert_parallel_mono_ep_precision=precision)
        _, reason = ExpertParallelPlugin._mono_ep_model_reason(cfg, build())
        assert match in reason
        ExpertParallelPlugin().post_model_build(cfg, build())
        assert cfg.expert_parallel_mono_ep_mode == "dispatch"


class TestForward:
    def test_fused_backend_routes_the_lora_precisions(self, monkeypatch):
        routed = []
        monkeypatch.setattr(
            fused_lora, "forward", lambda *args: routed.append(args) or args[2]
        )
        runtime = SimpleNamespace(precision="mxfp8_lora")
        experts_fn.set_mono_ep_runtime(runtime)
        experts = _qwen3_model(num_experts=4).mlp.experts
        x = torch.zeros(3, experts.gate_up_proj.shape[-1])
        out = experts_fn._MonoEPFusedBackend.forward(
            experts,
            x,
            torch.zeros(3, 2, dtype=torch.int64),
            torch.ones(3, 2),
            num_experts_global=4,
            num_local_experts=4,
        )
        assert out is x and routed[0][0] is runtime
        assert routed[0][5] is experts.gate_up_proj
        assert routed[0][6] is experts.down_proj

    def test_forward_hands_base_and_mapped_factors(self, monkeypatch):
        e, r, h, i, tokens = 2, 2, 256, 128, 5
        experts = SimpleNamespace(
            num_experts=e, num_experts_global=e, local_expert_offset=0
        )
        a1 = torch.randn(e * r, h, requires_grad=True)
        b1 = torch.randn(2 * i, r * e, requires_grad=True)
        a2 = torch.randn(e * r, i, requires_grad=True)
        b2 = torch.randn(h, r * e, requires_grad=True)
        experts._sonicmoe_lora = {
            "gate_up_proj": (a1, b1, 2.0),
            "down_proj": (a2, b2, 2.0),
        }
        experts._sonicmoe_lora_wrappers = {}
        layer = SimpleNamespace(scaling=2.0)
        state = SimpleNamespace(cache=object(), layer=lambda experts_, scaling: layer)
        monkeypatch.setattr(fused_lora, "get_state", lambda: state)
        calls = []

        def forward_lora(x, idx, w, *factors, w1, w2):
            calls.append((factors, w1, w2))
            return x

        w1, w2 = torch.zeros(e, 2 * i, h), torch.zeros(e, h, i)
        runtime = SimpleNamespace(precision="bf16_lora", forward_lora=forward_lora)
        x = torch.randn(tokens, h)
        fused_lora.forward(
            runtime,
            experts,
            x,
            torch.zeros(tokens, 2, dtype=torch.int64),
            torch.ones(tokens, 2),
            w1,
            w2,
        )
        (ma1, mb1, ma2, mb2, got_layer, cache), got_w1, got_w2 = calls[0]
        assert got_w1 is w1 and got_w2 is w2
        assert got_layer is layer and cache is state.cache
        assert ma1.shape == (e, r, h) and mb1.shape == (e, 2 * i, r)
        assert ma2.shape == (e, r, i) and mb2.shape == (e, h, r)

    def test_mismatched_scalings_are_refused(self, monkeypatch):
        experts = SimpleNamespace(num_experts=1, num_experts_global=1)
        experts._sonicmoe_lora = {
            "gate_up_proj": (torch.zeros(1, 4), torch.zeros(4, 1), 2.0),
            "down_proj": (torch.zeros(1, 2), torch.zeros(4, 1), 1.0),
        }
        monkeypatch.setattr(fused_lora, "get_state", lambda: SimpleNamespace())
        with pytest.raises(ValueError, match="share one LoRA scaling"):
            fused_lora.forward(
                SimpleNamespace(),
                experts,
                torch.zeros(1, 4),
                None,
                None,
                torch.zeros(1, 4, 4),
                torch.zeros(1, 4, 2),
            )


class TestState:
    def test_runtime_build_creates_and_unload_clears_the_state(
        self, mono_ep_package, cuda_capability, monkeypatch
    ):
        import sys
        import types

        class MergedWeightCache:
            def __init__(self):
                self.mode, self.version = "layer", 0

            def invalidate(self):
                self.version += 1

        lora = types.ModuleType("mono_ep.lora")
        lora.MergedWeightCache = MergedWeightCache
        monkeypatch.setitem(sys.modules, "mono_ep.lora", lora)
        mono_ep_package.lora = lora
        monkeypatch.setattr(
            ExpertParallelPlugin, "_resolve_ep_group", staticmethod(lambda cfg: None)
        )
        cuda_capability((10, 0))
        cfg = _bf16_lora_cfg(expert_parallel_mono_ep_lora_merged_cache="all")
        plugin = ExpertParallelPlugin()
        plugin.post_model_build(cfg, _qwen3_model())
        plugin._build_mono_ep_runtime(cfg)
        state = fused_lora.get_state()
        assert state is not None
        import os

        assert os.environ["MONO_EP_LORA_MERGED_CACHE"] == "all"
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        optimizer.step()
        assert state.cache.version == 1
        plugin.post_train_unload(cfg)
        assert fused_lora.get_state() is None
