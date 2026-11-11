"""Config validation for the fused mono_ep expert-parallel backend."""

import logging

import pytest

from axolotl.utils.config import prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

EP_PLUGIN = "axolotl.integrations.expert_parallel.ExpertParallelPlugin"
KERNELS_PLUGIN = "axolotl.integrations.kernels.KernelsPlugin"


def _validate(min_base_cfg, **kw):
    base = {
        "plugins": [EP_PLUGIN],
        "expert_parallel_size": 2,
        "expert_parallel_backend": "mono_ep",
        "activation_checkpointing": True,
        "fsdp_version": 2,
        "fsdp_config": {"state_dict_type": "FULL_STATE_DICT"},
    }
    cfg = DictDefault({**base, **kw}) | min_base_cfg
    prepare_plugins(cfg)
    return validate_config(cfg)


@pytest.fixture
def warnings(caplog):
    caplog.set_level(logging.WARNING)
    return lambda: "\n".join(r.getMessage() for r in caplog.records)


class TestMonoEPValidation:
    def test_accepts_defaults(self, min_base_cfg, warnings):
        cfg = _validate(min_base_cfg)
        assert cfg.expert_parallel_backend == "mono_ep"
        assert cfg.expert_parallel_mono_ep_precision == "auto"
        assert "mono_ep" not in warnings()

    @pytest.mark.parametrize(
        "extra,ignored",
        [
            ({"use_sonicmoe": True}, "use_sonicmoe"),
            ({"use_scattermoe": True}, "use_scattermoe"),
            ({"experts_implementation": "eager"}, "experts_implementation: eager"),
        ],
    )
    def test_warns_local_kernel_ignored(self, min_base_cfg, warnings, extra, ignored):
        plugins = [EP_PLUGIN, KERNELS_PLUGIN] if "use_" in ignored else [EP_PLUGIN]
        _validate(
            min_base_cfg, plugins=plugins, expert_parallel_mono_ep_mode="fused", **extra
        )
        assert "local experts kernel" in warnings() and ignored in warnings()

    def test_bounded_needs_compact(self, min_base_cfg):
        with pytest.raises(ValueError, match="needs expert_parallel_mono_ep_compact"):
            _validate(min_base_cfg, expert_parallel_mono_ep_bounded=True)
        cfg = _validate(
            min_base_cfg,
            expert_parallel_mono_ep_mode="dispatch",
            expert_parallel_mono_ep_compact=True,
            expert_parallel_mono_ep_bounded=True,
        )
        assert cfg.expert_parallel_mono_ep_bounded

    def test_workspace_bound_key(self, min_base_cfg):
        assert _validate(min_base_cfg).expert_parallel_mono_ep_workspace_bound == "auto"
        cfg = _validate(min_base_cfg, expert_parallel_mono_ep_workspace_bound="max")
        assert cfg.expert_parallel_mono_ep_workspace_bound == "max"
        with pytest.raises(ValueError):
            _validate(min_base_cfg, expert_parallel_mono_ep_workspace_bound="small")

    def test_warns_dispatch_chunks_ignored(self, min_base_cfg, warnings):
        _validate(min_base_cfg, expert_parallel_dispatch_chunks=2)
        assert "dispatch_chunks is ignored" in warnings()

    def test_warns_precision_without_backend(self, min_base_cfg, warnings):
        _validate(
            min_base_cfg,
            expert_parallel_backend="torch",
            expert_parallel_mono_ep_precision="bf16",
        )
        assert "only applies to" in warnings()

    def test_rejects_reentrant_checkpointing(self, min_base_cfg):
        with pytest.raises(ValueError, match="mono_ep requires non-reentrant"):
            _validate(min_base_cfg, activation_checkpointing={"reentrant": True})

    def test_rejects_unknown_precision(self, min_base_cfg):
        with pytest.raises(ValueError):
            _validate(min_base_cfg, expert_parallel_mono_ep_precision="fp4")
