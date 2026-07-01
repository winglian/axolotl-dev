"""Mistral-Large-4 kernel adapter.

Mistral-Large-4 is a DeepSeek-V3-style MLA + MoE model (Pixtral vision + MLA/MoE text), shipped as
a trust_remote_code checkpoint with genuine FineGrainedFP8 (128x128 block) weights. Unlike DSV4 its
MLA is *clean* — a single ``q_proj`` (``q_lora_rank=null``) and no gradientless fused indexer — so
the attention projections are safe to LoRA-adapt.

The adapter:
  * matches by ``use_mistral_large4_kernels`` or the checkpoint's ``model_type``
    (``mistral_large4`` / ``mistral_large4_text``), honoring ``trust_remote_code``;
  * registers the MLA RMSNorm class for the FSDP2 quantized-mixed-dtype fp32 shard group;
  * after PEFT, swaps the MLA attention LoRA layers for the fused quant-dispatching kernel.

The routed MoE experts are fused-3D (``gate_up_proj``/``down_proj``) and dispatch through the
scattermoe ExpertsInterface automatically (the modeling carries ``@use_experts_implementation``);
the plugin's ``use_scattermoe`` wiring owns them. The genuine fp8-block experts are *not*
reinterpreted as NVFP4 (that would corrupt them — they are real block-fp8, not NVFP4-in-disguise).
"""

from __future__ import annotations

from axolotl.integrations.kernels.adapters import ModelAdapter, load_base_model_config
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# MLA layernorm kept in fp32 and sharded separately by the FSDP2 quantized path.
MISTRAL_LARGE4_FP32_SHARD_CLASSES = ("MistralLarge4TextRMSNorm", "MistralLarge4RMSNorm")

_TEXT_TYPES = {"mistral_large4", "mistral_large4_text"}


def _model_type(cfg) -> str:
    return str(getattr(load_base_model_config(cfg), "model_type", ""))


def _is_mistral_large4(cfg) -> bool:
    return _model_type(cfg) in _TEXT_TYPES


def _has_fp8_linear(model) -> bool:
    try:
        from transformers.integrations.finegrained_fp8 import FP8Linear
    except Exception:
        return False
    return any(isinstance(m, FP8Linear) for m in model.modules())


class MistralLarge4Adapter(ModelAdapter):
    name = "mistral_large4"

    def matches(self, cfg) -> bool:
        if bool(cfg.get("use_mistral_large4_kernels")):
            return True
        return _is_mistral_large4(cfg)

    def pre_model_load(self, cfg) -> None:
        # Keep the MLA RMSNorms in their own fp32 shard group under the FSDP2 quantized path.
        from axolotl.monkeypatch.accelerate.fsdp2_quantized import (
            register_fp32_shard_classes,
        )

        register_fp32_shard_classes(MISTRAL_LARGE4_FP32_SHARD_CLASSES)

    def pre_lora_load(self, cfg, model) -> None:
        # NOTE: grouped-expert requant (mxfp8 / nvfp4) is model-agnostic and runs at the plugin level
        # (requant_grouped_experts) BEFORE this hook, so the fp8-linear patch below sees only the
        # non-expert linears. moe_grouped_mode='fp8_block' keeps the block-fp8 experts native.
        #
        # A FineGrainedFP8 base can't backprop through transformers' fp8 GEMM (no autograd). If any
        # fp8 (non-expert) linear remains, install the dequant-autograd forward so LoRA works.
        if _has_fp8_linear(model):
            from axolotl.integrations.kernels.libs.mistral_large4.fp8_training import (
                patch_fp8_linear_training,
            )

            patch_fp8_linear_training()

    def post_model_load(self, cfg, model) -> None:
        self._patch_attn_lora(cfg, model)

    @staticmethod
    def _patch_attn_lora(cfg, model) -> None:
        """Fused quant-dispatching LoRA for the MLA attention projections. Default on for a LoRA
        run; disable with ``mistral_large4_attn_lora_kernel: false``."""
        if cfg.get("adapter") != "lora":
            return
        if cfg.get("mistral_large4_attn_lora_kernel") is False:
            return
        from axolotl.integrations.kernels.libs.mistral_large4 import (
            patch_mistral_large4_attn_lora,
        )
        from axolotl.integrations.kernels.libs.mistral_large4.lora_attn import (
            MLA_PROJ_SUFFIXES,
        )

        # Optional: quantize the frozen attention bases to a low-bit format (kernel then runs the
        # 4-/8-bit fused path). Must run post-PEFT-wrap (here), before the kernel patch.
        attn_quant = cfg.get("mistral_large4_attn_quantization")
        if attn_quant and str(attn_quant).lower() != "none":
            from axolotl.integrations.kernels.libs.mistral_large4.lora_lowbit import (
                quantize_attn_bases,
            )

            quantize_attn_bases(model, str(attn_quant).lower(), MLA_PROJ_SUFFIXES)

        native_fp8 = cfg.get("mistral_large4_attn_native_fp8") is not False
        patch_mistral_large4_attn_lora(model, enabled=True, native_fp8=native_fp8)
