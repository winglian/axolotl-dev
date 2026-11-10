# Running outside Axolotl

## The runtime API

`MonoEPRuntime` is the integration surface. One runtime per process serves every MoE layer;
it owns one symmetric buffer sized for `max_tokens_per_rank`, reserves `reserved_sms` SMs for
NCCL for its lifetime, and exposes `forward` as a custom op pair with fake implementations, so
it compiles.

```python
from mono_ep import MonoEPRuntime, check_nccl_env, nccl_env, unsupported_reason

reason = unsupported_reason(
    capability=torch.cuda.get_device_capability(),
    hidden=2048,
    intermediate=768,
    has_bias=False,
    activation="silu",
    precision="auto",
)
assert reason is None, reason

os.environ.update(
    nccl_env(reserved_sms=16)
)  # NCCL_MAX_CTAS, before the process group exists
dist.init_process_group("nccl")
check_nccl_env(reserved_sms=16)

runtime = MonoEPRuntime(
    group=dist.group.WORLD,  # the EP group; all its ranks on one NVLink node
    num_experts=128,  # global count; each rank holds num_experts / world experts
    max_tokens_per_rank=16384,  # rounded up to MonoEPRuntime.token_alignment()
    top_k=8,
    hidden=2048,
    intermediate=768,
    precision="auto",  # bf16 on SM90, mxfp8 on SM100; "fp8" for blockwise E4M3 on SM90
)
y = runtime.forward(x, topk_idx, topk_weights, w1, w2)
y.float().square().mean().backward()
runtime.destroy()  # after every graph that used it has finished
```

Constructor arguments (all keyword-only):

| Argument | Meaning |
|---|---|
| `group` | The expert-parallel process group. All its ranks must be on one NVLink node. |
| `num_experts` | Global expert count; each rank holds `num_experts / world` experts. |
| `max_tokens_per_rank` | Upper bound on tokens per rank per call; rounded up to `MonoEPRuntime.token_alignment()`. Sizes the symmetric buffer. |
| `top_k` | Experts per token. |
| `hidden`, `intermediate` | Expert input width and per-expert intermediate width (multiples of 256 and 128; 256 for both with `nvfp4_lora`). |
| `precision` | `auto` (`bf16` on SM90, `mxfp8` on SM100), `bf16`, `mxfp8` (SM100), `fp8` (blockwise E4M3, SM90) or `nvfp4_lora` (SM100, see below). |
| `reserved_sms` | SMs left free for NCCL while the persistent kernels run (default `DEFAULT_RESERVED_SMS = 16`); pair it with `nccl_env(reserved_sms=...)`. |
| `activation_clamp` | Clamped-SwiGLU limit. Only `None` is accepted today; any other value raises, because the fused backward does not implement the clamp. |

`forward(x, topk_idx, topk_weights, w1, w2)` takes `x` BF16 `[T, H]`, `topk_idx` int64 global
expert ids `[T, K]` (`-1` marks a dropped slot), `topk_weights` FP32 `[T, K]` (applied at
combine), `w1 [E_local, 2I, H]` (gate rows then up rows) and `w2 [E_local, H, I]` in BF16 or
FP32, and returns `y` BF16 `[T, H]`. Gradients `dx`, `dtopk_weights`, `dw1`, `dw2` come back
in the master dtype. Nothing but `x`, the routing and the weights is saved for backward.
`destroy()` releases the workspace and restores DeepGEMM's SM budget; call it after every graph
that used the runtime has finished.

The weight layouts the kernels need (casts, interleaving, transposes, FP8 forms) are cached per
weight storage within an epoch. A plain `torch.optim` optimizer needs nothing: a global
post-step hook ends the epoch after every `step()`. Code that rewrites weights without a
version bump (FSDP2's unsharded storage, checkpoint loads into existing parameters, `.data`
writes, weight sync from another process) must call
`mono_ep.weight_cache.invalidate_weight_caches()` after the write, or bind the cache to its own
step counter with `set_weight_cache_epoch(step)`. [architecture.md](architecture.md#weight-layout-cache)
has the full contract.

### Exported names

| Name | Use |
|---|---|
| `MonoEPRuntime` | The fused runtime above. |
| `MonoEPDispatchRuntime` | Peer dispatch and combine around your own local experts kernel (`dispatch`, `combine`, `forward_bounded` with `compact=True, bounded=True`); no DeepGEMM or SM reservation. |
| `unsupported_reason(...)` | Why the fused kernels cannot run a geometry and precision, or `None`. |
| `dispatch_unsupported_reason(...)` | The same for the dispatch runtime (SM80+, one host). |
| `resolve_precision(precision, capability)` | What `auto` resolves to on a device. |
| `nccl_env(...)`, `check_nccl_env(...)` | The `NCCL_MAX_CTAS` to export before the process group exists, and a check that it is set. |
| `DEFAULT_RESERVED_SMS` | The default SM reservation, 16. |
| `PersistentMoE`, `MonoEP`, `FP8Experts`, `SonicExperts`, `pipelined_moe`, `sonic_experts`, `sonic_fp8_experts` | Internal: the engines and reference paths the benchmarks and validators use. Their signatures can change without notice. |

## NVFP4 LoRA

`precision="nvfp4_lora"` (SM100 only) trains BF16 LoRA adapters on frozen packed NVFP4 experts.
Each optimizer step rebuilds the merged NVFP4 weights (base plus the scaled LoRA delta,
requantized) in a `RequantCache`, and the persistent forward and backward run FP4 GEMMs on
them. It is reachable only through the package API for now; Axolotl configs cannot select it.
The construction below follows the validated EP8 timing harness
[`examples/benchmark_nvfp4_lora_b200.py`](../examples/benchmark_nvfp4_lora_b200.py), which calls
`nvfp4_lora_moe` on the runtime's engine, the function `forward_lora` wraps:

```python
from mono_ep import MonoEPRuntime
from mono_ep.nvfp4_lora import Nvfp4LoraExperts, RequantCache, attach_post_step_hook

# This rank's experts from a packed NVFP4 checkpoint: q1 [E, 2I, H/2] uint8 (gate rows then up
# rows), s1 [E, 2I, H/16] E4M3, pts1 [E] FP32; q2 [E, H, I/2], s2 [E, H, I/16], pts2 [E].
# (The harness builds them from BF16 weights with requant_modules().requant's
# per_tensor_scale_from_amax and quantize_nvfp4_packed_ref.)
base = Nvfp4LoraExperts(q1, s1, pts1, q2, s2, pts2, scaling=lora_alpha / rank)
cache = RequantCache()
attach_post_step_hook(
    optimizer, cache
)  # requantize from the updated adapters after step()

runtime = MonoEPRuntime(
    group=ep_group,
    num_experts=128,
    max_tokens_per_rank=16384,
    top_k=8,
    hidden=2048,
    intermediate=768,
    precision="nvfp4_lora",
)
# Adapters (trainable, BF16): A1 [E, r, H], B1 [E, 2I, r], A2 [E, r, I], B2 [E, H, r]
y = runtime.forward_lora(x, topk_idx, topk_weights, a1, b1, a2, b2, base, cache)
```

The kernel selectors of this path are `MONO_EP_NVFP4_REQUANT` (the per-step rebuild kernel),
`MONO_EP_NVFP4_BACKWARD_RING_WAVES` and `MONO_EP_SM100_NVFP4_BACKWARD_OPTIONS`
([configuration.md](configuration.md#environment-selectors)); the design, including the static
activation scales of real checkpoints (`input_scale`), is in
[nvfp4-lora-design.md](nvfp4-lora-design.md).

## TRL `SFTTrainer`

Transformers calls a registered experts implementation with the flattened tokens, the global
expert ids and the router weights, which is exactly the runtime's `forward`:

```python
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

RUNTIME: dict[str, MonoEPRuntime] = {}


def mono_ep_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    return RUNTIME["rt"].forward(
        hidden_states.to(torch.bfloat16),
        top_k_index,
        top_k_weights.float(),
        self.gate_up_proj,
        self.down_proj,
    )


ALL_EXPERTS_FUNCTIONS.register("mono_ep", mono_ep_experts_forward)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, experts_implementation="mono_ep"
)
```

The complete script, expert parallel across the whole world with the dense weights
replicated under DDP, is [`examples/trl_sft_mono_ep.py`](../examples/trl_sft_mono_ep.py).
It slices each layer's `gate_up_proj` / `down_proj` to this rank's experts, keeps those
parameters out of DDP and scales their gradients by `1 / world` (expert gradients are already
the sum over every rank's tokens), builds one runtime from the model config, and trains with
non-reentrant `gradient_checkpointing`. The DDP exclusion and gradient scaling are written
against the Transformers and TRL interfaces but have not been run end to end; the Axolotl
seam has.

What the Axolotl seam does that the script leaves out: it builds the runtime from the config
(tokens per rank from `sequence_len x micro_batch_size`), checks the model layout before
committing to the backend, saves the fused op's output and the router `topk` under selective
activation checkpointing so recompute replays the same routing, composes EP with FSDP2 on
orthogonal mesh axes with the right gradient divide factors, and gathers every EP group's
experts into the saved checkpoint. Full checkpointing (`gradient_checkpointing=True` in the
script) recomputes the whole decoder layer including the router; the routing is deterministic
for identical inputs, which is what a non-reentrant recompute gets.

## Lower-level entry points

The benchmarks use `mono_ep.PersistentMoE` (the fixed-geometry runtime behind
`MonoEPRuntime`, with `compute_dtype`, `backward_backend` and `retain_activations` exposed)
and `mono_ep.MonoEP` (an engine that also selects the older NCCL chunk pipeline and SonicMoE
paths and caches runtimes per group and geometry). Their docstrings and
`tests/test_persistent_selection.py` describe them.
