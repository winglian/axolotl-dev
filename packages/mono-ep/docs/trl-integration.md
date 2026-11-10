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

Shapes: `x` BF16 `[T, H]`; `topk_idx` int64 global expert ids `[T, K]` (`-1` marks a dropped
slot); `topk_weights` FP32 `[T, K]`; `w1 [E_local, 2I, H]` (gate rows then up rows); `w2
[E_local, H, I]`. Gradients `dx`, `dtopk_weights`, `dw1`, `dw2` come back in the master
dtype. `precision="auto"` resolves to `bf16` on SM90 and `mxfp8` on SM100; `fp8` is SM90 only.

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
