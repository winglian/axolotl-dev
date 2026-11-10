# mono-ep

Fused expert-parallel MoE training kernels for Hopper (SM90) and Blackwell (SM100). Each
direction of a mixture-of-experts layer runs as one persistent kernel per GPU: it pulls the
tokens routed to its experts over NVLink symmetric memory, runs the gate/up GEMM, the SwiGLU,
the down GEMM, and scatters the weighted results back to the owning ranks, without leaving
the kernel between those steps. The backward is the same shape of kernel run in reverse,
with the gate/up projections recomputed instead of saved. The kernels are JIT-compiled
through a vendored fork of DeepGEMM; the Python package wraps them as `torch.library` custom
ops with fake implementations, so a whole MoE layer compiles with `fullgraph=True`.

On 8xH200 at the Qwen3-30B-A3B layer shape the checkpointed layer step (forward, recomputed
forward and backward) takes 12.3 ms against 15.9 ms for the published prime-rl fused kernels
on the same hardware, and 37.3 ms against 43.7 ms at the Qwen3-235B-A22B shape. The
[benchmarks](#benchmarks) section has the tables and what they do and do not measure.

The package has no Axolotl, Transformers or TRL dependency. Axolotl selects it with
`expert_parallel_backend: mono_ep`; everything else (model layout, FSDP, checkpoints,
fallback) stays in Axolotl.

## Installation

Requirements: a Hopper (H100, H200) or Blackwell (B200) GPU, all GPUs of the expert-parallel
group on one NVLink node, CUDA 12.9 or newer (13.0 is what the Axolotl images and the
benchmarks use), PyTorch 2.9 or newer, Triton 3.4 or newer, a C++20 compiler with
`<format>` (GCC 13), and the ELF/DWARF development headers DeepGEMM's build wants
(`libelf-dev`, `libdw-dev`). SM120 (RTX PRO 6000, RTX 50-series) and older architectures
are not supported.

Install the package and build the vendored DeepGEMM:

```bash
pip install -e packages/mono-ep
python packages/mono-ep/tools/fetch_build_dependencies.py   # CUTLASS and DeepJIT at the pinned revisions
DG_USE_LOCAL_VERSION=0 DG_FORCE_BUILD=1 CC=gcc-13 CXX=g++-13 \
  pip install --no-build-isolation --no-deps packages/mono-ep/vendor/deepgemm
```

`cicd/install_mono_ep.sh` at the repository root does the same inside the Axolotl CI images
(it also installs GCC 13). The kernels themselves are compiled on first use and cached under
`DG_JIT_CACHE_DIR`; the first forward and backward of a new shape take a few minutes.

The optional extras pull SonicMoE for the BF16 chunk pipeline (`sonic`) and the experimental
SonicMoE FP8 path (`sonic-fp8-experimental`); neither is needed for the persistent kernels.

## Running with Axolotl

Full fine-tuning of Qwen3-30B-A3B on one node of eight H100, H200 or B200 GPUs, expert
parallel 8, every option the fused backend needs spelled out:

```yaml
base_model: Qwen/Qwen3-30B-A3B

plugins:
  - axolotl.integrations.expert_parallel.ExpertParallelPlugin
expert_parallel_size: 8               # world size 8: one EP group, no dp_shard axis
expert_parallel_backend: mono_ep
expert_parallel_mono_ep_precision: auto   # bf16 on Hopper, mxfp8 on Blackwell (table below)

datasets:
  - path: mhenrichsen/alpaca_2k_test
    type: alpaca
dataset_prepared_path:
output_dir: ./outputs/qwen3-30b-a3b-mono-ep

sequence_len: 4096
sample_packing: true
micro_batch_size: 1
gradient_accumulation_steps: 4
num_epochs: 1

optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 5e-6
warmup_ratio: 0.05
bf16: true
tf32: false

activation_checkpointing: true        # non-reentrant; reentrant checkpointing is rejected with EP
flash_attention: true
logging_steps: 1
saves_per_epoch: 1
save_only_model: true
seed: 42

fsdp_version: 2
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen3MoeDecoderLayer
  state_dict_type: FULL_STATE_DICT   # required with EP; SHARDED_STATE_DICT would keep one EP group's experts
  reshard_after_forward: true
```

```bash
axolotl train qwen3-30b-a3b-mono-ep.yaml
```

For four GPUs set `expert_parallel_size: 4`; for sixteen GPUs on two nodes set
`expert_parallel_size: 8` and `dp_shard_size: 2` (the EP group must stay inside one node).
`examples/expert_parallel/` has more configurations.

### Config keys

| Key | Values | Meaning |
|---|---|---|
| `expert_parallel_size` | `1` (off) or a divisor of the world size | Number of EP ranks; experts are split evenly across them. |
| `expert_parallel_backend` | `auto`, `deep_ep`, `torch`, `mono_ep` | Token dispatch backend. `auto` never picks `mono_ep`; set it explicitly. With `mono_ep` the local experts kernel (`use_sonicmoe`, `use_scattermoe`, `experts_implementation`) is not used. |
| `expert_parallel_mono_ep_precision` | `auto`, `bf16`, `mxfp8`, `fp8`, `fp8_bwd` | Compute precision of the fused kernels (table below). Master weights keep their dtype and weight gradients come back in it. |
| `expert_parallel_fallback_on_unsupported` | `true` (default), `false` | When `mono_ep` cannot run (table below), fall back to `expert_parallel_backend: torch` with a warning, or raise. |
| `dp_shard_size` | | FSDP shard axis next to EP; `expert_parallel_size x dp_shard_size` must equal the world size. |

| Precision | SM90 (H100, H200) | SM100 (B200) |
|---|---|---|
| `auto` | `bf16` | `mxfp8` |
| `bf16` | yes | yes |
| `mxfp8` | no | yes |
| `fp8` (blockwise E4M3 forward and backward, the kernels timed in [benchmarks](#benchmarks)) | yes | no |
| `fp8_bwd` (BF16 forward, FP8 backward) | not implemented | no |

The backend falls back (or raises) when the `mono_ep` package or its DeepGEMM build is
missing, an adapter or quantized loading is configured, the GPU is not SM90/SM100, the
model's `hidden_size` is not a multiple of 256 or its expert intermediate size not a
multiple of 128, the experts have biases, a non-SiLU gate, a clamped SwiGLU or a layout other
than HF's `gate_up_proj [E, 2I, H]` / `down_proj [E, H, I]`, or the precision does not match
the architecture. The Axolotl-side details (checkpointing policy, the FSDP mesh, what is
checked when) are in `src/axolotl/integrations/expert_parallel/README.md`.

### Environment selectors

The plugin sets everything the kernels need; the variables below exist for experiments and
for restoring a previous default. Defaults are the values measured in the last Hopper and
Blackwell tuning rounds; every one of them was checked bitwise or against an FP32 reference
against its restore value before it became a default.

| Variable | Default | Effect |
|---|---|---|
| `NCCL_MAX_CTAS` | exported by the plugin: `reserved_sms // 2` (`// 3` under HSDP) | NCCL must fit in the SMs the persistent kernels leave free (16 by default, `DEFAULT_RESERVED_SMS`) or FSDP collectives deadlock against the spinning grid. A user value wins but may not exceed the plugin's. |
| `MONO_EP_WEIGHT_CACHE` | `1` (`0`, `legacy`) | Cache the per-call weight layouts (casts, transposes, interleaving) across calls, invalidated after every optimizer step. |
| `MONO_EP_WEIGHT_CACHE_OPTIMIZER_HOOK` | `1` | Register the global `torch.optim` post-step hook that invalidates the cache. |
| `MONO_EP_WEIGHT_CACHE_KEY` | `auto` (`storage`, `storage_epoch`, `object`) | Cache key; `auto` uses the storage key, or storage plus the caller's epoch when one is set. |
| `MONO_EP_WEIGHT_CACHE_ENTRIES` | `2` | Entries per layout name. |
| `MONO_EP_WEIGHT_CACHE_COMPILED` | `1` | Build the layouts with compiled builders instead of eager ops. |
| `MONO_EP_WEIGHT_CACHE_FUSED` | `0` | Fuse the compiled builds of one call (SM100 experiment). |
| `MONO_EP_SM90_FP8_WEIGHT_CACHE` | `1` | SM90 FP8: cache the quantized weight forms per optimizer step. |
| `MONO_EP_SM90_FP8_FUSED_PREP` | `1` | SM90 FP8: fused token quantization and prefix preparation. |
| `MONO_EP_SM90_FP8_FUSED_WEIGHT_QUANT` | `1` | SM90 FP8: fused per-expert transpose, scale reduction and FP8 conversion of the weights. |
| `MONO_EP_SM90_FP8_TRANSPORT`, `MONO_EP_SM90_FP8_COMBINE` | unset (BF16) | SM90 FP8 backward: FP8 x/dy transport and FP8 dX combine; the benchmarks set both to `1`. |
| `MONO_EP_SM90_FP8_FORWARD_STAGES` | `5` (2..5) | SM90 FP8 forward TMA pipeline stages. |
| `MONO_EP_SM90_FP8_BACKWARD_STAGES` | `3` (0..8) | SM90 FP8 backward stages with the cached-dZ layout. |
| `MONO_EP_SM90_FP8_FORWARD_MAINLOOP`, `MONO_EP_SM90_FP8_BACKWARD_MAINLOOP` | `1`, `0` | Mainloop variants (register allocation and accumulator handling). |
| `MONO_EP_SM90_FP8_FORWARD_ROLE_REGISTERS`, `MONO_EP_SM90_FP8_ROLE_REGISTERS` | `2`, `2` | `setmaxnreg` placement per warpgroup role. |
| `MONO_EP_SM90_FP8_TAIL_SCHEDULE` | `1` (0..3) | Backward: split the last wave's M-blocks so the CTA tail is shorter. |
| `MONO_EP_SM90_FP8_OVERLAP_EPILOGUE` | `0` | Backward epilogue/mainloop overlap sites (measured slower at EP8 with the round-4 defaults). |
| `MONO_EP_SM90_FP8_DISPATCH_XDY_PACK` | `0` | Pack x/dy columns on the dispatch warps (faster single-rank, slower at EP8). |
| `MONO_EP_SM90_FP8_FORWARD_SCATTER_MODE` | `0` (0, 1, 3) | Forward combine scatter ordering. |
| `MONO_EP_SM90_FP8_DISPATCH_PULL_BYTES` | 4 KB per row, capped at 2 KB when the stage counts would exceed shared memory | NVLink pull granularity. |
| `MONO_EP_SM90_FP8_ACTIVATION` | `silu` | Gated activation (`swiglu_fast` is the measured opt-in). |
| `MONO_EP_SM90_FP8_COMPACT_FORWARD`, `MONO_EP_SM90_FP8_COMPACT_BACKWARD` | unset | 384-thread layouts kept for experiments; the 512-thread layouts are the defaults. |
| `MONO_EP_SM100_BACKWARD_STAGES` | `6` | SM100 backward pipeline stages. |
| `MONO_EP_SM100_BACKWARD_OPTIONS` | `11` | SM100 backward epilogue option bits. |
| `MONO_EP_SM100_BACKWARD_PULL_BYTES` | two rows (8 KB) when whole rows fit, else `4096` | SM100 backward NVLink pull size. |
| `MONO_EP_SM100_FORWARD_OPTIONS` | lookahead pull on when the pull buffer is one row (`0` restores) | SM100 forward option bits. |
| `MONO_EP_SM100_FORWARD_PULL_BYTES` | per shape | SM100 forward pull size. |
| `MONO_EP_SM100_DW_DIRECT` | `0` | SM100: direct weight-gradient operands (experiment). |
| `MONO_EP_SM90_FP8_TRAP_TIMEOUTS`, `*_PHASE_TRACE`, `*_DIST_TRACE`, `*_DIAG_ABLATE` | unset | Diagnostics: trap-only timeouts for compiler experiments, per-phase and per-rank traces, epilogue ablations. Not for training. |

## Running outside Axolotl

`MonoEPRuntime` is the integration surface. One runtime per process serves every MoE layer;
it owns one symmetric buffer sized for `max_tokens_per_rank`, reserves `reserved_sms` SMs
for NCCL for its lifetime, and exposes `forward` as a custom op pair with fake
implementations, so it compiles.

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
# x BF16 [T, H]; topk_idx int64 global expert ids [T, K] (-1 = dropped slot);
# topk_weights FP32 [T, K]; w1 [E_local, 2I, H] (gate rows then up rows); w2 [E_local, H, I]
y = runtime.forward(x, topk_idx, topk_weights, w1, w2)
y.float().square().mean().backward()  # dx, dtopk_weights, dw1, dw2 in the master dtype
runtime.destroy()  # after every graph that used it has finished
```

A TRL `SFTTrainer` run that uses it with a Transformers Qwen3-MoE model, expert parallel
across the whole world and the dense weights replicated (DDP). This is the shortest wiring
that matches the API above; the pieces marked *untested* are written against the Transformers
and TRL interfaces but have not been run end to end (Axolotl's seam, which has, also handles
FSDP2, checkpoint gathering and selective activation checkpointing):

```python
import os
import torch, torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
from trl import SFTConfig, SFTTrainer
from mono_ep import MonoEPRuntime, nccl_env, check_nccl_env

os.environ.update(nccl_env(reserved_sms=16))  # must precede init_process_group
dist.init_process_group("nccl")
check_nccl_env(reserved_sms=16)
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)

RUNTIME = {}


def mono_ep_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    # Transformers calls the registered experts implementation with the flattened tokens
    # [T, H], global expert ids [T, K] and the router weights [T, K].
    return RUNTIME["rt"].forward(
        hidden_states.to(torch.bfloat16),
        top_k_index,
        top_k_weights.float(),
        self.gate_up_proj,
        self.down_proj,
    )


ALL_EXPERTS_FUNCTIONS.register("mono_ep", mono_ep_experts_forward)

model_id = "Qwen/Qwen3-30B-A3B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    experts_implementation="mono_ep",
).cuda()
cfg = model.config
E, H, I, K = (
    cfg.num_experts,
    cfg.hidden_size,
    cfg.moe_intermediate_size,
    cfg.num_experts_per_tok,
)
assert E % world == 0
e0 = rank * (E // world)

expert_params = []
for layer in model.model.layers:
    experts = layer.mlp.experts
    # keep this rank's expert slice as plain parameters; the runtime routes by global id
    experts.gate_up_proj = torch.nn.Parameter(
        experts.gate_up_proj.data[e0 : e0 + E // world].clone()
    )
    experts.down_proj = torch.nn.Parameter(
        experts.down_proj.data[e0 : e0 + E // world].clone()
    )
    expert_params += [experts.gate_up_proj, experts.down_proj]
# DDP averages dense gradients over ranks; expert gradients are already the sum over every
# rank's tokens, so keep them out of DDP and scale them to match.    (untested)
torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
    model,
    [n for n, p in model.named_parameters() if any(p is q for q in expert_params)],
)
for p in expert_params:
    p.register_hook(lambda g: g / world)

seq_len, micro_batch = 4096, 1
RUNTIME["rt"] = MonoEPRuntime(
    group=dist.group.WORLD,
    num_experts=E,
    max_tokens_per_rank=seq_len * micro_batch,
    top_k=K,
    hidden=H,
    intermediate=I,
    precision="auto",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tok,
    train_dataset=load_dataset("mhenrichsen/alpaca_2k_test", split="train"),
    args=SFTConfig(
        output_dir="./outputs/qwen3-30b-a3b-trl-mono-ep",
        max_length=seq_len,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # reentrant recompute cannot replay the fused op
        ddp_find_unused_parameters=False,
        save_strategy="no",
        report_to="none",
    ),
)
trainer.train()
RUNTIME["rt"].destroy()
```

Things the Axolotl seam does that this example leaves out: it builds the runtime from the
config (tokens per rank from `sequence_len x micro_batch_size`), checks the model layout
before committing to the backend, saves the fused op's output and the router `topk` under
selective activation checkpointing so recompute replays the same routing, composes EP with
FSDP2 on orthogonal mesh axes with the right gradient divide factors, and gathers every EP
group's experts into the saved checkpoint. Full checkpointing (`gradient_checkpointing=True`
above) recomputes the whole decoder layer and is correct but recomputes the router too; the
routing is deterministic for identical inputs, which is what a non-reentrant recompute gets.

The lower-level entry points the benchmarks use are `mono_ep.PersistentMoE` (the
fixed-geometry runtime behind `MonoEPRuntime`, with `compute_dtype`, `backward_backend` and
`retain_activations` exposed) and `mono_ep.MonoEP` (an engine that also selects the older
NCCL chunk pipeline and SonicMoE paths and caches runtimes per group and geometry). Their
docstrings and `tests/test_persistent_selection.py` describe them.

## How it works

Forward, per GPU, one persistent kernel:

1. Every rank writes its routed rows into its own symmetric buffer and publishes per-expert
   counts. Each CTA (SM90) or CTA pair (SM100) owns M-blocks of 128 routed rows of one local
   expert and pulls them from the owning ranks over NVLink in fixed-size chunks, spinning on
   readiness flags rather than returning to the host. Dispatch and compute overlap because
   the pulls are issued ahead of the GEMM (the SM100 forward uses a lookahead pull, the SM90
   forward a 2-5 stage TMA pipeline).
2. The gate/up GEMM runs from the pulled tiles (WGMMA on SM90, UMMA with tensor memory on
   SM100); on SM90 FP8 the activations are quantized in the epilogue with per-row scales and
   the down GEMM consumes them directly. SwiGLU and the top-k weighting are fused into the
   epilogues.
3. The down GEMM's epilogue scatters each row's weighted contribution back into the owner's
   combine slot (exclusive slots, so BF16 stores and no remote atomics), and fences
   completion per rank.

Backward is one kernel of the same structure in the other direction: it restages `x` and
`dy` (nothing from the forward is retained; the forward output is what activation
checkpointing saves), recomputes the gate/up projection per owned M-block, runs the SwiGLU
derivative and the score-gradient reduction in the epilogue, produces `dh` and the `dz` the
weight gradients need, and scatters `dx` back to the owning ranks. `dH` stays CTA-local;
`h` and `dz` are written to a per-rank workspace once per K-tile, and the weight gradients
run afterwards as grouped GEMMs on GPU-resident expert offsets (no host size sync). On SM90
FP8 the x/dy transport, the dX combine and the weight-gradient operands are all E4M3 with
128-element scale groups; the final outputs are BF16 and weight gradients come back in the
master dtype.

Around the kernels: `MonoEPRuntime` reserves SMs by setting DeepGEMM's SM budget to
`physical - reserved_sms`, so NCCL collectives issued by FSDP during the persistent kernel
have somewhere to run; `nccl_env` sizes `NCCL_MAX_CTAS` to that budget. Weight layouts that
the kernels want (BF16 casts, interleaved gate/up rows, transposes, FP8 forms and scales) are
built once per optimizer step by `weight_cache.py`, keyed in a way that survives FSDP2's
re-allocation of unsharded parameters and invalidated by a global optimizer post-step hook.
The ops are registered with fake implementations and no host synchronization in the Python
layer, so a layer compiles with `fullgraph=True`; the launcher reads the shape and
architecture, picks stage counts and pull sizes that fit shared memory (`mono_ep_dispatch_pull.hpp`
caps the pull at 2 KB at hidden 4096), and the JIT key includes every selector.

## What was added, and what is vendored

`vendor/deepgemm` is DeepSeek's DeepGEMM at revision `5081f851` (MIT), without its tests and
CI, built against the CUTLASS and DeepJIT revisions in `vendor/sources.json`. Everything
under `csrc/jit_kernels/impls/*mono_ep*`, `csrc/jit_kernels/heuristics/mono_ep.hpp`,
`csrc/apis/mono_ep.hpp` and `deep_gemm/include/deep_gemm/{impls,layout,scheduler}/*mono_ep*`
is new code in this tree; upstream's fused-MoE feature prefix is spelled `mono_ep` /
`MonoEP` throughout (`tools/rename_vendor_names.py` is the mapping). `sources.json` lists
every local modification; by theme:

- **SM90 persistent FP8 forward and backward** (`sm90_fp8_mono_ep_forward.cuh`,
  `sm90_fp8_mono_ep_backward.cuh`): one-CTA M-block ownership, WGMMA accumulation,
  blockwise E4M3 with per-row activation scales, FP8 x/dy transport, FP8 dX combine, cached
  dZ, direct column-scaled weight-gradient operands, dispatch ring-slot spin, the 512-thread
  warp layout with per-role `setmaxnreg`, tail dX scheduling, pluggable gated activations.
- **SM90 persistent BF16 forward and backward** (`sm90_bf16_mono_ep*.cuh`): the one-CTA
  WGMMA port of the SM100 design, with K-grouped weight gradients on GPU-resident offsets.
- **SM100 persistent BF16 forward and backward** (`sm100_bf16_mono_ep*.cuh`): two-CTA pairs
  with UMMA and tensor memory, the recompute SwiGLU-derivative epilogue that emits `h`, `dz`
  and score partials without storing `z`, six-stage backward pipeline, two-row backward
  pulls and forward lookahead pulls, masked M-group tile suppression.
- **SM100 FP8/FP4 forward** (`sm100_fp8_fp4_mono_ep.cuh`): MXFP8 forward with optional
  gate/up retention.
- **Shared pieces**: the readiness-queue scheduler (`scheduler/mono_ep.cuh`, adapted from
  TeraMoE's BF16 chunk queue design), layouts (`layout/mono_ep.cuh`), the activation
  selector and the dispatch pull sizing, exclusive-slot scatter-store epilogues, and the
  grouped GEMM entry points with in-kernel readiness and GPU row bounds.
- **Host side** (`src/mono_ep`): the `torch.library` ops and fake implementations, the
  `PersistentMoE` runtime and `MonoEPRuntime` API, the NCCL reference backward, the weight
  layout cache and its FSDP2 contract, SM90 FP8 weight forms and fused token preparation
  (Triton), the quantization primitives, and the older NCCL chunk pipeline and SonicMoE
  integration the prototype started from.

`src/mono_ep/_vendor/sonic_fp8.py` is an extract of SonicMoE's functional entry points
(Apache-2.0, revision in `sources.json`) with the modifications listed there; it is only
used by the experimental Sonic FP8 path. `src/mono_ep/include` holds the CUDA sources of an
earlier SM100 owned-backward extension (`owned_backward.py`, built with `cpp_extension`).

The raw measurement files and the development log that records what was tried, kept and
rejected are kept outside the tree, next to the package's `benchmarks/` directory, which is
gitignored.

## Credits

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (DeepSeek, MIT): the JIT framework,
  the SM90/SM100 GEMM mainloops, TMA and UMMA helpers, and the grouped GEMM APIs the kernels
  are built from, together with [DeepJIT](https://github.com/deepseek-ai/DeepJIT) and
  NVIDIA [CUTLASS](https://github.com/NVIDIA/cutlass).
- [DeepEP](https://github.com/deepseek-ai/DeepEP): the intranode symmetric-memory
  dispatch/combine model this design and the Axolotl `deep_ep` backend are compared against;
  no DeepEP code is used here.
- [TeraMoE](https://github.com/PFCCLab/TeraMoE) (PFCCLab, MIT): the BF16 chunk queue and
  scheduler design the readiness queue adapts (listed as a design reference in
  `sources.json`; no code copied).
- [SonicMoE](https://github.com/bdepyzy/sonic-moe) (Apache-2.0), with
  [quack](https://github.com/Dao-AILab/quack) and the CUTLASS DSL: the BF16 and FP8 local
  expert kernels the prototype measured against and the extracted FP8 functional path.
- Prime Intellect's prime-rl fused MoE kernels: the op contract this package follows
  (custom ops with fake implementations and an integer runtime key, nothing retained from
  the forward, one symmetric buffer per process, SMs reserved for NCCL, `NCCL_MAX_CTAS`
  exported before process-group initialization) and the published 8xH200 numbers used as the
  comparison target.

## Benchmarks

The contract: a Qwen3-30B-A3B MoE layer at expert parallel 8, 16,384 tokens per GPU, hidden
2048, intermediate 768, 128 experts (16 per GPU), top-8, so each GPU receives about 131,072
routed rows; FP32 master weights; the whole layer `fullgraph` compiled; five warmups, fifteen
measured iterations, median over iterations of the slowest rank. "Checkpointed 2F+B" is the
forward, the recomputed forward and the backward of one layer, grouped weight gradients and
runtime staging included, router and optimizer excluded. The "NCCL control" shares the same
persistent forward and runs a fixed-capacity NCCL all-to-all backward in the same process and
allocation. Hopper runs are blockwise FP8 (E4M3, 1x128 and 128x128 scales); B200 runs are
BF16 compute. The 235B rows use hidden 4096, intermediate 1536, 128 experts, top-8.
The Modal app that produced each table is listed under it.

8xH200, balanced routing, max-rank median ms, against the published prime-rl numbers on the
same hardware (ratio below 1 is faster):

| Shape | | mono-ep | prime-rl | ratio | NCCL control |
|---|---|---:|---:|---:|---:|
| 30B-A3B | forward | 3.277 | 3.9 | 0.840 | 3.272 |
| 30B-A3B | backward | 6.601 | 8.2 | 0.805 | 40.653 |
| 30B-A3B | checkpointed 2F+B | 12.317 | 15.9 | 0.775 | 46.490 |
| 235B-A22B | forward | 8.447 | 9.5 | 0.889 | 8.445 |
| 235B-A22B | backward | 21.124 | 24.6 | 0.859 | 90.498 |
| 235B-A22B | checkpointed 2F+B | 37.333 | 43.7 | 0.854 | 107.124 |

Run ap-V06KM1a7K9KSAOsvOo4CU4 (one allocation, `NVIDIA H200` asserted in-process).

Skewed routing (one rank owns the hot experts) costs 26% on the 30B 2F+B (15.504 ms) and
28% on the 235B 2F+B (47.799 ms) on the same build; the NCCL control moves by 5-8%.

8xH100 (strict `H100!`), 30B-A3B, balanced, checkpointed 2F+B, one allocation per row:

| Build | Forward | Backward | Checkpointed 2F+B | NCCL control 2F+B |
|---|---:|---:|---:|---:|
| epilogue integration (start of the rounds) | 4.297 | 9.026 | 16.958 | 51.682 |
| round 2b | 3.365 | 8.157 | 14.327 | 49.831 |
| round 3 | 3.255 | 7.534 | 13.480 | 49.911 |
| round 4 (this tree's SM90 defaults) | 3.385 | 7.036 | 12.998 | 49.647 |
| this branch, gate run | 3.458 | 7.266 | 13.021 | 49.616 |

Runs, top to bottom: ap-optFhUPUcC5ywxp9zJPPRe, ap-EsrRmakdPLt73RGAS69kK4,
ap-ICtWLrsXD20dMONnCSd4Nz, ap-9N7kM7V67pDfcDRXl81ZP5, ap-05NnV2Ph9NslFfBVYrnbLq. Skewed
routing on the gate run: 4.028 / 8.996 / 16.198 ms, control 52.146 (ap-tsjPCzmlgpvljSIRdiLOVi).

8xB200, 30B-A3B, BF16, balanced, max-rank median ms, weight versions bumped every iteration:

| Build | Forward | Backward | Checkpointed 2F+B | Peer control | NCCL control |
|---|---:|---:|---:|---:|---:|
| start of the B200 rounds (6 backward stages) | 2.36 | 4.73 | 9.17 | 17.89 | 25.00 |
| round 1 | 2.19 | 4.40 | 8.31 | 17.39 | 24.53 |
| round 2 | 2.137 | 4.256 | 7.845 | 17.260 | 24.537 |
| round 3 (this tree's SM100 defaults) | 2.112 | 4.138 | 7.659 | 17.196 | 24.531 |
| round 3 with 16 reserved SMs (the FSDP2 setting) | 2.114 | 4.272 | 7.823 | 17.391 | 24.670 |
| this branch, gate run | 2.098 | 4.172 | 7.660 | 17.046 | 24.242 |

Runs, top to bottom: ap-6Js1FDCUJz1HfsUbBOqL3b, ap-PFK8nCymI9bow1IChnsdrD,
ap-VBok02Nuf6psSFDeShXEHF, ap-rxwVgU6xxIwEFdt6fNcX3o (both round-3 rows),
ap-afG15T7ayDA5cPEJnqpPcC.

What these do not measure: the H100 and B200 rows are not hardware- or precision-matched
with the prime-rl H200 FP8 figure; the controls share the persistent forward, so they are
not prime-rl's complete baseline; absolute timings move by up to 16% between allocations of
the same GPU type, so rows from different allocations are reproductions rather than A/Bs;
and these are layer timings, not end-to-end training throughput. Correctness rests on the
two-GPU FP32-reference and FP8-combine validators (ap-VN6mzQm22BwbfnE68DZ5rO,
ap-wufiJKL44xoazIZkfVaBQv, ap-gK4emj87Im7xQLO4dvCUXF on 2xH100), the four-GPU FSDP2
weight-cache contract test with optimizer steps (ap-TCBnqZV1beO55aH6SDxtBO on 4xH100,
ap-mkqekSW1wzK9nne3lsmOne on 4xB200 with 24 reserved SMs), the bitwise parity of every
default against its restore value, and the Axolotl parity test
(`tests/e2e/multigpu/test_ep_mono_ep.py`); long-training
convergence has only the round-2a loss-curve check (2xH100, 739M Qwen3-MoE, 300 steps, two
seeds).

## License

New code in this package is Apache-2.0 (`LICENSE`). `vendor/deepgemm` keeps DeepSeek's MIT
license and `vendor/sonic-moe` SonicMoE's Apache-2.0 license; both are shipped with the
package (`license-files` in `pyproject.toml`). Build dependencies fetched by
`tools/fetch_build_dependencies.py` (CUTLASS, DeepJIT) keep their own licenses. No UniEP
code is included. Revisions and the list of local modifications are in `vendor/sources.json`.
