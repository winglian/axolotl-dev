# mono-ep

Fused expert-parallel MoE training kernels for Hopper and Blackwell. Each direction of a
mixture-of-experts layer runs as one persistent kernel per GPU: it pulls the tokens routed to
its experts over NVLink symmetric memory, runs the gate/up GEMM, the SwiGLU and the down GEMM,
and scatters the weighted results back to the owning ranks without leaving the kernel. The
backward is the same kernel in reverse, with the gate/up projections recomputed instead of
saved. The kernels are JIT-compiled through a vendored fork of DeepGEMM; the Python package
wraps them as `torch.library` custom ops with fake implementations, so a whole MoE layer
compiles with `fullgraph=True`. The package has no Axolotl, Transformers or TRL dependency.

- Hardware: H100 and H200 (SM90), B200 (SM100); every rank of the expert-parallel group on
  one NVLink node. SM120 (RTX PRO 6000, RTX 50-series) and older are not supported.
- Precisions: SM90 `bf16` and blockwise `fp8` (E4M3 forward and backward); SM100 `bf16` and
  `mxfp8` (forward). `auto` picks `bf16` on SM90 and `mxfp8` on SM100.
- Interfaces: Axolotl's `expert_parallel_backend: mono_ep`, or `MonoEPRuntime` from any
  trainer. The fused kernels need plain SwiGLU experts and full fine-tuning; the dispatch mode
  (`expert_parallel_mono_ep_mode: dispatch`) runs mono-ep's peer transfers around Axolotl's
  local experts kernel, so adapters and quantized experts work there.
- Memory: the backward pools and both rings are sized from the rows each step actually routes
  (`MONO_EP_WORKSPACE_BOUND=auto`, Axolotl `expert_parallel_mono_ep_workspace_bound: auto`).
  Resident per rank at EP8 on B200, 16,384 tokens per GPU: 4.9 / 7.9 / 9.0 GB at the
  Qwen3-30B-A3B / Qwen3-235B-A22B / Beam shapes instead of 18.1 / 35.5 / 41.0 GB, with
  bitwise-identical dx; SM90 BF16 and blockwise FP8 take the same bounds.
- SM100 backward: activation-checkpoint recomputes store the gate/up pre-activations and the
  backward skips recomputing them at wide experts (`MONO_EP_STORED_CONTEXT=auto`, Axolotl
  `expert_parallel_mono_ep_stored_context`; Beam checkpointed layer 51.7 -> 45.6 ms on 8xB200),
  and the dW grouped GEMMs write their output directly (`MONO_EP_SM100_DW_DIRECT=1`). The MXFP8
  backward and MXFP8 dW (`MONO_EP_SM100_BACKWARD_PRECISION`, `MONO_EP_SM100_DW_PRECISION`) are opt-in.

## Installation

Requirements: CUDA 12.9 or newer (the benchmarks use 13.0), PyTorch 2.9 or newer, Triton 3.4
or newer, a C++20 compiler with `<format>` (GCC 13), and the ELF/DWARF headers DeepGEMM's
build wants (`libelf-dev`, `libdw-dev`).

```bash
pip install -e packages/mono-ep
python packages/mono-ep/tools/fetch_build_dependencies.py
DG_USE_LOCAL_VERSION=0 DG_FORCE_BUILD=1 CC=gcc-13 CXX=g++-13 \
  pip install --no-build-isolation --no-deps packages/mono-ep/vendor/deepgemm
```

`cicd/install_mono_ep.sh` at the repository root does the same inside the Axolotl CI images.
The kernels compile on first use and are cached under `DG_JIT_CACHE_DIR`; the first forward
and backward of a new shape take a few minutes.

To rebuild an existing tree (after pulling or editing `vendor/deepgemm`), remove the build
directory first, then run the same install:

```bash
rm -rf packages/mono-ep/vendor/deepgemm/build
DG_USE_LOCAL_VERSION=0 DG_FORCE_BUILD=1 CC=gcc-13 CXX=g++-13 \
  pip install --no-build-isolation --no-deps packages/mono-ep/vendor/deepgemm
```

The extension now lists its headers as build dependencies, but the clean build stays the safe
procedure: a reused object from before a header-only change surfaces at run time as
`incompatible function arguments` from a `deep_gemm._C` function. `DG_FORCE_BUILD=1` only makes
the install build from source instead of downloading a prebuilt wheel; it does not force
recompiling objects in an existing build directory. The JIT kernel cache needs no clearing:
its key hashes the generated kernel source, every header it includes (read from the installed
`deep_gemm/include`), the compiler flags and the compiler version, so an edited header or launcher
compiles a new kernel once it is installed. To reclaim space or start cold, remove
`$DG_JIT_CACHE_DIR` (default `~/.dj`).

## Quickstart

### Axolotl

The keys that select the fused backend, from
[`examples/expert_parallel/qwen3_30ba3b_ep_mono_ep_fft_8gpu.yaml`](../../examples/expert_parallel/qwen3_30ba3b_ep_mono_ep_fft_8gpu.yaml)
(Qwen3-30B-A3B full fine-tune, one node of eight GPUs):

```yaml
plugins:
  - axolotl.integrations.expert_parallel.ExpertParallelPlugin
expert_parallel_size: 8
expert_parallel_backend: mono_ep
expert_parallel_mono_ep_precision: auto

activation_checkpointing: true   # non-reentrant; reentrant is rejected with EP
fsdp_version: 2
fsdp_config:
  state_dict_type: FULL_STATE_DICT   # required with EP
```

```bash
axolotl train examples/expert_parallel/qwen3_30ba3b_ep_mono_ep_fft_8gpu.yaml
```

LoRA on a packed NVFP4 checkpoint (SM100) is the same backend with `adapter: lora`,
`use_sonicmoe: true` and `nvfp4_merge_aware: false`, which `auto` resolves to the
`nvfp4_lora` precision:
[`examples/expert_parallel/qwen3_30ba3b_nvfp4_lora_mono_ep.yaml`](../../examples/expert_parallel/qwen3_30ba3b_nvfp4_lora_mono_ep.yaml).

Every config key, the precision matrix, the fallback conditions and every environment
selector with its default and restore value are in [docs/configuration.md](docs/configuration.md).

### Other trainers

One `MonoEPRuntime` per process serves every MoE layer. Transformers models call a registered
experts function with exactly the runtime's inputs:

```python
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
from mono_ep import MonoEPRuntime, nccl_env

os.environ.update(nccl_env(reserved_sms=16))  # before init_process_group
runtime = MonoEPRuntime(
    group=ep_group,
    num_experts=128,
    max_tokens_per_rank=16384,
    top_k=8,
    hidden=2048,
    intermediate=768,
    precision="auto",
)
ALL_EXPERTS_FUNCTIONS.register(
    "mono_ep",
    lambda self, x, idx, w: runtime.forward(
        x.to(torch.bfloat16), idx, w.float(), self.gate_up_proj, self.down_proj
    ),
)
model = AutoModelForCausalLM.from_pretrained(model_id, experts_implementation="mono_ep")
```

[`examples/trl_sft_mono_ep.py`](examples/trl_sft_mono_ep.py) is the complete TRL `SFTTrainer`
script (expert slicing, DDP exclusion, runtime lifetime); the API and what the Axolotl seam
adds are in [docs/trl-integration.md](docs/trl-integration.md).

### NVFP4 LoRA (SM100)

LoRA on frozen packed NVFP4 experts runs through the package API
(`MonoEPRuntime(precision="nvfp4_lora").forward_lora(...)`,
[docs/trl-integration.md](docs/trl-integration.md#nvfp4-lora)) and from Axolotl with
`expert_parallel_mono_ep_precision: nvfp4_lora`. The merged NVFP4 weights are built just in time
per layer (`MONO_EP_NVFP4_MERGED_CACHE=layer`, the two most recent layers resident) and the
transposed backward forms are quantized from the merged tile (`MONO_EP_NVFP4_REQUANT=cuda_fused`):
Qwen3-30B-A3B-NVFP4 LoRA at EP4 on B200 peaks at 12.2 GiB per rank instead of 24.7 GiB.

## Benchmarks

Timing contracts, used as labels in every table:

- **(a)** forward + backward with the forward's context stored, no recompute.
- **(b)** forward + partial recompute (the gate/up projection only) + backward: Mixture of
  Kittens' checkpointed mode (`recompute_forward_context`).
- **(c)** forward + the full forward recomputed + backward: Axolotl's non-reentrant activation
  checkpointing (`activation_checkpointing: true`). Every mono-ep "2F+B" number here is (c), and
  so is the Mega MoE number Prime Intellect published.
- **(d)** forward + backward with the fused op's output saved under selective checkpointing
  (`activation_checkpointing: {selective: {save: ["mono_ep::persistent_forward"]}}`), the same as
  prime-rl's "selective" mode.

The Mega MoE rows compare (c) with (c). The Mixture of Kittens rows compare mono-ep's (c) with
MoK's (b), which charges mono-ep one more full forward than MoK. No (d) number appears in these
tables: none was measured in these allocations. Per layer, (c) keeps the layer input
(tokens x hidden BF16: 64 / 128 / 160 MiB at Qwen3-30B-A3B / Qwen3-235B-A22B / Beam with 16,384
tokens per GPU) plus the routing (1.5 MiB); (d) also keeps the output, twice that. Prime
Intellect publishes no memory numbers.

One MoE layer at expert parallel 8 (16,384 tokens per GPU, top-8), checkpointed layer step
(forward, recomputed forward, backward) in max-rank median ms, balanced routing, router and
optimizer excluded, one Modal allocation per GPU type. Qwen3-30B-A3B is hidden 2048 /
intermediate 768 / 128 experts, Qwen3-235B-A22B hidden 4096 / intermediate 1536 / 128 experts,
Beam hidden 5120 / intermediate 2560 / 256 experts. FP8 on Hopper, BF16 on B200, FP32 masters,
`fullgraph` compiled:

| Hardware | Shape | mono-ep, (c) (ms) | Routed TFLOPS per GPU (% of dense peak) | NCCL all-to-all backward, same run, (c) (ms) | Reference (ms) |
|---|---|---:|---:|---:|---|
| 8xH200 | 30B-A3B | 12.63 | 392 (20%) | 46.67 | Mega MoE (Prime Intellect, published), (c): 15.9 |
| 8xH200 | 235B-A22B | 38.30 | 517 (26%) | 107.52 | Mega MoE (Prime Intellect, published), (c): 43.7 |
| 8xH200 | Beam | 79.50 | 519 (26%) | 190.40 |  |
| 8xH100 | 30B-A3B | 13.55 | 365 (18%) | 49.91 |  |
| 8xH100 | 235B-A22B | 42.73 | 463 (23%) | 113.76 |  |
| 8xH100 | Beam | 90.86 | 454 (23%) | did not fit (80 GB) |  |
| 8xB200 | 30B-A3B | 7.79 | 635 (28%) | 24.89 | MoK BF16 / MXFP8, same run, (b): 6.96 / 6.40 |
| 8xB200 | 235B-A22B | 21.65 | 914 (41%) | 65.35 | MoK BF16 / MXFP8, same run, (b): 21.05 / 17.06 |
| 8xB200 | Beam | 49.29 | 837 (37%) | 113.38 | MoK BF16 / MXFP8, same run, (b): 40.63 / 30.33 |

Mega MoE is Prime Intellect's published prime-rl measurement on 8xH200 with the same shapes,
routing and FP8 precision, not a rerun. Mixture of Kittens (MoK) ran in the same B200
allocation with its best knobs; its times include a shared expert (1/8 more FLOPs) that
mono-ep does not compute. Its (b) time is below mono-ep's (c) time at every shape, by
0.6-0.8 ms at the Qwen3 shapes and 8.7 ms at Beam, while (c) charges mono-ep one more full
forward; mono-ep's forward alone is faster at all three shapes, and a comparison under one
contract has not been measured. TFLOPS are routed GEMM FLOPs over
the whole layer time against dense peaks of 1,979 (FP8, Hopper) and 2,250 (BF16, B200).

The same layer through Axolotl's expert-parallel backends with SonicMoE as the local experts
kernel and BF16 weights, every column interleaved in one process (the fused column is the same
kernels uncompiled, with parameters quantized per call):

| Hardware | Shape (all (c), ms) | mono-ep fused | mono-ep dispatch (bounded) + SonicMoE | DeepEP + SonicMoE, best SM count | DeepEP + SonicMoE, 20 SMs (Axolotl default) | torch all-to-all + SonicMoE |
|---|---|---:|---:|---:|---:|---:|
| 8xH200 | 30B-A3B | 13.41 | 19.09 | 18.12 (40 SMs) | 20.10 | 22.09 |
| 8xH200 | 235B-A22B | 42.19 | 46.67 | 45.87 (40 SMs) | 47.31 | 50.28 |
| 8xH200 | Beam | 88.56 | 85.09 | 85.24 (40 SMs) | 85.86 | 89.49 |
| 8xH100 | 30B-A3B | 14.07 | 19.66 | 26.94 (40 SMs) | 28.56 | 22.70 |
| 8xH100 | 235B-A22B | 44.62 | 48.85 | 51.79 (40 SMs) | 53.73 | 52.69 |
| 8xH100 | Beam | 94.69 | 87.57 | 89.21 (40 SMs) | 91.11 | 91.92 |
| 8xB200 | 30B-A3B | 7.77 | 11.53 | 12.71 (64 SMs) | 21.74 | 14.13 |
| 8xB200 | 235B-A22B | 21.25 | 25.52 | 26.39 (64 SMs) | 40.82 | 29.12 |
| 8xB200 | Beam | 45.72 | 45.35 | 44.45 (64 SMs) | 60.77 | 49.71 |

mono-ep's lead over the dispatch paths shrinks as the expert GEMMs grow; at the Beam shape every
path is within a few percent. The H100 allocation had slow NVLink, which penalized DeepEP most,
so the H200 rows are the clean Hopper comparison. Skewed routing, forward/backward splits,
the SM100 opt-ins, the ring floor, NVFP4 LoRA, the Modal run behind every number and what these
do not measure are in [docs/benchmarks.md](docs/benchmarks.md).

## How it works

Dispatch warps pull routed rows over NVLink into a TMA pipeline while the expert GEMMs run
(WGMMA on SM90, UMMA with tensor memory on SM100); SwiGLU, top-k weighting and the FP8
quantization of the activations are fused into the epilogues, and the down GEMM's epilogue
scatters each row into an exclusive combine slot on its owner. The backward restages `x` and
`dy`, recomputes the gate/up projection, and runs the weight gradients as grouped GEMMs on
GPU-resident offsets, so nothing from the forward is retained and no host sync is needed.
Weight layouts are built once per optimizer step by a cache with an explicit FSDP2
invalidation contract, and the runtime reserves SMs so FSDP's NCCL collectives can run
alongside the spinning kernels. Details: [docs/architecture.md](docs/architecture.md).

## What was added, and what is vendored

`vendor/deepgemm` is DeepSeek's DeepGEMM at revision `5081f851` (MIT) built against pinned
CUTLASS and DeepJIT revisions; the persistent dispatch, GEMM and combine kernels for SM90
(FP8 and BF16) and SM100 (BF16, MXFP8 forward), the readiness-queue scheduler, the layouts
and the host-side package are new code in this tree. `vendor/sources.json` lists every
revision and local modification; the themes and the vendor name mapping are in
[docs/vendored-changes.md](docs/vendored-changes.md).

## Credits

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (DeepSeek, MIT), with
  [DeepJIT](https://github.com/deepseek-ai/DeepJIT) and NVIDIA
  [CUTLASS](https://github.com/NVIDIA/cutlass): the JIT framework, GEMM mainloops, TMA and
  UMMA helpers and grouped GEMM APIs the kernels are built from.
- [DeepEP](https://github.com/deepseek-ai/DeepEP): the intranode symmetric-memory
  dispatch/combine model this design is compared against; no DeepEP code is used.
- [TeraMoE](https://github.com/PFCCLab/TeraMoE) (PFCCLab, MIT): the BF16 chunk queue and
  scheduler design the readiness queue adapts; no code copied.
- [SonicMoE](https://github.com/bdepyzy/sonic-moe) (Apache-2.0), with
  [quack](https://github.com/Dao-AILab/quack) and the CUTLASS DSL: the local expert kernels
  the prototype measured against and the extracted FP8 functional path.
- Prime Intellect's Mega MoE kernels in prime-rl: the op contract this package follows and the
  published 8xH200 numbers used as the comparison target.

## License

New code in this package is Apache-2.0 (`LICENSE`). `vendor/deepgemm` keeps DeepSeek's MIT
license and `vendor/sonic-moe` SonicMoE's Apache-2.0 license; both ship with the package.
Build dependencies fetched by `tools/fetch_build_dependencies.py` keep their own licenses.
