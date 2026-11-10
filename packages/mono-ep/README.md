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
  trainer. Full fine-tuning only; adapters and quantized loading fall back.

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

## Benchmarks

Qwen3-30B-A3B layer, expert parallel 8, 16,384 tokens per GPU, FP32 masters, `fullgraph`
compiled, max-rank median ms over fifteen iterations; "2F+B" is the checkpointed layer step
(forward, recomputed forward, backward), router and optimizer excluded.

| Hardware | Precision | Shape | Forward | Backward | 2F+B | prime-rl 2F+B (8xH200 FP8) |
|---|---|---|---:|---:|---:|---:|
| 8xH200 | blockwise FP8 | 30B-A3B | 3.277 | 6.601 | 12.317 | 15.9 |
| 8xH200 | blockwise FP8 | 235B-A22B | 8.447 | 21.124 | 37.333 | 43.7 |
| 8xH100 | blockwise FP8 | 30B-A3B | 3.385 | 7.036 | 12.998 | not matched hardware |
| 8xB200 | BF16 | 30B-A3B | 2.112 | 4.138 | 7.659 | not matched precision |

The same-run NCCL all-to-all control takes 46.5 ms (8xH200, 30B) and 24.5 ms (8xB200).
Forward/backward splits per tuning round, skewed routing, the controls, the Modal run of
every row, the validators and what these numbers do not measure are in
[docs/benchmarks.md](docs/benchmarks.md).

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
- Prime Intellect's prime-rl fused MoE kernels: the op contract this package follows and the
  published 8xH200 numbers used as the comparison target.

## License

New code in this package is Apache-2.0 (`LICENSE`). `vendor/deepgemm` keeps DeepSeek's MIT
license and `vendor/sonic-moe` SonicMoE's Apache-2.0 license; both ship with the package.
Build dependencies fetched by `tools/fetch_build_dependencies.py` keep their own licenses.
