# What was added, and what is vendored

`vendor/deepgemm` is DeepSeek's [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) at
revision `5081f851` (MIT), without its tests and CI, built against the CUTLASS and DeepJIT
revisions pinned in `vendor/sources.json`. Everything under
`csrc/jit_kernels/impls/*mono_ep*`, `csrc/jit_kernels/heuristics/mono_ep.hpp`,
`csrc/apis/mono_ep.hpp` and `deep_gemm/include/deep_gemm/{impls,layout,scheduler}/*mono_ep*`
is new code in this tree. By theme:

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
  selector, the dispatch pull sizing (`mono_ep_dispatch_pull.hpp`), the combine chunk
  sizing (`mono_ep_combine.cuh`), exclusive-slot scatter-store epilogues, and the grouped
  GEMM entry points with in-kernel readiness and GPU row bounds.
- **Host side** (`src/mono_ep`): the `torch.library` ops and fake implementations, the
  `PersistentMoE` runtime and `MonoEPRuntime` API, the NCCL reference backward, the weight
  layout cache and its FSDP2 contract, SM90 FP8 weight forms and fused token preparation
  (Triton), the quantization primitives, and the older NCCL chunk pipeline and SonicMoE
  integration the prototype started from.

`src/mono_ep/_vendor/sonic_fp8.py` is an extract of SonicMoE's functional entry points
(Apache-2.0, revision in `sources.json`) with the modifications listed there; it is only used
by the experimental Sonic FP8 path. `src/mono_ep/include` holds the CUDA sources of an earlier
SM100 owned-backward extension (`owned_backward.py`, built with `cpp_extension`).

## Vendor mapping

Upstream names its fused-MoE feature with a prefix that this tree spells `mono_ep` /
`MonoEP` everywhere: file names, C++ and Python identifiers, pybind names. The mapping is
applied by `tools/rename_vendor_names.py`, which assembles the upstream spelling from
fragments so the tree itself stays free of it; after moving the DeepGEMM pin, run it on
`vendor/deepgemm` before re-applying the local modifications. `sources.json` records the
mapping under `deepgemm.vendored_name_mapping`.

## `vendor/sources.json`

| Entry | Fields |
|---|---|
| `deepgemm` | `url`, `revision`, `license`, `excluded` (tests and CI), `local_modifications`, `dependencies` (CUTLASS and DeepJIT revisions fetched by `tools/fetch_build_dependencies.py`), `design_references` (TeraMoE, DeepEP), `vendored_name_mapping` |
| `sonic_fp8` | `url`, `revision`, `license`, `source`, `destination`, `local_modifications` |

`deepgemm.local_modifications` lists the changes relative to upstream up to the epilogue
integration; the tuning-round changes (fused weight quantization, tail scheduling, the SM100
pipeline work) are summarised above and still have to be appended to that list.
