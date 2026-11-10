# Architecture

## Forward: one persistent kernel per GPU

1. **Dispatch.** Every rank writes its routed rows into its own symmetric buffer (one per
   process, sized for `max_tokens_per_rank`) and publishes per-expert counts. Each CTA (SM90)
   or CTA pair (SM100) owns M-blocks of 128 routed rows of one local expert and pulls them
   from the owning ranks over NVLink in fixed-size chunks (`*_DISPATCH_PULL_BYTES`), spinning
   on readiness flags rather than returning to the host. The dispatch warps issue the pulls
   ahead of the GEMM: a 2-5 stage TMA pipeline on SM90, a lookahead pull on SM100.
2. **Expert GEMMs.** The gate/up GEMM runs from the pulled tiles (WGMMA on SM90, UMMA with
   tensor memory on SM100). SwiGLU and the top-k weighting are fused into the epilogues; on
   SM90 FP8 the activations are quantized in the epilogue with per-row scales and the down
   GEMM consumes them directly.
3. **Combine.** The down GEMM's epilogue scatters each row's weighted contribution back into
   the owner's combine slot. Slots are exclusive, so the stores are plain BF16 writes with no
   remote atomics, and completion is fenced per rank.

The readiness-queue scheduler that hands M-blocks to CTAs and tracks which chunks have landed
is adapted from TeraMoE's BF16 chunk-queue design (`scheduler/mono_ep.cuh`).

## Backward: the same kernel in reverse

Nothing from the forward is retained; the forward output is what activation checkpointing
saves. The backward kernel restages `x` and `dy`, recomputes the gate/up projection per owned
M-block, runs the SwiGLU derivative and the score-gradient reduction in the epilogue, produces
`dh` and the `dz` the weight gradients need, and scatters `dx` back to the owning ranks. `dH`
stays CTA-local; `h` and `dz` are written to a per-rank workspace once per K-tile, and the
weight gradients run afterwards as grouped GEMMs on GPU-resident expert offsets, so there is
no host size sync. The last wave's dX tiles are split across idle CTAs
(`MONO_EP_SM90_FP8_TAIL_SCHEDULE`) so the CTA tail is short.

On SM90 FP8 the x/dy transport, the dX combine and the weight-gradient operands are all E4M3
with 128-element scale groups (1x128 activations, 128x128 weights); outputs are BF16 and
weight gradients come back in the master dtype. The SM100 BF16 backward uses two-CTA pairs,
a six-stage pipeline with two-row pulls, and a recompute SwiGLU-derivative epilogue that
emits `h`, `dz` and the score partials without storing `z`. An opt-in MXFP8 variant
(`MONO_EP_SM100_BACKWARD_PRECISION=mxfp8`) runs dh, the recompute and dx as block-scaled MXFP8 UMMA
with E4M3 x/dy transport, UTCCP-ordered scale pools written by the dispatch warps, dz quantized in
the epilogue, and BF16 weight gradients from exact dequantized copies.

The native backward requires `hidden % 256 == 0`, `intermediate % 256 == 0` and every rank of
the EP group on one host. Otherwise the same op runs the NCCL reference backward (an
all-to-all restage of `x` and `dy` and grouped GEMMs), which is correct and slower.

## Around the kernels

- **Custom ops.** `mono_ep::persistent_forward` and `mono_ep::persistent_backward` are
  `torch.library` ops with fake implementations and an integer runtime key; the Python layer
  does no host synchronization, so a MoE layer compiles with `fullgraph=True` and composes
  with non-reentrant checkpointing. Reentrant recompute cannot replay the op pair.
- **Runtime lifetime.** `MonoEPRuntime` builds one symmetric buffer per process, reserves
  `reserved_sms` SMs (DeepGEMM's SM budget becomes `physical - reserved_sms`) so NCCL
  collectives issued by FSDP during a persistent kernel have somewhere to run, and must be
  destroyed only after every graph that used it has finished. `PersistentMoE` is the
  fixed-geometry runtime behind it (`compute_dtype`, `backward_backend`,
  `retain_activations` exposed); `MonoEP` is the older engine that also selects the NCCL
  chunk pipeline and SonicMoE paths and caches runtimes per group and geometry.
- **Launcher.** The host launcher reads shape and architecture, picks stage counts and pull
  sizes that fit shared memory (`mono_ep_dispatch_pull.hpp` mirrors the SM90 shared-storage
  layouts and caps the pull at 2 KB at hidden 4096), and includes every selector in the JIT
  key.
- **SM90 FP8 weight forms and token prep.** The quantized weight forms (values and scales for
  w1, w1ᵀ, w2, w2ᵀ) are produced in one Triton launch and the token quantization and prefix
  preparation are fused; both are cached per optimizer step through the weight cache below.
  The transposed forms are quantized from transposed reads rather than transposed after the
  cast, because ptxas 12.9 miscompiles 8-bit register layout conversions
  ([upstream-issue-ptxas-fp8-transpose.md](upstream-issue-ptxas-fp8-transpose.md)).

## Weight layout cache

The kernels want weights in layouts the model does not store (BF16 casts, interleaved
gate/up rows, transposes, FP8 forms and scales). `weight_cache.py` builds them once per cache
epoch with Inductor-compiled cast/permute kernels and reuses them across forward, checkpoint
recompute and backward of a layer.

Contract: an entry belongs to one weight *storage* (the `UntypedStorage`, its data pointer,
the tensor's offset/shape/stride/dtype and autograd version) within one epoch, and is reused
by every call in that epoch over the same storage, including fresh views (`DTensor.to_local()`,
saved-tensor unpacks). The epoch ends, dropping every entry, when `invalidate_weight_caches()`
is called, when `set_weight_cache_epoch()` is called with a new value, or when any
`torch.optim` optimizer finishes `step()` (a global post-step hook, the safety net). FSDP2 is
why the contract exists: it all-gathers each step into the same persistent unsharded storage
under `_unsafe_preserve_version_counter`, so the parameter keeps its object, storage and
version while its values change. Callers whose weights are rewritten without a version bump
(FSDP2, checkpoint resume, weight sync) must invalidate after every such write; the Axolotl
seam does it from an optimizer post-step hook and sets the epoch per step. Until a caller
supplies an epoch the key also holds the data pointer, which misses whenever FSDP2 re-gathers
into a different address; with an epoch the pointer is dropped so forward, recompute and
backward share one build. All cached layouts are bitwise identical to computing them per
call. The FSDP2 validators (`examples/modal_fsdp2_*weight_cache*.py`) exercise the contract
with optimizer steps on `dp_shard x EP` meshes, including a stale-cache detector.

## Dispatch-only mode

A composable mode that runs only the dispatch and combine in the persistent kernels and
leaves the expert GEMMs to the caller (the contract the Axolotl `deep_ep` backend uses) is
developed on the `features/mono-ep-compose` branch with its own design note and measurements;
it is not part of this tree.
