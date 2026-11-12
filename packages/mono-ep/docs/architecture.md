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
no host size sync. Those GEMMs reduce over each expert's pool rows padded to 128, so on SM100
the last recompute epilogue of an expert zeroes the x/dy tail rows of its last block; stale
rows from an earlier launch, possibly non-finite, would otherwise reach dW as 0 x NaN. The last wave's dX tiles are split across idle CTAs
(`MONO_EP_SM90_FP8_TAIL_SCHEDULE`) so the CTA tail is short.

On SM90 FP8 the x/dy transport, the dX combine and the weight-gradient operands are all E4M3
with 128-element scale groups (1x128 activations, 128x128 weights; `MONO_EP_SM90_FP8_TRANSPORT=0`
keeps the transport and combine BF16); outputs are BF16 and
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
- **Load producers before a consumer can spin.** Every producer kernel (CUDA, Triton, and
  helpers such as `torch.cuda._sleep`) must have been launched once in the process before a
  consumer that spins on it is enqueued. With CUDA lazy loading, the first launch of a kernel
  loads its module, and the load waits for the kernels already running, including a consumer
  spinning on that same producer; the consumer then traps on its spin timeout
  (`ReadyScheduler`, `scheduler/gemm.cuh`, about 15 s; the combine wait and epoch barrier
  have their own). Across ranks the same thing forms a cycle when every rank enqueues its
  consumer before a first-launch producer a peer waits on. The symptom is `CUDA error:
  unspecified launch failure` (719) at an unrelated later launch, with compute-sanitizer
  reporting only a `Trace/breakpoint trap`. The persistent kernels satisfy the rule by
  construction: each phase is one launch per rank on one stream, and peers wait only on
  the same launch on other ranks. `PeerSchedule` runs its first forward and backward
  without overlap (send, barrier, calculate, barrier, consume) before overlapping streams,
  and still enqueues producers ahead of consumers. Tests and examples that delay a producer
  launch it, and `torch.cuda._sleep(1)`, once before the timed loop.
  `CUDA_MODULE_LOADING=EAGER` can help confirm a suspected case but is not the fix: it does
  not cover Triton kernels, which load on first launch regardless.

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

## Backward ring floor

The owned backward keeps one pool block live per owning cluster through all of its phases, but
its ring was sized by the forward's interleaved schedule. Wide-expert shapes (hidden 5120) then had fewer ring blocks than owners
(30 blocks against 132 CTAs on H100 and 74 pairs on B200) and serialized on ring slots. The
backward now takes a floor of one wave of owners, `num_sms x 128` tokens on SM90 and
`num_sms / 2 x 128` on SM100 (`MONO_EP_BACKWARD_RING_WAVES`, `0` restores the forward sizing;
`MONO_EP_NVFP4_BACKWARD_RING_WAVES` for the NVFP4 path). The floor only raises the ring, so the
Qwen3-30B-A3B shape keeps its ring and its kernels; at the Beam shape (8,192 tokens per GPU) the
8xB200 BF16 backward went from 25.7 to 16.7 ms and the 8xH100 blockwise FP8 backward from 70.1 to
37.6 ms. The ring size is a template parameter, so a changed ring compiles a
new kernel.

## In-kernel weight gradients (SM100, opt-in)

`MONO_EP_SM100_INKERNEL_DW=1` (option bit 512) computes dW1 = dz^T x and dW2 = dy^T h inside
the persistent backward instead of as grouped GEMMs after it. Each task is a 256 x 128 FP32
tile (256 x 256 with `_WIDE=1`, bit 1024) reduced over all of one expert's pool rows. The
scheduler interleaves dW tiles between M-blocks by readiness: per expert, a counter of finished
h/dz M-block halves is released after the recompute epilogue and acquired by the TMA loaders,
and after each M-block the scheduler claims up to `_PER_BLOCK` ready tiles before draining the
rest. The pool operands are read MN-major through the existing 64 x 64 pool boxes, the epilogue
stores FP32 straight from TMEM, and the dW pointers and counters ride in an unused tensor-map
slot. At most 32 local experts. It pays at wide shapes
(-1.49 ms per backward with wide tiles at the Beam shape, 8xB200) and costs about 0.1 ms per
backward at the Qwen3-30B-A3B shape, so it stays opt-in.

## MXFP8 backward (SM100)

`MONO_EP_SM100_BACKWARD_PRECISION=mxfp8` (option bit 2048) runs dh, the z recompute and dx as
`kind::mxf8f6f4` GEMMs with E4M3 values and UE8M0 scales per 32 K elements, at BLOCK_K 128. The
sender quantizes x and dy per token, so NVLink carries 4 KB of E4M3 rows plus 128 bytes of
scales per routed row instead of 8 KB of BF16. The dispatch warps store the E4M3 pools and
UTCCP-ordered scale pools and write exact BF16 copies for the unchanged BF16 weight-gradient
GEMMs; dz is quantized in the staged epilogue stores. The scale factors are copied into TMEM
after the accumulators, as in the MXFP8 forward. The E4M3 weight forms (W1 interleaved, W2^T,
W1^T) are built in one Inductor graph and cached under the weight-cache epoch contract. It is
not bitwise (about 6-7% relative error against an FP32 reference, BF16 0.2%; the loss curve
stayed within a tenth of the seed gap) and saves about 0.4 ms per backward at the
Qwen3-30B-A3B shape on 8xB200, so it stays opt-in for the BF16 runtime. It does not combine
with in-kernel dW: both use the kernel's unused shared-expert operand slots and they need
different BLOCK_K.

`precision: mxfp8` (the SM100 `auto`) pairs the MXFP8 forward with this backward. The forward's
interleaved W1 is the backward's `sm100_mx_w1` form (the same bytes), so the first forward of a
step builds W1, W2^T and W1^T in one pass, plus the forward's W2, and the checkpoint recompute
and the backward hit the cache. The backward still quantizes x and dy into its own transport
buffer (the forward's E4M3 rows live in the forward ring, which the recompute overwrites), and it
recomputes z: the MXFP8 forward does not write the compact stored-z layout. Hidden sizes that are
not multiples of 512 fall back to the BF16 persistent backward with one log line. Before this the
MXFP8 runtime's backward was the generic NCCL path, about ten times slower.
With MXFP8 dW the kernel writes no BF16 x / dy / dz copies, and by default
(`MONO_EP_SM100_BF16_POOLS=lazy`) the backward does not allocate them either: they appear at the
first backward that writes them (BF16 dW, the BF16 backward, LoRA on BF16 pools), and the launcher
refuses empty copies for any launch that would. That frees 1.8 / 3.7 / 5.1 GiB per rank at the
30B-A3B / 235B-A22B / Beam EP8 shapes.
This is what `precision: auto` runs on SM100. MXFP8 dW is the default there
(`MONO_EP_SM100_DW_PRECISION=bf16` restores BF16 dW), and stored-context
`auto` does not store for an MXFP8 backward on either runtime: its recompute is one FP8 GEMM and
costs less than gathering the scattered BF16 z (see
[configuration.md](configuration.md#axolotl-config-keys)).

## NVFP4 LoRA path and the CUDA requant (SM100)

`precision="nvfp4_lora"` keeps the experts as frozen NVFP4 weights with LoRA adapters. Each
optimizer step rebuilds merged NVFP4 weights (base plus the scaled LoRA delta, requantized) in a
requant cache, and the persistent forward and backward run `kind::mxf4nvf4` GEMMs on them; the
adapter gradients are BF16 grouped GEMMs on the pools. The rebuild is an SM100 CUDA kernel
(`MONO_EP_NVFP4_REQUANT=cuda`, the default), which writes the same bytes as the Triton reference
in two paired passes (each weight with its transposed form) and cut the Beam rebuild from 10.3
to 3.2 ms per layer per step on 8xB200. `cuda_fused` builds the transposed forms from the
merged tile, which is faster still but changes `w1t` / `w2t` numerics, so it is opt-in.
`nvfp4-lora-design.md` has the design; the requant source is vendored under
`src/mono_ep/nvfp4_requant/` until the Axolotl copy and this one are reconciled.

The NVFP4 backward's Dh and recompute tasks skip the per-task fence: each CTA fences once at its
block's last recompute tile and adds all its recompute tiles to the ring's empty count there
(NVFP4 options bit 2048, the default; `MONO_EP_SM100_NVFP4_BACKWARD_OPTIONS=3` restores the
per-task fences). This works because the NVFP4 backward always gives a CTA pair whole blocks. A
cross-M-block pipeline that overlapped one block's Dx with the next block's recompute removed the
Dx wait but did not shorten the backward, so it was not merged.

## Fused LoRA on BF16 experts (`bf16_lora`, `mxfp8_lora`, `fp8_lora`)

The other precisions train a LoRA adapter on frozen BF16 experts with the full fine-tune
kernels, unchanged. `mono_ep.lora` holds what is shared with `nvfp4_lora`: the
`MergedWeightCache` (per layer, `layer` or `all` residency, invalidated by an optimizer post-step
hook) and the rank-r adapter gradients on the backward's pools. Per step and layer
`LoraExperts` merges `bf16(W + scaling * B_e @ A_e)` with one `baddbmm` per projection (the
factors enter in BF16, as for the NVFP4 requant) from the base FSDP2 has gathered for that
forward or backward. The merged tensors go through the same forward and native backward as a
full fine-tune, so the kernel layouts and the MXFP8 / blockwise FP8 forms come from the
weight-layout cache keyed by the merged storage; the bounded workspace records its receive
counts under the frozen base instead, which stays put across merges and regathers.

The backward skips the dW GEMMs (`NativePersistentBackward.backward(weight_grads=False)`) and
returns the BF16 x / dy / h / dz pools, whose padded tails the kernels zero; `lora_weight_grads`
computes `Q = X Aᵀ`, `P = dZ B`, `dA = s Pᵀ X`, `dB = s dZᵀ Q` (and the same for W2 with dY and
H) as eight grouped GEMMs. The full dW is never formed. The MXFP8 backward is launched with BF16
dW for this, so it writes the BF16 pools (x and dy are the dequantized MXFP8 operands the kernel
used). Under `mxfp8_lora` the merged BF16 weight is never written: the CUDA MXFP8 forms builder
loads each 128 x 128 base tile, adds `scaling * B @ A` for it on the tensor cores in shared memory
and quantizes the merged tile into all four forms (`MONO_EP_LORA_MXFP8_FORMS=fused`, default). The
SM90 blockwise FP8 backward keeps only FP8 operand copies, so `fp8_lora` instead
projects the full FP8 dW the Hopper wgrad already computes (`dA = s Bᵀ dW`, `dB = s dW Aᵀ`). The
forward and backward run as the `mono_ep::lora_forward` / `lora_backward` custom ops, so
selective activation checkpointing saves the fused output exactly as it does for
`persistent_forward`.

The adapter gradients default to two Triton passes on SM100 (`mono_ep.lora_grads`: rank-r
projections per 128-row block, then FP32 row contractions per expert). `mxfp8_lora` feeds them
the MXFP8 backward's E4M3 x / dy / dz pools instead of BF16 copies by default
(`MONO_EP_LORA_MXFP8_POOLS=fp8`; `bf16` restores the copies): the E4M3 bytes are read as int32 words
and each byte lane is its own K slice of the dot product, decoded by placing the byte's bits in an
FP32 and two power-of-two multiplies. With the lazy BF16 pools the copies are then never allocated
(1.8 / 3.7 / 5.1 GiB per rank at the EP8 shapes, about 1 % step time at 30B-A3B / 235B-A22B), and
the adapter gradients sit at the MXFP8 full fine-tune dW envelope.

Declined: computing the rank-r projections in the backward epilogue. Of the four projections only
`P1 = dZ B1` and `Q2 = H A2ᵀ` see their pool operand in an epilogue (the recompute epilogue holds
dZ and H tiles); `X` and `dY` exist only as GEMM A operands and in the dispatch warps, and the four
row contractions read the pools again in any case. Appending the factors as extra weight rows to
the existing GEMMs instead costs whole 256-row UMMA tiles under swap-AB (one extra tile per 6-20),
more than it saves. Measured on one B200 at the EP8 pools (131,072 rows, rank 16) the eight
Python-side GEMMs take 0.86 / 1.57 / 1.98 ms per layer at 30B-A3B / 235B-A22B / Beam, 60-73 % of
their two-pass bandwidth floor, and the epilogue-reachable P1 + Q2 0.18 / 0.30 / 0.43 ms, about 2-3 %
of the SAC step before any cost on the backward's critical epilogue warps.

## Dispatch mode and the bounded compact receive

The dispatch mode (`expert_parallel_mono_ep_mode: dispatch`, `MonoEPDispatchRuntime`) runs only
the peer transfers and leaves the expert GEMMs to the configured local kernel, so it serves the
same dispatch/combine contract as the Axolotl `deep_ep` and `torch` backends. Dispatch publishes
each rank's routed rows and pulls them from the peers' symmetric buffers with bandwidth-shaped
streaming kernels (32 KB tiles, 16-byte vectors, one peer per program); combine pushes the
partial rows back and reduces them in one fused FP32 pass. The fixed layout hands the local
kernel `ep_size x tokens_per_rank` rows. The compact receive (`expert_parallel_mono_ep_compact`)
hands it only the routed rows, which costs one device-to-host count per layer. The bounded
receive (`expert_parallel_mono_ep_bounded`) removes the stall on that count: it receives into a
bound from each layer's high-water routed count, filled to SonicMoE's own row bucket, copies the
exact count to pinned memory behind the dispatch, and reads it only after the local kernel has
been issued. Outputs are bitwise those of the compact receive; a count above the bound re-pulls
and re-runs the layer exactly, and the bound is only updated outside the backward so checkpoint
recomputes see their forward's bound. `compose-design.md` has the design and the measurements.
