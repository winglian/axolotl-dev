# Configuration

Everything a user sets, and every selector the kernels read. The plugin sets what the kernels
need; the environment variables exist for experiments and for restoring a previous default.

## Axolotl config keys

| Key | Values | Meaning |
|---|---|---|
| `expert_parallel_size` | `1` (off) or a divisor of the world size | Number of EP ranks; experts are split evenly across them. |
| `expert_parallel_backend` | `auto`, `deep_ep`, `torch`, `mono_ep` | Token dispatch backend. `auto` never picks `mono_ep`; set it explicitly. In the fused mode the local experts kernel (`use_sonicmoe`, `use_scattermoe`, `experts_implementation`) is not used; in the dispatch mode it is. |
| `expert_parallel_mono_ep_mode` | `auto` (default), `fused`, `dispatch` | `fused`: one persistent kernel per direction (plain SwiGLU experts; full fine-tuning, or a LoRA adapter on the experts through the `*_lora` precisions). `dispatch`: mono-ep's peer dispatch and combine around the configured local experts kernel, so DoRA and other adapters the fused LoRA path does not cover, quantized experts, biases, other activations and clamped SwiGLU work. `auto`: `fused` when the config and model allow it, else `dispatch`. |
| `expert_parallel_mono_ep_compact` | `false` (default), `true` | Dispatch mode: hand the local kernel only the routed receive rows instead of the fixed `ep_size x tokens_per_rank` layout, at the cost of one device-to-host count per layer (saved under activation checkpointing). |
| `expert_parallel_mono_ep_bounded` | `false` (default), `true` | With `compact`: receive into a bound from each layer's high-water routed count (filled to SonicMoE's row bucket) and read the count only after the local kernel is issued, so dispatch never stalls on the host. Bitwise equal to `compact`; a count above the bound re-runs that layer exactly. Restore: `false`. |
| `expert_parallel_mono_ep_precision` | `auto`, `bf16`, `mxfp8`, `fp8`, `fp8_bwd`, `nvfp4_lora`, `bf16_lora`, `mxfp8_lora`, `fp8_lora` | Compute precision of the fused kernels (table below). Master weights keep their dtype and weight gradients come back in it. With an expert LoRA adapter on a BF16 base, `auto` / `bf16` / `mxfp8` / `fp8` resolve to their `*_lora` precision. |
| `expert_parallel_mono_ep_workspace_bound` | `auto` (default), `max` | Fused mode: size the backward pools and the rings from each step's routed rows (`MONO_EP_WORKSPACE_BOUND`). `max` restores the worst-case sizing. |
| `expert_parallel_mono_ep_nvfp4_merged_cache` | `layer` (default), `all` | `nvfp4_lora` only: how long a layer's merged NVFP4 weights (base + LoRA, rebuilt after every optimizer step) stay resident. `layer` builds them just in time per layer and keeps the two most recent layers; `all` keeps every layer's forms for the step. Exported as `MONO_EP_NVFP4_MERGED_CACHE`. Restore: `all`. |
| `expert_parallel_mono_ep_fused_lora` | `true` (default), `false` | Fused mode with an expert LoRA adapter on BF16 experts: train it on the fused kernels (`bf16_lora`, `mxfp8_lora`, `fp8_lora`, below). `false` restores the dispatch mode for every adapter except `nvfp4_lora`. Restore: `false`. |
| `expert_parallel_mono_ep_lora_merged_cache` | `layer` (default), `all` | `bf16_lora` / `mxfp8_lora` / `fp8_lora`: how long a layer's merged BF16 weights (base + LoRA, rebuilt after every optimizer step) stay resident, like `expert_parallel_mono_ep_nvfp4_merged_cache`. `all` holds one BF16 copy of every layer's routed experts. Exported as `MONO_EP_LORA_MERGED_CACHE`. |
| `expert_parallel_fallback_on_unsupported` | `true` (default), `false` | When `mono_ep` cannot run (table below), fall back to `expert_parallel_backend: torch` with a warning, or raise. |
| `dp_shard_size` | | FSDP shard axis next to EP; `expert_parallel_size x dp_shard_size` must equal the world size. The EP group must stay inside one NVLink node. |
| `activation_checkpointing` | `true` | Non-reentrant checkpointing; reentrant checkpointing is rejected with EP because its recompute cannot replay the fused op. |
| `fsdp_config.state_dict_type` | `FULL_STATE_DICT` | Required with EP; `SHARDED_STATE_DICT` would keep only one EP group's experts. |

| Precision | SM90 (H100, H200) | SM100 (B200) |
|---|---|---|
| `auto` | `bf16` | `mxfp8` |
| `bf16` | yes | yes |
| `mxfp8` (MXFP8 forward and the fused MXFP8 persistent backward, sharing one cached set of MXFP8 weight forms; hidden sizes that are not multiples of 512 take the BF16 persistent backward, logged once) | no | yes |
| `fp8` (blockwise E4M3 forward and backward, the kernels timed in [benchmarks.md](benchmarks.md)) | yes | no |
| `fp8_bwd` (BF16 forward, FP8 backward) | not implemented | no |
| `nvfp4_lora` (frozen packed NVFP4 experts with BF16 LoRA adapters; hidden and intermediate multiples of 256; native backward only, one host) | no | yes |
| `bf16_lora` (frozen BF16 experts with LoRA adapters on the `bf16` kernels; native backward only, one host) | yes (single-H100 and 2xH100 validated) | yes |
| `mxfp8_lora` (the same on the `mxfp8` kernels, BF16 dW pools) | no | yes |
| `fp8_lora` (the same on the blockwise `fp8` kernels; adapter gradients projected from the FP8 dW) | yes (single-H100 and 2xH100 validated) | no |

`auto` on SM100 (full fine-tuning) resolves to `mxfp8`: the MXFP8 forward with the fused MXFP8
persistent backward. That backward recomputes the gate/up context (the MXFP8 forward has no
stored-context output, so `MONO_EP_STORED_CONTEXT` does not apply), computes the weight gradients
in MXFP8 (`MONO_EP_SM100_DW_PRECISION=bf16` restores BF16 dW: dW relative error against FP32 0.068
instead of 0.062, FSDP2 five-step floor 0.038 instead of 0.027 and 0.7-1.6 GiB more peak memory,
for a Beam SAC step of 24.23 instead of 27.43 ms balanced on 8xB200), and falls back to the BF16 fused
persistent backward with one warning when `hidden % 512 != 0` (an explicit
`MONO_EP_SM100_BACKWARD_PRECISION=mxfp8` raises there instead): the MXFP8 backward currently
supports hidden sizes that are multiples of 512, and if you need an odd multiple of 256, open an
issue at https://github.com/axolotl-ai-cloud/axolotl/issues. Its forward, backward and weight
gradients carry about 6-7 % relative error against an FP32 reference (BF16: 0.2 %); set
`expert_parallel_mono_ep_precision: bf16` for the BF16 kernels.

`MONO_EP_STORED_CONTEXT=auto` does not store when the backward is MXFP8, which is the SM100
default under `precision: mxfp8` / `auto` and the opt-in `MONO_EP_SM100_BACKWARD_PRECISION=mxfp8`
on the BF16 runtime: the MXFP8 recompute is one FP8 GEMM and costs less than gathering the
scattered BF16 z (8xB200 checkpointed 2F+B, storing vs recomputing: 235B-A22B 21.00 vs 19.42 ms,
Beam 44.83 vs 41.83 ms). `recompute` and `always` still force storing.

`auto` resolves to `nvfp4_lora` on SM100 when the experts load as packed NVFP4 (a modelopt NVFP4
checkpoint through the kernels plugin) and `adapter: lora` targets them; `nvfp4_lora` can also be
set explicitly. It is the one precision that trains with an adapter: the config needs
`use_sonicmoe: true` (the NVFP4 checkpoint loader and the expert-LoRA forward of the kernels
plugin), `nvfp4_merge_aware: false` (merge-aware training is the SonicMoE path),
`lora_dropout: 0` and `lora_target_parameters` on both `experts.gate_up_proj` and
`experts.down_proj`
([examples/expert_parallel/qwen3_30ba3b_nvfp4_lora_mono_ep.yaml](../../../examples/expert_parallel/qwen3_30ba3b_nvfp4_lora_mono_ep.yaml)).
Each rank keeps its EP slice of the packed experts frozen and outside FSDP (the dp_shard peers
hold the same slice) and rebuilds the merged NVFP4 weights (base + adapters, requantized) after
every optimizer step. By default the rebuild happens just in time per layer
(`expert_parallel_mono_ep_nvfp4_merged_cache: layer`): the forward builds the two forward forms,
the checkpoint recompute builds them again, the backward builds the three backward forms, and only
the two most recent layers stay resident (about two local-expert copies per rank in total, not per
layer), with the transposed forms quantized from the merged tile (`MONO_EP_NVFP4_REQUANT=cuda_fused`,
so no transposed copy of the base is stored). Qwen3-30B-A3B at EP4, 3,840 tokens per rank, 48 layers:
allocator peak 24.7 -> 12.2 GiB per rank (reserved 27.2 -> 14.8, device 30.3 -> 17.4; the SonicMoE
merge-aware reference is 15.2 / 17.8), 0.95 -> 1.03 s per step for the two extra rebuilds per layer
(`all` with the fused forms and the bounded workspace: 19.7 GiB, 0.98 s). The first-layer log line
names the backend, the cache mode and what the layer adds besides the packed base. The checkpoint's
`input_scale` tensors set the static activation scales; a weight-only export runs with scale 1
and a warning. Only the adapter is saved; merging it into the NVFP4 base is `merge-lora` through
the SonicMoE merge-aware path. The package API is `MonoEPRuntime(precision="nvfp4_lora")` and
`forward_lora(...)` ([trl-integration.md](trl-integration.md#nvfp4-lora)).

With an expert LoRA adapter on a BF16 base, the fused mode trains it on the full fine-tune
kernels of the configured precision: `auto` and `mxfp8` resolve to `mxfp8_lora` on SM100, `bf16`
to `bf16_lora`, and on SM90 `bf16` to `bf16_lora` and `fp8` to `fp8_lora`. The config needs
`adapter: lora` with `lora_target_parameters` on both `experts.gate_up_proj` and
`experts.down_proj`, `use_sonicmoe: true` (its ParamWrapper fast path hands the factors to the
experts module), `lora_dropout: 0` and no DoRA; anything else keeps the dispatch mode, and
`expert_parallel_mono_ep_fused_lora: false` restores the dispatch mode outright. The base stays a
frozen parameter (FSDP2-sharded with the rest of the layer, no optimizer state); after every
optimizer step each layer's merged weights `bf16(W + scaling * B @ A)` are rebuilt once per use
under `expert_parallel_mono_ep_lora_merged_cache: layer` (forward, checkpoint recompute and,
when evicted, backward; the last two layers stay resident) or once per step under `all`, and the
kernels run on them. Adapter gradients come from rank-r grouped GEMMs on the backward's BF16 pools
(blockwise FP8: projected from the FP8 dW); the base gets none. Adapter-gradient error against an
FP32 merge-aware reference sits at the precision's own full fine-tune dW envelope (BF16 0.49 % vs
0.43 %, MXFP8 6.1-6.3 % vs 6.6-6.8 %, blockwise FP8 6.5-6.8 % vs 6.6-6.8 %). On 8xB200 at EP8 the
`mxfp8_lora` SAC step is 1.75-2.2x faster than the dispatch mode around SonicMoE LoRA with a
quarter of its peak memory, and at 235B-A22B / Beam faster than the full fine-tune
([benchmarks.md](benchmarks.md#fused-lora-on-bf16-experts-mxfp8_lora-bf16_lora)). The package API is
`MonoEPRuntime(precision="mxfp8_lora")` with `forward_lora(..., w1=, w2=)`, or `mono_ep.lora.lora_moe`.

### Fallback conditions

The backend falls back to `torch` (or raises, per `expert_parallel_fallback_on_unsupported`)
when:

| Condition | Checked |
|---|---|
| the `mono_ep` package or its DeepGEMM build is missing | config load |
| an adapter is configured and the config does not qualify for a fused LoRA precision (not `adapter: lora`, no `use_sonicmoe`, `lora_dropout > 0`, DoRA, an expert projection missing from `lora_target_parameters`, `expert_parallel_mono_ep_fused_lora: false`; for `nvfp4_lora` also `nvfp4_merge_aware` on or a base that is not a modelopt NVFP4 checkpoint; a modelopt NVFP4 base with another precision); a `*_lora` precision without an adapter | config load |
| quantized loading (`load_in_4bit`, `load_in_8bit`, `gptq`, `model_quantization_config`) | config load |
| no CUDA, or a compute capability other than SM90 / SM100 | after model build |
| `hidden_size % 256 != 0` or expert intermediate size `% 128 != 0` (`% 256` for `nvfp4_lora`) | after model build |
| expert biases, a non-SiLU gate, a custom `_apply_gate` (e.g. `alpha`), or a clamped SwiGLU (`limit` / `swiglu_limit`) | after model build |
| experts not in HF `gate_up_proj [E, 2I, H]` / `down_proj [E, H, I]` layout, or MoE layers with different shapes | after model build |
| with an adapter: for `nvfp4_lora`, experts that are not torchao `NVFP4Tensor`s with row-major block-16 scales and a per-expert per-tensor scale, or not SM100; for the other LoRA precisions, packed NVFP4 experts | after model build |
| the precision does not match the architecture | after model build |

The Axolotl side (checkpointing policy, the FSDP mesh, the runtime's lifetime) is described
in `src/axolotl/integrations/expert_parallel/README.md`.

## FSDP2, reserved SMs and `NCCL_MAX_CTAS`

The persistent kernels occupy every SM they are given and spin on readiness flags, so a
NCCL collective issued by FSDP while a kernel runs (an all-gather of the next layer, a
reduce-scatter) must fit in the SMs the kernel leaves free, or the two deadlock.
`MonoEPRuntime(reserved_sms=16)` (`DEFAULT_RESERVED_SMS`) sets DeepGEMM's SM budget to
`physical - 16`; `nccl_env(reserved_sms, dp_replicate)` returns `NCCL_MAX_CTAS =
reserved_sms // 2` (`// 3` under HSDP, where two communicators run at once), which must be in
the environment before the process group exists, and `check_nccl_env` verifies it. The Axolotl
plugin exports it through its `get_distributed_env` hook; a user value wins but may not exceed
the plugin's. On the four-GPU `dp_shard 2 x EP 2` mesh the B200 validator needed 24 reserved
SMs; the plugin's default is 16.

Under FSDP2 the unsharded parameters are re-gathered into persistent storage every step
without a version bump, so the weight-layout cache cannot rely on tensor identity. The
contract is in [architecture.md](architecture.md#weight-layout-cache): the cache key is the
storage plus the caller's epoch, the Axolotl seam sets the epoch per optimizer step, and a
global optimizer post-step hook invalidates every entry as a safety net.

## Environment selectors

Defaults are the values measured in the last Hopper and Blackwell tuning rounds. Each default
was checked bitwise, or against an FP32 reference, against the value in the "restores" column
before it was made the default; setting that value restores the previous behaviour.

| Variable | Default | Restores | Effect |
|---|---|---|---|
| `NCCL_MAX_CTAS` | exported by the plugin, `reserved_sms // 2` (`// 3` under HSDP) | | See above. |
| `MONO_EP_WEIGHT_CACHE` | `1` | `0` (rebuild per call), `legacy` (per-call op sequence) | Cache the per-call weight layouts (casts, transposes, interleaving, FP8 forms) across calls within an epoch. |
| `MONO_EP_WEIGHT_CACHE_OPTIMIZER_HOOK` | `1` | `0` | Register the global `torch.optim` post-step hook that invalidates the cache. |
| `MONO_EP_WEIGHT_CACHE_KEY` | `auto` | `storage`, `storage_epoch`, `object` (not safe under FSDP2) | Cache key; `auto` is `storage` until a caller sets an epoch, then `storage_epoch`. |
| `MONO_EP_WEIGHT_CACHE_ENTRIES` | `2` | | `(w1, w2)` pairs kept per layout; about 300 MB per pair at the Qwen3-30B-A3B EP8 shape. |
| `MONO_EP_WEIGHT_CACHE_COMPILED` | `1` | `0` (eager builders) | Build the layouts with Inductor-compiled cast/permute kernels. |
| `MONO_EP_WEIGHT_CACHE_FUSED` | `0` | | Fuse the compiled builds of one call (SM100 experiment). |
| `MONO_EP_SM90_FP8_WEIGHT_CACHE` | `1` | `0` | SM90 FP8: cache the quantized weight forms per optimizer step. |
| `MONO_EP_SM90_FP8_FUSED_PREP` | `1` | `0` | SM90 FP8: fused token quantization and prefix preparation. |
| `MONO_EP_SM90_FP8_FUSED_WEIGHT_QUANT` | `1` | `0` | SM90 FP8: one launch for the per-expert transpose, scale reduction and FP8 conversion of the weights. |
| `MONO_EP_SM90_FP8_TRANSPORT`, `MONO_EP_SM90_FP8_COMBINE` | unset (BF16) | | SM90 FP8 backward: FP8 x/dy transport and FP8 dX combine; the benchmarks set both to `1`. |
| `MONO_EP_SM90_FP8_FORWARD_STAGES` | `5` | `2`..`4` | SM90 FP8 forward TMA pipeline stages. |
| `MONO_EP_SM90_FP8_BACKWARD_STAGES` | `3` | `0`..`8` | SM90 FP8 backward stages with the cached-dZ layout. |
| `MONO_EP_SM90_FP8_FORWARD_MAINLOOP` | `1` | `0` | Forward mainloop variant (register allocation and accumulator handling). |
| `MONO_EP_SM90_FP8_BACKWARD_MAINLOOP` | `0` | `2` (weight-scale prefetch) | Backward mainloop variant. Mode 2 was slower on top of the dispatch-warp pack in round 2b; it excludes the epilogue overlap, and it has not been re-measured since round 4 turned both of those off (one later A/B saw -0.05 ms, within noise). |
| `MONO_EP_SM90_FP8_FORWARD_ROLE_REGISTERS`, `MONO_EP_SM90_FP8_ROLE_REGISTERS` | `2`, `2` | `0` | `setmaxnreg` placement per warpgroup role. |
| `MONO_EP_SM90_FP8_TAIL_SCHEDULE` | `1` | `0` | Backward: idle CTAs take the last wave's dX tiles, partial blocks last; mode 1 allows more than 32 local experts. |
| `MONO_EP_SM90_FP8_OVERLAP_EPILOGUE` | `0` | `1` | Backward epilogue/mainloop overlap sites (measured slower at EP8 with the current defaults). |
| `MONO_EP_SM90_FP8_DISPATCH_XDY_PACK` | `0` | `1` | Pack x/dy columns on the dispatch warps (faster single-rank, slower at EP8). |
| `MONO_EP_SM90_FP8_FORWARD_SCATTER_MODE` | `0` | `1`, `3` | Forward combine scatter ordering. |
| `MONO_EP_SM90_FP8_DISPATCH_PULL_BYTES` | 4 KB per row, capped at 2 KB when the stage counts would exceed shared memory (hidden 4096) | | NVLink pull granularity. |
| `MONO_EP_SM90_FP8_ACTIVATION` | `swiglu_fast` on the 512-thread launches, `swiglu` otherwise | `swiglu` (the bitwise IEEE reference) | Gated activation of the SM90 FP8 epilogues. `swiglu_fast` is not bitwise: it passed the FP32-reference validators and a 300-step loss-curve check against `swiglu`. Other values (`swiglu_tanh`, `swiglu_oai`, `geglu`, ...) exist for kernel experiments; Axolotl falls back for non-SiLU gates. |
| `MONO_EP_SM90_FP8_ACTIVATION_ALPHA`, `MONO_EP_SM90_FP8_ACTIVATION_LIMIT` | unset (`swiglu_oai`: limit 7) | | Activation parameters: `ALPHA` is the `swiglu_oai` gate slope, `LIMIT` the clamp of the SwiGLU forms. Kept for experiments, not for training: Axolotl falls back for clamped or non-SiLU gates. |
| `MONO_EP_SM90_FP8_ACTIVATION_LOOP` | `2` with `swiglu_fast`, `1` otherwise | `0` | Columns per lane in the SwiGLU epilogue loop. |
| `MONO_EP_SM90_FP8_ACTIVATION_CHUNK` | `16` with the activation loop on the 512-thread launch, else `0` (build default) | `0` | Backward: rows per dscore reduction chunk in the activation loop (2, 4, 8, 16, 32; at most 16 with loop mode 2). |
| `MONO_EP_SM90_FP8_FORWARD_VECTOR_SCATTER` | `1` | `0` | Forward: vectorized combine scatter in the down-GEMM epilogue. |
| `MONO_EP_SM90_FP8_FORWARD_VECTOR_SWIGLU` | `1` | `0` | Forward: vectorized SwiGLU in the gate/up epilogue. |
| `MONO_EP_SM90_FP8_FORWARD_VECTOR_HQUANT` | `1` | `0` | Forward: vectorized FP8 quantization of h in the gate/up epilogue. |
| `MONO_EP_SM90_FP8_FORWARD_SWIZZLE_ACCUM_SMEM` | `1` when the vector scatter and SwiGLU are on | `0` | Forward: swizzled shared-memory staging of the accumulators (needs both vector paths). |
| `MONO_EP_SM90_FP8_FORWARD_DEFER_COMBINE_FENCE` | `1` | `0` | Forward: one deferred system fence before the combine signal instead of one per store. |
| `MONO_EP_SM90_FP8_FORWARD_RELAXED_SPINS` | `1` on the 512-thread launch, `0` on compact384 | `0` | Forward: relaxed polls plus one acquire on the dispatch ring, A-loader, epilogue and combine spins. |
| `MONO_EP_SM90_FP8_RELAXED_SPINS` | `0` | | Backward: the same relaxed polls on the A-loader, dh epilogue and combine-readiness spins. Kept for experiments, not for training. |
| `MONO_EP_SM90_FP8_DISPATCH_SPIN` | `4` | `0` (acquire spin), `1`, `2`, `3` | Backward: ring-slot release wait. `0` ld.acquire spin, `1` relaxed poll and one acquire, `2`-`4` relaxed poll with nanosleep backoff (32-256, 64-1024, 128-4096 ns). |
| `MONO_EP_SM90_FP8_XDY_VECTOR_PACK` | `1` with FP8 transport and the caches, else `0` | `0` | Backward: vectorized packing of the x/dy columns on the math warps. |
| `MONO_EP_SM90_FP8_VECTOR_HDZ_PACK` | `2` | `0`, `1` | Backward: vectorized h/dZ column pack; `1` keeps the original fences, `2` also drops them (the packed columns are read by a later launch). |
| `MONO_EP_SM90_FP8_VECTOR_DZ_QUANT` | `1` | `0`, `2` | Backward: load-batch depth of the vectorized dZ row-cache quantize (`2` spills in the combined build). |
| `MONO_EP_SM90_FP8_DX_COMBINE` | `7` (with `MONO_EP_SM90_FP8_COMBINE=1`) | `0` | Backward FP8 dX combine bits: 1 vectorized scatter, 2 one system fence per thread, 4 vectorized tail. Ignored without the FP8 combine. |
| `MONO_EP_SM90_FP8_DISPATCH_PACK_REGISTERS` | `64/40/200` | `48/40/208` (the split without the pack) | Backward with `MONO_EP_SM90_FP8_DISPATCH_XDY_PACK=1`: `setmaxnreg` split for the dispatch / non-epilogue / epilogue warpgroups. Kept for experiments, not for training. |
| `MONO_EP_SM90_FP8_COMPACT_FORWARD`, `MONO_EP_SM90_FP8_COMPACT_BACKWARD` | unset | | 384-thread layouts kept for experiments; the 512-thread layouts are the defaults. |
| `MONO_EP_SM100_BACKWARD_STAGES` | `6` | `2` (the old default) | SM100 BF16 backward TMA/UMMA ring depth; unset takes the deepest ring that fits shared memory. |
| `MONO_EP_SM100_BACKWARD_BLOCK_K` | `64` | | SM100 BF16 backward BLOCK_K: `64` or `128` (the MXFP8 backward uses 128; in-kernel dW needs 64). |
| `MONO_EP_SM100_BACKWARD_PRECISION` | `mxfp8` under `precision: mxfp8`, else `bf16` | `bf16` | `mxfp8`: SM100 backward GEMMs (dh, z recompute, dx) as MXFP8 (E4M3, UE8M0 per 32 along K) with E4M3 x/dy over NVLink and dz quantized in the epilogue; the weight gradients stay BF16 GEMMs on exact BF16 copies of the E4M3 x/dy. Not bitwise: about 6-7% relative error against an FP32 reference (BF16: 0.2%) at every measured hidden size (2048, 4096, 5120); the loss curve (hidden 1024) stayed within a tenth of the seed gap. Needs `hidden % 512 == 0`. 8xB200 backward: 30B-A3B 4.21 to 3.83 ms, 235B-A22B 15.3 to 11.6 ms, Beam 34.5 to 34.0 ms (dispatch-side work grows with hidden).  Under `precision: mxfp8` (the SM100 `auto`) the MXFP8 runtime runs this backward by default and its forward reads the interleaved W1 from the same cached MXFP8 forms; `bf16` selects the BF16 persistent backward, and an explicit `mxfp8` with `hidden % 512 != 0` raises. 8xB200 EP8, 16,384 tokens per GPU, BF16 weights rebuilt per step, 16 reserved SMs, one variant per process: `precision: mxfp8` vs `precision: bf16` with this backward, F / B / SAC step, ms: 30B-A3B 2.21 / 3.74 / 5.89 vs 2.15 / 3.81 / 5.98, 235B-A22B 3.67 / 10.65 / 14.10 vs 4.45 / 11.30 / 15.43, Beam 7.04 / 22.18 / 28.69 vs 9.97 / 24.67 / 34.52; peak 5.1 / 12.7 / 31.7 vs 5.2 / 13.9 / 34.5 GiB. Same FP32-reference error (y 6.6 %, dx 6.8 %, dW 6.2 %); dx is bitwise equal between the two runtimes. |
| `MONO_EP_SM100_MXFP8_WEIGHT_FORMS` | `cuda` | `fused` (the Triton one-read kernel, the previous default), `compiled` (one Inductor graph), `triton` (per-group kernel) | Builder of the MXFP8 backward weight forms (W1 interleaved, W2ᵀ, W1ᵀ), rebuilt once per step; all four are bitwise identical (values and UTCCP scale words, BF16 and FP32 input, all-zero blocks). `cuda` is the SM100 kernel in `include/mxfp8_forms_launch.cu` (built on first use with the package's other CUDA kernels): one 128 x 128 tile per 256-thread CTA, TMA loads and stores with 128-byte swizzled tensor maps, the transposed pass first and the row form written in place, four CTAs per SM, which puts it at the DRAM bound. Off SM100, or if the extension fails to build (logged once, counted in `compile_fallbacks`), it falls back to `fused`. Single B200, EP8 local experts, Triton `fused` / `cuda` / MoK `mxfp8_quantize` (BF16 only), ms: Beam (32 experts) BF16 input 1.33 / 0.70 / 0.77, FP32 1.51 / 1.09 / n/a; Qwen3-235B (16) BF16 0.32 / 0.18 / 0.20, FP32 0.38 / 0.27; Qwen3-30B (16) BF16 0.095 / 0.055 / 0.063, FP32 0.109 / 0.081. `MONO_EP_SM100_MXFP8_FORMS_DEBUG` (bits 1 / 2 / 4 skip the load / quantization / stores; output is garbage) times the kernel's phases. The cache key and post-step invalidation are the builder's callers' and do not change with the selector. |
| `MONO_EP_SM100_MXFP8_COMPILED_QUANT` | unset | | Older selector, honoured when `_WEIGHT_FORMS` is unset: `1` is `compiled`, `0` is `triton`. |
| `MONO_EP_SM100_BACKWARD_OPTIONS` | `11` | `0` | SM100 backward epilogue option bits (staged pool stores, two-row pulls); bits 512 and 1024 are set by `MONO_EP_SM100_INKERNEL_DW` and `_WIDE`, bit 2048 by `MONO_EP_SM100_BACKWARD_PRECISION=mxfp8` (which requires bit 1 and excludes 4, 16 and 64), bit 8192 when the Python runtime launches the MXFP8 backward with MXFP8 dW (`MONO_EP_SM100_DW_PRECISION`; no BF16 x/dy/dz pool copies); the two features do not combine. |
| `MONO_EP_SM100_BACKWARD_PULL_BYTES` | two rows (8 KB) when whole rows fit, else `4096` | `4096` | SM100 backward NVLink pull size. |
| `MONO_EP_SM100_FORWARD_OPTIONS` | lookahead pull when the pull buffer is one row | `0` | SM100 forward option bits. |
| `MONO_EP_SM100_FORWARD_PULL_BYTES` | per shape | | SM100 forward pull size. |
| `MONO_EP_SM100_DW_DIRECT` | `1` | `0` (accumulate into a zero-filled output) | SM100 BF16 and MXFP8 backwards: the dW grouped GEMMs write their FP32 output directly and a small kernel zeroes only experts with no rows, instead of zero-filling all of dW first (5 GB per backward at the Beam shape). Not bitwise against `0` (FP32 accumulation order; FP64-anchored dW error unchanged, zero-row experts exactly zero). 8xB200 backward: 30B-A3B 4.19 to 4.14 ms, 235B-A22B 15.39 to 14.99 ms, Beam 35.50 to 34.06 ms. Python-side. |
| `MONO_EP_SM100_DW_PRECISION` | `mxfp8` under `precision: mxfp8`, else `bf16` | `bf16` | The Python runtime resolves the dW precision per backward call and passes it to the kernel launch: this variable when set (`bf16` or `mxfp8`), else the runtime's entry in `mono_ep.dw.MXFP8_BACKWARD_DW_DEFAULTS` when the MXFP8 backward runs (`mxfp8` for the MXFP8 runtime, `bf16` for the BF16 runtime's opt-in MXFP8 backward), else `bf16` (the BF16 backward, including the hidden % 512 fallback). `mxfp8` (needs the MXFP8 backward: `precision: mxfp8`, or `MONO_EP_SM100_BACKWARD_PRECISION=mxfp8` on the BF16 runtime): the SM100 dW grouped GEMMs on E4M3 operands with one UE8M0 scale per 32 rows and column (DeepGEMM k-grouped `recipe=(1, 1, 32)`, MN-major). The x, dy and dz operands are requantized along the rows from the backward's own E4M3 pools (one Triton pass, 0.5-0.6 ms per Beam pool on B200), h from its BF16 pool, and the kernel then writes no BF16 x/dy/dz copies (option bit 8192). Relative dW error against an FP32 reference 6.7% instead of 6.2% with BF16 dW (BF16 backward 0.4%); dx and dscore unchanged. 4xB200 EP4, 16,384 tokens per GPU, same process: Beam backward 32.96 to 30.53 ms (SAC step 41.8 to 41.5), Qwen3-235B neutral (12.4 to 12.8 host wall with 1.4 ms less device time; the eager dW path is launch-bound there). Evidence: FSDP2 dp_shard 2 x EP 2 with three AdamW steps passes with it (dW error 0.068 / 0.047 / 0.041 / 0.039 at steps 0-3 vs 0.062 / 0.040 / 0.031 / 0.029 with BF16 dW; `validate_fsdp2_weight_cache.py --precision mxfp8 --tolerance 0.1`, since the default 0.03 is set for BF16; every stale-cache detector, the frozen MXFP8 forms included, still fires at 0.75 or more); 300-step loss curves of a random-init Qwen3-MoE (two seeds) sit 0.011-0.016 per step from BF16 against a 0.089 seed gap, final eval +0.014 / -0.017 (seed spread 0.033).  Under `precision: mxfp8` it is the default, traded for speed (weight-gradient error 6.8 % vs 6.2 %, FSDP2 three-step floor 0.039 vs 0.031); `MONO_EP_SM100_DW_PRECISION=bf16` restores BF16 dW: SAC step 5.87 / 12.79 / 25.29 ms instead of 5.89 / 14.10 / 28.69 at 30B-A3B / 235B-A22B / Beam (8xB200), peak +0.7 to +1.6 GiB. Repeated on a second 8xB200 pod under `precision: mxfp8` (three single-variant processes per cell, 16,384 tokens per GPU): SAC step 5.83 / 12.56 / 24.23 ms instead of 5.92 / 13.71 / 27.43 with balanced routing and 7.07 / 15.67 / 30.48 instead of 7.24 / 17.29 / 34.90 skewed, peak +0.9 / +1.6 / +0.7 GiB; FP32-reference dW error 0.066-0.068 instead of 0.060-0.062 at all three shapes, balanced and skewed, with y, dx and dscore unchanged; FSDP2 dp_shard 2 x EP 2 over five AdamW steps 0.068 / 0.047 / 0.041 / 0.039 / 0.038 / 0.038 vs 0.062 / 0.040 / 0.031 / 0.029 / 0.028 / 0.027; 500-step loss curves on the MXFP8 runtime (three seeds) sit 0.010-0.011 per step from BF16 dW, against 0.005-0.012 between two runs of one seed and setting and 0.092-0.095 between seeds. dW of either precision repeats to about 1e-5 relative run to run, not to the byte. |
| `MONO_EP_SM100_REQUANT_CONFIG` | `1,2,1` | `1,4,0` (the original launch) | Launch of the MXFP8 dW requantization (`requant_k_blocks`): `column_blocks,num_warps,reciprocal[,persistent]` (`/` also separates). The default runs two warps per 128 x 128 block and multiplies by the exact inverse of the power-of-two scale instead of dividing: bitwise equal to `1,4,0` (values and scale words) and a third faster on one B200 (Beam pools 2.25 -> 1.57 ms per backward, Qwen3-235B 1.61 -> 1.12, 30B 0.82 -> ~0.55). `persistent` > 0 strides a fixed row grid over the used blocks (no gain over the default on worst-case pools). 8xB200 EP8, same process, MXFP8 + MXFP8 dW + stored z: backward 21.22 -> 20.06-20.23 ms (Beam), 10.49 -> 9.70-9.76 ms (Qwen3-235B); dx bitwise; dW deviation equal to the run-to-run control; FSDP2 dp_shard 2 x EP 2 with three AdamW steps gives identical dW errors with either launch. |
| `MONO_EP_SM100_MXFP8_FORMS_CONFIG` | `2,32,1` | `4,32,0` (the original launch) | Launch of the Triton `fused` MXFP8 weight-form builder (`fused_mx_backward_forms`, not used by the default `cuda` builder: W1 interleaved row form, W1^T, W2^T from the FP32 masters): `num_warps,pass_rows,reciprocal` (`/` also separates). The default runs two warps and multiplies by the exact inverse of the power-of-two scale: bitwise equal to `4,32,0` and ~18 % faster on one B200 (32 Beam experts 1.84 -> 1.51 ms, 16 Qwen3-235B experts 0.45 -> 0.37 ms, ~4.7 TB/s against a ~1.1 ms FP32-read floor at Beam). The forms are cached per optimizer step (storage + version + epoch key, post-step invalidation); FSDP2 dp_shard 2 x EP 2 gives identical errors with either launch and the stale-forms detector still fires. |
| `MONO_EP_SM100_INKERNEL_DW` | `0` | | SM100 BF16: compute dW1/dW2 inside the persistent backward (option bit 512) instead of the grouped GEMMs after it. At most 32 local experts. 8xB200: +0.10 ms per backward at the Qwen3-30B-A3B shape (+0.04 with `_WIDE=1`), -1.49 ms (-4.9%, with `_WIDE=1`) at the Beam shape. |
| `MONO_EP_SM100_INKERNEL_DW_WIDE` | `0` | | With in-kernel dW: 256 x 256 tiles over 32-row stages (option bit 1024) instead of 256 x 128. |
| `MONO_EP_SM100_INKERNEL_DW_PER_BLOCK` | `2` | | With in-kernel dW: ready dW tiles the scheduler takes after each M-block (the rest run as experts finish). Runtime value, not part of the JIT key. |
| `MONO_EP_WORKSPACE_BOUND` | `auto` | `max` (pools for ranks x max tokens x top-k rows) | SM100 BF16 / MXFP8 backward pools (x, dy, h, dz and the E4M3 pools; the MXFP8 scale pools keep the worst case) sized to the rows this rank actually receives, padded to 128 per local expert. The forward reports its per-expert receive counts (copied to pinned memory without a sync) and the backward over the same inputs sizes its pools to that exact count before launching, so nothing overflows; the kernel traps past `pool_rows` as a host-bug guard. Pools start at 1.25 x tokens x top-k + 128 per local expert and only grow (to 1.1 x a larger need, logged). Records are keyed by the layer's expert weight storage with a per-layer stack, so they survive checkpoint recomputes (which rebuild the MoE input and routing as new tensors), FSDP2 regathers and gradient accumulation; checkpoint-recompute forwards do not record unless their layer has no record (reentrant checkpointing runs the original forward under `no_grad`); only forwards that can reach a backward record (grad mode on and an input or weight requiring grad), so eval, `no_grad` / `inference_mode` and generation forwards leave none; records of forwards whose backward never ran are dropped after every `optimizer.step()`, and a layer stack deeper than 64 drops its oldest entry with a warning; a backward with no record uses the worst case and logs a warning each time. 4 checkpointed layers x 2 accumulated micro-batches with `no_grad` and `inference_mode` eval forwards in between: 8 of 8 backwards found their record (non-reentrant and reentrant), no stale records, an abandoned forward's 4 records dropped by the next step. The SM100 BF16 backward's symmetric ring is also capped at the owned-backward floor (one wave of CTA pairs, `MONO_EP_BACKWARD_RING_WAVES`) instead of the forward's worst-case live-block sizing: EP4, 16,384 tokens per GPU, 149,760 -> 9,600 ring tokens, backward symmetric buffer 1.42 -> 0.68 GB (30B-A3B) / 2.75 -> 1.32 GB (235B-A22B), Beam unchanged (already at the floor); same-process backward 4.29 / 4.29 ms, 16.57 / 16.68 ms, dx bitwise. The forward ring keeps its worst-case sizing (the routed rows are unknown before launch). SM90 (H200 8x, 16,384 tokens per GPU): the BF16 and blockwise FP8 backwards take the same bounded pools and owned ring (four waves for BF16, which loses 3-4 % at one wave), and the blockwise forward reports its receive counts; resident after construction 20.7 -> 5.2 / 40.4 -> 9.7 / 38.6 -> 9.0 GB blockwise FP8 (30B-A3B / 235B-A22B / Beam) and 18.0 -> 5.5 / 35.4 -> 10.4 GB BF16; blockwise backward within 0.6 %, BF16 30B +0.5 % balanced / +2.4 % skewed at four waves; dx bitwise, FSDP2 dp_shard 2 x EP 2 passes with the record key stable across steps. dx bitwise vs `max`; FSDP2 dp_shard 2 x EP 2 passes. The SM100 NVFP4 LoRA backward (`precision="nvfp4_lora"`) is bounded the same way: its BF16 x / dy / h / dz pools, FP4 x / dy / dz pools and their scale pools (`[K/64][rows]`, runtime-strided) follow the recorded receive counts and its ring sits at the owned-backward floor (`MONO_EP_NVFP4_BACKWARD_RING_WAVES`); Qwen3-30B-A3B EP4, 3,840 tokens per rank: pools 1.89 -> 0.63 GiB, backward ring 42,240 -> 9,600 tokens (0.39 -> 0.23 GiB); Beam-1.0-501B EP8 geometry, 3,840 tokens per rank: pools 10.35 -> 1.76 GiB (same parity results, forward+backward +0.4 %), y and dx bitwise against `max` for balanced, skewed and adversarial routing, dscore / dA within the kernel's own run-to-run atomics noise (1e-7 / 1e-4 relative, same as `max` vs `max`), forward+backward +1.4 %; adversarial routing grows the pools once to the worst case. FSDP2 dp_shard 2 x EP 2 with AdamW steps passes in `layer` and `all` cache modes. 8xB200, 16,384 tokens per GPU: resident after construction 18.1 -> 7.0 / 35.5 -> 13.2 / 41.0 -> 9.0 GB per rank (30B-A3B / 235B-A22B / Beam); backward unchanged; a backward issued right behind its own forward waits for the count (forward+backward of one layer +0.1-0.15 ms; in a model only the last layer of a step). With the MXFP8 backward the E4M3 pools follow the bound and the MXFP8 dW requantization reads the worst-case scale pool by its row stride: dx bitwise vs `max` at 30B-A3B / 235B-A22B / Beam for balanced, skewed, unrecorded (pool growth) and adversarial routing, with BF16 or MXFP8 dW; dW and dscore within the `max` build's own same-process control noise. Python-side. |
| `MONO_EP_FORWARD_RING_ROWS` | unset (1.25 x tokens x top-k, rounded to 128) | `MONO_EP_WORKSPACE_BOUND=max` or `0` | Any non-negative integer (`0` = the worst-case ring; negative or non-integer values raise). The ring rounds up to a multiple of 1,920 tokens (whole blocks of every candidate BLOCK_M), so the smallest ring is 1,920 tokens (eight 240-row blocks) and every budget down to `1` completes (the 1,920-token ring at the Qwen3-30B-A3B shape on 1, 2 and 8 ranks, balanced and skewed, BF16 and NVFP4 LoRA: y and dx bitwise against the worst-case ring). Routed-row budget the SM100 BF16 forward's symmetric ring is sized for when the workspace bound is `auto`. The forward scheduler gates L1 claims on the ring (an L1 block is claimed only after every L2 task one ring generation back is claimed, and a claimed L2 task is held until L1 of the block one generation back has finished, because the L2 readiness mask only carries the generation's parity), so a launch whose routing needs more live blocks runs gated and slower, never wrong; it switches on per launch from the received counts and is off whenever the routing fits. Wide experts (H x I >= 4096 x 1536) size the ring by the live-block bound for this budget; narrow experts give it one slot per budgeted row, because slot reuse inside a launch costs a dispatch-bound forward 4-8 %. Never above the worst-case ring. 8xB200, 16,384 tokens per GPU: forward symmetric buffer 2.15 -> 1.50 GB (30B-A3B), 4.19 -> 1.73 GB (235B-A22B), Beam unchanged (its live-block ring is already 3,840); forward+backward 5.78 / 5.80 ms, 19.06 / 19.08 ms balanced, skewed +0.0 / +1.3 %; y and dx bitwise against the worst-case ring for balanced, skewed and adversarial routing, and with a forced 2,048-row budget (gate active every launch). The SM100 NVFP4 LoRA forward uses the same budget and gate (3,840 tokens per rank: 42,240 -> 38,400 ring tokens; a forced 2,048-row budget at 16,384 tokens per rank runs gated at +36 % forward time, no deadlock for balanced, skewed or adversarial routing). MXFP8 and SM90 forwards keep the worst-case ring and refuse a bounded buffer. |
| `MONO_EP_FORWARD_RING_TOKENS` | unset | unset | Exact forward ring tokens (sweeps), capped at the worst-case ring and rounded up to a multiple of 1,920; `0` uses the live-block bound for the budget. Any non-negative integer is legal; negative or non-integer values raise. |
| `MONO_EP_STORED_CONTEXT` | `auto` | `off` | SM100 BF16: a storing forward writes z (BF16 gate/up pre-activations) by its own pool row into one runtime-owned buffer, and the next backward with the same inputs skips its gate/up recompute GEMMs (option bit 4096) and reads z instead; dx is bitwise equal to the recomputing backward. One buffer serves one layer at a time, so it pays when the storing forward runs right before that layer's backward, i.e. the activation-checkpoint recompute. `auto` stores in checkpoint-recompute forwards when the backward is the BF16 one, `hidden x intermediate >= 4096 x 1536` and the buffer fits in free device memory plus max(2 GiB, 5%) headroom, otherwise recomputes; it logs one line with the decision. `recompute` forces storing in checkpoint recomputes, `always` stores in every forward (only the last forward before a backward is used), `off` never stores. `recompute` and `always` also combine with `MONO_EP_SM100_BACKWARD_PRECISION=mxfp8` (BF16 z lowers the MXFP8 error: dx 5.3% against an FP32 reference instead of 6.8%) and with in-kernel dW; `auto` recomputes there because storing is slower (below). Nothing is stored when the fused op's output is saved instead (Axolotl `selective` checkpointing), since there is no recompute. Buffer: an index of `ranks x max tokens x top-k` int32 plus `capacity x 2I` BF16 rows; capacity starts at 1.25 x tokens x top-k + 128 per local expert and grows to 1.1 x the routed-row high-water mark each backward reports (read one step late, no host sync); a forward whose rows exceed it raises a device flag and that layer's backward recomputes (same results, about the off time). 8xB200, 16,384 tokens per GPU, balanced: buffer 0.51 / 1.02 / 1.72 GB at 30B-A3B / 235B-A22B / Beam (worst case 3.2 / 6.4 / 10.7 GB); checkpointed layer step off vs auto 7.78 / 7.83 ms (auto declines), 25.2 / 22.4 ms, 57.5 / 52.3 ms. One policy per process (8xB200, checkpointed layer, off vs auto, mean of two runs): 235B-A22B 21.51 / 20.69 ms balanced, 26.38 / 26.33 ms skewed; Beam 51.68 / 45.62 ms balanced, 63.92 / 57.81 ms skewed; the backward alone is unchanged.  The MXFP8 runtime (`precision: mxfp8`) recomputes z: its forward has no stored-z output. With the BF16 runtime's MXFP8 backward, `auto` recomputes: the MXFP8 gate/up recompute is one FP8 GEMM over H bytes per row, cheaper than reading 4I bytes of BF16 z per row in the backward epilogue. Stored vs recompute in the checkpointed 2F+B step (8xB200, 16 reserved SMs, one variant per process): 235B-A22B 21.2 vs 19.2 ms balanced, 26.6 vs 24.7 ms skewed; Beam 45.9 vs 41.9 ms balanced, 58.4 vs 51.6 ms skewed (backward kernel +1.0 to +3.4 ms, storing forward +0.4 to +0.8 ms). |
| `MONO_EP_STORED_CONTEXT_ROWS` | unset | | Fixed stored-z capacity in forward pool rows (no growth); for testing the overflow fallback. |
| `MONO_EP_BACKWARD_RING_WAVES` | `1` | `0` (the forward's interleaved-schedule ring sizing) | Owned-backward ring floor for the BF16 (SM90, SM100), MXFP8 (SM100) and blockwise-FP8 (SM90) backwards, in waves of owning clusters: one wave is `num_sms x 128` tokens on SM90 and `num_sms / 2 x 128` on SM100. It only raises the ring, so the Qwen3-30B-A3B shape is unchanged; wide-expert shapes (hidden 5120) gain most (the MXFP8 transport buffer took the floor late: Beam EP4 backward 37.9 to 28.4 ms, kernel 23.0 to 12.8, dispatch ring wait 15.4 to 7.0 ms per warp). The ring size is a kernel template parameter. |
| `MONO_EP_NVFP4_BACKWARD_RING_WAVES` | `1` | `0` | The same floor for the SM100 NVFP4 LoRA backward. |
| `MONO_EP_SM100_NVFP4_BACKWARD_OPTIONS` | `2051` | `3` (before the lean tail), `0` | NVFP4 LoRA backward bits: 1 staged h/dz pool stores, 2 TMEM release before the recompute/dH math, 2048 lean recompute tail (Dh and recompute tasks skip the per-task fence; each CTA fences once at its block's last recompute tile). Its own option space: bit 2048 here is not the BF16 backward's MXFP8 bit. |
| `MONO_EP_SM100_NVFP4_BACKWARD_PULL_BYTES` | whole x and dy rows with lookahead | | Chunked dispatch pull for the NVFP4 backward: four chunks, each a multiple of 128 bytes, at most 4096 and smaller than an x+dy row. |
| `MONO_EP_SM100_NVFP4_BACKWARD_STAGES` | the most that fit shared memory | | NVFP4 backward TMA/UMMA stages. |
| `MONO_EP_SM100_NVFP4_OPTIONS` | `1` when the pull buffer holds one row, else `0` | `0` | NVFP4 LoRA forward bits: 1 lookahead dispatch pull. |
| `MONO_EP_NVFP4_REQUANT` | `cuda_fused` | `cuda`, `triton` | Backend of the per-step NVFP4 merged-weight rebuild (requant cache). `cuda_fused` (SM100 kernel) quantizes the transposed backward forms `w1t` / `w2t` from the merged fp32 tile with fresh scales, so no transposed copy of the packed base is stored (81 MiB per layer at Qwen3-30B-A3B EP4, 3.8 GiB per rank over 48 layers; 0.84 GiB per layer at Beam EP8) and the backward is slightly closer to the fp32 reference (one B200, lean-tail default: dx 0.235 vs 0.267, dscore 0.23 vs 0.25, dA / dB 0.18-0.20 vs 0.19-0.22 relative). It moves away from SonicMoE merge-aware's double-rounded transposed bytes, which the Axolotl e2e parity test compares against: adapter-gradient cosine 0.941 -> 0.909 (bound 0.9), relative L2 0.356 -> 0.425 (bound 0.5); `cuda` restores the old bytes at the cost of the transposed base. `cuda` keeps a transposed base per layer and writes the Triton kernel's bytes for every form (the double-rounded transposed forms); `triton` is the vendored Triton kernel. Python-side, not a JIT key. |
| `MONO_EP_NVFP4_MERGED_CACHE` | `layer` | `all` | Residency of the merged NVFP4 forms. `layer`: built just in time, forward forms (`w1`, `w2`) in the forward and again in the checkpoint recompute, backward forms (`w1`, `w1t`, `w2t`) in the backward, LRU of the two most recent layers (`RequantCache(capacity=2)`), so 3 rebuilds per layer per step under non-reentrant checkpointing (the last two layers of a step hit the recompute forms). `all`: one rebuild of all four forms per layer per step, every layer resident. Both invalidate on the global optimizer post-step hook (version bump) and never serve stale forms: FSDP2 dp_shard 2 x EP 2, three AdamW steps, cached bytes equal a fresh requant of the live adapters every step in both modes. Qwen3-30B-A3B EP4, 3,840 tokens per rank, 48 layers: resident merged forms 7.6 GiB -> 0.26 GiB per rank, 0.98 -> 1.03 s per step (the Axolotl 50-step run; losses equal within the atomics noise, 0.938 vs 0.937 at step 50); one B200 rebuild of the forward forms 0.28 ms, of the backward forms 0.36 ms per layer (32 local 2048 x 768 experts). Beam-1.0-501B at EP8 (32 local 5120 x 2560 experts, 1,350 MiB of merged forms per layer): `all` would hold 68.6 GiB per rank over 52 layers, `layer` holds 2.2 GiB; rebuilds 1.78 / 2.43 ms per layer. |
| `MONO_EP_NVFP4_REQUANT_FFMA2` | `1` | `0` | Paired FP32 FMA (FFMA2) in the CUDA requant's delta math; `0` builds the scalar variant, 2-6% slower. |
| `MONO_EP_LORA_WGRAD` (or `MONO_EP_NVFP4_LORA_WGRAD`) | `triton` for `bf16_lora` / `mxfp8_lora` on SM100; `grouped` for `nvfp4_lora`, SM90 and the blockwise projection | `grouped` (`torch._grouped_mm`), `loop` (per-expert reference loop) | LoRA adapter gradients from the backward's pools. `triton`: two Triton passes (rank-r projections per 128-row block with FP32 accumulation, then FP32 row contractions per expert) instead of eight grouped GEMMs whose contractions store BF16. One B200 at the EP8 pools, per layer 0.93 / 1.61 / 2.08 -> 0.73 / 1.20 / 1.66 ms (30B-A3B / 235B-A22B / Beam) and error vs FP32 3.04e-3 -> 2.55e-3; 8xB200 EP8 SAC step `mxfp8_lora` 5.89 / 12.18 / 22.35 -> 5.65 / 11.49 / 21.66 ms, `bf16_lora` 6.45 / 16.74 / 36.59 -> 6.23 / 16.10 / 35.54 ms. `nvfp4_lora` keeps `grouped` until timed; on SM90 the kernels are unvalidated. Python-side. |
| `MONO_EP_LORA_MXFP8_FORMS` | `fused` | `merge` | `mxfp8_lora` with the MXFP8 backward: build the four MXFP8 weight forms straight from the BF16 base and the factors, the merge `W + s B@A` done per 128 x 128 tile in shared memory (rank-r product on the tensor cores, one FP32 fmaf, one BF16 rounding) inside the CUDA forms builder, so no merged BF16 copy exists. `merge` restores the BF16 merge (`baddbmm`, which rounds the delta to BF16 before adding) followed by the full fine-tune's forms. 8xB200 EP8, per layer rebuild 0.24 / 0.81 / 3.14 -> 0.12 / 0.35 / 1.31 ms, SAC step 6.10 / 12.56 / 23.76 -> 5.94 / 12.15 / 22.41 ms and peak 3.9 / 8.7 / 16.8 -> 3.8 / 8.1 / 14.4 GiB (30B-A3B / 235B-A22B / Beam). The forms equal those of a single-rounded FP32 merge up to 1-2 E4M3 bytes per 10^7 (FP32 summation order of the rank-r product) and the plain forms bitwise with zero adapters; against `merge` 1.3 % of the E4M3 bytes differ because of its double rounding (dequantized error vs the FP32 merged weight 0.02660 vs 0.02663). Python-side. |
| `MONO_EP_LORA_MERGED_CACHE` | `layer` | `all` | Residency of the merged BF16 weights of `bf16_lora` / `mxfp8_lora` / `fp8_lora`, as `MONO_EP_NVFP4_MERGED_CACHE` for NVFP4. The weight forms the kernels read derive from the merged tensors through the weight-layout cache (`MONO_EP_WEIGHT_CACHE_ENTRIES`). Python-side. |
| `MONO_EP_BACKWARD_SEGMENT_TIMING` | unset | | Diagnostics: `1` records host time and a CUDA event at the MXFP8 backward's segment boundaries (token quantization, weight forms, kernel, requant, dW1, dW2) into `persistent_backward.SEGMENT_MARKS`, for timing probes. |
| `MONO_EP_NVFP4_REQUANT_VERBOSE`, `MONO_EP_SM100_NVFP4_PHASE_TRACE` | unset | | Diagnostics: CUDA requant build log, NVFP4 forward phase trace. |
| `MONO_EP_SM90_FP8_TRAP_TIMEOUTS`, `*_PHASE_TRACE`, `*_DIST_TRACE`, `*_DIAG_ABLATE` | unset | | Diagnostics: trap-only timeouts for compiler experiments, per-phase and per-rank traces, epilogue ablations. Not for training. By name: `MONO_EP_SM90_FP8_PHASE_TRACE`, `MONO_EP_SM90_FP8_FORWARD_PHASE_TRACE`, `MONO_EP_SM90_FP8_DIST_TRACE`, `MONO_EP_SM90_FP8_FORWARD_DIST_TRACE`, `MONO_EP_SM90_FP8_DIAG_ABLATE`, `MONO_EP_SM90_FP8_FORWARD_DIAG_ABLATE`, `MONO_EP_SM100_PHASE_TRACE`, `MONO_EP_SM100_FORWARD_PHASE_TRACE`, `MONO_EP_SM100_DIAG_ABLATE`, `MONO_EP_SM100_FORWARD_DIAG_ABLATE`. |

Every kernel selector is part of the JIT key, so changing one compiles a new kernel variant under
`DG_JIT_CACHE_DIR`; the selectors marked Python-side change only host code.

## Troubleshooting

- **`TypeError: <function>(): incompatible function arguments` from `deep_gemm._C`.** The
  installed extension is older than the Python package, usually an object reused from an
  existing `vendor/deepgemm/build` after a header-only change. Remove the build directory and
  reinstall (`rm -rf packages/mono-ep/vendor/deepgemm/build`, then the install command in the
  [README](../README.md#installation)). `DG_FORCE_BUILD=1` does not clean it: it only skips the
  prebuilt-wheel download. The JIT kernel cache (`DG_JIT_CACHE_DIR`, default `~/.dj`) is keyed by
  the generated source, the contents of every included header, the compiler flags and version, so
  it cannot serve a kernel built from older installed headers; removing it only costs recompiles.
- **The first step takes minutes.** Every kernel variant compiles on first use and is cached
  under `DG_JIT_CACHE_DIR`; point it at a persistent, shared directory so later runs and other
  ranks reuse the builds. Changing a kernel selector or the layer shape compiles a new variant.
- **A hang, then `DeepGEMM grid sync timeout` or `DeepGEMM combine peers timeout`.** The
  persistent kernels occupy every SM they are given; a NCCL collective that needs more CTAs than
  the reserved SMs hold waits behind them. Export `nccl_env(reserved_sms=...)` (or let the
  Axolotl plugin do it) before the process group is created, keep `NCCL_MAX_CTAS` at or below
  `reserved_sms // 2`, and raise `reserved_sms` if FSDP collectives still stall; the four-GPU
  `dp_shard 2 x EP 2` mesh on B200 needed 24. `check_nccl_env` raises early when the variable
  is missing or too large.
- **`expert_parallel_backend=mono_ep is unsupported here: <reason>. Falling back to
  expert_parallel_backend=torch ...`** The model or config hits a fallback condition (table
  above). Set `expert_parallel_fallback_on_unsupported: false` to raise instead, or
  `expert_parallel_mono_ep_mode: dispatch` when only the fused mode is excluded (adapters the
  fused LoRA path does not cover, quantized experts, biases, other activations).
- **Restoring a previous default.** Every selector above lists the value that restores the
  behaviour before its current default; set it in the environment of every rank.
