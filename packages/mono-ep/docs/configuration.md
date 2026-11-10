# Configuration

Everything a user sets, and every selector the kernels read. The plugin sets what the kernels
need; the environment variables exist for experiments and for restoring a previous default.

## Axolotl config keys

| Key | Values | Meaning |
|---|---|---|
| `expert_parallel_size` | `1` (off) or a divisor of the world size | Number of EP ranks; experts are split evenly across them. |
| `expert_parallel_backend` | `auto`, `deep_ep`, `torch`, `mono_ep` | Token dispatch backend. `auto` never picks `mono_ep`; set it explicitly. In the fused mode the local experts kernel (`use_sonicmoe`, `use_scattermoe`, `experts_implementation`) is not used; in the dispatch mode it is. |
| `expert_parallel_mono_ep_mode` | `auto` (default), `fused`, `dispatch` | `fused`: one persistent kernel per direction (plain SwiGLU experts, full fine-tuning). `dispatch`: mono-ep's peer dispatch and combine around the configured local experts kernel, so LoRA, quantized experts, biases, other activations and clamped SwiGLU work. `auto`: `fused` when the config and model allow it, else `dispatch`. |
| `expert_parallel_mono_ep_compact` | `false` (default), `true` | Dispatch mode: hand the local kernel only the routed receive rows instead of the fixed `ep_size x tokens_per_rank` layout, at the cost of one device-to-host count per layer (saved under activation checkpointing). |
| `expert_parallel_mono_ep_bounded` | `false` (default), `true` | With `compact`: receive into a bound from each layer's high-water routed count (filled to SonicMoE's row bucket) and read the count only after the local kernel is issued, so dispatch never stalls on the host. Bitwise equal to `compact`; a count above the bound re-runs that layer exactly. Restore: `false`. |
| `expert_parallel_mono_ep_precision` | `auto`, `bf16`, `mxfp8`, `fp8`, `fp8_bwd` | Compute precision of the fused kernels (table below). Master weights keep their dtype and weight gradients come back in it. |
| `expert_parallel_fallback_on_unsupported` | `true` (default), `false` | When `mono_ep` cannot run (table below), fall back to `expert_parallel_backend: torch` with a warning, or raise. |
| `dp_shard_size` | | FSDP shard axis next to EP; `expert_parallel_size x dp_shard_size` must equal the world size. The EP group must stay inside one NVLink node. |
| `activation_checkpointing` | `true` | Non-reentrant checkpointing; reentrant checkpointing is rejected with EP because its recompute cannot replay the fused op. |
| `fsdp_config.state_dict_type` | `FULL_STATE_DICT` | Required with EP; `SHARDED_STATE_DICT` would keep only one EP group's experts. |

| Precision | SM90 (H100, H200) | SM100 (B200) |
|---|---|---|
| `auto` | `bf16` | `mxfp8` |
| `bf16` | yes | yes |
| `mxfp8` | no | yes |
| `fp8` (blockwise E4M3 forward and backward, the kernels timed in [benchmarks.md](benchmarks.md)) | yes | no |
| `fp8_bwd` (BF16 forward, FP8 backward) | not implemented | no |

### Fallback conditions

The backend falls back to `torch` (or raises, per `expert_parallel_fallback_on_unsupported`)
when:

| Condition | Checked |
|---|---|
| the `mono_ep` package or its DeepGEMM build is missing | config load |
| an adapter (`adapter: lora`, `qlora`, ...) is configured | config load |
| quantized loading (`load_in_4bit`, `load_in_8bit`, `gptq`, `model_quantization_config`) | config load |
| no CUDA, or a compute capability other than SM90 / SM100 | after model build |
| `hidden_size % 256 != 0` or expert intermediate size `% 128 != 0` | after model build |
| expert biases, a non-SiLU gate, a custom `_apply_gate` (e.g. `alpha`), or a clamped SwiGLU (`limit` / `swiglu_limit`) | after model build |
| experts not in HF `gate_up_proj [E, 2I, H]` / `down_proj [E, H, I]` layout, or MoE layers with different shapes | after model build |
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
| `MONO_EP_SM90_FP8_ACTIVATION_LOOP` | `2` with `swiglu_fast`, `1` otherwise | `0` | Columns per lane in the SwiGLU epilogue loop. |
| `MONO_EP_SM90_FP8_COMPACT_FORWARD`, `MONO_EP_SM90_FP8_COMPACT_BACKWARD` | unset | | 384-thread layouts kept for experiments; the 512-thread layouts are the defaults. |
| `MONO_EP_SM100_BACKWARD_STAGES` | `6` | | SM100 backward pipeline stages. |
| `MONO_EP_SM100_BACKWARD_PRECISION` | `bf16` | | `mxfp8`: SM100 backward GEMMs (dh, z recompute, dx) as MXFP8 (E4M3, UE8M0 per 32 along K) with E4M3 x/dy over NVLink and dz quantized in the epilogue; the weight gradients stay BF16 GEMMs on exact BF16 copies of the E4M3 x/dy. Not bitwise: about 6-7% relative error against an FP32 reference (BF16: 0.2%); the loss curve stayed within a tenth of the seed gap. Needs `hidden % 512 == 0`, `hidden <= 2048`. |
| `MONO_EP_SM100_MXFP8_COMPILED_QUANT` | `1` | `0` (Triton builders) | Build the MXFP8 backward weight forms (W1, W2ᵀ, W1ᵀ) in one Inductor graph; bitwise equal to the Triton path. |
| `MONO_EP_SM100_BACKWARD_OPTIONS` | `11` | `0` | SM100 backward epilogue option bits (staged pool stores, two-row pulls); bits 512 and 1024 are set by `MONO_EP_SM100_INKERNEL_DW` and `_WIDE`, bit 2048 by `MONO_EP_SM100_BACKWARD_PRECISION=mxfp8` (which requires bit 1 and excludes 4, 16 and 64); the two features do not combine. |
| `MONO_EP_SM100_BACKWARD_PULL_BYTES` | two rows (8 KB) when whole rows fit, else `4096` | `4096` | SM100 backward NVLink pull size. |
| `MONO_EP_SM100_FORWARD_OPTIONS` | lookahead pull when the pull buffer is one row | `0` | SM100 forward option bits. |
| `MONO_EP_SM100_FORWARD_PULL_BYTES` | per shape | | SM100 forward pull size. |
| `MONO_EP_SM100_DW_DIRECT` | `0` | | SM100: direct weight-gradient operands (experiment). |
| `MONO_EP_SM100_INKERNEL_DW` | `0` | | SM100 BF16: compute dW1/dW2 inside the persistent backward (option bit 512) instead of the grouped GEMMs after it. At most 32 local experts. 8xB200: +0.08-0.11 ms per backward at the Qwen3-30B-A3B shape, -1.29 ms (-5.9%, with `_WIDE=1`) at H 5120 / I 2560 / 32 local experts. |
| `MONO_EP_SM100_INKERNEL_DW_WIDE` | `0` | | With in-kernel dW: 256 x 256 tiles over 32-row stages (option bit 1024) instead of 256 x 128. |
| `MONO_EP_SM100_INKERNEL_DW_PER_BLOCK` | `2` | | With in-kernel dW: ready dW tiles the scheduler takes after each M-block (the rest run as experts finish). Runtime value, not part of the JIT key. |
| `MONO_EP_BACKWARD_RING_WAVES` | `1` | `0` (the forward's interleaved-schedule ring sizing) | Owned-backward ring floor for the BF16 (SM90, SM100) and blockwise-FP8 (SM90) backwards, in waves of owning clusters: one wave is `num_sms x 128` tokens on SM90 and `num_sms / 2 x 128` on SM100. It only raises the ring, so the Qwen3-30B-A3B shape is unchanged; wide-expert shapes (hidden 5120) gain most. The ring size is a kernel template parameter. |
| `MONO_EP_NVFP4_BACKWARD_RING_WAVES` | `1` | `0` | The same floor for the SM100 NVFP4 LoRA backward. |
| `MONO_EP_SM100_NVFP4_BACKWARD_OPTIONS` | `2051` | `3` (before the lean tail), `0` | NVFP4 LoRA backward bits: 1 staged h/dz pool stores, 2 TMEM release before the recompute/dH math, 2048 lean recompute tail (Dh and recompute tasks skip the per-task fence; each CTA fences once at its block's last recompute tile). Its own option space: bit 2048 here is not the BF16 backward's MXFP8 bit. |
| `MONO_EP_SM100_NVFP4_BACKWARD_PULL_BYTES` | whole x and dy rows with lookahead | | Chunked dispatch pull for the NVFP4 backward: four chunks, each a multiple of 128 bytes, at most 4096 and smaller than an x+dy row. |
| `MONO_EP_SM100_NVFP4_BACKWARD_STAGES` | the most that fit shared memory | | NVFP4 backward TMA/UMMA stages. |
| `MONO_EP_SM100_NVFP4_OPTIONS` | `1` when the pull buffer holds one row, else `0` | `0` | NVFP4 LoRA forward bits: 1 lookahead dispatch pull. |
| `MONO_EP_NVFP4_REQUANT` | `cuda` | `triton` | Backend of the per-step NVFP4 merged-weight rebuild (requant cache): `cuda` is the SM100 kernel and writes the Triton kernel's bytes; `cuda_fused` (opt-in) quantizes the transposed forms from the merged tile, which is a numerics change for `w1t` / `w2t`. Python-side, not a JIT key. |
| `MONO_EP_NVFP4_REQUANT_FFMA2` | `1` | `0` | Paired FP32 FMA (FFMA2) in the CUDA requant's delta math; `0` builds the scalar variant, 2-6% slower. |
| `MONO_EP_NVFP4_LORA_WGRAD` | unset (grouped GEMMs) | `loop` (per-expert reference loop) | LoRA adapter weight gradients of the NVFP4 path. Python-side. |
| `MONO_EP_NVFP4_REQUANT_VERBOSE`, `MONO_EP_SM100_NVFP4_PHASE_TRACE` | unset | | Diagnostics: CUDA requant build log, NVFP4 forward phase trace. |
| `MONO_EP_SM90_FP8_TRAP_TIMEOUTS`, `*_PHASE_TRACE`, `*_DIST_TRACE`, `*_DIAG_ABLATE` | unset | | Diagnostics: trap-only timeouts for compiler experiments, per-phase and per-rank traces, epilogue ablations. Not for training. |

Every kernel selector is part of the JIT key, so changing one compiles a new kernel variant under
`DG_JIT_CACHE_DIR`; the selectors marked Python-side change only host code.
