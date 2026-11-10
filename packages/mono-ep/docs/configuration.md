# Configuration

Everything a user sets, and every selector the kernels read. The plugin sets what the kernels
need; the environment variables exist for experiments and for restoring a previous default.

## Axolotl config keys

| Key | Values | Meaning |
|---|---|---|
| `expert_parallel_size` | `1` (off) or a divisor of the world size | Number of EP ranks; experts are split evenly across them. |
| `expert_parallel_backend` | `auto`, `deep_ep`, `torch`, `mono_ep` | Token dispatch backend. `auto` never picks `mono_ep`; set it explicitly. With `mono_ep` the local experts kernel (`use_sonicmoe`, `use_scattermoe`, `experts_implementation`) is not used. |
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
| `MONO_EP_SM90_FP8_BACKWARD_MAINLOOP` | `0` | `2` (weight-scale prefetch, slower on top of the dispatch-warp pack) | Backward mainloop variant. |
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
| `MONO_EP_SM100_BACKWARD_OPTIONS` | `11` | `0` | SM100 backward epilogue option bits (staged pool stores, two-row pulls). |
| `MONO_EP_SM100_BACKWARD_PULL_BYTES` | two rows (8 KB) when whole rows fit, else `4096` | `4096` | SM100 backward NVLink pull size. |
| `MONO_EP_SM100_FORWARD_OPTIONS` | lookahead pull when the pull buffer is one row | `0` | SM100 forward option bits. |
| `MONO_EP_SM100_FORWARD_PULL_BYTES` | per shape | | SM100 forward pull size. |
| `MONO_EP_SM100_DW_DIRECT` | `0` | | SM100: direct weight-gradient operands (experiment). |
| `MONO_EP_SM90_FP8_TRAP_TIMEOUTS`, `*_PHASE_TRACE`, `*_DIST_TRACE`, `*_DIAG_ABLATE` | unset | | Diagnostics: trap-only timeouts for compiler experiments, per-phase and per-rank traces, epilogue ablations. Not for training. |

Every selector is part of the JIT key, so changing one compiles a new kernel variant under
`DG_JIT_CACHE_DIR`.
