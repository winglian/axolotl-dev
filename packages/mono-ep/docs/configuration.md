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
| `expert_parallel_mono_ep_workspace_bound` | `auto` (default), `max` | Fused mode: size the backward pools and the rings from each step's routed rows (`MONO_EP_WORKSPACE_BOUND`). `max` restores the worst-case sizing. |
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
| `nvfp4_lora` (frozen packed NVFP4 experts with BF16 LoRA adapters; hidden and intermediate multiples of 256; native backward only, one host) | no | yes, package API only |

`nvfp4_lora` is reachable today only through the package API: `MonoEPRuntime(precision="nvfp4_lora")`
and `MonoEPRuntime.forward_lora(...)` with a `mono_ep.nvfp4_lora.Nvfp4LoraExperts` base and
`A1 / B1 / A2 / B2` adapters ([trl-integration.md](trl-integration.md#nvfp4-lora)). The Axolotl
key `expert_parallel_mono_ep_precision` does not accept it, and an Axolotl `adapter:` falls back
to `torch` (table below).

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
| `MONO_EP_SM100_BACKWARD_PRECISION` | `bf16` | | `mxfp8`: SM100 backward GEMMs (dh, z recompute, dx) as MXFP8 (E4M3, UE8M0 per 32 along K) with E4M3 x/dy over NVLink and dz quantized in the epilogue; the weight gradients stay BF16 GEMMs on exact BF16 copies of the E4M3 x/dy. Not bitwise: about 6-7% relative error against an FP32 reference (BF16: 0.2%) at every measured hidden size (2048, 4096, 5120); the loss curve (hidden 1024) stayed within a tenth of the seed gap. Needs `hidden % 512 == 0`. 8xB200 backward: 30B-A3B 4.21 to 3.83 ms, 235B-A22B 15.3 to 11.6 ms, Beam 34.5 to 34.0 ms (dispatch-side work grows with hidden). |
| `MONO_EP_SM100_MXFP8_WEIGHT_FORMS` | `fused` | `compiled` (one Inductor graph, the previous default) | Builder of the MXFP8 backward weight forms (W1 interleaved, W2ᵀ, W1ᵀ), rebuilt once per step: `fused` reads each 128 x 128 weight tile once and writes its row-wise and transposed forms from registers, `compiled` is one Inductor graph, `triton` the per-group kernel; all three are bitwise identical. B200 in isolation, 64 / 32 Beam experts: 3.6 / 1.8 ms fused vs 5.4 / 2.7 compiled; 32 Qwen3-235B experts 0.87 vs 1.26. |
| `MONO_EP_SM100_MXFP8_COMPILED_QUANT` | unset | | Older selector, honoured when `_WEIGHT_FORMS` is unset: `1` is `compiled`, `0` is `triton`. |
| `MONO_EP_SM100_BACKWARD_OPTIONS` | `11` | `0` | SM100 backward epilogue option bits (staged pool stores, two-row pulls); bits 512 and 1024 are set by `MONO_EP_SM100_INKERNEL_DW` and `_WIDE`, bit 2048 by `MONO_EP_SM100_BACKWARD_PRECISION=mxfp8` (which requires bit 1 and excludes 4, 16 and 64), bit 8192 by `MONO_EP_SM100_DW_PRECISION=mxfp8` under MXFP8 (no BF16 x/dy/dz pool copies); the two features do not combine. |
| `MONO_EP_SM100_BACKWARD_PULL_BYTES` | two rows (8 KB) when whole rows fit, else `4096` | `4096` | SM100 backward NVLink pull size. |
| `MONO_EP_SM100_FORWARD_OPTIONS` | lookahead pull when the pull buffer is one row | `0` | SM100 forward option bits. |
| `MONO_EP_SM100_FORWARD_PULL_BYTES` | per shape | | SM100 forward pull size. |
| `MONO_EP_SM100_DW_DIRECT` | `1` | `0` (accumulate into a zero-filled output) | SM100 BF16 and MXFP8 backwards: the dW grouped GEMMs write their FP32 output directly and a small kernel zeroes only experts with no rows, instead of zero-filling all of dW first (5 GB per backward at the Beam shape). Not bitwise against `0` (FP32 accumulation order; FP64-anchored dW error unchanged, zero-row experts exactly zero). 8xB200 backward: 30B-A3B 4.19 to 4.14 ms, 235B-A22B 15.39 to 14.99 ms, Beam 35.50 to 34.06 ms. Python-side. |
| `MONO_EP_SM100_DW_PRECISION` | `bf16` | | `mxfp8` (needs `MONO_EP_SM100_BACKWARD_PRECISION=mxfp8`): the SM100 dW grouped GEMMs on E4M3 operands with one UE8M0 scale per 32 rows and column (DeepGEMM k-grouped `recipe=(1, 1, 32)`, MN-major). The x, dy and dz operands are requantized along the rows from the backward's own E4M3 pools (one Triton pass, 0.5-0.6 ms per Beam pool on B200), h from its BF16 pool, and the kernel then writes no BF16 x/dy/dz copies (option bit 8192). Relative dW error against an FP32 reference 6.7% instead of 6.2% with BF16 dW (BF16 backward 0.4%); dx and dscore unchanged. 4xB200 EP4, 16,384 tokens per GPU, same process: Beam backward 32.96 to 30.53 ms (SAC step 41.8 to 41.5), Qwen3-235B neutral (12.4 to 12.8 host wall with 1.4 ms less device time; the eager dW path is launch-bound there). Not a default. Evidence: FSDP2 dp_shard 2 x EP 2 with three AdamW steps passes with it (dW error 0.068 / 0.047 / 0.041 / 0.039 at steps 0-3 vs 0.062 / 0.040 / 0.031 / 0.029 with BF16 dW); 300-step loss curves of a random-init Qwen3-MoE (two seeds) sit 0.011-0.016 per step from BF16 against a 0.089 seed gap, final eval +0.014 / -0.017 (seed spread 0.033). |
| `MONO_EP_SM100_REQUANT_CONFIG` | `1,2,1` | `1,4,0` (the original launch) | Launch of the MXFP8 dW requantization (`requant_k_blocks`): `column_blocks,num_warps,reciprocal[,persistent]` (`/` also separates). The default runs two warps per 128 x 128 block and multiplies by the exact inverse of the power-of-two scale instead of dividing: bitwise equal to `1,4,0` (values and scale words) and a third faster on one B200 (Beam pools 2.25 -> 1.57 ms per backward, Qwen3-235B 1.61 -> 1.12, 30B 0.82 -> ~0.55). `persistent` > 0 strides a fixed row grid over the used blocks (no gain over the default on worst-case pools). 8xB200 EP8, same process, MXFP8 + MXFP8 dW + stored z: backward 21.22 -> 20.06-20.23 ms (Beam), 10.49 -> 9.70-9.76 ms (Qwen3-235B); dx bitwise; dW deviation equal to the run-to-run control; FSDP2 dp_shard 2 x EP 2 with three AdamW steps gives identical dW errors with either launch. |
| `MONO_EP_SM100_INKERNEL_DW` | `0` | | SM100 BF16: compute dW1/dW2 inside the persistent backward (option bit 512) instead of the grouped GEMMs after it. At most 32 local experts. 8xB200: +0.10 ms per backward at the Qwen3-30B-A3B shape (+0.04 with `_WIDE=1`), -1.49 ms (-4.9%, with `_WIDE=1`) at the Beam shape. |
| `MONO_EP_SM100_INKERNEL_DW_WIDE` | `0` | | With in-kernel dW: 256 x 256 tiles over 32-row stages (option bit 1024) instead of 256 x 128. |
| `MONO_EP_SM100_INKERNEL_DW_PER_BLOCK` | `2` | | With in-kernel dW: ready dW tiles the scheduler takes after each M-block (the rest run as experts finish). Runtime value, not part of the JIT key. |
| `MONO_EP_WORKSPACE_BOUND` | `auto` | `max` (pools for ranks x max tokens x top-k rows) | SM100 BF16 / MXFP8 backward pools (x, dy, h, dz and the E4M3 pools; the MXFP8 scale pools keep the worst case) sized to the rows this rank actually receives, padded to 128 per local expert. The forward reports its per-expert receive counts (copied to pinned memory without a sync) and the backward over the same inputs sizes its pools to that exact count before launching, so nothing overflows; the kernel traps past `pool_rows` as a host-bug guard. Pools start at 1.25 x tokens x top-k + 128 per local expert and only grow (to 1.1 x a larger need, logged). Records are keyed by the layer's expert weight storage with a per-layer stack, so they survive checkpoint recomputes (which rebuild the MoE input and routing as new tensors), FSDP2 regathers and gradient accumulation; checkpoint-recompute forwards do not record unless their layer has no record (reentrant checkpointing runs the original forward under `no_grad`); only forwards that can reach a backward record (grad mode on and an input or weight requiring grad), so eval, `no_grad` / `inference_mode` and generation forwards leave none; records of forwards whose backward never ran are dropped after every `optimizer.step()`, and a layer stack deeper than 64 drops its oldest entry with a warning; a backward with no record uses the worst case and logs a warning each time. 4 checkpointed layers x 2 accumulated micro-batches with `no_grad` and `inference_mode` eval forwards in between: 8 of 8 backwards found their record (non-reentrant and reentrant), no stale records, an abandoned forward's 4 records dropped by the next step. The SM100 BF16 backward's symmetric ring is also capped at the owned-backward floor (one wave of CTA pairs, `MONO_EP_BACKWARD_RING_WAVES`) instead of the forward's worst-case live-block sizing: EP4, 16,384 tokens per GPU, 149,760 -> 9,600 ring tokens, backward symmetric buffer 1.42 -> 0.68 GB (30B-A3B) / 2.75 -> 1.32 GB (235B-A22B), Beam unchanged (already at the floor); same-process backward 4.29 / 4.29 ms, 16.57 / 16.68 ms, dx bitwise. The forward ring keeps its worst-case sizing (the routed rows are unknown before launch). SM90 (H200 8x, 16,384 tokens per GPU): the BF16 and blockwise FP8 backwards take the same bounded pools and owned ring (four waves for BF16, which loses 3-4 % at one wave), and the blockwise forward reports its receive counts; resident after construction 20.7 -> 5.2 / 40.4 -> 9.7 / 38.6 -> 9.0 GB blockwise FP8 (30B-A3B / 235B-A22B / Beam) and 18.0 -> 5.5 / 35.4 -> 10.4 GB BF16; blockwise backward within 0.6 %, BF16 30B +0.5 % balanced / +2.4 % skewed at four waves; dx bitwise, FSDP2 dp_shard 2 x EP 2 passes with the record key stable across steps. dx bitwise vs `max`; FSDP2 dp_shard 2 x EP 2 passes. 8xB200, 16,384 tokens per GPU: resident after construction 18.1 -> 7.0 / 35.5 -> 13.2 / 41.0 -> 9.0 GB per rank (30B-A3B / 235B-A22B / Beam); backward unchanged; a backward issued right behind its own forward waits for the count (forward+backward of one layer +0.1-0.15 ms; in a model only the last layer of a step). Python-side. |
| `MONO_EP_FORWARD_RING_ROWS` | unset (1.25 x tokens x top-k, rounded to 128) | `MONO_EP_WORKSPACE_BOUND=max` | Routed-row budget the SM100 BF16 forward's symmetric ring is sized for when the workspace bound is `auto`. The forward scheduler gates L1 claims on the ring (an L1 block is claimed only after every L2 task one ring generation back is claimed), so a launch whose routing needs more live blocks runs gated and slower, never wrong; it switches on per launch from the received counts and is off whenever the routing fits. Wide experts (H x I >= 4096 x 1536) size the ring by the live-block bound for this budget; narrow experts give it one slot per budgeted row, because slot reuse inside a launch costs a dispatch-bound forward 4-8 %. Never above the worst-case ring. 8xB200, 16,384 tokens per GPU: forward symmetric buffer 2.15 -> 1.50 GB (30B-A3B), 4.19 -> 1.73 GB (235B-A22B), Beam unchanged (its live-block ring is already 3,840); forward+backward 5.78 / 5.80 ms, 19.06 / 19.08 ms balanced, skewed +0.0 / +1.3 %; y and dx bitwise against the worst-case ring for balanced, skewed and adversarial routing, and with a forced 2,048-row budget (gate active every launch). Other forwards (MXFP8, NVFP4, SM90) keep the worst-case ring and refuse a bounded buffer. |
| `MONO_EP_FORWARD_RING_TOKENS` | unset | unset | Exact forward ring tokens (sweeps), capped at the worst-case ring; `0` uses the live-block bound for the budget. |
| `MONO_EP_STORED_CONTEXT` | `auto` | `off` | SM100 BF16: a storing forward writes z (BF16 gate/up pre-activations) by its own pool row into one runtime-owned buffer, and the next backward with the same inputs skips its gate/up recompute GEMMs (option bit 4096) and reads z instead; dx is bitwise equal to the recomputing backward. One buffer serves one layer at a time, so it pays when the storing forward runs right before that layer's backward, i.e. the activation-checkpoint recompute. `auto` stores in checkpoint-recompute forwards when `hidden x intermediate >= 4096 x 1536` and the buffer fits in free device memory plus max(2 GiB, 5%) headroom, otherwise recomputes; it logs one line with the decision. `recompute` forces storing in checkpoint recomputes, `always` stores in every forward (only the last forward before a backward is used), `off` never stores. Combines with `MONO_EP_SM100_BACKWARD_PRECISION=mxfp8` (BF16 z also lowers the MXFP8 error: dx 5.3% against an FP32 reference instead of 6.8%) and with in-kernel dW. Nothing is stored when the fused op's output is saved instead (Axolotl `selective` checkpointing), since there is no recompute. Buffer: an index of `ranks x max tokens x top-k` int32 plus `capacity x 2I` BF16 rows; capacity starts at 1.25 x tokens x top-k + 128 per local expert and grows to 1.1 x the routed-row high-water mark each backward reports (read one step late, no host sync); a forward whose rows exceed it raises a device flag and that layer's backward recomputes (same results, about the off time). 8xB200, 16,384 tokens per GPU, balanced: buffer 0.51 / 1.02 / 1.72 GB at 30B-A3B / 235B-A22B / Beam (worst case 3.2 / 6.4 / 10.7 GB); checkpointed layer step off vs auto 7.78 / 7.83 ms (auto declines), 25.2 / 22.4 ms, 57.5 / 52.3 ms. |
| `MONO_EP_STORED_CONTEXT_ROWS` | unset | | Fixed stored-z capacity in forward pool rows (no growth); for testing the overflow fallback. |
| `MONO_EP_BACKWARD_RING_WAVES` | `1` | `0` (the forward's interleaved-schedule ring sizing) | Owned-backward ring floor for the BF16 (SM90, SM100), MXFP8 (SM100) and blockwise-FP8 (SM90) backwards, in waves of owning clusters: one wave is `num_sms x 128` tokens on SM90 and `num_sms / 2 x 128` on SM100. It only raises the ring, so the Qwen3-30B-A3B shape is unchanged; wide-expert shapes (hidden 5120) gain most (the MXFP8 transport buffer took the floor late: Beam EP4 backward 37.9 to 28.4 ms, kernel 23.0 to 12.8, dispatch ring wait 15.4 to 7.0 ms per warp). The ring size is a kernel template parameter. |
| `MONO_EP_NVFP4_BACKWARD_RING_WAVES` | `1` | `0` | The same floor for the SM100 NVFP4 LoRA backward. |
| `MONO_EP_SM100_NVFP4_BACKWARD_OPTIONS` | `2051` | `3` (before the lean tail), `0` | NVFP4 LoRA backward bits: 1 staged h/dz pool stores, 2 TMEM release before the recompute/dH math, 2048 lean recompute tail (Dh and recompute tasks skip the per-task fence; each CTA fences once at its block's last recompute tile). Its own option space: bit 2048 here is not the BF16 backward's MXFP8 bit. |
| `MONO_EP_SM100_NVFP4_BACKWARD_PULL_BYTES` | whole x and dy rows with lookahead | | Chunked dispatch pull for the NVFP4 backward: four chunks, each a multiple of 128 bytes, at most 4096 and smaller than an x+dy row. |
| `MONO_EP_SM100_NVFP4_BACKWARD_STAGES` | the most that fit shared memory | | NVFP4 backward TMA/UMMA stages. |
| `MONO_EP_SM100_NVFP4_OPTIONS` | `1` when the pull buffer holds one row, else `0` | `0` | NVFP4 LoRA forward bits: 1 lookahead dispatch pull. |
| `MONO_EP_NVFP4_REQUANT` | `cuda` | `triton` | Backend of the per-step NVFP4 merged-weight rebuild (requant cache): `cuda` is the SM100 kernel and writes the Triton kernel's bytes; `cuda_fused` (opt-in) quantizes the transposed forms from the merged tile, which is a numerics change for `w1t` / `w2t`. Python-side, not a JIT key. |
| `MONO_EP_NVFP4_REQUANT_FFMA2` | `1` | `0` | Paired FP32 FMA (FFMA2) in the CUDA requant's delta math; `0` builds the scalar variant, 2-6% slower. |
| `MONO_EP_NVFP4_LORA_WGRAD` | unset (grouped GEMMs) | `loop` (per-expert reference loop) | LoRA adapter weight gradients of the NVFP4 path. Python-side. |
| `MONO_EP_NVFP4_REQUANT_VERBOSE`, `MONO_EP_SM100_NVFP4_PHASE_TRACE` | unset | | Diagnostics: CUDA requant build log, NVFP4 forward phase trace. |
| `MONO_EP_SM90_FP8_TRAP_TIMEOUTS`, `*_PHASE_TRACE`, `*_DIST_TRACE`, `*_DIAG_ABLATE` | unset | | Diagnostics: trap-only timeouts for compiler experiments, per-phase and per-rank traces, epilogue ablations. Not for training. By name: `MONO_EP_SM90_FP8_PHASE_TRACE`, `MONO_EP_SM90_FP8_FORWARD_PHASE_TRACE`, `MONO_EP_SM90_FP8_DIST_TRACE`, `MONO_EP_SM90_FP8_FORWARD_DIST_TRACE`, `MONO_EP_SM90_FP8_DIAG_ABLATE`, `MONO_EP_SM90_FP8_FORWARD_DIAG_ABLATE`, `MONO_EP_SM100_PHASE_TRACE`, `MONO_EP_SM100_FORWARD_PHASE_TRACE`, `MONO_EP_SM100_DIAG_ABLATE`, `MONO_EP_SM100_FORWARD_DIAG_ABLATE`. |

Every kernel selector is part of the JIT key, so changing one compiles a new kernel variant under
`DG_JIT_CACHE_DIR`; the selectors marked Python-side change only host code.

## Troubleshooting

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
  `expert_parallel_mono_ep_mode: dispatch` when only the fused mode is excluded (LoRA,
  quantized experts, biases, other activations).
- **Restoring a previous default.** Every selector above lists the value that restores the
  behaviour before its current default; set it in the environment of every rank.
