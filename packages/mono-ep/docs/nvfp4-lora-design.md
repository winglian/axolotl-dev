# NVFP4 LoRA expert path (`precision="nvfp4_lora"`) on the SM100 kernels

Branch `features/mono-ep-nvfp4-lora`, off `features/mono-ep` @ 9ae994be97 with
`features/nvfp4-requant` (packed requant kernel, requant cache, ModelOpt audit)
cherry-picked and the b200-r3 SM100 kernels and weight cache ported. This note
fixes the design before kernel work. Everything below is behind the
`nvfp4_lora` precision selector; the BF16 SM100 kernels, launchers and defaults
are not edited, so the BF16 path stays byte-identical.

Targets: `reflection/Beam-1.0-501B-A23B-NVFP4` (256 experts top-8, hidden 5120,
intermediate 2560, routed experts W4A4 NVFP4: E2M1 codes, E4M3 block scales
per 16 K, one fp32 per-tensor scale per expert and projection, static
`input_scale` per linear) and `nvidia/Qwen3.6-35B-A3B-NVFP4`. The base experts
stay frozen packed NVFP4 straight from the checkpoint; only the LoRA factors
`A [E*r, K]`, `B [N, r*E]` train (bf16 or fp32 masters, PEFT layout), no bf16
or fp32 master of the base exists anywhere. Every optimizer step the requant
kernel rebuilds `Q(bf16(DQ(W)) + scaling * B_e @ A_e)` on the checkpoint's
fixed per-tensor scales and the kernels consume those bytes for the whole step
(all micro-batches). The forward is the merged weight; gradients are
straight-through on the quantizer: `dW_merged` is never formed, `dA`, `dB`
come from rank-`r` projections of the pooled operands.

## 1. MMA kind and scale format

`tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X` with
E2M1 operands and **UE4M3** scales, swap-AB 2-CTA UMMA exactly as the BF16
kernels (UMMA_M = 256 weight rows, 128 per CTA; UMMA_N = BLOCK_M tokens).

- `scale_vec::4X` is the 16-element block variant: one scale byte per 16 K,
  four bytes (one 32-bit TMEM column) per 64 K, so **one instruction covers
  K = 64** and the instruction descriptor's SF_ID is always 0. The CUTLASS
  descriptor is `InstrDescriptorBlockScaled` with `a_format = b_format =
  MXF4Format::E2M1 (1)`, `scale_format = UE4M3 (0)`. `deep_gemm/ptx/tcgen05.cuh`
  only wraps the `kind::mxf4 ... block32` (2X, UE8M0) form; a
  `SM100_MMA_MXF4NVF4_2x1SM_SS` wrapper is added next to it (same operand
  order as CUTLASS `SM100_MMA_MXF4NVF4_2x1SM_SS`, VS = 16).
- Why this kind: the checkpoint is NVFP4 (16-wide E4M3 block scales). `kind::mxf4`
  needs UE8M0 per 32, which cannot carry an E4M3 scale without re-rounding
  the weights off the shipped grid (the requant branch's reuse-grid exactness
  and the merge writer's byte round trip depend on that grid).
  `kind::f8f6f4` with E2M1 operands has no block scales and runs at the FP8
  rate; `mxf4nvf4` runs at the FP4 rate (2x FP8, 4x BF16 per SM per clock).
- Both operands of one instruction share the kind, so **activations and
  gradients that meet an NVFP4 weight are quantized to NVFP4 with E4M3
  per-16 block scales too** (W4A4 forward, W4G4 backward). Section 3 says
  how their second-level scales are chosen so E4M3's range (2^-6 .. 448
  normal, 2^-9 subnormal) is not the limiting factor.
- Scale factors live in TMEM, filled per 64-K step by
  `tcgen05.cp.cta_group::2.32x128b.warpx4` (CUTLASS
  `SM100_UTCCP_4x32dp128bit_2cta`) from 512-byte smem atoms, each atom =
  128 rows x 4 K-blocks, byte `(r % 32) * 16 + ((r / 32) % 4) * 4 + (k % 4)`,
  landing in TMEM lane `r % 32`, column `r / 32` (+ atom index * 4), byte
  `k % 4`. The requant kernel's `scale_atoms` and the MXFP8 kernel's
  `transform_sf_token_idx` already produce exactly this byte order, so no
  per-step repack of weight scales and no per-token re-layout exist; only the
  K granularity changes from 128 (4 x 32-wide) to 64 (4 x 16-wide) per word.

## 2. Data layouts

### 2.1 Merged weights (requant cache output, written in the kernel's layout)

| tensor | shape / dtype | role |
|---|---|---|
| `w1.qdata` | `u8 [E, 2*inter, hidden/2]`, **interleaved rows** | L1 forward and backward recompute B operand (K = hidden) |
| `w1.scale_atoms` | `u8 [E, 2*inter/128, hidden/64, 512]` | its block scales, UTCCP atom order |
| `w2.qdata` | `u8 [E, hidden, inter/2]` | L2 forward B operand (K = inter) |
| `w2.scale_atoms` | `u8 [E, hidden/128, inter/64, 512]` | |
| `w1t.qdata` | `u8 [E, hidden, 2*inter/2]`, natural `[gate \| up]` K order | backward dx B operand (K = 2*inter) |
| `w1t.scale_atoms` | `u8 [E, hidden/128, 2*inter/64, 512]` | fresh scales along `2*inter` |
| `w2t.qdata` | `u8 [E, inter, hidden/2]` | backward dH B operand (K = hidden) |
| `w2t.scale_atoms` | `u8 [E, inter/128, hidden/64, 512]` | fresh scales along `hidden` |
| `pts_w1 [E, 2]`, `pts_w2 [E, 1]`, `pts_w1t [E, 1]`, `pts_w2t [E, 1]` | fp32 | per-tensor scales, folded in the epilogues |

- Packing: two codes per byte, low nibble = even `k`, K contiguous. This is
  the checkpoint's packing and what `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B`
  expects, so the TMA boxes are `BLOCK_K = 256` logical FP4 = 128 bytes per
  row with 128-byte swizzle (`fp4_unpacked_smem = false`; deep_gemm's
  `make_tma_*_desc` already size the smem box as `swizzle * 2` and count the
  gmem inner dim in logical elements; `kPackedFP4` is `torch::kInt8`, so the
  `u8` buffers are passed as `int8` views). UMMA descriptors use
  `make_umma_desc<Major::K, rows, 256, 128>` with `float_e2m1_t`
  (`get_smem_pack_factor() == 2`, and `advance_umma_desc_lo` already divides
  byte offsets by the pack factor).
- The BF16 kernels load `w1` through a 4-D TMA map `(k, n/2, half, expert)`
  on the 8-row gate/up **interleaved** copy (`weight_layouts(...,
  "w1_interleaved")`: row `16j + 8h + i` holds gate (`h = 0`) or up (`h = 1`)
  output column `8j + i`), so each CTA's 128-row tile carries the gate and up
  rows of the same 64 output columns and the SwiGLU happens in registers. The
  requant kernel writes that order directly: the frozen base `qdata`/`scale`
  are permuted once at load time (static tensors), and the kernel's `lora_B`
  tile load takes a row-permutation vector (`b_ptr + perm[rows] * ...`), so
  nothing is repacked per step. `scale_atoms` follow the output row order, so
  the atoms are interleaved-consistent for free. The transposed forms keep
  natural order, which is what the dz pool (natural `[dgate | dup]` columns)
  and dx need.
- Transposed forms: the requant kernel already holds the fp32 merged tile
  `[BM = 64 rows of N, BK = 64 .. 128 of K]`; 64 rows are four 16-blocks along
  N, so the same tile is quantized a second time with blocks along N and
  stored transposed (`[K, N/2]` bytes, one word = 8 consecutive N codes of one
  K row; atoms indexed by `(k / 128, n / 64)`). The transposed scales are
  fresh (`amax / 6 / pts`, torchao clamp `[2^-6, 448]`); no shipped scale
  exists along N to reuse. Per-tensor: `pts_w1t = max(pts_gate, pts_up)` and
  the tile values are pre-multiplied by `pts_row / pts_w1t` before
  quantization, so `w1t` has a single per-tensor scale even for a checkpoint
  whose gate and up scales differ (Beam: ratio exactly 1.0, so this is a
  no-op there). Consequence, stated up front: **dx is computed with a weight
  that sits on a different NVFP4 grid than the forward's** (same fp32 merged
  value, re-quantized along the other axis). The expected deviation is one
  NVFP4 quantization noise (relative Frobenius ~2-3e-2 per expert matrix),
  the same order as quantizing the gradient operand itself; the parity
  tables report dx against an oracle built from `w1t` (kernel correctness)
  and against one built from the forward `w1` (cost of the design). A 16x16
  two-dimensional block grid would make both orientations identical but
  would leave the shipped 1x16 grid and the exact export round trip, so it
  is not done.
- Memory per EP rank for the forward forms is the checkpoint size (Beam, 32
  local experts: 630 MB of codes, 79 MB of scales); the transposed forms
  double that. BF16 dequant of the base would be 4x the codes per copy.

### 2.2 Token operands and their scales in the symmetric / pool buffers

`MonoEPBuffer` gains a scale-factor granularity (`sf_gran = 32` for the
existing MXFP8 layout, `16` for NVFP4; unchanged sizes for every existing
caller). With `with_sf` and `sf_gran = 16`:

| buffer | per token | notes |
|---|---|---|
| `input_token_buffer` (source rank) | `hidden/2` bytes FP4 + `hidden/16` bytes scales + `rowscale` fp32 | **not used**: tokens leave the source rank in BF16 (unchanged) and are quantized by the destination's dispatch warps, which already hold the row in shared memory. This keeps the NVLink payload 2 bytes/elem but avoids a second quantization pass on the source and keeps the dispatch pull code identical to the BF16 kernel up to the store. |
| `l1_token_buffer` (ring) | `hidden/2` bytes | K-major packed FP4 rows, TMA A operand, box `256 x LOAD_BLOCK_M`, SW128 |
| `l1_sf_buffer` (ring) | `hidden/16` bytes | `uint32 [hidden/64][num_sf_ring_tokens]` MN-major; token position `ring_block * SF_BLOCK_M + transform_sf_token_idx(r)` (MXFP8 scheme, `SF_BLOCK_M = align(BLOCK_M, 128)`). One stage's SFB tile is the TMA box `<SF_BLOCK_M, BLOCK_K/64>` = `4 x 256 x 4 B = 4 KB`, two atoms per K-step. |
| `l1_topk_weights_buffer` | fp32 | unchanged (routing weight) |
| `l2_token_buffer` (ring) | `inter/2` bytes | FP4 h written by the L1 epilogue |
| `l2_sf_buffer` (ring) | `inter/16` bytes | `uint32 [inter/64][num_sf_ring_tokens]`, same token transform |
| `combine_token_buffer` | BF16 | unchanged |

Backward pools (`MoEBackwardBuffers`, rows = `world * max_tokens * topk +
experts * 128`, `BLOCK_M = 128` so `SF_BLOCK_M = BLOCK_M` and one atom per
K-step): the BF16 pools `x, dy, h, dz` stay (they feed `dA`, `dB`); new FP4
pools `x4 [rows, hidden/2]`, `dy4 [rows, hidden/2]`, `dz4 [rows, inter]`
(= `2*inter/2` bytes), scale pools `x4_sf [hidden/64][rows]`,
`dy4_sf [hidden/64][rows]`, `dz4_sf [2*inter/64][rows]` (uint32, token
position `pool_block * 128 + transform(r % 128)`), and row scales
`rowscale_dy [rows]`, `rowscale_dz [rows]` fp32. Weight-side SFA tiles for a
stage are the requant atoms `[e][n_block][k_step .. k_step+3]` = 2 KB
contiguous, loaded with one `cp.async.bulk` per stage per CTA on the weight
TMA warp's barrier (no tensor map needed).

## 3. Activation and gradient quantization

All quantization uses the hardware converts `cvt.rn.satfinite.e4m3x2.f32`
(scale) and `cvt.rn.satfinite.e2m1x2.f32` (codes), torchao numerics: block
scale `amax / 6 / pts` clamped to `[2^-6, 448]`, codes
`clamp(x / (sf * pts), -6, 6)` RNE. `pts` is the second-level (per-tensor or
per-row) scale and is what makes the E4M3 range sufficient: E4M3 normals span
2^14.8, so blocks more than ~2^15 below the operand's `pts * 6 * 448` lose
precision or flush to zero.

| operand | where quantized | second-level scale `pts` | folded in |
|---|---|---|---|
| x (forward L1 A, backward recompute A) | dispatch warps, from the BF16 row in the pull buffer, per 64-K chunk per lane: 4 blocks, 32 B of codes (`st.global.v4` x 2, 1 KB contiguous per warp per 64 K) and one scale word at the transformed SF slot | **static** `input_scale` of `gate_up_proj` from the checkpoint (`nvfp4_input_scale_mode = static`, default). `unit` (`pts = 1`) exists for A/B against the dynamic-per-16 fp4_cute engine. | L1 epilogue: accumulator x `pts_x * pts_w1[gate]` for gate lanes, `pts_x * pts_w1[up]` for up lanes, before SwiGLU. Backward recompute: same, so z recomputes bitwise from identical operands. |
| h (forward L2 A) | L1 epilogue. Each warp owns 32 interleaved N rows = 16 h columns = **one 16-block per token row**; the 16 values of a row sit in 4 lanes of the `16dp256b` fragment, so the block amax is two `shfl.xor` steps. Codes: 2 bytes per lane, assembled to 8 bytes per row through the existing STSM staging; scale byte `(row, block = n_block*4 + warp)` = word `n_block` of `l2_sf_buffer`, byte `warp_idx_in_wg` (the MXFP8 kernel's byte-store scheme, one word per row per L1 task). | **static** `input_scale` of `down_proj` | L2 epilogue: x `pts_h * pts_w2` before the BF16 cast / NVLink scatter |
| dy (backward dH A) | dispatch warps, chunkwise like x | **per row, dynamic**: `rowscale_dy[t] = amax(dy[t, :]) / (6 * 448)` computed by the launcher with one fused torch reduction over the local `dy` and published in the symmetric buffer next to `dy` (`[max_tokens]` fp32); the dispatch warp reads the remote scalar with the row (4 bytes, overlapped like the routing weight), quantizes without needing the whole row, stores `rowscale_dy[pool_row]` | dH epilogue: x `rowscale_dy[row] * pts_w2t` |
| dz (backward dx A) | **completer pass**: the recompute epilogues write BF16 `dz` to the pool as today and arrive on the L2 readiness mask with an `atom.xor` that returns the old mask; the warpgroup whose arrival completes the block's mask converts that block: per row amax over `2*inter` -> `rowscale_dz`, 16-blocks -> `dz4` + `dz4_sf`, then releases a generation-tagged `dz4_ready[ring_block]` counter that the dx A-loader waits on instead of the mask. ~2 KB read + 0.6 KB written per row from L2-resident data; 240 .. 128 rows per block on 256 threads, ~3 us (Qwen) .. 10 us (Beam) per block, once per block. | **per row, dynamic**, exact | dx epilogue: x `rowscale_dz[row] * pts_w1t` |

Why per row for gradients: a unit scale flushes most gradient blocks
(magnitudes 1e-5 .. 1e-2) below E4M3; a per-tensor scale would need an
all-reduce before dispatch for `dy` and does not exist before the kernel for
`dz`; a per-row factor is exact because it multiplies a whole A row and
factors out of every dot product of that row, and it costs one fp32 per row.
Gate/up per-tensor scales cannot be factored out of the dx reduction (they
would sit along K), which is why `w1t` carries one per-tensor scale
(Section 2.1). No Hadamard transform or stochastic rounding is used for the
gradient operands in this round; the parity envelope will say whether RNE
per-row NVFP4 gradients are within the fp4_cute engine's existing tolerance.
The dispatch warps count 448-saturated blocks into an optional diagnostic
counter (`MONO_EP_SM100_NVFP4_SAT_COUNT`) so static-scale drift under LoRA
training is observable.

Cost: x/dy quantization adds ~500 .. 1000 instructions per row per dispatch
warp (~220 rows per warp at the EP8 contract), overlapped with the NVLink
pulls those warps are already waiting on; h quantization adds two shuffles
and a convert per 16 values to the L1 epilogue.

## 4. Forward kernel (`sm100_nvfp4_mono_ep.cuh`)

A copy of the b200-r3 BF16 forward with these deltas (same warp roles,
scheduler, ring protocol, combine, lookahead pull, options):

- `BLOCK_K = 256` FP4 (128-byte rows, SW128) for both layers; `MmaKind::NVFP4`
  in the heuristics (`get_sf_uttcp_aligned_block_sizes` returns the 128-aligned
  sizes as for MXFP8; `block_k = 256`; per-stage smem counts the two SF tiles).
- TMA-A warp: token tile `256 x LOAD_BLOCK_M` + SFB tile `<SF_BLOCK_M, 4>` from
  `l1_sf_buffer` / `l2_sf_buffer` (MN-major map, as the MXFP8 kernel's
  `tensor_map_sfa`). TMA-B warp: packed weight tile (4-D map for `w1`, 3-D for
  `w2`) + one 2 KB bulk copy of the weight atoms.
- MMA warp: per stage 4 x (two UTCCPs for the token atoms, one for the weight
  atom, one `mxf4nvf4` MMA with K = 64), SF columns reused across K-steps
  (issue-ordered on the same thread, as the MXFP8 kernel and DeepGEMM's 1D1D
  FP8 GEMM do).
- L1 epilogue: per-tensor fold, SwiGLU, FP4 h + scales to the L2 ring
  (Section 3); the L2 `full` mask / `empty` counts unchanged. L2 epilogue:
  per-tensor fold, BF16 combine scatter unchanged.
- Shared experts, FP8 combine: not supported under `nvfp4_lora` (host assert).

Shared memory per stage at `BLOCK_M = 240`: A (tokens) `120 x 128 B =
15,360`, B (weights) `128 x 128 B = 16,384`, SFB `4 x 256 x 4 = 4,096`, SFA
`4 x 128 x 4 = 2,048` -> **37,888 B** per stage holding 256 K (the BF16 stage
is 31,744 B for 64 K). Fixed storage (pull buffers, expert counts, L1/L2 output
staging, barriers) is within 1 KB of the BF16 kernel's (the L1 staging shrinks
4x), so 4 .. 5 stages fit in 227 KB. `MONO_EP_SM100_NVFP4_BLOCK_K = 128`
(64-byte rows, SW64) is kept as a knob: ~19 KB per stage, 9 .. 10 stages,
in case the deeper ring beats the wider swizzle on the real traffic.

Tensor memory: accumulators `2 x 240 = 480` columns + SFA 4 + SFB 8 =
**492 <= 512** (the MXFP8 kernel's budget; SF columns are rewritten per
64-K step). `BLOCK_M = 240` therefore stays the forward default.

## 5. Backward (`sm100_nvfp4_mono_ep_backward.cuh` + Python)

Same phase structure and scheduler as the r3 BF16 backward (Dh, RecomputeLow,
RecomputeHigh, Dx per owned M-block, `BLOCK_M = 128`, dscore partials, combine
of dx), operands:

| phase | A (tokens) | B (weights) | K | epilogue scale |
|---|---|---|---|---|
| Dh: `dH = dy @ W2_mergedᵀ` | `dy4` + `dy4_sf` | `w2t` + atoms | hidden | `rowscale_dy[row] * pts_w2t`, then retained BF16 dH atoms as today |
| Recompute: `z = x @ W1_mergedᵀ` | `x4` + `x4_sf` | `w1` (interleaved) + atoms | hidden | `pts_x * pts_w1[gate/up]`; SwiGLU derivative with dH and the routing weight -> `h`, `dz` BF16 pools, dscore partials (unchanged code) |
| Dx: `dx = dz @ W1_merged` | `dz4` + `dz4_sf` (completer pass) | `w1t` + atoms | 2*inter | `rowscale_dz[row] * pts_w1t`, BF16 NVLink scatter unchanged |

Dispatch warps pull `x` and `dy` rows (two-row pull as today), store the BF16
pools, and additionally quantize both rows (x static, dy per-row) into the FP4
pools and scale pools. Shared memory per stage at `BLOCK_K = 256`: A `64 x 128
= 8,192`, B `16,384`, SFA `2,048`, SFB `2,048` -> 28.7 KB (BF16: 24.6 KB per
64 K), 5 .. 6 stages. Tensor memory `2 x 128 + 4 + 4 = 264` columns; the
four-stage TMEM opt-in (`options & 4`) would need 520 and is rejected for this
precision (it is not a default).

Ring floor. The symmetric buffer's L1 ring is sized for the forward's
interleaved schedule (`get_num_max_live_pool_blocks`). The backward's owned
schedule keeps one pool block per CTA pair live through all four phases, so at
wide experts that sizing starves it: at Beam (hidden 5120, inter 2560) the ring
holds 15-30 blocks for 74 CTA pairs and the dispatch warps spend most of the
kernel waiting on ring slots. `Nvfp4PersistentBackward` passes a floor of
`waves * (sms / 2) * 128` tokens to `get_symm_buffer_for_mono_ep`
(`num_min_ring_tokens`), default one wave; the floor only raises the ring, so
shapes whose forward sizing is already larger (the Qwen3-30B contract) are
unchanged. The BF16 / FP8 persistent backward takes the same floor opt-in
(`MONO_EP_BACKWARD_RING_WAVES`, default 0) so its defaults stay byte-identical.

Dispatch pull. Whole `x + dy` rows per warp (`hidden * 4` bytes) are the
default; at hidden 5120 that leaves 3 pipeline stages instead of 5. A chunked,
double-buffered pull (`MONO_EP_SM100_NVFP4_BACKWARD_PULL_BYTES` = 8192, 4096 or
2048: two slots of one x chunk and one dy chunk; x quantized per chunk with the
static scale, dy stored BF16 per chunk and quantized from the pool once the row
amax is known) restores 5-6 stages and is bitwise identical, but measured no
faster once the ring floor is in (the stage count stops mattering and the dy
re-read costs the dispatch warps), so it is opt-in only.

**dA, dB run outside the kernel** (`weight_grad_lora`, replacing
`weight_grad_bf16`), on the BF16 pools grouped by `expert_starts/counts`:

```
W1: P = dz @ B1_e            [rows, r]      dA1_e = scaling * Pᵀ @ x        [r, hidden]
    Q = x  @ A1_eᵀ           [rows, r]      dB1_e = scaling * dzᵀ @ Q       [2*inter, r]
W2: P = dy @ B2_e            [rows, r]      dA2_e = scaling * Pᵀ @ h        [r, inter]
    Q = h  @ A2_eᵀ           [rows, r]      dB2_e = scaling * dyᵀ @ Q       [hidden, r]
```

Eight grouped GEMMs (`torch._grouped_mm` with offsets, fp32 accumulation,
grads cast to the master dtype). FLOPs scale with `r` instead of `N`
(~2e11 per layer at the EP8 contract vs ~2.5e12 for the full `dW`), so this
is bandwidth-bound on two reads of the pools (~0.5 ms at the contract vs
~1.2 ms for today's full-rank `dW`). Reasons against epilogue tasks: the
rank-`r` products need tensor cores on operands that are in TMEM/registers
only slice-wise (`dz` 64 columns per task), the pooled operands are needed
anyway for the second factor, and the epilogue warps are the backward's
critical resource (phase traces, b200-r3). An in-kernel `P`/`Q` projection
fused into the pool store is a follow-up if the 0.5 ms shows in the 8xB200
numbers. The base weights get no gradient; `dscore` flows as today.

Autograd: `PersistentMoE.forward(x, topk_idx, topk_w, A1, B1, A2, B2)` saves
the routing, the pools' row map and the step key; `backward` returns `dx,
dscore, dA1, dB1, dA2, dB2`. The merged weights are not inputs of the autograd
graph (they are cache entries keyed by the adapter version), which is the
straight-through semantic the merge-aware path already has.

## 6. Requant cache and weight-cache epoch wiring

Dependency direction (Wing, 2026-11-10): `packages/mono-ep` keeps no Axolotl
import. The requant kernel, references and cache are a temporary copy in
`mono_ep/nvfp4_requant/` taken from `features/nvfp4-requant-default` @
227adde535; kernel-layout and transposed-output additions go on that copy
(commits titled "NVFP4 requant: ..."), and a later follow-up replaces the
Axolotl modules with imports from `mono_ep.nvfp4_requant`. The simpler
step-keyed cache used by the kernel path lives in `mono_ep/nvfp4_lora.py`
(`RequantCache` + `attach_post_step_hook`, invalidate-on-step); the vendored
`nvfp4_requant_cache` (adapter version, step keys, side stream) is kept for the
follow-up.

- `nvfp4_lora` bypasses `weight_layouts` (the requant output already is the
  kernel layout). One `LayerRequant` per MoE layer builds `{w1, w2, w1t, w2t}`
  from the frozen base and the live adapters (`full_tensor()` on FSDP2
  DTensors inside the build, the `fsdp2_cache_check.py` pattern, so the build
  runs on every rank inside the hook) with `requant_lora_packed` (forward
  forms with `scale_mode = "reuse"` and the shipped per-tensor scales;
  transposed forms with fresh scales, added as a second output set of the
  same launch). Keyed `RequantKey(adapter_version, step)`.
- `attach_post_step_hook(optimizer, cache, layers, adapter_version)` refreshes
  every layer after `optimizer.step()` on the side stream; `PersistentMoE`
  calls `cache.get(layer, key)` in the forward (stream wait on the build
  event). A stale or missing key is a hard error in training (never a silent
  fallback to the previous step's weights); `AdapterVersion.bump()` on
  adapter swaps invalidates everything.
- mono-ep's `invalidate_weight_caches()` / `set_weight_cache_epoch()` and the
  global `register_optimizer_step_post_hook` are routed to
  `RequantCache.invalidate` for this precision, so the FSDP2 cache validators
  (`validate_fsdp2_weight_cache.py`, `fsdp2_cache_check.py`) and the
  `MONO_EP_WEIGHT_CACHE_*` knobs keep their meaning.
- TMA descriptors are built per launch on the host from the cache tensors (as
  the BF16 launcher does), so a new step's buffers need no kernel-side epoch.
- The requant cost per step is reported separately (forward forms, transposed
  forms) and amortized per micro-batch in the timing tables; the transposed
  quantization roughly doubles the requant's store traffic, not its delta
  GEMM.

## 7. Plumbing and selectors (everything opt-in)

- `api.PRECISIONS += ("nvfp4_lora",)`, `_NATIVE_BACKWARD += {(10, "nvfp4_lora")}`;
  `MonoEPRuntime(..., precision="nvfp4_lora", input_scales=...)` takes the
  checkpoint's per-layer `input_scale` pair; `MmaKind::NVFP4` in
  `common/types.cuh` and `heuristics/mono_ep.hpp`; new binding
  `nvfp4_mono_ep` / `nvfp4_mono_ep_backward` in `csrc/apis/mono_ep.hpp`.
- Env knobs (unset = defaults above): `MONO_EP_SM100_NVFP4_BLOCK_K` (256 | 128),
  `MONO_EP_SM100_NVFP4_STAGES`, `MONO_EP_NVFP4_INPUT_SCALE_MODE` (static |
  unit), `MONO_EP_SM100_NVFP4_SAT_COUNT` (diagnostic). The BF16 options
  bitmasks apply unchanged where the code is shared.
- Backward knobs: `MONO_EP_SM100_NVFP4_BACKWARD_STAGES` (at most the deepest
  that fits), `MONO_EP_SM100_NVFP4_BACKWARD_OPTIONS` (bits 1 and 2),
  `MONO_EP_SM100_NVFP4_BACKWARD_PULL_BYTES` (whole rows by default, or a
  chunked 8192 / 4096 / 2048), `MONO_EP_NVFP4_BACKWARD_RING_WAVES` (ring floor
  in waves of CTA pairs, default 1, 0 = forward sizing),
  `MONO_EP_NVFP4_LORA_WGRAD=loop` (per-expert adapter-gradient fallback).
- Axolotl seam: `expert_parallel/args.py` precision literal, `plugin.py`
  accepts a LoRA adapter only for `nvfp4_lora` on a packed NVFP4 base and
  wires the requant cache + post-step hook, `experts_fn.py` passes the
  adapter tensors; this is the last step and only after the kernels pass
  parity.
- Checkpoints without `input_scale` (weight-only NVFP4 exports) run with
  `pts = 1` and a warning; a calibration pass that derives them is a
  follow-up, not part of this round.

## 8. Validation and error budget

1. Forward, single B200, EP world 1: `nvfp4_lora` vs an fp32 oracle from the
   same packed bytes (`dequantize_nvfp4_packed_ref`) with (a) fp32 activations
   and (b) activations fake-quantized with the same static scales in torch,
   and vs the fp4_cute grouped GEMM on the same merged bytes; envelope next to
   fp4_cute's `rtol = 1e-2, atol = 1e-2 * max|ref|` (observed rel-Fro
   2.2-2.5e-3 for W4A4 on these shapes). Routing balanced and skewed, Qwen3
   and Beam layer shapes, with and without the lookahead pull.
2. Backward, single B200: dx, dscore, dA, dB vs the fp32 oracle; dx reported
   against the `w1t` oracle and the forward-`w1` oracle (Section 2.1).
3. FSDP2 dp_shard x EP (2 x 2 on 4 B200), two optimizer steps, adapters train,
   base frozen, requant rebuilt per step by the post-step hook; after each
   step the kernel output and the adapter gradients match a torch reference
   computed from the live (gathered) adapters and the freshly requantized
   merged weight.
4. 8xB200 same-allocation interleaved timing (5 warmups / 15 samples,
   balanced and skewed routing, forward / backward / checkpointed 2F+B):
   `nvfp4_lora` vs the BF16 persistent path vs the fp4_cute requant-cache
   path with torch all-to-all, Qwen3-30B-A3B EP8 contract (16384 tok/GPU,
   hidden 2048, inter 768, 128 experts top-8) and the Beam-shaped layer
   (hidden 5120, inter 2560, 32 local experts). Requant cost per step shown
   separately and amortized.
5. BF16 default unchanged: the existing b200-r3 BF16 parity and paired-timing
   scripts are rerun once on the branch and must reproduce their numbers.

Known deviations to measure, not hide: the transposed-grid dx (2.1), RNE
per-row NVFP4 gradients (3), static-scale saturation under adapter drift (3).
