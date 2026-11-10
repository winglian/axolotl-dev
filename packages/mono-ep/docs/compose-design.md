# Composing mono-ep dispatch/combine with a local experts kernel

Status: implemented on `features/mono-ep-compose` as `expert_parallel_mono_ep_mode: dispatch`
(and the `auto` fallback); the transfer rewrite of section 5.3 / 5.4 (bandwidth-shaped
pull/push in `peer_stream.py`, one fused combine reduction, one barrier per op, opt-in compact
receive rows) sits on top. This note records the survey of the package's peer path that the
design rests on, the gaps that forced a new op pair rather than a reuse of `PeerSchedule`,
and the contract of the composable mode.

## 1. What the peer path provides today

The package has two peer-memory (NVLink symmetric memory) paths. Neither exposes
`dispatch(x, topk_idx, topk_w) -> (recv_x, recv_idx, recv_w)` / `combine(y_local) -> y` as
autograd ops independent of the expert GEMM.

### 1.1 `peer_runtime.PeerSchedule` (the "peer control" timing)

`PeerSchedule.__call__` is one `autograd.Function` (`_PeerFunction`) that owns dispatch,
expert compute and combine:

| Stage | Op | What it does |
|-------|----|--------------|
| dispatch | `mono_ep::peer_packed_bf16_publish` / `peer_packed_quantize_publish` (`peer.py`) | one launch per destination rank: gathers this rank's routed rows into the destination's symmetric buffer in **expert-packed** layout (`DeviceExpertLayout`, 128-row aligned per expert), writes routing metadata (`ids`, source `tokens`, `slots`, `ends`, `scores`) and releases per-expert readiness flags (system-scope fences, `queue[task, 3]`). Payload is BF16 (`dtype="bf16"`) or MXFP8 (E4M3 values + packed UE8M0 scales). |
| experts | `_peer_forward` (`bf16.py`) / `_packed_forward` (`peer_runtime.py`) | per *sender* lane: DeepGEMM `ready_grouped_*_out` gate/up GEMM consuming the readiness queue, SwiGLU, then `grouped_bf16_scatter` / `grouped_fp8_scatter` |
| combine | the down-proj GEMM **epilogue** (`combine.py::grouped_*_scatter`, `peer_scatter_add_publish`, `peer_reduce_scatter_publish`) | the down projection stores routed rows straight into the **source rank's** FP32 token accumulator (or BF16 slot buffer with `slot_combine`) over NVLink and publishes completion; `combine_wait_copy` / `combine_wait_reduce` on the source waits and copies (or reduces the K slots). |
| backward | `_peer_backward` / `_packed_backward` | same shape: `peer_gather_*_publish` sends `dy` in the forward's packed order, grouped GEMMs recompute, `peer_backward_scatter_publish` returns `dx` and `ds`. |

Properties that matter for composition:

- The combine is fused into the down-proj GEMM epilogue; there is no standalone
  "take `y_local [N, H]` and sum it onto source tokens" op. Pulling it out would mean a new
  kernel (the brief forbids kernel work on this branch).
- The receive layout is expert-packed and K-duplicated: one row per `(token, expert)` assignment
  with per-row `scores`, not the DeepEP layout `_run_local` expects (one row per
  `(token, rank)` with K-wide `recv_topk_idx` / `recv_topk_weights`).
- Hardware: `PeerSchedule.__init__` raises on anything but SM100 (`"PeerSchedule requires
  SM100"`); the scatter epilogues (`peer_scatter.peer_gemm_out`, DeepGEMM
  `PeerScatterStoreBF16`) are SM100-only. `pack_peer_rows` / `return_peer_rows` /
  `peer_epoch_barrier` are plain Triton and arch-agnostic.
- Capacity: fixed `tokens` per rank and `rows = roundup(tokens*topk + experts*127, 128)` per
  source; inputs must match exactly (`"Inputs do not match the fixed peer schedule
  capacity/dtypes"`), so a shorter last batch needs caller-side padding.
- Buffers: owned by the schedule (`symm_mem.empty` + `rendezvous` per tensor, peer views via
  `handle.get_buffer`); `_reset` zeroes the readiness/accumulation parts of this rank's buffers
  before a `dist.barrier` (or the `peer_epoch_barrier` custom op once warm). Forward retains the
  received payload and metadata for backward (`saved`), so the backward does not re-dispatch.
- Precision: BF16 or MXFP8 transport; weights are quantized per call.

### 1.2 `peer_backward.PeerBackward` (the persistent runtime's non-native backward)

This is the closest thing to a separable dispatch/combine and is the pattern the composable mode
follows:

1. every rank copies `x`, `dy`, `ids`, `scores` into **its own** symmetric buffers (plain
   `copy_`), then `peer_epoch_barrier`;
2. every rank reads **all** peers' `ids` (`torch.cat(self.ids)`, a direct read of the peer
   mappings), builds a `DeviceExpertLayout` of the assignments that land on its local experts and
   `pack_peer_rows` **pulls** the routed rows out of the peers' `x` / `dy` buffers (padding rows
   stay zero, no loads);
3. after the grouped GEMMs, `return_peer_rows` **pushes** `dx` / `ds` rows into unique
   `(source rank, token, slot)` slots of the source's buffers, barrier, and the source sums its
   `topk` slots locally.

So the package already has the three primitives a composable path needs: a routed pull-gather
from peer mappings (`mono_ep::pack_peer_rows`, custom op with fake, BF16/FP32, arbitrary width),
a slot push (`peer_packing.return_peer_rows`, plain launcher) and a system-scope barrier
(`mono_ep::peer_epoch_barrier`, or torch's own `_SymmetricMemory.barrier`). What is missing is
the DeepEP-layout bookkeeping around them and the autograd pairing.

### 1.3 How the Axolotl seam consumes a dispatch backend

`experts_fn._ep_forward` (non-fused branch): `backend.dispatch(x_bf16?, topk_idx_i64, topk_w_f32,
num_experts_global=, num_local_experts=) -> (recv_x, recv_topk_idx, recv_topk_weights, handle)`,
then `_run_local(experts, local_fn, recv_x, recv_topk_idx, recv_topk_weights)`, then
`backend.combine(local_out, handle)`.

- `recv_topk_idx` is `[N, K]` int64 with **local** expert ids in `[0, E_local)` and `-1` for slots
  owned elsewhere; `recv_topk_weights` is `[N, K]` fp32 with `0` on `-1` slots. `_run_local`
  rewrites `-1` to `E_local`/weight 0 and appends one zero pad row per local expert.
- The local kernel applies the routing weights itself and sums over K; **combine only sums the
  per-rank partial rows onto the source token** (`torch_dispatch.combine`: uneven all-to-all then
  `index_add_`). Combine never multiplies by routing weights.
- Routing-weight gradients flow through `recv_topk_weights` back to `top_k_weights`; the torch
  backend's dispatch all-to-all is differentiable for both `x` and the weights.
- `_DeepEPBackend` / `_TorchBackend` both send each token **once per destination rank** (K-wide
  ids), not once per expert.
- The torch backend needs one device-to-host copy per layer (`axolotl::ep_to_host`) for the
  all-to-all split sizes; selective checkpointing must save it (and the router `topk`) so the
  recompute cannot desync the collective shapes.

## 2. Gaps

1. **No standalone combine.** The peer path's combine lives in GEMM epilogues; reuse would need a
   kernel. Resolved with a push of `y_local` rows via `return_peer_rows` into a per-source slot
   buffer `[world, cap, H]` plus a local masked sum (BF16 transport, FP32 accumulation).
2. **Layout mismatch.** Expert-packed, K-duplicated rows vs the DeepEP K-wide layout. Resolved
   with a fixed-slot receive layout: row `s*cap + t` of the receive buffer is token `t` of source
   rank `s` (a zero row with all-`-1` ids when nothing of it routes here). No sort, no host
   count, no `.item()`; every rank computes the layout from the published routing alone.
3. **SM100-only pieces.** Only the fused scatter epilogues are SM100-bound. The composable mode
   uses `pack_peer_rows`, `return_peer_rows` and torch symmetric memory, so it runs on SM90 as
   well; it is gated on SM80+ and a single host (symmetric memory needs peer access).
4. **Fixed capacity.** Like the fused runtime the composable mode sizes its buffers once
   (`max_tokens_per_rank`) and accepts any `T <= cap`. The local kernel sees `world * cap` rows
   (sentinel rows are zero-compute in grouped_mm / SonicMoE / ScatterMoE); this is the price of
   a sync-free, `torch.compile`-clean layer. A compact variant would need one `ep_to_host`-style
   sync per layer, which is what the torch backend already does.
5. **Transport precision.** BF16 only. The FP8 publish kernels write the expert-packed layout
   consumed by DeepGEMM; the local kernels want BF16 rows.
6. **`PeerSchedule` retains activations and is SM100-only**; it stays as-is (research path).

## 3. Design

### 3.1 Config surface

```yaml
expert_parallel_backend: mono_ep
expert_parallel_mono_ep_mode: auto        # auto | fused | dispatch
expert_parallel_mono_ep_precision: auto   # fused mode only
```

| mode | dispatch/combine | local experts | when |
|------|------------------|---------------|------|
| `fused` | persistent kernels (`mono_ep::persistent_forward`/`_backward`) | ignored (kernel's own SwiGLU GEMMs); config validation warns about `use_sonicmoe` etc. | forced; falls back per `expert_parallel_fallback_on_unsupported` when the model is unsupported |
| `dispatch` | `mono_ep::peer_dispatch` / `mono_ep::peer_combine` over NVLink symmetric memory | whatever the config names (`use_sonicmoe`, `use_scattermoe`, `experts_implementation`, default `grouped_mm`) through `_run_local` | forced; LoRA, quantized experts, biases, non-SiLU gates, clamped SwiGLU, shared experts all allowed because the local kernel decides |
| `auto` | fused when the config + model pass `unsupported_reason`, else dispatch when the hardware supports it (CUDA, SM80+, single host), else the existing fallback (`torch` with the configured local kernel, or raise) | | default |

`deep_ep` and `torch` are untouched. `expert_parallel_mono_ep_precision`, `expert_parallel_dispatch_chunks`
do not apply in dispatch mode (validation warns when set explicitly).

### 3.2 Package API (`mono_ep.api`)

```python
runtime = MonoEPDispatchRuntime(
    group=ep_group,
    max_tokens_per_rank=cap,
    top_k=K,
    hidden=H,
    compact=False,  # True: only the routed rows, one host count per dispatch
    transfer="streamed",  # "legacy": the fused runtime's per-expert row kernels
)
recv_x, recv_idx, recv_w, handle = runtime.dispatch(
    x_bf16, topk_idx_i64, topk_w_f32, num_local_experts
)
y = runtime.combine(local_out, handle)
runtime.destroy()
```

- `recv_x` `[world*cap, H]` BF16, `recv_idx` `[world*cap, K]` int64 local ids / `-1`, `recv_w`
  `[world*cap, K]` fp32 (0 on `-1`): the DeepEP layout. `handle` is `(rows, topk_idx,
  num_local_experts)` with `rows` the routed mask (fixed layout) or the int64 receive-row
  index of every compact row; the Axolotl backend passes it through untouched.
- Compact (`compact=True`): `recv_*` have `routed` rows (about 86k of 131072 at the EP8
  contract), `dispatch` is the `mono_ep::peer_dispatch_publish` op (publish, barrier,
  routing, device prefix sum of the routed mask), the `mono_ep::peer_count_to_host` op (the
  one device-to-host copy, a stream sync per layer exactly like the torch backend's
  `axolotl::ep_to_host`; the plugin registers both as mandatory checkpoint saves so a
  recompute never re-syncs) and the `mono_ep::peer_dispatch_pull` op (the compact pull,
  whose backward is the fixed layout's `peer_dispatch_backward` with the saved slots).
  The `int()` of the count is a graph break, so compact mode is not fullgraph-compilable
  (the torch backend has the same break); the fixed layout stays sync-free.
- Both calls are `torch.library` custom ops (`mono_ep::peer_dispatch`, `mono_ep::peer_combine`)
  with fake implementations; their backwards (`mono_ep::peer_dispatch_backward`,
  `mono_ep::peer_combine_backward`) are the same two transfers with the payloads swapped:
  backward-of-dispatch returns `grad_recv_x` (and `grad_recv_w`) to the source tokens,
  backward-of-combine gathers `grad_y` into the receive layout. Routing is re-published by each
  op from the `topk_idx` it saved, so no state has to survive between forward and backward.
- Op names are exposed as `runtime.dispatch_op` / `runtime.combine_op` for the selective
  checkpointing policy.

### 3.3 One op, step by step (rank `r`, `world` ranks, capacity `cap`)

Symmetric buffers per rank (allocated once): `x [cap, H]` BF16, `idx [cap, K]` int64,
`w [cap, K]` fp32, `y [world*cap, H]` BF16, `gw [world*cap, K]` fp32.

`peer_dispatch(x, idx, w)`:

1. local writes: `x -> x_buf[:T]`, `idx -> idx_buf` (tail `-1`), `w -> w_buf`;
2. barrier (publish: every peer's `idx`/`w`/`x` is visible from here on);
3. read every peer's `idx_buf`/`w_buf` (small), remap to local ids: `recv_idx[s*cap+t, k]`,
   `recv_w` masked to 0 where not local; `routed[s*cap+t] = any_k(local)`;
4. `peer_stream.pull_rows` streams the routed rows of every source over NVLink
   (section 5.3); unrouted rows are zero.

`peer_combine(local_out, rows, topk_idx, num_local_experts)`:

1. `peer_stream.push_rows` writes row `s*cap+t` to `y_buf_s[r*cap + t]` (unique slot per
   `(producer, source token)`); 2. barrier (every producer's partial has landed in own `y`);
3. `peer_stream.reduce_slots`: `y = sum_p where(routed_to[p, t], y_buf_r[p*cap + t], 0)` in
   fp32, producer order, rounded once to `local_out.dtype` (one pass over `[T, H]`).

Backward of dispatch = `peer_combine` of `grad_recv_x` into `x`'s dtype plus the same push of
`grad_recv_w` through `gw`; backward of combine = the gather half of `peer_dispatch` applied to
`grad_y`. Each op is write → barrier → read.

**Barriers.** The ops fall into two buffer families: the *x family* (dispatch, combine
backward) writes this rank's `x`/`idx`/`w` and reads the peers'; the *y family* (combine,
dispatch backward) writes the peers' `y`/`gw` slots and reads this rank's. A layer always
alternates them (dispatch → combine in forward, combine backward → dispatch backward in
backward, and a checkpoint recompute replays the forward pair before the backward pair), so
the one barrier inside op *n* already orders every peer's reads of op *n-1*'s buffers ahead
of op *n+1*'s writes to them: a peer's read kernels are enqueued before its arrival at the
barrier of op *n*, and this rank's writes in op *n+1* are enqueued after it passed that
barrier. The leading barrier the first version put in front of every op (buffer reuse) is
therefore redundant whenever the families alternate, and `PeerDispatch._enter` inserts it
only when two ops of the same family run back to back (e.g. two combine backwards with no
dispatch backward between them because neither `x` nor the routing weights need a gradient;
`extra_barriers` counts them). The one barrier per op that remains is the publish/landing
barrier: without it a peer could pull `x` rows that are still being copied in, or the
reduction could read slots a slow producer has not written yet. The compact
`peer_dispatch_pull` op has no barrier of its own: it only reads, and the next op's barrier
(always y family) orders those reads before any re-publish. Barriers are
`_SymmetricMemory.barrier` (system-scope release/acquire on the allocation's signal pad,
10-minute timeout so a skewed rank behaves like a slow NCCL collective, not a device assert);
they cost 13 us on 2 x B200 and 26 us on 2 x H100 (section 5.3).

### 3.4 Axolotl seam

- `experts_fn._MonoEPDispatchBackend` (`fused=False`, `cast_bf16=True`): `dispatch` returns the
  runtime's three tensors plus a handle `(recv_idx, topk_idx, num_local_experts)`; `combine`
  calls the runtime. Padding sentinel, token capacity and the grad-scale hook are untouched
  because they live in `_ep_forward` and the plugin, before/around the backend.
- `experts_fn.set_mono_ep_mode("fused"|"dispatch")` picks the class behind `_BACKENDS["mono_ep"]`.
- Plugin: `post_model_build` resolves `auto`, `post_model_load` builds either runtime
  (`MonoEPRuntime` or `MonoEPDispatchRuntime`), `_register_checkpoint_saves` adds the dispatch
  op outputs as preferred saves under `expert_parallel_save_dispatch` (mirror of the torch
  all-to-all saves); the router `topk` save is already mandatory.
- `get_distributed_env` exports `NCCL_MAX_CTAS` only while the fused mode is still possible for
  the config (`mono_ep` backend, mode not forced to `dispatch`, no adapter, no quantized
  loading); the model-level fused checks run later, so an `auto` config that ends up in dispatch
  mode because of its model shape still carries the harmless reservation.

### 3.5 Compile and sync discipline

No `.item()`/`.tolist()`: shapes are fixed by the runtime, routing math is tensor-only, and
`peer_epoch_barrier`-style syncs happen inside the custom ops. The CPU tests compile `_ep_forward`
with `fullgraph=True` against a fake runtime to keep the seam itself break-free.

### 3.6 Cost of the fixed layout

The receive side is `world * capacity` rows per layer whatever the routing, with
`capacity = sequence_len * max(micro_batch_size, eval_batch_size) / context_parallel_size`
(the fused runtime's rule):

| EP | tokens/rank | receive rows | `recv_x` BF16 at H=2048 | `recv_x` BF16 at H=7168 |
|----|-------------|--------------|-------------------------|-------------------------|
| 2  | 4096        | 8192         | 32 MB                   | 112 MB                  |
| 8  | 16384       | 131072       | 512 MB                  | 1.8 GB                  |

The local kernel sees every row (sentinel rows are zero-compute in grouped_mm, SonicMoE and
ScatterMoE, but row-count-sized work such as SonicMoE's pow2 padding and the index/weight
tensors scales with it), and `expert_parallel_save_dispatch: true` holds that `recv_x` per
layer across the backward. The symmetric buffers themselves are `x [cap, H]`, `y [world*cap,
H]` and the small `idx`/`w`/`gw` tensors, allocated once per process. Compacting the rows needs
the host count the design avoids (follow-up below).

`collect()` (combine and dispatch-backward receive) is `peer_stream.reduce_slots`: one pass
over the own `[world, T, H]` return buffer that sums the producer slots of each token in FP32,
in producer order, and rounds once (bitwise what the earlier `world` masked `+=` passes
computed, 17-19x faster at EP8, section 5.3). The `torch.stack` of the `world` routing views
in `collect_routing()` and the `producer_mask` are small (12 MB of reads, 0.06-0.13 ms).

## 4. Follow-ups (not in this change)

Done since the first version: bandwidth-shaped pull/push (`peer_stream.py`), the fused
`collect()`, one barrier per op, compact receive rows behind one host count (sections 3.2,
3.3, 5.3, 5.4). Still open:

- Overlap: issue the gather of layer `l+1` under layer `l`'s local kernel (needs a second
  buffer set; the family rule of 3.3 would then hold per buffer set).
- FP8 transport for `recv_x` when the local kernel accepts it (halves the bytes the pull and
  the combine backward move; the combine push/reduce stay BF16).
- Fuse `producer_mask` and the routing remap of `collect_routing()` into the reduce / pull
  launches (0.1-0.3 ms per op at EP8), and the compact mode's `combine_backward` rebuilds
  `routed`/`prefix` from the saved slots (three small launches) rather than saving them.
- Multi-host EP groups: symmetric memory is node-local; a NCCL fallback inside the dispatch
  mode is still out of scope.
- `pull_rows` / `push_rows` compile one Triton specialization per `world`, `capacity` and
  row width; a growable capacity needs a re-rendezvous of the workspace.
- Tile size (16-64 KB) and programs per SM (2-8) are flat within 1% on both GPUs at this
  contract (section 5.3), so the defaults are not per-arch; revisit for wider rows (H 7168)
  or PCIe-only hosts.
- The fixed layout hands a non-padding local kernel 1.5x the rows (grouped_mm 1.24-1.4x
  behind torch all-to-all in 5.4); the compact rows fix it at the price of one host count per
  layer. A fullgraph-compilable compact mode needs the count to stay on device (a
  capacity-bounded `[world * capacity, H]` view with a device row count the kernels honor).
- Gap to the fused kernels at EP8 (section 5.4): the streamed SonicMoE line is 1.6x behind
  the current fused reference on both GPUs with the transfers now at the link rate; what
  remains is local-kernel time and the serialization of transfer and compute that the
  fused runtime overlaps.

## 5. Validation (2026-11-09)

- CPU: `packages/mono-ep/tests/test_peer_dispatch.py` (3-rank threaded simulation, autograd
  parity, receive layout, fullgraph compile, lifecycle) and
  `tests/integrations/expert_parallel/test_mono_ep_backend.py` / `test_validation.py` (mode
  resolution, fallback matrix, fake-runtime seam forward incl. fullgraph compile, SAC
  registration, validation warnings).
- GPU (`examples/modal_dispatch_parity.py`, Modal app `ap-F77wFpKRWjBAfX9QYF0e9Z`, 2 x H100 strict,
  EP2 x FSDP2, 2 AdamW steps, balanced + fully skewed routing, full fine-tune + PEFT expert
  LoRA): gradients and post-step weights of the dispatch mode match the torch backend with the
  same local kernel bitwise for **grouped_mm, ScatterMoE and SonicMoE** (12 cases,
  `benchmarks/dispatch_mode_parity_h100.json`). One MoE block, H 2048 / I 1024 / 64 experts /
  top-8 / 4096 tokens per rank, bf16, forward+backward, max over ranks:

  | local kernel | torch all-to-all | mono_ep dispatch |
  |--------------|------------------|------------------|
  | grouped_mm   | 13.09 ms         | 11.99 ms         |
  | ScatterMoE   | 9.91 ms          | 9.08 ms          |
  | SonicMoE     | 6.70 ms          | 6.41 ms          |

  Caveat: on this H100 image most quack autotune configs for SonicMoE's GEMMs failed to
  compile (the py3.14 forkserver / async-compile issue already known from the kernel lanes),
  so SonicMoE ran on the surviving configs; both backends share the kernel, so the comparison
  holds but the absolute SonicMoE numbers are pessimistic.

### 5.1 Fused-benchmark contract: one Qwen3-30B-A3B layer at EP8

`examples/dispatch_sonicmoe_contract.py` (driver `modal_dispatch_sonicmoe_contract.py`) runs the
fused line's contract: 16384 tokens per rank, H 2048, I 768, 128 experts top-8 (16 local),
balanced and skewed routing from the same generators, forward / backward / checkpointed 2F+B,
5 warmups + 15 samples, medians of the max rank, all variants interleaved in one allocation.
Expert weights are **BF16** (SonicMoE's native dtype); the fused numbers quoted for comparison
come from different allocations with FP32 masters quantized per call (FP8 compute on H100,
BF16 on B200). SonicMoE runs the way production does: the kernels plugin's device setup
(`_check_sonicmoe_gpu_compat`, which on cutlass-dsl 4.8 shims the pre-4.8 `nvvm.mbarrier_arrive`
call form the hub-vendored quack 0.6.1 still uses on its SM100 2-CTA path), the measured
no-sweep profiles seeded with `apply_autotune_profiles` (`examples/sonicmoe_contract_profiles.py`:
B200 entries from main `b0f55d39e3`, #4273; H100 entries from `7d4225c71f` on
`features/sonicmoe-quack-h100`), then `install_quack_key_policy` (pow2 buckets, in-process
compile). Per rank the 12 grouped quack keys got 1-4 candidates each (gated forward 1, not
benchmarked; `gemm_tuned` 3-4; `gemm_dgated_tuned` 3 on B200, 1 on H100), all compiled, 0 failed,
every winner `source: profile`; the first-step warm took 72 s (B200) / 82 s (H100) instead of
200 s with full sweeps. Each artifact records every key's call shapes, candidates offered /
compiled / failed, the winner and its timing table. (`token_gather_sum_kernel` is the bundled
Triton kernel with its own 60-config autotuner, untouched by the quack profiles.)

8 x H100 80GB (Modal `ap-DPDFL3sbYfPqkGAh5dNWqe`, 0.46 GPU-h), median max-rank ms:

| variant | F bal | B bal | 2F+B bal | F skew | B skew | 2F+B skew | peak GiB |
|---|---|---|---|---|---|---|---|
| torch all-to-all + SonicMoE (control) | 7.31 | 8.37 | 22.68 | 7.87 | 9.77 | 24.75 | 11.9 |
| mono_ep dispatch + SonicMoE | 13.77 | 16.24 | 43.54 | 13.77 | 16.53 | 43.34 | 21.7 |
| mono_ep dispatch + grouped_mm | 56.83 | 91.90 | 204.99 | 56.77 | 91.96 | 204.80 | 43.6 |
| torch all-to-all + grouped_mm | 35.80 | 58.97 | 129.52 | 41.21 | 68.22 | 149.58 | 34.1 |
| fused mono_ep r3 (FP8, other allocation) | 3.26 | 7.53 | 13.48 | | | 16.60 | |

8 x B200 (Modal `ap-R8iuf9H6IVXAWgpMWmfv9P`, 0.40 GPU-h), median max-rank ms:

| variant | F bal | B bal | 2F+B bal | F skew | B skew | 2F+B skew | peak GiB |
|---|---|---|---|---|---|---|---|
| torch all-to-all + SonicMoE (control) | 4.75 | 4.94 | 14.01 | 4.67 | 5.69 | 14.88 | 11.9 |
| mono_ep dispatch + SonicMoE | 8.68 | 9.81 | 26.68 | 8.83 | 10.20 | 27.22 | 21.7 |
| mono_ep dispatch + grouped_mm | 38.72 | 63.27 | 140.24 | 38.98 | 63.73 | 141.02 | 43.6 |
| torch all-to-all + grouped_mm | 24.50 | 41.11 | 89.64 | 28.11 | 47.40 | 103.23 | 34.1 |
| fused mono_ep r3 (BF16, other allocation) | 2.11 | 4.14 | 7.66 | | | | |

An earlier H100 run with full quack sweeps (`ap-gonVugQZPGReBJVajH1kkx`, 0.65 GPU-h, 124
`gemm_tuned` candidates over 8 keys) gave the same dispatch numbers (13.82 / 16.20 / 43.54) and
a control within 1-8% (6.76 / 8.23 / 21.24), so the bounded profiles cost the control a little
on this shape and the dispatch mode nothing. Mean-rank medians sit within 0.03 ms of the
max-rank ones on both GPUs: the fixed layout makes every rank do the same work.

Parity (first measured iteration, max over ranks, identical on both GPUs): `dgate_up`, `ddown`
and `dscore` bitwise for both pairs; `y` and `dx` differ by rel-L2 3.4e-3 (the torch backend
sums up to eight partial rows per token with a BF16 `index_add_`, the peer combine sums them in
FP32 and rounds once). Receive side per rank: 131072 rows, `recv_x` 512 MB BF16, symmetric
buffers 64 MB `x` + 512 MB `y` + 6 MB routing.

Reading: at EP8 the dispatch mode is 1.8-2x slower than torch all-to-all with the same local
kernel on both GPUs and 3.2x (H100) / 3.5x (B200) behind the fused kernels on 2F+B. The transfer kernels it reuses (`pack_peer_rows` pull,
`return_peer_rows` push) were sized for the fused runtime's per-expert row pulls: 1024 programs
walk 131072 rows x 8 column tiles with one remote load per tile, so each direction is NVLink
latency-bound (~6-7 ms for 512 MB that NCCL moves in ~1-2 ms), and the fixed layout makes
grouped_mm do 1.5x the rows (131072 vs ~86k routed) while SonicMoE pads to the same power of
two either way. Memory doubles for the same reasons. The EP2 numbers in 5 are the regime where
the layout is small and the latency is hidden. Before any EP8 use the pull/push need a
bandwidth-shaped kernel (wide per-program row ranges, vectorized 16 B loads, several rows in
flight) and the compact-rows follow-up; both are in section 4.

### 5.3 Transfer microbench at the EP8 pattern on 2 GPUs (2026-11-10)

`examples/peer_stream_microbench.py` (driver `modal_peer_stream_microbench.py`; 2 x H100 80GB
Modal `ap-9le0bhpevY0xzueZCCCamD`, 2 x B200 `ap-wGHazjO0gFtAOTDSgWKaLP`, 20 s and
0.011 GPU-h each) lays the two ranks' buffers out as if there were eight source ranks: the
peer's `x` buffer is eight `[16384, 2048]` BF16 source slices and the pull gathers the
`[131072, 2048]` receive layout from them (512 MB; the 5.2 copies move the same bytes), the
push returns the rows into eight `[16384, 2048]` destination slices of the peer. Routing is
either every row or the contract's balanced top-8 over 128 experts (66% of the rows routed,
~341 MB). Same timing as 5.2 (CUDA events, `dist.barrier` + `synchronize` around each of 15
samples after 3 warmups, median, max over active ranks). The streamed kernels were swept over
tile sizes 16/32/64 KB and 2/4/8 programs per SM: every pull/push cell moves within 1% on
both GPUs, so the defaults (32 KB tiles of 8 rows x 512 int64 lanes, 4 programs per SM,
`world * ceil(SMs * 4 / world)` programs each streaming one peer) are kept. Every streamed
and compact output is bitwise equal to the legacy kernels' (and the fused reduction to the
masked 8-pass sum) on both GPUs.

| transfer (both ranks active, 2 GPUs) | 2 x H100 ms / GB/s / of NVLink | 2 x B200 ms / GB/s / of NVLink |
|---|---|---|
| pull, all 131072 rows (512 MB): legacy `pack_peer_rows` | 1.74 / 308 / 0.68 | 2.37 / 226 / 0.25 |
| pull, 512 MB: streamed `pull_rows` | 1.67 / 322 / 0.71 | 0.85 / 631 / 0.70 |
| pull, 512 MB: streamed, one rank active | 1.51 / 355 / 0.79 | 0.74 / 727 / 0.81 |
| pull, 512 MB: plain Triton row copy (5.2) | 1.64 / 327 / 0.73 | 0.84 / 641 / 0.71 |
| pull, 512 MB: `Tensor.copy_` | 1.46 / 368 / 0.82 | 0.74 / 724 / 0.80 |
| pull, 512 MB: NCCL `all_to_all_single` | 1.59 / 337 / 0.75 | 1.02 / 528 / 0.59 |
| pull, contract routing (~87k rows, ~341 MB): legacy | 1.30 / 275 / 0.61 | 1.75 / 204 / 0.23 |
| pull, contract: streamed | 1.14 / 313 / 0.70 | 0.59 / 605 / 0.67 |
| pull, contract: streamed compact (`[routed, H]` + slots) | 1.17 / 308 / 0.68 | 0.60 / 598 / 0.66 |
| pull, contract: plain Triton row copy | 1.11 / 322 / 0.71 | 0.57 / 627 / 0.70 |
| pull, contract: NCCL `all_to_all_single` | 1.09 / 328 / 0.73 | 0.69 / 513 / 0.57 |
| pull, contract: streamed, 4 of 8 sources local (crossing half counted) | 0.67 / 266 / 0.59 | 0.32 / 550 / 0.61 |
| push, 512 MB: legacy `return_peer_rows` | 1.82 / 295 / 0.66 | 0.97 / 555 / 0.62 |
| push, 512 MB: streamed `push_rows` | 1.54 / 348 / 0.77 | 0.81 / 665 / 0.74 |
| push, contract: legacy | 1.22 / 293 / 0.65 | 0.73 / 492 / 0.55 |
| push, contract: streamed | 1.06 / 337 / 0.75 | 0.56 / 643 / 0.71 |
| push, contract: streamed compact (slots) | 1.12 / 320 / 0.71 | 0.58 / 612 / 0.68 |

| local work (per rank) | 2 x H100 ms | 2 x B200 ms |
|---|---|---|
| combine reduction, legacy 8-pass masked `+=` (512 MB read, BF16 out) | 3.76 | 2.39 |
| combine reduction, fused `reduce_slots` (BF16 out) | 0.236 | 0.124 |
| fused `reduce_slots` (FP32 out, the `gw` path's shape class) | 0.251 | 0.153 |
| `producer_mask` (torch) | 0.128 | 0.061 |
| symmetric-memory barrier | 26 us | 13 us |
| NCCL `dist.barrier` | 112 us | 43 us |

Reading. On B200 the legacy pull was the problem the 5.2 copy predicted: 25% of the link
against 70-71% for the streamed pull and the plain row copy (80% for `Tensor.copy_`), so the
dispatch's pull drops 2.8x at this pattern and the push 1.2-1.3x. On H100 the legacy pull was
already at 68% and the streamed one gains 4%: the EP8 loss on H100 is not per-row latency but
the combine reduction (3.7 ms per combine, 17x more than the fused pass; at EP8 the dispatch
mode runs it three times per checkpointed layer, 11 ms of the 43.5) plus the masked
`torch.where` / `masked_fill` passes around it. NCCL `all_to_all_single` of the same bytes
sits at 73-75% (H100) / 57-59% (B200) of the link, so the streamed pull is at NCCL's rate on
H100 and 1.2x faster on B200; the compact variants pay 2-5% for the prefix-sum stores and the
slot indirection. The streamed kernels address every peer buffer as `first buffer + delta`
with a `multiple_of(16)` hint; casting a loaded address to a pointer (the pointer-array
idiom) loses the alignment to Triton and halves every load to 8-byte scalars (an earlier
H100 run with that form: pull 1.67 ms, i.e. the same on H100, where the link latency rather
than the instruction count bounds it).

### 5.4 EP8 contract rerun with the streamed transfers (2026-11-10)

Same script, same contract, same profiles and timing as 5.1, nine variants interleaved in one
allocation: the 5.1 four (the legacy transfers are kept as `transfer="legacy"`), the streamed
transfers (`peer_stream.py`, default) and the compact rows (`compact=True`), each with
SonicMoE and grouped_mm, plus the fused mono_ep runtime (`mode=fused`) re-measured in the same
allocation instead of quoted from another one. The three dispatch runtimes and the fused
runtime are resident for the whole run, so the peak memory column of every variant includes
all their symmetric workspaces (as 5.1's included one). Extra same-family barriers inserted by `_enter`: 0 for every runtime on both GPUs.

8 x H100 80GB (Modal `ap-VUsL20asL4xebM9ByaCUSX`, 0.56 GPU-h, 253 s), median max-rank ms:

| variant | F bal | B bal | 2F+B bal | F skew | B skew | 2F+B skew | peak GiB |
|---|---|---|---|---|---|---|---|
| torch all-to-all + SonicMoE (control) | 7.36 | 8.38 | 22.55 | 7.63 | 9.71 | 24.54 | 13.0 |
| dispatch, legacy transfers + SonicMoE (5.1 line) | 12.88 | 15.05 | 40.37 | 12.33 | 15.00 | 38.91 | 22.8 |
| dispatch, streamed + SonicMoE | 6.53 | 8.71 | 21.07 | 7.16 | 9.97 | 23.73 | 22.8 |
| dispatch, streamed compact + SonicMoE | 6.43 | 8.07 | 20.68 | 6.79 | 9.43 | 22.90 | 13.0 |
| torch all-to-all + grouped_mm | 35.90 | 59.04 | 129.78 | 41.19 | 68.00 | 149.48 | 35.1 |
| dispatch, legacy transfers + grouped_mm | 55.76 | 90.75 | 201.54 | 55.30 | 90.32 | 200.05 | 44.7 |
| dispatch, streamed + grouped_mm | 49.49 | 83.98 | 182.32 | 50.38 | 85.22 | 185.26 | 44.7 |
| dispatch, streamed compact + grouped_mm | 35.11 | 58.81 | 128.35 | 40.54 | 67.81 | 148.30 | 35.1 |
| fused mono_ep, this branch's base build (mode=fused, same allocation) (b) | 4.77 | 9.78 | 18.57 | 5.73 | 12.06 | 22.52 | 19.8 (a) |

8 x B200 (Modal `ap-kqdcZKzZU8tgEHqRPGUIHb`, 0.49 GPU-h, 220 s), median max-rank ms:

| variant | F bal | B bal | 2F+B bal | F skew | B skew | 2F+B skew | peak GiB |
|---|---|---|---|---|---|---|---|
| torch all-to-all + SonicMoE (control) | 4.79 | 4.93 | 14.16 | 4.95 | 5.66 | 15.20 | 13.0 |
| dispatch, legacy transfers + SonicMoE (5.1 line) | 8.42 | 9.42 | 25.65 | 8.27 | 9.63 | 25.60 | 22.8 |
| dispatch, streamed + SonicMoE | 4.47 | 4.83 | 12.57 | 4.55 | 5.47 | 13.82 | 22.8 |
| dispatch, streamed compact + SonicMoE | 4.19 | 4.30 | 12.32 | 4.25 | 5.05 | 13.36 | 13.0 |
| torch all-to-all + grouped_mm | 24.79 | 41.64 | 90.68 | 28.13 | 47.37 | 103.06 | 35.2 |
| dispatch, legacy transfers + grouped_mm | 38.76 | 63.44 | 140.51 | 38.63 | 63.46 | 140.22 | 44.7 |
| dispatch, streamed + grouped_mm | 34.33 | 58.99 | 127.06 | 34.50 | 59.15 | 127.61 | 44.7 |
| dispatch, streamed compact + grouped_mm | 24.25 | 41.01 | 89.05 | 27.64 | 46.78 | 101.59 | 35.2 |
| fused mono_ep, this branch's base build (mode=fused, same allocation) (b) | 2.40 | 6.57 | 10.98 | 2.85 | 8.21 | 13.53 | 16.7 (a) |

(a) The peak GiB column comes from the same code run without the fused runtime resident
(H100 `ap-xJHCsDBnHQDkzZTIlyX4kY`, B200 `ap-iSuWi5SykSIvjdiNuUVvGO`, 0.98 GPU-h together, dispatch
timings within 0.1 ms of the tables above); in the runs with the fused runtime resident every
variant's peak rises by its workspace (torch + SonicMoE reads 26.5 GiB on B200), and the fused row's own
figure is the one with the three dispatch workspaces resident. Mean-rank medians sit within
0.07 ms of the max-rank ones in every dispatch and torch cell (the streamed kernels keep the
fixed layout's property that every rank does the same work; the compact rows differ per rank
by the routing only); the fused row's max rank runs up to 0.42 ms ahead of the mean on skewed routing. Parity, both GPUs, both routings:
`dgate_up`, `ddown`, `dscore` bitwise against the torch backend with the same local kernel
for all six dispatch variants; `y` / `dx` rel-L2 at most 3.39e-03 (H100) /
3.39e-03 (B200), the FP32-combine-vs-BF16-`index_add_` difference of 5.1; the streamed
and compact SonicMoE variants are **bitwise equal to the legacy transfers** on `y`, `dx`,
`dscore`, `dgate_up` and `ddown`, as the 2-GPU test (`tests/test_peer_dispatch_gpu.py`)
requires. Quack candidates failed: 0 on both GPUs.

(b) Provenance of the fused row: it is the production `MonoEPRuntime` of this branch's base,
`features/mono-ep` @ 9ae994be97, whose SM90/SM100 persistent kernels predate B200 rounds 1-3
(no two-row dispatch pulls, no forward lookahead, no weight-layout cache, none of the
`MONO_EP_SM100_*` selectors), run eager on the BF16 parameters (the reference runs keep FP32
masters) with 16 reserved SMs. It is a same-allocation sanity point for the dispatch numbers,
not the fused-vs-dispatch gap: the current fused reference for this contract is
the B200 round-3 build (f73dad341d) at **7.66 ms** checkpointed 2F+B on 8 x B200 and
**13.48 ms** (FP8) on 8 x H100, which is being ported into the integrated branch. The JSON
artifacts carry the same note under `fused_runtime.provenance`. On H100 the fused row ran at precision `fp8` with the `native` backward; against torch all-to-all + SonicMoE it is rel-L2 6.6e-02 on `y`, 7.3e-02 on `dx`, 6.8e-02 on the weight gradients. On B200 the fused row ran at precision `bf16` with the `native` backward; against torch all-to-all + SonicMoE it is rel-L2 5.9e-03 on `y`, 5.5e-03 on `dx`, 4.0e-03 on the weight gradients.

Reading, 2F+B balanced / skewed against torch all-to-all with the same local kernel:

| | H100 streamed | H100 compact | B200 streamed | B200 compact |
|---|---|---|---|---|
| SonicMoE, vs torch | 0.93x / 0.97x | 0.92x / 0.93x | 0.89x / 0.91x | 0.87x / 0.88x |
| grouped_mm, vs torch | 1.40x / 1.24x | 0.99x / 0.99x | 1.40x / 1.24x | 0.98x / 0.99x |
| SonicMoE, vs the base build's fused row (b) | 1.13x / 1.05x | 1.11x / 1.02x | 1.14x / 1.02x | 1.12x / 0.99x |
| SonicMoE, vs legacy transfers | 0.52x | 0.51x | 0.49x | 0.48x |

Reading. The target of this change holds on both GPUs: with the same local kernel the
streamed dispatch mode is now 3-11% faster than torch all-to-all on SonicMoE (it was 1.8-2x
slower in 5.1), the compact rows add another 1-3% and bring the memory back to the torch
backend's, and both are bitwise equal to the legacy path. With grouped_mm the fixed layout
still costs 1.24-1.4x against torch (the kernel runs over 131072 rows instead of ~86k routed
ones), which the compact rows remove (0.98-0.99x); so `expert_parallel_mono_ep_compact: true`
is the setting for a local kernel that does not pad to the receive count, and the fixed
layout is for SonicMoE and for fullgraph compilation. Against the current fused reference
(7.66 ms B200 / 13.48 ms H100) the streamed SonicMoE line is 1.6x behind on both GPUs: that
gap is the local kernel's forward and the two backward GEMMs plus the three serialized
transfers, no longer the transfer kernels, and section 4 lists what is left on the transfer
side (overlap of the next layer's gather, FP8 transport, fusing the routing remaps).

### 5.5 DeepEP baseline in the EP8 contract (2026-11-10)

The question behind this rerun: do the streamed and compact dispatch modes with SonicMoE beat
DeepEP with SonicMoE? DeepEP had never been timed in this program, so the contract script grew
`--deep-ep` rows that run axolotl's own `_DeepEPBackend` (`experts_fn.py`, normal-mode intranode
kernels, K-wide rows: one row per (token, destination rank) with the top-k ids and weights on
the row, `get_dispatch_layout` per forward) around the same local kernels, in the same
allocation, scopes, routings, warmups and samples as every other row. The Modal driver adds the
DeepEP layer only under `MONO_EP_CONTRACT_DEEP_EP=1`, so the 5.4 images and artifacts are
untouched; the new artifacts are `benchmarks/dispatch_sonicmoe_contract_deep_ep_{h100,b200}.json`.

**Build.** DeepEP `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee` (tag v1.2.1, the commit
`docker/Dockerfile-cloud-blackwell-uv` and `cicd/deep_ep.py` pin), package version
`1.2.1+9af0e0d`. NVSHMEM comes from torch's `nvidia-nvshmem-cu13` wheel, **3.4.5**, which
DeepEP's `setup.py` finds on its own (`cpp_links_nvshmem: libnvshmem_host.so.3`, the internode
and low-latency kernels are compiled in but unused here). Flags: `TORCH_CUDA_ARCH_LIST=9.0`
(Hopper layer) / `10.0` (the Blackwell cloud image's own build), `MAX_JOBS=16`,
`CFLAGS/CXXFLAGS/NVCC_PREPEND_FLAGS=-I/usr/local/cuda/include/cccl` (CUDA 13.0 moved
`cuda/std`), and setup.py's own `-O3 -rdc=true --ptxas-options=--register-usage-level=10
-DDISABLE_AGGRESSIVE_PTX_INSTRS -std=c++20`; wheel via `python setup.py bdist_wheel` (the
tag's build backend fails under `pip install .`). The Hopper layer built in 45 s on the Modal
builder (image `im-WuNsHfm3JwbMVvPIOFROFH`); the Blackwell image
(`axolotlai/axolotl-cloud-uv:main-py3.14-cu130-2.14.1-blackwell`) already ships the same
commit built for sm_100, so its layer is an import check (`im-LX8azkx0s5OX2y2WJMtPmL`). No
Blackwell-specific revision was needed: v1.2.1 builds and runs on sm_100 (the parity below is
from that build); the one revision constraint is the other way round, DeepEP main needs CUDA
13.1 and these images are CUDA 13.0.

**Backend settings (axolotl defaults, reported from the job).** `deep_ep.Buffer(group=WORLD,
num_nvl_bytes=256 MiB, num_rdma_bytes=0, low_latency_mode=False)` from `configure_buffer` with
the `ExpertParallelArgs` defaults; `Buffer.get_dispatch_config(8) = Config(Buffer.num_sms, 6,
256, 6, 128)` and `get_combine_config(8) = Config(Buffer.num_sms, 4, 256, 6, 128)`;
`Buffer.num_sms` = **20**, DeepEP's class default, since the plugin never calls
`set_num_sms`; `dispatch` / `combine` with `async_finish=False`,
`allocate_on_comm_stream=False`, `expert_alignment=1`, no `num_worst_tokens`, no
`previous_event`. The buffer-size hint of the 8-rank dispatch preset at hidden 2048 is 126 MB
for 20 SMs and grows 6.3 MB per SM pair (252 MB at 40 SMs, 403 MB at 64), so the plugin's 256
MiB buffer admits at most 40 SMs. Two knobs were run both ways:

- `barrier_ep` (a `torch.cuda.synchronize` + NCCL barrier before every combine and
  backward-of-dispatch, the plugin's autotune-skew guard): `deep_ep` rows run with
  `AXOLOTL_EP_NO_BARRIER=1`, the `deep_ep_barrier` row is the backend as shipped. It costs
  2% on H100 (below the 5% bar), so the kernel-only row is the comparison line.
- `num_sms`: a 3+10-iteration fwd+bwd sweep on balanced routing (recorded under
  `deep_ep_num_sms_sweep`), and because the best value moved the default by more than 5% the
  job added a full `deep_ep_sms<N>+sonicmoe` row. On H100 the sweep stopped at 40 (the buffer
  bound above); the B200 job was run with `--deep-ep-num-sms-max 64`, which sizes the buffer
  to the largest hint instead (the kernels address the buffer per channel, so the default
  rows do not see the larger allocation; both sizes are in the artifact).

8 x H100 80GB (Modal `ap-ML4Jf7M3m03WlwFVGclzL6`, 0.50 GPU-h, 225 s), median max-rank ms,
balanced / skewed routing:

| variant | F bal | B bal | 2F+B bal | F skew | B skew | 2F+B skew |
|---|---|---|---|---|---|---|
| torch all-to-all + SonicMoE (control) | 6.71 | 8.23 | 20.88 | 7.46 | 9.62 | 23.76 |
| **DeepEP + SonicMoE, 20 SMs (kernels only)** | 7.09 | 9.28 | 21.09 | 8.37 | 11.22 | 25.30 |
| DeepEP + SonicMoE, 20 SMs, as shipped (`barrier_ep`) | 7.27 | 9.41 | 21.48 | 8.54 | 11.36 | 25.68 |
| **DeepEP + SonicMoE, 40 SMs (kernels only)** | 5.61 | 7.72 | 17.56 | 6.50 | 9.24 | 20.73 |
| dispatch, legacy transfers + SonicMoE | 12.74 | 14.93 | 40.28 | 12.34 | 14.76 | 38.76 |
| dispatch, streamed + SonicMoE | 6.37 | 8.61 | 20.77 | 7.22 | 9.97 | 23.88 |
| dispatch, streamed compact + SonicMoE | 5.89 | 7.91 | 19.30 | 6.76 | 9.41 | 22.53 |
| torch all-to-all + grouped_mm | 35.79 | 58.96 | 129.68 | 41.25 | 68.19 | 149.80 |
| DeepEP + grouped_mm, 20 SMs (kernels only) | 36.28 | 60.42 | 123.97 | 42.19 | 70.06 | 144.25 |
| dispatch, legacy transfers + grouped_mm | 55.70 | 90.37 | 201.43 | 55.31 | 90.15 | 200.29 |
| dispatch, streamed + grouped_mm | 49.53 | 84.02 | 182.42 | 50.48 | 85.42 | 185.72 |
| dispatch, streamed compact + grouped_mm | 35.09 | 58.76 | 128.37 | 40.60 | 67.96 | 148.61 |
| fused mono_ep, base build (5.4 (b), fp8 on H100) | 4.73 | 9.74 | 18.55 | 5.70 | 11.97 | 22.64 |

H100 `num_sms` sweep (fwd+bwd, balanced, median max-rank ms): 20 SMs 16.15, 24 SMs 15.06,
32 SMs 13.63, 40 SMs 13.15 (0.81x of the default).

8 x B200 (Modal `ap-1frtFOq9kwibpsjjWsMTtR`, 0.54 GPU-h, 242 s; DeepEP buffer sized to the
64-SM hint, 403 MB, for the extended sweep), median max-rank ms, balanced / skewed routing:

| variant | F bal | B bal | 2F+B bal | F skew | B skew | 2F+B skew |
|---|---|---|---|---|---|---|
| torch all-to-all + SonicMoE (control) | 4.89 | 4.90 | 14.35 | 5.10 | 5.64 | 15.33 |
| **DeepEP + SonicMoE, 20 SMs (kernels only)** | 8.22 | 9.06 | 21.75 | 9.32 | 10.46 | 24.82 |
| DeepEP + SonicMoE, 20 SMs, as shipped (`barrier_ep`) | 8.44 | 9.25 | 22.13 | 9.52 | 10.66 | 25.24 |
| **DeepEP + SonicMoE, 64 SMs (kernels only)** | 4.44 | 5.10 | 12.80 | 4.94 | 5.94 | 14.00 |
| dispatch, legacy transfers + SonicMoE | 8.38 | 9.48 | 25.58 | 8.30 | 9.56 | 25.58 |
| dispatch, streamed + SonicMoE | 4.43 | 4.82 | 12.60 | 4.53 | 5.44 | 13.61 |
| dispatch, streamed compact + SonicMoE | 4.19 | 4.32 | 12.42 | 4.48 | 5.03 | 13.31 |
| torch all-to-all + grouped_mm | 24.56 | 41.18 | 89.74 | 28.28 | 47.49 | 103.55 |
| DeepEP + grouped_mm, 20 SMs (kernels only) | 28.69 | 45.20 | 95.44 | 33.11 | 52.31 | 110.31 |
| dispatch, legacy transfers + grouped_mm | 38.46 | 62.81 | 139.09 | 38.52 | 63.11 | 139.44 |
| dispatch, streamed + grouped_mm | 33.94 | 58.25 | 125.56 | 34.50 | 59.08 | 127.53 |
| dispatch, streamed compact + grouped_mm | 24.02 | 40.47 | 87.96 | 27.71 | 46.89 | 101.83 |
| fused mono_ep, base build (5.4 (b), bf16 on B200) | 2.41 | 6.50 | 10.89 | 2.88 | 8.23 | 13.50 |

B200 `num_sms` sweep (fwd+bwd, balanced, median max-rank ms): 20 SMs 17.12, 24 SMs 15.21,
32 SMs 12.75, 40 SMs 11.37, 48 SMs 10.34, 64 SMs 9.38 (0.55x of the default, and still
falling at the last point; the B200 has 148 SMs). The 5.4 rows reproduce within 0.3 ms on
both GPUs (streamed 12.60 vs 12.57, compact 12.42 vs 12.32, torch 14.35 vs 14.16 on B200;
20.77 / 19.30 / 20.88 vs 21.07 / 20.68 / 22.55 on H100), so the DeepEP rows can be read
against either table.

**Rows and bytes per direction.** From DeepEP's dispatch layout on the actual routing (the
`transfer_rows` field), per rank:

| routing | K-wide rows received (total / remote) | remote rows min / max over ranks | fixed-layout rows (streamed compute) | K-duplicated slots (fused only) | NVLink payload per direction per rank |
|---|---|---|---|---|---|
| balanced | 87453 / 76530 | 76401 / 76882 | 131072 | 131072 | 298.9 MiB |
| skewed | 86527 / 75689 | 63087 / 88548 | 131072 | 131072 | 295.7 MiB |

The payload rows are the same for DeepEP, streamed and compact: DeepEP moves one K-wide row per
(token, destination rank), and both dispatch-mode transfers mask unrouted rows at the load
(`peer_stream._pull_rows` / `_push_rows`), so the fixed layout costs the streamed mode compute
rows (131072 vs ~87k) and memory, not bytes on the wire; the K-duplicated layout belongs to the
fused runtime alone. Each of dispatch (x), combine (y), backward-of-combine (dy) and
backward-of-dispatch (dx partials) moves that payload once in BF16, so a checkpointed 2F+B
step moves 6 x 298.9 MiB = 1.75 GiB per rank over NVLink in every one of these rows.
Metadata differs slightly: DeepEP ships 8 int64 ids + 8 fp32 weights + a source index per
routed row (7.3 MiB per dispatch), mono-ep's `collect_routing` reads every peer's full
capacity-row `idx` / `w` slabs (10.5 MiB per dispatch) and pushes fp32 top-k weight grads per
routed row in the backward. With equal bytes, per-byte and per-layer rankings coincide; in
ms per GiB moved (2F+B, balanced / skewed), H100: DeepEP 20 SMs 12.04 / 14.61, DeepEP 40 SMs
10.03 / 11.98, streamed 11.86 / 13.78, compact 11.02 / 13.00, torch 11.92 / 13.72;
B200: DeepEP 20 SMs 12.42 / 14.33, DeepEP 64 SMs 7.31 / 8.09, streamed 7.19 / 7.85, compact
7.09 / 7.68, torch 8.19 / 8.85.

**Ratios, checkpointed 2F+B, balanced / skewed:**

| | H100 | B200 |
|---|---|---|
| torch all-to-all + SonicMoE vs DeepEP 20 SMs | 0.99x / 0.94x | 0.66x / 0.62x |
| streamed + SonicMoE vs DeepEP 20 SMs | 0.98x / 0.94x | 0.58x / 0.55x |
| compact + SonicMoE vs DeepEP 20 SMs | 0.92x / 0.89x | 0.57x / 0.54x |
| DeepEP as shipped (`barrier_ep`) vs DeepEP 20 SMs | 1.02x / 1.02x | 1.02x / 1.02x |
| DeepEP best SM count vs DeepEP 20 SMs | 0.83x / 0.82x (40 SMs) | 0.59x / 0.56x (64 SMs) |
| compact + SonicMoE vs DeepEP best SM count | 1.10x / 1.09x | 0.97x / 0.95x |
| streamed + SonicMoE vs DeepEP best SM count | 1.18x / 1.15x | 0.98x / 0.97x |
| DeepEP + grouped_mm vs torch + grouped_mm | 0.96x / 0.96x | 1.06x / 1.07x |
| compact + grouped_mm vs DeepEP + grouped_mm | 1.04x / 1.03x | 0.92x / 0.92x |

**What DeepEP's SM reservation does to the SonicMoE compute: nothing, in this path.** The
backend runs dispatch, the local kernel and combine serially on one stream, so DeepEP's
`num_sms` is never contending with the GEMMs. The profiler breakdown (one fwd+bwd per row
after the warm, self device time on rank 0, `profiler_breakdown_rank0`) puts the quack GEMM
kernels at 5.42-5.43 ms balanced / 7.06-7.12 ms skewed on H100 in every SonicMoE row (torch,
streamed, compact, DeepEP with and without the barrier), the same kernels at the same
timings. What the reservation does bound is DeepEP's own transfer kernels: at 20 SMs
`intranode::dispatch` is 1.63 ms per call and `intranode::combine` 2.12 ms per call on H100
(~190 / ~150 GB/s per rank one way for 299 MiB; the H100 NVLink one-way nominal is 450 GB/s),
against 0.98 ms per `_pull_rows` and ~0.9 ms per full-width `_push_rows` for the streamed
transfers; doubling the SM budget to 40 recovers 3 ms of the 16 ms fwd+bwd. On B200 the gap
is wider: the GEMM bucket is 2.69-2.78 ms in every SonicMoE row, DeepEP's dispatch is 2.57 ms
and its combine 3.58 ms per call at 20 SMs (~120 / ~90 GB/s per rank one way against a 900
GB/s one-way nominal), `_pull_rows` 0.46 ms and a full-width `_push_rows` ~0.44 ms (about 0.7
of nominal, the 5.3 microbench figure), and the SM sweep halves DeepEP's fwd+bwd from 17.1 ms
at 20 SMs to 9.4 ms at 64 without flattening. DeepEP's normal kernels move bytes from a fixed
SM subset, so on a GPU whose NVLink is twice as fast per SM they need proportionally more SMs
to fill the link; the streamed transfers launch `4 * SMs` programs and fill it by default.
(The torch row's NCCL kernels spin while waiting for peers, so their 15 / 10 ms of device time
is not a transfer cost; the `deep_ep_barrier` row's 6.6 ms in `notify_dispatch` on H100 is the
rank's wait after the host barrier, and the 2.4 ms in the symmetric-memory `barrier_kernel` of
the B200 dispatch-mode rows is the same kind of wait.) The 64-SM row was not profiled (the
breakdown covers the fixed list of five rows).

**Parity (both GPUs, both routings, `deep_ep_parity_tolerance`, bound rel-L2 1e-2, no
failures).** DeepEP + SonicMoE and DeepEP + grouped_mm against torch + the same local kernel:
`dscore`, `dgate_up`, `ddown` bitwise, `y` / `dx` rel-L2 3.37e-03 (balanced) / 3.39e-03
(skewed), the FP32-combine-vs-BF16-`index_add_` difference of 5.1. The compact dispatch mode
is **bitwise equal to DeepEP** on `y`, `dx`, `dscore`, `dgate_up` and `ddown` (both sum the
per-rank partials in FP32 in rank order and round once), as are the barrier and 40-SM DeepEP
rows against the 20-SM one. Quack candidates failed: 0; `_enter` extra barriers: 0; mean-rank
and max-rank 2F+B medians within 0.02 ms for the torch, DeepEP and compact rows. The peak
memory column of these runs is not comparable to 5.4: the three dispatch workspaces, the fused
runtime and the DeepEP buffer are resident throughout, so every row reads ~17 GiB above its 5.4
(a) figure; the relative ordering (fixed-layout rows +9.8 GiB, grouped_mm rows +19 GiB) is unchanged.

**Reading.** Against the DeepEP configuration axolotl ships (20 SMs, 256 MiB buffer,
`barrier_ep`), both dispatch modes with SonicMoE win on both GPUs: on H100 DeepEP ties torch
all-to-all and the streamed mode and the compact mode is 8-11% faster; on B200 DeepEP is
1.5-1.6x slower than torch and 1.7-1.9x slower than streamed / compact. That configuration is not
DeepEP's ceiling, because its normal kernels scale with `num_sms`: at 40 SMs (the most the
plugin's buffer admits) DeepEP + SonicMoE is the fastest composed row on H100, 10% ahead of
compact and 18% ahead of streamed; on B200 even 64 SMs (a buffer 1.5x the plugin default)
only brings DeepEP to within 2-3% of streamed and 3-5% behind compact, with the sweep still
improving. Same bytes on the wire and bitwise-identical results throughout. So: against DeepEP
as shipped, compact and streamed beat it everywhere; against DeepEP tuned, they lose by ~10% on
H100 and keep a 3-5% (compact) / 2-3% (streamed) lead on B200 at 64 SMs that a larger SM budget
would probably close. Two follow-ups fall out for the plugin regardless of the dispatch mode:
expose `num_sms` (`Buffer.set_num_sms`) and size `expert_parallel_num_nvl_bytes` from the
config's hint, since the shipped 20 SMs leave 17-44% of the DeepEP layer time on the table.
The structural difference that remains is that DeepEP's kernels are built to run on a reserved
SM subset alongside compute on another stream, which this serialized backend never exploits,
while the streamed transfers take the whole GPU for ~0.5-1 ms per op; the overlap of the next
layer's gather in section 4 is the dispatch mode's own headroom, and a compute-overlapped
DeepEP would be the baseline to beat after that.
