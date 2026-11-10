# Composing mono-ep dispatch/combine with a local experts kernel

Status: implemented on `features/mono-ep-compose` as `expert_parallel_mono_ep_mode: dispatch`
(and the `auto` fallback). This note records the survey of the package's peer path that the
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
    group=ep_group, max_tokens_per_rank=cap, top_k=K, hidden=H
)
recv_x, recv_idx, recv_w = runtime.dispatch(
    x_bf16, topk_idx_i64, topk_w_f32, num_local_experts
)
y = runtime.combine(local_out, recv_idx, topk_idx_i64, num_local_experts)
runtime.destroy()
```

- `recv_x` `[world*cap, H]` BF16, `recv_idx` `[world*cap, K]` int64 local ids / `-1`, `recv_w`
  `[world*cap, K]` fp32 (0 on `-1`): the DeepEP layout.
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

1. barrier (peers finished reading the previous op's buffers; stream-ordered, so it also
   orders this rank's previous reads);
2. local writes: `x -> x_buf[:T]`, `idx -> idx_buf` (tail `-1`), `w -> w_buf`;
3. barrier (publish);
4. read every peer's `idx_buf`/`w_buf` (small), remap to local ids: `recv_idx[s*cap+t, k]`,
   `recv_w` masked to 0 where not local; `routed[s*cap+t] = any_k(local)`;
5. `pack_peer_rows([x_buf_s for s], assign = where(routed, s*cap+t, -1), topk=1)` pulls exactly
   the routed rows over NVLink; unrouted rows are zero.

`peer_combine(local_out, routed, topk_idx, num_local_experts)` (`routed = (recv_idx >= 0).any(-1)`):

1. barrier; 2. `return_peer_rows(local_out_bf16, assign)` pushes row `s*cap+t` to
   `y_buf_s[r*cap + t]` (unique slot per `(producer, source token)`); 3. barrier;
4. `y = sum_p where(routed_to[p, t], y_buf_r[p*cap + t], 0)` in fp32, cast to `local_out.dtype`.

Backward of dispatch = `peer_combine` of `grad_recv_x` into `x`'s dtype plus the same push of
`grad_recv_w` through `gw`; backward of combine = the gather half of `peer_dispatch` applied to
`grad_y`. Each op is write → barrier → read; the leading barrier of the next op protects reuse of
the single buffer set, so no double buffering is needed and any op order is safe.

Barriers are `_SymmetricMemory.barrier` (system-scope release/acquire on the allocation's signal
pad, 10-minute timeout so a skewed rank behaves like a slow NCCL collective, not a device
assert).

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

`collect()` (combine and dispatch-backward receive) does one masked FP32 pass over the own
`[world, T, H]` return buffer per producer rank: `world` reads of `T x H` plus the FP32 sum. At
EP2 this is noise next to the GEMMs; at EP8 it is the first thing to fuse (one Triton pass that
reads all producer slots of a token once and writes BF16), together with the `torch.stack` of
the `world` routing views in `collect_routing()`.

## 4. Follow-ups (not in this change)

- Compact receive rows via an optional host count (trade one sync per layer for a smaller local
  kernel input, like the torch backend).
- Fuse `collect()` and `collect_routing()` into one Triton pass before EP8 timings (see 3.6).
- FP8 transport for `recv_x` when the local kernel accepts it.
- Overlap: issue the gather of layer `l+1` under layer `l`'s local kernel (needs a second buffer
  set).

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
