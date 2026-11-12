# Roadmap

What ships in this package, and what is next. The development history behind it is kept with
the benchmark logs, outside the package.

## Shipped

- Fused persistent forward and backward per direction (dispatch over NVLink symmetric memory,
  expert GEMMs with SwiGLU in the epilogue, combine) on SM90 (blockwise FP8, BF16) and SM100
  (BF16, MXFP8 forward), `fullgraph`-compilable custom ops, nothing retained from the forward.
- Weight-layout cache with an FSDP2 invalidation contract; SM reservation and `NCCL_MAX_CTAS`
  for FSDP2 next to the spinning kernels.
- Dispatch mode: mono-ep's peer dispatch and combine around any local experts kernel, with the
  bounded compact receive.
- Owned-backward ring floor (default on), SM100 MXFP8 backward and in-kernel weight gradients
  (opt-in).
- Bounded workspace (default on): backward pools, the backward ring and the SM100 BF16 forward
  ring sized from the routed rows instead of the worst case, on SM90 and SM100.
- SM100 backward stack: stored gate/up pre-activations in checkpoint recomputes (default `auto`,
  wide experts only) and direct-write weight gradients (default on); MXFP8 weight gradients with
  the CUDA weight-form builder and the row requantization (opt-in, with the MXFP8 backward).
- NVFP4 LoRA on SM100 (frozen packed NVFP4 experts with LoRA adapters, CUDA requant, lean
  recompute tail), through the package API and the Axolotl precision `nvfp4_lora` (PEFT
  `lora_target_parameters` adapters on modelopt NVFP4 checkpoints, the requant cache's
  optimizer hook, adapter-only checkpoints), with just-in-time merged weights per layer, fused
  transposed forms and bounded pools and rings.
- Axolotl backend `expert_parallel_backend: mono_ep` (fused and dispatch modes, fallback to
  `torch`), a TRL `SFTTrainer` example, validators and the benchmarks in
  [benchmarks.md](benchmarks.md).

## Next

- NVFP4 full fine-tuning (the LoRA wiring ships; the merged cache of the LoRA path is the
  seam for it).
- The backward: the gap to Mixture of Kittens at every B200 shape, and to the dispatch paths at
  the widest shapes, is in the backward's weight gradients and recompute.
- Backward pools: after the bounds, 2.1 / 4.2 / 5.9 GB per rank of x / dy / h / dz pools remain
  for the recompute-in-backward design (Mixture of Kittens sizes by its own tokens and replays
  macrobatches); shrinking them needs a pool design change, not sizing.
- Forward ring at narrow experts: the Qwen3-30B-A3B ring keeps one slot per budgeted row (1.50 GB)
  because the live-block ring cost 4-8 % of the forward; a dispatch order that reuses slots could
  reach the live bound.
- The MXFP8 and SM90 forwards keep the worst-case forward ring.
- The MXFP8 transport ring and the MXFP8 scale pools keep the worst case under the bound.
- Stored context: count hits and misses per layer. At Qwen3-235B-A22B skewed routing `auto` only
  ties `off` (26.33 / 26.38 ms), and it is not yet known whether the hot rank overflows the buffer
  or hides the saving behind its transfers.
- Layering refactor of the Python package into stateless ops with fake implementations, a
  functional layer and the API.
- SM120 (RTX PRO 6000 / RTX 50-series): no TMEM or tcgen05, so it needs its own kernels.
- `fp8_bwd` (BF16 forward, FP8 backward) and clamped SwiGLU in the fused backward.
- A stable C ABI for the kernels ([stable-abi-design.md](stable-abi-design.md)), then a
  standalone repository from `git subtree split --prefix=packages/mono-ep` and PyPI wheels.
