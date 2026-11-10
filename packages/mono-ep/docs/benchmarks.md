# Benchmarks

## Contract and methodology

A Qwen3-30B-A3B MoE layer at expert parallel 8: 16,384 tokens per GPU, hidden 2048,
intermediate 768, 128 experts (16 per GPU), top-8, so each GPU receives about 131,072 routed
rows; FP32 master weights; the whole layer `fullgraph` compiled; five warmups, fifteen
measured iterations, median over iterations of the slowest rank. The 235B rows use hidden
4096, intermediate 1536, 128 experts, top-8.

- "Checkpointed 2F+B" is the forward, the recomputed forward and the backward of one layer,
  grouped weight gradients and runtime staging included, router and optimizer excluded.
- The "NCCL control" shares the same persistent forward and runs a fixed-capacity NCCL
  all-to-all backward in the same process and allocation. The B200 "peer control" is the
  earlier symmetric-memory chunk pipeline.
- Hopper runs are blockwise FP8 (E4M3, 1x128 activation and 128x128 weight scales, with
  `MONO_EP_SM90_FP8_TRANSPORT=1` and `MONO_EP_SM90_FP8_COMBINE=1`); B200 runs are BF16
  compute. B200 runs bump the weight versions every iteration so the weight-layout cache
  rebuilds as it would in training.
- Balanced routing is uniform random; skewed routing concentrates the hot experts on one
  rank.
- Runner: `examples/modal_hopper_persistent_timing.py` (SM90) and the B200 paired-timing
  runner in `examples/`; the Modal app id of every run is listed under its table. Hopper
  runs assert the allocated GPU in-process (`H100!` strict, `NVIDIA H200`).

## 8xH200 against the published prime-rl numbers

Balanced routing, max-rank median ms, blockwise FP8, against the prime-rl fused MoE numbers
published for the same hardware (ratio below 1 is faster):

| Shape | | mono-ep | prime-rl | ratio | NCCL control |
|---|---|---:|---:|---:|---:|
| 30B-A3B | forward | 3.277 | 3.9 | 0.840 | 3.272 |
| 30B-A3B | backward | 6.601 | 8.2 | 0.805 | 40.653 |
| 30B-A3B | checkpointed 2F+B | 12.317 | 15.9 | 0.775 | 46.490 |
| 235B-A22B | forward | 8.447 | 9.5 | 0.889 | 8.445 |
| 235B-A22B | backward | 21.124 | 24.6 | 0.859 | 90.498 |
| 235B-A22B | checkpointed 2F+B | 37.333 | 43.7 | 0.854 | 107.124 |

Run ap-V06KM1a7K9KSAOsvOo4CU4 (one allocation). The 235B shape runs with 5 forward / 3
backward stages and the 2 KB dispatch pull, because the hidden-2048 stage counts exceed SM90
shared memory at hidden 4096.

Skewed routing costs 26% on the 30B 2F+B (15.504 ms) and 28% on the 235B 2F+B (47.799 ms) on
the same build; the NCCL control moves by 5-8%.

## 8xH100, Hopper tuning rounds

Strict `H100!`, 30B-A3B, balanced, ms, one allocation per row:

| Build | Forward | Backward | Checkpointed 2F+B | NCCL control 2F+B |
|---|---:|---:|---:|---:|
| epilogue integration (start of the rounds) | 4.297 | 9.026 | 16.958 | 51.682 |
| round 2b | 3.365 | 8.157 | 14.327 | 49.831 |
| round 3 | 3.255 | 7.534 | 13.480 | 49.911 |
| round 4 (this tree's SM90 defaults) | 3.385 | 7.036 | 12.998 | 49.647 |
| this branch, gate run | 3.458 | 7.266 | 13.021 | 49.616 |

Runs, top to bottom: ap-optFhUPUcC5ywxp9zJPPRe, ap-EsrRmakdPLt73RGAS69kK4,
ap-ICtWLrsXD20dMONnCSd4Nz, ap-9N7kM7V67pDfcDRXl81ZP5, ap-05NnV2Ph9NslFfBVYrnbLq. Skewed
routing on the gate run: 4.028 / 8.996 / 16.198 ms, control 52.146
(ap-tsjPCzmlgpvljSIRdiLOVi); round 4 measured 16.293 skewed.

## 8xB200, Blackwell tuning rounds

30B-A3B, BF16, balanced, max-rank median ms, weight versions bumped every iteration:

| Build | Forward | Backward | Checkpointed 2F+B | Peer control | NCCL control |
|---|---:|---:|---:|---:|---:|
| start of the B200 rounds (6 backward stages) | 2.36 | 4.73 | 9.17 | 17.89 | 25.00 |
| round 1 | 2.19 | 4.40 | 8.31 | 17.39 | 24.53 |
| round 2 | 2.137 | 4.256 | 7.845 | 17.260 | 24.537 |
| round 3 (this tree's SM100 defaults) | 2.112 | 4.138 | 7.659 | 17.196 | 24.531 |
| round 3 with 16 reserved SMs (the FSDP2 setting) | 2.114 | 4.272 | 7.823 | 17.391 | 24.670 |
| this branch, gate run | 2.098 | 4.172 | 7.660 | 17.046 | 24.242 |

Runs, top to bottom: ap-6Js1FDCUJz1HfsUbBOqL3b, ap-PFK8nCymI9bow1IChnsdrD,
ap-VBok02Nuf6psSFDeShXEHF, ap-rxwVgU6xxIwEFdt6fNcX3o (both round-3 rows),
ap-afG15T7ayDA5cPEJnqpPcC.

## Dispatch-only mode and the DeepEP contract

The Axolotl `deep_ep` and `torch` backends use a dispatch/combine contract (the collective
moves tokens, the local experts kernel runs between the two halves). A composable mono-ep
mode that serves that contract from the persistent kernels is developed on
`features/mono-ep-compose`, with its own design note and its measurements against torch
all-to-all and the fused path; no numbers for it are claimed here.

## What these do not measure

The H100 and B200 rows are not hardware- or precision-matched with the prime-rl H200 FP8
figure; the controls share the persistent forward, so they are not prime-rl's complete
baseline; absolute timings move by up to 16% between allocations of the same GPU type, so
rows from different allocations are reproductions rather than A/Bs; and these are layer
timings, not end-to-end training throughput.

## Correctness

- Two-GPU FP32-reference forward/backward validators with the x/dZ caches, the FP8 combine
  validator with forced ring wrap, and the forward FP32 reference: ap-VN6mzQm22BwbfnE68DZ5rO,
  ap-wufiJKL44xoazIZkfVaBQv, ap-gK4emj87Im7xQLO4dvCUXF (2xH100).
- The FSDP2 weight-cache contract test with optimizer steps on `dp_shard 2 x EP 2`:
  ap-TCBnqZV1beO55aH6SDxtBO (4xH100), ap-mkqekSW1wzK9nne3lsmOne (4xB200, 24 reserved SMs).
- Bitwise parity of every selector default against its restore value, and JIT kernel names
  and sources identical to the development branches modulo the package naming (SM90 and
  SM100).
- The Axolotl parity test `tests/e2e/multigpu/test_ep_mono_ep.py` (EP2 x FSDP2 against the
  `torch` backend with optimizer steps).
- Long-training convergence has only a loss-curve check: 2xH100, a 739M Qwen3-MoE, 300
  steps, two seeds, exact SwiGLU against `swiglu_fast` through the Axolotl seam.
