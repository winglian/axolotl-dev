# Benchmarks

Every number on this page comes from one re-benchmark of this tree: one Modal allocation per GPU
type, every row at three layer shapes, run through the harnesses in `examples/`. Numbers from the
tuning rounds that built these kernels are kept with the development logs, not here.

## Contract and methodology

One MoE layer at expert parallel 8, 16,384 tokens per GPU, top-8, at three shapes:

| Shape | Hidden | Intermediate | Experts (per GPU) | Routed TFLOP per GPU: forward / backward / 2F+B |
|---|---:|---:|---:|---|
| Qwen3-30B-A3B | 2048 | 768 | 128 (16) | 1.237 / 2.474 / 4.948 |
| Qwen3-235B-A22B | 4096 | 1536 | 128 (16) | 4.948 / 9.896 / 19.79 |
| Beam | 5120 | 2560 | 256 (32) | 10.31 / 20.62 / 41.23 |

- "Checkpointed 2F+B" is contract (c) below: the forward, the recomputed forward and the backward of one layer,
  weight gradients and runtime staging included, router and optimizer excluded. Five warmups,
  fifteen measured iterations, median over iterations of the slowest rank, paths interleaved
  with the order reversed every other iteration.
- The fused rows use FP32 master weights, the whole layer `fullgraph` compiled, and bump the
  weight versions every iteration, so the weight-layout cache rebuilds as it does in training.
  Hopper runs are blockwise FP8 (E4M3, 1x128 activation and 128x128 weight scales,
  `MONO_EP_SM90_FP8_TRANSPORT=1`, `MONO_EP_SM90_FP8_COMBINE=1`); B200 runs are BF16.
- The NCCL control shares the same persistent forward and runs a fixed-capacity NCCL
  all-to-all backward in the same process.
- Balanced routing is uniform random; skewed routing concentrates the hot experts on one rank.
- Routed TFLOPS per GPU is routed GEMM FLOPs (2 x rows x 3 x hidden x intermediate per
  forward, twice that per backward, four forwards per 2F+B) divided by the whole measured
  time, so dispatch, combine, quantization, the backward's gate/up recompute and weight-layout
  preparation all count against it. Dense peaks: 1,979 TFLOPS FP8 on H100 and H200, 2,250
  TFLOPS BF16 on B200.

Timing contracts, used as labels in every table:

- **(a)** forward + backward with the forward's context stored, no recompute.
- **(b)** forward + partial recompute (the gate/up projection only) + backward: Mixture of
  Kittens' checkpointed mode (`recompute_forward_context`).
- **(c)** forward + the full forward recomputed + backward: Axolotl's non-reentrant activation
  checkpointing (`activation_checkpointing: true`). Every mono-ep "2F+B" number here is (c), and
  so is the Mega MoE number Prime Intellect published.
- **(d)** forward + backward with the fused op's output saved under selective checkpointing
  (`activation_checkpointing: {selective: {save: ["mono_ep::persistent_forward"]}}`), the same as
  prime-rl's "selective" mode.

The Mega MoE rows compare (c) with (c). The Mixture of Kittens rows compare mono-ep's (c) with
MoK's (b), which charges mono-ep one more full forward than MoK. No (d) number appears in these
tables: none was measured in these allocations. Per layer, (c) keeps the layer input
(tokens x hidden BF16: 64 / 128 / 160 MiB at Qwen3-30B-A3B / Qwen3-235B-A22B / Beam with 16,384
tokens per GPU) plus the routing (1.5 MiB); (d) also keeps the output, twice that. Prime
Intellect publishes no memory numbers.

Allocations (Modal app, commit, device asserted in-process before the first row):

| Hardware | App | Commit | Notes |
|---|---|---|---|
| 8xH200 | ap-8NiyZxNCgoy9sN6CVq0J7y | ef76fd6804 | preflight all-to-all 311.6 GB/s |
| 8xH100 (`H100!`) | ap-QcAVGSp7VUaUBjZYPBHEiB | ef76fd6804 | slow NVLink, see below |
| 8xB200 | ap-rtKAANKTJOx3UX04OFsnQs | 2d6ef94f60 | about 3% slow, see below |
| 8xB200, NVFP4 rows | ap-0oxG5Q7sNY60DrAtVxUXlp | 5e210e1a3 (NVFP4 backward default 2051) | preflight all-to-all 317.1 GB/s |

`ef76fd6804` differs from `2d6ef94f60` only in three timing harnesses (path subsets for the
80 GB H100, the dispatch contract's fused-row description); the package sources and the JIT
kernels are identical. The NVFP4 rows ran after the NVFP4 backward's lean recompute tail became
the default; that change touches only the SM100 NVFP4 backward kernel, so the other rows stand.

## 8xH200 against Mega MoE

Balanced routing, max-rank median ms, blockwise FP8, against the Mega MoE numbers Prime
Intellect published for prime-rl on the same hardware (ratio below 1 is faster):

| Shape | | mono-ep | Mega MoE (published) | ratio | NCCL control |
|---|---|---:|---:|---:|---:|
| 30B-A3B | forward | 3.498 | 3.9 | 0.897 | 3.509 |
| 30B-A3B | backward | 6.962 | 8.2 | 0.849 | 40.900 |
| 30B-A3B | (c) checkpointed 2F+B | 12.630 | 15.9 | 0.794 | 46.668 |
| 235B-A22B | forward | 8.504 | 9.5 | 0.895 | 8.324 |
| 235B-A22B | backward | 21.275 | 24.6 | 0.865 | 90.789 |
| 235B-A22B | (c) checkpointed 2F+B | 38.302 | 43.7 | 0.876 | 107.518 |

Prime Intellect's published torch all-to-all + DeepGEMM FP8 baseline is 39.3 ms (30B-A3B) and
71.9 ms (235B-A22B). The published numbers are their measurements, not reruns.

## Fused training layer

Balanced routing unless marked, max-rank median ms:

| Hardware | Shape | Forward | Backward | (c) checkpointed 2F+B | TFLOPS (% of peak) | Skewed (c) | NCCL control (c) | vs control | Peak allocated GiB | Saved activations per layer, (c) / (d) MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8xH200 FP8 | 30B-A3B | 3.50 | 6.96 | 12.63 | 392 (20%) | 15.81 | 46.67 | 3.69x | 17.2 | 64 / 128 |
| 8xH200 FP8 | 235B-A22B | 8.50 | 21.27 | 38.30 | 517 (26%) | 47.40 | 107.52 | 2.81x | 35.8 | 128 / 256 |
| 8xH200 FP8 | Beam | 18.34 | 45.40 | 79.50 | 519 (26%) | 100.84 | 190.40 | 2.40x | 49.2 | 160 / 320 |
| 8xH100 FP8 | 30B-A3B | 3.66 | 7.27 | 13.55 | 365 (18%) | 16.58 | 49.91 | 3.68x | 17.2 | 64 / 128 |
| 8xH100 FP8 | 235B-A22B | 9.53 | 25.05 | 42.73 | 463 (23%) | 53.82 | 113.76 | 2.66x | 35.8 | 128 / 256 |
| 8xH100 FP8 | Beam | 19.72 | 55.30 | 90.86 | 454 (23%) | 112.80 | did not fit |  | 49.2 | 160 / 320 |
| 8xB200 BF16 | 30B-A3B | 2.15 | 4.20 | 7.79 | 635 (28%) | 9.26 | 24.89 | 3.19x | 21.9 | 64 / 128 |
| 8xB200 BF16 | 235B-A22B | 4.27 | 13.06 | 21.65 | 914 (41%) | 27.11 | 65.35 | 3.02x | 34.4 | 128 / 256 |
| 8xB200 BF16 | Beam | 8.95 | 30.48 | 49.29 | 837 (37%) | 60.23 | 113.38 | 2.30x | 66.7 | 160 / 320 |

Routed TFLOPS per GPU by direction (balanced):

| Hardware | Shape | Forward | Backward | (c) checkpointed 2F+B |
|---|---|---:|---:|---:|
| 8xH200 FP8 | 30B-A3B | 354 | 355 | 392 |
| 8xH200 FP8 | 235B-A22B | 582 | 465 | 517 |
| 8xH200 FP8 | Beam | 562 | 454 | 519 |
| 8xH100 FP8 | 30B-A3B | 338 | 340 | 365 |
| 8xH100 FP8 | 235B-A22B | 519 | 395 | 463 |
| 8xH100 FP8 | Beam | 523 | 373 | 454 |
| 8xB200 BF16 | 30B-A3B | 576 | 588 | 635 |
| 8xB200 BF16 | 235B-A22B | 1158 | 758 | 914 |
| 8xB200 BF16 | Beam | 1151 | 676 | 837 |
| Mega MoE (published), 8xH200 FP8 | 30B-A3B | 317 | 302 | 311 |
| Mega MoE (published), 8xH200 FP8 | 235B-A22B | 521 | 402 | 453 |

The share of peak rises with the GEMM size, because the transfers and the per-row epilogue
work are a larger share of the small 30B-A3B layer. The backward is the weaker direction
everywhere; it recomputes the gate/up GEMM per M-block (about two thirds of a forward that is
not counted) and runs the weight gradients as grouped GEMMs after the main loop.

## Dispatch mode and the Axolotl backends on one contract

The same layer, routing and timing method run through Axolotl's expert-parallel backends with
SonicMoE as the local experts kernel and BF16 expert weights (SonicMoE's dtype), every variant
of a column interleaved in one process (`examples/dispatch_sonicmoe_contract.py --deep-ep
--retime`). The mono-ep fused row in these tables is the same kernels run uncompiled with BF16
parameters quantized per call (FP8 transport on Hopper), so it is a same-run reference rather
than the fused-harness number above. DeepEP is v1.2.1 with the plugin's NVLink buffer; the
20-SM row is the `deep_ep` backend as shipped. Contract (c), balanced / skewed, ms.

8xH200:

| Variant | 30B-A3B | 235B-A22B | Beam |
|---|---|---|---|
| mono-ep fused (this tree, uncompiled, BF16 parameters quantized per call) | 13.41 / 16.70 | 42.19 / 54.58 | 88.56 / 110.09 |
| mono-ep dispatch, bounded compact receive | 19.09 / 21.59 | 46.67 / 56.59 | 85.09 / 104.54 |
| mono-ep dispatch, compact receive | 19.93 / 22.17 | 47.61 / 58.32 | 86.81 / 107.00 |
| DeepEP + SonicMoE, 40 SMs | 18.12 / 20.21 | 45.87 / 56.15 | 85.24 / 105.40 |
| DeepEP + SonicMoE, 20 SMs (Axolotl default) | 20.10 / 23.56 | 47.31 / 58.68 | 85.86 / 106.41 |
| torch all-to-all + SonicMoE | 22.09 / 24.36 | 50.28 / 60.60 | 89.49 / 111.32 |

8xH100:

| Variant | 30B-A3B | 235B-A22B | Beam |
|---|---|---|---|
| mono-ep fused (this tree, uncompiled, BF16 parameters quantized per call) | 14.07 / 17.42 | 44.62 / 56.52 | 94.69 / 118.08 |
| mono-ep dispatch, bounded compact receive | 19.66 / 22.09 | 48.85 / 58.24 | 87.57 / 106.09 |
| mono-ep dispatch, compact receive | 20.48 / 22.94 | 49.92 / 59.94 | 89.00 / 109.31 |
| DeepEP + SonicMoE, 40 SMs | 26.94 / 29.79 | 51.79 / 61.61 | 89.21 / 109.85 |
| DeepEP + SonicMoE, 20 SMs (Axolotl default) | 28.56 / 31.65 | 53.73 / 64.41 | 91.11 / 112.28 |
| torch all-to-all + SonicMoE | 22.70 / 24.78 | 52.69 / 62.35 | 91.92 / 112.43 |

8xB200:

| Variant | 30B-A3B | 235B-A22B | Beam |
|---|---|---|---|
| mono-ep fused (this tree, uncompiled, BF16 parameters quantized per call) | 7.77 / 9.32 | 21.25 / 26.05 | 45.72 / 56.47 |
| mono-ep dispatch, bounded compact receive | 11.53 / 12.48 | 25.52 / 29.68 | 45.35 / 56.94 |
| mono-ep dispatch, compact receive | 12.35 / 13.30 | 26.17 / 30.84 | 45.67 / 54.65 |
| DeepEP + SonicMoE, 64 SMs | 12.71 / 13.84 | 26.39 / 31.20 | 44.45 / 52.43 |
| DeepEP + SonicMoE, 40 SMs | 14.80 / 16.79 | 30.57 / 35.68 | 50.00 / 57.38 |
| DeepEP + SonicMoE, 20 SMs (Axolotl default) | 21.74 / 24.62 | 40.82 / 47.22 | 60.77 / 70.90 |
| torch all-to-all + SonicMoE | 14.13 / 15.32 | 29.12 / 33.63 | 49.71 / 58.48 |

The bounded compact receive (`expert_parallel_mono_ep_bounded`) is bitwise equal to the
compact receive (forward and backward, both routings) at every shape on every GPU, and no
re-timed row changed its output. On B200 it leads DeepEP at its best SM count at 30B-A3B and
235B-A22B; on H200 DeepEP at 40 SMs is about 1 ms faster than it at both shapes. At the Beam shape the dispatch paths, DeepEP and the fused kernels land within a
few percent of each other on every GPU: with 32 local experts and hidden 5120 the expert GEMMs
dominate the layer, so fusing the transfers into them buys little.

## Mixture of Kittens on 8xB200 (same allocation)

cursor/mixture-of-kittens at f3a8bd2432 built for SM100 in the B200 image (one-line
`kittens::detail::tcgen05::commit` qualification), run in the same allocation as the mono-ep
rows with the best knobs from an earlier sweep (fwd / bwd comm SMs, minibatch:
30B-A3B 52/52, 8192 (BF16) / 52/52, 8192 (MXFP8), 235B-A22B 36/36, 4096 (BF16) / 36/36, 4096 (MXFP8), Beam 24/24, 16384 (BF16) / 36/24, 4096 (MXFP8)). Balanced routing, ms; MoK's (b) is its
forward plus a backward that recomputes only its forward context (gate/up), while mono-ep's (c)
replays the whole forward. MoK computes a shared expert of the same intermediate
size inside every timed number (1/8 more FLOPs at top-8), so its times include work mono-ep
does not do, and its routed TFLOPS understate it:

| Shape | MoK BF16 F / B / (b) | MoK MXFP8 F / B / (b) | mono-ep BF16 F / B / (c) | mono-ep (c) with the MXFP8 backward | MoK BF16 TFLOPS | mono-ep TFLOPS |
|---|---|---|---|---:|---:|---:|
| 30B-A3B | 2.25 / 3.47 / 6.96 | 2.26 / 2.85 / 6.40 | 2.15 / 4.20 / 7.79 | 7.32 | 711 | 635 |
| 235B-A22B | 6.26 / 10.96 / 21.05 | 5.54 / 7.81 / 17.06 | 4.27 / 13.06 / 21.65 | n/a (hidden > 2048) | 940 | 914 |
| Beam | 11.68 / 21.59 / 40.63 | 9.11 / 15.25 / 30.33 | 8.95 / 30.48 / 49.29 | n/a (hidden > 2048) | 1015 | 837 |

MoK's (b) is below mono-ep's (c) at every shape, by 0.8 ms at 30B-A3B, 0.6 ms at 235B-A22B
and 8.7 ms at Beam, although (c) includes one more full forward than (b). mono-ep's forward
alone is faster at all three shapes and MoK's backward is faster; the two have not been timed
under one contract.

## SM100 opt-ins on 8xB200

MXFP8 backward (`MONO_EP_SM100_BACKWARD_PRECISION=mxfp8`, hidden up to 2048) and in-kernel
weight gradients (`MONO_EP_SM100_INKERNEL_DW=1`, `_WIDE=1` for 256 x 256 tiles), interleaved
with the default per iteration in the fused-harness rows above, backward / (c) ms:

| Shape | Routing | default BF16 B / 2F+B | MXFP8 backward B / 2F+B | in-kernel dW B / 2F+B | in-kernel dW, wide B / 2F+B |
|---|---|---|---|---|---|
| 30B-A3B | balanced | 4.20 / 7.79 | 3.77 / 7.32 | 4.30 / 7.81 | 4.25 / 7.75 |
| 30B-A3B | skewed | 5.00 / 9.26 | 4.51 / 8.83 | 5.18 / 9.34 | 5.15 / 9.28 |
| 235B-A22B | balanced | 13.06 / 21.65 | n/a | 13.45 / 21.84 | 13.19 / 21.33 |
| 235B-A22B | skewed | 15.88 / 27.11 | n/a | 16.41 / 26.59 | 16.01 / 26.38 |
| Beam | balanced | 30.48 / 49.29 | n/a | 30.33 / 48.76 | 28.99 / 47.99 |
| Beam | skewed | 38.18 / 60.23 | n/a | 39.11 / 61.75 | 36.68 / 60.59 |

Neither is a default: the MXFP8 backward is not bitwise (about 6-7% relative error against an
FP32 reference) and the in-kernel dW costs about 0.1 ms at 30B-A3B while it saves 1.5 ms per
backward at Beam.

## Backward ring floor

`MONO_EP_BACKWARD_RING_WAVES` (default 1) against `0` (the forward sizing), both resident in
one process unless marked; backward / (c) ms. The Beam case runs 8,192 tokens per GPU:

| Hardware | Shape (tokens per GPU) | Routing | ring floor off: B / 2F+B | default (1 wave): B / 2F+B | dx bitwise |
|---|---|---|---|---|---|
| 8xB200 BF16 | 30B-A3B (16,384) | balanced | 4.12 / 7.67 | 4.08 / 7.67 | True |
| 8xB200 BF16 | 30B-A3B (16,384) | skewed | 5.55 / 10.20 | 5.59 / 10.21 | True |
| 8xB200 BF16 | 235B-A22B (16,384) | balanced | 14.50 / 23.06 | 14.60 / 23.07 | True |
| 8xB200 BF16 | 235B-A22B (16,384) | skewed | 19.35 / 30.33 | 19.54 / 30.64 | True |
| 8xB200 BF16 | Beam (8,192) | balanced | 25.70 / 33.60 | 16.66 / 24.40 | True |
| 8xB200 BF16 | Beam (8,192) | skewed | 34.07 / 42.74 | 22.43 / 32.39 | True |
| 8xH100 FP8 | 30B-A3B (16,384) | balanced | 13.06 / 18.80 | 13.05 / 18.73 | True |
| 8xH100 FP8 | 30B-A3B (16,384) | skewed | 14.56 / 22.00 | 14.32 / 21.93 | True |
| 8xH100 FP8 | 235B-A22B (16,384) | balanced | 36.76 / 51.74 | 36.74 / 51.75 | not checked (separate sets) |
| 8xH100 FP8 | 235B-A22B (16,384) | skewed | 49.08 / 69.41 | 41.82 / 62.67 | not checked (separate sets) |
| 8xH100 FP8 | Beam (8,192) | balanced | 70.08 / 86.30 | 37.58 / 52.40 | True |
| 8xH100 FP8 | Beam (8,192) | skewed | 97.09 / 118.67 | 50.48 / 70.28 | True |

The floor only raises the ring, so the 30B-A3B and 235B-A22B shapes keep their kernels; at the
Beam shape it removes the backward's ring stalls. The 235B-A22B H100 rows ran the two settings
in separate processes, so they carry run-to-run noise rather than a paired difference.

## NVFP4 LoRA (SM100)

`precision="nvfp4_lora"` (frozen NVFP4 experts with LoRA) on 8xB200 against the BF16
persistent path and two NVFP4 floors with an all-to-all around them (the fp4_cute grouped GEMM
and the cuDNN frontend grouped NVFP4 GEMM), all resident in one process, ms. The rebuild column
is the per-step merged-weight requant per layer (`MONO_EP_NVFP4_REQUANT`). The Beam case runs
8,192 tokens per GPU. Own allocation (ap-0oxG5Q7sNY60DrAtVxUXlp), run after the NVFP4 backward
default changed:

| Shape (tokens per GPU) | Routing | nvfp4_lora F / B / (c) | BF16 persistent (c) | fp4_cute + all-to-all (c) | cuDNN grouped + all-to-all (c) | rebuild: `cuda` / Triton / `cuda_fused` |
|---|---|---|---|---|---|---|
| 30B-A3B (16,384) | balanced | 2.08 / 3.97 / 7.90 | 7.56 | 30.47 | 28.03 | 0.35 / 0.87 / 0.31 |
| 30B-A3B (16,384) | skewed | 2.79 / 6.35 / 11.70 | 10.00 | 38.24 | 35.15 |  |
| 235B-A22B (16,384) | balanced | 4.03 / 9.22 / 17.12 | 21.08 | 63.69 | 52.94 | 0.90 / 2.75 / 0.74 |
| 235B-A22B (16,384) | skewed | 5.60 / 13.99 / 24.69 | 29.01 | 75.11 | 66.88 |  |
| Beam (8,192) | balanced | 3.05 / 7.88 / 13.61 | 24.39 | 41.73 | 36.46 | 3.23 / 10.24 / 2.50 |
| Beam (8,192) | skewed | 4.20 / 11.92 / 20.09 | 32.17 | 55.61 | 48.40 |  |

The lean recompute tail (NVFP4 backward options bit 2048, the default) against the previous
default 3, each on a runtime built the same way in a process with only NVFP4 runtimes, ms:

| Shape (tokens per GPU) | Routing | lean tail, options 2051 (default): B / 2F+B | options 3 (before): B / 2F+B | backward change |
|---|---|---|---|---:|
| 30B-A3B (16,384) | balanced | 3.94 / 7.93 | 3.97 / 7.94 | -0.7% |
| 30B-A3B (16,384) | skewed | 6.26 / 11.47 | 6.25 / 11.48 | +0.3% |
| 235B-A22B (16,384) | balanced | 7.13 / 14.25 | 7.55 / 14.73 | -5.6% |
| 235B-A22B (16,384) | skewed | 10.22 / 19.56 | 10.86 / 20.19 | -5.8% |
| Beam (8,192) | balanced | 5.86 / 11.03 | 6.19 / 11.31 | -5.4% |
| Beam (8,192) | skewed | 8.35 / 15.00 | 8.74 / 15.38 | -4.4% |

The NVFP4 path is sensitive to what else is resident: with the BF16 runtimes and the
all-to-all baselines in the same process its 2F+B is slower at the larger shapes (balanced, ms):

| Shape | nvfp4_lora 2F+B, NVFP4 runtimes only | nvfp4_lora 2F+B, with the BF16 runtimes and all-to-all baselines resident |
|---|---:|---:|
| 30B-A3B | 7.92 | 7.90 |
| 235B-A22B | 14.27 | 17.12 |
| Beam | 11.03 | 13.61 |

## Fused LoRA on BF16 experts (`mxfp8_lora`, `bf16_lora`)

One MoE layer through the Axolotl experts seam (`examples/benchmark_lora_b200.py`), 8xB200
mono-ep-dev, EP8, 16,384 tokens per GPU, top-8, PEFT expert adapter rank 16, BF16 base. One
variant per process, 5 warmups and 40 measured iterations with the two contracts interleaved
(order reversed every other iteration), host wall max over ranks, median ms; peak is the
allocator peak within a contract, max over ranks. Every iteration counts as an optimizer step: the
fused paths rebuild their weight forms inside the step, the LoRA variants also merge
`W + s B @ A`; the dispatch baseline (mono-ep peer dispatch around SonicMoE LoRA, MXFP8 not
applicable) builds its effective weight per call. (c) is checkpointed 2F+B, (d) the step with the
fused output (or the dispatch / combine ops) saved under SAC.

| Shape | Routing | `mxfp8_lora` (c) / (d) | `mxfp8` full FT (c) / (d) | dispatch + SonicMoE LoRA (c) / (d) | `bf16_lora` (c) / (d) | `bf16` full FT (c) / (d) | peak GiB: `mxfp8_lora` / full / dispatch | merge / forms per layer |
|---|---|---|---|---|---|---|---|---|
| 30B-A3B | balanced | 7.53 / 6.05 | 7.34 / 5.82 | 12.48 / 11.50 | 8.08 / 6.68 | 8.01 / 6.49 | 3.9 / 5.1 / 20.6 | 0.11 / 0.13 |
| 30B-A3B | skewed | 9.13 / 7.24 | 9.02 / 7.11 | 13.81 / 12.68 | 9.67 / 7.84 | 9.74 / 7.83 | 3.9 / 5.1 / 20.6 | |
| 235B-A22B | balanced | 15.05 / 12.50 | 15.80 / 13.16 | 29.92 / 27.65 | 21.87 / 16.75 | 24.86 / 18.98 | 8.7 / 11.3 / 41.7 | 0.44 / 0.37 |
| 235B-A22B | skewed | 18.81 / 15.39 | 19.82 / 16.41 | 34.81 / 32.25 | 27.31 / 20.64 | 31.48 / 23.23 | 8.7 / 11.3 / 41.7 | |
| Beam | balanced | 27.88 / 23.96 | 30.54 / 26.43 | 54.07 / 50.92 | 48.51 / 36.50 | 55.35 / 43.40 | 16.8 / 22.1 / 62.8 | 1.78 / 1.33 |
| Beam | skewed | 34.19 / 29.06 | 37.48 / 32.12 | 62.47 / 60.52 | 59.96 / 45.44 | 68.40 / 55.34 | 16.8 / 22.1 / 62.8 | |

The fused LoRA step is 1.75-2.2x faster than the dispatch mode it replaces under SAC and uses a
fifth to a quarter of its memory. Against the fused full fine-tune it costs the merge and the
rank-r adapter GEMMs but skips the dW GEMMs: about +0.1-0.2 ms at 30B-A3B, where both are
bandwidth-bound, and 0.7-3.1 ms (MXFP8) or 2.2-9.9 ms (BF16) less at 235B-A22B and Beam. The merge
and forms columns are one layer's per-step rebuild on one B200 (MXFP8 forms; BF16 layouts 0.15 /
0.39 / 1.46 ms); under the `layer` cache a multi-layer model merges a layer twice per step
(forward and checkpoint recompute), like the full fine-tune rebuilds its forms twice.

## SM90 BF16 at the Beam shape (8xH100)

`precision: bf16` (the SM90 `auto` precision) against the blockwise FP8 path above, both from
one strict-H100 allocation (ap-wiPJGlhb3XTlfmuInQej7M, NVIDIA H100 80GB HBM3), one process per
row, fused path only, 16,384 tokens per GPU, max-rank median ms over 15 iterations after 5
warmups; peak allocated / reserved GiB per rank is the maximum over the four scopes:

| Precision | Routing | Forward | Backward | F+B (no checkpoint) | (c) checkpointed 2F+B | Peak GiB allocated / reserved |
|---|---|---:|---:|---:|---:|---:|
| SM90 BF16 | balanced | 34.73 | 110.91 | 145.72 | 179.50 | 51.7 / 57.7 |
| SM90 BF16 | skewed | 43.91 | 144.83 | 189.02 | 233.56 | 51.7 / 57.7 |
| SM90 blockwise FP8 | balanced | 15.71 | 54.25 | 69.07 | 85.79 | 49.2 / 55.1 |
| SM90 blockwise FP8 | skewed | 19.70 | 69.29 | 88.30 | 107.18 | 49.2 / 55.1 |

The BF16 forward combine at hidden 5120 runs in four chunks per token (one or two at hidden
2048 / 4096: the smallest chunk count whose three per-warp slots fit the reusable shared memory,
`impls/mono_ep_combine.cuh`). Blockwise FP8 stays the Hopper path for this shape: the BF16
kernels are 2.1-2.2x slower and hold 2.5 GiB more. In the same allocation the Qwen3-30B-A3B BF16
rows were timed on both builds of the kernels (balanced (c) 33.34 / 33.34 ms, skewed 43.00 /
42.97 ms, after / before), whose SASS is identical.

## Hopper backward GEMM roofline

The three backward GEMMs alone on one H100 (prepacked, balanced, blockwise FP8), ms:

| Shape | dh | z recompute | dx | Sum | TFLOPS |
|---|---:|---:|---:|---:|---:|
| 30B-A3B | 0.341 | 0.641 | 0.782 | 1.764 | 1169 |
| 235B-A22B | 1.193 | 2.306 | 2.645 | 6.144 | 1342 |
| Beam | 2.734 | 5.409 | 5.455 | 13.598 | 1263 |

## What these do not measure

- The H100 allocation had slow NVLink: against an earlier H100 allocation of the same commit
  the expert GEMMs were unchanged, the fused path was 2-5% slower, and DeepEP 36-54% slower.
  The H100 dispatch table therefore understates DeepEP on H100; the H200 table is the clean
  Hopper comparison.
- The B200 allocation ran about 3% slower than an earlier B200 gate run (NCCL control 24.89 ms
  against 24.24); compare B200 rows with each other, not across allocations.
- On the 80 GB H100 the 235B-A22B and Beam rows did not fit with every compared path
  resident, so each path ran in its own process in the same allocation; the Beam NCCL control
  did not fit even alone.
- H200 has no ring-floor rows: the re-benchmark plan ran the ring floor on H100 and B200 only.
- The Hopper fused rows are FP8 and the B200 rows BF16, and none of them is hardware- or
  precision-matched with MoK's MXFP8 rows. These are layer timings, not end-to-end training
  throughput, and only stationary routing was measured.

## Correctness

- Two-GPU FP32-reference forward/backward validators, the FP8 combine validator with forced
  ring wrap, and the forward FP32 reference on 2xH100; the in-kernel dW and MXFP8 validators
  on one and two B200s; the native backward on 2xB200.
- The FSDP2 weight-cache contract test with optimizer steps on `dp_shard 2 x EP 2`, 4xH100 and
  4xB200 (24 reserved SMs), and the NVFP4 FSDP2 contract on 4xB200 (rerun on the lean-tail
  default: three AdamW steps, cached merged weights equal a fresh requant every step).
- The NVFP4 LoRA backward on the lean-tail default against an FP32 reference on one B200: dx
  relative error 0.267 / 0.267 / 0.263 and dscore 0.246 / 0.245 / 0.245 (30B balanced, 30B
  skewed, Beam), unchanged from the previous default.
- The dispatch mode's gradients against the `torch` backend on 2xH100 (grouped_mm, ScatterMoE
  and SonicMoE, with and without LoRA, balanced and skewed): relative L2 0.0 after two
  optimizer steps.
- The SM90 kernels are byte-identical in SASS to the pre-integration build; the SM100 default
  backward differs only by the padded-tail zeroing before the dW GEMMs.
- The SM90 BF16 forward at the Beam shape against an FP32 reference on 2xH100 (64 experts, top-8,
  4,096 tokens per rank, balanced and skewed with rank 0's ring wrapping): relative L2 0.00186,
  next to 0.00172 / 0.00181 at the Qwen3-30B-A3B / 235B-A22B shapes in the same job; the two
  smaller shapes are bitwise equal to the build before the chunk rule and their SASS is identical.
- The Axolotl parity test `tests/e2e/multigpu/test_ep_mono_ep.py` (EP2 x FSDP2 against the
  `torch` backend with optimizer steps).
- Long-training convergence has only a loss-curve check: 2xH100, a 739M Qwen3-MoE, 300
  steps, two seeds, exact SwiGLU against `swiglu_fast` through the Axolotl seam.
