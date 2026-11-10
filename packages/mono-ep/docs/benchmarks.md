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

## 8xH200 against Mega MoE

Balanced routing, max-rank median ms, blockwise FP8, against the Mega MoE numbers Prime
Intellect published for prime-rl on the same hardware (ratio below 1 is faster):

| Shape | | mono-ep | Mega MoE (published) | ratio | NCCL control |
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

## Achieved TFLOPS

Routed TFLOPS per GPU, with the share of the GPU's dense peak in parentheses:

| Build | Hardware | Shape | Precision | Forward | Backward | Checkpointed 2F+B | Source |
|---|---|---|---|---:|---:|---:|---|
| mono-ep | 8xH200 | 30B-A3B | blockwise FP8 | 377 (19.1%) | 375 (18.9%) | 402 (20.3%) | ap-V06KM1a7K9KSAOsvOo4CU4 |
| mono-ep | 8xH200 | 235B-A22B | blockwise FP8 | 586 (29.6%) | 468 (23.7%) | 530 (26.8%) | ap-V06KM1a7K9KSAOsvOo4CU4 |
| mono-ep | 8xH100 | 30B-A3B | blockwise FP8 | 358 (18.1%) | 340 (17.2%) | 380 (19.2%) | ap-05NnV2Ph9NslFfBVYrnbLq |
| mono-ep | 8xB200 | 30B-A3B | BF16 | 590 (26.2%) | 593 (26.4%) | 646 (28.7%) | ap-afG15T7ayDA5cPEJnqpPcC |
| Mega MoE (published) | 8xH200 | 30B-A3B | blockwise FP8 | 317 (16.0%) | 302 (15.2%) | 311 (15.7%) | Prime Intellect |
| Mega MoE (published) | 8xH200 | 235B-A22B | blockwise FP8 | 521 (26.3%) | 402 (20.3%) | 453 (22.9%) | Prime Intellect |
| torch all-to-all + DeepGEMM FP8 (published baseline) | 8xH200 | 30B-A3B | blockwise FP8 | 103 (5.2%) | 161 (8.1%) | 126 (6.4%) | Prime Intellect |
| torch all-to-all + DeepGEMM FP8 (published baseline) | 8xH200 | 235B-A22B | blockwise FP8 | 239 (12.1%) | 324 (16.4%) | 275 (13.9%) | Prime Intellect |

Definition: routed FLOPs per GPU per direction = 2 x rows x 3 x hidden x intermediate, with
rows = tokens per rank x top-k (the gate/up GEMM is 2 x hidden x 2 x intermediate per row,
the down GEMM 2 x hidden x intermediate). A backward costs twice a forward (input and weight
gradients for each GEMM), and the checkpointed step is forward + recomputed forward +
backward, four forwards in all: 1.237 / 2.474 / 4.948 TFLOP per GPU at the Qwen3-30B-A3B
shape and four times that at Qwen3-235B-A22B. Achieved TFLOPS is routed FLOPs divided by the
measured max-rank median time of the whole layer step, so it counts only useful GEMM work
and charges dispatch, combine, quantization, the backward's gate/up recompute (the backward
kernel recomputes the gate/up GEMM per M-block, about two thirds of a forward that is not
counted) and weight-layout preparation to the denominator. Prime Intellect's published times
are converted the same way.

Dense peaks: H100 SXM and H200 SXM 1,979 TFLOPS FP8 (NVIDIA's datasheets list 3,958 TFLOPS
with 2:4 sparsity); B200 2,250 TFLOPS BF16 (NVIDIA's HGX B200 specification lists 36 PFLOPS
BF16 with sparsity per eight GPUs, 4.5 per GPU, half of that dense). Hopper rows compare
against the FP8 peak, the B200 row against the BF16 peak.

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

## Baselines: the Axolotl backends on the same contract

The same layer shape, routing generators and timing method (5 warmups, 15 samples, median
of the max rank, checkpointed 2F+B), run through Axolotl's expert-parallel backends with
SonicMoE or `grouped_mm` as the local experts kernel. Expert weights are BF16 (SonicMoE's
native dtype) and every variant of a column ran interleaved in one allocation
(`examples/dispatch_sonicmoe_contract.py --deep-ep` on the `features/deepep-contract-baseline`
branch; 8xH100 ap-ML4Jf7M3m03WlwFVGclzL6, 8xB200 ap-1frtFOq9kwibpsjjWsMTtR; artifacts
`dispatch_sonicmoe_contract_deep_ep_h100.json` and `dispatch_sonicmoe_contract_deep_ep_b200.json`). DeepEP is v1.2.1 with
the plugin's 256 MiB NVLink buffer; "as shipped" is the `deep_ep` backend with its
`barrier_ep` before every combine and DeepEP's default 20 SMs; "kernels only" drops the
barrier; the swept rows give DeepEP the best SM count found (the H100 sweep stopped at 40
SMs, the B200 sweep was still improving at 64). Balanced / skewed, ms:

| Variant | 8xH100 | 8xB200 |
|---|---|---|
| torch all-to-all + SonicMoE | 20.88 / 23.76 | 14.35 / 15.33 |
| DeepEP + SonicMoE (20 SMs, Axolotl default), as shipped | 21.48 / 25.68 | 22.13 / 25.24 |
| DeepEP + SonicMoE (20 SMs, Axolotl default), kernels only | 21.09 / 25.30 | 21.75 / 24.82 |
| DeepEP + SonicMoE (SM-tuned: 40 H100 / 64 B200) | 17.56 / 20.73 | 12.80 / 14.00 |
| torch all-to-all + grouped_mm | 129.68 / 149.80 | 89.74 / 103.55 |
| DeepEP + grouped_mm, 20 SMs | 123.97 / 144.25 | 95.44 / 110.31 |
| mono-ep fused, pre-tuning build, same job | 18.55 / 22.64 | 10.89 / 13.50 |

The last row is the fused kernel at the state before the Hopper and B200 tuning rounds,
measured inside the same job as a reference point; this tree's defaults measured 12.998 /
16.198 (H100) and 7.659 (B200) in their own allocations (tables above). SonicMoE's GEMM time
is identical in every SonicMoE row (H100 5.4 ms, B200 2.7-2.8 ms per forward+backward):
nothing overlaps in that serialized path, so DeepEP's SM count only bounds its own dispatch
and combine kernels. An earlier job with full quack sweeps (8xH100 ap-DPDFL3sbYfPqkGAh5dNWqe,
8xB200 ap-R8iuf9H6IVXAWgpMWmfv9P) put the torch all-to-all + SonicMoE control at 22.68 /
14.01 ms balanced. Mixture of Kittens (MoK) has not been run on this contract.

## Dispatch mode on the same contract

The dispatch mode serves the `deep_ep` / `torch` dispatch/combine contract from mono-ep's peer
transfers, with SonicMoE as the local kernel and BF16 expert weights, on the same layer,
routing and timing method as the baselines above; every row of a column ran interleaved in
one allocation. Checkpointed 2F+B, max-rank median ms, balanced / skewed
(`compose-design.md` 5.7; 8xH100 ap-u57jzYtyxSFbY5pHpZn9xj, 8xB200 ap-O0F3PmsfwBnZTjWsCzB25T):

| Variant | 8xH100 | 8xB200 |
|---|---|---|
| mono-ep dispatch, bounded compact receive (`expert_parallel_mono_ep_bounded`) | 18.72 / 21.36 | 11.91 / 12.60 |
| mono-ep dispatch, compact receive (count stall) | 19.22 / 22.23 | 12.50 / 13.51 |
| DeepEP + SonicMoE, 40 SMs | 17.57 / 20.40 | 14.89 / 16.86 |
| DeepEP + SonicMoE, 64 SMs | 17.24 / 20.02 | 12.94 / 14.09 |
| torch all-to-all + SonicMoE | 20.85 / 23.53 | 14.51 / 15.31 |

On B200 the bounded receive leads DeepEP at its best measured SM count; on H100 DeepEP keeps
a lead of about 1.4 ms.

## Mixture of Kittens (separate allocation)

cursor/mixture-of-kittens at f3a8bd2432 built for SM100 on the same B200 image (one-line
`kittens::detail::tcgen05::commit` qualification), best knobs from a sweep, **shared expert
included in its timing**, in a **separate allocation** from the mono-ep rows (Modal
ap-tnRPLy5qgP47OeeadOfmRO). Not a same-run A/B. F / B / checkpointed 2F+B ms:

| Shape | MoK BF16 | MoK MXFP8 | mono-ep (this tree) |
|---|---|---|---|
| Qwen3-30B-A3B | 2.27 / 3.50 / 7.01 | 2.20 / 2.79 / 6.32 | BF16 2.10 / 4.17 / 7.66 |
| Qwen3-235B-A22B | 6.34 / 11.06 / 21.22 | 5.46 / 7.90 / 16.63 | |
| Beam (256 experts) | 11.77 / 21.73 / 40.78 | 9.19 / 15.31 / 30.41 | |

On one B200 at the 30B shape (BF16, F / B) MoK takes 2.80 / 5.06 ms against mono-ep's 1.27 /
3.26. At EP8 mono-ep's forward is faster and MoK's backward is about 0.5 ms faster, which
matches mono-ep's serial weight-gradient phase (0.93 ms).

## SM100 opt-ins on 8xB200

MXFP8 backward (`MONO_EP_SM100_BACKWARD_PRECISION=mxfp8`), BF16 and MXFP8 interleaved per
iteration in one allocation (ap-vNYQUgp8WL4TzRvFuKprVS), weights bumped every step, ms:

| Qwen3-30B-A3B | BF16 | MXFP8 |
|---|---|---|
| backward, balanced | 4.164 | 3.794 (-8.9%) |
| checkpointed 2F+B, balanced | 7.714 | 7.236 (-6.2%) |
| checkpointed 2F+B, skewed | 9.334 | 8.760 (-6.1%) |

In-kernel weight gradients (`MONO_EP_SM100_INKERNEL_DW`), off / on / wide, one allocation
(ap-HXzwyX5kvITeMHznIApH69), backward and checkpointed 2F+B ms:

| Shape | off | on (256 x 128) | wide (256 x 256) |
|---|---|---|---|
| Qwen3-30B-A3B, balanced | 4.164 / 7.665 | 4.273 / 7.798 | 4.243 / 7.757 |
| H 5120 / I 2560, 32 local experts | 21.914 / 32.208 | 21.254 / 31.903 | 20.625 / 30.860 |

## NVFP4 LoRA (SM100)

`precision="nvfp4_lora"` (frozen NVFP4 experts with LoRA), 8xB200, F / B / checkpointed 2F+B
ms with the ring floor and the CUDA requant (ap-dJj9XYEA3BDSZ7TDQv3bzS), and the per-step
merged-weight rebuild per layer:

| Case | nvfp4_lora | rebuild, Triton / `cuda` / `cuda_fused` |
|---|---|---|
| Qwen3-30B-A3B, balanced | 2.11 / 4.03 / 7.98 | 0.87 / 0.35 / 0.31 |
| Qwen3-30B-A3B, skewed | 2.74 / 6.31 / 11.53 | |
| Beam, balanced | 2.98 / 8.11 / 13.68 | 10.24 / 3.24 / 2.50 |
| Beam, skewed | 3.64 / 10.35 / 17.44 | |

The ring floor took the Beam backward from 11.35 to about 8 ms; with the `cuda` rebuild a Beam
optimizer step without gradient accumulation goes from 23.9 to 16.9 ms per layer.

GEMM floor: the cuDNN frontend grouped NVFP4 GEMM on one B200 runs the backward's GEMMs in
0.568 ms (30B) and 1.551 ms (Beam) against 2.11 and 4.44 ms for the whole persistent backward
kernel, and its k-loops are 2.2-2.4x faster than the persistent kernel's. As an EP8 layer with an
all-to-all around it (`cudnn_grouped_a2a`, ap-zZAErn1BHT6XxTyVS0PVIs) it takes 27.82 ms (30B)
and 36.35 ms (Beam) checkpointed 2F+B against nvfp4_lora's 7.93 and 13.08 in the same run.

## What these do not measure

The H100 and B200 rows are not hardware- or precision-matched with the Mega MoE H200 FP8
figures; the controls share the persistent forward, so they are not Prime Intellect's
published baseline (torch all-to-all + DeepGEMM FP8: 39.3 ms at 30B-A3B, 71.9 ms at
235B-A22B); absolute timings move by up to 16% between allocations of the same GPU type, so
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
