# MonoEP implementation roadmap

Last updated: 2026-11-09.

This is the persistent roadmap approved by the user. Update task status and link
validation artifacts as work progresses. A successful primitive test does not
complete a whole-layer or full-model milestone. Recheck live jobs and current code
before relying on this snapshot.

## Objective and constraints

Build an open-source equivalent of Prime's MoE training-kernel capabilities,
combining expert-compute fusion and communication overlap. Use DeepGEMM and
SonicMoE where appropriate; evaluate license-compatible UniEP transport behind a
coherent interface. The package and import namespace are `mono-ep` and
`mono_ep`.

- After fusion and communication overlap stabilize, prioritize merge-aware NVFP4
  LoRA (frozen NVFP4 expert weights, adapters at full alpha), then NVFP4 full
  fine-tuning, then BF16 weights with FP8 AMP and LoRA (reordered 2026-11-09).
- Support BF16, Hopper blockwise FP8, and SM100 MXFP8; track master-weight dtype
  separately from compute dtype. Verify additional claimed formats before adding
  them to the support matrix.
- Target whole-layer `torch.compile`, not merely compiled local callbacks.
- Keep routing IDs and weights from forward when recomputing activations; never
  reroute during backward and risk reduced-precision expert flips.
- Respect Axolotl activation checkpointing, selective activation checkpointing,
  and offloading configuration.
- Use the HF Hub SonicMoE loading path and Axolotl compatibility setup for the
  BF16 baseline. Keep experimental source-pinned FP8 Sonic work explicit.
- Keep backend, lane, and transport choices internal where practical; avoid a
  proliferation of public configuration options.
- Validate local SM120 capability separately from Hopper and SM100. Do not assume
  consumer Blackwell supports the SM100 implementation.
- Preserve upstream licenses and notices; appropriate Axolotl sources carry
  SPDX headers and Copyright (c) 2026 Axolotl AI.
- Modal GPU runs are authorized. PR creation still requires human review of the
  complete diff under the repository contribution policy.

## Feedback assessment and priority update (2026-11-09)

The user requested evaluating the external redirect brief, not blindly adopting
its stop/delete instructions, and explicitly required preserving all progress.
Existing implementations, tests, results and provenance remain available. A local
source/results snapshot is `/private/tmp/mono-ep-progress-2026-11-09.tar.gz`
(excludes build and cache directories); it supplements this worktree, not a remote
backup. Do not delete experimental paths or evidence as a consequence of the brief.

Verified against prime-rl's public MoE layer (`src/prime_rl/trainer/models/layers/`): autograd saves
only input, routing and weights; backward receives no forward z/h; precision modes
include BF16, MXFP8, blockwise FP8 and BF16-forward/FP8-backward. Its wrapper reuses
buffers across layers and reserves compute resources for concurrent NCCL. These
are stronger design requirements than inferring implementation from blog timings.

Static source-rank/producer-consumer SM partition tuning is parked. Implement the
fused backward with GPU-resident dynamic M-block task acquisition and per-CTA
warp-specialized dispatch, math and combine. A CTA/pair owns the block through
phase A (column-tiled dH + recompute + SwiGLU) and phase B (dX + combine); phase
N-tile loops are internal, not separate global scheduling tasks. Tune phase-specific warp roles and
ring depth only after this pipeline exists and passes skew/lifetime validation.
Keep the NCCL coexistence budget separate from MoE task scheduling.

Near-term priorities now emphasize a recomputing persistent BF16 backward on B200,
then MXFP8, with Hopper persistent forward/backward as the architectural port.
Close currently running correctness work and retain its evidence; do not start a
new series of retention/peer-lane microbenchmarks. Retention remains an optional
validated memory/performance tradeoff, not the parity baseline. The actual saved-z
allocation at Qwen geometry is ~3.4 GB per rank per outstanding layer, so a
single-layer retained result does not establish whole-model viability.

The current public-op boundary/shared-workspace/resource-budget work is useful,
but wrapping NCCL split `.cpu().tolist()` inside a custom op only hides host sync
from Dynamo. Fullgraph tracing, CUDA graph capture and truly GPU-resident execution
must have separate acceptance evidence. A registry key must also preserve safe
buffer lifetime and shared-workspace ordering across layers/streams.

Do not assume the suggested persistent backward algorithm is already correct:
SwiGLUBackward currently consumes saved z with dH GEMM accumulators, so attaching
it directly to a recomputed z GEMM needs a new epilogue contract. Local dW GEMMs
outside the persistent dispatch/dH/z/dX/combine kernel are an explicit intermediate
milestone, not completion of full backward fusion. Recompute cost and resulting
2F+B must be measured; a fast forward alone does not make backward recompute free.

Retain the user's full scope: FFT and LoRA, checkpoint/offload compatibility,
SonicMoE exploration, expanded dtypes, whole-layer compile, Hopper and SM120,
and license-compatible multi-node transport. The brief does not revoke these.
Existing milestones below remain historical requirements; this section sets the
near-term emphasis without discarding prior work.

## SM120 (RTX PRO 6000 / RTX 50-series Blackwell): what carries over

Analysis only; nothing below has been measured on SM120 yet. SM120 has neither
SM100's tcgen05/TMEM (UMMA, 2-CTA pairs) nor SM90's WGMMA: tensor cores are driven
by warp-level `mma.sync` (including block-scaled FP8/FP4 variants) with register
accumulators. Shared memory per block is far smaller than H100/B200, device memory
is GDDR7 (well under half of H200's HBM bandwidth), and multi-GPU boxes are
PCIe-only with platform-dependent peer access. Today `unsupported_reason` rejects
SM120 and Axolotl falls back to `torch` dispatch with the configured local kernel
(SonicMoE/ScatterMoE/grouped_mm).

Carries over (architecture-independent):

- The op contract: two custom ops with fake impls and an int key, nothing retained
  from the forward, restage-and-recompute backward, one workspace per process,
  routing saved under SAC. Recompute is relatively cheaper where bandwidth is the
  scarce resource, so not retaining z/h matters more on GDDR7, not less.
- GPU-resident, fixed-capacity expert layouts (expert starts/padded counts on
  device, no `.item()`/`.tolist()`), the K-grouped weight-gradient formulation, FP32
  combine accumulation, and capacity/padding sentinels.
- Fused epilogues: SwiGLU backward (dz, gate/up gradients, score partials),
  weighted activation and recompute-from-accumulators. The SM90 port already uses
  register accumulators, so its epilogues are the closest template; only the
  fragment layout changes from WGMMA to `mma.sync`.
- Quantize-on-pack (FP8 x/dy/dz written once while packing) and per-call weight
  quantization. Byte reduction is worth more on GDDR7/PCIe. Block-scaled
  `mma.sync` consumes 32-element E8M0 scales natively, so the MXFP8 recipe maps
  more directly than Hopper's 1x128/128x128 blockwise FP8 with FP32 promotion.

Does not carry over:

- In-kernel dispatch/combine over NVLink symmetric memory. Over PCIe, remote stores
  and atomics are slow and peer access may be disabled; keep NCCL all-to-all with
  chunked overlap (the `torch` backend's `expert_parallel_dispatch_chunks`). SM
  reservation and `NCCL_MAX_CTAS` then stop being correctness requirements.
- tcgen05/TMEM mainloops, 2-CTA scheduling, WGMMA async pipelines, and ring depths
  sized for 228 KB of shared memory (fewer stages, smaller tiles).
- Register-redistribution warp specialization tuned for 512-thread SM90 CTAs;
  re-derive role counts for `mma.sync` occupancy.

Next steps, in order, on the local 2x RTX PRO 6000 host before any kernel work:
check peer access and topology; profile `torch` + SonicMoE/grouped_mm at the Qwen
shape to find whether GEMMs, memory traffic or PCIe dominate; then decide between
(a) a fused `mma.sync` grouped GEMM carrying the SwiGLU/recompute and
quantize-on-pack epilogues behind the same two ops with NCCL transport, and
(b) MXFP8/NVFP4 local GEMMs only. Report results per section 6 of the redirect
brief's reporting rules (hardware, geometry, dtypes, scope, samples).

## Current implementation boundary

### Follow-up integration brief assessment (2026-11-09)

Read the external Axolotl composition follow-up. Use `_MonoEPBackend`
(user's requested MonoEP naming), `mono_ep` as the backend value, and
MonoEP names for every public runtime API, config key and operator namespace.
The project is not called after Prime Intellect's kernels; cite their published
numbers as "Prime Intellect (prime-rl)". The vendored DeepGEMM fused-MoE file
and API names carry the `mono_ep` prefix as well; `vendor/sources.json` and
`tools/rename_vendor_names.py` record how to re-apply that when the pin moves.

The proposed fused backend seam is the intended integration: it bypasses local
dispatch/compute/combine while torch/deep_ep retain their independent SonicMoE
and ScatterMoE choices. Evaluate shared workspace lifecycle, DTensor local
weights/gradient scaling, model capability checks, explicit fallback and whole
layer compile at that seam. The brief's precision table is a target, not proof
that the current Hopper or FP8 paths are ready. Verify checkpoint policy against
router stability and actual SAC/offload behavior before registering save rules.

The distributed-environment hook merged in PR #4284 (commit
`f914504565692d24f22e2badfec31823af142770`), verified via GitHub on 2026-11-09.
The merged patch is now applied locally (without a history merge). Consume
`get_distributed_env(cfg)`; do not duplicate the hook or silently leave required
NCCL resource constraints unenforced. Preserve existing adapters, pipelines and
evidence until replacements pass parity checks. Host synchronization removal via
compact GPU-native communication remains the immediate priority; this brief
does not authorize its stop/delete list or supersede that priority.

The common entry point selects BF16 Hub SonicMoE or FP8 DeepGEMM with the shared
NCCL chunk scheduler. The native readiness/peer scheduler is experimental and
single-host SM100. UniEP has a separate H100 prototype, not an integrated transport.
Local expert callbacks compile; the full EP layer does not yet compile end to end.
Small training and checkpoint tests do not establish full-model convergence.

Completed building blocks include peer readiness/publication, packed dispatch,
down-projection/output scatter fusion, input-gradient/scatter fusion, backward
SwiGLU epilogues, GPU barriers, retained routing, and outstanding-forward/reverse-
backward reuse checks. BF16 forward projection/SwiGLU fusion is numerically tested
but is not the default because it has not established a whole-layer speedup.

## Ordered milestones

### 1. Make BF16 fusion improve whole-layer latency

- [ ] Isolate weight interleaving, initialization, launch gaps, and synchronization
  costs with controlled A/B experiments.
- [x] Remove unused full-sized FP8 scale buffers from BF16 scheduling.
- [x] Implement an uninitialized activation-scratch variant and validate inactive
  rows using NaN poisoning, delayed publication, empty experts, and compiled calls.
- [x] Complete the same-allocation zeroed/uninitialized scratch comparison.
- [ ] Reduce remaining preparation and temporary-materialization costs.
- [ ] Demonstrate a repeatable fused-forward improvement over separate activation
  at the complete retained forward/backward layer boundary.

### 2. Beat the single-node SonicMoE + NCCL baseline

- [ ] Select compute lanes and SM allocation by workload without new public knobs.
- [ ] Reduce per-source launches and serialization; handle skewed and empty routes.
- [ ] Pipeline dispatch, compute, and return at a granularity that preserves GEMM
  efficiency; remove unnecessary phase barriers and host launch bubbles.
- [ ] Bound workspace, reduce capacity-wide clearing and saved-buffer copies, and
  retain safe epoch/lifetime management for concurrent invocations.
- [ ] Establish consistent whole-layer gains with matched dtype, hardware,
  routing, retention policy, and baseline compatibility setup.

### 3. Complete backward fusion and overlap

- [ ] Reduce routing-score reduction/publication launches and intermediate storage.
- [ ] Overlap weight-gradient preparation/GEMMs with gradient communication.
- [ ] Pipeline dH/SwiGLU, dX/scatter, and weight gradients without rerouting.
- [ ] Validate the resulting complete backward under stream stress and optimizer
  updates, including multiple outstanding forwards.

### 4. Extend the winning schedule to FP8 and complete capture

- [ ] Fuse peer-path activation/FP8 quantization and reduce intermediate writes.
- [ ] Pipeline W1 → SwiGLU → W2 across kernel boundaries or persistent execution
  without full-capacity intermediate materialization where feasible.
- [ ] Remove host count/split dependencies from the full EP compilation path.
- [ ] Validate whole-layer `torch.compile` and CUDA graph capture with changing
  routing, empty experts, retained activations, and recomputation.
- [ ] Verify BF16/FP32 master weights independently of supported compute formats.

### 5. Integrate transport, hardware coverage, and full-model FFT

- [ ] Integrate a multi-node transport behind the common runtime interface;
  evaluate UniEP/NVSHMEM/other applicable transports with license provenance.
  - [ ] NVSHMEM multi-node EP (backlog, 2026-11-09): keep the single-node NVLink
    symmetric-memory path as is and add an NVSHMEM transport for the dispatch
    pulls, combine stores and readiness flags, so one kernel design spans a node
    (NVLink) and IB/EFA across nodes (IBGDA). Driver: 256-expert models such as
    Beam-1.0-501B on 2x8 B200 need EP beyond one node. Reference design:
    FlashMoE (osayamenja/FlashMoE, arXiv 2506.04667, BSD-3) builds its global
    address space on NVSHMEM with one-sided device-initiated puts plus signals;
    it is forward-only and TF32/BF16 mma.sync, so only the transport and
    signalling pattern carry over, not the kernels. Gates before starting:
    license check for NVSHMEM linkage (TeraMoE's MIT grant excepts
    NVSHMEM-related code), symmetric-heap sizing per rank, a CoreWeave IB
    validation plan, and a full-step multi-node measurement rather than layer
    timing.
- [ ] Validate cross-node readiness, completion, buffer reuse, and backward overlap.
- [ ] Port the relevant peer/fusion path to Hopper; implement an appropriate SM120
  path or explicit capability-based fallback.
- [ ] Integrate the winning peer engine into Axolotl's common EP entry point.
- [ ] Complete AC/SAC/offload coverage, including full HF/FSDP combinations.
- [ ] Validate merge-aware NVFP4 LoRA first (frozen NVFP4 expert weights
  consumed by the fused kernels, in-smem dequant to the compute format where the
  MMA kind needs it, adapters at full alpha; reference = the dense merge-aware
  NVFP4 LoRA path), then NVFP4 full fine-tuning; scope other quantized adapters
  explicitly rather than implying support from kernel dtype coverage.
- [ ] Validate BF16 weights + FP8 AMP + LoRA after that, including adapter/input
  gradients, optimizer behavior, checkpoint/resume, memory, and throughput.

## Benchmark and release gates

- [ ] Repeat a matched single-node eight-H200 Qwen3-30B-A3B comparison against
  Prime's reported setup, including retained activations and actual checkpointing.
- [ ] Keep manual 2F+B, actual checkpoint-engine execution, retained F+B, layer
  latency, and full training-step throughput distinct.
- [ ] Validate multi-node full-step performance; do not infer it from layer timings.
- [ ] Cover skew, masked/empty routes, varying token counts, concurrency, and large
  offsets on the appropriate hardware CI lanes.
- [ ] Pin reproducible dependency/kernel revisions, audit licenses and packaging,
  document the support matrix, and obtain human diff review before opening a PR.

Always accompany timing reports with model reference sizing, GPU count/type,
EP degree, tokens per GPU, hidden/intermediate widths, global/local expert counts,
top-k, compute/master dtypes, activation policy, and measured scope. The current
Qwen3-30B-A3B synthetic layer uses 8 B200s, EP8, T=16384 per GPU, H=2048, I=768,
128 experts (16/rank), top-k 8, BF16 compute and FP32 master weights. It excludes
router logits/top-k, attention, optimizer updates, and full-model execution.

Report median maximum-rank latency across measured iterations. Kernel durations
may overlap and must not be summed into a claimed layer runtime. Keep controls in
the same allocation when attributing performance changes to an implementation.

## Latest evidence and next action

- `benchmarks/b200_qwen3_30b_ep8_bf16_compact_scales.json`: eight-B200 retained F+B,
  SonicMoE + NCCL **34.53 ms**, separate peer **39.40 ms**, fused peer **39.88 ms**.
  This establishes the remaining gap, not equivalence to Prime's H200 numbers.
- `benchmarks/bf16_uninitialized_activation_tests.json`: **305 B200 tests passed**,
  including NaN-filled activation scratch followed by bounded down-projection.
- `benchmarks/b200_qwen3_30b_ep8_bf16_uninitialized_activation.json`: eight-rank
  controlled retained F+B medians were **34.46 ms NCCL**, **39.15 ms separate**,
  **39.65 ms fused zeroed**, and **38.77 ms fused uninitialized**. The latter
  improves on zeroing by 2.2% and separate activation by 1.0% in this run, but
  remains 12.5% slower than NCCL. Maximum fused/separate relative L2 was 5.318e-5.
  Modal app `ap-BKwaw1xqE7Va9ygg6jdhUd` completed successfully; all eight rank
  records and maximum-rank samples are saved. This is one controlled allocation,
  not proof of repeatability across workloads.

- BF16 saved-preactivation clearing is now removed separately. The derivative
  tests poison unpublished rows before backward; **305 B200 tests passed** in
  `benchmarks/bf16_uninitialized_z_tests.json`. Four-rank compiled optimizer and
  outstanding-forward/reverse-backward checks passed through epoch 42 in
  `benchmarks/bf16_uninitialized_z_training.json`. This eliminates 3.046875 GiB
  of clearing writes per rank at Qwen capacity, not retained memory allocation.
  Its controlled latency impact is recorded below.

The benchmark performs an untimed parity forward/backward per variant, then
three discarded warmup iterations and ten measured iterations. CUDA is
synchronized around the timed host forward/backward region; variant order
alternates. Future raw records explicitly include warmup/sample counts and scope.
This is a fixed warmup policy, not an adaptive steady-state detector.

- `benchmarks/b200_qwen3_30b_ep8_bf16_uninitialized_z.json`: the completed
  same-allocation comparison measured **34.41 ms NCCL**, **39.01 → 37.27 ms**
  separate peer and **39.33 → 36.84 ms** fused peer when removing preactivation
  clearing (4.5% and 6.3% reductions). The fully uninitialized fused path is
  still 7.1% slower than NCCL. All eight ranks passed parity, max relative L2
  5.318e-5. The timing run did not include the subsequent backward scratch change.

The first uninitialized-backward suite failed three standalone derivative cases
with empty experts (`benchmarks/bf16_uninitialized_backward_tests.json`), despite
passing four-rank training. Standalone dX uses an unmasked grouped GEMM, so dZ
initialization has been restored for that path; peer scatter masks invalid
destinations and retains uninitialized dZ. Backward h remains uninitialized.

The corrected kernel suite passed **305 tests** in
`benchmarks/bf16_uninitialized_backward_scatter_tests.json`. The four-rank peer
optimizer/reverse-backward stress passed through epoch 42 in
`benchmarks/bf16_uninitialized_backward_training.json`; restoring initialization
for standalone dX does not alter that peer path. All jobs in this handoff are
terminal. Routing-score partials remain zeroed because their reduction covers
full capacity.

- `benchmarks/b200_qwen3_30b_ep8_bf16_batched_timing.json`: instrumentation
  sensitivity run completed. Isolated/batched ms per call were **36.13/36.67
  NCCL**, **37.76/39.63 separate peer**, and **37.55/39.32 fused peer**. The
  fused/separate gain is only 0.5–0.8%; do not promote it on this evidence.
  Batched mode includes inter-call gradient resets and runs after isolated mode;
  these results do not isolate a pure synchronization effect. This run includes
  backward scratch removal but predates fused score reduction/scatter.

Fused BF16 score reduction/scatter passed **323 B200 tests** in
`benchmarks/bf16_score_reduce_scatter_tests.json`, including compiled/eager
publication, duplicate score slots, empty/masked sources, NaN inactive rows,
delayed concurrent producers, and repeated queue reuse. Four-B200 optimizer/reuse
checks passed through epoch 42 in `benchmarks/bf16_score_reduce_scatter_training.json`.
The peer backward avoids a separate score-vector reduction/materialization and
masks inactive partial rows before reading, allowing uninitialized partial scratch.
Standalone backward retains zeroed partials and its existing score output.

Next action: collect the fused/separate score-publication comparison dispatched
with `--qwen-bf16-forward --timing-batch-size 10`. Local session `78418`; log
`/private/tmp/ultra-bf16-score-fusion-timing.log`. Recheck its live state before
acting. The harness now includes `peer_fused_forward_separate_scores` as a
same-allocation control with zeroed partials and separate reduction/scatter. Keep separate
forward activation as the default; report shape and timing mode with every
performance claim. Score-fusion correctness is established for the tested cases;
its latency benefit is not measured yet.


FP8 follow-up in progress: the SM100 packed FP8 peer backward now calls the same
FP32 reduction/scatter publication operator. Only score partial initialization
and the intermediate score vector are removed; FP8 h/dZ initialization remains
because quantization consumes padded storage. Four-B200 validation was dispatched
as `ap-yYbSoZSxbwCR8lysjKaLRP`, local session `45987`, log
`/private/tmp/ultra-fp8-score-reduce-scatter-training.log`. It was confirmed live
alongside the BF16 timing session at this update. Collect both existing jobs;
do not relaunch based solely on an observation timeout. No FP8 speedup is claimed.


## Priority update: incremental readiness and persistent scheduling

User direction: stop buffer-clear tuning and prioritize incremental expert
readiness, persistent dispatch → W1/SwiGLU → W2 → return, then backward overlap.
The eventual comparison target remains matched H200 FP8 checkpointed 2F+B.

The final score-fusion comparison completed in
`benchmarks/b200_qwen3_30b_ep8_bf16_score_fusion.json`: fused/separate score
publication measured 37.75/37.40 ms isolated and 39.29/38.84 ms batched on the
same eight-B200 Qwen-sized BF16 retained layer. No speedup was established;
separate score publication remains/restores the default. FP8 correctness passed
in `benchmarks/fp8_score_reduce_scatter_training.json`, but its fused score path
is not selected pending performance evidence. Do not spend further turns tuning
these buffer/publication micro-optimizations.

The native BF16/FP8 grouped kernels previously acquired every ready descriptor
before any tiles. An opt-in `incremental` native argument now selects a persistent
expert-task tile scheduler: each descriptor is acquired when needed, empty tasks
are skipped, and cluster-aligned tile pairs remain together. It admits descriptors
in queue order; it does not yet schedule arbitrary ready tasks out of order or
fuse the two projections into one kernel. Existing all-ready scheduling remains
the default. BF16 custom-op forward/backward boundaries expose the opt-in flag.

Next action: inspect native build/validation app `ap-Js83iZKViVAqRSxeUzBdou`,
local session `24822`, log `/private/tmp/ultra-incremental-ready-tests.log`.
The suite adds eager/compiled BF16 forward/backward numerical coverage for
incremental scheduling, empty tasks, and delayed publication. Completion and
performance are unproven. After numerical validation, add an explicit staggered-
expert progress check to prove early computation before all descriptors publish,
then integrate and measure the full layer. Do not infer full persistent-pipeline
completion from this first scheduler primitive.


Incremental-readiness validation follow-up: `test_ready_bf16_staggered_progress`
now compares opt-in and legacy admission in eager/compiled modes. A bounded GPU
observer checks for the first expert's output before writing/publishing the
second expert's payload; legacy admission should make no early progress, while
incremental admission should. It repeats queue reuse three times and checks final
outputs exactly. This is an observable-progress probe, not a latency benchmark.
The test was added after the currently running Modal invocation began building;
inspect its collected test count before assuming the new test ran. Native build
session `24822` was rechecked and remained live. No scheduler result is available
yet. Once the existing invocation is terminal, run any newly added progress cases
not covered by it, then integrate incremental scheduling into the peer runtime.


Incremental scheduler validation completed: native build and **347 B200 tests**
passed in `benchmarks/incremental_ready_tests.json`. The subsequently added
**four staggered-progress cases also passed** in
`benchmarks/incremental_ready_progress_tests.json`. In eager and compiled runs,
first-expert output was observed before second-expert publication only with
incremental scheduling; both paths produced exact final outputs over three reuses.
All runs listed in this handoff are terminal.

Next action: thread incremental admission through complete peer forward/backward
callbacks (including fused forward), validate distributed reuse/optimizer behavior,
and cover the FP8 opt-in scheduler. Then measure complete MoE-layer latency under
controlled dispatch skew and the Qwen workload. The larger remaining task is to
pipeline W1/SwiGLU/W2/return within persistent GPU scheduling; this primitive
removes the wait-for-all preamble but still admits experts in queue order and
does not yet overlap the two projections. Do not return to buffer-clear tuning.


Incremental BF16 peer integration: `PeerSchedule(incremental_ready=True)` now
binds the native admission flag into both forward and backward callbacks before
local compilation. The fused-forward wrapper and benchmark callback also forward
the flag; FP8 runtime opt-in is rejected until its path is wired and validated.
`benchmarks/incremental_ready_peer_training.json` records a successful four-B200
compiled run with one compute lane, two experts per rank, BF16, slot combine,
skipped padding, delayed dispatch, poisoned buffers, three AdamW steps, and three
outstanding forwards followed by reverse-order backwards on alternating streams.
All four ranks reached barrier epoch 42. This establishes distributed correctness,
not a speedup or full-layer graph capture. Fused-forward eager/compiled incremental and legacy admission tests completed
successfully: 24 passed, 339 deselected, saved in
`benchmarks/incremental_ready_fused_forward_tests.json`. These cover empty expert
groups, delayed publication and reuse at inner widths 128 and 768. Both Modal
runs are terminal. Next: fused-forward
end-to-end optimizer validation, FP8 admission, then full-layer timing and the
persistent two-projection pipeline.


Incremental FP8 and fused BF16 training validation completed. Both four-B200
compiled runs passed three AdamW steps and the alternating-stream / three
outstanding-forward reuse checks, all ranks reaching epoch 42. Artifacts:
`benchmarks/incremental_ready_fused_training.json` and
`benchmarks/incremental_ready_fp8_training.json`. FP8 now forwards the opt-in
admission flag through its ready forward custom op and backward SwiGLU epilogue;
its numerical tolerances remain the existing FP8 tolerances. This does not prove
FP8 early-progress behavior independently of the BF16 observer test.

Full Qwen BF16 timing A/B is running in session 19817, log
`/private/tmp/ultra-qwen-incremental-timing.log`. Command: `modal run
packages/mono-ep/examples/modal_qwen_peer.py --compute-dtype bf16
--qwen-bf16-forward --qwen-incremental --timing-batch-size 10`. This compares
SonicMoE+NCCL with separate/fused forward, each with legacy/incremental admission,
without buffer-clear variants. Single-node 8 B200, EP8, T16384/rank, H2048, I768,
E128, top-k8; retained activations; full MoE F+B excluding router and optimizer.
Three warmup trials and ten measured trials, both isolated and batch10 modes.
Inspect this live handle before starting another timing job, persist terminal
results, and report median maximum-rank samples with scope. Next architecture
work remains the persistent W1/SwiGLU/W2/return pipeline and backward overlap;
queue-order readiness alone does not implement those.


B200 incremental timing completed successfully, all eight ranks recovered in
`benchmarks/b200_qwen3_30b_ep8_incremental_timing.json`. Median maximum-rank full
retained BF16 MoE F+B, isolated / batch10 per iteration (ms): NCCL 35.266 / 36.195;
separate legacy 37.291 / 38.606; separate incremental 36.944 / 38.327; fused legacy
37.206 / 38.329; fused incremental 37.158 / 38.321. Same Qwen geometry and warmup
method above. Incremental admission is correct but adds essentially no measured
fused-path speedup here. Do not continue tuning this preamble.

User priority: primary agent owns B200 full end-to-end implementation. Hopper
port proceeds independently with Sol subagent `/root/hopper_port`; Astra reviewer
`/root/hopper_review` reviews its native code. Hopper H100 validation session87857
is owned by Sol; do not restart or duplicate it. Hopper changes are experimental
and must not relax the B200 peer runtime architecture guard.

B200 next implementation direction: the existing vendor persistent forward already
pipelines dispatch/W1/SwiGLU/W2/combine, but `runtime.py` backward recomputes W1
and uses host-count NCCL exchanges. The experimental retained peer runtime has
correct backward but separate projection launches. Close this gap deliberately:
inspect persistent-forward ring-buffer lifetime and add training activation /
routing retention suitable for outstanding forwards, then connect GPU-driven
backward and weight gradients. Avoid presenting the existing recomputing wrapper
as a complete retained persistent training implementation. In parallel with that
architecture work, preserve BF16/FP8 parity and compile boundaries already tested.


User priority refinement: isolate complete forward first against the 3.9 ms Prime
reference before continuing broad backward optimization. Existing H200 EP8
blockwise-FP8+NCCL artifact `benchmarks/h200_qwen3_30b_ep8_fused_mask.json` already
records forward median 9.842741 ms (2.52x 3.9); quantization and host split sizes
included, local callbacks compiled but whole layer not compiled. Five discarded
warmup iterations, fifteen measured, maximum rank per sample. Same Qwen geometry.
Next measurement: B200 persistent fused forward, separating prepared kernel-only
from complete operand preparation+dispatch+expert+combine forward, BF16 and MXFP8,
with activation retention recorded separately. Do not call B200 timings an H200
hardware match or subtract unrelated medians to infer forward.


Persistent BF16 activation retention primitive validated on two B200s:
`benchmarks/persistent_bf16_retention.json`, three forwards with 137/769/137
tokens, H512/I512/E8/topk4, all gate/up projections exactly matched independent
BF16 GEMMs on both ranks; earlier saved buffers remained bitwise unchanged.
Native optional `saved_z` uses dense [source_rank,max_tokens,topk,2I] storage on
each expert owner. This is a correctness-first retention layout, not a compact
final training workspace. Shared experts and clamped activations are rejected.
`runtime.py` now has opt-in BF16 `retain_activations` and uses saved projections
in existing NCCL backward, but that integration has NOT yet been validated; it
still exchanges source token IDs and uses host split counts. Do not promote it
as the GPU-driven backward implementation.

Forward-only benchmark now running: exec session74266,
`/private/tmp/ultra-persistent-forward-timing.log`, examples/modal_persistent_forward.py.
Eight B200s, Qwen geometry, BF16 then MXFP8, FP32 master weights, no activation
retention; compare prepared-input persistent forward versus complete preparation
(including weight conversion/quantization, input/routing copies and output
allocation)+forward. Router excluded. Five warmup iterations and fifteen measured
per scope, alternating scope order; synchronized host-wall maximum-rank timings.
Independent sampled BF16 reference validation precedes timings. Poll this same
handle and persist its terminal output before making latency claims.

Forward timing session74266 terminated before timing due to benchmark input-copy
shape mismatch: symmetric capacity is padded to 17280, active T is 16384. Failure
saved in `benchmarks/persistent_forward_capacity_failure.json`. Fixed all input,
scale and routing copies to active-token slices. Rerun is live session40205, log
`/private/tmp/ultra-persistent-forward-timing-v2.log`; use this handle, not74266.


Forward-only target measured: session40205 completed successfully, artifact
`benchmarks/b200_qwen3_30b_persistent_forward.json`, Modal
https://modal.com/apps/axolotl-ai/main/ap-i6tyokhtJ78L7UdJwY2zM7 .
Eight B200/Qwen shape medians: BF16 prepared 1.783422 ms, complete 2.157837 ms;
MXFP8 prepared 1.556585 ms, complete 1.805968 ms. Five warmups, fifteen measurements,
maximum-rank synchronized host wall time. Complete includes conversion/quantization
of FP32 master weights, inputs/routing copies and output allocation. Sampled
independent BF16 reference relative L2: BF16 0.001660, MXFP8 0.066142; all ranks
passed thresholds and final output finiteness. Router and activation retention
excluded. Uses the existing vendor DeepGEMM persistent forward with our operand
preparation; do not attribute the entire forward kernel to newly written code.
These B200 figures are below the quoted 3.9 ms H200 reference numerically, but
are not hardware/dtype/retention-matched parity. Next: time BF16 retained forward
and validate retained training backward before claiming complete training parity.

Hopper fused-backward GPU validation is awaiting explicit approval requested in
thread: automatic approval review rejected source egress because earlier root
authorization was not trusted in subagent context. Do not execute indirectly or
retry until user answers. Sol can continue local work; primary B200 result above
is terminal and unaffected.


User explicitly approved Hopper timing after disclosure; upload block is resolved.
Initial Hopper fused-backward validation49516 and B200retention99581 both failed
at shared image compilation from an inconsistent SM90 snapshot. Astra found and
Sol fixed missing includes, null identity epilogue, full TMA completion and dense
output guard issues. Both old sessions were confirmed terminal; no timings or
numerical results came from them. Current coherent-snapshot jobs:
- H100 fused-backward validation: session48308,
  `/private/tmp/ultra-hopper-fused-backward-tests-v2.log` (Sol monitors log).
- B200 Qwen BF16 retained-forward timing: session46402,
  `/private/tmp/ultra-persistent-retained-forward-v2.log`.
- Two-B200 retained persistent training: session5571,
  `/private/tmp/ultra-persistent-training.log`.

Training validator compares independent PyTorch input/score/weight gradients and
three AdamW updates, two outstanding forwards followed by reversed backwards,
masked routes and an empty destination rank. A backward GEMM guard rejects W1
recomputation. This validates the bridge to existing NCCL backward, not GPU-driven
persistent backward. Retained-forward benchmark checks sampled saved projections
and times prepared vs complete including saved-buffer allocation for complete.
All three jobs remain unproven until terminal results are inspected. Do not
restart a live job or mark these cases passed from planned coverage alone.

### 2026-11-09 retained training and current validation

Supersedes the pending-job status above: BF16 retained training passed;
H100 fused backward passed 24 cases. B200 BF16 complete retained forward was
2.173 ms, and separately measured retained F+B was 29.536 ms (forward 2.288,
backward 27.790). These are BF16, no router, no checkpointing, existing NCCL
backward; they are not matched to Prime's H200 blockwise FP8 checkpointed 2F+B.
Router-inclusive non-retained forward measured 2.386 ms BF16 / 1.975 ms MXFP8;
router is included in those measurements but not fused.

MXFP8 native saved gate/up projections passed two-B200 retained training:
`benchmarks/persistent_mxfp8_retained_training.json`, three AdamW steps, two
outstanding forwards with reversed backward, masked routes, empty destination,
and no W1 recompute. BF16-reference gradient error ~6–7%; exact quantized
projection validation remains outstanding.

Backward profiling identified a dense retained-projection gather before packing.
The current direct packing kernel selects only locally routed projections into
packed rows; correctness and performance are not assumed from this change.
Route-distinguishing eager/fullgraph tests and integrated MXFP8 training are
running under session 78283, log `/private/tmp/ultra-direct-packed-mxfp8-training.log`.
The training benchmark now accepts bf16/mxfp8, but fresh timings must follow
correctness. Existing timings predate direct packing.

Hopper fused forward's previous snapshot failed compilation; current corrected
snapshot (separate raw scratch avoids cross-N tile alias, explicit z destination,
BF16 staging type fix) runs under session 89554, log
`/private/tmp/ultra-hopper-fused-forward-v2.log`. Static Astra review found no
remaining concrete blocker; runtime evidence remains pending. Both Modal jobs
are authorized. Persistent GPU-driven backward and end-to-end overlap, compact
retention, fullgraph training and matched H200 FP8 checkpoint timing remain open.

Independent MXFP8 saved-projection reference validation added: per-32-value E4M3 with power-of-two scales computed in PyTorch, FP32 reference GEMM rounded to BF16, plus ownership/padding/lifetime checks. Running session 85997, log `/private/tmp/ultra-mxfp8-projection-reference.log`; unproven until terminal. BF16 integrated direct-packing training also launched, log `/private/tmp/ultra-direct-packed-bf16-training.log`. Direct-packing tests are selected by existing CUDA kernel CI (`tests -m gpu`).

### 2026-11-09 direct packing verified; real checkpoint validation

All previously pending B200 correctness runs completed successfully:
- `benchmarks/persistent_mxfp8_direct_packed_training.json`
- `benchmarks/persistent_bf16_direct_packed_training.json`
- `benchmarks/persistent_mxfp8_projection_reference.json`

Each includes four eager/fullgraph direct-packing cases passing. The independent
MXFP8 projection reference has max relative L2 ~2.13e-5 across both ranks and
three input lengths; ownership, unused padding and outstanding-buffer lifetime
checks pass. This is stronger saved-projection evidence than BF16-reference
training tolerance. H100 fused-forward v2 also passed 36 cases (Sol persisting).

Eight-B200 retained timing runs use the validated direct packing snapshot:
MXFP8 session74414 (`/private/tmp/ultra-direct-packed-mxfp8-timing.log`),
BF16 session4817 (`/private/tmp/ultra-direct-packed-bf16-timing.log`). These
snapshots predate the new checkpoint timing scope and are still building.

Added actual PyTorch nonreentrant checkpoint validation with preserved route
inputs, two outstanding forwards, reversed backwards, and an explicit forward
call count (four calls for two microbatches verifies 2F). MXFP8 validation is
session72473, `/private/tmp/ultra-mxfp8-checkpoint-training.log`. It is not yet
proven. Training benchmark now includes a separate `checkpointed_layer` scope
using that checkpoint engine; launch only after correctness passes. It still
has host-split NCCL backward and is not whole-layer fullgraph, so it is not a
matched Prime comparison even after real 2F+B timing is collected.

### 2026-11-09 checkpoint pass and initial direct-packing timing

Actual nonreentrant MXFP8 checkpoint training passed three AdamW steps on two
B200s, with two outstanding graphs reversed during backward and exactly four
forward calls per two microbatches. Evidence:
`benchmarks/persistent_mxfp8_checkpoint_training.json`.

BF16 eight-B200 direct-packing timing completed: forward 2.292 ms, backward
22.836 ms, measured retained F+B 23.604 ms (previous dense gather F+B 29.536 ms).
Evidence `benchmarks/b200_qwen3_30b_direct_packed_bf16_timing.json` retains all
samples, including large outliers (one layer sample 221.94 ms). This is initial
median evidence, not a tail-latency improvement claim. MXFP8 session74414 still
awaits terminal result. Both timing snapshots exclude the new checkpoint scope.

Root connected the existing native grouped MXFP8 GEMM SwiGLU backward epilogue
into the retained runtime when fuse_activation=True. This removes materialized
dH and the separate derivative launch. New direct fused-vs-unfused eager and
fullgraph tests cover empty rows and padded expert tiles, mounted in the retention
runner and selected in Blackwell CI. Changes pass pinned lint but are NOT yet GPU
validated. Hold new Modal snapshots until Sol declares shared Hopper FP8 native
changes build-ready; existing live snapshots are unaffected. After correctness,
measure fused MXFP8 retained and real checkpointed 2F+B versus the saved unfused
baseline. Persistent backward comms overlap remains the larger unfinished work.

MXFP8 direct-packing retained timing completed: forward **1.961 ms**, backward **28.480 ms**, separately measured F+B **29.675 ms**, same eight-B200 geometry and warmup method as BF16 above. Evidence: `benchmarks/b200_qwen3_30b_direct_packed_mxfp8_timing.json`. This snapshot uses the unfused activation derivative; the newly wired fused epilogue remains unvalidated. MXFP8 forward is faster, but this full training path is slower than BF16; FP8 quantization and backward scheduling remain optimization targets.

### 2026-11-09 fused MXFP8 and Hopper FP8 validation launched

B200 fused MXFP8 checkpointed-training validation: session89999, log
`/private/tmp/ultra-fused-mxfp8-checkpoint-training.log`. Includes new direct
fused-vs-unfused derivative tests (empty and padded tiles, eager/fullgraph),
packing tests, three AdamW steps and real checkpoint forward-call count.
No correctness/performance claim until its terminal result is inspected.

H100 blockwise FP8 readiness plus BF16 regressions: session7475, log
`/private/tmp/ultra-hopper-fp8-ready.log`. Snapshot is source-coherent but predates
Astra's scale-layout validation finding. Sol is adding metadata guards and
negative tests; preserve valid-input evidence from this run, then rerun corrected
snapshot after terminal. Do not restart while building. Current shared native
source edits are owned by Sol; coordinate snapshots.

Paired fusion benchmark added (`modal_persistent_forward.py --training --compute-dtype mxfp8 --compare-fusion`): alternates fused/unfused cases on the same allocation, independently warms each, retains per-sample maximum-rank timing for forward, backward, retained F+B and actual checkpointed 2F+B. Pinned lint passes; launch after session89999 correctness succeeds. Sessions89999/7475 remain live after build. Sol fixed reviewer compact SFA-stride guard and negative coverage; follow-up H100 snapshot pending original terminal result.

Fused MXFP8 validation terminal PASS: eight primitive/compile tests and three
checkpointed AdamW steps on two B200s. Evidence:
`benchmarks/persistent_fused_mxfp8_checkpoint_training.json`. Paired eight-B200
fusion timing launched, log `/private/tmp/ultra-mxfp8-fusion-paired-timing.log`;
includes actual checkpointed 2F+B scope. Inspect terminal before claiming speedup.

H100 initial FP8 readiness run terminal FAIL: 36 existing BF16 cases passed,
first FP8 case triggered CUDA launch failure and poisoned following cases.
Evidence `benchmarks/h100_fp8_ready_initial.json`. Sol diagnosing first failure;
do not treat metadata guard corrections as sufficient resolution or rerun merely
because source is coherent. Current H100 FP8 path is not validated.

### 2026-11-09 transpose quantization draft and live runs

Root added direct quantization of transposed expert-weight views, avoiding the
intermediate FP32/BF16 contiguous transpose in MXFP8 derivative GEMMs. Uses the
same per-row power-of-two E4M3 recipe; no cached weight reuse. New tests compare
values/scales bit-for-bit across BF16/FP32, irregular dimensions, empty experts,
and eager/fullgraph. Selected by existing CUDA kernel CI (`tests -m gpu`) and
mounted in the retention runner. Pinned lint passes, GPU evidence pending.
Integrated checkpointed-training validation session77606, log
`/private/tmp/ultra-transpose-quant-training.log`. Do not claim speedup yet.

Paired fusion timing session45086 remains live (native build), log
`/private/tmp/ultra-mxfp8-fusion-paired-timing.log`; snapshot predates transpose
quantization. H100 isolated failure diagnosis session60977, log
`/private/tmp/ultra-hopper-fp8-debug.log`: fresh CUDA subprocesses isolate delayed
queue/A/SFA/B/SFB/IDs, preventing one sticky failure from obscuring the cause.
Current shared native metadata-guard checkpoint is stable per Sol.

Transpose quantization first GPU run: 16 passed / 4 fullgraph failures, before
training started. Eager values/scales matched exactly; compiler failed lowering
unused optional None pointer (`_unflatten_ir`). Evidence:
`benchmarks/transposed_quantization_initial_failure.json`. Passed an unused tensor
pointer for the statically disabled expert mask; no numerical recipe change.
Rerun session19685, `/private/tmp/ultra-transpose-quant-training-v2.log`.

Hopper isolated debug snapshot contains a harness flaw: CUDA_LAUNCH_BLOCKING can
block consumer launch before producer publication is enqueued. Sol removed it
from delayed cases in current source. Session60977 still live; a delayed-stage
timeout from that snapshot is not kernel evidence. Preserve any ready-warmup
failure, and wait for terminal before corrected rerun. Paired fusion session45086
remains live; no new timing result yet.

### 2026-11-09 paired fusion regression; gradient-return overlap draft

Paired B200 MXFP8 result: retained F+B unfused 29.550 / fused 31.367 ms;
actual checkpointed 2F+B unfused 31.550 / fused 33.327 ms. This is a regression,
not a fusion speedup. Runtime's default fuse_activation=False remains appropriate.
Evidence `benchmarks/b200_qwen3_30b_mxfp8_fusion_paired.json`. Same allocation,
5 warmups/15 measured per case, alternating order; router excluded, NCCL backward,
not whole-layer fullgraph. Snapshot predates direct transpose quantization.

Root reordered BF16/MXFP8 explicit backward: compute dx/dscore, enqueue their
asynchronous return collectives, then compute dw2/dw1 before waiting for returned
gradients. An internal callback exposes this scheduling boundary; standalone
expert callers retain synchronous output semantics. This permits NCCL return
traffic to overlap weight-gradient compute, but actual overlap and speedup must
be measured. GPU validations launched:
`/private/tmp/ultra-return-overlap-mxfp8-training.log` (checkpointed),
`/private/tmp/ultra-return-overlap-bf16-training.log` (retained).
These include the pending transpose quantization compiler fix.

Corrected Hopper diagnostics session53893, `/private/tmp/ultra-hopper-fp8-debug-v2.log`.
Original debug run terminal; launch-blocking snapshot's delayed-stage failures
are invalid evidence. Warm-ready launch succeeded. No scheduler tuning based on
that harness failure is justified.

### 2026-11-09 user priority: remove actual host synchronization

User explicitly prioritizes host sync. Root added `fixed_dispatch.py` as an
interim fixed-capacity NCCL backward: equal known splits, no count exchange or
GPU-to-CPU split conversion; rank/token addresses derived from configured capacity.
Forward route validation now uses a device assertion rather than `.item()`.
The existing compact host-split branch remains intact for comparisons via an
internal runtime switch. Current runtime uses fixed capacity; GPU validation is
pending and it is not yet a performance recommendation.

This trades extra payload/padding for removal of data-dependent host splits.
It is not the final persistent transport or proof of whole-layer compile/capture.
Measure before choosing the production schedule. Main destination remains a
GPU-resident persistent recomputing backward; no existing work has been deleted.

Validation now covers unequal/changing rank token counts, workspace padding,
masked routes, empty destination, reversed outstanding graphs, optimizer updates,
and Python Tensor cpu/tolist/item guards around actual forward/backward. These
guards catch explicit Python reads, not arbitrary hidden native synchronization.
BF16 session43891 (`/private/tmp/ultra-fixed-dispatch-bf16.log`), MXFP8 checkpoint
session40071 (`/private/tmp/ultra-fixed-dispatch-mxfp8.log`). Collect terminal
correctness, then timings and a separate trace to inspect remaining host sync.

Prior early-return schedule validations passed for both BF16 and MXFP8, preserved
as `benchmarks/persistent_return_overlap_{bf16,mxfp8}_training.json`.

Fixed-capacity BF16 validation PASS: unequal rank sizes, padded capacity, masked
and empty destinations, outstanding graphs, three optimizer steps, and explicit
Python host-read guards. Evidence `benchmarks/persistent_fixed_dispatch_bf16.json`.
Eight-B200 latency and separate trace launched:
`/private/tmp/ultra-fixed-dispatch-bf16-timing.log`,
`/private/tmp/ultra-fixed-dispatch-bf16-profile.log`. Profiler output now includes
GPU kernel/memcpy intervals and synchronization events, with its explicit final
synchronize labeled; use this to verify actual overlap and native host sync.
MXFP8 fixed-capacity validation remains pending.

Corrected H100 no-launch-blocking diagnosis still fails readiness publication in
immediate, priority, delayed, and all-payload cases. Evidence preserved in
`benchmarks/h100_fp8_ready_debug_v2.json`. Warm-ready success does not validate
incremental readiness. Keep unresolved without further broad tuning while host
sync and persistent backward take priority.

### 2026-11-09 host-sync baseline result and compiler contract

Fixed-capacity eight-B200 results (same Qwen geometry, 5 warmups/15 samples,
median maximum-rank latency): BF16 retained F+B 30.793 ms / real checkpointed
2F+B 32.999 ms; MXFP8 unfused retained F+B 40.387 ms / checkpointed 42.306 ms.
These regress against compact dispatch; extra traffic and padding outweigh the
removed host read. Preserve as a host-count-free correctness/compiler baseline,
not a performant final transport. Native GPU-count compact dispatch is required.
Evidence: `benchmarks/b200_qwen3_30b_fixed_dispatch_{bf16,mxfp8}_timing.json`.

Refactored runtime forward/backward bodies for a recomputing custom-op boundary
(`mono_ep::persistent_forward/backward`), fake implementations and explicit
registered autograd. Context saves only input/routes/weights; retained legacy
path remains intact. Registry handle has explicit close after outstanding graphs
finish. This is a compiler boundary around native forward + fixed NCCL backward,
NOT a newly fused persistent backward, nor proof of capture safety.

Whole-layer fullgraph validation launched: BF16 session44441,
`/private/tmp/ultra-recompute-fullgraph-bf16.log`; MXFP8 checkpoint session31072,
`/private/tmp/ultra-recompute-fullgraph-mxfp8.log`. Tests assert one W1 recompute
per backward and forbid Python cpu/tolist/item reads, plus varying unequal rank
sizes and three optimizer steps. Pending GPU evidence. Shared workspace lifecycle
and FSDP SM reservations remain open, as does native persistent backward.

Fullgraph initial tests failed before kernel execution: Python Tensor-method
MagicMocks used as eager host-read guards break Dynamo introspection (`__name__`).
Saved both failures. Kept those guards for eager validation and separated them
from compiled validation (metadata reports this explicitly). Reruns: BF16
session55081 `/private/tmp/ultra-recompute-fullgraph-bf16-v2.log`; MXFP8 session22794
`/private/tmp/ultra-recompute-fullgraph-mxfp8-v2.log`. Runtime registry keys now
monotonically increase so a stale compiled handle cannot alias a new runtime after
close/id reuse. Profiler session37718 remains live; don't restart from log silence.

### 2026-11-09 fullgraph GPU results and checkpoint accounting

Sessions55081/22794 completed successfully: BF16 recomputing training and MXFP8
recomputing training with non-reentrant checkpointing, both fullgraph, two B200s,
unequal changing token counts, reversed outstanding backwards and three AdamW
steps. Each also passed 20 packing/epilogue/transpose-quantization cases. Saved
raw results in `benchmarks/recompute_fullgraph_{bf16,mxfp8}.json`. This proves the
opaque compiler boundary and tested gradient tolerances, not capture safety or a
native fused backward. MXFP8 relative gradient errors versus the BF16 reference
are about 6–7%; BF16 errors are below 0.4% in these cases.

The checkpoint wrapper call count does not establish a second native forward:
non-reentrant checkpointing may stop after recovering saved inputs. Added a
counter around the actual DeepGEMM entry point and explicit replay metadata;
also made the remaining backward Tensor.tolist mock conditional in compiled
tests, consistent with the other eager-only guards. Native-call accounting run:
session36393, `/private/tmp/ultra-recompute-native-count-mxfp8.log`. Do not label
that test 2F+B until native replay is observed. Eight-B200 profiling session37718
was polled and remains live with no result yet.

Native replay check completed successfully: `recompute_native_count_mxfp8.json`
records four native forward calls for two checkpointed microbatches on both ranks
at each of three steps. This is actual replay, not just Python wrapper entry.

Profiler37718 completed. Six intact rank records (0,1,2,3,4,6; two records suffered
interleaved torchrun stdout) show 1113–1304 us intersection between NCCL return
kernels and the two KGroupedContiguousWithPsumLayout weight-gradient kernels.
No DtoH or scalar-read event appears in those records; only the explicit final
device synchronize and profiler shutdown synchronize were recorded. This is
evidence of overlap, not a latency improvement from profiling. Raw artifact:
`b200_fixed_dispatch_bf16_profile.json`; extracted timestamp intersections:
`b200_fixed_dispatch_bf16_overlap_summary.json`.

Implemented `peer_packing.py::pack_peer_rows`: GPU routing assignments directly
gather BF16/FP32 rows from peer mappings into expert-packed storage. It requires
caller-owned publication/lifetime ordering and performs no barrier or host count
read. This is a transport building block, not yet wired into persistent backward.
Local-mapping indexing/fullgraph tests plus BF16 regression running in session98507,
`/private/tmp/ultra-peer-packing-validation.log`; real multi-GPU peer mappings and
end-to-end integration still required. No speed claim before those checks.

Session98507 passed all 28 local GPU tests (including eight peer gather indexing,
padding and fullgraph cases) and BF16 fullgraph training; saved as
`peer_packing_local_and_bf16_training.json`. Real-peer validation now exercises
symmetric mappings on two B200s, GPU epoch publication/reuse, unequal live token
counts, all-masked routes, empty packed outputs, BF16/FP32 and eager/fullgraph.
It runs before training in the validation runner. MXFP8 job session42418 uses
`/private/tmp/ultra-real-peer-packing-validation.log` and is still building/running.

Extracted `grouped.expert_backward_packed` so receive-side packing can feed BF16
compute directly, preserving the callback that starts gradient return before dW
GEMMs. Existing unpacked entry point delegates to it. GPU regression launched in
`/private/tmp/ultra-packed-backward-bf16-validation.log`; this extraction does not
yet wire a peer transport into the public runtime.

Hopper agent parameterized native scheduler CTA cluster size (default two for
SM100, optional one for SM90) and corresponding ring bounds, warmup, locality and
publication. Static review passed and vendor manifest updated. A native SM90
kernel/host launcher remains unimplemented; new native-image GPU runs are needed
to confirm unchanged SM100 behavior after this preparation.

### 2026-11-09 direct-peer BF16 backward integration

Added experimental `PeerBackward` behind the internal runtime `_peer_backward`
injection (validation flag `--peer-backward`, not a public config option). One
symmetric input/gradient copy per source rank feeds GPU-count expert layouts;
`pack_peer_rows` gathers only routed payloads. A return kernel writes each unique
source token/top-k gradient slot directly into peer storage, followed by local
FP32 summation. GPU epoch barriers protect publication/reuse, and a local event
orders successive calls across streams. No NCCL payload exchange or CPU count
read occurs in this backward path. Metadata is still gathered across all ranks;
packed allocations retain worst-case capacity. These costs remain to measure.

This recomputes z in separate grouped GEMMs and returns gradients before weight
GEMMs; it is NOT the final persistent fused backward or evidence of performance.
BF16, same-node, recompute-only initially. Python epochs preclude CUDA graph
replay; fullgraph is a separate gate. Existing transports remain available.

First end-to-end validation: session23073,
`/private/tmp/ultra-direct-peer-backward-bf16.log`, two B200s, checkpointing,
unequal changing token counts, masked/empty experts, reverse outstanding graphs,
three AdamW steps, eager Python cpu/tolist/item guards. Real peer gather + MXFP8
regression session42418 and packed backward BF16 regression session85986 remain
live after native image construction; do not restart due to quiet logs.

All three completed successfully. Real-peer BF16/FP32 gather tests (both eager
and fullgraph, with GPU epochs) passed, as did BF16 and MXFP8 fullgraph training
on the native image containing the scheduler parameterization. Artifacts:
`real_peer_packing_mxfp8_training.json`, `packed_backward_bf16_training.json`.
Direct-peer BF16 checkpointed training passed with eager cpu/tolist/item guards,
actual native replay, unequal token counts, empty experts, reverse outstanding
graphs and three optimizer steps: `direct_peer_backward_bf16_initial.json`.

Launched direct-peer whole-layer fullgraph checkpoint validation in
`/private/tmp/ultra-direct-peer-backward-fullgraph.log`, and eight-B200 Qwen timing
in `/private/tmp/ultra-direct-peer-backward-qwen-timing.log`. Benchmark now has
explicit recompute/peer flags and reports recomputed layer scope, actual retention
and activation-fusion settings. Five warmups / fifteen max-rank samples remain
unchanged. No new timing result yet; worst-case packed allocation cost remains.

Direct-peer BF16 fullgraph checkpoint validation passed
(`direct_peer_backward_bf16_fullgraph.json`). Eight-B200 recomputing timing is
F 2.2850 ms, B 32.2790 ms, F+B 34.3025 ms, checkpointed 2F+B 36.5164 ms
(`b200_qwen3_direct_peer_bf16_timing.json`). Same Qwen geometry and warmup policy,
router excluded. This is not a speed claim versus earlier retained NCCL results:
activation policy differs. Same-policy NCCL baseline launched in
`/private/tmp/ultra-recompute-nccl-qwen-timing.log`.

Extended the packed derivative entry point to MXFP8 without changing existing
retained/unpacked APIs; direct-peer payloads remain BF16 while compute quantizes
to MXFP8. Fullgraph checkpoint validation session15458 is running at
`/private/tmp/ultra-direct-peer-mxfp8-validation.log`. Uses frozen native snapshot
`/private/tmp/mono-ep-vendor-pre-hopper-kernel-20261109` via build-only
`MONO_EP_VENDOR_DIR`, excluding the two unfinished SM90 kernel/host files.
This keeps Python validation independent of the Hopper agent's native edits.

Hopper agent is actively replacing the isolated kernel's UMMA/TMEM pipeline with
a dedicated WGMMA/register/shared-memory warpgroup plus scheduler/TMA warpgroup.
Those new SM90 files are not buildable or API-hooked yet; do not count the skeleton
as completed support. The large remaining B200 gap likewise needs bounded ring
storage and persistent compute/communication fusion, not just transport changes.

### 2026-11-09 recompute epilogue for persistent backward

MXFP8 direct-peer fullgraph checkpoint training passed (session15458), including
native replay and three optimizer steps. Saved `direct_peer_mxfp8_fullgraph.json`.
Matching recompute NCCL timing session8018 remains live.

Implemented native SM100 `SwiGLURecomputeBackward` epilogue: consume the z-GEMM
accumulators (adjacent gate/up interleave, rounded to BF16) and precomputed BF16 dH;
emit weighted h, concatenated dz, and four-column score-gradient partials directly.
No z output is materialized. Existing SwiGLUForward semantics are preserved via a
shared templated implementation. This supplies the distinct epilogue contract
needed by a recomputing persistent backward; it is not the old dH epilogue attached
to the wrong GEMM. Current grouped wrapper still materializes dH, h and dz.

New `recompute.py::grouped_recompute_swiglu` custom op/fake and six GPU cases cover
empty rows, masked expert tiles, two inner widths, eager/fullgraph and comparison
with separate BF16 projection plus derivative. Native build/test session86389:
`/private/tmp/ultra-recompute-epilogue-tests.log`, frozen native snapshot
`/private/tmp/mono-ep-vendor-recompute-epilogue-20261109` excludes Hopper WIP.
Not yet switched into runtime; await numerical/native build validation first.

Integrated the new epilogue into `expert_backward_packed` for SM100 BF16 when
activation fusion is enabled and no saved z is supplied. It computes dH first,
then recomputes z and its derivatives in the fused epilogue; retained and SM90
paths retain their previous schedule. Validation records fused recompute call
counts separately from ordinary W1 recomputation. This integration is still
unvalidated while the native build runs, and no speed claim is made.
Fullgraph checkpoint training session19539:
`/private/tmp/ultra-recompute-epilogue-training.log`, using the same frozen native
snapshot as standalone epilogue session86389. NCCL baseline session8018 was also
polled and remains live. Do not infer terminal status from log silence.

### 2026-11-09 epilogue validation failure and correction

Same-policy recompute NCCL baseline completed: F 2.2982 ms, B 30.2757 ms,
F+B 32.1587 ms, checkpointed 34.4569 ms. Direct-peer backward was ~6.6% slower
in these separate allocations, despite eliminating NCCL payload transfers.
Saved `b200_qwen3_recompute_nccl_bf16_timing.json`; neither transport closes the
fusion gap, and the peer path is not a performance recommendation.

Initial epilogue build succeeded but four nonempty correctness cases failed:
masked M-group tiles still ran the direct-output epilogue. The native empty-tile
condition only covered SwiGLUBackward. Extended it to SwiGLUForward (including
the new recompute specialization). A fifth test failed from Dynamo's shared
recompile cache limit after adding cases; new compiled tests now reset that cache
between independent cases. Saved failures in `recompute_epilogue_initial.json`
and `recompute_epilogue_training_initial.json`. Training was not reached.

Added the epilogue/peer tests to the Blackwell CI lane. Corrected native snapshot
and vendor local_modifications entries; first v2 launch session2081 stopped locally
because snapshot creation failed on an incorrect manifest key (no GPU execution).
Actual rerun session38039 uses `/private/tmp/ultra-recompute-epilogue-v2-run.log`
and `/private/tmp/mono-ep-vendor-recompute-epilogue-v2-20261109`. Numerical
agreement and integrated training remain unproven pending this run.

Corrected epilogue run38039 PASSED: 34 kernel tests, real-peer gather tests and
fullgraph checkpointed BF16 training with two fused recompute calls per step.
Saved `recompute_epilogue_v2_training.json`. Native replay and three AdamW steps
also passed; z is no longer materialized in the fused recomputing BF16 path.

Changed peer gather/return launch topology from worst-case row-by-column grids
to fixed-size GPU worker grids looping to a device-resident padded row end.
This targets millions of inactive scheduled work items without a host count
read; allocation capacity and output initialization remain unchanged. Added
changing-row-bound eager/fullgraph tests. Integration validation running at
`/private/tmp/ultra-peer-resident-work-validation.log`. A separate NCCL run at
`/private/tmp/ultra-recompute-fused-nccl-qwen-timing.log` isolates recompute-epilogue
timing from these peer scheduler changes. No new performance result yet.

Both runs passed. Resident-work scheduling: 42 kernel tests and fullgraph
checkpointed BF16 training (`peer_resident_work_bf16_training.json`). Isolated
fused-recompute NCCL timing: F 2.3018 ms, B 28.8589 ms, F+B 30.8687 ms,
checkpointed 33.0892 ms (`b200_qwen3_fused_recompute_nccl_timing.json`). Previous
same-policy B was 30.2757 ms; these are separate allocations, not paired samples.
One checkpoint sample was 144.84 ms and remains in the raw artifact.

Now running eight-B200 resident-peer timing at
`/private/tmp/ultra-peer-resident-qwen-timing.log`, plus MXFP8 resident-peer
fullgraph regression at `/private/tmp/ultra-peer-resident-mxfp8-validation.log`.

Hopper agent's first Modal request was rejected for missing upload authorization
in its transcript. Root presented the user's explicit prior Modal and Hopper
approvals; approval review accepted the single-H100 run. Session73594,
`/private/tmp/ultra-hopper-persistent-single-root.log`, is building the coherent
SM90 WGMMA prototype with API dispatch. Static review passed; GPU validity is
still pending. Do not duplicate this launch from the agent.

### 2026-11-09 resident peer results and Hopper compile correction

Resident-peer BF16 timing completed successfully on eight B200s: F 2.3038 ms,
B 23.8582 ms, F+B 25.9099 ms, checkpointed 28.1738 ms. Geometry remains
16384 tokens/rank, H2048, I768, E128, top8, EP8; router excluded, five warmups,
15 measured maximum-rank synchronized samples. Raw results saved as
`b200_qwen3_resident_peer_bf16_timing.json`. This combines resident scheduling
and fused recomputation; separate allocations do not isolate their individual
contributions. Full persistent backward is still unfinished.

MXFP8 resident-peer fullgraph checkpointed training passed, including 42 kernel
tests and the real-peer gather checks. Saved `peer_resident_work_mxfp8_training.json`.
MXFP8 does not yet use the BF16 fused recompute epilogue.

H100 first run failed JIT compilation at six CUTLASS/native BF16 conversions.
Agent corrected these conversions; single-H100 rerun log:
`/private/tmp/ultra-hopper-persistent-single-v2.log`. No H100 numerical result yet.
B200 resident-peer backward profile is running (session21089), log:
`/private/tmp/ultra-resident-peer-profile.log`, to select the next overlap target.

H100 corrected run is live as session84473; Astra re-review found no additional
static issues. Both session84473 and B200 profile21089 were polled live.
Profile reporting now gathers records after the measured/profiled region and
prints one rank-zero JSON object, preventing torchrun interleaving from losing
rank evidence on future runs. Pinned Ruff check/format passed; current profile
launch predates this reporting change.

### 2026-11-09 direct-peer return overlap

Resident-peer profile21089 completed with all eight rank records intact. Saved
`b200_resident_peer_profile.json` and `b200_resident_peer_overlap_summary.json`.
The return kernel took 857–888 us of profiled service time and had zero overlap
with the two weight-gradient kernels on every rank. No DtoH or scalar-extraction
sync appeared; explicit final/profiler synchronizations remain. Profiler barrier
waits are rank-skew-sensitive and are not unprofiled latency estimates.

Moved direct-peer gradient return to a dedicated stream after waiting for the
producer stream. Recorded allocator stream ownership for dx, ds, assignments,
and row_end; the main stream waits for return completion after weight gradients
and before the peer barrier/workspace reuse. This remains an experimental
recomputing backward path, not full persistent fusion. Pinned Ruff passed.
Fullgraph checkpointed BF16 validation session80664 is running at
`/private/tmp/ultra-peer-return-overlap-validation.log`; separate overlap profile
runs at `/private/tmp/ultra-peer-return-overlap-profile.log`. No correctness or
performance claim for this change until those runs complete.

### 2026-11-09 return overlap validated; projection scheduling

Return-stream validation80664 passed fullgraph checkpointed BF16 training and
42 kernel tests. Saved `peer_return_overlap_training.json`. Profile11971 also
passed; `b200_peer_return_overlap_profile.json` contains all eight gathered rank
records. Return/dW overlap is 64.6–943.5 us across ranks, versus zero before.
This proves concurrency, not full-layer speedup; profiled service times vary.

BF16 grouped projections and the fused recompute epilogue now optionally consume
cumulative padded expert row ends from DeviceExpertLayout, selecting DeepGEMM's
psum M scheduler. This skips worst-case unused capacity without host count reads;
legacy ExpertLayout still uses per-row IDs. Padded ends preserve 128-row expert
boundaries and zero payloads. Fullgraph checkpoint validation1805 is running:
`/private/tmp/ultra-peer-psum-validation.log`. Pinned Ruff passed. New scheduling
is unvalidated pending that run.

Hopper v2 passed C++ compilation but PTX rejected Blackwell-only st.bulk. Agent
replaced it with warp-cooperative shared stores and retained the CTA barrier.
Root v3 session23755 is running at `/private/tmp/ultra-hopper-persistent-single-v3.log`.

Added eager/fullgraph recompute-epilogue cases for cumulative-count scheduling,
with an empty final expert and unused trailing allocation rows. Pinned Ruff
passed. Existing live validation1805 predates these six additional cases; they
still need GPU execution. Validation1805 and Hopper23755 were re-polled live.
Combined BF16 projection-count scheduling plus peer-return overlap timing is
running at `/private/tmp/ultra-peer-psum-timing.log`; results must be conditioned
on correctness validation, and cannot isolate the two changes individually.

### 2026-11-09 projection-count training passed

BF16 validation1805 passed fullgraph checkpointed recomputing training, three
AdamW steps, unequal tokens and empty experts, with the return stream enabled.
Saved `peer_psum_bf16_training.json`. Its 42 kernel tests predate the six newly
added psum epilogue cases. Eight-B200 timing39587 remains active.

Extended optional cumulative-count scheduling to FP8 grouped projections and
recomputing packed backward. Quantization still uses per-row IDs for padding;
GEMM scheduling receives padded expert ends. Fused MXFP8 activation mode retains
its prior dH scheduler; default MXFP8 activation mode is unfused. Validation4548
runs the expanded 48-test suite plus MXFP8 fullgraph checkpointed training:
`/private/tmp/ultra-peer-psum-mxfp8-validation.log`. No FP8 pass claimed yet.

H100 v3 compiled successfully but execution failed with CUDA illegal instruction
encoding (Xid13), before parity checks. Hopper agent is diagnosing runtime
instructions; log preserved at `/private/tmp/ultra-hopper-persistent-single-v3.log`.
No Hopper persistent-forward correctness result exists yet.

### 2026-11-09 count scheduling timing and input overlap

Eight-B200 timing39587 passed: F 2.2929 ms, B 15.7580 ms, F+B 17.8648 ms,
checkpointed 20.1167 ms. Saved `b200_peer_psum_bf16_timing.json`. Same Qwen3
geometry/warmup policy as prior BF16 results, router excluded. Separate allocation
versus previous B23.858/F+B25.910/AC28.174; combined scheduler and stream changes
cannot be attributed individually from this comparison.

Extended communication stream to gather x/scores while main stream gathers dy
and computes dH. BF16 fused recompute waits for x/scores only after dH; other
paths wait before consuming x. Cross-stream allocator ownership is recorded.
Validation78219: `/private/tmp/ultra-peer-input-overlap-validation.log`. No pass or
speedup claimed yet. Full persistent producer/consumer kernel fusion remains open.

MXFP8 validation4548 passed the expanded 48 kernel tests but training stopped
because its instrumentation wrapper lacked the new group_ends keyword. Updated
wrapper; failure preserved in `peer_psum_mxfp8_initial.json`. Rerun47851 at
`/private/tmp/ultra-peer-psum-mxfp8-v2-validation.log` includes current stream code.
Hopper agent identified divergent per-role setmaxnreg as a concrete v3 runtime
hazard and hoisted it to converged warpgroups; awaiting Astra check before v4.
Astra confirmed the setmaxnreg convergence fix; root launched H100 v4 at
`/private/tmp/ultra-hopper-persistent-single-v4.log`. Runtime confirmation pending.

### 2026-11-09 input overlap correctness passed

Validation78219 passed: 48 kernel tests, real-peer gather checks, BF16 fullgraph
checkpoint replay and three optimizer steps. Saved
`peer_input_overlap_bf16_training.json`. Input gather/dH overlap profile30261
is active at `/private/tmp/ultra-peer-input-overlap-profile.log`; unprofiled
eight-B200 timing45831 at `/private/tmp/ultra-peer-input-overlap-timing.log`.
No speedup claim for input overlap until measurements complete. MXFP8 rerun47851
and H100 v4 session4264 remain live on polling.

### 2026-11-09 MXFP8 scheduling validated

MXFP8 rerun47851 passed all 48 kernel tests and fullgraph checkpointed training
with cumulative-count projection scheduling and current communication streams.
Saved `peer_psum_mxfp8_training.json`. Reference gradient differences remain
within the existing MXFP8 tolerance (~6–7% versus BF16 reference); optimizer
weight differences remain below 0.1%. No accuracy tightening claimed.
Eight-B200 MXFP8 timing11039 is running at
`/private/tmp/ultra-peer-psum-mxfp8-timing.log`. BF16 input overlap profile30261,
timing45831 and Hopper v4 build4264 were polled live. Full fused recomputation
still applies only to BF16; MXFP8 retains its separate z/activation path.

Poll correction: the preceding batch returned terminal status for30261/45831/4264,
not live. BF16 profile30261 and timing45831 passed; H1004264 failed again.
BF16 timing: F2.2755, B13.5585, F+B15.5564, AC17.8192 ms. Saved
`b200_peer_input_overlap_timing.json` and `b200_peer_input_overlap_profile.json`.
Profile shows x/dy gather kernels overlapping on separate streams (rank0 ~2.3ms),
but zero gather/dH overlap on all eight ranks. The implemented deferred wait
permits gather/dH overlap; this trace does not demonstrate it. Do not attribute
the timing change to GEMM overlap; separate allocation and concurrent gathers
are material qualifications.
Hopper v4 still reports illegal instruction after the convergence fix. Agent
asked to prepare runtime instruction/sanitizer localization. MXFP8 timing11039
remains outstanding; no persistent Hopper forward correctness claim.

### Priority update: eliminate collective/split dependencies end to end

User explicitly prioritizes NCCL all-to-alls and host split synchronization
without discarding existing progress. Current scope distinction:
- Experimental recomputing PeerBackward already avoids NCCL payload all-to-alls
  and host split reads on a single node; it still stages full packed arrays and
  launches separate GEMMs, so it is not the final fused persistent backward.
- runtime.py legacy compact branch retains cpu().tolist() split extraction;
  fixed-capacity branch removes that host read but retains NCCL payload traffic.
- Axolotl torch_pipeline and the current FP8 adapter still route through older
  collective schedulers. Standalone peer results do not prove integration removal.

Next priority is GPU-resident dispatch/readiness/gradient-return inside persistent
backward, followed by _MonoEPBackend integration without falling through the
legacy dispatch/local/combine path. Preserve current validated paths as references;
defer further standalone packing/clear micro-optimizations. Multi-node transport
still requires an explicit implementation; same-node peer mappings do not prove it.
Added eager validation guards rejecting dist/functional all-to-alls alongside
existing Tensor.cpu/tolist/item guards. Compile validation retains profiler-based
checks because monkeypatching tensor methods conflicts with Dynamo introspection.
Guard run: `/private/tmp/ultra-peer-no-collective-guard.log`. Not yet passed.

### 2026-11-09 per-expert peer publication

Eager guard64702 passed: direct-peer checkpoint backward did not invoke guarded
functional/dist all-to-alls or Tensor.cpu/tolist/item. Saved
`peer_no_collective_guard.json`. This validates the standalone single-node path,
not the existing Axolotl collective integration or multi-node operation.

Implemented `pack_peer_rows_ready_out`: eight resident workers per expert gather
peer payloads, aggregate completion with system-scope acquire/release counters,
and publish each expert queue flag independently. Empty experts also publish.
Caller owns zeroed output/counters/flags and protects source lifetime. This is
producer infrastructure for the existing ReadyScheduler, not yet a fused backward
or a demonstrated concurrent consumer. Added real-peer BF16/FP32 eager/fullgraph
checks for exact payloads, all-masked cases and completion flags. Validation2040
runs at `/private/tmp/ultra-peer-ready-producer-validation.log`.

Hopper sanitizer19141 localized cluster-shared stores in a one-CTA launch. Agent
specialized scheduler publication for local shared memory; review pending.
MXFP8 timing11039 passed and is saved as `b200_peer_psum_mxfp8_timing.json`.

### 2026-11-09 readiness consumer validation

Added a real-peer readiness-aware BF16 GEMM test. It warms JIT paths first,
poisons packed input, starts the GEMM consumer before the gather producer on
separate streams, reserves 16 SMs for producer progress, and compares output to
the existing grouped reference. Includes all-empty expert case; metadata remains
GPU-resident and completion uses per-expert release/acquire publication. Outputs
and metadata stay alive until both streams join. This tests protocol correctness,
not actual timeline overlap or speed. Validation37407 is running at
`/private/tmp/ultra-peer-ready-consumer-validation.log`; producer-only2040 remains
live on last poll. Pinned Ruff passed. Not integrated into training backward yet.
Hopper review requested truly local `.arrive()` barrier overloads before rerun;
agent is addressing that finding without altering the SM100 branch.

### 2026-11-09 readiness protocol passed; backward connection

Producer2040 and concurrent consumer37407 both passed; raw artifacts saved as
`peer_ready_producer_validation.json` and `peer_ready_consumer_validation.json`.
Consumer-first launch correctly waited for real peer payload publication, including
empty experts. This proves protocol correctness, not measured overlap.

Reordered producer grid so workers for one expert are adjacent. Connected ready
gather to dH GEMM in experimental BF16 fused PeerBackward; consumer reserves16SMs,
then joins producer before later derivative stages. Existing staged path retained
via internal `PeerBackward(..., ready_gather=False)`. MXFP8 stays staged. Fullgraph
training validation running at `/private/tmp/ultra-peer-ready-backward-validation.log`.
No pass or latency claim for integrated path yet. Full persistent kernel fusion,
Axolotl backend integration and multi-node transport remain open.
Hopper local-barrier/cluster-store fix is under sanitizer v5, session50865,
`/private/tmp/ultra-hopper-persistent-diagnostic-v5.log`.

### User validation priorities: larger shapes and BF16 master weights

After fusion and communication-overlap pieces stabilize, final validation must
include the user's larger Qwen 397B reference geometry. Verify the exact model
configuration first; do not infer expert dimensions from parameter count alone.
Compare correctness, full-layer latency, memory and overlap against the existing
Qwen3-30B-A3B geometry, with varied token counts/routing, checkpointing and BF16/FP8
compute. This is a generalization gate, not a replacement for the current target.

BF16 master/model weights with FP8 AMP-style compute are a required user-facing
configuration to validate after fusion/overlap. Preserve FP32 master weights with
FSDP/FP8 compute for comparison to Prime. Validate actual BF16 parameter/gradient
storage, FP8 compute/scaling, optimizer updates and FSDP interaction; accepting
BF16 inputs or internally casting FP32 weights is not proof of this combination.
Do not silently maintain or require an FP32 master copy for the BF16 case. Report
optimizer-state precision separately from parameter/master-weight precision.

### 2026-11-09 Hopper first numerical pass and readiness profiling

H100 diagnostic v5 completed numerical parity for one rank/137 tokens: retained
projections exact; output relative L2 0.001666. Sanitizer still reports21 CUDA API
errors (cuMemCreate invalid value and missing-image function attribute queries),
so this is not a clean sanitizer pass. No illegal shared-write report appears in
this run. Agent investigating API report attribution and extending reuse/masking.

B200 readiness backward47969 remains live. Added ready_gather metadata to future
training/timing/profile results to distinguish the staged path. Profile74069 runs
at `/private/tmp/ultra-peer-ready-backward-profile.log`. Its interpretation remains
conditional on integrated training correctness. Pinned Ruff passed.

### User requirement: NVFP4 FFT and merge-aware LoRA

NVFP4 training and NVFP4 merge-aware LoRA are required scope alongside BF16/FP8.
Keep fusion/communication-overlap stabilization as the immediate priority; do not
lose this requirement when completing dtype or adapter support.

Validate NVFP4 full fine-tuning with explicit parameter/master, compute, scaling,
gradient and optimizer-state precision. Determine supported GPU architectures
from the actual kernel path; do not infer SM120 support from SM100 support.
For merge-aware LoRA, establish reference semantics for base-plus-adapter weights,
quantization/scaling and merge/requantization timing, then verify adapter/input
and any trainable base-weight gradients, checkpoint replay, FSDP/EP and compile.
Evaluate epilogue fusion for applicable adapter accumulation/scaling/activation
steps; do not assume a post-GEMM low-rank addition is equivalent to quantizing
merged weights, or that an epilogue alone performs the low-rank GEMMs. Preserve
router assignments across checkpoint replay. No NVFP4 or merge-aware LoRA support
is claimed from existing BF16/MXFP8 standalone tests.

### 2026-11-09 integrated readiness training passed

Readiness backward47969 passed fullgraph checkpointed BF16 training, 48 kernel
tests and real-peer concurrent consumer validation. Saved
`peer_ready_backward_training.json`. Three optimizer steps retain prior reference
agreement. Profile74069 remains active; unprofiled eight-B200 timing32112 runs at
`/private/tmp/ultra-peer-ready-backward-timing.log`. No overlap/speed claim yet.

Two-H100 validation68340 is running at
`/private/tmp/ultra-hopper-persistent-pair-v6.log`, with three workspace reuses and
masked routes in the middle iteration. Single-H100 initial evidence saved by
agent in `h100_persistent_bf16_single_initial.json`; its21 sanitizer API reports
remain explicitly recorded. Agent attributes them to capability/function-image
probes; do not turn that attribution into a clean sanitizer claim.

Profile74069 completed in the last poll (supersedes preceding active status).
Saved `b200_peer_ready_backward_profile.json` and
`b200_peer_ready_overlap_summary.json`. Gather kernel lifetime overlaps dH kernel
lifetime on all ranks (~14.2–14.7ms), but dH includes readiness polling. This is
not proof of useful FMA/gather overlap. Producer service time is much longer than
the staged gather, suggesting resource reservation/scheduling needs work. Await
unprofiled32112 before selecting the path for performance. Preserve staged baseline.

### User-ordered dtype/adapter rollout

After fusion and communication overlap stabilize, the explicit priority order is
(reordered by the user on 2026-11-09: NVFP4 + LoRA ahead of FP8 + LoRA):
1. NVFP4 LoRA, including the previously requested merge-aware semantics.
2. NVFP4 full fine-tuning.
3. BF16 weights + FP8 AMP-style compute + LoRA.

This ordering supersedes the earlier FP8-LoRA-first order and any implied
FFT-before-LoRA sequence for this rollout. Preserve FP32-master Prime comparisons and larger Qwen397B final shape
validation, but do not let those displace this user-facing order. Validate actual
parameter/gradient precision, adapter and input gradients, quantization/merge
semantics, checkpoint replay, EP/FSDP and compile for each applicable stage.

### 2026-11-09 readiness regression and resource rebalance

Timing32112 passed but regressed: F2.2860, B27.5547, F+B29.5562,
AC31.8014ms versus staged AC17.8192. Saved
`b200_peer_ready_backward_timing.json`. Readiness is not a performance recommendation.
Changed producer workers/expert8->32 and dH consumer reservation from GPU-minus16
SMs to at most32SMs, leaving substantially more producer capacity. Validation86549
and timing76488 run at `/private/tmp/ultra-peer-ready-balanced-validation.log` and
`/private/tmp/ultra-peer-ready-balanced-timing.log`. No improvement claimed yet.

Added benchmark-only --compare-ready for alternating staged/ready cases on the
same allocation (five warmups,15measurements), independent of fusion comparison.
Not run yet. Also skip readiness dH when activation fusion is disabled, avoiding
an unused extra GEMM in fusion A/B runs. Pinned Ruff passed.
Two-H10068340 passed all six rank/step numerical checks with reuse and masking;
agent preserving deduplicated evidence and preparing larger ring-reuse cases.

### 2026-11-09 finer readiness chunks

Rebalanced timing76488 returned F2.2870/B17.4787/F+B19.4866/AC21.7416ms,
still slower than staged AC17.8192. Saved `b200_peer_ready_balanced_timing.json`.
Paired staged/ready benchmark16978 runs at
`/private/tmp/ultra-peer-ready-paired-timing.log` on one allocation, before the
following chunk subdivision change.

Readiness descriptors now optionally split each expert into aligned chunks using
GPU tensor arithmetic. Experimental backward uses eight chunks/expert and eight
producer workers/chunk, retaining at most32SMs for dH. This targets whole-expert
publication delay; no CPU split reads or NCCL payload traffic introduced.
Validation50235: `/private/tmp/ultra-peer-ready-chunks-validation.log`. Earlier
whole-expert results remain intact; no numerical or speed claim for subdivision.
Hopper ring-wrap/imbalanced-receive test7051 is running at
`/private/tmp/ultra-hopper-persistent-ring-wrap.log`.

Balanced validation86549 passed; saved `peer_ready_balanced_training.json`.

### 2026-11-09 paired scheduling result and safe default

Paired16978 passed: staged B13.5698/F+B15.5691/AC17.7959ms versus whole-expert
ready B17.4515/F+B19.4946/AC21.7396ms on the same allocation. Saved
`b200_peer_ready_paired_timing.json`. Restored staged gather as PeerBackward's
default; validation explicitly opts into readiness so research coverage remains.
This preserves the faster no-NCCL/no-host-split baseline, not an abandonment of
readiness or persistent fusion. Chunked readiness validation50235 remains live;
paired chunked benchmark74817 runs at `/private/tmp/ultra-peer-ready-chunks-paired-timing.log`.

Hopper ring-wrap7051 stopped at geometry guard (240 pool blocks <=285 ring blocks),
not a numerical mismatch. Agent seeded all64 receiver-local experts while retaining
hot-expert skew to produce296 blocks. Root rerun log:
`/private/tmp/ultra-hopper-persistent-ring-wrap-v2.log`; no wrap pass yet.

### 2026-11-09 direct-peer GEMM scatter primitive started

Added native `PeerScatterStoreBF16` epilogue, reusing the existing scatter register
mapping but selecting output peer pointer and unique token/top-k slot from packed
assignment IDs. This targets removing the packed dX output write/read and separate
return kernel. Existing local scatter specializations are unchanged. Native API
validates pointer-table/assignment/queue metadata and global slot bound. Caller
must provide real symmetric mappings and unique slots and retain their lifetime.
This primitive is not yet compiled, GPU validated or integrated; next work is
functional wrapper and numerical/real-peer tests before any runtime enablement.
Vendor local modification manifest updated. Current running Modal jobs use frozen
pre-change native source; their results cannot validate this epilogue.

Chunk validation50235 passed. Paired74817: staged B13.4415/F+B15.4392/AC17.6636ms,
chunk-ready B15.7735/F+B17.7937/AC20.0655ms. Saved
`peer_ready_chunks_training.json` and `b200_peer_ready_chunks_paired_timing.json`.
Chunk publication improves readiness but remains slower; staged stays default.
Next focus is direct epilogue gradient return/full persistent fusion, not treating
kernel lifetime overlap as success.
Hopper28095 passed ring-wrap guard296>285 and output parity (~0.001662); rank0's
empty-expert projection diagnostic reports0/0 NaNs. Agent correcting reporting;
these do not establish nonempty projection errors or NaN output payloads.

### 2026-11-09 peer scatter wrapper and first GPU build

Added `peer_gemm_out` custom op/fake with explicit mutated peer-output list and
GPU pointer/assignment metadata. Added eager/fullgraph numerical tests covering
H512/H2048, shuffled unique destination slots, masked assignments, empty expert,
and untouched destination sentinel values. These first tests use local tensors
as peer-pointer stand-ins; real symmetric-peer validation is still required.
Scratch allocation remains for native shape metadata; epilogue avoids its stores
but no allocation-elimination claim is made. Added tests to Blackwell CI lane.
Pinned Ruff passed. Native snapshot:
`/private/tmp/mono-ep-vendor-peer-scatter-v1-20261109` (corrected B200 base plus
four explicit epilogue files, independent of Hopper changes). Build/validation12302:
`/private/tmp/ultra-peer-scatter-v1-validation.log`. No GPU pass yet.
Hopper agent completed bounded port work; shared-expert behavior and performance
remain unvalidated, and the broader integration/backward goals remain active.

### 2026-11-09 real peer scatter validation added

Added real symmetric-output peer scatter checks: BF16 GEMM output goes to unique
source slots across ranks, eager/fullgraph, three reuses, masked routes and an
empty expert-owner rank. A separate reference all-reduce assembles expected values
outside the operation under test; it does not prove collective-free operation by
itself. Runtime scatter still uses pointer stores only. Native library build12302
completed; JIT/numerical tests remain running. Real-peer validation runs at
`/private/tmp/ultra-peer-scatter-real-peer-validation.log`. Pinned Ruff passed.
No backward integration or GPU correctness claim for scatter yet.

### 2026-11-09 direct scatter local tests passed; opt-in backward

Local scatter12302 passed (52 kernel tests plus existing peer/training checks).
Saved `peer_scatter_local_validation.json`. That run predates real peer scatter;
real-peer24575 is still live.
Added opt-in `PeerBackward(..., fused_peer_return=True)` for BF16. dX GEMM now
writes unique peer slots directly through the native epilogue; the existing return
kernel gains a score-only specialization when dx is None. Existing path remains
default, and MXFP8 does not claim fused return. Added grouped backward output-GEMM
hook and explicit training metadata. Pointer mappings are runtime-owned and peer
barrier still joins completion before source reduction/reuse. Pinned Ruff passed.

Integrated validation98231 at `/private/tmp/ultra-peer-scatter-training-validation.log`
executes primitive tests, then real-peer stores, then fullgraph checkpoint training;
a failure in an earlier gate stops later stages. No integrated pass or speed claim
until results arrive. Shape scratch allocation is still present despite removed
dX payload stores/reads; full persistent backward remains unfinished.

### 2026-11-09 fused peer return validated; isolated timing running

Real-peer24575 and integrated98231 completed successfully. Evidence saved in
`benchmarks/peer_scatter_real_peer_validation.json` and
`benchmarks/peer_scatter_training_validation.json`. The latter passed 52 primitive
tests, real peer-memory scatter, and BF16 fullgraph checkpointed training across
three optimizer steps with two outstanding forwards and reverse backward order.
This validates the opt-in dX GEMM peer-store epilogue, not full persistent backward
or Axolotl integration. Training validation used readiness gathering; timing holds
staged gathering fixed to isolate return fusion.

Added benchmark-only `--compare-peer-return`, mutually exclusive with readiness,
activation-fusion comparisons and profiling. Both cases share one initialized
workspace, alternate order, discard five warmups, and measure fifteen iterations
using maximum rank latency. Eight-B200 Qwen3-30B-A3B geometry remains T16384/rank,
H2048, I768, E128, top8, FP32 weights/BF16 compute, excluding router and optimizer.
Session40136 is running at `/private/tmp/ultra-peer-scatter-paired-timing.log` with
frozen native snapshot `/private/tmp/mono-ep-vendor-peer-scatter-v1-20261109`.
No speedup claim until terminal evidence. Readiness rebalancing remains parked.

### 2026-11-09 direct GEMM peer stores did not improve whole-layer timing

Session40136 completed successfully; evidence:
`benchmarks/b200_peer_scatter_paired_timing.json`. Paired medians (ms), separate
return versus peer epilogue: forward 2.2917 / 2.3024, backward 13.5190 / 13.8711,
recomputed F+B 15.5193 / 15.8428, checkpointed 2F+B 17.7687 / 18.0568.
Thus direct dX peer-store fusion regressed backward ~2.6% and checkpointed layer
~1.6% for this geometry; keep it opt-in and preserve the separate-return default.
Staged gather was held fixed. Removing a separate kernel is not itself a latency
win. The peer epilogue puts remote stores on the GEMM stream, while the baseline
return overlaps dW on a communication stream; this is a candidate explanation,
not a profile-proven cause. Full persistent backward, bounded tile pipelines and
Axolotl integration remain open; do not resume readiness resource rebalancing
based on this result.

### 2026-11-09 scheduler direction and NCCL coexistence

User feedback confirms dynamic M-block scheduling as the next architectural work,
not further static source-rank or role partition rebalancing. Current vendored
`MonoEPScheduler` uses atomic task counters and locality-domain work stealing;
reuse/adapt its dependency and ring-lifetime machinery for dH (N=I), z recompute
(N=2I) and dX (N=H). Dynamic CTA scheduling within a GPU does not redistribute
expert ownership between GPUs; hot expert owners can still be bottlenecks.
Validate concentrated routing, empty owners and ring wrap without deadlock.

Prime scaling docs checked 2026-11-09 specify 16 reserved SMs by default, with
NCCL_MAX_CTAS <= reserved_sms // 2 for FSDP and // 3 for HSDP, to permit concurrent
collectives. This is an independent residency/deadlock constraint, not a routing
balance knob or a universal optimal constant. Source:
https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/scaling.md

Important current gap: vendored `get_num_sms_for_mono_ep()` asserts the entire
GPU SM count for locality domains. Do not assume Prime's reservation is already
supported or change the SM count blindly. Adapt locality mapping/grid sizing and
validate concurrent FSDP collectives before enabling the persistent runtime in
Axolotl. Backward warp roles, tile geometry and ring depth should become internal
architecture/shape-specific JIT configuration, with a distinct SM90 policy for
register accumulators/WGMMA versus SM100 TMEM/UMMA. Full backward remains open.

### 2026-11-09 dynamic three-phase backward scheduling primitive

Added original Apache-2.0 `src/mono_ep/include/backward_schedule.cuh`, packaged
as a JIT header. A free CTA claims an available N tile within a bounded ring of
M blocks. dH, recompute/SwiGLU and dX have independent tile counts. Publication
requires completion of every tile in the prior phase; generation/phase/tile are
claimed with one tagged CAS so delayed claimants cannot consume recycled slots.
No CTA is statically assigned a source rank or producer/consumer role. This
conservative phase barrier can later become finer-grained K-tile readiness.

Added CUDA stress tests with out-of-order service delays, 96 competing CTAs,
unequal phase widths, empty work, one-slot repeated reuse and partial rings.
They check exactly-once execution and acquire-visible generation-tagged payloads.
Registered tests in Blackwell CI and the existing Modal regression runner; pinned
Ruff passed. Validation92533 runs at
`/private/tmp/ultra-dynamic-backward-scheduler-validation.log`. No GPU pass claim
until terminal. This is scheduler infrastructure, not connected GEMMs: full
warp-specialized backward, peer tile dispatch/combine, and dW operand lifetime
integration remain necessary. Executor must retain dW operands outside recycled
ring storage until their consumers finish.

Dynamic scheduler validation92533 completed successfully: all 12 new device
scheduler stress cases passed, with 64 total primitive tests, existing real-peer
checks, and existing fullgraph checkpointed training regressions. Saved
`benchmarks/dynamic_backward_scheduler_validation.json`. The training regression
still uses the existing grouped-GEMM path; it does not exercise the new scheduler
with GEMMs. Next required work is the task-to-expert descriptor bridge and fused
phase executor, including dispatch publication and safe dW operand retention.

### 2026-11-09 merged distributed-environment hook contract

Verified https://github.com/axolotl-ai-cloud/axolotl/pull/4284 is MERGED at
2026-11-09T14:08:07Z, merge commit f914504565692d24f22e2badfec31823af142770.
Read the actual diff. BasePlugin.get_distributed_env(cfg) returns a string mapping;
PluginManager merges it and rejects conflicting plugin values. The hook executes
before distributed initialization, including DeepSpeed, and must not touch CUDA
or torch.distributed (it also executes on GPU-less Ray drivers). Application uses
setdefault, preserving user values; Ray forwards effective values before workers
initialize, with explicit limitations for an already initialized Ray driver and
job-supplied runtime_env.

MonoEP should supply its NCCL_MAX_CTAS default through this hook only for the
persistent backend. Validate effective user/worker values against the actual SM
reservation separately; setdefault alone does not enforce the residency budget.
Late-environment warnings do not prove an existing communicator adopted the cap.
The dependency is not present in this worktree yet; preserve local work when
bringing it in. Native partial-SM support and concurrent-collective validation
remain required independently of this merged hook.

### 2026-11-09 dependency applied and expert tile bridge

Applied PR4284's reviewed six-file patch to this worktree, adjusting only a patch
context line for the older unannotated do_cli kwargs signature. Existing modified
files were preserved. No separate environment-hook implementation was invented.
Initial CPU test attempts hit local environment issues (missing pytest in Python
3.12 environments; the installed Python3.11 pytest plugins require a sandboxed
socket, and repository conftest assumes sys.monitoring). The focused self-contained
hook tests are being run with plugin autoload and conftest disabled, without
changing repository test infrastructure.

Added `resolve_backward_tile` to the original dynamic scheduler header: GPU prefix
ends map a claimed M block to expert, packed row and valid row count, independently
of expert alignment padding. CUDA tests cover empty expert prefixes, uneven tails,
all-empty work, concentrated ownership and out-of-range sentinel blocks. Mapping
metadata stays on GPU. This still needs connection to the fused GEMM executor.
Validation49268 runs at `/private/tmp/ultra-dynamic-backward-mapping-validation.log`
with the same frozen native dependency; pinned Ruff passed for the new tests.

PR4284 dependency validation: 14 focused CPU tests passed on local Python3.11
with pytest plugin autoload/conftest disabled and the repository's log-propagation
behavior restored by a temporary pytest plugin. The first isolated attempt had
13 passes and one caplog failure because that conftest fixture was absent; the
warning itself was emitted. No production/test source change was needed. Pinned
Ruff passed for all imported Python files. Upstream six-file changes are applied
as working-tree edits, not yet represented by a merge/cherry-pick in history.

Mapping validation49268 passed: 67 primitive tests (12 scheduler, 3 expert-mapping, 52 existing), real-peer tests and existing fullgraph checkpointed training. Saved `benchmarks/dynamic_backward_mapping_validation.json`. Scheduler and mapping remain separately tested primitives, not yet connected to GEMMs. Hopper follow-up delegated to the existing authorized port agent.

### 2026-11-09 correction: M-block ownership is the primary backward design

User feedback correctly challenges the unmeasured need for cross-CTA phase
handoffs. No register/TMEM occupancy measurements justify choosing three global
phases. Park the generation-tagged three-phase scheduler as an alternative; retain
its tests/evidence, but do not connect it as the default fused backward executor.

Primary design: a CTA/pair dynamically acquires a routed M-block and owns it until
dX/combine finishes. Phase A loops over intermediate-column tiles: compute
unweighted dH_j, recompute interleaved gate/up_j, apply SwiGLU backward immediately
and accumulate dscore. Keep dH_j local, with no full global dH workspace. Write
weighted h and dz to stable per-expert workspace for later dW2/dW1. Phase B loops
over output tiles, streams dz, computes dX and returns unique source slots.
Dispatch staging uses CTA/pair-owned full/empty buffer synchronization. Global
h/dz lifetime extends past kernel completion through the weight-gradient GEMMs;
it must not be mistaken for reusable dispatch scratch.

At BLOCK_M128/I768, BF16 dz is 393216 bytes (384 KiB), so shared-memory residency
of the whole dz block is not assumed. Keep rounding and score semantics identical
to the existing recompute epilogue: dscore uses unweighted silu(gate)*up and
unweighted dH; saved h for dW2 is score-weighted. Measure actual TMEM/register
pressure before considering any cross-CTA phase split.

Two evidence qualifications: pinned DeepGEMM forward get_next_task() actually
issues separate Linear1/Linear2 N-cluster tasks with dependency tracking, so the
owned-block backward is a deliberate alternative, not a literal copy of forward.
Prime's reported timings cannot prove whether an internal dH round trip exists.
Neither qualification is a reason to retain unnecessary backward handoffs.

An SM100 tile-math helper was started in include/sm100_backward_tile.cuh using
DeepGEMM UMMA/TMEM primitives. It is uncompiled/unvalidated, not wired into runtime,
and may be adapted for the owned-block executor; no performance/correctness claim.

### 2026-11-09 owned M-block SM100 math executor implemented

Added `include/sm100_owned_backward.cuh`, using the DeepGEMM UMMA/TMEM tile
helper. A CTA claims a 64-row M block, computes column-local dH/gate/up and
SwiGLU derivatives, persists weighted h/dZ for dW, then computes all dX tiles
before acquiring another M block. There is no global dH tensor or cross-CTA
phase queue. FP32 score-gradient accumulation and BF16 projection rounding
match the existing reference contract.

This first executor consumes packed x/dy and emits packed dX; peer dispatch and
combine, warp-specialized asynchronous staging, 2-CTA/128-row configuration,
FP8 and integration into autograd remain open. It uses serial 64-wide K stages
and a one-CTA UMMA path to validate the owned-block math; it is not a throughput
claim or a replacement for the final persistent EP design.

Added GPU numerical tests for irregular and concentrated expert token counts,
empty experts, and H2048/I768 geometry, plus dedicated single-B200 runner for
subsequent targeted iterations. Initial compilation/numerical validation53567
runs in the existing regression harness at
`/private/tmp/ultra-owned-backward-validation.log`; no pass claim yet. Headers
are original Apache-2.0/Copyright2026 Axolotl AI and remain package data.

### User reporting requirement: regular timing deltas

For each completed performance comparison report absolute forward/backward/F+B/AC
latency, percentage change versus the same-run control, and whether it improves
the best validated matching configuration. Preserve geometry, hardware, precision,
router scope and warmup/aggregation metadata. Separate correctness updates from
speed claims; do not imply new speedups while debugging. Current paired control
is B200x8 Qwen3-30B-A3B BF16/FP32-master staged peer F2.2917, B13.5190,
F+B15.5193, AC17.7687ms (router excluded). Fused peer-store comparison regressed
B2.6%/AC1.6%, so remains opt-in.

### 2026-11-09 owned backward initial timeout and convergence fix

Validation53567 terminated with a 300-second kernel-test subprocess timeout; no
correctness pass. Inspection of the owned UMMA helper found the completion call
inside a thread0-only branch, while CUTLASS umma_arrive internally elects a lane.
Moved the call to the converged first warp and added bounded try_wait diagnostics.
This is a candidate cause until verified, not a proven fix yet. Dedicated one-B200
validation80095 is running at
`/private/tmp/ultra-owned-backward-warp-commit-validation.log`. No new timing data.

Owned validation80095 finished in 10.51s after the warp-convergence correction,
so the initial stall did not recur. First numerical case failed h relative-L2
0.9896 against the staged BF16 equations. No correct executor or latency claim.
Added an identity/progression tile-layout diagnostic; validation45604 runs at
`/private/tmp/ultra-owned-tile-layout-validation.log` to isolate TMEM/shared layout
from backward math. Existing production path remains unchanged.

Owned tile diagnostic45604 initially failed to compile because the probe tried
to default-construct CUTLASS barrier storage; changed only the probe to raw
aligned shared storage. Diagnostic25433 then exposed M64 TMEM row mapping: only
the first16 datapaths per warp contain output, not all32. Corrected extraction
across four warps. Validation98169 PASSED all four focused tests: exact tile
layout plus three owned-backward numerical cases, including H2048/I768. Saved
`benchmarks/b200_owned_backward_initial_validation.json`. This proves the first
packed-input owned-M-block math executor against its BF16 reference, not peer
communication, autograd integration, dW output, or throughput. No timing change
reported; next is interleaved gate/up issue and peer dispatch/combine integration.

### 2026-11-09 interleaved owned recompute and custom-op boundary

Interleaved gate/up recompute now uses one 64x128 UMMA tile per 64 intermediate
columns; dH and dX keep 64x64 tiles. This removes the separate up-projection pass
over x. Shared h/dZ score semantics remain unchanged. Validation36917 passed all
four numerical cases; evidence `b200_owned_backward_interleaved_validation.json`.
The original validated headers are preserved at
`/private/tmp/ultra-owned-backward-initial-validated`. No kernel timing collected.

Added production JIT binding and `mono_ep::owned_backward` custom-op/fake
boundary. Shapes/dtypes/devices are checked; expert counts/prefixes stay on GPU,
with async bounds/nonoverlap assertions. Outputs are weighted h, dZ, dX, dscore;
the operation remains standalone and is not yet wired into training backward.
Header/C++/CUDA sources are packaged for JIT. Eager/fullgraph numerical validation
22915 runs at `/private/tmp/ultra-owned-custom-op-validation.log`. Initial Ruff
found one unused test-local ends variable from the test-to-production transition;
removed it without changing kernel behavior.

Custom-op validation22915 failed in the test wrapper before kernel invocation
because it passed Python reference counts instead of counts_gpu. Corrected the
test argument. Validation5796 PASSED all seven cases: exact UMMA tile layout plus
three owned-backward shapes in eager and fullgraph compilation. Saved
`benchmarks/b200_owned_backward_custom_op_validation.json`. This is standalone
backward-math compile validation; full-layer/autograd integration and timings
remain outstanding. No change to the 17.7687ms eight-B200 checkpointed baseline.

### 2026-11-09 owned-block training validation and progress reporting

Owned-block backward is now connected to the experimental peer training path.
Modal app ap-Vtsat3TlWJ02ycJ1TDE8W9 completed successfully: 74 kernel tests
passed, plus compiled/checkpointed BF16 distributed training validation with
multiple outstanding forwards and reverse backward order. Evidence is saved in
`benchmarks/b200_owned_backward_training_validation.json`. Peer gather and return
remain separate operations; in-kernel communication and asynchronous pipelining
are still outstanding. This is a correctness milestone, not a measured speedup.

Report performance regularly as work proceeds: absolute milliseconds, percent
change against the same-run matching control, and whether the best validated
matching configuration improved. Report forward, backward, full F+B, and
checkpointed full layer separately. Include GPU count/type, dtype, model shapes,
routing settings, router inclusion, warmup and sampling policy. Keep correctness
progress separate from performance; pytest duration is not kernel timing.
Current comparison reference: 8 B200, EP8, T16384/rank, H2048, I768, E128, top8,
FP32 master/BF16 compute, router excluded, 5 warmup/15 samples, max-rank median:
F 2.291675 ms, B 13.519014 ms, F+B 15.519291 ms, checkpointed 17.768690 ms.
Owned-block paired timings are pending; no validated improvement claimed yet.

### 2026-11-09 owned-block paired timing and operand pipeline

Added `--compare-owned` to the eight-B200 training benchmark. It alternates
owned-block and grouped backward with the same staged peer gather and separate
return; JIT loads before timing, with five warmups and fifteen measured samples
per case. Forward, backward, F+B and checkpointed F+B are timed separately.
Initial serial-stage comparison is running in Modal, local session 5722, log
`/private/tmp/ultra-owned-paired-timing.log`.

Preserved the serial helper at `/private/tmp/ultra-owned-serial-baseline/`.
The next candidate doubles shared operand buffers and fills the next K tile
while UMMA consumes the current tile. This remains cooperative scalar staging,
not TMA or warp-specialized peer dispatch. Focused eager/fullgraph validation
is running in session 25802, log
`/private/tmp/ultra-owned-double-buffer-validation.log`. No performance claim
or default promotion pending validation and paired timings.

Double-buffer focused validation25802 passed all seven eager/fullgraph numerical
cases. Evidence: `benchmarks/b200_owned_backward_double_buffer_validation.json`.
Full compiled/checkpointed training validation is now running; log
`/private/tmp/ultra-owned-double-buffer-training.log`. Serial paired benchmark
container ta-01M70CB2TVMKGR1YHKYAJT6NNR was confirmed live with all eight GPUs
96–100% utilized. No benchmark result yet.

Serial paired timing5722 completed successfully. Evidence:
`benchmarks/b200_owned_backward_serial_paired_timing.json`. Same-run grouped vs
owned: forward 2.2733/2.2819 ms; backward 13.5519/350.1550 ms; F+B
15.5054/351.8106 ms; checkpointed 17.7510/354.3350 ms. The owned prototype is
~20x slower for checkpointed layer, not a speedup. It remains opt-in. This large
gap makes efficient operand movement and MMA pipelining prerequisites for
performance promotion; no small scheduler tuning or communication claim can
stand in for that work. Double-buffer validation is still pending at training
scope.

Double-buffer full training validation is live in session35845. Its paired
8-B200 timing was launched after focused numerical validation, in parallel
with the full training checks; log
`/private/tmp/ultra-owned-double-buffer-paired-timing.log`. Timing is not grounds
for promotion if full training validation fails. Next operand-movement issue
to examine is noncoalesced scalar B loads for transposed dH/dX weights; current
loaders walk K across lanes even though those weight matrices are contiguous
in N. This is code inspection, not yet profiler attribution.

### 2026-11-09 persistent-port priority correction

User explicitly directs porting the validated owned math into the vendored
SM100 persistent kernel structure next. Stop optimizing the standalone M64
cpp_extension prototype; retain it as a numerical/TMEM mapping reference.
Inherit dispatch warps, TMA/2-CTA UMMA operand pipelines, ring lifetime and
combine epilogue. The persistent backward must hold M-block ownership across
column-local dH/recompute/SwiGLU and dX, with h/dZ workspace for dW. Existing
forward schedules separate N tiles; its scheduler cannot be copied unchanged.
No full persistent-backward implementation or performance claim exists yet.

Added --compare-nccl to the training benchmark and Modal runner. It alternates
recomputing grouped peer backward and fixed-capacity NCCL backward on the same
allocation and same tensors, five warmups/fifteen measurements. Native forward
is identical. This is our grouped NCCL control, not SonicMoE. Session9707 is
running, log `/private/tmp/ultra-recompute-peer-nccl-paired-timing.log`. Existing
17.77 ms peer-only measurements do not prove a win over NCCL or SonicMoE.

Same-allocation NCCL comparison9707 completed successfully, Modal app
ap-jkBajqFHBUWzoSzkjkAQn7. Evidence:
`benchmarks/b200_recompute_peer_nccl_paired_timing.json`. NCCL/peer medians:
F 2.282520/2.288163 ms; B 20.708357/13.644592 ms; F+B
22.629894/15.540381 ms; checkpointed 24.873840/17.782530 ms. Peer reduced
backward 34.1%, F+B 31.3%, checkpointed 28.5% against this same-run
fixed-capacity grouped NCCL control. This is not SonicMoE+NCCL. The peer result
is consistent with the previous 17.768690 ms result, not a new cumulative win.

Started the native epilogue math port in
`vendor/deepgemm/deep_gemm/include/deep_gemm/epilogue/sm100_moe_backward.cuh`.
The atom accepts dH in registers and gate/up in the persistent forward's
16dp256b TMEM register order, produces weighted h and dZ in BF16, and retains
two-row FP32 dscore partials. It has no global dH input or output. This helper
is not yet wired to the native persistent executor or GPU-validated; it does
not constitute a completed persistent-backward kernel. Next work must wire
the actual native TMA/UMMA and dispatch/combine loops, not tune the M64 prototype.
Care is needed in dH versus gate/up tile placement: forward's 2-CTA UMMA uses
256 output columns total, while one recompute tile produces only 128
intermediate columns after SwiGLU. dH register ownership must match those
intermediate columns across the two CTAs, without a cross-CTA global dH spill.

### 2026-11-09 native persistent M-block ownership

Added opt-in `kOwnMBlocks` to the actual SM100 BF16 persistent kernel. Native
`mainloop_owned` claims one M-block from the GPU task counter and emits all
its N tiles to that same CTA pair through the existing metadata pipeline.
The forward path retains native dispatch warps, TMA/2-CTA UMMA and combine.
Default remains unchanged. `examples/modal_persistent_owned.py` overlays the
modified native headers and enables ownership only in the validation container.
Two-B200 compiled/checkpointed three-step training passed with multiple
outstanding forwards in reverse backward order. Initial evidence:
`benchmarks/b200_native_owned_schedule_validation.json`; after backward phase
metadata/epilogue definitions, `benchmarks/b200_native_owned_phase_validation.json`
(app ap-RdpXYJdGDOtCiTGzmcJ9LK). These runs use existing grouped peer backward;
they validate native forward ownership, NOT a persistent backward executor.

The same owned scheduler now defines a backward task stream: for each dH
column cluster, dH then two gate/up recompute halves; after all intermediate
columns, dX tiles. These are local GEMM substeps on one owner, not separately
claimable global phase tasks. Recompute N-block mapping is
`4*j + 2*cta + half`, while dH uses `2*j + cta`, so each CTA keeps its own
128 intermediate columns across the two 64-column recomputes. This backward
stream is not instantiated by the executor yet.

Native epilogue now includes CTA-local BF16 dH storage (BlockM*128*2 bytes,
32 KiB at BlockM128) plus register-atom conversion for persistent 16dp256b
TMEM loads. A dH atom maps row `atom_row+2*(lane%4)+i%2`, column
`32*warp+lane/4+8*(i/2)`. Recompute reads columns
`64*half+16*warp+lane/4+8*k`. This avoids cross-warp register ownership changes
and a global dH spill. Header compiles via the native forward validation, but
this mapping still needs actual backward/TMEM numerical validation. Shared
storage and stage-count resource budgeting remain to be measured.

Next implementation is a backward-specific native kernel derived from the
existing forward structure (preserve original forward behavior):
- Dispatch x and dy through existing remote TMA pull warps into full pool
  workspaces, retaining both for dW. Separate dy peer SymBuffer; reuse route
  metadata and ring readiness generations. No host split counts.
- Three weight descriptors: W2 transposed for dH, interleaved W1 for recompute,
  W1 transposed for dX. A descriptors select dy pool, x pool, dZ pool.
- Phase-A epilogue caches BF16 dH locally, then emits weighted h and concatenated
  dZ to full pool workspaces. Recompute-low/high use the mapping above.
- dX TMA consumers wait for the complete recompute readiness mask for their
  M-block, not the forward per-K mask: dZ gate/up writes span two K regions.
  Publish readiness only after stores complete. No global dH tensor.
- Reuse dX remote combine epilogue; dZ already carries routing weights.
  dscore needs two unique per-route CTA partials and source-side addition,
  avoiding remote atomic accumulation. Gate combine readiness on both outputs.
- Expose pool x/dy/h/dZ and expert layout for grouped dW after this kernel.
  Allocate extra shared dH storage explicitly and account for it in launch
  stage-count/resource heuristics before enabling this path.
This is incomplete; do not reinterpret ownership validation as fused backward
validation. No new performance result from these changes.

### 2026-11-09 native persistent backward executor written

Implemented `impls/sm100_bf16_mono_ep_backward.cuh` as a derivative of the
existing native forward (MIT copyright retained), plus
`layout/moe_backward.cuh`. It uses native dispatch warps to TMA-pull x and dy
into retained full-pool workspaces, the existing 2-CTA TMA/UMMA pipelines with
phase-specific descriptors, CTA-local BF16 dH, interleaved recompute/activation
backward, global h/dZ for dW, and native dX combine. Source dscore combines two
unique CTA partials per route. Full-mask readiness gates dX loads; expert starts
and counts are exported during cleanup. No host split counts or global dH.
This is unvalidated executor code, not an enabled production backend.

Compile14709 found a const-pointer mismatch in the dispatch byte-offset helper.
Corrected it. Compile30377 passed, producing a 135496-byte cubin for
T16384/H2048/I768/E128/top8/EP8/BM128/BK64/two stages. ptxas reports
128 registers, 112-byte stack frame, zero spill loads/stores, 16 barriers.
Vector-type warnings remain; no sanitizer-clean or performance claim. Evidence:
`benchmarks/b200_native_persistent_backward_compile.json`.

Added a direct native test launcher in `tests/native/persistent_backward_probe.cu`,
`examples/validate_native_persistent_backward.py`, and
`examples/modal_native_persistent_backward.py`. This cpp_extension is a test
launcher for the vendored persistent executor, not the parked M64 kernel. It
uses two B200s, H2048/I768/E16/top4, BM128/BK64/two stages/128 resident CTAs,
retains the actual native ring capacity, and compares dX/dscore/expert counts
against BF16 reference for balanced then rank-skewed routing with masked slots.
It reserves the device's maximum opt-in shared memory for this initial launch;
production launch resource budgeting still needs exact shared-size accounting.
Validation2099 is live; log
`/private/tmp/ultra-native-persistent-backward-validation.log`. No duplicate run
until its process is terminal. Native dW/autograd integration, h/dZ validation,
ring-wrap stress, compile integration and timings are still outstanding.

Native validation2099 initially failed in NVCC's host launch stub: the inherited
infinity floating template argument generated an undefined `inf`. Removed
the unused clamp/fast-math template arguments from this backward-only kernel
(unclamped SwiGLU/expf only); no math fallback or alternate executor was used.
Validation25736 PASSED balanced and skewed two-B200 dX/dscore/expert counts.
Maximum relative error: dX 0.00078294, dscore 0.00020826. Evidence:
`benchmarks/b200_native_persistent_backward_validation.json`, app
ap-0j3FYlREqGpw17hWj8F4X0. This executes actual native remote dispatch, TMA,
2-CTA UMMA, local dH, recompute, dX combine and dscore source reduction.

Extended validator matches every retained pool row exactly to source x/dy,
recovers its routing weight, and independently checks h/dZ plus dW1=dZ^T x
and dW2=dy^T h. Validation89639 PASSED both routes. Maximum dW1 relative
error 0.00034246, dW2 0.00027743. Evidence:
`benchmarks/b200_native_persistent_backward_weight_validation.json`. These
weight contractions use PyTorch in the validator; grouped production dW and
autograd integration are not yet wired to the native executor.

Native ring stress12055 is live, log
`/private/tmp/ultra-native-persistent-backward-ring-validation.log`. Skew case
now uses 1800/1813 source tokens and routes all tokens to four rank-zero
experts; actual capacity is1920/rank and ring13440 rows. Validator asserts the
routed padded pool exceeds ring capacity, then checks dX/dscore/h/dZ/dW and
expert metadata. This also gives more M-blocks than the64 active CTA pairs.
No timing claim yet. Next after correctness: native API/host descriptor and
resource-budget launcher, runtime custom-op/fake/autograd integration, then
paired full-layer comparison against grouped peer and fixed-capacity NCCL.

Ring stress12055 PASSED. Rank0 used14336 padded pool rows against13440 ring
rows (112 M-blocks against64 CTA pairs); rank1 owned no expert tokens. Balanced
and skewed dX/dscore/h/dZ/dW checks passed, with source x/dy exact and expert
counts exact. Evidence:
`benchmarks/b200_native_persistent_backward_ring_validation.json`. No running
root Modal jobs remain from this validation sequence. The next work is the
production native launcher/runtime integration; do not return to standalone
M64 tuning or scheduler-only experiments.

### 2026-11-09 native launcher and runtime integration

Added native `_C.bf16_mono_ep_backward` binding with tensor/device/dtype/shape,
symmetric-local-pointer, geometry and workspace-capacity checks. Host launcher
builds TMA descriptors for x/dy/dZ and interleaved/transposed weights, uses the
existing DeepGEMM JIT, and calculates shared allocation for two operand stages
plus33KiB of local dH/dscore storage. This replaces the validation-only raw
launcher. SM count currently follows the existing native forward setting;
concurrent FSDP/NCCL SM reservation remains unimplemented here.

`src/mono_ep/persistent_backward.py::NativePersistentBackward` now stages
invocation-saved inputs into its own symmetric buffer and calls the native
executor, then uses existing grouped BF16 dW operations with GPU-resident
expert end offsets. It reuses the existing runtime backward custom-op/fake/
autograd boundary. Runtime integration is opt-in via the internal backend
object; BF16/recompute/single-host/SM100/H,I%256/grouped alignment128 only.
An event prevents cross-stream scratch reuse before dW completes. Native h/dZ
epilogue now writes zero for padded rows so reused scratch cannot contaminate
grouped weight-gradient tails. x/dy pools start zeroed; no whole-pool clears
are performed per backward call.

Snapshot `/private/tmp/mono-ep-vendor-native-backward-v1-20261109` contains
coherent current csrc/native headers. Validation25081 is actively building the
new extension, then runs the existing kernel regression suite and compiled/
checkpointed training with two outstanding forwards and three optimizer steps.
Log `/private/tmp/ultra-native-persistent-training-validation.log`, Modal app
ap-0yCwsn5FclXhWMFSGCiLpN. Command adds --native-backward to the training
validation runner. Do not duplicate a live build/run.

Prepared --compare-native in the eight-B200 benchmark/runner. It alternates
three same-allocation controls: fixed-capacity grouped NCCL, grouped peer, and
native persistent backward plus grouped dW. Same inputs/weights, five warmups
and fifteen measured iterations; scopes F, B, F+B and checkpointed layer.
Timing has NOT been launched; wait for native training validation first.

### 2026-11-09 native production training validation passed

Validation25081 completed successfully: 74 kernel regressions, exact peer
packing, and three AdamW steps on two B200s with two outstanding forwards
backpropagated in reverse order, fullgraph compilation and non-reentrant
checkpoint replay. Production grouped dW is exercised. Maximum reported
input-gradient relative error 0.0034891, score-gradient error 0.0010073,
weight-gradient error 0.0018621, updated-weight error 0.00011036. Evidence:
`benchmarks/b200_native_persistent_training_validation.json`. This is
correctness evidence; it does not establish a speedup.

The eight-B200 native/grouped-peer/fixed-capacity-NCCL comparison is running
as session6265, log `/private/tmp/ultra-native-three-way-timing.log`. Same
allocation, inputs, weights, five warmups and fifteen samples per scope.
The NCCL control is our fixed-capacity grouped implementation, not SonicMoE.
Native transport metadata now explicitly identifies all three controls.

### 2026-11-09 native persistent backward first matched timing passed

Session6265 completed successfully, Modal app ap-K9TXTyDQToou8AiyzNwLOD.
Evidence: `benchmarks/b200_native_peer_nccl_paired_timing.json`. Eight B200s,
EP8, 16384 tokens/rank, H2048/I768/E128/top8, FP32 master weights and BF16
compute; five warmups and fifteen alternating samples, max-rank host-wall
medians. Router and optimizer excluded. Grouped dW and input/weight staging
are included. Timing is eager autograd; fullgraph correctness was validated
separately, not timed here.

| Scope (ms) | Fixed-capacity NCCL | Grouped peer | Native persistent |
| --- | ---: | ---: | ---: |
| Forward | 2.2831 | 2.2786 | 2.3002 |
| Backward | 20.6712 | 13.4861 | 6.1952 |
| Recomputing F+B | 22.6404 | 15.4402 | 8.2821 |
| Checkpointed 2F+B | 24.8728 | 17.6680 | 10.5351 |

Native lowers backward latency 54.1% versus same-run peer and 70.0% versus
NCCL; checkpointed layer latency falls 40.4% and 57.6%, respectively. This
is a new best validated matching native result. Forward is unchanged in
architecture, so its minor variation is not a forward speedup claim. NCCL
is our fixed-capacity grouped transport, not SonicMoE. Prime H200/FP8
figures remain an unmatched hardware/dtype comparison.

Owned M64 executor stays parked as a numerical reference. Native backward
remains internal opt-in, BF16/recompute/SM100/single-host. Next integration
work: audited backend selection, reserved SM handling for FSDP overlap,
further shape/routing stress, Hopper port, then FP8 and adapter sequence.

### 2026-11-09 explicit persistent runtime selection

Added `PersistentMoE(..., backward_backend="native")`; `nccl` remains the
reference default (the former class alias has since been removed). Native selection rejects
FP8 or retained activations before allocation; the backend still enforces
SM100/single-host/shape/alignment restrictions. Existing training validator
now uses this constructor instead of attaching its private backend manually.
Fallback-only imports are deferred so CPU configuration checks do not require
Triton. Fourteen CPU selection/entry-point tests pass, pinned Ruff passes;
new selection tests are included in the CUDA-kernels lane's CPU test command.
The constructor change has not yet been GPU revalidated. Kernel code did not
change. README now documents the native API and matched timing scope instead
of claiming all persistent backward remains unfused. Common MonoEP and
Axolotl FP8 entry points still use the NCCL pipeline.

Hopper agent compile upload was rejected by automatic approval review as
private headers/harness uploaded to Modal without payload-specific approval.
User approval question is pending; no retry or alternate upload mechanism
is authorized by elapsed time. Local Hopper review continues independently.

### 2026-11-09 common MonoEP persistent selection

The previous goal turn made concrete progress: explicit PersistentMoE native
selection, CPU validation and CI wiring. This turn connects it to the common
MonoEP interface using backend="persistent" and explicit max_tokens.
BF16/recompute/chunks1 are required; compile_local is rejected for this mode.
`prepare_persistent` collectively creates and caches runtimes by process group,
device and geometry before tracing. Compatible layers can share one engine;
`close` releases its runtime handles after outstanding graphs complete.
Default pipeline behavior and the Axolotl FP8 adapter remain unchanged.

Twenty-three CPU tests pass across persistent selection and common entry-point
configuration/cache/lifecycle checks. Pinned Ruff passes. The native GPU training
validator now prepares its runtime through the common engine, but this revised
entry path has not been GPU validated; existing native GPU evidence predates
these Python selection changes. No kernel or timing changes claimed.

Hopper static review found and fixed activation TMA box height64 vs expected128.
The isolated SM90 backward and compile/parity harnesses remain unregistered
and GPU-unvalidated. Payload-specific Modal approval remains pending.

### 2026-11-09 SM reservation audit and expanded training validation

The prior turn made progress on common runtime selection. Source audit found
`get_num_sms_for_mono_ep` explicitly asserts runtime SM count equals physical
SM count, because forward uses physical locality domains. Small task queues
disable cross-domain stealing. A global `set_num_sms(total-16)` is therefore
not a supported reservation mechanism and was not applied. Proper reservation
requires coordinated launch/scheduler/ring-capacity changes and GPU validation;
no FSDP-overlap completion claim is made. Native backward's owned scheduler
does not by itself remove the forward restriction.

Training validator now accepts `--master-dtype bf16|float32` and Modal runner
accepts `--world-size 2|4|8`. Four experts per rank preserve the single-owner
skew case at every world size; previously fixed E8/top4 was only valid at EP2.
Reference and local parameters use the same selected master dtype, including
optimizer updates. Reports include world/master dtype and full shape metadata.
The existing compiled/checkpointed path still exercises reverse backward over
two outstanding forwards for three AdamW steps through the public preparation
API. Planned commands add `--master-dtype bf16 --world-size 8` to the existing
`--training --recompute --peer-backward --native-backward --compiled --checkpointed`
validation. Neither expanded GPU case has run. Syntax, pinned lint and local
Modal registration for all three resource sizes pass without remote execution.
Modal upload approval remains pending; no upload was attempted this turn.

### 2026-11-09 validation blocker rechecked

No Modal containers are running (read-only container list returned []). Local
host is Darwin with CUDA unavailable and no nvcc. GPU validation therefore
cannot be completed locally. The same payload-specific Modal upload approval
has remained unanswered across more than three continuation turns. Local
selection, cache/lifecycle coverage, expanded validation preparation and Hopper
static review are complete for the current changes; further kernel/integration
changes should wait for their GPU evidence rather than compound unvalidated
work. Goal blocked pending upload approval; original scope remains incomplete.
Resume with Hopper compile/parity and public-entry BF16-master compiled training
validation. Preserve the validated native B200 timing and all existing sources.

### 2026-11-09 Modal source upload explicitly reapproved

User approved uploading MonoEP sources, vendored headers and validation
harnesses to Modal for H100/B200 testing. Launched Hopper compile session24939
(log `/private/tmp/ultra-hopper-native-compile-approved.log`) and public native
entry BF16-master EP8 compiled/checkpointed training session65752 (log
`/private/tmp/ultra-native-bf16-master-ep8-validation.log`). Hopper agent owns
compile monitoring/fixes and subsequent parity; root owns B200 validation.
Do not duplicate live jobs. Previous upload blocker is resolved by explicit
user approval; original goal remains unfinished.

### 2026-11-09 matched Hopper comparison requested

User requests latest native timings on Hopper with Prime-matched dtype/shapes.
Rechecked Prime blog: per-layer setup is ONE node8H200 EP8,T16384/rank,
E128/top8,FP8,random routing,no router,fullgraph,full AC2F+B. H2048/I768
comes from Qwen3-30B-A3B reference configuration. Keep FP32 master comparison
separate from BF16-master validation. H100 BF16 port validation is a gate,
not a matched H200/blockwise-FP8 latency result. Native Hopper blockwise FP8
forward/backward remains missing; report that gap instead of comparing BF16
B200 timings as equivalent. Prime layer warmup/aggregation is not specified
in blog text; their full-step protocol must not be attributed to layer timing.

Added --compiled timing mode to paired B200 training harness/Modal runner.
It compiles the recomputing runtime with fullgraph=True and retains identical
five-warmup/fifteen-measured alternating cases. Previous native timing is
eager; prepared compiled timing has not run. Profiling is kept separate.

Native public-entry BF16-master EP8 validation65752 PASSED, app
ap-CwMsIEp3Sk7G9GJEfJMwvZ. Evidence:
`benchmarks/b200_native_bf16_master_ep8_training_validation.json`. All24
rank/step records present (8ranks x3AdamWsteps), fullgraph+checkpointed,
two outstanding forwards/reverse backward. E32,H512,I512,top4,unequal
source rows and all-to-one-owner skew. Max input/score relative error
0.00302773, weight-gradient error0.000083009, updated-weight error0.000004043.
This validates BF16 masters with BF16 compute, not FP8 AMP. Kernel regression
suite also passed.

Compiled same-allocation B200 native/peer/NCCL timing now session97361, log
`/private/tmp/ultra-native-three-way-compiled-timing.log`; five warmups/fifteen
measurements, reference Qwen shape, FP32 masters/BF16 compute. Do not duplicate.
Hopper compile passed; agent owns live parity session89252/app
ap-EH57sbbKEfEDmZXK5uhEnj.

### 2026-11-09 native Hopper blockwise FP8 backward in validation

SM90 BF16 parity passed (`h100_native_persistent_backward_validation.json`),
including skew/ring wrap. Hopper agent is wiring unique native SM90 API and
fullgraph/checkpointed training; shared files runtime.py/persistent_backward.py
and csrc/apis/mono_ep.hpp are agent-owned during that work.

Root added isolated `sm90_fp8_mono_ep_backward.cuh`, derived from the proven
SM90 persistent BF16 executor. Dispatch/scheduler/combine remain native; math
uses FP8 WGMMA with K128 and per-block FP32 scale promotion. Weight operands
are 128x128 E4M3; activations are quantized 1x128 inside their owning CTA.
Transport and retained dW pools remain BF16, and dW test contractions are
BF16/FP32: this is NOT yet the complete Prime-matched FP8 training path.
Native FP8 forward and quantized production dW remain outstanding.

Compile79018 passed, artifact `h100_native_fp8_backward_compile.json`:128regs,
88B spill loads/stores,104Bstack; register/WGMMA serialization warnings remain.
Static review found BF16 TMA A layout must be [K/64,row,64], fixed both address
mapping and explicit two64x128 loads before runtime test. Compile artifact
predates this correction. Reviewer found no further concrete FP8-specific
barrier/reuse/scale mapping issue.

Corrected native FP8 parity is live session49283, log
`/private/tmp/ultra-hopper-fp8-native-parity.log`. New probe/validator/Modal
runner names contain hopper_fp8_persistent_backward. Reference explicitly
quantizes interleaved/transposed weights and per-row operands, then uses
FP32 matmul with TF32 disabled; tests balanced and skew/ring-wrap routing.
This job recompiles corrected header. Do not duplicate.

Compiled B200 paired timing session97361/appap-jMUHYlCgZOmRRfhQYxwiLU remains
live with no terminal output; app listing showed ephemeral/0tasks, so treat
as waiting for allocation, not a stopped job.

FP8 backward parity49283 PASSED, appap-FXERGNMTSkKULotaN2dwcJ; evidence
`benchmarks/h100_native_fp8_backward_validation.json`. Max quantized-reference
relative errors: dx0.00532475,dscore0.00129752,h0.00122837,dZ0.00126009,
dW1reference0.00126284,dW2reference0.00123060. Balanced and14336-row
physical ring reuse pass. dW is reference contraction of retained BF16 pools,
not production FP8 dW. Empty-owner zeros are unsampled projections.

Added isolated `sm90_fp8_mono_ep_forward.cuh` by applying the validated
K128 FP8 WGMMA/scale-promotion loop to native SM90 forward. BF16 transport
and activation ring retained; shared experts explicitly rejected. Root
forward compile29021 live, log `/private/tmp/ultra-hopper-fp8-native-forward-compile.log`,
via `modal_compile_hopper_backward.py --fp8 --forward`. Existing BF16 source
is preserved; new FP8 kernels remain unregistered. Astra static review active.
Forward runtime parity, host binding, FP8 dW, compiled training and matched
H200 timing remain outstanding.

FP8 forward compile29021 PASSED, appap-6bjcl6RpM70kQ9JwPSf57h; evidence
`benchmarks/h100_native_fp8_forward_compile.json`:85040-byte cubin,128regs,
68Bspills,96Bstack. Astra found no concrete static blocker, including correct
BK128 waiting for two L1 publications of64 h columns. Runtime parity is NEXT;
no forward numerical or latency claim yet. Compiled B200 benchmark97361 still
confirmed live by write_stdin with no terminal output; do not relaunch.


### Hopper matched-timing gate (2026-11-09)

User requested the latest timings on Hopper with Prime's matched dtypes and
shapes. Target remains one node of 8 H200, EP8, T16384 per rank, H2048,
I768, E128, top8, blockwise FP8, random routing, router excluded,
fullgraph compile; report forward, backward including production dW,
and full-AC 2F+B. Keep same-allocation NCCL control, explicit warmup and
uninstrumented end-to-end timings. H100 BF16 validation and B200 BF16
latencies are not this comparison. BF16 transport in the new FP8 path
must remain disclosed until quantized transport is implemented.

FP8 forward parity94972 failed at launch; diagnostics87303 isolated CUDA400
to cudaFuncSetAttribute (before launch), consistent with kernel registration
rather than operand math. Testing integer bit-pattern activation-clamp NTTP
instead of float NTTP with unchanged numerical clamp: session14178,
log /private/tmp/ultra-hopper-fp8-forward-bits.log. No numerical pass yet.
Hopper BF16 production training is owned by hopper_port, retry94351,
app ap-TuKqsO2Pu3Ci3yHKIvMeNt. B200 compiled paired timing97361 still live.

FP8 dW audit: existing Hopper k-grouped NT API requires host ks_cpu and
rejects prefix-sum layout. Native FP8 training must add a GPU-offset contract
(and capacity-safe packing) rather than silently introducing a host split
read. This is a remaining gate to matched full-layer timings.

FP8 forward retry14178 PASSED on 2H100, app ap-MNYkB0TDLjl7nZSiyErOUE.
Integer-bit activation clamp NTTP resolves host-stub registration failure;
clamp value/math unchanged. Evidence h100_native_fp8_forward_validation.json:
max relative output error0.00518592,projection0.000718202, balanced/skewed/
ring-wrap/reuse checks passed. Forward/backward raw FP8 kernels now both
validated; production host wiring, GPU-offset grouped dW, fullgraph FP8
training and matched8H200 timings remain outstanding. Transport remainsBF16.
Hopper BF16 compiled checkpointed training also passed (both master dtypes),
evidence h100_native_persistent_training.json; dense per-expert dW fallback
is correctness-only and not ready for timing comparisons.


### GPU-offset dW and FP8 binding progress (2026-11-09)

Prior turn was progress: native FP8 forward runtime passed after integer-bit
clamp NTTP change. Current turn adds capacity-safe prefix-end FP8 transpose
packing. Empty groups and unused capacity avoid OOB metadata/input reads;
physical scales index capacity K-blocks. End offsets must be monotonic,
128-padded, and <=capacity; runtime must derive padded ends from native
starts/counts. Twelve independent-reference H100 tests passed eager/fullgraph
with poisoned tails and all-empty experts; evidence
benchmarks/h100_fp8_wgrad_packing_validation.json. Added test to Blackwell CI
lane and standalone Hopper runner. First test caught integer-zero to FP8
Triton conversion restriction; float-zero retry47450 passed.

Added checked sm90_fp8_mono_ep_backward.hpp and registered
_C.fp8_mono_ep_backward_sm90. Static review matches passing raw probe;
registered binding test is live61307, app/log from
/private/tmp/ultra-hopper-fp8-backward-binding.log. Immutable vendor snapshot
/private/tmp/mono-ep-vendor-hopper-fp8-binding-v1-20261109 preserves known
base and only FP8 binding/kernel changes. Do not claim bound-launch parity
until this job passes. Native training FP8 still disabled pending grouped dW
and forward binding. hopper_port owns BF16 GPU-offset grouped dW replacement
of dense fallback; root owns FP8 packing/host wiring. B200 compiled timing
97361 was re-polled live this turn, still no output; no duplicate launched.


### Compiled B200 result and Hopper API validation (2026-11-09)

Compiled B200 paired job97361 FINISHED successfully (app
ap-jMUHYlCgZOmRRfhQYxwiLU); no longer pending. Saved
benchmarks/b200_native_peer_nccl_compiled_paired_timing.json. Same Qwen30B
shape T16384,H2048,I768,E128,top8,EP8, BF16 compute/FP32 masters,
5 warmups,15samples, fullgraph, max-rank synchronized host-wall medians:
native F2.330001,B6.283126,F+B8.370968,AC2F+B10.598459ms.
Peer AC17.798704; fixed-capacity NCCL AC24.860493ms, respectively40.45%
and57.37% latency reductions. Native AC is0.60% above previous eager result.
NCCL control is not SonicMoE; B200 BF16 is not matched Prime H200 FP8.

Added registered native FP8 forward host launcher with BF16 ring views,
FP8 weight/scale validation and optional saved_z. Astra static review found
no concrete blocker. Backward bound build61307 failed on missing scheduler
header dependency; fixed explicit include in both new host headers.
Corrected immutable snapshot hopper-fp8-binding-v2-20261109 is validating
forward31262/log /private/tmp/ultra-hopper-fp8-forward-binding.log and
backward42707/log /private/tmp/ultra-hopper-fp8-backward-binding-v2.log.
Both are distinct tests, do not duplicate. Agent owns BF16 GPU-offset
training38236/app ap-ftRAfel2EMMfu0gSZWtXQ0, source snapshot preserved.
FP8 dW can reuse existing SM90 K-grouped scheduler with padded GPU sizes:
its scale index is already physical current_k_start/BLOCK_K. New host
capacity contract should bypass ks_cpu/scale transform and use MN-major
capacity scales, plus device validation after PDL dependency synchronize.
Implementation and parity still outstanding; no full FP8 training claim.


### Native FP8 bindings passed; grouped dW implementation (2026-11-09)

Forward31262 and backward42707 registered host-launcher tests PASSED2H100;
evidence h100_native_fp8_forward_binding_validation.json and
h100_native_fp8_backward_binding_validation.json. Errors match raw probes.
BF16 transport remains explicit; backward dW checks here are reference
contractions, not grouped FP8 production dW.

Implemented k_grouped_fp8_gemm_nt_gpu_sizes host API and capacity-checked
specialization of existing SM90 FP8 KGroupedContiguous kernel. GPU padded
sizes drive group offsets; physical MN-major capacity SF descriptors bypass
host ks_cpu transforms. Device size/alignment/cumulative-capacity checks
run after PDL dependency acquisition and before scheduler/TMA. Existing
kernel callers default validation=false. New custom op hopper_wgrad.py
packs from GPU padded ends and initializes FP32 accumulation output.
Packing now optionally emits MN-major scales directly, avoiding transpose.
Astra static review found no concrete blocker. New header is MIT-derived.

Validation5847 live /private/tmp/ultra-hopper-fp8-wgrad-gemm.log, immutable
snapshot /private/tmp/mono-ep-vendor-hopper-fp8-wgrad-v1-20261109.
24 packing cases and18 grouped GEMM cases include empty groups, unused
poisoned capacity, eager/fullgraph, and Qwen dW widths. Added grouped test
with gpu/gpu_blackwell/slow markers to gpu_slow_kernels H100 lane. No runtime
parity or performance claim for new grouped FP8 dW until this job passes.
Agent BF16 GPU-offset retry33351 running after fixing _C binding lookup.
Root awaiting file-ownership handoff before persistent runtime FP8 wiring.


### FP8 runtime integration and padding correction (2026-11-09)

Root wired compute_dtype=blockwise into PersistentMoE/native SM90 recompute
and MonoEP(dtype=fp8,backend=persistent). FP32/BF16 masters quantize
from their original values; W1 interleave occurs before quantization.
Transpose weights quantize separately for backward. BF16 dispatch preserved.
Explicitly rejects blockwise SM100/non-native/retention. Local selection/common
API tests28passed. Full GPU FP8 training still UNVALIDATED; prepared runner
examples/modal_hopper_native_fp8_training.py must wait for dW parity.

Integration audit found stale padded x/dy pool values could perturb FP8amax.
Packing now accepts separate valid_ends and zeros padded rows before amax,
while padded_ends determine transpose stride/physical SF index. Native call
passes logical starts+counts plus128-rounded padded ends. Astra review found
no concrete integration blocker for producer-generated metadata.

Initial FP8 dW5847 completed with24packing+6all-empty cases passing and12
nonempty GEMM elementwise FP32-reference checks failing (e.g.maxabs0.01866).
Do not call this a parity pass. Retry6140 compares exact packed operands
against existing host-size Hopper kernel at1e-5 tolerance plus independent
FP32-reference normalized bound0.002; adds poisoned interiorpadding cases.
Now24packing+36GEMM cases. Log /private/tmp/ultra-hopper-fp8-wgrad-gemm-v2.log;
same compiled vendor snapshot, current Python. No duplicated live job.
BF16 groupedGPUoffset fullgraph/checkpointed training meanwhile PASSED both
masterdtypes; artifact h100_gpu_offset_grouped_dw_training.json, agent owns
additional focusedparity34729. Root owns runtime integration edits now.


FP8 grouped dW retry6140 PASSED60cases onH100, app
ap-aXgMBOiEf4GQUzGdR9KpAS; artifact
benchmarks/h100_fp8_grouped_wgrad_validation.json. NewGPU-size kernel matches
existing host-size Hopper kernel at1e-5, independent FP32 norm bound0.002;
poisoned internal padding/unusedcapacity and fullgraph all pass. First
strictFP32 elementwise mismatch is reproduced by existing kernel, not a
GPU-size regression. Launched native FP8 training97661, log
/private/tmp/ultra-hopper-native-fp8-training.log, same immutable FP8wgradv1
vendor + current source/validator. Tests3AdamWsteps, twoforwards/reversebackward,
fullgraph+AC, FP32thenBF16masters. This run is active; do not duplicate.


### Full FP8 training passed; H200 paired timing launched (2026-11-09)

Native FP8 training97661 PASSED app ap-VXlAsUP0gsqWqFfzRI768v, evidence
h100_native_fp8_training_validation.json. Both FP32/BF16masters complete
3AdamWsteps,twooutstandingforwards/reversebackward,fullgraph+AC with4native
forwardcalls/checkpoint replay. Gradients differ around7% from BF16reference
within declared FP8tolerance0.15; separate quantized-reference tests passed.
This is short numerical/optimizer validation, not convergence proof.

Added blockwiseGPU dW to fixed-capacity NCCL control; runtime blockwise now
allows recomputing native/NCCL backends, rejects retention and nonSM90.
Control training82315 PASSED bothmastertypes, app ap-cyMVoTiThobo3HTldqSRvm;
evidence h100_fp8_nccl_control_training_validation.json. Native/BF16 unchanged.
Local selection/commonAPI28passed. Documentation reflects currentstatus.

Matched-shape8H200 timing83123 is LIVE, log
/private/tmp/ultra-h200-native-fp8-paired-timing.log via
examples/modal_hopper_persistent_timing.py, immutable FP8wgradv1vendor.
Shared benchmark supports isolated blockwise native/NCCL pair without
unsupported peer backend. T16384,H2048,I768,E128,top8,EP8,FP32masters,
blockwiseFP8compute,BF16transport,fullgraph,recompute,5warmups/15samples.
Scopes F,B,F+B,checkpoint2F+B; router/optimizer excluded. BOTH cases use
same native persistentforward; only backward differs, so NCCLcontrol is
not Prime's fullNCCLforward baseline or SonicMoE. NoH200latencyresultyet.


### H200 matched-shape FP8 timing completed (2026-11-09)

Run83123 completed successfully, app ap-khRHUxaWC5oOV8XGJLpXT1. Raw samples
and metadata: benchmarks/h200_native_fp8_nccl_compiled_paired_timing.json.
This supersedes the LIVE status above. No timing job remains active.

8H200, EP8, T16384, H2048, I768, E128, top8; FP32 masters, blockwise FP8
compute, BF16 transport, fullgraph, recompute, 5 warmups and 15 samples.
Median synchronized host-wall maximum-rank milliseconds:

| Scope | Native | Fixed-capacity NCCL backward control |
| --- | ---: | ---: |
| Forward | 50.545 | 50.463 |
| Backward | 112.553 | 40.635 |
| F+B | 162.608 | 90.762 |
| Checkpointed 2F+B | 212.970 | 141.226 |

Both cases share the same persistent forward; this is not SonicMoE or
Prime's complete NCCL baseline. Prime reports 3.9/8.2/15.9 ms for F/B/AC;
native is respectively 12.96x/13.73x/13.39x the reported latency. Native
backward is 2.77x this run's NCCL backward, AC 1.51x. The Hopper port is
correctness-validated but not performance-competitive. These measurements
do not identify the bottleneck; profile before attributing the gap solely
to BF16 transport. Prime's per-layer aggregation protocol is unspecified.

Next Hopper gate: separate persistent GEMM/pipeline time, weight quantization,
and grouped dW time without adding instrumentation to headline timings.
Port-agent design review identified a bounded forward-only FP8 L1 transport
step with FP32 K128 scales; keep L2/backward BF16 and original BF16 X/dY
for dW. No transport edits made. Avoid wholesale SM100 with_sf layout reuse.


### H200 bottleneck isolation in progress (2026-11-09)

Previous goal turn was progress: completed paired H200 timing and persisted
the evidence. Extended separate profiler mode to capture compiled forward
and backward, native and fixed-capacity NCCL, after five warmups. Normal
timing path remains uninstrumented. Python compilation and pinned ruff pass.
H200 diagnostic run65503 is active, log
/private/tmp/ultra-h200-native-fp8-profile.log, immutable FP8wgradv1vendor.
Do not duplicate based on delayed captured output.

Port audit found per-N redundant activation quantization: forward x12x/h16x,
backward dy6x/x12x/dz16x. Each K128 waits WGMMA group0 before promotion.
Prior ptxas also reported outlined-call WGMMA serialization and spills.
Hopper agent owns bounded F/B power-of-two reciprocal multiply experiment
and compile inspection; no cache or transport redesign authorized in this
subtask yet. Root owns profiler harness and results.

H200 profile65503 completed successfully, app ap-ZvCgKIAt2D4tnBQoGBtItl.
Evidence benchmarks/h200_native_fp8_nccl_profile.json. No root GPU job live.
Rank0 native persistent forward49.163ms, backward99.270ms; native grouped
dW kernels0.866+0.469ms, with packing kernels~9.7ms. These are profiled
kernel durations, not replacements for uninstrumented headline latencies.
Persistent pipeline is the dominant target, rather than host launch overhead
or dW GEMM speed. Agent reciprocal-multiply codegen experiment still active.
CPU exhaustive FP8 conversion bit parity passed9,400,320 finite BF16/value
scale pairs over144 power-of-two scales; GPU validation remains required.


### Hopper activation reuse implementation (2026-11-09)

User confirmed removing repeated activation quantization before the next
Prime comparison. Root implemented backward caches: dispatch quantizes gathered
X/dY once into local FP8+FP32-scale pools, preserving original BF16 pools for
dW. dH/recompute TMA-load cached values; dX still quantizes BF16 dZ. Cache
publication uses per-writer global async-proxy fence, threadfence and warp
sync before readiness release. Host pull uses largest power-of-two divisor
<=4096, avoiding non-K128 chunks for widths such as4352. Existing 4-pool
host call preserves uncached test control; 8-pool selects cache internally.

Root cache validationv1 app ap-m3AWNRGs3IOYwGX1Zol1Rj was deliberately stopped
after review caught FP8 destination pointer cast/proxy fence issues. Both fixed.
Current backward cache validation15076 LIVE, log
/private/tmp/ultra-hopper-fp8-backward-cache-validation-v2.log, immutable
/private/tmp/mono-ep-vendor-hopper-fp8-bwd-cache-v2-20261109.
Validator --bound --cached checks exact cache bits/scales and existing
quantized reference dx/dscore/h/dz/dW across skew/ring/reuse. Not yet passed.

Reciprocal-only F/B validation81083/91130 PASSED; artifacts
h100_fp8_reciprocal_forward_validation.json and backward counterpart.
Agent compile found F cubin82992bytes (2048bytes smaller), but C7510
WGMMA serialization, C7507 register warning and68B spills remain. No speedclaim.

New finiteBF16 activation quantization test sweeps all finite BF16 anchors
with128-column row variation, CPU float64 mathematical scale reference and
exact q-bit comparison. Eager passed first; fullgraph failed due omitted
Triton TRANSPOSE arg. token_quantization.py now passes explicit optional
args in token/MXFP8 launchers. Retry98055 PASSED2cases onH100, artifact
h100_activation_cache_quantization_validation.json. CPU21tests also pass.
Added GPUslow CI selection for this test.

Hopper agent owns forward hybrid buffer (fp8xbf16), FP8 input/L1 transport,
FP32 scales and runtime forward integration. L2 remainsBF16. Astra reviewed
several draft synchronization/alignment issues; agent fixing remaining
large-width scale pull-buffer bound. Root hybrid compile36496 LIVE, log
/private/tmp/ultra-hopper-hybrid-forward-compile.log, compile-only H2048.
Child upload rejected for missing prior approval context; root same Modal
action accepted with existing explicit source/header upload authorization.

Outstanding: finish forward GPU parity; backward cache parity; combined
fullgraph/checkpointed training bothmasters; remove remaining h/dZ repeated
quantization and pipeline serialization; uninstrumented paired8H200 timing.
Do not report new speedup before timing or claim all activation reuse done.

Latest status: hybrid compile36496 PASSED (H2048 only), artifact
h100_fp8_hybrid_forward_compile.json; serialization warnings/spills remain.
Forward bound parity35691 LIVE, log
/private/tmp/ultra-hopper-fp8-hybrid-forward-validation.log, immutable
/private/tmp/mono-ep-vendor-hopper-fp8-hybrid-v1-20261109. Includes latest
forward max(pull,SF-row) shared allocation and backward cachev2.
Backward parity15076 remains LIVE; do not duplicate either job.
Local selection/common API28passed. Hopper agent now investigating h/L2
cache reuse design without editing validated-in-flight sources. Review's
backward pull alignment finding was fixed in cachev2.


### Full Hopper activation caches under GPU validation (2026-11-09)

Previous goal turn was progress (cache code, GPU quantizer parity, live
validation launches); intervening user question verified Prime memory/AC
claims without changing kernel direction. Re-polled jobs, then both terminal:
BW15076 PASSED exactX/dYcache and existing math/skew/ring/reuse, artifact
h100_fp8_backward_xy_cache_validation.json, appap-lXfKaEgXPmziOaME5ynUOr.
F35691 BUILDFAILED: hybrid snapshot mixed currentapis with oldvendor missing
sm90_bf16_mono_ep_backward.hpp. This was snapshot assembly, not GPU parity.

Root extended BWcache to dZ after ownedCTA RecomputeHigh completes each
gate/up K128 group. OriginalBF16dZ retained for dW; qdz/scales published
beforeHighreadinessbit, dX waitsfullmask. 10poolsselectcacheall,8preservesXY,
4uncached. Static review clear. Validator nowchecks exact qdz/scales for
validrows and zero/scalefloor for paddedrows.

Agent added forward hcache: oddL1task acquires even sibling bit, quantizes
assembledK128, writes qh/scales thenpublishesoddbit. L2 existingpairedmask
protects consumption, preserving ringgeneration. Astra liveness proof depends
on scheduler1/locality1/FIFO/fullgridresidency; dependencyorder decreasesM/N.
Globalasyncproxyfence added beforepublication. Staticreview clear.

Complete coherent vendor snapshot copied from currenttree:
/private/tmp/mono-ep-vendor-hopper-fp8-full-cache-v1-20261109.
H100 F3620 LIVE log /private/tmp/ultra-hopper-fp8-full-cache-forward-validation.log.
H100 B20029 LIVE log /private/tmp/ultra-hopper-fp8-full-cache-backward-validation.log.
Do notduplicate. Both newfullcachepaths notyet GPUvalidated. F log metadata
still saysBF16transport fromoldvalidator label: actualinputtransportFP8,
combineBF16. Currentvalidator/benchmarklabels corrected forfutureuploads.

Benchmark nowreports local explicitworkspace componentbytes and maxrank
peakallocated/reserved acrosspostwarmupiterations. Memoryqueries/resets
outside timedinterval. Allpairedbackends resident, so notisolated perbackend
peak; externalallocators maynotappear inTorchpeak. Workspaceinventory is
separate, not additive topeak. Pinnedruff/Pythoncompilationpass; GPUs pending.


### Full-cache training passed; H200 measurement active (2026-11-09)

Full-cacheF3620/B20029 PASSED H100: artifacts
h100_fp8_full_cache_forward_validation.json and backward counterpart. F
artifact corrects stale stdouttransportlabel (actualFP8input/BF16combine).
Native75670 andcontrol14114 compiled/checkpointed training PASSED both
FP32/BF16masters, artifacts h100_fp8_full_cache_native_training.json and
control counterpart. Earliergoalturnwasprogress; latestjobsrevalidatedterminal.

Agent implemented directquantize_tokens_into callerFP8/stridedSF buffers;
runtime avoids qx/sx temporaries+copies. GPU eager/fullgraph finiteBF16 +
stridedout/sentineltests82308 PASSED6cases; artifact
h100_direct_activation_quantization_validation.json. CPU21pass andstatic
reviewclear. No router overlap implemented.

Current H200 timing7608 LIVE, log
/private/tmp/ultra-h200-fp8-full-cache-paired-timing.log, immutablefull-cache-v1
vendor, currentdirectquantPython+harnesswithmemoryreport. SamePrimegeometry
and5warmup15samples; samepersistentF/NCCLBcontrol. Do notduplicate.

Compileinspection88187/76583 completed. All PTXcalls in cachedBW arevprintf
from comm/barrier.cuh timeoutdiagnostics; not FP8division. 72Bspills,128regs,
C7507/C7510 remain. Evidence h100_fp8_full_cache_backward_codegen.json.
Added opt-in DG_TRAP_ONLY_TIMEOUTS in shared wait_until; default unchanged,
timeoutcyclechecksandtrap preserved whenenabled. Compileprobe57025 LIVE log
/private/tmp/ultra-hopper-fp8-trap-timeouts-codegen.log with --trap-timeouts;
no productionenablement or latencyclaim yet. Rootownsheader/compileprobe.


### H200 full-cache measured improvement (2026-11-09)

Timing7608 PASSED, artifact h200_fp8_full_cache_paired_timing.json.
8H200 Qwen30Bshape EP8,T16384,H2048,I768,E128,top8; FP32masters,FP8compute,
FP8forwardinput/BF16backwardtransport,fullgraph,5warmup15samples.
Maxrank synchronizedhostwall medians(ms):

| Scope | Native | Sameforward/NCCLbackwardcontrol |
| --- | ---: | ---: |
| F | 7.874 | 7.858 |
| B | 32.183 | 40.499 |
| F+B | 39.816 | 47.942 |
| AC2F+B | 47.360 | 55.594 |

Againstpriornative50.545/112.553/212.970, F6.42x/B3.50x/AC4.50x faster.
NativeB20.5% belowcontrol, AC14.8%belowcontrol. Prime3.9/8.2/15.9ms still
about2.02x/3.92x/2.98x lowerlatency; timingaggregation/transport differences
remain. Changes includeactivationreuse,FP8forwardtransport,directinputquant
writes; do not attributeisolatedcausality beyondthiscombinedchange.

Explicitworkspaceinventorybytes: Fsym1,921,318,784; Bsym2,294,321,024;
BF16pools14,181,990,400; FP8cache6,435,078,144; sourceDY70,778,880;
dscore1,105,920; metadata128. Nativepeakallocated25,614,785,536bytes;
NCCLcontrol46,615,457,792; reserved48,834,281,472. Allpairedbackendsresident;
Torchallocator excludes someexternalallocations, explicitinventorynotadditive.

Trap-onlycompile57025 PASSED, artifact h100_fp8_trap_timeouts_codegen.json.
ZeroPTXcalls, C7510 serializationwarninggone; spills96Bvs72B,128regs and
C7507 remain. Onlycompileexperiment;productionJITnotenabled. Noactive root
GPUjobs now. Agent auditingregisterreconfiguration beforefurtherchange.


### Hopper forward register-layout experiment (2026-11-09)

Previous goal turn made progress: full-cache H200 timings and codegen evidence
were saved. This turn preserved the validated 512-thread default and added an
explicit opt-in 384-thread forward instantiation. One non-math warpgroup owns
dispatch/A loader/B loader/scheduler; two math warpgroups remain unchanged.
Static review checked cleanup, barrier participants, task releases, and uniform
register directives. Compact scope is at most 32 ranks and 64 experts per rank.

H100 compile-only matrix (all four succeeded):

| Layout | Timeout diagnostics | Registers | Spill loads/stores | Stack |
| --- | --- | ---: | ---: | ---: |
| 512 threads | verbose | 128 | 76 / 76 B | 96 B |
| 384 threads | verbose | 151 | 0 / 0 B | 56 B |
| 512 threads | trap only | 128 | 76 / 76 B | 48 B |
| 384 threads | trap only | 152 | 0 / 0 B | 0 B |

All retain C7507 (register redistribution ignored). Trap-only variants remove
C7510 and PTX function calls while retaining timeout checks and traps. Neither
change has runtime correctness or timing evidence yet; no production switch.
Artifacts: h100_fp8_full_cache_forward_codegen.json,
h100_fp8_compact_forward_codegen.json,
h100_fp8_baseline_trap_forward_codegen.json, and
h100_fp8_compact_trap_forward_codegen.json. Sessions 21284, 79750, 10600,
77890 are terminal. Latest probe records NVCC version/command, instantiation
source/hash, and PTX register/thread directives.

Found the raw forward correctness probe stale after full-cache signature and
FP8-input changes; the previously validated bound path uses the current host
wrapper. Repairing the raw probe and adding explicit compact selection before
runtime validation. Backward layout remains unchanged.


Compact forward runtime correctness now PASSED on two H100s (session90314),
as did repaired raw 512-thread control (39139). Both are terminal. Artifacts:
h100_fp8_compact_forward_validation.json and
h100_fp8_raw_control_forward_validation.json. Tests exercise unequal tokens,
random/skew routing, empty destination rank, invalid routes, saved projections,
ring wrap (14336 used rows >13440 ring rows), and reuse. Compact output relative
error against independent quantized reference is 0.00464–0.00519; projection
errors are approximately 0.00065–0.00072. This is correctness evidence only,
not a speed comparison or training validation of the compact layout.

Next: measure the compact and baseline forward in one H200 allocation with
identical inputs, warmup, timing scopes and backward/NCCL controls. Keep verbose
diagnostics for the first layout comparison; trap-only remains a separate
compile-only factor. Production default is still512; backward unchanged.
No root GPU jobs remain active at this checkpoint.


### Shape and architecture selection direction (2026-11-09)

User asked for new-model performance without manual tuning. Plan: maintain a
small validated configuration family per architecture, filter candidates by
shape/alignment/shared-memory/register constraints, then optionally autotune
end-to-end communication-inclusive latency during warmup and cache the result.
Cache keys must include architecture/device, shape/token bucket, dtype, EP size
and topology, kernel version, and relevant communication policy. Keep selection
stable before graph capture; all participating ranks must agree on compatible
choices. Dynamic M-block work scheduling remains responsible for routing skew;
configuration selection must not introduce static per-rank SM partitions.
No complete autotuner exists yet, and near-optimal performance is an empirical
target, not a guarantee. Current 512/384 Hopper comparison establishes candidate
evidence before introducing automatic selection. The preceding Q&A turn did not
change code; this continuation resumes implementation and measured validation.


### Paired Hopper forward-layout timing launched (2026-11-09)

Added internal experimental host selector
MONO_EP_SM90_FP8_COMPACT_FORWARD using call-time std::getenv and exact
unset/0/1 validation. Default remains512; compact384 enforces<=64 local experts
and<=32 ranks. Distinct JIT instantiations use matching launch dimensions.
Benchmark --compare-forward-layouts compares both layouts in one allocation,
with unchanged native/NCCL backward controls, same tensors/routes, alternating
case order and5warmup/15measured iterations. Fullgraph remains enabled; CUDA
graphs explicitly disabled for this experiment to avoid replay freezing selection.
This environment override is an experiment, not the future automatic heuristic.

H200 timing session6827 launched, log
/private/tmp/ultra-h200-fp8-forward-layout-paired-timing.log. Immutable complete
vendor snapshot /private/tmp/mono-ep-vendor-hopper-fp8-layout-pair-v1-20261109.
Image build observed active; do not duplicate. No new latency result yet.

H200 layout-pair checkpoint: extension image built successfully (deep-gemm2.8.2, Python3.14.8); benchmark function created. Session6827 re-polled and remains live, no result JSON yet. Static reviewer cleared final std::getenv selector, opaque compiled dispatch, graph-replay disablement, and16 scope/backend/layout cases. Do not restart on silent captured subprocess output.


### Compact backward candidate (2026-11-09)

Previous goal turn made progress (paired host/harness implementation and H200
launch). Revalidated H200 session6827 live; Modal app
ap-sFgx3TDwKNAF1Xvi03pbHw is ephemeral with0tasks at inspection, so likely
awaiting capacity, not a terminal failure. No duplicate launch.

Added default-off compact backward template after kCacheDz. Keeps owned M-block
scheduling, per-warp cached quantization/publication, and original math mapping;
uses one uniform non-math WG and two math WGs. Production host remains512.
H100 compile52077 PASSED:166registers,0spill bytes,56Bstack versus baseline
128registers,72Bspills,104Bstack. C7507/C7510 remain. Artifact:
h100_fp8_compact_backward_codegen.json. No timing claim.

Extended raw backward validation to cached x/dy/dZ and explicit compact selection;
fixed prefix pointer ordering and appended cache pointers match host descriptors.
Bound control remains supported; bound+compact rejected. Ruff/Python compile pass.
H1002 full-cache compact parity76968 and raw512control16153 launched; logs
/private/tmp/ultra-hopper-fp8-backward-compact-validation.log and
/private/tmp/ultra-hopper-fp8-backward-raw-control-validation.log. Results pending.


Compact backward parity76968 and raw512control16153 PASSED and are terminal.
Artifacts h100_fp8_compact_backward_validation.json and
h100_fp8_raw_control_backward_validation.json. Both match the quantized FP32
matmul reference with dX relative errors0.00494–0.00532, dscore0.00127–0.00130,
and h/dZ/dW about0.00110–0.00127 for nonempty cases. Exact x/dy/dZ cache checks,
padded dZ zeros/scales, empty destination rank and14336-row wrap through13440-row
ring all pass. These are raw-kernel math/cache tests, not compact training or
latency validation. Production backward remains512. H200 session6827 remains
outstanding; do not replace/restart it based on absent captured output.


### H200 forward-layout result and compact training (2026-11-09)

H200 session6827/app ap-sFgx3TDwKNAF1Xvi03pbHw is terminal PASSED; result
h200_fp8_forward_layout_paired_timing.json. Same-allocation FP8 Qwen30Bshape,
EP8/fullgraph/5warmup15samples. Native F baseline5128.0625ms vs compact384
7.7840ms (3.45% lower); F+B40.0101 vs39.7754; AC47.6547 vs47.4393 (0.45%
lower). Backward remains512 in both and measured32.4114 vs32.2875; do not
attribute its noise-level difference to a backward change. Same-forward NCCL
control AC55.9734 vs55.5793. Removing forward spills is a small improvement,
not the primary remaining opportunity. Preserve default512 pending broader
selection evidence; prioritize backward/pipeline serialization next.

Compact backward now has exact opt-in call-time host selector analogous to
forward, matching template/block dimensions and bounds. Training runner
--compact enables both variants before process startup and reports selection.
Astra static review cleared kernel, raw cache probe, host and runner. Immutable
vendor snapshot /private/tmp/mono-ep-vendor-hopper-fp8-compact-training-v1-20261109.
Compiled/checkpointed two-H100 training session38480 LIVE, log
/private/tmp/ultra-hopper-fp8-compact-training.log. Tests FP32/BF16 masters.
Preparing a three-configuration same-allocation timing (baseline, compact F only,
compact F+B) but do not launch until training is validated.

Three-configuration --compare-kernel-layouts harness prepared (baseline F/B, compact F only, compact F+B), preserving old forward-only comparison. Resets both selectors per case and prewarms backward variants. Timing launch still gated on compact training38480, revalidated live during image build. No H200 timing job currently active; prior6827 succeeded and was archived to artifact.


### Timeout-diagnostic serialization experiment (2026-11-09)

Previous turn made progress by saving terminal H200 paired measurements and
integrating compact backward for training. Revalidated training38480 live;
updated extension built successfully, validation pending.

Added raw F/B validation --trap-timeouts flags that define
DG_TRAP_ONLY_TIMEOUTS only for the test extension; timeout clock checks and
traps remain, device diagnostic printf calls omitted. Bound+trap rejected,
production host unchanged. Compact cached BW compile56204 PASSED:
168regs,0stack,0spills,0PTXcalls; C7510 removed, C7507 remains. Artifact
h100_fp8_compact_trap_backward_codegen.json. No latency inference.

Raw compact trap-only parity jobs active: backward74251 and forward31424.
Logs /private/tmp/ultra-hopper-fp8-backward-compact-trap-validation.log and
/private/tmp/ultra-hopper-fp8-forward-compact-trap-validation.log. These are
separate from compact verbose compiled training38480. Wait for results;
do not combine correctness evidence or timing claims across variants.

Trap-only raw compact F31424/B74251 parity both PASSED and terminal; artifacts h100_fp8_compact_trap_forward_validation.json and h100_fp8_compact_trap_backward_validation.json. Production/compiled training still use verbose timeout handling; no trap-only latency measurement yet. Training38480 remains live after function creation.


Compact verbose training38480 terminal PASSED both FP32/BF16 masters. Artifact
h100_fp8_compact_native_training.json. Fullgraph+AC,3AdamW steps,twooutstanding
forwards/reversebackward,unequal ranks/emptyexperts; native forward replay counted.
Gradient comparisons to BF16 reference remain about6–7% as expected for this
existing FP8 training test; raw quantized-reference tests provide tighter math
checks. Short numerical validation does not establish long-run convergence.

Launched eight-H200 three-configuration paired timing35621 with previously
reviewed --compare-kernel-layouts. Log
/private/tmp/ultra-h200-fp8-kernel-layout-paired-timing.log; same immutable
compact-training-v1 vendor. All use verbose timeout handling. Matrix baselineF/B,
compactFonly,compactF+B; native/NCCLcontrols,5warmup15samples,fullgraph with
CUDAgraphreplaydisabled. NCCL ignores native backward selector, metadata explicit.
No duplicate launches; this is the only active root GPU job at checkpoint.


Next bottleneck audit: after the current three-layout timing completes, collect
an updated full-cache Hopper profile. The older profile predates activation
reuse and cannot quantify today's persistent-kernel vs quantization/transpose/
weight-gradient costs. Do not infer the main bottleneck from compiler warnings
alone. Trap-only compiled validation is being prepared independently; keep its
latency comparison separate from the thread-layout ablation.


### Trap-only compiled integration (2026-11-09)

Previous turn made progress: compact verbose training passed, full layout timing
launched, raw trap-only F/B validation passed. Current35621 layout timing remains
live with no result JSON; keep samehandle/app ap-QcMuB4AJxqa2RoyaEznWTM.

SM90FP8 host F/B now accept exact call-time opt-in
MONO_EP_SM90_FP8_TRAP_TIMEOUTS; unset/0 retains diagnostics. Selected JIT
source defines DG_TRAP_ONLY_TIMEOUTS before includes, separate tags encode
layout/direction/handler. No timeout removal. Static review clear. Training
runner --trap-timeouts sets/clears all selectors explicitly and rejects control.

Compact trap-only compiledACtraining11991 launched on H1002 for FP32/BF16
masters, log /private/tmp/ultra-hopper-fp8-compact-trap-training.log. Snapshot
/private/tmp/mono-ep-vendor-hopper-fp8-compact-trap-training-v1-20261109.
Do not substitute this variant's validation for verbose layout timing or claim
latency improvement from removing the compiler warning. Two rootGPUjobs live:
35621 pairedH200layouttiming and11991 trap-onlyH100training.


### Compact backward rejected by timing (2026-11-09)

Three-layout H20035621 terminal PASSED; artifact
h200_fp8_kernel_layout_paired_timing.json. Sameallocation Qwen30B FP8/FP32master,
EP8,5warmup15samples,fullgraph; timeout diagnostics enabled. Native medians(ms):

| Layout F/B | F | B | F+B | AC2F+B |
| --- | ---: | ---: | ---: | ---: |
| 512/512 | 7.854 | 32.172 | 39.789 | 47.569 |
| 384/512 | 7.622 | 32.191 | 39.663 | 47.104 |
| 384/384 | 7.787 | 48.369 | 55.828 | 63.286 |

Compact backward is50.3% slower than512 with compactFfixed, AC34.4%slower.
NCCLBstays40.52–40.55ms. No default change: preserve512backward. Zero spills
are insufficient performance evidence; reduced dispatcher throughput or another
pipeline effect needs profiling, not assumption. CompactB remains explicit
experimental comparison only. Need benchmark-based selection for futureheuristics.

Launched updated full-cache H200 baseline profile using samecompact-training-v1
vendor and --profile. Log /private/tmp/ultra-h200-fp8-full-cache-profile.log.
This profile uses512F/B and NCCLcontrol; profiler output is diagnostic, not timing.
Trap-only compact training11991 remains independent; do not stop/restart it
based on this timing result. Trap-only test may still illuminate serialization.

Updated full-cache profile handle is70708; trap-only training handle11991. These are the active root jobs after35621 completed.

### Ordered structural FP8 work (2026-11-09)

1. Owning CTA emits dual-quantized dW operands: column scales across the owned
   128-token block on SM90, 32-token MXFP8 groups on SM100. Emit h/dZ in the
   epilogue and x/dY from staged input, mask padding before amax, preserve BF16
   rounding. Feed DeepGEMM 1D1D k-grouped wgrad directly. Remove separate
   packing passes and full BF16 pools retained solely for dW. Python packed
   consumer exists; persistent producer and integration are still in progress.
2. Verify existing once-per-owned-block h/dZ activation quantization and reuse
   across N tiles; eliminate remaining BF16 staging as part of step 1.
3. FP8 backward transport and combine output; label results mixed transport
   until complete.

After each step: paired H200 native/NCCL in the same allocation, warmed timings,
max-rank median AND median of per-iteration mean-rank latency, alongside Prime
3.9/8.2/15.9 ms F/B/checkpointed 2F+B. Target about16ms at FP32 masters; above
25ms after all three requires structural investigation. Historical rank means
were not recorded and cannot be reconstructed.

Weight layout audit confirms per-call BF16 W1 interleave and backward W1/W2
transposes. Replace with paired64-row forward loads and MN-major BF16 backward
operands. FP8 forward needs paired gate/up scales; FP8 backward needs fused
strided quantization into its required K-major layout, not a transpose of the
forward FP8 payload. Preserve scale grouping and optimizer-update correctness.

FSDP gate: real runtime SM reservation and frozen buffer budgets implemented,
but concurrent NCCL GPU validation remains pending. Do not wire persistent
Axolotl integration before that proof.

Fresh full-cache H200 profile completed: four native backward packing kernels
sum about9.77ms/rank and grouped dW GEMMs about1.36ms; persistent backward
18.22–23.00ms. These are profiler diagnostics, not benchmark latency.
Compact trap-only H100 fullgraph/checkpointed training passed FP32 and BF16
masters; saved h100_fp8_compact_trap_native_training.json. This does not reverse
the measured compact-backward regression or change the default512 threads.

Direct-pack integration checkpoint: Python now allocates four column-scaled
FP8 payloads and four MN-major FP32 scale arrays, plus GPU padded expert counts,
and connects them to the prepacked wgrad consumer. This is pending the matching
vendor producer and must not run against the old extension. Existing BF16
pools have NOT yet been removed, so this is not completion of structural step1.
The independent H100 packed-consumer matrix is running as terminal88983, log
/private/tmp/ultra-hopper-dual-packed-wgrad-validation.log, using the prior
compact-trap-training-v1 vendor snapshot. Runtime reservation CPU tests pass
21cases, including invalid budgets and frozen allocation-time SM counts.
The reserved-SM GPU gate now requires CUPTI evidence of overlap in BOTH
directions; merely enqueueing collectives before weight preparation is not proof.

H100 packed-consumer validation88983 completed successfully:96 tests passed
in36.87s, including compiled/uncompiled directpacked input, emptyexperts,
NaN-poisoned padded rows, Qwen widths. Artifact
benchmarks/h100_dual_packed_wgrad_validation.json. This validates the external
packer-to-consumer contract only; it does not validate the new persistent
producer or show a speedup. App ap-wstCXC5rGZkR014LC5RjAf.

Unused-capacity safety gate89864 also passed96 H100 tests in38.41s after
poisoning unused FP8 payload and scale tails with NaNs; artifact
benchmarks/h100_dual_packed_wgrad_tail_validation.json. Thus producer needs
to initialize padded rows within live expert groups, not unused capacity.
The initial in-kernel packing implementation still rereads BF16 pools: it is
an intermediate parity check only. Direct register/shared epilogue emission
and bounded staging remain required before claiming step1 complete.

Persistent producer parity gate staged:18-pool native ABI, optional trailing
padded_counts preserving old Python arity, CTA-local metadata with mbarrier,
single global metadata writer, masked columnquantization. Review caught/fixed
divergent fullwarp reduction, duplicateglobal metadata races, and initial
barrier-order circularwait. v1 Modal initialization was stopped before kernel
execution; do not reuse its snapshot. Reviewedv2 snapshot:
/private/tmp/mono-ep-vendor-hopper-dual-pack-parity-v2-20261109.
2H100 exactpayload/SF anddW parity running terminal69099, log
/private/tmp/ultra-hopper-direct-packing-parity-v2.log. Balanced/skewed/balanced
reuse, unequalranktokens, FP32/BF16masters, fullgraph. Next implementation
continues direct staged/shared packing and bounded BF16 scratch; fullpools
still exist in v2. No H200 timing claim from this intermediate implementation.

Direct producer v2 parity69099 PASSED on2H100: exact FP8payloads/SF anddW
against externalpacker forFP32/BF16masters, balanced/skewed/balancedreuse,
unequalranktokens, emptyexpert-rank case, fullgraph. Artifact
benchmarks/h100_persistent_direct_packing_parity.json; app
ap-dRvmeCNq9dUKaSarPrQRsQ. This v2 stillretainedfullBF16pools.
Currenttree nowallocates first4BF16buffers at ringcapacity, derivesfullpacked
capacity fromrowFP8cache, andnative packs h/dg/du fromregistersnapshots via
shared128x64 tiles; these NEWchanges await GPUvalidation. x/dy still use
boundedglobalring stageddata. Newvalidator can rerun legacy10pool native
backward toobtain independentfullBF16reference afterringwrap; --ring-wrap
uses16activeSMs and2049tokens/rank and requires>2ringcapacities ofskewedwork.

Shared/register dual-pack checkpoint (2026-11-09): currentreviewednative
usesring-bounded first4BF16scratch, x/dycolumnpack from stagedringblocks,
h/dg/duregister snapshots stagedthroughshared128x64tile. Reviewfixed
snapshot-before-overwrite barrier. Python allocations/capacity updated;
21CPUreservation tests andpinnedRuff pass. This removescapacity-sizedBF16
dW retention butstillhasring-globalx/dy staging anddZ forrowcache.

Frozen /private/tmp/mono-ep-vendor-hopper-shared-dual-pack-v1-20261109:
-2H100 smallparity running31105, log
 /private/tmp/ultra-hopper-shared-dual-pack-parity-v1.log.
-H100 codegen running15985, log
 /private/tmp/ultra-hopper-shared-dual-pack-codegen-v1.log; inspectregisters/spills.
Both lastverifiedlive duringimagebuild. Donotrestart fromquietlogs.

Updatedvalidator --ring-wrap pendinglaunch:16activeSMs,2049tokens/rank,
requires>2ringcapacities under skew. Exactpacking checks onlynonwrapped
current-invocation scratch. Separatelegacy10pool reference has nondeterministic
atomicrowordering/FP8groups: wrappedcases usegradient numericalcomparisons,
NOTbitexactQ/SF. Allranks collectivelydecide whetherlegacyreferenceisneeded,
soemptyranks cannot skipthepeer kernel. Nativefirstv2producer validation
remainsvalidbecauseitcomparedsame-invocationfullBF16pools.
Next: obtainsharedsmallparity/codegen, runforcedwrap+reservedSM/NCCLoverlap
gates, thenpairedH200 withmaxrankmedian+meanrank vsNCCL+Prime. Do notclaim
step1timing improvement fromtheoldv2correctness result.

Compilerresult15985: shared-dual-pack-v1 builds but128regs/688Bstack,
1076Bspillstores/1484Bspillloads. Artifact h100_shared_dual_pack_codegen_v1.json.
Do nottime/adoptthisimplementation asfast: replace96BF16perthread arrays
withcached-mode-unused16KiB quantized_a sharedscratch, onefieldatime.

Sharedsmallparity31105 terminalfailed at exactpackedcomparison; capturedolder
validator usedseparatelegacylaunch wheneverringrows!=capacity. Atomicrouting
reordersrows so thiscomparisonisnotvalid. Do notmarkkernelpassed orfailed
fromthat result. Artifact h100_shared_dual_pack_parity_v1.json retained.
Corrected same-invocation/nonwrapped reference rerun57522, log
/private/tmp/ultra-hopper-shared-dual-pack-parity-corrected-v1.log.
ReservedSM/CUPTI-overlap gate75064running, log
/private/tmp/ultra-hopper-reserved-sms-dual-pack-v1.log. Both usefrozen
shared-dual-pack-v1 vendor andwereverifiedfunctioncreated/live.
Benchmarkmemory inventory nowsplits rowFP8cache vsdirectpackedwgrad, includes
paddedcounts andactive/physicalSMs, reportsBF16scratch vsoperandcapacityrows.

Corrected sharedv1 smallparity57522 PASSED: exactsame-invocationQ/SF andzero
dWrelativeerror forbothmasterdtypes andallroutingcases. Ring17280rows exceeds
live0–1536rows, so thisdoesnotprovewrap. Artifact
h100_shared_dual_pack_parity_corrected_v1.json.

HopperreservedSMgate75064 PASSED:132physical/116active/16reserved,
NCCL_MAX_CTAS4,3PGs/streams,20parity/reuseiterations,postallocationbudget
rejection. CUPTIshowed3simultaneousNCCLstreams overlapping BOTHF/B onboth
ranks. dxexact, dscore~2.5e-7relative, dW~0.00095relative. Artifact
h100_reserved_sms_dual_pack_v1.json. Kernelconcurrencyproofonly; notfullFSDP
training orSM100concurrencyproof.

Sharedscratchv2snapshot /private/tmp/mono-ep-vendor-hopper-shared-dual-pack-v2-20261109
removes96BF16/threadarrays andusesexisting16KiBsharedtile onefieldatime.
Staticreviewclear. Newjobs:
-47567 codegen /private/tmp/ultra-hopper-shared-dual-pack-codegen-v2.log
-19722 forcedwrap /private/tmp/ultra-hopper-shared-dual-pack-ring-v2.log
Do notclaimspillremoval untilcodegenreturns.

Sharedscratchv2codegen47567 PASSEDcompile:128regs,304Bstack,528Bspillstores,
816Bspillloads (downfromv1 688/1076/1484 butaboveoldbaseline). Artifact
h100_shared_dual_pack_codegen_v2.json. Measurebeforefurtherpackingchanges.
Forcedwrap19722 terminalFAILEDcoverageassertion:live16384,ring19200; no
wrapoccurred. Thisisatestcoveragefailure,notevidenceofkernelincorrectness.
Artifact h100_shared_dual_pack_ring_v2.json. Updatedtest uses4096tokens and
max_tokens4096 (avoidscapacityrounding), balancedH1024/I512 default, optional
QwenH2048/I768/E128/top8. Qwenforcedwrap17553running on2H100,16activeSMs,
log /private/tmp/ultra-hopper-shared-dual-pack-qwen-ring-v2.log, samev2snapshot.

Step3read-onlyaudit: directFP8x/dYtransport needs lifetime-owned symmetric
FP8+FP32K128scale buffers; cannotaliasforwardbuffer acrossoutstandingforwards.
FP8-onlytransport intentionallychangescolumn dWquantization to row-FP8
decode/BF16round thencolumnquant. Validateagainstthatreference; do not add
BF16sidecars tothemaintarget orclaim bit-exactoriginalBF16dW. FP8combine needs
per-token/per-topkslot/per-K128scale, FP32sum, BF16finalgradient. PublicPrime
docs confirmFP8GEMMs butnotallprivatewirelayoutdetails; precisionmatching
flagmuststayfalseuntilknown.
Packingaftertiming:128-rowunrolledloops likelyinflate livevalues/spills.
Candidate: no-unrollx/dy loops (preservescoalescedrowmajorloads); h/dg/du
warpcolumnreduction4rows/lane, coalescedpackedKstores. Sharedbankconflicts
needmeasurement, notassumptions. Noeditsfortheseexperimentsyet.

Qwenforcedwrap17553 PASSED:2H100,16activeSMs,H2048/I768/E128/top8,4096
tokens/rank,FP32/BF16masters,fullgraph. Ring28800rows,skewedlive65536
(>2ringgenerations),emptypeer-rank,balanced/skewed/balancedreuse. Relative
dWerrorvslegacy~0.00021FP32 /~0.00089BF16; dx/dscorechecks passed. Artifact
h100_shared_dual_pack_qwen_ring_v2.json. Crosslaunchroworderingdifferent,
so thisisnumericalparity,notbitexactpackedparity.

Launchedpaired8H200Qwen30B benchmark usingreviewedshared-dual-pack-v2vendor,
log /private/tmp/ultra-h200-shared-dual-pack-paired-v2.log. Standarddefault512
F/Bthreads,physicalSMbudget,noFSDPcollectives,mixedtransport,FP32masters,
5warmups15samples. Newreportincludesmaxrankmedian+meanrankmedian andsame
NCCLcontrol, separateBF16ring/FP8cache/packedwgradworkspace inventory.
Do notclaimlatencyimprovementuntilthisresultreturns.

PairedH200shared-dual-pack-v2 livehandle88636; frozenkernel remainsunchanged
whiletiming. Nextpackingexperiments deferreduntilthat evidence.

Owner-local dual quantization status (2026-11-09): Hopper uses each owned
128-row M-block to emit column-scaled FP8 x/dy/h/dz and directly feeds
DeepGEMM 1D1D grouped wgrad. x/dy are read from bounded BF16 ring scratch;
h/dz are emitted from the epilogue. This removes the full-capacity BF16 dW
pools and external packing launches, but does not eliminate bounded BF16
staging or make backward transport FP8. SM100 needs the corresponding
32-row MXFP8 implementation and independent validation.

The next warp-cooperative Hopper producer has static review clearance:
four rows per lane for h/dz, no-unroll x/dy loops, no shared packing tile.
GPU parity and codegen measurements remain outstanding. The existing H200
v2 benchmark stays frozen; latest container inventory is empty and its live
session has returned no result. Do not report a new speedup yet.

tools/report_persistent_training.py now renders paired max-rank/mean-rank
medians, Prime references, geometry, precision and warmup/sample counts.
Historical missing rank means remain N/A. Pinned Ruff and a smoke run on
h200_fp8_full_cache_paired_timing.json passed.

Warp producer v3 validation launched from frozen vendor snapshot
/private/tmp/mono-ep-vendor-hopper-warp-dual-pack-v3-20261109:
- H100 codegen session 7651, app ap-6MBvwoYqbqI1cWeXq0HsUb,
  log /private/tmp/ultra-hopper-warp-dual-pack-codegen-v3.log.
- Qwen forced-wrap parity session 69951, app ap-v0UvpV0He5aJ7bKyLcDTgH,
  log /private/tmp/ultra-hopper-warp-dual-pack-parity-v3.log.
Both are building; no validation result yet. Compile probe now honors
MONO_EP_VENDOR_DIR for its mounted headers as well as the built extension.
Pinned Ruff passed for this change.

H200 v2 session 88636 remains live; latest Modal app inventory now reports
one task (previously zero), so the allocation wait has ended. No timings yet.
Hopper agent is implementing opt-in FP8 backward x/dy transport. Require
independent backward buffer lifetime across outstanding forwards and delayed
backwards; no BF16 wire sidecar. Python edits deferred until the v3 validation
mount is captured. The existing v2 timing and v3 native snapshot stay frozen.

H200 v2 paired timing completed successfully (artifact
h200_shared_dual_pack_paired_v2.json). Qwen30B H2048/I768/E128/top8,
T16384/rank EP8, FP32 masters, mixed transport, fullgraph, 5 warmups and
15 samples, router excluded. Milliseconds (max-rank median / mean-rank median):

| Scope | Native | Same-run NCCL | Prime reference |
| --- | --- | --- | --- |
| Forward | 8.060 / 8.058 | 8.033 / 8.030 | 3.9 |
| Backward | 36.051 / 36.022 | 40.645 / 40.592 | 8.2 |
| F+B | 43.795 / 43.769 | 48.249 / 48.181 | unavailable |
| Checkpointed 2F+B | 51.540 / 51.513 | 55.917 / 55.855 | 15.9 |

This regressed against retained full-cache results from a different allocation:
backward +12.0%, checkpointed +8.8%. Same-run checkpointed native is 7.8%
lower latency than NCCL. Removing external packing has not yet produced a
speed win over the earlier native path. Prime remains an unmatched external
precision reference. Workspace inventory is recorded in the artifact;
allocator peaks include all paired backends and are not standalone memory.

Warp producer v3 codegen completed successfully: 128 registers, 104-byte
stack, 72-byte spill stores and loads, versus v2 304/528/816 bytes. Artifact
h100_warp_dual_pack_codegen_v3.json. This restores spill counts to baseline,
but is not a measured latency improvement. Qwen forced-wrap parity69951 is
still live. Timing runner now accepts MONO_EP_SOURCE_DIR so subsequent
transport edits can coexist with frozen v3 BF16-wire timing; pinned Ruff passed.

FP8 backward x/dy transport implemented behind exact opt-in
MONO_EP_SM90_FP8_TRANSPORT=1 (unset/0 remains BF16). Separate backward-owned
FP8 symmetric buffers carry x/dy and K128 scales; dispatch writes row caches
directly. Python requantizes saved inputs after the previous completion event,
including contiguous normalization. Native and Python static review passed;
device parity and delayed/reversed autograd validation remain outstanding.
Combine output is still BF16, so do not label the whole path fully FP8.

Independent transport validator compares with a separate BF16-wire backend
fed PyTorch row-FP8-decode/BF16-round operands. It checks dx/dscore/dW for
FP32/BF16 masters, balanced/skewed/reused routing, unequal rank tokens, and
more than two ring generations at Qwen30B shapes. Direct backend test only;
it does not establish torch.compile or reverse-forward autograd correctness.
Pinned Ruff passed. H100 job1936 launched, app ap-VGK3oEBwLyyuArTFh93xRy,
log /private/tmp/ultra-hopper-fp8-wire-parity-v1.log; frozen native snapshot
/private/tmp/mono-ep-vendor-hopper-fp8-wire-v1-20261109.

Frozen BF16-wire Python source for v3 timing is available at
/private/tmp/mono-ep-source-hopper-warp-dual-pack-v3-20261109. Native v3
parity69951 remains live (container ta-01M70RBP7NVQC0WNA8FZHWADRR observed).
Do not launch its timing until numerical validation passes.

v3 Qwen forced-wrap validation69951 PASSED (artifact
h100_warp_dual_pack_qwen_ring_v3.json): FP32/BF16 masters, fullgraph,
balanced/skewed/balanced reuse, ring28800 and skewed65536 routed rows,
empty expert rank. dW relative errors below0.000895; dx/dscore assertions pass.
Launched paired8H200 v3 timing session26110 using frozen v3 native AND Python
snapshots, log /private/tmp/ultra-h200-warp-dual-pack-paired-v3.log. It remains
BF16 backward wire to isolate the packing change from FP8 transport.

Independent FP8-wire validator received static review clearance for the
intended two-GPU finite-input test. It establishes numerical comparison only,
not bit-exact wire transfers, memory savings, or compiled/reversed autograd.

Compiled/checkpointed FP8-input-transport training gate launched session71184,
log /private/tmp/ultra-hopper-fp8-wire-training-v1.log, same frozen fp8-wire-v1
vendor. Existing training validator covers two outstanding forwards, reverse
backwards, three AdamW updates, unequal rank tokens, missing routes/skew, and
FP32/BF16 masters. Runner now explicitly selects FP8 transport and reports
input FP8 / combine BF16 separately; pinned Ruff passed. Results pending.

FP8 combine next: production layout should replace the internal BF16 return
allocation with E4M3[slot,token,H] plus FP32[slot,token,H/128] scales. Quantize
BF16-rounded route contributions in each owned K128 tile, publish scales and
payload before existing peer readiness, accumulate decoded top-k in FP32,
store final external BF16 dx. Keep dscore and dW semantics unchanged. Agent
is implementing native layout behind an opt-in; Python edits held until live
validation source mounts are captured. Require read-only allocation-derived
combine views for independent per-route quantization tests and capacity
poisoning, plus a check that no internal BF16 return allocation is retained.

FP8 input transport Qwen ring validation1936 PASSED. Artifact
h100_fp8_wire_qwen_ring_v1.json: independent decoded-BF16-wire reference,
dx exact, dscore relative error <=2.82e-7, dW relative error <=2.16e-4,
FP32/BF16 masters, balanced/skewed/reused routes, live65536/ring28800,
empty expert rank and unequal source counts.

Compiled training71184 PASSED (h100_fp8_wire_training_v1.json). Each master
dtype has six rank-step records covering three AdamW steps, two outstanding
forwards, reverse backward and checkpointed fullgraph execution. This uses
the existing dense BF16 reference with 15% FP8 gradient tolerance; max dW
error6.84%, updated-weight error0.116%. Do not confuse this broad full-training
comparison with the tighter decoded-wire transport reference above.

Frozen v1 Python source: /private/tmp/mono-ep-source-hopper-fp8-wire-v1-20261109.
Launched paired8H200 FP8-input/BF16-combine timing session80516,
log /private/tmp/ultra-h200-fp8-wire-paired-v1.log, frozen v1 native+Python.
Runner now explicitly controls input transport and benchmark records include
input/combine precision separately plus source dy scale bytes. Pinned Ruff
passed. No result yet; v3 BF16-wire timing26110 also remains live.

v3 BF16-wire H200 timing26110 completed successfully, but regressed again.
Artifact h200_warp_dual_pack_paired_v3.json; same Qwen30B/EP8/T16384,
FP32 masters, mixed transport, fullgraph, 5 warmups/15 samples, no router.
Times are max-rank median / mean-rank median milliseconds:

| Scope | Native | Same-run NCCL | Prime reference |
| --- | --- | --- | --- |
| Forward | 8.035 / 8.033 | 8.042 / 8.039 | 3.9 |
| Backward | 39.428 / 39.411 | 40.498 / 40.429 | 8.2 |
| F+B | 47.139 / 47.118 | 48.136 / 48.090 | unavailable |
| Checkpointed 2F+B | 55.093 / 55.071 | 55.982 / 55.931 | 15.9 |

Versus v2 in a different allocation: backward +9.4%, checkpointed +6.9%.
Lower compiler spills did not produce a speedup. Shared-memory bank conflicts
in warp-per-column reads and serial column packing remain hypotheses requiring
measurement; do not promote v3 as faster. FP8 input transport timing80516 is live.

Added independent FP8 combine validator and runner, not yet launched. It
quantizes each legacy BF16 route contribution separately before FP32 sum,
checks dx/dscore/dW, poisons candidate FP8 payload/scales including unused
capacity, checks exact zero output for missing routes, and requires exact
internal buffer savings topk*max_tokens*(H-4H/128). Pinned Ruff passed.
Review found a native blocker: host rereads global FP8_COMBINE while backend
layout is frozen at construction. Require explicit per-backend bool passed
through native ABI (default false) so mixed-layout instances coexist safely.
Agent is fixing this before snapshot/device validation.

Combine selector blocker fixed with explicit trailing native bool defaultfalse;
Python passes frozen per-backend flag. Final static review clear. FP8 combine
validator launched22116, log /private/tmp/ultra-hopper-fp8-combine-parity-v1.log,
frozen vendor /private/tmp/mono-ep-vendor-hopper-fp8-combine-v1-20261109.
No GPU result yet. Native API exposes allocation-derived combine views for
both layouts; all legacy defaults retained.

Packing regression source-level finding: v3 warp-per-column gate/dh reads
BF16 index row*128+fixedcol. All32 lanes access distinct addresses in the
same4-byte shared-memory bank. This is an address-level conflict analysis,
not a measured runtime attribution. Agent is preparing a swizzled16KB scratch
candidate using row*64+(col XOR 2*(row&31)); warp-column reads then visit32
distinct banks while row writers retain contiguous bank coverage. Keep all
other packing choices fixed and require correctness/codegen/paired timing.

FP8 combine compiled training gate launched10714 using frozen combine-v1
vendor; log /private/tmp/ultra-hopper-fp8-combine-training-v1.log. Existing
validator checks reverse-order backwards, two outstanding forwards,
checkpointed fullgraph, three optimizer steps, both master dtypes. Combine
per-route test22116 remains live. Source frozen for later timing at
/private/tmp/mono-ep-source-hopper-fp8-combine-v1-20261109.

Training/timing runners now explicitly pass both diagnostic transport flags,
reject combine without input transport, and clear defaults to0. Benchmark
metadata distinguishes input/combine wire precision and BF16 final outputs;
Prime precision matching stays false. Pinned Ruff passed. No combine timing
launched before correctness gates. H200 input-wire benchmark80516 is live;
latest container inventory showed no assigned H200 container, so continue
waiting on the existing handle rather than restarting.

Swizzled packing static review clear; frozen vendor
/private/tmp/mono-ep-vendor-hopper-swizzled-dual-pack-v4-20261109.
Launched H100 codegen3486 (/private/tmp/ultra-hopper-swizzled-dual-pack-codegen-v4.log)
and Qwen ring parity71376 (/private/tmp/ultra-hopper-swizzled-dual-pack-parity-v4.log).
No correctness or speed result yet; tests use default BF16 backward transport.

FP8 combine parity22116 PASSED. Artifact h100_fp8_combine_qwen_ring_v1.json:
dx exactly matches independent per-route BF16→FP8 decode→FP32sum→BF16
reference; dscore relative error <=2.80e-7; dW <=2.17e-4. Both masters,
balanced/skewed/reuse, empty expert rank, unequal counts, invalid-route exact
zeros, poisoned unused FP8/SF capacity. Skewed63488 rows exceed2*ring28800.
Internal return allocation saved91422720 bytes (87.1875MiB) PER RANK in this
test configuration, exactly matching the expected BF16-to-FP8+SF replacement.
Do not extrapolate this figure to other token capacities.

FP8 combine training10714 PASSED, artifact h100_fp8_combine_training_v1.json:
both master dtypes, six rank-step records each, fullgraph checkpointed,
two outstanding forwards/reverse backward/three AdamW steps. Dense BF16
reference dW max6.84%, updated weights max0.116% (existing FP8 tolerance).
Launched paired8H200 combine timing90873 from frozen combine-v1 native+src,
log /private/tmp/ultra-h200-fp8-combine-paired-v1.log. Input-wire-only timing
80516 remains live. Keep both separate from v4 swizzled packing experiment.

Added explicit top-k1 coverage to combine validator/runner (top-k8 remains
the default; forced multi-generation wrap requires top-k8). Pinned Ruff passed.
Top-k1 H100 job98579 PASSED: artifact h100_fp8_combine_topk1_v1.json,
12 rank/dtype/routing records, dx exact, dscore <=2.83e-7 relative,
dW <=9.52e-5 relative, poisoned unused capacity and invalid routes. Internal
return savings3809280 bytes/rank for this smaller capacity. This supplements
the top-k8 multi-generation wrap gate; it is not an additional timing result.
Swizzle codegen3486/parity71376 and H200 timings80516/90873 remain live.

Swizzle v4 codegen3486 PASSED:128regs,104Bstack,72Bstores/72Bloads,
artifact h100_swizzled_dual_pack_codegen_v4.json. Qwen ring71376 PASSED:
12 rank/dtype/routing records, dW max0.000895, existing dx/dscore assertions,
artifact h100_swizzled_dual_pack_qwen_ring_v4.json. Launched paired8H200
v4 timing45693 with frozenv4native + frozenv3BF16-wire Python source,
log /private/tmp/ultra-h200-swizzled-dual-pack-paired-v4.log. This isolates
packing from transport; no speed result yet.

Modal app inventory confirms input-wire80516/ap-m1MnVQ7MFVd1yuqRQA52zk
and combine90873/ap-mrcfFBHQtok4Eh5uCHbpp9 both ephemeral Tasks0, notstopped.
Continue existing jobs. Benchmark audit confirmed NCCL cases set the runtime
peer backend toNone and do not inherit native transport selection.

While H200 allocations queue, agent is implementing fused strided Hopper
128x128 backward weight quantization to eliminate explicit full W1/W2
transpose copies before FP8 conversion. Preserve originalmasters and exact
existing quantization blocks/scales; do not transpose forward-quantized weights.
Kernel integration and new GPU jobs deferred until focused validation; this
does not change frozen packing/transport runs or solve forward interleaving.

Standalone fused transpose quantizer implemented in weight_quantization.py:
quantize_transposed_weights_128x128 reads positive-strided BF16/FP32 experts,
writes contiguous transposed E4M3 and128x128pow2FP32scales. No backward
integration yet. Review fixed int32 address overflow with int64 products;
tests now independently cover scale boundaries, tails, contiguous/transposed/
sliced strides, Qwen W1/W2 shapes and eager/fullgraph. Static review clear.
H100 exact-parity job77370 launched using modal_hopper_wgrad --weight-quantization;
log /private/tmp/ultra-hopper-strided-weight-quantization-v1.log. Await GPU result
before replacing existing transpose+quantize in the runtime.

Modal native build now copies only fetch_build_dependencies.py into its build
tools layer; unrelated report edits no longer invalidate that layer. Existing
jobs unchanged. Report comparator now also rejects different-device baselines.
Pinned Ruff and retained H200 report smoke check passed.

Verified new quantizer tests are selected by cicd_cuda_kernels.sh's package
GPU suite; SM89+ guard matches FP8 conversion capability. Quantizer77370
still building; no runtime integration or GPU success claim.

Forward layout audit CORRECTION: current SM90/SM100 BF16 B-weight loaders
also issue one128-row interleaved load; there is no existing paired B-loader.
The analogous two64-row loads are FP8 forward's BF16 A-operand loads.
Current WGMMA/fragment store preserves canonical output-column order. To
remove W1 interleave, original-layout FP8 W1 must use paired64gate+64up
scales per K128; two64 B-loads fill gate cols0:63/up64:127; L1 epilogue must
read p andp+64, not currentgran8 mapping. Pair row = n_block_idx*64, not
current128-rowhalf calculation. W2 stays unchanged, W1 scale index stays
n_block_idx, total TMA expected bytes unchanged. Agent implementing native
opt-in explicit layout flag/JITtag first; Python/quantizer edits held until
current validation source capture completes. BF16 transform removal stillpending.


### 2026-11-09: Direct column packing and structural regression checkpoint

The owning SM90 CTA emits column FP8 x/dy/h/dz with one scale per 128-token
M-block directly for DeepGEMM k-grouped 1D1D wgrad. This is the required
contract, not a justification to keep the current scalar packing schedule.
SM100 requires independent 32-row MXFP8 groups; that implementation remains
outstanding. Preserve row-quantized operands for activation GEMMs separately.

New paired H200 results (max-rank median / mean-rank median, milliseconds):
8 GPUs, T=16384/rank, H=2048, I=768, E=128, top-k=8, FP32 masters,
blockwise FP8, fullgraph, 5 warmup / 15 measured iterations, router excluded.
Prime external reference F/B/AC = 3.9/8.2/15.9; not precision-matched.

| Variant | Scope | Native max/mean | Same-run NCCL max/mean |
| --- | --- | --- | --- |
| FP8 input, BF16 combine v1 | F | 7.817/7.814 | 7.804/7.800 |
| FP8 input, BF16 combine v1 | B | 40.803/40.768 | 40.532/40.460 |
| FP8 input, BF16 combine v1 | F+B | 48.276/48.238 | 47.974/47.890 |
| FP8 input, BF16 combine v1 | 2F+B | 55.870/55.832 | 55.599/55.514 |
| Swizzled packing v4, BF16 wire | F | 8.031/8.028 | 8.032/8.028 |
| Swizzled packing v4, BF16 wire | B | 39.082/39.063 | 40.505/40.467 |
| Swizzled packing v4, BF16 wire | F+B | 46.831/46.812 | 48.195/48.135 |
| Swizzled packing v4, BF16 wire | 2F+B | 54.672/54.653 | 55.926/55.868 |
| FP8 input and combine v1 | F | 7.826/7.824 | 7.818/7.813 |
| FP8 input and combine v1 | B | 42.586/42.555 | 40.509/40.455 |
| FP8 input and combine v1 | F+B | 50.019/49.993 | 47.934/47.865 |
| FP8 input and combine v1 | 2F+B | 57.611/57.576 | 55.574/55.514 |

Artifacts: benchmarks/h200_fp8_wire_paired_v1.json,
h200_swizzled_dual_pack_paired_v4.json, h200_fp8_combine_paired_v1.json.
Swizzle improves AC only 0.8% versus retained v3 (different allocation).
FP8 combine worsens AC 3.1% versus input-only (different allocation).
All three requested pieces exist on SM90 but performance is structurally wrong;
do not promote or claim a win. Fresh H200 profile session63951/log
/private/tmp/ultra-h200-fp8-combine-profile-v1.log uses frozen source and vendor
/private/tmp/mono-ep-{source,vendor}-hopper-fp8-combine-v1-20261109.
Investigate persistent packing costs first; one-column-per-thread x/dy stores
are a concrete coalescing concern, not yet a measured dominant bottleneck.

Transpose-free W1T/W2T FP8 quantizer validation: initial run77370 failed only
8 fullgraph cases due to cumulative Dynamo recompile limit; 26 passed.
Reset compiler between independent parameter cases. Repeat26267 passed all34
exact Q/SF parity cases (both masters, strides, tails, Qwen, eager/fullgraph).
Artifacts h100_strided_weight_quantization_v1.json and v2.json. Integrated
quantize_transposed_weights_128x128 into backward preparation only; recompute
W1 interleave and BF16 paths remain. Training gate70546 pending, log
/private/tmp/ultra-hopper-fused-weight-training-v1.log, frozen combine-v1 vendor.

Original-layout forward native paired64 TMA path and Python paired-scale
quantizer both passed independent static review. GPU parity still required.
New paired_weight_quantization.py and test_paired_weight_quantization.py cover
exact legacy-interleave equivalence (Q bytes/SF), both masters, fullgraph,
sliced strides, tails, zero blocks and gate/up dominance. H100 gate74510
launched via modal_hopper_wgrad.py --paired-weights; log
/private/tmp/ultra-hopper-paired-weight-quantization-v1.log. No forward runtime
integration or speed claim yet. Native opt-in keeps legacy default unchanged.


Validation update: training70546 PASSED for both FP32/BF16 masters, fullgraph,
checkpointing, two outstanding forwards/reverse backward and three AdamW steps.
Artifact benchmarks/h100_fused_weight_training_v1.json. Paired-weight gate74510
PASSED all16 exact Q/SF tests; artifact h100_paired_weight_quantization_v1.json.
These validate weight preparation, not new performance claims or native paired
forward parity. H200 structural profile63951 still pending.

SM90 x/dy column packing v5 candidate now in progress: shared64-column transpose,
coalesced source reads and contiguous column output writes, preserving numerical
rounding. Existing scalar loop has two global read passes and scattered one-byte
stores. Scratch lifetime statically audited; proposed schedule adds barriers and
must earn its place in paired measurement. No assumed dominant-cost claim.


v5 x/dy shared64 implementation and independent static review complete. Frozen
vendor/source snapshots: /private/tmp/mono-ep-{vendor,source}-hopper-coalesced-dual-pack-v5-20261109.
H100 Qwen FP8 combine ring gate47444 pending, log
/private/tmp/ultra-hopper-coalesced-dual-pack-qwen-ring-v5.log; codegen28612
pending, log /private/tmp/ultra-hopper-coalesced-dual-pack-codegen-v5.log.
Cached x/dy retains row-FP8 decode -> BF16 round -> column-FP8 semantics;
this is not dual quantization independently from the original pre-transport
high-precision source. Do not imply TransformerEngine numerical equivalence.


SM100 consumer audit: csrc/apis/gemm.hpp::k_grouped_fp8_gemm_tn_contiguous
accepts recipe(1,1,32) and uses MN-major contiguous [sumK,width] Q operands;
SM90 NT consumer uses per-expert column-major K-contiguous packing. Thus
column-scaled describes the reduction axis, not identical byte layout across
architectures. Emit four 32-row scales per128 owned rows on SM100 directly in
the required UE8M0/TMA scale layout; avoid copying Hopper's transposed byte
layout or adding a runtime scale-layout transform. The current SM100 API calls
transform_k_grouped_sf_into_required_layout, so direct producer/consumer scale
ABI needs an explicit bypass or already-required-layout acceptance test.


v5 codegen28612 and ring47444 FAILED compile on ambiguous __nv_bfloat16/cute
ternary conversion, before runtime execution. Failure artifacts saved as
h100_coalesced_dual_pack_codegen_v5.json and h100_coalesced_dual_pack_qwen_ring_v5.json.
Fixed true arm with explicit BF16->float->d_dtype_t exact conversion; manifest
updated. Frozen v6 vendor/source snapshots preserve correction separately.
New cached x/dy independent exact-Q/SF harness statically reviewed:
modal_hopper_cached_xdy_packing.py (both masters, fullgraph, poisoned padding,
empty experts and unequal ranks). This checks packing, not transport correctness.


Active v6 validation jobs (no outcomes yet): cached exact-Q/SF19629,
BF16-wire Qwen direct-packing/ring17829, FP8 combine Qwen ring52137,
codegen65351. Logs respectively /private/tmp/ultra-hopper-cached-xdy-packing-v6.log,
ultra-hopper-bf16-wire-direct-packing-v6.log, ultra-hopper-fp8-combine-ring-v6.log,
ultra-hopper-coalesced-dual-pack-codegen-v6.log. Fresh H200 profile63951 remains
pending on frozen full-FP8 v1, app ap-LWa9gwzpVOjPJG4Mhywzxy. Do not rerun or
promote v6 before collecting these results. If validation passes, run paired
H200 max/mean-rank timings against same-run NCCL; preserve original profile.


Continuation: previous turn made progress (v6 packing implementation, exact
reference harness, weight quantization/training evidence), not a no-progress turn.
All five inherited job handles re-polled live; no jobs restarted.

Separate v7 candidate removes three repeated SwiGLU computations during h/dz
column packing: first epilogue writes rounded h/dg/du into dead gate/up/current-dh
shared slots after using original values for dscore. Pair mapping is exclusive;
Low preserves High dh half. Existing barrier publishes before pack; old no-column
path unchanged. Independent exact-code lifetime/rounding review clear. Frozen
vendor/source /private/tmp/mono-ep-{vendor,source}-hopper-epilogue-reuse-v7-20261109.
Keep v6 x/dy performance measurement separate from v7, no gain assumed.

Original-W1 native forward parity probe61501 launched against v6 native snapshot
(which already contains the opt-in original-layout forward), log
/private/tmp/ultra-hopper-original-w1-forward-parity-v1.log. Both masters, Qwen
geometry, routing variants, exact y and saved-projection layout parity. This
checks layout equivalence, not independent completeness of saved activation mask.


v6 codegen65351 PASSED:128 registers,104Bstack,72Bspill stores/loads, unchanged
from v4. Artifact benchmarks/h100_coalesced_dual_pack_codegen_v6.json. Numerical
gates remain pending; no performance conclusion from codegen. v7 codegen77558
running, log /private/tmp/ultra-hopper-epilogue-reuse-codegen-v7.log.


v6 cached19629, BF16-wire Qwen ring17829, FP8 combine ring52137 all PASSED.
Artifacts: h100_cached_xdy_packing_v6.json (exact Q/SF both masters),
h100_bf16_wire_direct_packing_v6.json (ring-replay reference, not exact across
independent dispatch order), h100_fp8_combine_ring_v6.json (dx exact, dW ~2.2e-4).
Original-W1 parity61501 PASSED both master dtypes/routing variants; artifact
h100_original_w1_forward_parity_v1.json (exact y and saved projection layout).

Paired H200 v6 timing81099 running; log
/private/tmp/ultra-h200-coalesced-dual-pack-paired-v6.log, frozen v6vendor AND
v6Python. Compared with older fullFP8 v1 this includes v4swizzle, coalescedx/dy,
and transpose-free backward weight preparation. Do not attribute aggregate gain
to only x/dy. v7 timing must use same v6Python to isolate epilogue reuse.

Root integrated original-layout W1 quantization/explicit native flag in Hopper
blockwise forward only. BF16/MXFP8 and backward recompute W1 still unchanged.
Pinned Ruff passes; training8809 pending using frozenv6native plus current Python,
log /private/tmp/ultra-hopper-original-w1-integrated-training-v1.log. Newforward
integration is NOT in v6/v7 frozen Python timing snapshots.

v7 direct-packing Qwen ring17289 pending, log
/private/tmp/ultra-hopper-epilogue-reuse-direct-packing-v7.log. Probe mounts live
Python; runtime forward integration may be present, so this is correctness only.


v7 FP8 combine ring16077 running, log
/private/tmp/ultra-hopper-epilogue-reuse-combine-ring-v7.log. Exact runtime.py
original-W1 integration reviewed clear. Pending handles63951/81099/77558/17289/
16077/8809 revalidated live at turn end; continue these rather than restarting.
Next: collect v7 correctness/codegen + original-W1 compiled training; ifv7passes,
launch pairedH200 using v7native and v6Python to isolate epilogue reuse. Collect
freshv1profile and v6timing, report max/meanrank and same-runNCCL. No new latency
result yet. Broad SM100 MXFP8 packed wgrad, BF16 no-transform paths, and all other
ROADMAP requirements remain incomplete; goal stays active.


v7 codegen77558 PASSED but spills increased:128regs,144Bstack,132Bspill loads/
stores versus v6 104/72/72. Artifact h100_epilogue_reuse_codegen_v7.json.
Removing repeated math is not by itself evidence of a speedup; await numerical
checks and paired timing before promotion. Previous turn made progress (v6 GPU
parity, v7 implementation, original-forward integration), no blocker audit.


v7 direct17289 and combine16077 PASSED Qwen ring validation, both masters;
artifacts h100_epilogue_reuse_direct_packing_v7.json and
h100_epilogue_reuse_combine_ring_v7.json. Ring reference is independent replay
so operand equality is numerical, not bitwise; combine dx exact, dW ~2.2e-4.
Original-W1 integration8809 PASSED fullgraph checkpointed training, FP32/BF16
masters, reverse outstanding forwards,3AdamWsteps; artifact
h100_original_w1_integrated_training_v1.json.

v7 pairedH200 timing3426 submitted with v7native + SAME frozen v6Python as
v6timing81099; log /private/tmp/ultra-h200-epilogue-reuse-paired-v7.log.
This isolates native h/dz reuse between v6/v7, excluding newforwardintegration.
H200profile63951 remains pending; retain job. Sol auditing no-transform Hopper
BF16forward/backward prerequisites without native edits yet.


## Parked Hopper BF16 weight-layout work

SM90 BF16 forward can consume original W1 with the same paired 64-column
gate/up loads validated for FP8: two K64xN64 TMAs fill the existing K64xN128
shared tile, and the epilogue reads canonical columns `p` and `p + 64`.
W2 is already in its native forward layout. Backward recompute can use the same
paired W1 path; dH and dX can read original W2/W1 through SM90 BF16 MN-major B
descriptors, which are already supported by the generic SM90 BF16 GEMM. This
would remove interleave and transpose copies, while FP32-master to BF16
conversion would remain. Stage this behind explicit opt-ins and validate
shared-expert handling, expert-aware descriptor strides, saved projections,
empty/skewed routing, and ring reuse before changing runtime defaults.


### User priority redirect: diagnose persistent throughput before more packing

Park swizzle/packing and no-transform layout work. Preserve all candidates and
artifacts; no new packing tuning runs. v6/v7 paired runs had already finished
when cancellation status was checked, so no jobs were stopped.
Order: (1) trap-only timing, (2) no-remote-traffic matched per-GPU math and ncu
stall reasons on real multi-rank persistent run, (3) only if producer-bound,
deepen in-flight pulls/larger chunks/remove store-before-next-load and move
x/dy packing to non-math warp, (4) fix ignored setmaxnreg regardless with SASS
proof and legal CTA register budget, (5) plain grouped dH/recompute/dX math
roofline on packed rows. Use H100 for dev/correctness; H200 final Prime
comparison. Timing equivalence between H100/H200 has NOT been established.

v6 pairedH200 completed: F8.013/8.011, B31.808/31.793, F+B39.464/39.444,
AC47.179/47.149 (max/meanrankmedian). NCCL7.994/7.978,40.541/40.490,
48.153/48.095,55.952/55.898. v7: F8.027/8.022,B31.644/31.615,
F+B39.296/39.270,AC46.948/46.925; NCCL8.046/8.042,40.559/40.477,
48.117/48.063,56.109/55.891. Prime3.9/8.2/15.9 external unmatched.
Artifacts h200_coalesced_dual_pack_paired_v6.json and h200_epilogue_reuse_paired_v7.json.

Trap-onlyH20053322 completed with same frozenv6native/Python, separateallocation:
F8.110/8.108,B32.428/32.406,F+B40.110/40.091,AC47.932/47.907;
NCCL8.095/8.085,40.551/40.495,48.465/48.411,55.878/55.827.
Artifact h200_trap_only_paired_v6.json. No large speedup observed. Removing
C7510 does not fix ignored register redistribution; both remain separate claims.

H100 matched local-only6941 PASSED:16experts,T16384,top8 ->131072receivedrows,
nativehostentryCUDAevent median29.12694ms (15samples,5warmups). Excludes weight
preparation/dW, retains local dispatch/combine. Artifact h100_single_rank_trap_v6.json.
Do not compare this directly to H20019ms as hardware/scope equivalence.
Ncu88926 failed clocklocking, not missingprofiler; NCU2025.3.1.0 present.
Saved h100_single_rank_ncu_v6_failed_clock.json; retry83360 uses --clock-control
none. Need realmulti-rank profiling with supported concurrent-kernel replay;
default ncu serialization can deadlock mandatory peer work. NVIDIA CLI docs
https://docs.nvidia.com/nsight-compute/NsightComputeCli/ describe shmemcommunicator
and lockstep options; check installedversion supports them before fullrun.

H100 synthetic prepacked groupedGEMM roofline2540 running, log
/private/tmp/ultra-h100-backward-gemm-roofline-v1.log. Geometry corrected from
incorrect16384rows to131072receivedrows: sourceT is perrank.16localexperts,
H2048,I768; allprep outside timings. Only totalFLOPs/summedtime is combined
TFLOPS. Actualpersistent-cache mode follows after sourcecapture; no fullmath
correctness or actualrouting-parity claim from syntheticfloor.

Astra preparing standalone compiler experiments for legal full512 CTA dynamic
registerallocation, including64/56/192/192 budget and earlier redistribution.
Do not adopt based on PTX intent alone: require retained SETMAXNREG SASS and
spills plus runtimecorrectness. Production native unchanged during experiments.


### Pause and restart handoff (2026-11-09)

User requested pause after matched H100 results. See the hand-off note
(HANDOFF-2026-11-09.md, kept untracked beside the benchmark artifacts) for scope, frozen snapshots, completed diagnostics, failures, and pending work.
Do not launch further optimization/diagnostics before user resumes.
Modal H100 requests can automatically upgrade to H200: first four paired jobs
were H200, preserved as h200_auto_upgrade_* artifacts. Strict comparison uses
gpu="H100!:8" and validates allocated GPU before work. First real H100 v6:
F8.393/8.390 B32.506/32.477 F+B40.422/40.391 AC48.539/48.516;
NCCL AC59.733/59.646 (max/mean-rank median ms). Other variants pending.
Single-rank NCU completed; real distributed profile still pending. New profiler
capability probe failed because executable selection chose ARM binary.
Actual-cache roofline and fixed register compiler fixture are ready but unrun.

All four strict H100 jobs finished successfully; PAUSED at user request.
See benchmarks/hopper_comparison_20261109.md for full native/control F/B/F+B/AC
max/mean-rank tables and H100/H200 deltas. Native AC max-rank medians:
v1 59.816, v6 48.539, trap-v6 49.116, v7 48.412 ms. H100 overhead2.5–3.8%;
NCCL control overhead6.5–7.2%. Both hardware labels verified in artifacts.
Handoff finalized in that hand-off note; do not resume automatically.
