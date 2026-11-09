# mono-ep

Independent open-source MoE training kernels and expert-parallel scheduling.

Experimental SonicMoE- and DeepGEMM-based MoE training kernels. This standalone package has no
Axolotl, Transformers, or TRL dependency. Axolotl's experimental expert-parallel FP8 adapter
imports this package; model configuration and FSDP policy stay in Axolotl.

The common training entry point is `from mono_ep import MonoEP`:

```python
runtime = MonoEP(dtype="fp8", compile_local=True, retain_activations=True)
y = runtime(x, indices, scores, w1, w2, group=ep_group)
y.float().square().mean().backward()
```

`dtype="fp8"` selects DeepGEMM blockwise FP8 on Hopper or MXFP8 on SM100.
With the default `backend="pipeline"`, `dtype="bf16"` selects SonicMoE.
Both dtypes use the same NCCL chunk scheduler;
the experimental peer-ready queue and UniEP work are not selected here yet.
Compilation covers local expert callbacks, not whole-graph communication.
The Axolotl FP8 adapter now delegates to this entry point. Its existing explicit
format configuration remains accepted and checked for hardware compatibility.
SM120 remains unsupported by this entry point.

Select `MonoEP(backend="persistent", dtype="bf16", retain_activations=False,
max_tokens=16384)` to use native persistent forward/backward through the same
call interface. Runtime buffers are cached per process group, device and
geometry within that engine. Reuse the engine across compatible layers rather
than constructing one per call. All ranks must prepare geometries in the same
order. Selection/cache behavior has CPU coverage. Its prepared native runtime also
passes eight-B200 BF16-master compiled/checkpointed training; see
[public-entry validation](benchmarks/b200_native_bf16_master_ep8_training_validation.json).
That check uses BF16 compute. On Hopper, `dtype="fp8"` selects native
blockwise E4M3 compute with FP8 forward-input transport, BF16 backward
transport, cached FP8 activations and GPU-offset grouped weight gradients.
[Compiled checkpointed training](benchmarks/h100_fp8_full_cache_native_training.json)
passes with both FP32 and BF16 master weights; this is a short numerical and
optimizer validation, not a long-training convergence result.

For fullgraph compilation, allocate outside tracing with
`engine.prepare_persistent(group=ep_group, device=x.device, experts=128,
topk=8, hidden=2048, intermediate=768)`, then compile that returned runtime.
`compile_local` applies only to pipeline callbacks. Native retention is not yet
implemented; unsupported retention/dtype/chunk combinations raise an error.
Call `engine.close()` only after graphs using its runtimes have finished.

The fixed-geometry runtime is also available directly:

```python
from mono_ep import PersistentMoE

runtime = PersistentMoE(
    ep_group, experts=128, max_tokens=16384, topk=8,
    hidden=2048, intermediate=768,
    compute_dtype="bf16", retain_activations=False,
    backward_backend="native",
)
y = runtime(x, indices, scores, w1, w2)
y.backward(dy)
# Release only after all graphs using this runtime have finished.
runtime.close()
```

Native backward supports single-host SM90 BF16/blockwise FP8 and SM100 BF16
recomputation with H and I divisible by 256. One CTA on SM90, or a CTA pair on
SM100, owns each routed M-block through dH, gate/up recomputation,
SwiGLU/dscore and dX/combine. Both use native dispatch and TMA pipelines;
Hopper uses WGMMA and Blackwell uses UMMA. dH stays CTA-local; h/dZ are
written to the workspace for grouped weight gradients. The constructor fails
explicitly for unsupported native combinations. `backward_backend="nccl"`
(the default) selects the fixed-capacity grouped reference path. The historical
`MonoEP` name remains an alias for `PersistentMoE`.

Hopper's direct weight-gradient path now emits column-scaled FP8 operands from
the owning block, using 128-token scale groups. BF16 staging is bounded by the
ring size; h/dZ column packing runs in the shared epilogue, while the existing
row-scaled dZ cache feeds dX. Numerical validation and compiler experiments are
recorded in the development log kept outside the tree; its H200 timings predate this change.

To reserve SMs, call `deep_gemm.set_num_sms(physical_sms - reserved_sms)` before
constructing `PersistentMoE(..., reserved_sms=reserved_sms)`. The SM budget is
fixed for the lifetime of its buffers; later changes are rejected. SM100 requires
an even active count. A two-H100 test with 16 reserved SMs and `NCCL_MAX_CTAS=4`
verified concurrent progress with three NCCL streams during both forward and
backward, plus 20 numerical/reuse checks. See the
[Hopper overlap result](benchmarks/h100_reserved_sms_dual_pack_v1.json).
This is a kernel concurrency check, not an end-to-end FSDP or SM100 validation.

The native path passed fullgraph/checkpointed training with three AdamW steps,
two outstanding forwards backpropagated in reverse order, unequal rank token
counts and empty experts. See
[training validation](benchmarks/b200_native_persistent_training_validation.json).
The compiled correctness and timing runs are separate. The
default `MonoEP` mode and Axolotl FP8 adapter still select the NCCL chunk
pipeline; persistent execution requires explicit selection.

A same-allocation eight-B200 comparison used 16384 tokens/rank,
H2048/I768/E128/top8, FP32 master weights, BF16 compute, five warmups and
fifteen measured samples, with whole-layer fullgraph compilation. Router and optimizer are excluded; grouped weight
gradients and runtime staging are included:

| Median latency | Fixed-capacity NCCL | Grouped peer | Native persistent |
| --- | ---: | ---: | ---: |
| Forward | 2.33 ms | 2.33 ms | 2.33 ms |
| Backward | 20.77 ms | 13.62 ms | 6.28 ms |
| Recomputing F+B | 22.70 ms | 15.62 ms | 8.37 ms |
| Checkpointed 2F+B | 24.86 ms | 17.80 ms | 10.60 ms |

See [compiled raw samples](benchmarks/b200_native_peer_nccl_compiled_paired_timing.json).
This NCCL control is not SonicMoE, and these B200/BF16 measurements are not a
matched comparison with Prime's H200/FP8 results.

The Hopper activation-cache implementation also has a same-allocation eight-H200
comparison at the same Qwen3-30B-A3B geometry, with blockwise FP8 compute,
FP32 masters, fullgraph compilation, five warmups and fifteen samples:

| Median latency | Same-forward/NCCL-backward control | Native persistent |
| --- | ---: | ---: |
| Forward | 7.86 ms | 7.87 ms |
| Backward | 40.50 ms | 32.18 ms |
| Recomputing F+B | 47.94 ms | 39.82 ms |
| Checkpointed 2F+B | 55.59 ms | 47.36 ms |

See [raw samples and memory inventory](benchmarks/h200_fp8_full_cache_paired_timing.json).
The previous native checkpointed result was 212.97 ms; activation reuse,
FP8 forward-input transport and direct quantizer output writes together reduce
that to 47.36 ms (4.50x faster). This is not an isolated ablation of each change.
Prime reports 15.9 ms for its checkpointed H200 FP8 layer; that target is not
reached. Both local variants share the same persistent forward, so the control
is not Prime's complete baseline. Router and optimizer remain excluded.

Explicit local workspace inventory is 23.19 GiB, including 5.99 GiB of backward
FP8 caches. Native peak PyTorch allocation is 23.86 GiB; all paired backends
are resident and external allocations may be absent from this allocator metric.
The inventory and allocator peak are separate measurements and must not be added.

An experimental 384-thread Hopper forward layout removes compiler spills but
provides a modest measured gain: a separate same-allocation comparison gives
8.06 → 7.78 ms forward (3.5%) and 47.65 → 47.44 ms checkpointed (0.5%), with
backward unchanged. See [paired layout samples](benchmarks/h200_fp8_forward_layout_paired_timing.json).
The default remains the validated 512-thread layout. Compact forward and backward have passed raw two-H100 gradient/cache validation
and [compiled checkpointed training](benchmarks/h100_fp8_compact_native_training.json)
with FP32/BF16 master weights. A subsequent [three-layout comparison](benchmarks/h200_fp8_kernel_layout_paired_timing.json)
rejects compact backward for this shape: backward increases from 32.19 to 48.37 ms,
and checkpointed latency from 47.10 to 63.29 ms with compact forward fixed.
The 512-thread backward remains the default. Compiler spill counts alone do not establish a speedup.

`PersistentMoE(..., compute_dtype="mxfp8")` selects the existing fused MXFP8
forward with grouped FP8 backward. Hopper's `compute_dtype="blockwise"` uses
native FP8 forward and accepts native or NCCL backward with recomputation.
Native SM100 FP8 backward, FP8 backward peer transport, SM120, multi-node transport,
FSDP communication reservations and adapter integration remain unfinished. The package is experimental and is
not yet equivalent to the complete Prime training implementation.

`mono_ep.quantization.quantize_fp8` prepares E4M3 matrices and FP32
power-of-two scales for 1x32, 1x128, and 128x128 blocks, preserving independent
expert scales. CPU format tests pass, and B200 eager and full-graph Inductor
conversions match the pinned DeepGEMM reference exactly for BF16/FP32 inputs.
Reports are in `benchmarks/fp8_quantization_validation.json`.
FP8 training integration remains unfinished. The returned
scales still require DeepGEMM's architecture-specific layout transform.

`mono_ep.fp8.expert_gemm_fp8` now connects quantization to DeepGEMM grouped
GEMMs, including scale layout conversion inside the custom operator. H100
blockwise FP8 and B200 MXFP8 pass eager and full-graph checks for empty inputs
and two packed experts with padding. Against dequantized FP8 matrix products,
relative L2 error was 0.00070 on H100 and zero on B200; both differed from BF16
by approximately 0.038. See `benchmarks/fp8_gemm_validation.json`. This primitive
does not itself register autograd.

The explicit `mono_ep.ops.weight_grad_fp8` primitive computes FP32 expert
weight gradients with FP8 operands. Eager and full-graph checks pass on H100 and
B200, including empty experts and padding. B200 also passes with GPU cumulative
group counts; Hopper still requires host group sizes and per-expert transposes.
Relative L2 error against dequantized FP8 matrix products was 0.00012 on H100
and approximately 2e-8 on B200. Reports are in
`benchmarks/fp8_weight_grad_validation.json`.

Hopper weight-gradient operand preparation now uses a Triton kernel that fuses
per-expert transpose, scale reduction, FP8 conversion, and concatenated packing.
Exact value/scale checks against the prior PyTorch preparation pass in eager and
compiled modes, including empty experts; see
`benchmarks/fused_wgrad_quantization_validation.json`.

`expert_forward_fp8` and `expert_backward_fp8` connect these primitives into
explicit local expert computation and derivatives, including routing-score
gradients and FP32 weight-gradient output. H100 blockwise FP8 and B200 MXFP8 pass
eager/full-graph checks for masked and entirely unrouted inputs. On the tested
137-token layer, FP8 output and gradient relative L2 differences from BF16 were
approximately 0.058–0.069. With `bf16_forward=True`, forward outputs matched BF16
and gradient differences were approximately 0.033–0.054. These are approximate
quantized derivatives, not differentiation through rounding. Reports are in
`benchmarks/fp8_expert_validation.json`. The complete distributed autograd entry
point, training convergence, and performance still need validation. Hopper's
packing layout is built outside the compiled region and reads counts on the host.

`mono_ep.fp8_autograd.fp8_experts(x, indices, scores, w1, w2, format,
bf16_forward=False)` provides a local autograd entry point using these explicit
derivatives. Eager and full-graph capture checks pass on Hopper and B200 with
BF16/FP32 master weights, masked/unrouted inputs, and either forward precision.
Results are in `benchmarks/fp8_autograd_validation.json`. This captures opaque
forward/backward custom operators; it does not compile their internal Python
operations or eliminate Hopper's host synchronization. Only first-order
derivatives are implemented. It is not the optimized explicitly compiled path
used for the H200 timing results below, and has not yet been benchmarked as a
distributed training entry point.

`FP8Experts` exposes that local forward plus an explicit `.backward` method for
the communication scheduler:

```python
from mono_ep import FP8Experts, pipelined_moe

compute = FP8Experts("blockwise")  # "mxfp8" on SM100
y = pipelined_moe(
    x, global_indices, scores, local_w1, local_w2,
    group=ep_group, local_kernel=compute, local_backward=compute.backward,
)
```

Construct `FP8Experts(..., compile_local=True)` to fullgraph-compile the explicit
forward-with-activations and backward callbacks. Layout construction and NCCL
remain outside the compiled region; the standalone `__call__` autograd boundary
is unchanged. Different routing shapes, Hopper host group sizes, dtypes, and
forward precisions can specialize separate graphs. The validation matrix resets
Dynamo between independent configurations; it does not establish bounded
recompilation during changing-routing training.

Pass `local_forward=compute.forward_with_activations` alongside
`local_backward=compute.backward` to retain local expert activations across
forward/backward. The scheduler saves each chunk's tensors through
`save_for_backward`; backward reuses the packed inputs, projection, and weighted
activation. Omitting `local_forward` preserves the recomputing mode. This option
currently applies to the eager distributed scheduler, not the standalone
`fp8_experts` compiler boundary. Two-rank Hopper and B200 checks cover
both retained and recomputing modes (64 rank/case checks total), plus four
SonicMoE regression checks; records are in
`benchmarks/retained_pipeline_validation.json`.

Two-GPU distributed autograd checks pass on Hopper and B200 for BF16/FP32
masters, FP8 or BF16 forward, and serial/overlapping backward schedules. Three
chunks include an entirely unrouted first chunk. Callback counters verify that
backward avoids reconstructing a full local forward graph. Weight gradients
accumulate in FP32 across chunks before conversion to the master dtype. The
existing SonicMoE pipeline also passes regression checks. Results are in
`benchmarks/fp8_pipeline_validation.json`. This establishes distributed first-order
gradients, not whole-layer compilation, convergence, or model/FSDP integration.

## Development

Install this package with `pip install -e packages/mono-ep`. Build the vendored
DeepGEMM separately in a CUDA development environment with C++20 `<format>`
support, CUDA 12.9 or newer, PyTorch, setuptools, wheel, packaging, and ELF/DWARF
development headers:

```bash
python packages/mono-ep/tools/fetch_build_dependencies.py
DG_USE_LOCAL_VERSION=0 DG_FORCE_BUILD=1 pip install --no-build-isolation --no-deps packages/mono-ep/vendor/deepgemm
```

GPU validation with Modal:

```bash
modal run packages/mono-ep/examples/modal_validate.py --gpu H100
modal run packages/mono-ep/examples/modal_validate.py --gpu B200
modal run packages/mono-ep/examples/modal_validate.py --gpu B200 --ep
modal run packages/mono-ep/examples/modal_validate.py --gpu B200 --compare
modal run packages/mono-ep/examples/modal_validate.py --gpu B200 --quantization
modal run packages/mono-ep/examples/modal_validate.py --gpu H100 --fp8-gemm
modal run packages/mono-ep/examples/modal_validate.py --gpu B200 --fp8-gemm
```

`--compare` runs eager/compiled PyTorch grouped GEMMs, eager/compiled SonicMoE,
and both fused/unfused backward variants of this prototype in separate processes on one B200,
with identical fixed routing and BF16 parameters, timing forward plus backward.
It is an EP=1 layer benchmark without checkpointing, not a
multi-GPU or end-to-end training comparison. No speedup over SonicMoE is assumed.
The initial host-packed prototype took 12.81 ms versus SonicMoE's 1.87 ms for
4096 tokens, hidden size 2048, expert intermediate size 768, 128 experts, top-8.
GPU packing and removal of EP=1 collectives reduced this to 8.53 ms; SonicMoE
measured 1.84 ms on that repeat. The remaining gap motivates deeper epilogue and
communication fusion. These results do not measure Prime's implementation.
Raw measurements are in `benchmarks/b200_bf16_ep1.json`.

The BF16 SM100 backward epilogue replaces the
materialized activation gradient with recomputed weighted activations, writes
gate/up gradients, and reduces routing-score gradients through deterministic
partial sums. B200 eager and local full-graph checks pass, as do two-B200 gradient
checks with BF16 and FP32 master weights. Hopper's separate activation path also
passes regression checks. Reports are in `benchmarks/bf16_validation.json`.
The measurements above predate this epilogue extension. A same-B200 comparison
measured 8.47 ms without the epilogue and 9.24 ms with it; SonicMoE measured
1.42 ms. The epilogue is therefore opt-in through `MonoEP(...,
fuse_activation=True)` while further tuning is needed. Removing an intermediate
tensor alone did not improve performance in this workload.

A subsequent matched run measured PyTorch grouped GEMMs at 3.31 ms eager and
1.83 ms compiled; SonicMoE at 1.59 ms eager and 1.83 ms compiled; the prototype at
8.51 ms unfused and 9.24 ms fused. Compilation uses `fullgraph=True` and includes
the generated backward. PyTorch baseline outputs and gradients are checked
against an independent expert-loop reference before timing. These BF16 EP=1
measurements do not predict Prime's FP8 EP=8 checkpointed benchmark.

## Source and licensing

New package code is Apache-2.0. `vendor/deepgemm` retains DeepSeek's MIT license;
its revision and build dependency revisions are recorded in `vendor/sources.json`.
The snapshot excludes upstream tests and CI. Local epilogue extensions are
recorded in `vendor/sources.json`; the snapshot is no longer unmodified. Build dependencies
are fetched explicitly and retain their own licenses. No UniEP code has been
copied into this package. Any future reuse must preserve the applicable notices.
