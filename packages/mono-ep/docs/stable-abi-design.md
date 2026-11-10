# mono-ep on stable ABIs: design, prototype, packaging and CI

Status: design. The one-op prototype it measures stays on `features/mono-ep-stable-abi` (off
`features/mono-ep` @ 9ae994be97, commit c7aaf5fc2e) and is not part of this tree; its result file is
kept with the untracked benchmark artifacts. Nothing here is meant to merge as is; the layering
refactor (stateless `ops` + `_fake_impls` + `functional` + `api`) lands first and this plan slots in
behind it (§6.4).

Goal: build `mono-ep` once per CUDA major, publish it to PyPI as `cp310-abi3` wheels and to the
Hugging Face Hub as a `kernel` repo, and stop compiling `deep_gemm._C` per torch version in CI.

Scope of the artifact today: `deep_gemm._C` (vendored DeepGEMM host extension, `vendor/deepgemm/csrc`,
one `python_api.cpp` plus 48 headers, 11.7k lines) and `mono_ep.owned_backward` (one `.cu` + one
pybind shim built at runtime with `cpp_extension.load`). Device code is JIT-compiled by DeepJIT from
headers shipped with the package; that part is unchanged by this design and dominates the packaging
story (§4).

---

## 1. Dependency inventory of the host extension

Method: `grep` over `vendor/deepgemm/csrc` (counts are occurrences per file; a file with zero torch,
ATen, c10 and pybind references is omitted) plus a read of every runtime/API header. 44 of 54 source
files reference torch, ATen, c10 or pybind. The ten torch-free files are `apis/config.hpp` (pybind
only), `jit_kernels/heuristics/{common,mega_gate,mono_ep,runtime,utils}.hpp`, `runtime/jit.hpp`,
`runtime/mlopart.hpp` (pybind only), `utils/compatibility.hpp`, `utils/exception.hpp`; the three
arch-spec headers `heuristics/{config,sm90,sm100}.hpp` are torch-free too.

### 1.1 Per-file table

| file | `torch::Tensor` | `from_blob` | factories | stream | capture check | pybind | `m.def` | cuBLASLt | c10 cuda |
|---|---|---|---|---|---|---|---|---|---|
| apis/gemm.hpp | 120 | 0 | 2 | 0 | 0 | 137 | 33 | 0 | 0 |
| apis/attention.hpp | 41 | 2 | 9 | 2 | 1 | 23 | 8 | 0 | 3 |
| apis/mono_ep.hpp | 37 | 13 | 1 | 0 | 0 | 12 | 10 | 0 | 0 |
| apis/einsum.hpp | 28 | 0 | 1 | 1 | 0 | 8 | 2 | 0 | 1 |
| apis/mega_mhc.hpp | 23 | 0 | 2 | 2 | 1 | 28 | 1 | 0 | 3 |
| apis/mega_gate.hpp | 17 | 0 | 3 | 2 | 1 | 15 | 1 | 0 | 3 |
| apis/locality_domain.hpp | 11 | 2 | 5 | 0 | 0 | 4 | 0 | 0 | 0 |
| apis/layout.hpp | 7 | 0 | 1 | 0 | 0 | 10 | 8 | 0 | 0 |
| apis/epilogue_class.hpp | 5 | 0 | 0 | 0 | 0 | 24 | 0 | 0 | 0 |
| apis/hyperconnection.hpp | 4 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 |
| apis/config.hpp | 0 | 0 | 0 | 0 | 0 | 1 | 11 | 0 | 0 |
| jit_kernels/impls/epilogue_class.hpp | 39 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | 3 |
| jit_kernels/impls/smxx_cublaslt.hpp | 13 | 0 | 0 | 1 | 0 | 0 | 0 | 31 | 10 |
| jit_kernels/impls/smxx_layout.hpp | 10 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| jit_kernels/impls/runtime_utils.hpp | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| jit_kernels/impls/sm90_fp8_mono_ep_forward.hpp | 7 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| jit_kernels/impls/* (other 24 launchers) | 3–31 each, 301 total | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| runtime/runtime.hpp | 4 | 0 | 2 | 2 | 0 | 0 | 0 | 5 | 4 |
| runtime/locality_domain.hpp | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| runtime/mlopart.hpp | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| utils/layout.hpp | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| utils/math.hpp | 0 (`torch::kInt8` constant) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Totals: 873 `torch::Tensor` mentions, 20 `from_blob`, 35 allocating factory calls
(`empty` 18, `zeros` 9, `arange` 3, `empty_strided` 3, `empty_like` 2), 7 current-stream sites,
4 graph-capture checks, 75 `m.def`, 9 pybind classes + the DeepJIT `Runtime` class.

### 1.2 By category

**Tensor arguments (873 refs, every launcher).** Launchers consume `data_ptr()`, `size(i)`,
`stride(i)`, `scalar_type()`, `is_contiguous()`, `numel()`, `device()` and nothing else from the
tensor. The TMA helpers in `runtime_utils.hpp` (`make_tma_2d_desc`, `make_tma_3d_desc`,
`make_tma_4d_desc`, `make_tma_weights_{2d,3d}_desc`, `make_tma_sf_desc`) are the central users and
are the first thing to port (the prototype re-implements the two it needs on a 40-byte `View`).

**Factories / allocation (35 sites).** Three kinds:

- Output allocation inside the API layer (`apis/gemm.hpp`, `apis/layout.hpp`, `smxx_layout.hpp`:
  `torch::empty`/`empty_like` for the result of `transform_sf_into_required_layout`, the k-grouped
  size/offset helpers, etc.). Route A moves these to Python (the op returns into a caller-provided
  output, or the Python wrapper allocates and passes it in); route B rewrites them with
  `torch::stable::new_empty`.
- Per-call scratch (`torch::zeros({num_sms}, kUInt8)` for `sm_locality_domains` on SM90 in
  `apis/mono_ep.hpp`, the per-call `workspace` in `apis/attention.hpp`, `torch::arange` in
  `locality_domain.hpp`). These become explicit workspace arguments, which is exactly what the MoK-style
  `ops` layer wants anyway.
- Lazily-created persistent buffers guarded by a capture check (next item).

**Streams (7 sites).** `at::cuda::getCurrentCUDAStream()` in `apis/attention.hpp:291`,
`apis/mega_gate.hpp:14`, `apis/mega_mhc.hpp:27`, `apis/einsum.hpp:33`,
`jit_kernels/impls/epilogue_class.hpp:125`, `smxx_cublaslt.hpp:75`, and `runtime/runtime.hpp`
(cuBLASLt workspace map keyed on `c10::cuda::CUDAStream`). Every kernel launch goes through
`jit->launch(kernel, LaunchOptions{...})` whose `stream` is `std::optional<CUstream>`; when unset,
DeepJIT's `kernel.hpp:166` falls back to `at::cuda::getCurrentCUDAStream().stream()`. All mono-ep
launchers leave it unset today. The fix is mechanical: thread an explicit `cudaStream_t` from the
binding into `LaunchOptions.stream` (the prototype does this).

**Graph-capture checks (4 sites).** `c10::cuda::currentStreamCaptureStatusMayInitCtx() ==
CaptureStatus::None` at `attention.hpp:297` (16 MiB workspace), `mega_gate.hpp:20` (score barriers),
`mega_mhc.hpp:33` (split barriers) and `epilogue_class.hpp:130` (`FP8CastWithClamp` per-stream
`ready` map). All four are the same idiom: "allocate and zero a persistent per-stream/per-shape
buffer the first time, but refuse to do so while a graph is being captured". None of the mono-ep
training ops use these four APIs. In the stable design the buffer lives in the Python `functional`
layer (allocated eagerly at warm-up, passed as a workspace argument), and the C library exposes
`cudaStreamIsCapturing(stream)` only as a defensive assertion. There is no torch stable-ABI query for
capture status (§2.2), so route B needs the same move.

**pybind classes holding state.** `Runtime` (DeepJIT, `get_jit()`), and the 9 epilogue classes in
`apis/epilogue_class.hpp` (`Epilogue`, `SwiGLUForward`, `SwiGLURecomputeBackward`, `SwiGLUBackward`,
`PeerScatterStoreBF16`, `Identity`, `Alpha`, `FP8CastWithClamp`, `BF16StochasticRounding`). Each is a
small record of `torch::Tensor` members plus scalars that the GEMM launcher reads to build the
kernel template string and the launch arguments. They carry no device-side state except
`FP8CastWithClamp`'s lazily created `ready` tensor. In the C ABI they become plain structs
(`mono_ep_epilogue { int32 kind; mono_ep_tensor a, b, c; float alpha; ... }`) built per call in
Python; no opaque handle is needed. The only true opaque handle is the JIT runtime itself, which is
process-global behind `mono_ep_capi_init` and never crosses the boundary.

**cuBLASLt.** `runtime/runtime.hpp` (`cublasLtCreate/Destroy`, optional
`at::cuda::getCurrentCUDABlasLtHandle()` reuse, 16 MiB workspace `torch::Tensor` per
`CUDAStream`) and `smxx_cublaslt.hpp` (31 refs: matmul descriptors, preference, heuristic, the
`smxx_cublaslt_gemm` path used by `bf16_gemm_*` when DeepGEMM decides to defer to cuBLASLt). The host
`.so` does not link `libcublasLt`; the symbols resolve at load time because torch has already loaded
it `RTLD_GLOBAL`. Stable design: the C library owns one `cublasLtHandle_t` per device, keeps a
`stream -> workspace` map allocated with `cudaMallocAsync` (or takes the workspace as an argument),
and a `mono_ep_capi_set_cublaslt_handle()` lets the host hand in torch's handle. mono-ep's training
path does not hit cuBLASLt (grouped/k-grouped/mono-ep kernels are all DeepJIT), so this can be
deferred.

**Locality domains / mlopart.** `runtime/locality_domain.hpp` uses `c10::cuda::current_device()` and
the driver VMM API (`cuMemCreate`, `cuMemMap`, `cuMemImportFromShareableHandle`, over DeepJIT's lazy
dlsym table); `runtime/mlopart.hpp` *imports a Python module from C++*
(`pybind11::module_::import("deep_gemm.utils.mlopart")`) to obtain an IPC fd. `apis/locality_domain.hpp`
wraps allocations in `torch::from_blob` with a custom deleter. This is the one piece that cannot be
made torch-free by mechanical substitution: SM100 locality-domain memory becomes a host-supplied
allocator callback (`mono_ep_capi_set_allocator(alloc_fn, free_fn, ctx)`) or a Python-side
allocation whose pointer is passed in. SM90 does not use it.

**Python-side state outside `_C`.** `deep_gemm.mega.SymmBuffer` (`symm_mem.rendezvous` /
`buffer_ptrs`, views via the `slice` lambda returned by `get_symm_buffer_size_for_mono_ep`),
`_C.init(package_dir)`, `set_num_sms/get_num_sms`, `set_pdl`, heuristics flags; on the mono-ep side
`runtime.PersistentMoE` (`_RUNTIMES[int runtime_key]`), ~30 `torch.library.custom_op`s with inline
fakes, and `owned_backward.py` (`cpp_extension.load` of a pybind shim over an already torch-free
`int launch(std::vector<uintptr_t>, int, int, int, int, uintptr_t stream)`).

### 1.3 DeepJIT

`third-party/deep_jit` (header-only, pinned 2efdab42) depends on torch in exactly two places:
`backend/cuda/kernel.hpp:166` (current-stream fallback, only reached when `LaunchOptions.stream` is
unset) and `utils/gil.hpp` (pybind11 `gil_scoped_release` guarded by `Py_IsInitialized()`), plus the
optional `python_api.hpp` that registers `Runtime` with pybind. Everything else is CUDA driver API
through a lazily `dlopen`ed `libcuda.so.1` (`cuLibraryLoadFromFile`, `cuLaunchKernelEx`,
`cuTensorMapEncodeTiled`, `cuMem*`) and `libdw` for symbolised backtraces. Making it torch-free is a
two-line patch behind a macro (`DJ_NO_TORCH`): drop the stream fallback (assert instead) and compile
`GilScopedRelease` to a no-op.

### 1.4 What becomes an opaque handle

| today | stable design |
|---|---|
| `deep_gemm._C` module globals (`jit`, `runtime`, `heuristics_runtime` `LazyInit`s) | process-global inside `libmono_ep.so`, initialised by `mono_ep_capi_init(library_root)`; never exposed |
| `Runtime` pybind class | not exposed (debug accessors `get_cache_dir`, `get_nvcc_version` as C functions if needed) |
| epilogue pybind classes | plain C structs, no lifetime |
| `SymmBuffer` + `slice` lambda | Python object holding the `symm_mem` handle; byte offsets of the views come from `mono_ep_bf16_symm_buffer_layout()` (prototype) rather than from `torch::from_blob` in C++ |
| `PersistentMoE` / `runtime_key` | unchanged Python-side registry (int key is fine for `torch.library`) |
| cuBLASLt handle + workspace map | owned by the C library per device, overridable |
| locality-domain allocator (SM100) | allocator callback or pre-allocated pointer |

---

## 2. Route A (torch-free C ABI) vs route B (torch stable C++ ABI)

### 2.1 Route A: C ABI + `torch.library` in Python

Shape: `libmono_ep_cuda.so` exports `extern "C"` entry points taking device pointers, sizes/strides,
dtype codes, scalars and a `cudaStream_t`; it links `libcudart.so.<major>` and `libdw` only and
contains the DeepJIT runtime. Python binds the symbols with `ctypes` (stdlib, allowed in a Hub kernel)
and wraps each in `torch.library.custom_op` with a `register_fake`, exactly like the ~30 existing
`mono_ep::*` ops. Device code is unchanged (DeepJIT, nvcc at runtime, headers shipped).

- What breaks: nothing in the kernels. `python_api.cpp` is replaced wholesale by a C surface
  (one function per `m.def`, 75 today, of which mono-ep uses ~25). `torch::from_blob` views become
  offset tables. Factories move to Python. pybind classes become structs. mlopart/locality-domain
  allocation becomes a callback. `utils/layout.hpp` and `runtime_utils.hpp` get a tensor-view type.
- Current stream / capture: explicit `cudaStream_t` argument on every call (the custom op reads
  `torch.cuda.current_stream().cuda_stream`); capture checks replaced by eager allocation in
  `functional` + `cudaStreamIsCapturing` assertion.
- Python/torch coupling: zero at the ABI; Python interop is `ctypes` (~1 µs per call, see §3) or,
  if that matters later, a tiny abi3 CPython shim (`PyModuleDef` + `PyCapsule`) that does not touch
  torch.
- Fit with the planned layering: this *is* the ops layer. `ops.py` = thin custom ops over the C
  symbols (stateless, explicit workspaces); `_fake_impls.py` = Python fakes; `functional.py` owns
  workspaces/symm buffers and the per-stream buffers that used to hide behind capture checks;
  `api.py` unchanged.
- Portability bonus: the same `.so` serves JAX/XLA FFI, vLLM, or a C++ trainer, and the Hub
  `torch-noarch`-style loading is possible in principle (§5.4).

### 2.2 Route B: torch stable C++ ABI (`torch/csrc/stable`, `STABLE_TORCH_LIBRARY`, abi3)

Shape: keep C++ bindings but written against `torch::stable::Tensor`, `torch::stable::empty`,
`getCurrentStream(idx).nativeHandle()` and `STABLE_TORCH_LIBRARY_IMPL`, compiled with
`-DTORCH_TARGET_VERSION=0x020d...` (2.13) against the newest torch headers, `py_limited_api=True`, no
pybind11. This is what FA3, torchao's sm90a modules, xformers 0.0.34+ and vLLM 0.33 did.

- What breaks: every pybind construct (`pybind11::class_`, `py::tuple`, `std::optional<Tensor>`
  arguments through pybind casters, `module_::import` in mlopart) and every unstable header
  (`c10/cuda/*`, `ATen/cuda/CUDAContext.h`, `torch/python.h` which every launcher includes). The
  schema becomes a string per op (75 schema strings, hand-written boxed wrappers or `TORCH_BOX`).
  Fakes still live in Python (`register_fake` has no stable C++ equivalent; vLLM did the same).
- Current stream: available (`aoti_torch_get_current_cuda_stream` pre-2.10;
  `getCurrentStream().nativeHandle()` 2.13+). Capture status: **not available** in either stable
  shim; falls back to `cudaStreamIsCapturing` or eager allocation, same as route A.
- cuBLASLt handle: not available (only the cuBLAS handle, 2.10+); own it, same as route A.
- pybind objects: `torch::stable::Tensor` arguments and scalars only; the epilogue classes still
  have to become argument structs or `Tensor?` lists, because there is no stable custom-class
  facility. So route B does not save the class rewrite.
- `from_blob` with deleter: available since 2.11 (`torch::stable::from_blob` + `deleter(void*, ctx)`).
- Coupling: the artifact still imports libtorch symbols (`torch_*`/`aoti_torch_*` shims), so it is one
  wheel per (CUDA major) that works for torch >= target, but it is unusable without torch, and the
  DeepJIT runtime stays inside a torch extension.
- Fit with layering: the C++ file *is* the ops layer, fakes in Python. Workspaces still have to move
  to arguments because allocation inside a stable op is fine but the capture-guarded lazy buffers are
  not expressible. So the `functional` work is identical.

### 2.3 Estimated lines touched

Counted against the current `csrc` (11.7k lines, 48 headers), mono-ep subset first.

| area | route A | route B |
|---|---|---|
| tensor view type + `runtime_utils` TMA helpers + `utils/layout.hpp` | ~450 new / ~300 replaced | ~250 replaced (API-compatible `stable::Tensor` has `sizes()/strides()/data_ptr()`) |
| mono-ep launchers (`sm90_bf16_mono_ep{,_backward}`, `sm90_fp8_*`, `sm100_*`, grouped/chunk/k-grouped GEMMs, epilogue impls) | ~1,100 (signature rewrite `torch::Tensor` -> view, explicit stream) | ~900 (type swap + stream) |
| `apis/*` for the mono-ep subset (~25 of 75 entry points) | ~700 new C functions + structs; pybind removed | ~800 (schemas + boxed wrappers + `stable::empty` for outputs) |
| epilogue classes | ~150 (structs) | ~200 (structs + `Tensor?` plumbing) |
| runtime (`runtime.hpp` cuBLASLt/workspace, `locality_domain`, `mlopart`) | ~250 | ~250 |
| DeepJIT patch (`kernel.hpp`, `gil.hpp`) | 2 sites, ~30 lines | 2 sites, ~30 lines |
| Python: `capi`-style binding + custom ops + fakes | ~900 (ctypes structs + ~25 ops) | ~400 (fakes + `_ops` glue) |
| `owned_backward` | ~20 (already torch-free; drop pybind shim) | ~40 |
| build (CMake/scikit-build, abi3 wheel, Hub `build.toml`) | ~300 | ~300 |
| **total (mono-ep subset)** | **~3.9k** | **~3.2k** |
| rest of DeepGEMM (attention, einsum, mega_*, hyperconnection, cuBLASLt GEMM) if ever needed | ~2.5k | ~2.5k |

Route B is ~20% less typing for the first cut; route A pays that back the first time the artifact
is loaded outside torch or the stable C++ wrappers move (they are "subject to the same FC/BC policy
as the rest of PyTorch", only the C shim has a two-year guarantee).

### 2.4 Recommendation: route A

1. The expensive parts are identical in both routes: the pybind classes, factories, capture-guarded
   buffers, mlopart and cuBLASLt all have to move regardless. Route B's only real saving is the
   tensor-view type, which the prototype shows is ~60 lines.
2. mono-ep already has the Python op layer that route A needs (~30 `custom_op`s with fakes,
   `owned_backward` already crosses a torch-free `uintptr_t` boundary). Route A finishes a pattern the
   package already uses; route B introduces a second binding style.
3. The planned MoK split (stateless ops with explicit workspaces, Python fakes, functional layer
   owning state) maps one-to-one onto a C ABI. Route A makes the layering refactor and the ABI
   refactor the same change, applied twice instead of once.
4. Independence from torch's ABI story: no `TORCH_TARGET_VERSION` floor, no limited-API caveats
   (free-threaded Python is excluded by torch's abi3 wheels), no exposure to `torch/csrc/stable`
   churn. The wheel depends on `libcudart.so.<major>` only.
5. Hub fit: a Hub kernel must not have dynamic dependencies beyond torch and torch's CUDA libs; a
   C-ABI `.so` linking only `cudart` satisfies that trivially, and `kernel-builder` ships
   `.abi3.so` + arbitrary data files (`pyext`) which is all it needs (§5).

Cost accepted: host overhead. The prototype measures +20 to +35 µs per call for custom op +
ctypes over a bare pybind call (§3.1), fully hidden behind the kernel at training shapes but visible
at decode-sized ones. The custom-op dispatch is paid by both routes; the ctypes share shrinks with the
struct caching and pointer-table changes listed in §3.1, and the remainder can be removed with a
100-line abi3 CPython module that forwards to the same C symbols, still torch-free.

Route B remains the right answer for a *different* artifact: if DeepGEMM upstream (which already
ships a `TORCH_LIBRARY` binding on the Hub, `kernels-community/deep-gemm`) is ever consumed directly
instead of vendored.

---

## 3. Prototype: SM90 BF16 persistent forward over a C ABI

Files on the prototype branch (all additive; the pybind path is untouched and remains the
reference):

- `src/mono_ep/include/capi/mono_ep_capi.h` — the C contract: `mono_ep_tensor` (ptr, ndim, dtype,
  sizes, strides), `mono_ep_sm90_bf16_forward_args` (struct_size-versioned), `mono_ep_bf16_layout`
  (byte offsets of the symm-buffer views), `mono_ep_capi_{init,set_num_sms,get_num_sms}`,
  `mono_ep_bf16_symm_buffer_layout`, `mono_ep_sm90_bf16_forward`. Errors come back as a status code
  plus a message copied into a caller buffer; no exception crosses the boundary.
- `src/mono_ep/include/capi/sm90_bf16_mono_ep_capi.cpp` — torch-free launcher (no
  `torch::Tensor`, no ATen stream, explicit `LaunchOptions.stream`). It re-implements the two TMA
  helpers on a 40-byte `View`, mirrors `get_symm_buffer_size_for_mono_ep` for `bf16xbf16` to derive
  the ring views from byte offsets instead of `torch::from_blob`, and formats the **byte-identical**
  JIT source string of `csrc/jit_kernels/impls/sm90_bf16_mono_ep.hpp`, so both paths hit the same
  DeepJIT cache entry and the same cubin.
- `src/mono_ep/capi.py` — `ctypes` structs mirroring the header, `torch.library.custom_op
  ("mono_ep::sm90_bf16_forward_capi", mutates_args=...)` with a `register_fake`, and
  `capi.bf16_mono_ep(...)`, a drop-in for `deep_gemm.bf16_mono_ep`. For the prototype the `.so` is
  built on first use with `torch.utils.cpp_extension.load(..., is_python_module=False)` so the
  vendored `setup.py` and the Modal image stay untouched; the production build is a CMake target
  (§6).
- `examples/probe_capi_parity.py`, `examples/modal_capi_parity.py` — single-rank `torchrun` probe
  on one `H100!`.

Prototype caveats, by design: the TU still *includes* `runtime/runtime.hpp` and `utils/math.hpp`
through the heuristics headers (so it links against libtorch for the cuBLASLt workspace map and
`kPackedFP4`), and DeepJIT's `kernel.hpp` still references `getCurrentCUDAStream` in the untaken
fallback branch. The code paths exercised are torch-free; the link line is not yet. Splitting
`runtime.hpp` (SM count + device) from the cuBLASLt part and the DeepJIT macro (§1.3) removes the
remaining references; that is step 2 of §6.4.

Two load-time lessons from the Modal iterations, both relevant to the production build:

- `DG_HOST_ASSERT` (`csrc/utils/exception.hpp`) must be defined before any `deep_gemm/*.cuh` header
  is included, because `common/exception.cuh` binds `DG_UNIFIED_ASSERT` to it at the point of use.
  The pybind TU gets this for free through `apis/config.hpp`; a standalone TU must order includes.
- The pybind `_C` only gets `cublasLtCreate`/`cublasGetStatusString` resolved because it references
  `at::cuda::getCurrentCUDABlasLtHandle`, which keeps `libtorch_cuda` (and transitively cuBLAS) in
  its lookup scope. A torch-free TU references nothing in `libtorch_cuda`, `--as-needed` drops it,
  and the `.so` fails to load. The prototype links the cuBLAS libraries torch already mapped by exact
  filename (`-l:libcublasLt.so.13`); the production library must either link `cublasLt` explicitly
  (and declare it: both PyPI and the Hub allow torch's CUDA libraries) or, better for mono-ep, not
  pull `runtime.hpp`'s cuBLASLt member into the build at all.

### 3.1 Results

Modal run `ap-oZ86ZFOJAR8Bt5OAkCFbg3` (app `axolotl-mono-ep-capi-parity`, `gpu="H100!"`, one GPU,
NVIDIA H100 80GB HBM3, torch 2.14.1+cu130, CUDA image of `examples/modal_validate.py`,
`num_sms = 132 - 16 = 116`). Earlier runs of the same job (`ap-G6Y2…`, `ap-ATi6…`, `ap-Emb8…`,
`ap-wjbz…`, `ap-VL7l…`) failed before the probe body: one image-build error and four build/load
iterations (include order, then the cuBLAS link scope described above); none produced numbers.
The C ABI TU compiles in 22.8 s with `g++-13`.

**Parity.** Two shapes, three seeds each (one with 25% of routes masked to -1), `fast_math`
off and on for the large shape; `saved_z` requested on both paths:

| tokens / hidden / inter / experts / topk | cases | `y` bitwise | `saved_z` bitwise | masked `saved_z` stays NaN | rel. L2 vs fp32 reference |
|---|---|---|---|---|---|
| 2048 / 2048 / 768 / 16 / 8 | 6 | 6/6 | 6/6 | 6/6 | 1.66e-3 |
| 256 / 512 / 512 / 8 / 4 | 3 | 3/3 | 3/3 | 3/3 | 1.67e-3 |

The pybind path is itself bitwise reproducible across repeats (`y_bitwise_pybind_repeat` true in
all 9 cases), so "bitwise equal" is a meaningful statement. The byte offsets and ring size returned
by `mono_ep_bf16_symm_buffer_layout` match the pybind `SymmBuffer` views exactly (`num_bytes`,
`num_ring_tokens`, `x`, `topk_idx`, `topk_weights`, `l1_acts`, `l2_acts`) for both shapes.

**Host overhead.** 200 back-to-back launches per path after 10 warm-ups, same inputs, no `saved_z`.
`host wall` is `perf_counter` around the Python call without synchronisation (what the training loop
pays on the CPU); `event/call` is CUDA-event time divided by the number of launches (what the GPU
sees); `isolated` is one launch bracketed by events after a sync (includes launch latency).

| path | host wall median µs | host wall min µs | event/call µs | isolated µs |
|---|---|---|---|---|
| large shape | | | | |
| `deep_gemm.bf16_mono_ep` (pybind, no custom op) | 17.5 | 16.7 | 837.0 | 863.7 |
| `capi.bf16_mono_ep` (custom op + ctypes + checks) | 51.3 | 40.8 | 835.5 | 891.7 |
| `capi.sm90_bf16_forward_capi` (custom op + ctypes) | 38.7 | 36.1 | 834.4 | 888.9 |
| same, allocating the `sm_locality_domains` zeros per call | 51.5 | 44.0 | 837.5 | 903.6 |
| pybind repeated at the end of the run | 17.8 | 17.1 | 837.5 | 865.8 |
| small shape | | | | |
| pybind | 16.7 | 15.8 | 51.5 | 69.9 |
| `capi.bf16_mono_ep` | 45.4 | 39.5 | 49.4 | 98.5 |
| `capi.sm90_bf16_forward_capi` | 40.1 | 36.1 | 49.2 | 97.6 |
| per-call zeros | 53.1 | 46.1 | 55.6 | 105.7 |

Reading:

- GPU time per launch is identical within noise (835 vs 837 µs; 49 vs 51 µs), as expected from a
  shared cubin. At the large shape host overhead is entirely hidden behind the kernel on both paths.
- The C ABI path costs **+20 to +35 µs of host time per call** over the bare pybind call. That is
  more than the ctypes marshalling alone: the measured path also pays the `torch.library.custom_op`
  dispatch (14 arguments, including a `list[int]` of symm-buffer pointers and two `Optional[Tensor]`),
  which the pybind baseline does not pay. The production comparison is custom-op-over-pybind vs
  custom-op-over-ctypes, since every mono-ep training op is already a `custom_op`; that difference
  was not isolated in this job and is bounded above by the numbers here.
- Where the remaining cost goes, and the obvious fixes, in order: (1) build the `_ForwardArgs`
  struct once per `(buffer, weights)` pair in `functional` and only patch `y`/`stream`/`num_tokens`
  per call (the ctypes struct fill is a few µs of Python); (2) pass the symm-buffer pointer table as
  a persistent `int64` tensor instead of a `list[int]` (the list goes through the custom-op schema
  on every call); (3) hoist `get_num_sms()` out of the per-call path (the `capi.bf16_mono_ep` vs
  `sm90_bf16_forward_capi` gap, ~5–12 µs, is that ctypes round trip plus the Python checks);
  (4) if the residual still matters, an abi3 CPython module over the same C symbols removes the
  ctypes layer. A per-call `torch.zeros` for `sm_locality_domains` costs another ~8–13 µs, which is
  why the stable design keeps it as a cached workspace argument.
- The small shape shows the same ~25 µs host delta sitting on top of a 50 µs kernel, so for
  decode-sized or tiny-expert calls the overhead work above is needed before switching the default.


---

## 4. CUDA packaging

### 4.1 Can DeepJIT use NVRTC? No (verified)

`third-party/deep_jit/include/deep_jit/backend/cuda/backend.hpp` compiles exclusively by spawning
`nvcc ... --cubin` (command assembled in `compile()`), asserts `NVCC version must be at least 12.9`
from `nvcc --version` **when the `Runtime` is constructed**, and `find_cuda_toolkit()` asserts an
`nvcc` executable (`CUDA_HOME`/`CUDA_PATH`, then `which nvcc`, then `/usr/local/cuda`). There is no
NVRTC code path anywhere in the pinned tree (the upstream DeepGEMM `compiler.hpp` has an opt-in
`DG_JIT_USE_NVRTC`; the vendored DeepJIT rewrite dropped it). Consequences:

- A process without nvcc cannot even construct the runtime, so **a precompiled cubin cache alone does
  not remove the nvcc requirement** without a DeepJIT change.
- Cache key = `extra_signature` (`cutlass-<ver>`) + nvcc version hash + compiler options + post-hook +
  the parsed source including tracked includes; cache roots `DG_JIT_CACHE_DIR` /
  `DJ_JIT_CACHE_DIR` are colon-separated, misses are written to the first root, later roots may be
  read-only. A shipped cache is therefore valid only for the exact nvcc version string it was built
  with.
- The kernels use TMA, WGMMA/UMMA, clusters, `sm_90a`/`sm_100a` features via CUTLASS/CuTe; NVRTC can
  compile that in principle, but adopting it means re-implementing include tracking, `--cubin` link
  flags, PDL/cluster attributes and `libdevice` handling in DeepJIT. Not in scope; recorded as a
  possible later DeepJIT contribution.

Recommended DeepJIT changes (small, upstreamable): (a) make the nvcc check lazy (only when a compile
is actually needed), (b) allow a `read-only cache root` lookup before probing nvcc, (c) the
`DJ_NO_TORCH` macro from §1.3. With (a)+(b) a wheel can ship cubins for the default instantiations and
run on a toolkit-less host until it meets a new shape.

### 4.2 Runtime requirements of a wheel

Three layers, all needed at runtime for a cache miss:

1. **Host library** (`libmono_ep_cuda.so`): needs `libcudart.so.12` or `.13` (from torch's
   `nvidia-cuda-runtime[-cu12]` dependency) and `libcuda.so.1` via dlopen. DeepJIT `dlopen`s
   `libdw.so.1` lazily for symbolised backtraces (`utils/exception.hpp`), so it is an optional
   runtime nicety, not a link dependency. cuBLASLt is needed only while `runtime.hpp` keeps the
   handle in the shared `Runtime` object (prototype); the production library drops it from the
   mono-ep build or receives a handle.
2. **Device headers** shipped in the package: `deep_gemm/include/{deep_gemm,cute,cutlass}` (as the
   vendored `setup.py` already does) plus the CUDA headers nvcc needs. nvcc finds its own `include/`
   relative to `CUDA_HOME`, so the toolkit must provide `cuda_runtime.h`, `cccl/` (CUB/libcu++,
   needed by CUTLASS) and `crt/`.
3. **nvcc >= 12.9** with `cicc`, `ptxas`, `fatbinary`, `nvlink`, `crt/`, `nvvm/`.

What PyPI can supply (verified by downloading the wheels on 2026-11-10):

| package | contents | usable as DeepJIT toolkit? |
|---|---|---|
| `nvidia-cuda-nvcc-cu12` 12.9.86 | `bin/ptxas`, `include/crt/*`, `nvvm/lib64/libnvvm.so`, `libdevice.10.bc`; **no `nvcc`, `cicc`, `cudafe++`, `fatbinary`** | no (it is the Triton/Numba ptxas bundle) |
| `nvidia-cuda-nvcc` 13.5.37 (CUDA 13 wheels are unsuffixed, `site-packages/nvidia/cu13/`) + deps `nvidia-nvvm`, `nvidia-cuda-crt`, `nvidia-cuda-runtime` = `cuda-toolkit[nvcc]` | `bin/{nvcc,ptxas,cudafe++,fatbinary,nvlink,bin2c}`, `nvvm/bin/cicc`, `include/crt`, `include/cuda_runtime*.h`, `lib/libcudart.so.13` | yes, with `CUDA_HOME=<site-packages>/nvidia/cu13` |
| `nvidia-cuda-cccl` 13.x | `include/cccl/{cub,thrust,cuda,nv}` (note the extra `cccl/` path element on cu13) | needed in addition; torch does not pull it |
| torch 2.13–2.15 `Requires-Dist` | `cuda-toolkit[cublas,cudart,...,nvrtc,nvtx]`, never `nvcc`, `crt`, `cccl` | — |

Therefore:

- **cu13 wheel**: declare `cuda-toolkit[nvcc]>=13.0` and `nvidia-cuda-cccl` as dependencies (or an
  extra `mono-ep[jit]`), locate the toolkit at import (`CUDA_HOME` > `nvidia/cu13` in site-packages >
  `which nvcc` > `/usr/local/cuda`) and pass it to `mono_ep_capi_init`. Fully pip-installable, no
  system toolkit needed.
- **cu12 wheel**: PyPI has no cu12 `nvcc` binary at all (`nvidia-cuda-nvcc-cu12` stops at 12.9.86 and
  ships only ptxas), so a cu12 wheel must document a **system CUDA >= 12.9** toolkit. A 12.8 toolkit
  fails DeepJIT's assertion and lacks `sm_100f`. Recommendation: cu13 primary (torch 2.15 is cu13x
  only on PyPI), cu12 secondary and only while the CI image matrix still contains cu12x.
- **Precompiled cubin cache**: ship `build/<variant>/jit_cache/<nvcc-version>/...` for the default
  instantiations and the benchmark-contract shapes (`benchmarks.md`) built with the same
  nvcc the wheel declares, and prepend it as a read-only `DG_JIT_CACHE_DIR` root. Only valid for
  that exact nvcc version string; misses still need nvcc; requires DeepJIT change (a)/(b) above to
  help on toolkit-less hosts. Worth doing for cold-start time, not as a substitute for the toolchain.

### 4.3 CUDA major/minor compatibility and the wheel matrix

- The host `.so` has `DT_NEEDED libcudart.so.12` or `.13` (sonames verified in the runtime wheels), so
  there is **one wheel per CUDA major** matching the torch flavour: cu126/cu129 torch -> `+cu12`
  wheel, cu130/cu132/cu134 torch -> `+cu13` wheel. Python ABI is `cp310-abi3`, torch version is not in
  the tag (nothing in the artifact depends on torch).
- Within a major, CUDA minor-version compatibility (NVIDIA docs) says a 13.0-built binary runs on
  any R580+ driver and any 13.x runtime; building the host library with the **lowest** toolkit of the
  major (13.0 for cu13, 12.9 for cu12 because of DeepJIT) keeps every `cudart` symbol resolvable in
  older torch environments. Cubins generated at runtime by whatever nvcc is installed are
  `sm_90a`/`sm_100a` SASS and need a driver at least as new as that toolkit's major (R580+ for any
  13.x nvcc). Shipping cubins built by 13.x nvcc therefore excludes R5xx drivers; a cu12 wheel keeps
  the 525+ floor.
- Build flags: `-std=c++20`, GCC 13 (std::format), `-D_GLIBCXX_USE_CXX11_ABI=1`, manylinux_2_28
  (GLIBC 2.28; the Hub also requires GLIBCXX <= 3.4.24 which rules out some C++20 library symbols; a
  `kernel-builder check-abi` pass decides whether `std::format` survives or the hot spots switch to
  `fmt`/`snprintf`).

---

## 5. Hugging Face Hub kernel

Sources: `huggingface/kernels` at 32b3dc8c (2026-11-04; `kernel-builder` moved into this repo and
the old repo is archived), docs at huggingface.co/docs/kernels, `kernels-community/deep-gemm`
(`build.toml`, `torch-ext/deep_gemm/__init__.py`, Hub tree), `kernels-community/activation` v1 branch.

### 5.1 What a mono-ep kernel repo needs

- `build.toml` edition 6 with `[general]` (`name`, integer `version` -> Hub branch `v<version>`,
  `license`, `backends = ["cuda"]`, `[general.cuda] minver`), one framework section, and
  `[kernel.<name>]` sections. `kernel-builder create-pyproject` (the renamed `build2cmake`) generates
  CMake/`setup.py`; Nix builds the compliance variant set.
- Python package under `torch-ext/mono_ep/` with `__init__.py`, relative imports only, stdlib + torch
  + curated `python-depends` only (`triton` is assumed through torch; `ctypes` is stdlib and allowed).
- Torch ops must be registered under the per-build namespace `torch.ops.<id>` where
  `id = _<name>_cuda_<shortsha>`; Python registrations must use `from ._ops import
  add_op_namespace_prefix` (a build hook rejects literal names in `custom_op`/`register_fake`). For
  route A that means the `custom_op` names in `ops.py` take the namespace from `_ops.py` when built as
  a Hub kernel and `mono_ep::` when installed from PyPI (one `_namespace()` helper).
- `metadata.json` (generated), digest over every shipped file. **The JIT cache must not be written
  inside the variant directory** (digest validation would fail); keep `~/.deep_gemm`/`DG_JIT_CACHE_DIR`
  as today and treat a shipped cubin cache as read-only input.
- Import must succeed on a GPU-less, nvcc-less sandbox (`get-kernel-check`), so `mono_ep_capi_init`
  is lazy and wrapped in `try/except` exactly as `kernels-community/deep-gemm` does
  (`_InitializedOps` proxy).
- Variant layout: `build/<variant>/__init__.py` plus the compat dir `build/<variant>/mono_ep/`;
  `.so` named `_mono_ep_cuda_<sha>.abi3.so`.

### 5.2 Shipping DeepJIT and CUTLASS headers inside the artifact

Verified mechanism (deep-gemm precedent): runtime headers live under `torch-ext/mono_ep/include/...`
and are shipped by adding their extensions to `[torch] pyext = ["py", "cuh", "hpp", "h", "inl"]`;
`add_kernels_install_target` globs `torch-ext/<pkg>/*.<ext>` recursively and preserves
sub-directories. deep-gemm ships the whole CUTLASS tree per variant this way
(`include/third-party/cutlass/{include,...}`); `__init__.py` derives `lib_root` from `__file__` and
sets `DG_CUTLASS_INCLUDE`. For mono-ep: `torch-ext/mono_ep/include/{deep_gemm,cute,cutlass}` (the
same set `vendor/deepgemm/setup.py` already copies into the wheel) and DeepJIT's own headers are not
needed at runtime (DeepJIT is compiled into the host `.so`; only its generated `-I` flags matter).

Compile-time includes are different: the Nix sandbox copies **only files listed in `src`** (plus
`torch-ext/` files matching `pyext`), so every header the host TU includes must be enumerated in
`[kernel.*].src` (deep-gemm lists ~100). CUTLASS for the host build comes from `depends =
["cutlass_4_4"]` (closed enum: `cutlass_2_10, 3_5, 3_6, 3_8, 3_9, 4_0, 4_4, 4_5`; our pin f3fde583 is
4.2.1, so the host build uses the nearest enum version while the runtime headers stay our pinned
copy). DeepJIT headers are vendored into `csrc/third-party/deep_jit/include` and listed in `src`.

### 5.3 Stable ABI / abi3 universal builds: supported today (experimental)

- `[torch.stable-abi]` (table `backend -> torch version`) builds one `torch-stable-abi<MAJ><MIN>-
  cu<ver>-<arch>-<os>` variant per CUDA minor instead of one per torch version
  (`docs/source/builder/writing-kernels.md`; `nix-builder/lib/build.nix`: "For backends that use the
  Torch stable ABI, we only need a build set for the latest Torch"). `kernels-community/activation`
  v1 already ships `torch-stable-abi210-cu{126,128,130,132,134}-{x86_64,aarch64}-linux`. The loader
  accepts it when the environment torch >= that version (`kernels/src/kernels/variants.py`,
  `TorchStableAbi`), and `kernel-builder check-abi --torch-stable-abi` verifies the `.so` only
  imports versioned shim symbols. All arch builds are `.abi3.so` (`USE_SABI 3`).
- There is **no "torch-free arch" variant**: a compiled kernel is either per-torch, torch-stable-abi,
  or `tvm-ffi`; `torch-noarch` builds no `.so` at all. A route-A `.so` that imports zero torch symbols
  passes `check-abi` trivially, so the practical choice is to declare `[torch.stable-abi] cuda =
  "2.13"` (the lowest torch in our matrix) and ship the C-ABI `.so` under that variant name, with a
  tiny `STABLE_TORCH_LIBRARY`-free `PyInit_` stub (`REGISTER_EXTENSION`). The variant name then
  encodes a torch floor the artifact does not actually have; that is cosmetic. Alternative to
  evaluate once the loader lands it: `tvm-ffi` variants (`[tvm-ffi]` section exists today) are the
  framework-agnostic path flashinfer uses.
- Compliance variant set (`docs/source/builder/build-variants.md`): torch 2.10–2.15 ×
  cu126/128/130/132/134 × x86_64/aarch64. With stable-abi 2.13 the Nix matrix still builds per-torch
  variants for 2.10–2.12 (release note v0.18.0). DeepJIT's nvcc >= 12.9 floor means
  `[general.cuda] minver = "12.9"` and no cu126/cu128 variants; `cuda-minver` per kernel section
  skips those components.

### 5.4 Draft `build.toml`

```toml
[general]
name = "mono-ep"
version = 1
edition = 6
license = "Apache-2.0"
backends = ["cuda"]
upstream = "https://github.com/axolotl-ai-cloud/axolotl"
source = "https://github.com/axolotl-ai-cloud/axolotl/tree/main/packages/mono-ep"

[general.cuda]
minver = "12.9"   # DeepJIT asserts nvcc >= 12.9

[general.hub]
repo-id = "axolotl-ai-co/mono-ep"

[torch]
# Only the torch.library glue is torch-aware; the compiled library is torch-free, so one build per
# CUDA minor is enough. 2.13 is the lowest torch in the axolotl CI matrix.
stable-abi = { cuda = "2.13" }
pyext = ["py", "pyi", "cuh", "hpp", "h", "inl", "json"]
src = ["torch-ext/torch_binding.cpp"]          # PyInit_ stub + REGISTER_EXTENSION, nothing else

[kernel.mono_ep_host]
language = "cpp-cuda"
cuda-capabilities = ["9.0a", "10.0a"]          # metadata gate for check_arch; device code is JIT
cuda-minver = "12.9"
cxx-flags = ["-std=c++20", "-O3", "-Wno-psabi", "-Wno-deprecated-declarations", "-DDJ_NO_TORCH"]
depends = ["torch", "cutlass_4_4"]
include = ["csrc", "csrc/third-party/deep_jit/include", "torch-ext/mono_ep/include"]
src = [
  "csrc/capi/mono_ep_capi.h",
  "csrc/capi/mono_ep_capi.cpp",                 # all extern "C" entry points, one TU
  "csrc/capi/launchers/sm90_bf16_mono_ep.hpp",
  # ... every host header the TU includes (Nix copies only listed files)
  "csrc/third-party/deep_jit/include/deep_jit/runtime/runtime.hpp",
  # ... DeepJIT headers
]
```

Open question for kernel-builder: a `cpp-cuda` component whose `src` has no `.cu` is schema-legal
but unverified under Nix (the source set still names the variant correctly only when a
`[kernel.*]` section exists, and `cuda_kernel_component` compiles `.cpp` as CXX). deep-gemm sidesteps
it with a host-only `csrc/impl.cu`; mono-ep can do the same (rename the C ABI TU to `.cu`, or keep
`owned_backward_launch.cu` in the same component, which is a real device TU anyway).

### 5.5 Loader snippet (`torch-ext/mono_ep/__init__.py`, Hub build) and consumer

```python
import ctypes, os
from pathlib import Path
from ._ops import add_op_namespace_prefix  # generated; namespace is _mono_ep_cuda_<sha>

_HERE = Path(__file__).resolve().parent
_LIB = next(_HERE.parent.glob("_mono_ep_cuda_*.abi3.so"))
_INCLUDE = _HERE / "include"  # deep_gemm/, cute/, cutlass/ shipped via pyext


def _cuda_home():
    for candidate in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        _pip_toolkit("nvidia/cu13"),
        _which_nvcc_root(),
        "/usr/local/cuda",
    ):
        if candidate and (Path(candidate) / "bin" / "nvcc").exists():
            return str(candidate)
    return None


class _Lazy:
    """Bind the C ABI on first op call so import works on GPU-less, nvcc-less hosts."""

    lib = None

    @classmethod
    def get(cls):
        if cls.lib is None:
            lib = ctypes.CDLL(str(_LIB))  # torch-free: links cudart only
            cls.lib = _declare(lib)  # argtypes/restypes from capi.py
            _check(
                lib.mono_ep_capi_init,
                str(_INCLUDE.parent).encode(),
                (_cuda_home() or "").encode(),
            )
        return cls.lib
```

Consumer (axolotl side, `expert_parallel/mono_ep.py`):

```python
from kernels import get_kernel

mono_ep = get_kernel(
    "axolotl-ai-co/mono-ep", version=1
)  # version is an int -> branch v1
y = mono_ep.api.persistent_forward(...)  # same api.py as the PyPI wheel
```

`get_kernel` needs `trust_remote_code=True` (or an allowlist) until the org is a trusted publisher;
`kernels lock` + `get_locked_kernel` pins the Hub SHA in axolotl's `pyproject.toml`.

### 5.6 Variant naming summary

| artifact | name |
|---|---|
| Hub, stable-abi (one per CUDA minor) | `build/torch-stable-abi213-cu130-x86_64-linux/` (`cu132`, `cu134`, `cu129` likewise) |
| Hub, per-torch for 2.10–2.12 (if kept) | `build/torch212-cu130-x86_64-linux/` |
| Hub, aarch64 (GB200 hosts) | `...-aarch64-linux` |
| PyPI | `mono_ep-<ver>+cu13-cp310-abi3-manylinux_2_28_x86_64.whl` (local version for the CUDA major as flash-attn/deep_gemm do, or two distributions `mono-ep-cu13` / `mono-ep-cu12`; PyPI rejects local versions, so the two-name scheme or a self-hosted index is required for cu12) |

---

## 6. CI plan

### 6.1 Build jobs (new workflow `mono-ep-build.yml`, path-filtered like `tree-recurrence.yml`)

| step | what | effort |
|---|---|---|
| B1 | CMake project for `libmono_ep_cuda.so` + abi3 stub (`scikit-build-core`, `wheel.py-api = "cp310"`); links `cudart` only, `-DDJ_NO_TORCH` | 1 day |
| B2 | Matrix on `ubuntu-latest` × `quay.io/pypa/manylinux_2_28_x86_64` with CUDA 12.9 and 13.0 toolkits (install via `nvidia/cuda:*-devel` images or the `cuda-toolkit[nvcc]` wheel for 13.x); GCC 13 is in manylinux_2_28 (`gcc-toolset-13`) | 1 day |
| B3 | `auditwheel repair --exclude libcudart.so.1x --exclude libcuda.so.1` (nothing else to bundle: `libdw` is dlopened lazily), upload `mono-ep-dist-cu12`/`-cu13` artifacts | 0.5 day |
| B4 | Optional cubin pre-cache job on the `H100!`/B200 Modal lanes: run `tools/warm_jit_cache.py` over the default instantiations + contract shapes, tar the cache keyed by nvcc version, attach as `jit-cache-cu13-<nvcc>` artifact and ship it inside the wheel under `mono_ep/_jit_cache/` | 1 day, after DeepJIT (a)/(b) |

No GPU is needed for B1–B3 (host code only; nvcc is not invoked at build time), so the build runs
on hosted runners.

### 6.2 Test matrix: install the wheel instead of compiling

Today `cicd/install_mono_ep.sh` installs `g++-13` + `libdw-dev`, fetches CUTLASS/DeepJIT, and
compiles `vendor/deepgemm` with `DG_FORCE_BUILD=1 --no-build-isolation` inside every Modal lane that
needs it (`gpu_slow.sh` ×2, `cicd_blackwell.sh`; `cicd_cuda_kernels.sh` only does the `--no-deps`
install of `packages/mono-ep` and relies on the lane image). Replacement:

| step | what | effort |
|---|---|---|
| T1 | `install_mono_ep.sh` becomes: download the `mono-ep-dist-cu<major>` artifact of the current run (or `pip install mono-ep==<sha>` from the PR's artifact index), `uv pip install --no-deps <wheel>`; apt installs go away except `libdw1` if not bundled | 0.5 day |
| T2 | Modal images (`cicd/image_build`, matrix cu130/py3.12/torch2.13, cu130/py3.14/torch2.14.1 ±blackwell, cu132/py3.14/torch2.15) stop building deep_gemm; add `cuda-toolkit[nvcc]` + `nvidia-cuda-cccl` to the image so the JIT works without the system toolkit (or keep the system toolkit the image already has and document that as the CI choice) | 0.5 day |
| T3 | the same wheel runs on all three torch versions in the matrix; add one assertion test that the loaded `.so` imports no `torch_*`/`c10` symbols (`nm -D` check, mirrors `kernel-builder check-abi`) | 0.5 day |
| T4 | parity gate: the existing `tests/test_*` GPU suite plus the §3 bitwise probe generalised per op (`pytest -m gpu` on the kernels lane) | included in the port |

### 6.3 Publish steps

| step | what | effort |
|---|---|---|
| P1 | `mono-ep-publish.yml` on tag `mono-ep/v*`, mirroring `tree-recurrence-publish.yml`: write `VERSION` from the tag, reuse the B2/B3 build jobs, `pypa/gh-action-pypi-publish` with trusted publishing environment `pypi-mono-ep` (two distributions if cu12 is kept) | 0.5 day |
| P2 | Hub: `huggingface/kernel-builder-job` action (`kernel-builder build-and-upload --repo-id axolotl-ai-co/mono-ep`, token from `HF_TOKEN`, Nix on the HF runner) on the same tag; the Hub `version` integer bumps only on incompatible API changes, otherwise the `v1` branch is replaced in place | 1 day (first run: kernel-creation access for the org, flake pin, `src` enumeration) |
| P3 | axolotl `pyproject.toml`: `mono-ep` becomes a pinned optional dependency (`[mono-ep]` extra) instead of a path install; `kernels.lock` entry for the Hub path | 0.5 day |

### 6.4 Ordering relative to the layering refactor

1. **Layering refactor first** (Wing's MoK split: `ops.py` stateless custom ops with explicit
   workspaces, `_fake_impls.py`, `functional.py`, `api.py`; bitwise + timing-neutral gate). The ABI
   work needs every op to already take its workspaces and stream-dependent buffers as arguments; doing
   that in Python first is the cheapest place.
2. **DeepJIT patch + `runtime.hpp` split** (`DJ_NO_TORCH`; separate SM-count/device state from the
   cuBLASLt workspace map; lazy nvcc check). Small, independent, unblocks a torch-free link line.
3. **Port launchers op by op behind `capi/`**, in the order of the training path: SM90 BF16 forward
   (done), SM90 BF16 backward + k-grouped wgrad, SM100 BF16 forward/backward, grouped/chunk/ready
   GEMMs with the epilogue structs, FP8 variants, `transform_weights`/layout helpers. Each op swaps
   its `ops.py` body from `torch.ops.deep_gemm`/pybind to the C symbol with the §3 probe as the gate.
   Estimated 3.9k lines (§2.3), ~3 weeks of focused work with GPU validation on dev2/Modal.
4. **Build + CI** (§6.1–6.2) can start in parallel with step 3 once the first two ops are in, since
   the CMake target and wheel plumbing do not depend on op count.
5. **Publish** (§6.3) when the training path is fully on the C ABI; pybind `deep_gemm._C` is deleted
   from the vendored tree at that point (the vendored DeepGEMM kernels and headers stay).

---

## 7. Verified facts vs assumptions

Verified (read from source, docs, or measured):

- Inventory counts (§1) from grep over the vendored tree at 9ae994be97; call sites named by file:line.
- DeepJIT: nvcc-only compilation, >= 12.9 assertion at runtime construction, cache key composition,
  colon-separated cache roots, exactly two torch couplings (`kernel.hpp` stream fallback, `gil.hpp`).
- PyPI wheel contents of `nvidia-cuda-nvcc-cu12` 12.9.86 (ptxas only) and `nvidia-cuda-nvcc`
  13.5.37 + `nvidia-nvvm`/`nvidia-cuda-crt`/`nvidia-cuda-runtime` (full nvcc), `cuda-toolkit[nvcc]`
  extra, torch 2.13/2.14/2.15 `Requires-Dist` (no nvcc/cccl/crt).
- torch stable ABI surface: `torch/csrc/stable/*`, `TORCH_TARGET_VERSION` semantics, shim versions
  (`shim_function_versions.txt`), current-stream and cuBLAS-handle shims, **no** capture-status or
  cuBLASLt shim, `py_limited_api` behaviour of `cpp_extension`; FA3/torchao/vLLM/xformers practice.
- kernel-builder: moved into `huggingface/kernels`; edition-6 `build.toml` schema and `depends`
  enum; `pyext` header shipping and the Nix "only listed `src`" rule; `[torch.stable-abi]` variants
  and loader acceptance; `_ops.py` namespace requirement; integer `version` -> `v<N>` branch;
  `kernel-builder-job` action; `kernels-community/deep-gemm` as a shipped-headers + nvcc-JIT precedent.
- Prototype parity and overhead numbers (§3.1) from Modal `H100!` job `ap-oZ86ZFOJAR8Bt5OAkCFbg3`.

Assumptions (not verified in this session):

- The CUDA-kernels Modal lane obtains `deep_gemm` from its image rather than from
  `install_mono_ep.sh` (no Dockerfile reference found; only `gpu_slow.sh` and `cicd_blackwell.sh`
  call the script).
- `manylinux_2_28` + GCC 13 + `std::format` passes the Hub's GLIBCXX <= 3.4.24 rule; if not, the
  host code uses `fmt` or `snprintf` for the JIT source formatting.
- A `cpp-cuda` kernel component without `.cu` sources builds and is named correctly under Nix
  (unverified; mitigated by keeping a `.cu` in the component).
- Cubins produced by a 13.x nvcc need an R580+ driver (NVIDIA states the rule at application level,
  not per cubin).
- The split of the measured +20–35 µs between `custom_op` dispatch and ctypes was not isolated
  (no custom-op-over-pybind baseline in the job); the abi3 CPython shim fallback was not prototyped.
