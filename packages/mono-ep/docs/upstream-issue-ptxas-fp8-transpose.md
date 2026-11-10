# Draft upstream issue (not filed): wrong bytes from an 8-bit layout conversion of a fresh e4m3 cast on sm_90 with the bundled ptxas 12.9

Target: `triton-lang/triton` (release 3.8.x bundles ptxas 12.9.86 for pre-Blackwell targets). A
copy can go to NVIDIA as a ptxas bug report; the PTX below reproduces it with `ptxas` alone once
compared against `-O0` output or ptxas 13.x.

## Title

`tl.trans` of a freshly cast `float8_e4m3fn` tile stores wrong bytes on H100 (Triton 3.8.0,
bundled ptxas 12.9.86); correct with ptxas 13.x or `--opt-level 0`

## Summary

A 128x128 (or 64x128) FP32 tile is cast to `float8_e4m3fn` and the transposed tile is stored.
The transpose lowers to `ttg.convert_layout` on the FP8 tensor (blocked `[1,16]` -> blocked
`[16,1]`, through shared memory as `st.shared.v4.b8` of bytes taken from
`cvt.rn.satfinite.e4m3x2.f32` results). On H100 with the ptxas that Triton 3.8.0 bundles for
`sm_90` (`backends/nvidia/bin/ptxas`, 12.9.86) 25% or 50% of the stored bytes are wrong; the
untransposed store of the same tile and the per-tile scale are right.

- Same TTGIR, same PTX up to the position of 32 `shr.s16 ..., 8` instructions: one schedule is
  right, the other wrong, so the failure is below Triton's IR (SASS differs by ~1800 lines; the
  wrong variant fuses `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` with `PRMT` byte selects).
- Wrong output rows are exactly element lanes `1 mod 4` (25% cases) or `0,1 mod 4` (50% cases)
  of each thread's packed 4-byte word; the bytes are not a permutation of the tile.
- `int8` tiles transposed the same way are always right, so the e4m3 cast feeding the byte
  shuffle is part of the trigger.
- Right with: `TRITON_PTXAS_PATH=<triton>/backends/nvidia/bin/ptxas-blackwell` (13.3.33,
  targeting sm_90), the CUDA 13.0.88 ptxas, `DISABLE_PTXAS_OPT=1`. Wrong with
  `PTXAS_OPTIONS="--opt-level 1"` and `2`.
- Not reproducible on an RTX PRO 6000 (sm_120), which Triton compiles with `ptxas-blackwell`.
- Transposing the FP32 tile in the source does not help reliably: `RemoveLayoutConversions`
  moves the conversion back onto the FP8 tensor, and the result depends on the schedule (it is
  right for one of our store orders and wrong for the other).

## Environment

- Triton 3.8.0 as bundled with torch 2.14.1+cu130 (Python 3.14), CUDA 13.0 driver stack
- NVIDIA H100 80GB HBM3 (sm_90), Modal
- `triton/backends/nvidia/bin/ptxas --version`: `release 12.9, V12.9.86`
- `triton/backends/nvidia/bin/ptxas-blackwell --version`: `release 13.3, V13.3.33`

## Reproducer

`tools/triton_fp8_trans_repro.py` in this package (standalone, torch + triton only). Minimal
kernel:

```python
@triton.jit
def tile_transpose(W, Q, QT, N: tl.constexpr):
    cb = tl.program_id(0)
    r = tl.arange(0, 128).to(tl.int64)
    c = (cb * 128 + tl.arange(0, 128)).to(tl.int64)
    w = tl.load(W + r[:, None] * N + c[None, :])
    s = tl.max(tl.max(tl.abs(w), 1), 0) / 448.0  # any per-tile scale, or 1.0
    q = tl.minimum(tl.maximum(w / s, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + r[:, None] * N + c[None, :], q)  # correct
    tl.store(
        QT + c[:, None] * 128 + r[None, :], tl.trans(q)
    )  # 25% wrong bytes on sm_90


w = torch.randn(128, 1024, device="cuda")
q = torch.empty(128, 1024, device="cuda", dtype=torch.float8_e4m3fn)
qt = torch.empty(1024, 128, device="cuda", dtype=torch.float8_e4m3fn)
tile_transpose[(8,)](w, q, qt, 1024, num_warps=8)
assert torch.equal(qt.view(torch.uint8), q.view(torch.uint8).t())  # fails
```

Matrix on H100 (`tools/triton_fp8_trans_repro.py`, 60 variants): every 128x128 variant that
transposes after the cast is wrong (`tl.trans`, `tl.permute`, reshape+permute, a scattered store
that also lowers to a layout conversion); 64x128 is wrong when the untransposed store comes after
the transposed one and right when it comes first; 8 int8 variants right. The IR dumps
(`TRITON_KERNEL_DUMP`) of a wrong and a right variant with identical TTGIR are attached.

## Expected

`qt` bitwise equal to `q.T`, as with ptxas 13.x.

## Ask

Bundle a ptxas 13.x for pre-Blackwell targets in 3.8.x as `main` already does (13.5.37), or
document `TRITON_PTXAS_PATH` as the workaround for sm_90 FP8 kernels that transpose or scatter
8-bit tensors.
