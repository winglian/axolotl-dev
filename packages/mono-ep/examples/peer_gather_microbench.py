# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Two-GPU peer-gather microbench: can quack's gathered-A GEMM (SonicMoE's gate_up GEMM)
read its A rows from the peer GPU over NVLink at local speed, and how do SM-driven peer
copies compare with NCCL?

Rank ``r`` reads rank ``1 - r``. Three A placements for the same gathered grouped GEMM at the
EP8 fused-benchmark contract shape (131072 rows of H 2048 BF16, 16 local experts,
B [16, 2048, 1536], balanced ``cu_seqlens_m``, indices a random permutation):

- ``local``: A is a local tensor;
- ``peer_symm``: A is the peer's torch symmetric-memory buffer (its local VA mapping);
- ``window``: A is a contiguous peer-mapped VA window ``[2 * cap, H]`` built with the CUDA
  VMM API (``cuMemAddressReserve`` + ``cuMemMap`` of exported allocation handles at
  consecutive offsets; slice ``s`` is rank ``s``'s memory), with indices in the peer slice,
  in the local slice, or spanning both.

Alongside: raw peer-read bandwidth from SMs (a Triton row copy with 16 B vector loads and
``Tensor.copy_``), today's ``pack_peer_rows`` pull, and NCCL ``all_to_all_single`` /
``send``+``recv``, each one-directional and both-directions-at-once. One quack config per
GEMM is pinned (the production profile winner), nothing is swept. Run under
``torch.distributed.run --nproc-per-node=2`` with ``--worker``.
"""

import argparse
import ctypes
import json
import os
import socket
import statistics
import subprocess  # nosec B404
import sys
import tempfile
import time
import traceback

import torch
import torch.distributed as dist
import triton
import triton.language as tl

CAP, HIDDEN, INNER, LOCAL_EXPERTS = 131072, 2048, 768, 16
ROW_BYTES = HIDDEN * 2
NCCL_350MB_ROWS = 350 * 1024 * 1024 // ROW_BYTES  # 89600 rows
WARMUP, MEASURED = 3, 15
# NVLink per-direction nominal GB/s per GPU (H100 SXM NVLink 4: 900 GB/s bidirectional;
# B200 NVLink 5: 1800 GB/s bidirectional)
NVLINK_ONE_WAY_GBPS = {"H100": 450.0, "B200": 900.0}

GEMM_CONFIGS = {
    # production profile winners for gemm_gated_tuned at this key (sonicmoe_contract_profiles)
    "H100": {
        "cp_async": dict(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=False,
            cluster_m=2,
            cluster_n=1,
            swap_ab=False,
            max_swizzle_size=8,
            device_capacity=9,
            use_tma_gather=False,
        )
    },
    "B200": {
        "tma_gather4": dict(
            tile_m=256,
            tile_n=512,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            cluster_n=1,
            swap_ab=False,
            max_swizzle_size=8,
            device_capacity=10,
            use_tma_gather=True,
        ),
        # same tiles on the cp.async gather path; falls back to 256x256 if that fails to build
        "cp_async": dict(
            tile_m=256,
            tile_n=512,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            cluster_n=1,
            swap_ab=False,
            max_swizzle_size=8,
            device_capacity=10,
            use_tma_gather=False,
        ),
    },
}


# --- raw peer copy -------------------------------------------------------------------


@triton.jit
def _copy_rows_i64(SRC, DST, rows, WIDTH: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    """Grid-stride row copy over int64-viewed rows: 8 B elements, 4 per thread, so each
    thread issues 16 B vector loads; BLOCK_ROWS rows per tile keep many rows in flight."""
    cols = tl.arange(0, WIDTH)
    for start in range(
        tl.program_id(0) * BLOCK_ROWS, rows, tl.num_programs(0) * BLOCK_ROWS
    ):
        r = start + tl.arange(0, BLOCK_ROWS)
        offs = r[:, None].to(tl.int64) * WIDTH + cols[None, :]
        mask = (r < rows)[:, None]
        tl.store(DST + offs, tl.load(SRC + offs, mask=mask), mask=mask)


def triton_copy_rows(src, dst, programs):
    src64 = src.view(torch.int64)
    dst64 = dst.view(torch.int64)
    _copy_rows_i64[(programs,)](
        src64, dst64, src64.shape[0], src64.shape[1], 8, num_warps=8
    )


# --- CUDA VMM peer window ------------------------------------------------------------


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocFlags),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_DL_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DL_DELETER),
    ]


def _bf16_tensor_from_pointer(ptr, rows, cols, device_index):
    """A BF16 ``[rows, cols]`` tensor over a raw device address via a DLPack capsule.

    DLPack carries the device id explicitly; ``__cuda_array_interface__`` would make torch
    look the pointer's owner up with ``cudaPointerGetAttributes``, which for a peer-mapped
    window names the peer GPU and ends in a silent device copy. Returns the tensor and the
    ctypes objects that must outlive it."""
    shape = (ctypes.c_int64 * 2)(rows, cols)
    strides = (ctypes.c_int64 * 2)(cols, 1)
    managed = _DLManagedTensor()
    managed.dl_tensor.data = ptr
    managed.dl_tensor.device = _DLDevice(2, device_index)  # kDLCUDA
    managed.dl_tensor.ndim = 2
    managed.dl_tensor.dtype = _DLDataType(4, 16, 1)  # kDLBfloat
    managed.dl_tensor.shape = shape
    managed.dl_tensor.strides = strides
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    deleter = _DL_DELETER(lambda _p: None)
    managed.deleter = deleter
    new_capsule = ctypes.pythonapi.PyCapsule_New
    new_capsule.restype = ctypes.py_object
    new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = new_capsule(ctypes.addressof(managed), b"dltensor", None)
    tensor = torch.from_dlpack(capsule)
    return tensor, (managed, shape, strides, deleter)


class PeerWindow:
    """One VA reservation of ``world * size`` bytes; slice ``s`` maps rank ``s``'s allocation.

    Allocation handles are exported as POSIX file descriptors and exchanged over an AF_UNIX
    socket (``socket.send_fds``), path shared through the process group.
    """

    def __init__(self, group, device_index, rows, cols, dtype=torch.bfloat16):
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self._declare()
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.device_index = device_index
        self.rows, self.cols = rows, cols
        self.size = rows * cols * dtype.itemsize
        self.handles = []
        self.base = ctypes.c_ulonglong(0)
        self.steps = []
        prop = _CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_index
        granularity = ctypes.c_size_t(0)
        self._check(
            "cuMemGetAllocationGranularity",
            ctypes.byref(granularity),
            ctypes.byref(prop),
            CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        self.granularity = granularity.value
        if self.size % self.granularity:
            raise ValueError(f"{self.size} B is not a multiple of {self.granularity} B")
        own = ctypes.c_ulonglong(0)
        self._check("cuMemCreate", ctypes.byref(own), self.size, ctypes.byref(prop), 0)
        fd = ctypes.c_int(-1)
        self._check(
            "cuMemExportToShareableHandle",
            ctypes.byref(fd),
            own,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0,
        )
        self.steps.append("export_fd")
        peer_fds = self._exchange_fds(group, fd.value)
        os.close(fd.value)
        self.steps.append("exchange_fd")
        for s in range(self.world):
            if s == self.rank:
                self.handles.append(own)
                continue
            imported = ctypes.c_ulonglong(0)
            self._check(
                "cuMemImportFromShareableHandle",
                ctypes.byref(imported),
                ctypes.c_void_p(peer_fds[s]),
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            )
            os.close(peer_fds[s])
            self.handles.append(imported)
        self.steps.append("import_fd")
        self._check(
            "cuMemAddressReserve",
            ctypes.byref(self.base),
            self.size * self.world,
            0,
            0,
            0,
        )
        self.steps.append("reserve")
        for s, handle in enumerate(self.handles):
            self._check(
                "cuMemMap", self.base.value + s * self.size, self.size, 0, handle, 0
            )
        self.steps.append("map")
        access = _CUmemAccessDesc()
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device_index
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._check(
            "cuMemSetAccess", self.base, self.size * self.world, ctypes.byref(access), 1
        )
        self.steps.append("set_access")
        self.window, self._keepalive = _bf16_tensor_from_pointer(
            self.base.value, rows * self.world, cols, device_index
        )
        assert self.window.data_ptr() == self.base.value
        assert self.window.device.index == device_index
        self.slices = [
            self.window[s * rows : (s + 1) * rows] for s in range(self.world)
        ]
        dist.barrier(group)

    def _declare(self):
        c = ctypes
        signatures = {
            "cuGetErrorString": [c.c_int, c.POINTER(c.c_char_p)],
            "cuMemGetAllocationGranularity": [
                c.POINTER(c.c_size_t),
                c.POINTER(_CUmemAllocationProp),
                c.c_int,
            ],
            "cuMemCreate": [
                c.POINTER(c.c_ulonglong),
                c.c_size_t,
                c.POINTER(_CUmemAllocationProp),
                c.c_ulonglong,
            ],
            "cuMemExportToShareableHandle": [
                c.c_void_p,
                c.c_ulonglong,
                c.c_int,
                c.c_ulonglong,
            ],
            "cuMemImportFromShareableHandle": [
                c.POINTER(c.c_ulonglong),
                c.c_void_p,
                c.c_int,
            ],
            "cuMemAddressReserve": [
                c.POINTER(c.c_ulonglong),
                c.c_size_t,
                c.c_size_t,
                c.c_ulonglong,
                c.c_ulonglong,
            ],
            "cuMemMap": [
                c.c_ulonglong,
                c.c_size_t,
                c.c_size_t,
                c.c_ulonglong,
                c.c_ulonglong,
            ],
            "cuMemSetAccess": [
                c.c_ulonglong,
                c.c_size_t,
                c.POINTER(_CUmemAccessDesc),
                c.c_size_t,
            ],
            "cuMemUnmap": [c.c_ulonglong, c.c_size_t],
            "cuMemRelease": [c.c_ulonglong],
            "cuMemAddressFree": [c.c_ulonglong, c.c_size_t],
        }
        for name, argtypes in signatures.items():
            fn = getattr(self.cuda, name)
            fn.argtypes = argtypes
            fn.restype = c.c_int

    def _check(self, name, *args):
        result = getattr(self.cuda, name)(*args)
        if result != 0:
            message = ctypes.c_char_p()
            self.cuda.cuGetErrorString(result, ctypes.byref(message))
            raise RuntimeError(
                f"{name} failed: CUresult {result} ({(message.value or b'?').decode()})"
            )

    def _exchange_fds(self, group, fd):
        path = [
            os.path.join(
                tempfile.gettempdir(), f"peer_gather_window_{os.getpid()}.sock"
            )
            if self.rank == 0
            else None
        ]
        dist.broadcast_object_list(path, src=0, group=group)
        path = path[0]
        peers = {}
        if self.rank == 0:
            if os.path.exists(path):
                os.unlink(path)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(path)
            listener.listen(self.world)
            connections = []
            for _ in range(self.world - 1):
                conn, _addr = listener.accept()
                peer_rank = int.from_bytes(conn.recv(4), "little")
                connections.append((peer_rank, conn))
            for peer_rank, conn in connections:
                socket.send_fds(conn, [b"x"], [fd])
                _msg, fds, _flags, _addr = socket.recv_fds(conn, 1, 1)
                peers[peer_rank] = fds[0]
                conn.close()
            listener.close()
            os.unlink(path)
        else:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            for _ in range(600):
                try:
                    conn.connect(path)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    time.sleep(0.05)
            else:
                raise RuntimeError("rank 0 never opened the fd-exchange socket")
            conn.sendall(self.rank.to_bytes(4, "little"))
            _msg, fds, _flags, _addr = socket.recv_fds(conn, 1, 1)
            peers[0] = fds[0]
            socket.send_fds(conn, [b"x"], [fd])
            conn.close()
        # two ranks only need the pairwise exchange with rank 0
        if self.world != 2:
            raise NotImplementedError("the fd exchange is written for two ranks")
        return peers

    def destroy(self):
        if self.base.value:
            self.slices = []
            self.window = None
            torch.cuda.synchronize(self.device_index)
            self.cuda.cuMemUnmap(self.base, self.size * self.world)
            for handle in self.handles:
                self.cuda.cuMemRelease(handle)
            self.cuda.cuMemAddressFree(self.base, self.size * self.world)
            self.base = ctypes.c_ulonglong(0)


# --- quack gathered gated GEMM --------------------------------------------------------


def _load_quack():
    """Load SonicMoE the way the kernels plugin does and return the hub-vendored quack's
    ``gemm_interface`` and ``gemm_config`` modules (not any site-packages quack)."""
    from transformers.integrations.sonicmoe import load_sonicmoe_kernel

    from axolotl.integrations.kernels.plugin import _check_sonicmoe_gpu_compat

    _check_sonicmoe_gpu_compat()
    load_sonicmoe_kernel()
    candidates = [
        name
        for name, module in sys.modules.items()
        if name.endswith(".quack.gemm_interface") and module is not None
    ]
    if not candidates:
        raise RuntimeError("the sonic-moe hub kernel did not import its vendored quack")
    name = candidates[0]
    interface = sys.modules[name]
    config = sys.modules[name.rsplit(".", 1)[0] + ".gemm_config"]
    return interface, config, {"module": name, "file": interface.__file__}


class GatedGemm:
    def __init__(self, interface, config_module, config_kwargs, weight):
        self.fn = interface.gemm_gated_tuned.fn
        self.config = config_module.GemmConfig(**config_kwargs)
        self.config_kwargs = dict(config_kwargs)
        self.B = weight.transpose(1, 2)  # [E, K=H, N=2I] view of a [E, 2I, H] tensor
        rows = CAP
        device = weight.device
        self.preact = torch.empty(rows, 2 * INNER, device=device, dtype=torch.bfloat16)
        self.postact = torch.empty(rows, INNER, device=device, dtype=torch.bfloat16)
        self.cu_seqlens_m = (
            torch.arange(LOCAL_EXPERTS + 1, device=device, dtype=torch.int32)
            * (rows // LOCAL_EXPERTS)
        ).contiguous()

    def __call__(self, A, A_idx):
        self.fn(
            A,
            self.B,
            self.preact,
            self.postact,
            None,
            None,
            "swiglu",
            self.cu_seqlens_m,
            A_idx,
            False,
            config=self.config,
        )


# --- harness --------------------------------------------------------------------------


def _timed(fn, group, participate=True, warmup=WARMUP, measured=MEASURED):
    """Median of CUDA-event times over ``measured`` iterations after ``warmup``, with
    ``dist.barrier`` + ``synchronize`` around each; per-rank medians gathered."""
    samples = []
    for i in range(warmup + measured):
        dist.barrier(group)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if participate:
            fn()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            samples.append(start.elapsed_time(end))
    per_rank = [None] * dist.get_world_size(group)
    dist.all_gather_object(per_rank, (participate, samples), group=group)
    active = [r for r, (p, _s) in enumerate(per_rank) if p]
    medians = {r: statistics.median(s) for r, (p, s) in enumerate(per_rank) if p}
    return {
        "median_ms": max(medians.values()),
        "per_rank_median_ms": [medians.get(r) for r in range(len(per_rank))],
        "active_ranks": active,
        "iterations": measured,
        "warmup": warmup,
        "samples_ms_per_rank": [s if p else None for p, s in per_rank],
    }


def _with_bandwidth(entry, bytes_per_active_rank, nominal_gbps):
    gbps = bytes_per_active_rank / (entry["median_ms"] / 1000) / 1e9
    entry["peer_bytes_per_active_rank"] = bytes_per_active_rank
    entry["gbps_per_active_rank"] = gbps
    entry["fraction_of_nominal_nvlink_one_way"] = gbps / nominal_gbps
    return entry


def _compare(out, ref):
    diff = (out.float() - ref.float()).abs()
    return {
        "bitwise_equal": bool(torch.equal(out, ref)),
        "max_abs_diff": float(diff.max()),
        "mismatched_elements": int((out != ref).sum()),
    }


def _all_ranks_true(flag, group):
    t = torch.tensor([float(flag)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(t.item())


def _nvidia_smi(args):
    try:
        return subprocess.run(  # nosec B603 B607
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception as error:  # noqa: BLE001
        return f"unavailable: {error!r}"


def run(gpu):
    from mono_ep.peer_dispatch import SymmetricWorkspace
    from mono_ep.peer_packing import pack_peer_rows

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    group = dist.group.WORLD
    world = dist.get_world_size()
    assert world == 2, "this microbench is written for two ranks"
    peer = 1 - rank
    device = torch.device("cuda", rank)
    nominal = NVLINK_ONE_WAY_GBPS[gpu]
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    payload = {
        "benchmark": "peer_gather_microbench",
        "device": torch.cuda.get_device_name(),
        "world_size": world,
        "rows": CAP,
        "hidden": HIDDEN,
        "intermediate": INNER,
        "local_experts": LOCAL_EXPERTS,
        "row_bytes": ROW_BYTES,
        "a_bytes": CAP * ROW_BYTES,
        "timing": "CUDA events per rank; dist.barrier + synchronize around each iteration; median over measured iterations; median_ms is the max over active ranks",
        "nominal_nvlink_one_way_gbps": nominal,
        "versions": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "python": sys.version.split()[0],
        },
        "topology": {
            "nvidia_smi_topo": _nvidia_smi(["topo", "-m"]) if rank == 0 else None,
            "nvlink_status_gpu0": _nvidia_smi(["nvlink", "--status", "-i", "0"])
            if rank == 0
            else None,
        },
        "stages_completed": [],
        "errors": {},
        "raw_transfer": {},
        "gemm": {},
    }

    def snapshot(stage):
        payload["stages_completed"].append(stage)
        if rank == 0:
            print("RESULT " + json.dumps(payload), flush=True)

    torch.manual_seed(1000 + rank)
    x = torch.randn(CAP, HIDDEN, device=device, dtype=torch.bfloat16)
    workspace = SymmetricWorkspace(
        group, device, {"x": ((CAP, HIDDEN), torch.bfloat16)}
    )
    views = workspace.views["x"]
    views[rank].copy_(x)
    torch.cuda.synchronize()
    dist.barrier(group)
    x_all = torch.empty(world * CAP, HIDDEN, device=device, dtype=torch.bfloat16)
    dist.all_gather_into_tensor(x_all, x, group=group)
    x_all = x_all.view(world, CAP, HIDDEN)
    peer_local = x_all[peer].contiguous()  # the peer's rows, resident locally
    assert torch.equal(views[peer], peer_local), "peer symmetric view mismatch"
    dst = torch.empty_like(x)
    programs = sms * 4

    # ---- 1. raw transfers ----
    raw = payload["raw_transfer"]
    copy_bytes = CAP * ROW_BYTES
    for direction, participate in (("one_way", rank == 0), ("both_ways", True)):
        raw[f"triton_copy_peer_rows/{direction}"] = _with_bandwidth(
            _timed(
                lambda: triton_copy_rows(views[peer], dst, programs), group, participate
            ),
            copy_bytes,
            nominal,
        )
        assert torch.equal(dst, peer_local) or not participate
        raw[f"torch_copy_peer_rows/{direction}"] = _with_bandwidth(
            _timed(lambda: dst.copy_(views[peer]), group, participate),
            copy_bytes,
            nominal,
        )
    raw["triton_copy_local_rows"] = _with_bandwidth(
        _timed(lambda: triton_copy_rows(peer_local, dst, programs), group),
        copy_bytes,
        nominal,
    )
    raw["triton_copy_programs"] = programs
    assignments_peer = (
        peer * CAP + torch.arange(CAP, device=device, dtype=torch.int64)
    ).contiguous()
    assignments_local = (
        rank * CAP + torch.arange(CAP, device=device, dtype=torch.int64)
    ).contiguous()
    for direction, participate in (("one_way", rank == 0), ("both_ways", True)):
        raw[f"pack_peer_rows_peer/{direction}"] = _with_bandwidth(
            _timed(
                lambda: pack_peer_rows(list(views), assignments_peer, 1),
                group,
                participate,
            ),
            copy_bytes,
            nominal,
        )
    raw["pack_peer_rows_local"] = _with_bandwidth(
        _timed(lambda: pack_peer_rows(list(views), assignments_local, 1), group),
        copy_bytes,
        nominal,
    )
    assert torch.equal(pack_peer_rows(list(views), assignments_peer, 1), peer_local)
    snapshot("raw_sm_copies")

    a2a_in = torch.empty(2 * CAP, HIDDEN, device=device, dtype=torch.bfloat16)
    a2a_in[:CAP].copy_(x)
    a2a_in[CAP:].copy_(x)
    a2a_out = torch.empty_like(a2a_in)
    raw["nccl_all_to_all_single_512MB_to_peer/both_ways"] = _with_bandwidth(
        _timed(lambda: dist.all_to_all_single(a2a_out, a2a_in, group=group), group),
        copy_bytes,
        nominal,
    )
    rows_350 = NCCL_350MB_ROWS
    raw["nccl_all_to_all_single_350MB_per_rank/both_ways"] = _with_bandwidth(
        _timed(
            lambda: dist.all_to_all_single(
                a2a_out[:rows_350], a2a_in[:rows_350], group=group
            ),
            group,
        ),
        (rows_350 // 2) * ROW_BYTES,
        nominal,
    )
    raw["nccl_all_to_all_single_350MB_per_rank/both_ways"]["note"] = (
        "350 MB input per rank as in the EP8 torch backend; half of it crosses NVLink at EP2"
    )
    send_buf, recv_buf = a2a_in[:CAP], a2a_out[:CAP]

    def send_recv_one_way():
        if rank == 0:
            dist.send(send_buf, dst=1, group=group)
        else:
            dist.recv(recv_buf, src=0, group=group)

    def send_recv_both_ways():
        ops = [
            dist.P2POp(dist.isend, send_buf, peer, group),
            dist.P2POp(dist.irecv, recv_buf, peer, group),
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    raw["nccl_send_recv_512MB/one_way"] = _with_bandwidth(
        _timed(send_recv_one_way, group), copy_bytes, nominal
    )
    raw["nccl_send_recv_512MB/both_ways"] = _with_bandwidth(
        _timed(send_recv_both_ways, group), copy_bytes, nominal
    )
    del a2a_in, a2a_out
    snapshot("nccl")

    # ---- 2. gathered GEMM ----
    gemm_section = payload["gemm"]
    try:
        interface, config_module, quack_info = _load_quack()
        payload["quack"] = quack_info
        import cutlass

        payload["versions"]["cutlass_dsl"] = getattr(cutlass, "__version__", None)
    except Exception:  # noqa: BLE001
        payload["errors"]["load_quack"] = traceback.format_exc()
        snapshot("gemm_load_failed")
        workspace.destroy()
        dist.barrier(group)
        dist.destroy_process_group()
        return
    torch.manual_seed(7)
    weight = (
        torch.randn(LOCAL_EXPERTS, 2 * INNER, HIDDEN, device=device) / HIDDEN**0.5
    ).to(torch.bfloat16)
    generator = torch.Generator(device=device).manual_seed(11 + rank)
    perm = torch.randperm(CAP, device=device, generator=generator).to(torch.int32)
    half = CAP // 2
    span = torch.cat(
        [
            torch.randperm(CAP, device=device, generator=generator)[:half],
            CAP + torch.randperm(CAP, device=device, generator=generator)[:half],
        ]
    )
    span = span[torch.randperm(CAP, device=device, generator=generator)].to(torch.int32)
    span_peer_rows = int((span // CAP == peer).sum())
    window = None
    try:
        window = PeerWindow(group, rank, CAP, HIDDEN)
        window.slices[rank].copy_(x)
        torch.cuda.synchronize()
        dist.barrier(group)
        payload["window"] = {
            "built": True,
            "granularity_bytes": window.granularity,
            "base": hex(window.base.value),
            "own_slice_roundtrip_equal": bool(torch.equal(window.slices[rank], x)),
            "peer_slice_equal_to_nccl_copy": bool(
                torch.equal(window.slices[peer], peer_local)
            ),
        }
        window_local = x_all.view(2 * CAP, HIDDEN)  # same rows, resident locally
        assert torch.equal(window.window, window_local)
    except Exception:  # noqa: BLE001
        payload["window"] = {
            "built": False,
            "steps_completed": getattr(window, "steps", None),
            "error": traceback.format_exc(),
        }
        window = None
    if not _all_ranks_true(window is not None, group):
        if window is not None:
            window.destroy()
            payload["window"]["built_on_all_ranks"] = False
        window = None
    snapshot("window")

    placements = [
        ("local", lambda: (peer_local, perm), lambda: (peer_local, perm)),
        ("peer_symm", lambda: (views[peer], perm), lambda: (peer_local, perm)),
    ]
    if window is not None:
        placements += [
            (
                "window_local_slice",
                lambda: (window.window, rank * CAP + perm),
                lambda: (window_local, rank * CAP + perm),
            ),
            (
                "window_peer_slice",
                lambda: (window.window, peer * CAP + perm),
                lambda: (window_local, peer * CAP + perm),
            ),
            (
                "window_both_slices",
                lambda: (window.window, span),
                lambda: (window_local, span),
            ),
        ]
    peer_rows = {
        "local": 0,
        "peer_symm": CAP,
        "window_local_slice": 0,
        "window_peer_slice": CAP,
        "window_both_slices": span_peer_rows,
    }
    gemm_section["placements"] = {
        "local": "A = local copy of the peer's rows (caching allocator)",
        "peer_symm": "A = peer's torch symmetric-memory buffer (local VA mapping of remote memory)",
        "window_local_slice": "A = VMM window [2*cap, H], indices in this rank's slice",
        "window_peer_slice": "A = VMM window, indices in the peer's slice",
        "window_both_slices": "A = VMM window, 65536 rows from each slice, shuffled",
    }
    gemm_section["peer_rows_per_placement"] = peer_rows
    gemm_section["variants"] = {}
    config_sets = GEMM_CONFIGS[gpu]
    # cp.async path first on SM100: a TMA fault on peer memory would be sticky
    order = [k for k in ("cp_async", "tma_gather4") if k in config_sets]
    for variant in order:
        kwargs = dict(config_sets[variant])
        entry = {"config": dict(kwargs), "placements": {}, "errors": {}}
        gemm_section["variants"][variant] = entry
        gemm = None
        for attempt in (kwargs, {**kwargs, "tile_n": 256}):
            try:
                gemm = GatedGemm(interface, config_module, attempt, weight)
                started = time.perf_counter()
                gemm(peer_local, perm)
                torch.cuda.synchronize()
                entry["compile_and_first_call_seconds"] = time.perf_counter() - started
                entry["config"] = dict(attempt)
                break
            except Exception:  # noqa: BLE001
                entry["errors"][f"build:{json.dumps(attempt, sort_keys=True)}"] = (
                    traceback.format_exc()
                )
                gemm = None
        built = _all_ranks_true(gemm is not None, group)
        if not built:
            entry["skipped"] = "kernel did not build on every rank"
            snapshot(f"gemm_{variant}_build_failed")
            continue
        reference = {}
        for placement, args, ref_args in placements:
            A, idx = args()
            A_ref, idx_ref = ref_args()
            gemm(A_ref, idx_ref)
            torch.cuda.synchronize()
            reference["preact"] = gemm.preact.clone()
            reference["postact"] = gemm.postact.clone()
            gemm.preact.zero_()
            gemm.postact.zero_()
            try:
                gemm(A, idx)
                torch.cuda.synchronize()
                parity = {
                    "preact": _compare(gemm.preact, reference["preact"]),
                    "postact": _compare(gemm.postact, reference["postact"]),
                }
                ok = True
            except Exception:  # noqa: BLE001
                entry["errors"][f"{placement}:first_call"] = traceback.format_exc()
                parity = None
                ok = False
            if not _all_ranks_true(ok, group):
                entry["placements"][placement] = {"failed": True, "parity": parity}
                snapshot(f"gemm_{variant}_{placement}_failed")
                continue
            timing = _timed(lambda A=A, idx=idx, gemm=gemm: gemm(A, idx), group)
            timing["parity_vs_local_data"] = parity
            timing["peer_rows"] = peer_rows[placement]
            timing["peer_bytes"] = peer_rows[placement] * ROW_BYTES
            entry["placements"][placement] = timing
            snapshot(f"gemm_{variant}_{placement}")
        local_ms = entry["placements"].get("local", {}).get("median_ms")
        if local_ms is not None:
            for _placement, timing in entry["placements"].items():
                if timing.get("failed"):
                    continue
                extra = timing["median_ms"] - local_ms
                timing["extra_ms_over_local"] = extra
                if timing["peer_bytes"]:
                    timing["effective_peer_gbps_if_unhidden"] = (
                        timing["peer_bytes"] / max(extra, 1e-6) / 1e6
                    )
        del gemm
        torch.cuda.empty_cache()

    if window is not None:
        window.destroy()
    workspace.destroy()
    snapshot("done")
    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", choices=("H100", "B200"), required=True)
    args = parser.parse_args()
    if not args.worker:
        parser.error("run under torch.distributed.run with --worker")
    run(args.gpu)
