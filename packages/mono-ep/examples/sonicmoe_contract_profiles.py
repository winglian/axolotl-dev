# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Measured SonicMoE quack autotune profiles for the dispatch-mode contract benchmark.

Copied verbatim from axolotl's ``quack_default_profiles.py`` so the harness seeds the same
bounded candidate lists production uses: the B200 entries from main commit b0f55d39e3
(#4273) and the H100 entries from 7d4225c71f on ``features/sonicmoe-quack-h100`` (no PR).
This branch predates both."""

from typing import Any

PROFILE_SOURCES = {
    "B200": {
        "commit": "b0f55d39e38bf5e392f26b8b0d97f9df89699558",
        "path": "src/axolotl/integrations/kernels/quack_default_profiles.py",
        "ref": "main (#4273)",
    },
    "H100": {
        "commit": "7d4225c71f9164e5c7e5472e8e29c6c0fec0bbf0",
        "path": "src/axolotl/integrations/kernels/quack_default_profiles.py",
        "ref": "features/sonicmoe-quack-h100",
    },
}

_VARLEN_M = {"cu_seqlens_m.dim0": {"ge": 2}}
_VARLEN_K = {"cu_seqlens_k.dim0": {"ge": 2}}


def _config(
    tile_m, tile_n, cluster_m, cluster_n, persistent, tma_gather, swap_ab=False
):
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "is_dynamic_persistent": persistent,
        "use_tma_gather": tma_gather,
        "swap_ab": swap_ab,
    }


def _sm90(tile_m, tile_n, cluster_m, cluster_n, pingpong, swap_ab=False):
    """An sm90 config: never persistent or ``use_tma_gather`` there, so those are left out."""
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "pingpong": pingpong,
        "swap_ab": swap_ab,
    }


def _profile(kernel, match, *configs, gpu="B200"):
    return {
        "kernel": kernel,
        "backend": "quack",
        "gpu": gpu,
        "match": match,
        "configs": list(configs),
    }


# gemm_tuned entries go narrowest first: the first matching entry decides a call.
B200_SONICMOE_PROFILES: list[dict[str, Any]] = [
    # gate/up forward, with or without the LoRA addend: one config, not benchmarked
    _profile("gemm_gated_tuned", _VARLEN_M, _config(256, 512, 2, 1, True, True)),
    _profile(
        "gemm_dgated_tuned",
        _VARLEN_M,
        _config(256, 256, 2, 1, True, True),
        _config(128, 256, 2, 1, True, True),
        _config(256, 256, 2, 1, False, True),
    ),
    # expert weight grads
    _profile(
        "gemm_tuned",
        {**_VARLEN_K, "out.dim-1": {"gt": 128}, "out.dim-2": {"gt": 128}},
        _config(256, 512, 2, 1, True, True),
        _config(256, 256, 2, 1, True, True),
        _config(256, 256, 2, 1, False, False),
    ),
    # LoRA grads with r rows ([E, r, 2I])
    _profile(
        "gemm_tuned",
        {**_VARLEN_K, "out.dim-2": {"le": 128}},
        _config(256, 64, 2, 1, True, False, swap_ab=True),
        _config(128, 64, 1, 1, True, False, swap_ab=True),
    ),
    # LoRA grads with r columns ([E, H or I, r])
    _profile(
        "gemm_tuned",
        _VARLEN_K,
        _config(128, 64, 1, 1, True, False),
        _config(128, 64, 1, 1, False, False),
        _config(256, 64, 2, 1, True, False),
        _config(128, 64, 2, 1, True, False),
    ),
    # rank-r x P (N = r); the tma_gather ones only apply with A_idx
    _profile(
        "gemm_tuned",
        {**_VARLEN_M, "out.dim-1": {"le": 128}},
        _config(128, 64, 1, 1, True, False),
        _config(128, 64, 1, 1, False, False),
        _config(256, 64, 2, 1, True, False),
        _config(128, 64, 1, 1, True, True),
        _config(128, 64, 2, 1, True, True),
    ),
    # rank-r u Q (K = r)
    _profile(
        "gemm_tuned",
        {**_VARLEN_M, "B.dim-2": {"le": 128}},
        _config(256, 256, 2, 1, False, False),
        _config(128, 256, 1, 1, False, False),
    ),
    # down forward, gate/up dX, with or without the LoRA addend
    _profile(
        "gemm_tuned",
        _VARLEN_M,
        _config(256, 256, 2, 1, True, False),
        _config(256, 256, 2, 1, False, False),
        _config(256, 512, 2, 1, True, False),
        _config(128, 256, 1, 2, False, False),
    ),
]


# sm90 offers 9-44 candidates per grouped key: every one was re-timed on H100
H100_SONICMOE_PROFILES: list[dict[str, Any]] = [
    _profile("gemm_gated_tuned", _VARLEN_M, _sm90(128, 256, 2, 1, False), gpu="H100"),
    _profile("gemm_dgated_tuned", _VARLEN_M, _sm90(128, 256, 2, 1, False), gpu="H100"),
    _profile(
        "gemm_tuned",
        {**_VARLEN_K, "out.dim-1": {"gt": 128}, "out.dim-2": {"gt": 128}},
        _sm90(128, 256, 2, 1, False),
        _sm90(128, 192, 2, 1, True),
        _sm90(256, 192, 2, 1, False),
        _sm90(256, 160, 2, 1, False),
        gpu="H100",
    ),
    _profile(
        "gemm_tuned",
        {**_VARLEN_K, "out.dim-2": {"le": 128}},
        _sm90(128, 128, 1, 2, True),
        _sm90(256, 128, 2, 1, False, swap_ab=True),
        _sm90(128, 192, 1, 2, True),
        gpu="H100",
    ),
    _profile(
        "gemm_tuned",
        _VARLEN_K,
        _sm90(128, 128, 2, 1, True),
        _sm90(256, 128, 2, 1, False),
        _sm90(192, 128, 2, 1, True),
        gpu="H100",
    ),
    _profile(
        "gemm_tuned",
        {**_VARLEN_M, "out.dim-1": {"le": 128}},
        _sm90(128, 128, 2, 1, True),
        _sm90(256, 128, 2, 1, False),
        _sm90(192, 128, 2, 1, True),
        _sm90(256, 128, 1, 2, False),
        gpu="H100",
    ),
    _profile(
        "gemm_tuned",
        {**_VARLEN_M, "B.dim-2": {"le": 128}},
        _sm90(128, 256, 2, 1, False),
        _sm90(128, 256, 1, 2, False),
        _sm90(256, 128, 2, 1, False),
        gpu="H100",
    ),
    _profile(
        "gemm_tuned",
        _VARLEN_M,
        _sm90(192, 128, 1, 2, True),
        _sm90(256, 128, 1, 2, False),
        _sm90(128, 256, 1, 2, False),
        gpu="H100",
    ),
]


PROFILE_SETS = {"B200": B200_SONICMOE_PROFILES, "H100": H100_SONICMOE_PROFILES}
