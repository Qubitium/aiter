# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cross-M stability policy tests for mHC launch selection.

mHC has two independent M-dependent launch policies:

  * the pre GEMM split-K selector (``get_mhc_pre_splitk``), and
  * the fused pre RMSNorm row-group dispatch in ``csrc/kernels/mhc_kernels.cu``.

Both are individually deterministic, but each changes its FP32 reduction tree
with the runtime batch size, so the low bits of ``h_post``/``h_res`` differ
between prefill, decode and speculative verification of the same token.  For a
registered (arch, cu_num, problem) family the selectors are pinned to one
topology; these tests assert the pin holds across M and that unregistered
families keep their (faster) dynamic policy.
"""

import pytest

from aiter.ops import mhc
from aiter.ops.mhc import (
    _MHC_STABLE_PRE_FAMILIES,
    _MHC_STABLE_PRE_LOGGED,
    _MHC_STABLE_RMSNORM_FAMILIES,
    _MHC_STABLE_RMSNORM_LOGGED,
    get_mhc_pre_splitk,
    get_mhc_pre_splitk_large_m,
)

# The GLM-5.3 / DeepSeek-style mHC family: hc_mult=4, hidden_size=4096.
STABLE_ARCH = "gfx950"
STABLE_CU = 256
STABLE_HC_HIDDEN = 4 * 4096

# M values that straddle every boundary of the former dynamic ladders
# (m < 4*cu_num, m <= 8*cu_num, and the large-M helper's m >= 8192).
BOUNDARY_M = [1, 2, 8, 16, 64, 1023, 1024, 2047, 2048, 4095, 4096, 8191, 8192, 16384]


@pytest.fixture(autouse=True)
def _reset_selector_state():
    """Each test starts from a cold selector/telemetry cache."""
    get_mhc_pre_splitk.cache_clear()
    _MHC_STABLE_PRE_LOGGED.clear()
    _MHC_STABLE_RMSNORM_LOGGED.clear()
    yield
    get_mhc_pre_splitk.cache_clear()
    _MHC_STABLE_PRE_LOGGED.clear()
    _MHC_STABLE_RMSNORM_LOGGED.clear()


@pytest.fixture
def stable_family(monkeypatch):
    """Pretend we are on the registered gfx950 / 256-CU family."""

    def _install(arch, num_cu):
        monkeypatch.setattr(mhc, "get_gfx_runtime", lambda: arch)
        monkeypatch.setattr(mhc, "get_cu_num", lambda: num_cu)

    _install(STABLE_ARCH, STABLE_CU)
    return _install


def test_registry_keys_match_documented_family():
    assert _MHC_STABLE_PRE_FAMILIES[(STABLE_ARCH, STABLE_CU, STABLE_HC_HIDDEN)] == (
        16,
        128,
    )
    # hc_hidden_size is hc_mult * hidden_size, so the RMSNorm registry (keyed
    # on hidden_size) must describe the same family.
    assert _MHC_STABLE_RMSNORM_FAMILIES[(STABLE_ARCH, STABLE_CU, 4096)] == (
        2,
        512,
        512,
    )


def test_gfx950_mhc_selector_family_is_cross_m_stable(stable_family):
    """Every M in the family selects the same (split_k, tile_k)."""
    seen = {get_mhc_pre_splitk(m, STABLE_HC_HIDDEN) for m in BOUNDARY_M}
    assert seen == {(16, 128)}, f"selector drifted across M: {sorted(seen)}"


def test_large_m_helper_honours_the_pin(stable_family):
    """The large-M helper must not short-circuit away from the pinned family.

    ``get_mhc_pre_splitk_large_m`` returns (8, 64) for gfx950 at m >= 8192 by
    default; without the pin the family would flip trees exactly at that
    boundary (8191 -> (16, 128), 8192 -> (8, 64)).
    """
    seen = {get_mhc_pre_splitk_large_m(m, STABLE_HC_HIDDEN) for m in BOUNDARY_M}
    assert seen == {(16, 128)}, f"large-M helper drifted across M: {sorted(seen)}"


def test_pre_and_large_m_helpers_agree_for_the_family(stable_family):
    """Both entry points share one tree, so callers cannot disagree."""
    for m in BOUNDARY_M:
        assert get_mhc_pre_splitk(m, STABLE_HC_HIDDEN) == get_mhc_pre_splitk_large_m(
            m, STABLE_HC_HIDDEN
        ), f"helpers disagree at m={m}"


def test_unregistered_mhc_family_keeps_dynamic_policy(stable_family):
    """Unregistered families keep the tuned (M-dependent) occupancy policy."""
    other_hc_hidden = 4 * 7168  # GLM-4.5 / DeepSeek hidden_size=7168
    small = get_mhc_pre_splitk(1, other_hc_hidden)
    large = get_mhc_pre_splitk(16384, other_hc_hidden)
    assert small != large, (
        "the dynamic policy is expected to keep varying with M for "
        f"unregistered families, got {small} and {large}"
    )


def test_unregistered_cu_num_does_not_pin(monkeypatch):
    """The pin is keyed on cu_num, so a different part count stays dynamic."""
    monkeypatch.setattr(mhc, "get_gfx_runtime", lambda: STABLE_ARCH)
    monkeypatch.setattr(mhc, "get_cu_num", lambda: 128)
    configs = {get_mhc_pre_splitk(m, STABLE_HC_HIDDEN) for m in (1, 16384)}
    assert (16, 128) not in configs or len(configs) > 1


def test_stable_family_logs_once(stable_family):
    """Telemetry is emitted once per family, not once per launch."""
    get_mhc_pre_splitk(1, STABLE_HC_HIDDEN)
    assert (STABLE_ARCH, STABLE_CU, STABLE_HC_HIDDEN) in _MHC_STABLE_PRE_LOGGED
    get_mhc_pre_splitk(16384, STABLE_HC_HIDDEN)
    assert len(_MHC_STABLE_PRE_LOGGED) == 1
