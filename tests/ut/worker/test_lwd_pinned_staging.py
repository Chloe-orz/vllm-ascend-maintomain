# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Table-driven equivalence tests for LwdPinnedStaging (the equivalence

anchor of the S1 R2 consolidation).
Correspondence (equivalence not written down cannot be trusted):
  - ``stage_to_device`` ~= the old ``LwdEdgeWorker._lwd_embed_ids_h2d`` /
  ``LwdCloudModelRunner._lwd_h2d_stage_i64`` "pinned write + non_blocking
  H2D + return dev[:n]": 4-slot rotation, capacity growth to max(n, floor),
rotation restart after a growth rebuild — semantics correspond term by term;
  - ``next_pinned`` ~= the old ``_lwd_meta_pinned`` pure pinned 4-slot

rotation (capacity floor 4096).
Real-environment rerun condition: a training-stack host (torch + NPU/CUDA;
pinned memory needs an accelerator host allocator — environments without
torch or with CPU-only torch skip wholesale). Structural guards (source
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not (getattr(torch, "npu", None) or torch.cuda.is_available()),
    reason="pinned memory needs an accelerator host allocator; "
    "rerun on the training-stack host",
)

_ROOT = Path(__file__).resolve().parents[3]
_MODULE = _ROOT / "vllm_ascend/worker/lwd_base_model/lwd_pinned_staging.py"
_SPEC = importlib.util.spec_from_file_location("lwd_pinned_staging", _MODULE)
_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_module)
LwdPinnedStaging = _module.LwdPinnedStaging

_DEVICE = "npu" if getattr(torch, "npu", None) else "cuda"


@pytest.mark.parametrize(
    "min_capacity,sizes",
    [
        (8, [3, 5, 7]),  # plain rotation, no growth
        (8, [3, 5, 7, 9, 4]),  # 5th upload returns to slot 0 (4-slot ring)
        (8, [64]),  # over capacity -> growth rebuild
        (8, [64, 3]),  # small payload still correct after growth rebuild
        (1024, [1]),  # min_capacity floor in effect
        (4, [4, 4, 4, 4, 4]),  # fixed-size repeated uploads (common steady state)
    ],
)
def test_upload_roundtrip_per_step(min_capacity: int, sizes: list[int]) -> None:
    """Read back right after each upload: the view equals this staging."""
    ring = LwdPinnedStaging(
        dtype=torch.int64, min_capacity=min_capacity, device=_DEVICE,
    )
    for step, n in enumerate(sizes):
        src = list(range(step * 100, step * 100 + n))
        view = ring.stage_to_device(src, n)
        assert view.numel() == n
        assert view.tolist() == src


def test_upload_rotation_depth_window() -> None:
    """No overlap within the 4-slot window; the 5th returns to slot 0."""
    ring = LwdPinnedStaging(dtype=torch.int64, min_capacity=8, device=_DEVICE)
    ptrs = [ring.stage_to_device([i], 1).data_ptr() for i in range(4)]
    assert len(set(ptrs)) == 4
    assert ring.stage_to_device([9], 1).data_ptr() == ptrs[0]


def test_upload_growth_rebuild_restarts_rotation() -> None:
    """Over-capacity triggers a whole-ring rebuild (max(n, floor)); behavior unchanged."""
    ring = LwdPinnedStaging(dtype=torch.int64, min_capacity=8, device=_DEVICE)
    ring.stage_to_device([1], 1)
    big = list(range(100))
    assert ring.stage_to_device(big, 100).tolist() == big
    assert ring.stage_to_device([7, 8], 2).tolist() == [7, 8]


def test_take_pinned_rotation_and_growth() -> None:
    """Pinned-only mode (same rotation as meta): 4-slot rotation, slot-0 return, rebuild."""
    ring = LwdPinnedStaging(dtype=torch.int32, min_capacity=4096)
    slots = [ring.next_pinned(n) for n in (10, 20, 30, 40)]
    assert len({t.data_ptr() for t in slots}) == 4
    assert ring.next_pinned(50).data_ptr() == slots[0].data_ptr()
    grown = ring.next_pinned(5000)
    assert grown.numel() >= 5000
