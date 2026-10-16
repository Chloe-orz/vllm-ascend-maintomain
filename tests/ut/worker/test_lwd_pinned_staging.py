# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdPinnedStaging 表驱动等价测试(S1 R2 收敛的等价锚)。

对当关系(不写进注释的等价性不可信):
- ``stage_to_device`` ≍ 旧 ``LwdEdgeWorker._lwd_embed_ids_h2d`` / 旧
  ``LwdCloudModelRunner._lwd_h2d_stage_i64`` 的"pinned 写入 +
  non_blocking H2D + 返回 dev[:n]":4 槽轮换、容量 max(n, floor)
  增长、增长重建后槽序归零,三语义逐条对应;
- ``next_pinned`` ≍ 旧 ``_lwd_meta_pinned`` 的纯 pinned 4 槽轮换
  (容量下限 4096)。

真实环境补跑条件:训练栈主机(torch + NPU/CUDA;pinned 内存需要
加速器宿主分配器,无 torch 或仅 CPU torch 的环境整体 skip)。
结构层面的守卫(源码扫描)在 test_lwd_s1s2_refactor_guard.py,任何
环境必跑。
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
        (8, [3, 5, 7]),  # 常规轮换,无增长
        (8, [3, 5, 7, 9, 4]),  # 第 5 次上传回到槽 0(4 槽环)
        (8, [64]),  # 超容量 → 增长重建
        (8, [64, 3]),  # 增长后小载荷仍正确(槽序归零后继续轮换)
        (1024, [1]),  # min_capacity 下限生效
        (4, [4, 4, 4, 4, 4]),  # 定长重复上传(最常见稳态)
    ],
)
def test_upload_roundtrip_per_step(min_capacity: int, sizes: list[int]) -> None:
    """每次上传后立即读回:视图值即本次暂存(同流序,legacy 同款保证)。"""
    ring = LwdPinnedStaging(
        dtype=torch.int64, min_capacity=min_capacity, device=_DEVICE,
    )
    for step, n in enumerate(sizes):
        src = list(range(step * 100, step * 100 + n))
        view = ring.stage_to_device(src, n)
        assert view.numel() == n
        assert view.tolist() == src


def test_upload_rotation_depth_window() -> None:
    """4 槽窗口内互不重叠(在飞传输不被下一步覆写),第 5 次回到槽 0。"""
    ring = LwdPinnedStaging(dtype=torch.int64, min_capacity=8, device=_DEVICE)
    ptrs = [ring.stage_to_device([i], 1).data_ptr() for i in range(4)]
    assert len(set(ptrs)) == 4
    assert ring.stage_to_device([9], 1).data_ptr() == ptrs[0]


def test_upload_growth_rebuild_restarts_rotation() -> None:
    """超容量触发整环重建(max(n, floor)),重建后功能不变。"""
    ring = LwdPinnedStaging(dtype=torch.int64, min_capacity=8, device=_DEVICE)
    ring.stage_to_device([1], 1)
    big = list(range(100))
    assert ring.stage_to_device(big, 100).tolist() == big
    assert ring.stage_to_device([7, 8], 2).tolist() == [7, 8]


def test_take_pinned_rotation_and_growth() -> None:
    """纯 pinned 模式(meta 轮换同款):4 槽轮换、回到槽 0、超容量重建。"""
    ring = LwdPinnedStaging(dtype=torch.int32, min_capacity=4096)
    slots = [ring.next_pinned(n) for n in (10, 20, 30, 40)]
    assert len({t.data_ptr() for t in slots}) == 4
    assert ring.next_pinned(50).data_ptr() == slots[0].data_ptr()
    grown = ring.next_pinned(5000)
    assert grown.numel() >= 5000
