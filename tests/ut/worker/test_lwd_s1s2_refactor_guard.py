# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""S1/S2 重构守卫(纯源码断言,无 torch,任何环境必跑)。

锚定 docs/refactor/lwd_worker_packages_refactor_plan.md S1/S2 的关键
等价与结构承诺,防止回归:
- R1: 死代码 compute_top_id_th 已删;
- R2: 三处手写 pinned 环收敛到 LwdPinnedStaging(槽序变量清零);
- R3: 端点判定单源(runner 不再重算 edge_npu_count);
- F6a/R4: 收割逐 seqno 交错拆分(_lwd_stage_up_batch → _lwd_harvest
  __tensors_one → _lwd_check_up_windows),事件序与拆分前单循环逐点
  一致,整体位于 super()._prepare_inputs 之前(暂缓项 D1 的边界);
- R4: 注入段复用收割携带的批位,不再重查 req_id_to_index;
- 三类 fail-fast 消息(静默乱码防线)原文保留;
- F6b: DOWN 采集拆分后 runner 经 worker.note_lwd_meta_ready 记事件,
  不再直写 worker 私有属性;
- R9: 完结清理单主(runner.lwd_release_finished 唯一落点,worker 只
  转发)。

时序拆分的已知良性差异(对当关系):meta 段核验先于张量段 host 门,
失败路径下异常消息逐字一致、仅在等待前后区别;成功路径设备序不变。
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER = (
    _REPO_ROOT
    / "vllm_ascend"
    / "worker"
    / "lwd_cloud_model"
    / "lwd_cloud_model_runner.py"
)
_WORKER = (
    _REPO_ROOT
    / "vllm_ascend"
    / "worker"
    / "lwd_cloud_model"
    / "lwd_cloud_worker.py"
)
_EDGE = (
    _REPO_ROOT / "vllm_ascend" / "worker" / "lwd_edge_model" / "lwd_edge_worker.py"
)


def _method(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found in {path.name}")


def _segment(path: Path, name: str) -> str:
    fn = _method(path, name)
    seg = ast.get_source_segment(path.read_text(encoding="utf-8"), fn)
    assert seg is not None
    return seg


def _call_names(fn: ast.FunctionDef) -> list[tuple[int, str]]:
    """(lineno, called name) for attribute calls, in source order."""
    calls = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.append((node.lineno, node.func.attr))
    return sorted(calls)


def test_dead_code_removed() -> None:
    src = _EDGE.read_text(encoding="utf-8")
    assert "compute_top_id_th" not in src, "dead helper must not come back"


def test_no_is_lwd_gates_inside_lwd_runner() -> None:
    """团队规范:lwd 类/函数内部不再判断"是否 lwd"。

    入口不变量:platform.py 仅在 lwd_config.enabled 时选择
    LwdCloudWorker,而 LwdCloudModelRunner 全仓唯一构造点在其
    init_device——类内任何 enabled / is-not-None 门都是恒真死判断
    (mode 维度的分支不属于"是否 lwd",允许保留)。
    """
    src = _RUNNER.read_text(encoding="utf-8")
    assert "_lwd_enabled" not in src
    assert "lwd_config.enabled" not in src
    assert 'getattr(self.vllm_config, "lwd_config", None)' not in src


def test_draft_cache_weave_removed() -> None:
    """LwdDraftCache 删除守卫(与基类 spec recovery chain 删除同款论证)。

    pdmix 现行线上恢复路径零消费:decode 不被驱逐(prefill notify 整批
    defer)、原生 max_tokens 守卫是 depart-and-finish(请求必然完结,
    不产生回归步)、V2 节拍守卫在云侧(V1 强制)是死路。删除后云
    runner 的投机路径全原生(_update_states/_prepare_input_ids 不再
    覆写)。若相位臂(phase_scheduler)重新接线,须连同"离批草稿
    保管"机制一并恢复。
    """
    runner_src = _RUNNER.read_text(encoding="utf-8")
    for symbol in ("LwdDraftCache", "_lwd_draft_cache", "_lwd_restore_drafts"):
        assert symbol not in runner_src, f"draft-cache weave leaked back: {symbol}"
    for override in ("def _update_states(", "def _prepare_input_ids("):
        assert override not in runner_src, (
            f"native {override[4:-1]} must stay unoverridden on the cloud runner"
        )
    assert not (
        _REPO_ROOT / "vllm_ascend/worker/lwd_cloud_model/lwd_draft_cache.py"
    ).exists(), "deleted module must not come back"


def test_pinned_staging_single_implementation() -> None:
    runner = _RUNNER.read_text(encoding="utf-8")
    edge = _EDGE.read_text(encoding="utf-8")
    assert runner.count("LwdPinnedStaging(") == 2  # meta + h2d staging
    assert edge.count("LwdPinnedStaging(") == 1  # ids staging
    for stale in ("_lwd_pinned_ring_idx", "_lwd_h2d_ring_idx"):
        assert stale not in runner, f"hand-rolled rotation leaked: {stale}"
    assert "_lwd_ids_ring_idx" not in edge


def test_endpoint_single_source() -> None:
    runner = _RUNNER.read_text(encoding="utf-8")
    assert "edge_npu_count" not in runner, (
        "runner must reuse worker._lwd_early_recv_endpoint, not recompute"
    )
    assert "_lwd_early_recv_endpoint" in runner


def test_harvest_split_and_pre_super_order() -> None:
    prepare = _method(_RUNNER, "_prepare_inputs")
    harvest_lineno = None
    super_lineno = None
    for lineno, name in _call_names(prepare):
        if name == "_lwd_harvest_up_batches" and harvest_lineno is None:
            harvest_lineno = lineno
        if name == "_prepare_inputs" and super_lineno is None:
            super_lineno = lineno
    assert harvest_lineno is not None and super_lineno is not None
    assert harvest_lineno < super_lineno, (
        "harvest (both phases) must stay before super()._prepare_inputs; "
        "moving the tensor phase is deferred item D1, not this refactor"
    )

    orch = _segment(_RUNNER, "_lwd_harvest_up_batches")
    stage_pos = orch.index("_lwd_stage_up_batch")
    tensor_pos = orch.index("_lwd_harvest_tensors_one")
    window_pos = orch.index("_lwd_check_up_windows")
    assert stage_pos < tensor_pos < window_pos, (
        "per-seqno order must mirror the pre-split loop: length check, "
        "host gate + broadcast, window check + mrope (log order included)"
    )
    for helper in ("_lwd_stage_up_batch", "_lwd_harvest_tensors_one",
                   "_lwd_check_up_windows"):
        assert f"def {helper}(" in _RUNNER.read_text(encoding="utf-8")


def test_harvest_failfast_messages_preserved() -> None:
    src = _RUNNER.read_text(encoding="utf-8")
    for fragment in (
        "req not in input_batch",
        "INJECT window mismatch",
        "prompt chunk offset mismatch",
        "mrope payload missing",
        "LWD requires one prompt offset and mrope flag per request",
    ):
        assert fragment in src, f"fail-fast line dropped: {fragment}"


def test_inject_reuses_carried_index() -> None:
    seg = _segment(_RUNNER, "_lwd_inject_remote_embeds")
    assert "req_id_to_index" not in seg, (
        "inject must reuse the index carried by the harvest meta phase"
    )
    assert "offsets[idx]" in seg


def test_meta_ready_via_worker_api() -> None:
    runner = _RUNNER.read_text(encoding="utf-8")
    worker = _WORKER.read_text(encoding="utf-8")
    assert "note_lwd_meta_ready()" in runner
    # The runner must not touch the worker's private attribute; the
    # executor still reads it, so the writer stays on the worker.
    assert "_lwd_meta_ready_event" not in runner
    assert "def note_lwd_meta_ready" in worker
    assert "_lwd_meta_ready_event" in _segment(_WORKER, "note_lwd_meta_ready")


def test_collect_split_and_payload_protocol() -> None:
    src = _RUNNER.read_text(encoding="utf-8")
    for helper in ("_lwd_pack_plain_down_rows", "_lwd_pack_spec_down_rows",
                   "_lwd_stage_down_meta"):
        assert f"def {helper}(" in src
    orch = _segment(_RUNNER, "_lwd_collect_down_payload")
    assert "_lwd_stage_down_meta(" in orch
    # The 5-tuple DOWN payload protocol (hidden, pinned, None, req_ids,
    # None) with req_ids filtered by the captured snapshot mask.
    assert "[r for i, r in enumerate(batch_req_ids) if valid[i]]" in orch


def test_finished_release_single_owner() -> None:
    worker_src = _WORKER.read_text(encoding="utf-8")
    runner_src = _RUNNER.read_text(encoding="utf-8")
    assert "def lwd_release_finished" in runner_src
    flush = _segment(_WORKER, "_lwd_cloud_flush_finished")
    assert "lwd_release_finished" in flush
    assert "hash_state.discard" not in worker_src, (
        "release logic must live on the runner only"
    )

def test_layer_trace_rehomed_and_installed() -> None:
    """layer-trace 对拍插桩归位 lwd_debug 并恢复安装点。

    c22c04b53 清基座探针时摘除了 NPUModelRunner.load_model 的安装调用
    且无人接手,约 120 行 hook/校验和机制成为孤儿。现归位:边/云各自
    load_model 安装(env 门控,缺省零开销);集中式对照臂为手动
    opt-in(基类保持零 LWD 符号,weave guard 白名单为空)。
    """
    edge = _EDGE.read_text(encoding="utf-8")
    worker = _WORKER.read_text(encoding="utf-8")
    assert "install_lwd_layer_trace(self.model_runner.get_model())" in edge
    assert "install_lwd_layer_trace(self.model_runner.get_model())" in worker
    assert "vllm_ascend.worker.lwd_debug.lwd_layer_trace" in edge
    assert (_REPO_ROOT / "vllm_ascend/worker/lwd_debug/lwd_layer_trace.py").exists()
    assert not (_REPO_ROOT / "vllm_ascend/worker/lwd_layer_trace.py").exists()
