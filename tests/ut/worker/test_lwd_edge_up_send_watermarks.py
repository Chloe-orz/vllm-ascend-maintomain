# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for per-DP contiguous UP send-completion watermarks."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pytest

pytest.importorskip(
    "torch",
    reason="edge runner imports torch; rerun on a host with the training stack",
)

from vllm.v1.core.sched.output import (  # noqa: E402
    LwdEmbedBatch,
    LwdUnembedBatch,
    SchedulerOutput,
)

from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import (  # noqa: E402
    LwdTransferFuture,
)
from vllm_ascend.worker.lwd_edge_model.lwd_edge_model_runner import (  # noqa: E402
    LwdEdgeModelRunner,
)
from vllm_ascend.worker.lwd_edge_model.lwd_edge_worker import (  # noqa: E402
    LwdEdgeWorker,
)


class _FakeFuture:
    def __init__(self, completed: bool = False, error: RuntimeError | None = None) -> None:
        self._completed = completed
        self._error = error

    def done(self) -> bool:
        return self._completed

    def complete(self) -> None:
        self._completed = True

    def result(self) -> None:
        if self._error is not None:
            raise self._error


def _make_runner() -> LwdEdgeModelRunner:
    runner = LwdEdgeModelRunner.__new__(LwdEdgeModelRunner)
    runner._lwd_up_send_inflight = {}
    runner._lwd_up_send_watermarks = {}
    return runner


def _add_send(
    runner: LwdEdgeModelRunner,
    dp_idx: int,
    seqno: int,
    future: _FakeFuture,
) -> None:
    pending = runner._lwd_up_send_inflight.setdefault(dp_idx, deque())
    pending.append((seqno, cast(LwdTransferFuture, future)))


def test_watermark_advances_across_completed_prefix() -> None:
    runner = _make_runner()
    futures = [_FakeFuture(completed=True), _FakeFuture(completed=True), _FakeFuture()]
    for seqno, future in enumerate(futures):
        _add_send(runner, 0, seqno, future)

    assert runner._lwd_watermarked_output(0).lwd_up_send_watermarks == {0: 1}

    futures[2].complete()
    assert runner._lwd_watermarked_output(0).lwd_up_send_watermarks == {0: 2}


def test_out_of_order_completion_cannot_advance_watermark() -> None:
    runner = _make_runner()
    futures = [_FakeFuture(), _FakeFuture(completed=True), _FakeFuture(completed=True)]
    for seqno, future in enumerate(futures):
        _add_send(runner, 0, seqno, future)

    assert runner._lwd_watermarked_output(0).lwd_up_send_watermarks == {0: -1}

    futures[0].complete()
    assert runner._lwd_watermarked_output(0).lwd_up_send_watermarks == {0: 2}


def test_same_seqno_is_isolated_and_embed_returns_current_dp() -> None:
    runner = _make_runner()
    dp0_future = _FakeFuture(completed=True)
    dp1_future = _FakeFuture()
    _add_send(runner, 0, 0, dp0_future)
    _add_send(runner, 1, 0, dp1_future)
    scheduler_output = cast(SchedulerOutput, SimpleNamespace(scheduled_new_reqs=[]))

    with patch.object(runner, "get_model", return_value=None):
        dp0_output = runner.execute_lwd_embed(0, LwdEmbedBatch(dp_idx=0), scheduler_output)
        dp1_output = runner.execute_lwd_embed(0, LwdEmbedBatch(dp_idx=1), scheduler_output)

    assert dp0_output.lwd_up_send_watermarks == {0: 0}
    assert dp1_output.lwd_up_send_watermarks == {1: -1}


def test_empty_unembed_returns_all_dp_watermarks() -> None:
    runner = _make_runner()
    _add_send(runner, 0, 0, _FakeFuture(completed=True))
    _add_send(runner, 1, 0, _FakeFuture())

    with patch.object(runner, "get_model", return_value=None):
        output = runner.execute_lwd_unembed(0, LwdUnembedBatch(dp_idx=1))

    assert output.lwd_up_send_watermarks == {0: 0, 1: -1}


@pytest.mark.parametrize(
    "lwd_batch",
    [None, SimpleNamespace(batch_type=object(), batch_meta=None)],
    ids=["empty-batch", "unknown-batch"],
)
def test_worker_step_without_compute_returns_all_dp_watermarks(lwd_batch: object | None) -> None:
    runner = _make_runner()
    _add_send(runner, 0, 0, _FakeFuture(completed=True))
    _add_send(runner, 1, 0, _FakeFuture())
    worker = LwdEdgeWorker.__new__(LwdEdgeWorker)
    worker.model_runner = runner
    worker.comm_service = SimpleNamespace(poll_completions=Mock())
    scheduler_output = cast(
        SchedulerOutput,
        SimpleNamespace(lwd_batch=lwd_batch, finished_req_ids=[]),
    )

    output = worker.execute_model(scheduler_output)

    assert output.lwd_up_send_watermarks == {0: 0, 1: -1}
    worker.comm_service.poll_completions.assert_called_once_with()


def test_send_error_does_not_pop_or_advance_watermark() -> None:
    runner = _make_runner()
    future = _FakeFuture(completed=True, error=RuntimeError("send failed"))
    _add_send(runner, 0, 0, future)

    with pytest.raises(RuntimeError, match="send failed"):
        runner._lwd_watermarked_output(0)

    assert runner._lwd_up_send_watermarks == {}
    assert [seqno for seqno, _ in runner._lwd_up_send_inflight[0]] == [0]
