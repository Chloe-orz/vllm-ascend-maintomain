# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Simulate receive hints while exercising the original Worker receive thread."""

from __future__ import annotations

import time
from types import TracebackType
from typing import TYPE_CHECKING

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

from .pre_recv import LwdRecvFailure, create_hint_mq, make_recv_hint
from .types import LwdChannelType

if TYPE_CHECKING:
    from vllm_ascend.worker.lwd_worker import LwdWorker

    from .future import LwdCommFuture

_POLL_INTERVAL_S = 0.001


class LwdProbeEarlyRecv:
    def __init__(self, worker: LwdWorker) -> None:
        self.worker = worker
        self.hints: MessageQueue | None = None
        self.reports: MessageQueue | None = None
        self.preposted = 0
        self.verified_reports = 0

    def __enter__(self) -> LwdProbeEarlyRecv:
        try:
            self.hints = create_hint_mq()
            handle = self.worker.lwd_start_early_recv(self.hints.export_handle())
            if handle is None:
                raise RuntimeError("[LWD][wire-test] missing early-receive endpoint")
            self.reports = MessageQueue.create_from_handle(handle, 0)
            self.hints.wait_until_ready()
            self.reports.wait_until_ready()
            return self
        except Exception:
            self.close()
            raise

    def post(self, seqno: int, numel: int) -> LwdCommFuture:
        assert self.hints is not None
        rows, remainder = divmod(numel, self.worker._lwd_up_hidden_size)
        if remainder:
            raise ValueError("[LWD][wire-test] early receive requires whole hidden rows")
        timeout = self.worker.lwd_wire.timeout_s
        self.hints.enqueue(make_recv_hint(seqno, rows), timeout=timeout)
        deadline = time.monotonic() + timeout
        while True:
            if self.worker._early_recv_error is not None:
                raise RuntimeError("[LWD][wire-test] receive thread failed") from self.worker._early_recv_error
            with self.worker._early_recv_lock:
                future = self.worker._early_recv_futures.get(seqno)
            if future is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"[LWD][wire-test] hint not posted seqno={seqno}")
            time.sleep(_POLL_INTERVAL_S)
        # Wait for the hint thread to post first: a fallback receive would hide a broken hint path.
        consumed = self.worker.get_or_post_early_recv(seqno, rows, 0)
        if consumed is not future:
            raise RuntimeError("[LWD][wire-test] early-receive future was not reused")
        self.preposted += 1
        return consumed

    def verify_report(self, seqno: int) -> None:
        assert self.reports is not None
        report = self.reports.dequeue(timeout=self.worker.lwd_wire.timeout_s)
        if isinstance(report, LwdRecvFailure):
            raise RuntimeError(f"[LWD][wire-test] receive thread failed: {report.message}")
        if report != (LwdChannelType.UP, seqno):
            raise ValueError(f"[LWD][wire-test] unexpected completion report: {report!r}, expected seqno={seqno}")
        self.verified_reports += 1

    def close(self) -> None:
        try:
            self.worker.shutdown_early_recv()
        finally:
            if self.hints is not None:
                self.hints.shutdown()
            if self.reports is not None:
                self.reports.shutdown()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()
        if exc_type is None:
            if self.preposted != self.verified_reports:
                raise RuntimeError("[LWD][wire-test] missing early-receive completion reports")
            if self.worker._early_recv_futures or self.worker._early_recv_pending:
                raise RuntimeError("[LWD][wire-test] early-receive buffers were not released")
