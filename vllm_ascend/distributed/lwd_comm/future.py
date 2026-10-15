# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Completion and buffer ownership for one held or submitted wire request."""

from __future__ import annotations

import threading
import time
from typing import Any

import torch
from vllm.logger import logger

from .types import LwdCommRequest, LwdCommResult, LwdCommStatus


class LwdCommFuture:
    def __init__(self, request: LwdCommRequest, *, producer_event: Any = None) -> None:
        self.request = request
        self.producer_event = producer_event
        self._done_event: Any = None
        self._tensor: torch.Tensor | None = None
        self._status = LwdCommStatus.PENDING
        self._error: Exception | None = None
        self._condition = threading.Condition()

    def _trace(self, stage: str) -> None:
        request = self.request
        logger.debug(
            "[LWD][comm] %s connection=%s channel=%s op=%s seqno=%s num_elements=%d",
            stage,
            request.connection_key,
            request.channel.value,
            request.op,
            request.seqno,
            request.num_elements,
        )

    def _bind(self, done_event: Any, tensor: torch.Tensor | None = None) -> None:
        with self._condition:
            self._done_event = done_event
            self._tensor = tensor
            self._condition.notify_all()

    def _fail(self, error: Exception) -> None:
        with self._condition:
            self._status = LwdCommStatus.ERROR
            self._error = error
            self._condition.notify_all()

    def done(self) -> bool:
        with self._condition:
            if self._status is not LwdCommStatus.PENDING:
                return True
            try:
                if self._done_event is not None and self._done_event.query():
                    self._status = LwdCommStatus.OK
                    self._trace("complete")
                    self._condition.notify_all()
            except Exception as exc:
                self._fail(exc)
            return self._status is not LwdCommStatus.PENDING

    def result(self) -> LwdCommResult:
        self.done()
        with self._condition:
            if self._error is not None:
                raise RuntimeError(
                    f"[LWD] {self.request.channel.value} seqno={self.request.seqno} failed"
                ) from self._error
            return LwdCommResult(status=self._status, tensor=self._tensor)

    def wait(self, timeout: float | None = None) -> LwdCommResult:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self.done():
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    stage = "queued" if self._done_event is None else "device_pending"
                    raise TimeoutError(
                        f"[LWD] {self.request.channel.value} seqno={self.request.seqno} timed out "
                        f"stage={stage} op={self.request.op} connection={self.request.connection_key} "
                        f"num_elements={self.request.num_elements}"
                    )
                self._condition.wait(min(0.01, remaining) if remaining is not None else 0.01)
            return self.result()

    def wait_for_comm(self) -> None:
        with self._condition:
            self.result()
            if self._done_event is None:
                raise RuntimeError("[LWD] cannot consume an unposted receive")
            torch.npu.current_stream().wait_event(self._done_event)

    def _finalize(self) -> None:
        with self._condition:
            # An error is not proof that DMA stopped. Retain its buffers until
            # process teardown; only successful completion permits reclamation.
            if self.done() and self._status is LwdCommStatus.OK:
                self.request.tensor = None
                self.producer_event = None
