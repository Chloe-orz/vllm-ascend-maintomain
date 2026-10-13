# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Sequence tensor operations before posting to tag-less HCCL P2P."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from .future import LwdCommFuture
from .types import LwdChannelType, LwdCommRequest

if TYPE_CHECKING:
    from .lwd_wire import LwdWire


class LwdChannel:
    def __init__(self, channel_type: LwdChannelType, op: str, wire: LwdWire) -> None:
        self.channel_type, self.op, self._wire = channel_type, op, wire
        self._lock = threading.RLock()
        self._next_seqno = 0
        self._held: dict[int, LwdCommFuture] = {}
        self._skipped: set[int] = set()
        self._pending: deque[LwdCommFuture] = deque()
        self._failed: list[LwdCommFuture] = []
        self._error: Exception | None = None
        self._closed = False

    def submit(self, request: LwdCommRequest) -> LwdCommFuture:
        with self._lock:
            if self._closed or self._error is not None:
                raise RuntimeError("[LWD] channel closed or failed") from self._error
            self._validate_request(request)
            seqno = request.seqno
            assert seqno is not None
            if seqno < self._next_seqno or seqno in self._held or seqno in self._skipped:
                raise ValueError(f"[LWD] duplicate or skipped seqno={seqno}")
            self.reap()
            producer_event = None
            if request.op == "send":
                assert request.tensor is not None
                owned = request.tensor.detach().clone(memory_format=torch.contiguous_format)
                request = replace(request, tensor=owned)
                # Capture the producing stream now, not when a held request is
                # eventually posted by a different host thread.
                producer_event = torch.npu.Event()
                producer_event.record()
            else:
                request = replace(request)
            future = LwdCommFuture(request, producer_event=producer_event)
            self._held[seqno] = future
            self._drain()
            return future

    def _validate_request(self, request: LwdCommRequest) -> None:
        if request.op != self.op or request.channel != self.channel_type:
            raise ValueError("[LWD] channel direction mismatch")
        if request.connection_key != self._wire.key:
            raise ValueError("[LWD] request belongs to another connection")
        expected = "send" if (request.channel == LwdChannelType.UP) == self._wire.is_edge else "recv"
        if request.op != expected or not self._wire.member:
            raise ValueError("[LWD] request direction or endpoint membership mismatch")
        if request.src_dst is not None and request.src_dst != 1 - self._wire.domain_rank:
            raise ValueError("[LWD] src_dst must be the peer's domain rank")
        if request.seqno is None or request.seqno < 0 or request.num_elements <= 0:
            raise ValueError("[LWD] nonnegative seqno and positive tensor size required")
        if request.op == "send":
            tensor = request.tensor
            if (
                tensor is None
                or tensor.numel() != request.num_elements
                or tensor.dtype != self._wire.dtype
                or tensor.device != self._wire.device
            ):
                raise ValueError("[LWD] send tensor size/dtype/device mismatch")
        elif request.tensor is not None:
            raise ValueError("[LWD] receive buffers are owned by the channel")

    def _drain(self) -> None:
        while True:
            if self._next_seqno in self._skipped:
                self._skipped.remove(self._next_seqno)
            elif self._next_seqno in self._held:
                future = self._held.pop(self._next_seqno)
                # Keep ownership even if submission fails after touching HCCL.
                self._pending.append(future)
                try:
                    self._execute(future)
                except Exception as exc:
                    self._error = exc
                    future._fail(exc)
                    for held in self._held.values():
                        held._fail(exc)
                    raise
            else:
                return
            self._next_seqno += 1

    def _execute(self, future: LwdCommFuture) -> None:
        self._wire.ensure_joined()
        stream = self._wire.get_channel_stream(self.channel_type)
        request = future.request
        with torch.npu.stream(stream):
            if request.op == "send":
                stream.wait_event(future.producer_event)
                self._wire.p2p_send(self.channel_type, request.tensor, request.src_dst)
                tensor = None
            else:
                tensor = torch.empty(request.num_elements, dtype=self._wire.dtype, device=self._wire.device)
                # Keep the receive buffer alive if the enqueue/event call fails.
                future._tensor = tensor
                self._wire.p2p_recv(self.channel_type, tensor, request.src_dst)
            event = torch.npu.Event()
            event.record()
        future._bind(event, tensor)

    def skip_seqno(self, seqno: int) -> None:
        with self._lock:
            if self._closed or self._error is not None:
                raise RuntimeError("[LWD] channel closed or failed") from self._error
            if seqno < 0:
                raise ValueError("[LWD] negative skipped seqno")
            if seqno < self._next_seqno:
                return
            held = self._held.pop(seqno, None)
            if held is not None:
                held._fail(RuntimeError(f"[LWD] seqno={seqno} aborted before posting"))
            # Only coordinated, never-posted sequence holes may be skipped.
            self._skipped.add(seqno)
            self._drain()

    def reap(self) -> list[LwdCommFuture]:
        with self._lock:
            completed = []
            while self._pending and self._pending[0].done():
                future = self._pending.popleft()
                try:
                    future.result()
                except RuntimeError as exc:
                    self._failed.append(future)
                    self._error = exc
                    raise
                future._finalize()
                completed.append(future)
            return completed

    def shutdown(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            self._closed = True
            for future in self._held.values():
                future._fail(RuntimeError("[LWD] channel closed before posting"))
            self._held.clear()
            if self._error is not None:
                raise RuntimeError("[LWD] failed channel retains device buffers") from self._error
            for future in self._pending:
                future.wait(None if deadline is None else max(0, deadline - time.monotonic()))
            self.reap()
