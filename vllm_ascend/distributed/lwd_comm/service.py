# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Worker-owned transport service; no process-wide connection singleton."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from .channel import LwdChannel
from .future import LwdCommFuture
from .types import LwdChannelType, LwdCommRequest

if TYPE_CHECKING:
    from .lwd_wire import LwdWire


class LwdCommService:
    def __init__(self, wire: LwdWire) -> None:
        self._wire = wire
        self._channels: dict[tuple[LwdChannelType, str], LwdChannel] = {}
        self._lock = threading.Lock()
        self._closed = False

    def submit_send(self, request: LwdCommRequest) -> LwdCommFuture:
        if request.op != "send":
            raise ValueError("[LWD] submit_send requires a send request")
        return self._channel(request.channel, request.op).submit(request)

    def submit_recv(self, request: LwdCommRequest) -> LwdCommFuture:
        if request.op != "recv":
            raise ValueError("[LWD] submit_recv requires a receive request")
        return self._channel(request.channel, request.op).submit(request)

    def _channel(self, channel: LwdChannelType, op: str) -> LwdChannel:
        with self._lock:
            if self._closed:
                raise RuntimeError("[LWD] communication service closed")
            expected = "send" if (channel == LwdChannelType.UP) == self._wire.is_edge else "recv"
            if not self._wire.member or op != expected:
                raise ValueError("[LWD] channel direction or endpoint membership mismatch")
            key = (channel, op)
            if key not in self._channels:
                self._channels[key] = LwdChannel(channel, op, self._wire)
            return self._channels[key]

    def skip_seqno(self, channel_type: LwdChannelType, seqno: int, op: str = "recv") -> None:
        # Record holes even if no request has created this channel yet.
        self._channel(channel_type, op).skip_seqno(seqno)

    def poll_completions(self) -> int:
        with self._lock:
            channels = tuple(self._channels.values())
        return sum(len(channel.reap()) for channel in channels)

    def shutdown(self, timeout: float | None = None) -> None:
        with self._lock:
            self._closed = True
            channels = tuple(self._channels.values())
        deadline = time.monotonic() + (self._wire.timeout_s if timeout is None else timeout)
        errors = []
        for channel in channels:
            try:
                channel.shutdown(max(0, deadline - time.monotonic()))
            except Exception as exc:
                errors.append(exc)
        if errors:
            # Outstanding DMA must retain buffers and its PG until process exit.
            raise RuntimeError("[LWD] transport could not drain; stop the worker process") from errors[0]
        self._wire.shutdown()
