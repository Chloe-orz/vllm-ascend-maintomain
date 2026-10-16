# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdCommService: Worker entry point of the PR #63 duplex comm layer.

Compute side only ever calls ``submit_send`` / ``submit_recv`` and
consumes ``LwdCommFuture`` notifications.  ``poll_completions`` is meant
to be invoked from an existing host loop head (worker busy loop);
it drives keepalive reclamation via lazy reap.

Ported (simplified) from the demo branch's ``lwd_comm/service.py``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from vllm.logger import logger

from .channel import LwdChannel
from .future import LwdCommFuture
from .types import LwdChannelType, LwdCommRequest

if TYPE_CHECKING:
    from .lwd_wire import LwdWire


class LwdCommService:
    """PR #63 submission and lifecycle, bound to the Worker YAML connection."""

    def __init__(self, wire: LwdWire) -> None:
        self._wire = wire
        # HCCL ordering is per physical (channel, direction): send and
        # recv have INDEPENDENT seqno streams (each peer's send order must
        # match the other side's recv-post order), so the FIFO key is
        # (channel, op) — merging them would make a process's sends and
        # recvs on one channel collide on a shared seqno counter.
        self._channels: dict[tuple[LwdChannelType, str], LwdChannel] = {}
        self._lock = threading.Lock()
        self._shutting_down = False

    # ------------------------------------------------------------------ #
    # Submission                                                          #
    # ------------------------------------------------------------------ #

    def submit_send(self, request: LwdCommRequest) -> LwdCommFuture:
        assert request.op == "send", "submit_send requires op='send'"
        return self._submit(request)

    def submit_recv(self, request: LwdCommRequest) -> LwdCommFuture:
        """Submit a recv.  The irecv is posted immediately — "early" is a
        property of when you submit, not a separate API."""
        assert request.op == "recv", "submit_recv requires op='recv'"
        return self._submit(request)

    # ------------------------------------------------------------------ #
    # Completion driving                                                  #
    # ------------------------------------------------------------------ #

    def poll_completions(self) -> int:
        """Reap every channel once.  Call from an existing loop head;
        costs one head-of-line ``event.query()`` per channel with
        pending traffic."""
        with self._lock:
            channels = list(self._channels.values())
        completed = 0
        for channel in channels:
            completed += len(channel.reap())
        return completed

    def skip_seqno(self, channel_type: LwdChannelType, seqno: int, op: str = "recv") -> None:
        """Mark a seqno as never-to-arrive (aborted request) on the given
        channel+direction FIFO.  Recv managers' drop path passes
        op="recv"; the sender-side abort glue passes op="send".  Both
        peers must skip the same seqnos."""
        with self._lock:
            channel = self._channels.get((channel_type, op))
        if channel is not None:
            channel.skip_seqno(seqno)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def shutdown(self, timeout: float | None = None) -> None:
        with self._lock:
            if self._shutting_down:
                return
            self._shutting_down = True
            channels = list(self._channels.values())
        try:
            for channel in channels:
                channel.shutdown(timeout=timeout)
        finally:
            with self._lock:
                self._channels.clear()
                self._shutting_down = False
        logger.info("[lwd-comm] service shut down")

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    def _submit(self, request: LwdCommRequest) -> LwdCommFuture:
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("lwd-comm service is shutting down")
            key = (request.channel, request.op)
            channel = self._channels.get(key)
            if channel is None:
                channel = LwdChannel(request.channel, request.op, self._wire)
                self._channels[key] = channel
        return channel.submit(request)
