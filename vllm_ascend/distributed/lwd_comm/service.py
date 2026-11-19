# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdCommService: singleton entry point of the duplex comm layer.

Compute side only ever calls ``submit_send`` / ``submit_recv`` and
consumes ``LwdCommFuture`` notifications.  ``poll_completions`` is meant
to be invoked from an existing host loop head (worker busy loop);
it drives keepalive reclamation via lazy reap.

Ported (simplified) from the demo branch's ``lwd_comm/service.py``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import torch
from vllm.logger import logger

from vllm_ascend.distributed.lwd_comm.channel import LwdChannel
from vllm_ascend.distributed.lwd_comm.future import LwdCommFuture
from vllm_ascend.distributed.lwd_comm.memory_diag import LwdMemoryDiagnostics, LwdReceiveMemorySample
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest


class LwdCommService:
    """Process-wide singleton.  All public methods are thread-safe."""

    _instance: "LwdCommService | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "LwdCommService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        # HCCL ordering is per physical (engine, channel, direction): send
        # and recv have INDEPENDENT seqno streams (each peer's send order
        # must match the other side's recv-post order), and multi-DP gives
        # each cloud engine its own channel pair — so the FIFO key is
        # (engine_index, channel, op); merging any of the three would make
        # seqno streams collide across engines/directions.
        self._channels: dict[tuple[int, LwdChannelType, str], LwdChannel] = {}
        self._lock = threading.Lock()
        self._shutting_down = False
        self._memory_diag: LwdMemoryDiagnostics | None = None

    def start_memory_diagnostics(
        self, role: str, rank: int, device: torch.device,
        receive_snapshot: Callable[[], LwdReceiveMemorySample] | None = None,
    ) -> None:
        """Start process-local sampling after distributed device initialization."""
        with self._lock:
            if self._memory_diag is not None:
                return
            self._memory_diag = LwdMemoryDiagnostics(role, rank, device, receive_snapshot)
        self._memory_diag.start()

    def activate_memory_diagnostics(self) -> None:
        """Activate all-rank sampling when request execution first begins."""
        if self._memory_diag is not None:
            self._memory_diag.mark_traffic_started()

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

    def skip_seqno(
        self,
        channel_type: LwdChannelType,
        seqno: int,
        op: str = "recv",
        engine_index: int = 0,
    ) -> None:
        """Mark a seqno as never-to-arrive (aborted request) on the given
        engine+channel+direction FIFO.  Recv managers' drop path passes
        op="recv"; the sender-side abort glue passes op="send".  Both
        peers must skip the same seqnos."""
        with self._lock:
            channel = self._channels.get((engine_index, channel_type, op))
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
            memory_diag = self._memory_diag
            self._memory_diag = None
        if memory_diag is not None:
            memory_diag.stop()
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
            key = (request.engine_index, request.channel, request.op)
            channel = self._channels.get(key)
            if channel is None:
                channel = LwdChannel(
                    request.channel, request.op, request.engine_index
                )
                self._channels[key] = channel
        memory_diag = self._memory_diag
        if memory_diag is None:
            return channel.submit(request)
        token = memory_diag.begin_transfer(request)
        try:
            future = channel.submit(request)
        except Exception as error:
            memory_diag.fail_transfer(token, error)
            raise
        memory_diag.finish_transfer(token, future)
        return future


def get_lwd_comm_service() -> LwdCommService:
    return LwdCommService.instance()
