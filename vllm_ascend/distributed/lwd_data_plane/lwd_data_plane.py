# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdDataPlane: singleton entry point of the LWD data plane.

Compute side only ever calls ``submit_send`` / ``submit_recv`` and
consumes ``LwdTransferFuture`` notifications.  ``poll_completions`` is meant
to be invoked from an existing host loop head (worker busy loop);
it drives keepalive reclamation via lazy reap.

Ported (simplified) from the demo branch's ``lwd_comm/service.py``.
"""

from __future__ import annotations

import threading

from vllm.logger import logger

from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_queue import LwdTransferQueue
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType, LwdTransferRequest


class LwdDataPlane:
    """Process-wide singleton.  All public methods are thread-safe."""

    _instance: LwdDataPlane | None = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> LwdDataPlane:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        # HCCL ordering is per physical (channel, direction): send and
        # recv have INDEPENDENT seqno streams (each peer's send order must
        # match the other side's recv-post order), so the FIFO key is
        # (channel, op) — merging them would make a process's sends and
        # recvs on one channel collide on a shared seqno counter.
        self._queues: dict[tuple[LwdChannelType, str], LwdTransferQueue] = {}
        self._lock = threading.Lock()
        self._shutting_down = False

    # ------------------------------------------------------------------ #
    # Submission                                                          #
    # ------------------------------------------------------------------ #

    def submit_send(self, request: LwdTransferRequest) -> LwdTransferFuture:
        assert request.op == "send", "submit_send requires op='send'"
        return self._submit(request)

    def submit_recv(self, request: LwdTransferRequest) -> LwdTransferFuture:
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
        # TODO(lwd): confirm whether the edge should reap too. On the mm branch
        # only the cloud worker calls this; submit() reaps its own queue only,
        # so an edge UP send snapshot (one prefill batch) stays alive until the
        # next UP send.
        with self._lock:
            queues = list(self._queues.values())
        completed = 0
        for queue in queues:
            completed += len(queue.reap())
        return completed

    def skip_seqno(self, channel_type: LwdChannelType, seqno: int, op: str = "recv") -> None:
        """Mark a seqno as never-to-arrive (aborted request) on the given
        channel+direction FIFO.  Recv managers' drop path passes
        op="recv"; the sender-side abort glue passes op="send".  Both
        peers must skip the same seqnos."""
        with self._lock:
            queue = self._queues.get((channel_type, op))
        if queue is not None:
            queue.skip_seqno(seqno)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def shutdown(self, timeout: float | None = None) -> None:
        with self._lock:
            if self._shutting_down:
                return
            self._shutting_down = True
            queues = list(self._queues.values())
        try:
            for queue in queues:
                queue.shutdown(timeout=timeout)
        finally:
            with self._lock:
                self._queues.clear()
                self._shutting_down = False
        logger.info("[lwd-comm] service shut down")

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    def _submit(self, request: LwdTransferRequest) -> LwdTransferFuture:
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("lwd-comm service is shutting down")
            key = (request.channel, request.op)
            queue = self._queues.get(key)
            if queue is None:
                queue = LwdTransferQueue(request.channel, request.op)
                self._queues[key] = queue
        return queue.submit(request)


def lwd_get_data_plane() -> LwdDataPlane:
    return LwdDataPlane.instance()
