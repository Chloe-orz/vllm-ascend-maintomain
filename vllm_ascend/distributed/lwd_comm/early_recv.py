# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Early receive queue lifecycle extracted from 982f776 LwdCloudWorker."""

from __future__ import annotations

import threading

import torch
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.logger import logger

from .future import LwdCommFuture
from .pre_recv import LwdRecvFailure, LwdRecvHint, create_report_mq, hint_fields
from .types import LwdChannelType

_LWD_EARLY_RECV_MAX_INFLIGHT = 3


class LwdEarlyRecv:
    def _init_lwd_early_recv_state(self) -> None:
        self._early_recv_futures: dict[int, LwdCommFuture] = {}
        self._early_recv_pending: dict[int, LwdRecvHint] = {}
        self._early_recv_reported: set[int] = set()
        self._early_recv_consumed: set[int] = set()
        self._last_consumed_seqno = -1
        self._early_recv_lock = threading.Lock()
        self._early_recv_stop = threading.Event()
        self._early_recv_error: Exception | None = None
        self._irecv_hint_mq: MessageQueue | None = None
        self._irecv_done_mq: MessageQueue | None = None
        self._lwd_early_recv_thread: threading.Thread | None = None

    def lwd_start_early_recv(self, handle: Handle) -> Handle | None:
        """Exchange queue handles once, before model requests can be scheduled."""
        if not self.lwd_wire.member:
            return None
        if self.lwd_wire.is_edge or not self._early_recv_enabled or self._lwd_early_recv_thread is not None:
            raise RuntimeError("[LWD] invalid early-receive initialization")
        self._irecv_hint_mq = MessageQueue.create_from_handle(handle, 0)
        self._irecv_done_mq = create_report_mq()
        self._lwd_early_recv_thread = threading.Thread(
            target=self._lwd_early_recv_loop,
            name="lwd-early-recv",
            daemon=True,
        )
        self._lwd_early_recv_thread.start()
        return self._irecv_done_mq.export_handle()

    def _lwd_early_recv_loop(self) -> None:
        hints, done = self._irecv_hint_mq, self._irecv_done_mq
        assert hints is not None and done is not None
        try:
            # NPU device selection is thread-local, including communication threads.
            hints.wait_until_ready()
            done.wait_until_ready()
            torch.npu.set_device(self.device)
            logger.info("[LWD] cloud receive thread ready: connection=%s", self.lwd_wire.key)
            while not self._early_recv_stop.is_set():
                try:
                    hint = hints.dequeue(timeout=0.0005)
                except TimeoutError:
                    pass
                else:
                    self.start_early_irecv(hint)
                self._report_irecv_completions()
                self._early_recv_stop.wait(0.002)
        except Exception as exc:
            if not self._early_recv_stop.is_set():
                self._early_recv_error = exc
                logger.exception("[LWD] cloud early receive failed")
                done.enqueue(LwdRecvFailure(str(exc)))

    def start_early_irecv(self, hint: LwdRecvHint) -> None:
        fields = hint_fields(hint)
        if fields is None:
            raise ValueError(f"[LWD] invalid receive hint: {hint!r}")
        seqno = fields[0]
        with self._early_recv_lock:
            if seqno <= self._last_consumed_seqno or seqno in self._early_recv_futures:
                return
            self._early_recv_pending[seqno] = hint
        self._early_recv_pump()

    def _early_recv_pump(self) -> None:
        with self._early_recv_lock:
            while self._early_recv_pending and len(self._early_recv_futures) < _LWD_EARLY_RECV_MAX_INFLIGHT:
                seqno, hint = next(iter(self._early_recv_pending.items()))
                del self._early_recv_pending[seqno]
                fields = hint_fields(hint)
                assert fields is not None
                self._early_recv_futures[seqno] = self.comm_service.submit_recv(
                    self._lwd_build_up_recv_request(*fields)
                )

    def _report_irecv_completions(self) -> None:
        assert self._irecv_done_mq is not None
        with self._early_recv_lock:
            pending = [
                (seqno, future)
                for seqno, future in self._early_recv_futures.items()
                if seqno not in self._early_recv_reported
            ]
        for seqno, future in pending:
            if not future.done():
                continue
            future.result()  # A failed receive must never advance the scheduler's watermark.
            self._irecv_done_mq.enqueue((LwdChannelType.UP, seqno))
            with self._early_recv_lock:
                self._early_recv_reported.add(seqno)
                self._release_early_recv(seqno)
        self._early_recv_pump()

    def _release_early_recv(self, seqno: int) -> None:
        # Both consumers must finish: the engine needs the completion report,
        # and the runner still needs the tensor after that report unblocks it.
        if seqno in self._early_recv_reported and seqno in self._early_recv_consumed:
            self._early_recv_futures.pop(seqno)
            self._early_recv_reported.remove(seqno)
            self._early_recv_consumed.remove(seqno)

    def get_or_post_early_recv(self, seqno: int, num_tokens: int, aux_rows: int) -> LwdCommFuture:
        if self._early_recv_error is not None:
            raise RuntimeError("[LWD] receive thread failed") from self._early_recv_error
        with self._early_recv_lock:
            future = self._early_recv_futures.get(seqno)
            if future is None:
                self._early_recv_pending.pop(seqno, None)
                future = self.comm_service.submit_recv(self._lwd_build_up_recv_request(seqno, num_tokens, aux_rows))
                self._early_recv_futures[seqno] = future
            self._last_consumed_seqno = max(self._last_consumed_seqno, seqno)
            self._early_recv_consumed.add(seqno)
            self._release_early_recv(seqno)
        self._early_recv_pump()
        return future

    def shutdown_early_recv(self) -> None:
        self._early_recv_stop.set()
        if self._irecv_hint_mq is not None:
            self._irecv_hint_mq.shutdown()
        if self._lwd_early_recv_thread is not None:
            self._lwd_early_recv_thread.join(timeout=5)
            if self._lwd_early_recv_thread.is_alive():
                raise RuntimeError("[LWD] early receive thread is still running; stop the worker process")
        if self._irecv_done_mq is not None:
            self._irecv_done_mq.shutdown()
