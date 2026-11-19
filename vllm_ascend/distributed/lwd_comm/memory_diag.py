# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Observe retained communication payloads and device memory without polling HCCL."""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import torch
from vllm.logger import logger

from vllm_ascend.distributed.lwd_comm.future import LwdCommFuture
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest, LwdCommStatus

_SAMPLE_INTERVAL_SECONDS = 2.0
_MIB = 2**20
_BF16_BYTES = 2
_INT64_BYTES = 8


@dataclass(frozen=True)
class _LwdTransferRecord:
    engine: int
    channel: LwdChannelType
    operation: Literal["send", "recv"]
    seqno: int | None
    payload_bytes: int
    started_at: float
    future: weakref.ReferenceType[LwdCommFuture] | None = None


@dataclass
class _LwdTransferBacklog:
    pending_count: int = 0
    pending_bytes: int = 0
    retained_count: int = 0
    retained_bytes: int = 0
    completed_retained_count: int = 0
    error_retained_count: int = 0
    oldest: _LwdTransferRecord | None = None
    latest: _LwdTransferRecord | None = None

    def observe(self, record: _LwdTransferRecord, status: LwdCommStatus, retained_bytes: int) -> None:
        if retained_bytes:
            self.retained_count += 1
            self.retained_bytes += retained_bytes
            self.completed_retained_count += status is LwdCommStatus.OK
            self.error_retained_count += status is LwdCommStatus.ERROR
        if status is LwdCommStatus.PENDING:
            self.pending_count += 1
            self.pending_bytes += retained_bytes
            if self.oldest is None or record.started_at < self.oldest.started_at:
                self.oldest = record
            if self.latest is None or record.started_at > self.latest.started_at:
                self.latest = record


@dataclass(frozen=True)
class _LwdMemorySample:
    sampled_at: float = 0.0
    free_bytes: int = -1
    total_bytes: int = -1
    allocated_bytes: int = -1
    reserved_bytes: int = -1
    active_bytes: int = -1
    inactive_split_bytes: int = -1
    peak_allocated_bytes: int = -1
    allocation_retries: int = -1
    ooms: int = -1


@dataclass(frozen=True)
class LwdReceiveMemorySample:
    engine: int
    endpoint: bool
    host_stage: bool
    recv_limit: int
    inflight: int = -1
    queued: int = -1
    inflight_main_bytes: int = -1
    oldest_seqno: int | None = None
    recv_lock_busy: bool = False
    slot_limit: int = 0
    allocated_slots: int = 0
    busy_slots: int = 0
    device_pool_bytes: int = 0
    host_bytes: int = 0
    slot_owners: tuple[int, ...] = ()
    stage_lock_busy: bool = False


class LwdMemoryDiagnostics:
    """Keep CPU-only communication metadata and sample memory on a separate thread."""

    def __init__(
        self,
        role: str,
        rank: int,
        device: torch.device,
        receive_snapshot: Callable[[], LwdReceiveMemorySample] | None = None,
    ) -> None:
        self.role = role
        self.rank = rank
        self.device = device
        self._receive_snapshot = receive_snapshot
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._submitting: dict[int, _LwdTransferRecord] = {}
        # Weak references must not extend the payload lifetime being measured.
        self._sends: dict[int, _LwdTransferRecord] = {}
        self._receives: dict[int, _LwdTransferRecord] = {}
        self._next_token = 0
        self._memory = _LwdMemorySample()
        self._query_started_at = 0.0
        self._query_error = "none"

    def start(self) -> None:
        for name, target in (("report", self._report_loop), ("device", self._memory_loop)):
            threading.Thread(target=target, name=f"lwd-memory-{name}", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def begin_transfer(self, request: LwdCommRequest) -> int:
        if request.op == "send":
            payload_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (request.tensor, request.aux_tensor)
                if tensor is not None
            )
        else:
            payload_bytes = request.num_elements * _BF16_BYTES
            payload_bytes += request.aux_num_elements * getattr(request.aux_dtype, "itemsize", _INT64_BYTES)
        record = _LwdTransferRecord(
            engine=request.engine_index,
            channel=request.channel,
            operation=request.op,
            seqno=request.seqno,
            payload_bytes=payload_bytes,
            started_at=time.monotonic(),
        )
        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._submitting[token] = record
        return token

    def finish_transfer(self, token: int, future: LwdCommFuture) -> None:
        with self._lock:
            record = self._submitting.pop(token)
            records = self._sends if record.operation == "send" else self._receives
            previous = records.get(id(future))
            if previous is None or previous.future is None or previous.future() is not future:
                records[id(future)] = replace(record, future=weakref.ref(future))

    def fail_transfer(self, token: int, error: Exception) -> None:
        with self._lock:
            record = self._submitting.pop(token)
        logger.warning(
            "[LWD][mem] %s_submit_error role=%s rank=%d device=%s channel=%s "
            "engine=%d seqno=%s payload_mib=%.3f error=%r wall=%.3f ts=%.6f",
            record.operation,
            self.role,
            self.rank,
            self.device,
            record.channel.value,
            record.engine,
            record.seqno,
            record.payload_bytes / _MIB,
            error,
            time.time(),
            time.monotonic(),
        )

    def _backlogs(self, operation: Literal["send", "recv"] = "send") -> dict[LwdChannelType, _LwdTransferBacklog]:
        with self._lock:
            tracked = self._sends if operation == "send" else self._receives
            records = list(tracked.items())
        backlogs = {channel: _LwdTransferBacklog() for channel in LwdChannelType}
        released = []
        for key, record in records:
            future = record.future() if record.future is not None else None
            if future is None:
                released.append((key, record))
                continue
            status, retained_bytes = future.retained_payload_state()
            backlogs[record.channel].observe(record, status, retained_bytes)
            if not retained_bytes and status is not LwdCommStatus.PENDING:
                released.append((key, record))
        with self._lock:
            for key, record in released:
                if tracked.get(key) is record:
                    del tracked[key]
        return backlogs

    def _report_loop(self) -> None:
        sample = 0
        while not self._stop.is_set():
            try:
                self._report(sample)
            except Exception:
                logger.warning("[LWD][mem] report_error role=%s rank=%d", self.role, self.rank, exc_info=True)
            sample += 1
            self._stop.wait(_SAMPLE_INTERVAL_SECONDS)

    def _report(self, sample: int) -> None:
        with self._lock:
            submitting = list(self._submitting.values())
            memory = self._memory
            query_started = self._query_started_at
            query_error = self._query_error
        now, wall = time.monotonic(), time.time()
        for operation in ("send", "recv"):
            self._report_transfers(operation, sample, submitting)
        logger.info(
            "[LWD][mem] device role=%s rank=%d device=%s sample=%d "
            "hbm_free_mib=%.3f hbm_total_mib=%.3f allocated_mib=%.3f "
            "reserved_mib=%.3f active_mib=%.3f inactive_split_mib=%.3f "
            "peak_allocated_mib=%.3f allocation_retries=%d ooms=%d "
            "memory_age_s=%.3f query_running_s=%.3f query_error=%s wall=%.3f ts=%.6f",
            self.role,
            self.rank,
            self.device,
            sample,
            memory.free_bytes / _MIB if memory.free_bytes >= 0 else -1.0,
            memory.total_bytes / _MIB if memory.total_bytes >= 0 else -1.0,
            memory.allocated_bytes / _MIB if memory.allocated_bytes >= 0 else -1.0,
            memory.reserved_bytes / _MIB if memory.reserved_bytes >= 0 else -1.0,
            memory.active_bytes / _MIB if memory.active_bytes >= 0 else -1.0,
            memory.inactive_split_bytes / _MIB if memory.inactive_split_bytes >= 0 else -1.0,
            memory.peak_allocated_bytes / _MIB if memory.peak_allocated_bytes >= 0 else -1.0,
            memory.allocation_retries,
            memory.ooms,
            now - memory.sampled_at if memory.sampled_at else -1.0,
            now - query_started if query_started else 0.0,
            query_error,
            wall,
            now,
        )
        if self._receive_snapshot is not None:
            self._report_receive(sample, now, wall)

    def _report_transfers(
        self, operation: Literal["send", "recv"], sample: int, submitting: list[_LwdTransferRecord]
    ) -> None:
        now, wall = time.monotonic(), time.time()
        for channel, backlog in self._backlogs(operation).items():
            active = [record for record in submitting if record.channel is channel and record.operation == operation]
            oldest_submit = min(active, key=lambda record: record.started_at, default=None)
            oldest, latest = backlog.oldest, backlog.latest
            logger.info(
                "[LWD][mem] %s role=%s rank=%d device=%s sample=%d channel=%s "
                "pending=%d pending_mib=%.3f retained=%d retained_mib=%.3f "
                "completed_retained=%d error_retained=%d oldest_engine=%s oldest_seqno=%s "
                "latest_seqno=%s oldest_age_s=%.3f submitting=%d submitting_mib=%.3f "
                "submit_engine=%s submit_seqno=%s submit_age_s=%.3f wall=%.3f ts=%.6f",
                "send" if operation == "send" else "recv_buffer",
                self.role,
                self.rank,
                self.device,
                sample,
                channel.value,
                backlog.pending_count,
                backlog.pending_bytes / _MIB,
                backlog.retained_count,
                backlog.retained_bytes / _MIB,
                backlog.completed_retained_count,
                backlog.error_retained_count,
                oldest.engine if oldest else None,
                oldest.seqno if oldest else None,
                latest.seqno if latest else None,
                now - oldest.started_at if oldest else 0.0,
                len(active),
                sum(record.payload_bytes for record in active) / _MIB,
                oldest_submit.engine if oldest_submit else None,
                oldest_submit.seqno if oldest_submit else None,
                now - oldest_submit.started_at if oldest_submit else 0.0,
                wall,
                now,
            )

    def _report_receive(self, sample: int, now: float, wall: float) -> None:
        assert self._receive_snapshot is not None
        state = self._receive_snapshot()
        logger.info(
            "[LWD][mem] recv role=%s rank=%d device=%s sample=%d channel=lwd_up "
            "engine=%d endpoint=%s host_stage=%s recv_limit=%d inflight=%d queued=%d "
            "inflight_main_mib=%.3f oldest_seqno=%s recv_lock_busy=%s "
            "slot_limit=%d allocated_slots=%d busy_slots=%d device_pool_mib=%.3f "
            "slot_owners=%s host_mib=%.3f stage_lock_busy=%s wall=%.3f ts=%.6f",
            self.role,
            self.rank,
            self.device,
            sample,
            state.engine,
            state.endpoint,
            state.host_stage,
            state.recv_limit,
            state.inflight,
            state.queued,
            state.inflight_main_bytes / _MIB if state.inflight_main_bytes >= 0 else -1.0,
            state.oldest_seqno,
            state.recv_lock_busy,
            state.slot_limit,
            state.allocated_slots,
            state.busy_slots,
            state.device_pool_bytes / _MIB if state.device_pool_bytes >= 0 else -1.0,
            state.slot_owners,
            state.host_bytes / _MIB if state.host_bytes >= 0 else -1.0,
            state.stage_lock_busy,
            wall,
            now,
        )

    def _memory_loop(self) -> None:
        # Keep reporting CPU metadata even if a Runtime memory query stalls.
        try:
            torch.npu.set_device(self.device)
        except Exception:
            with self._lock:
                self._query_error = "set_device_failed"
            logger.warning("[LWD][mem] device_init_error role=%s rank=%d", self.role, self.rank, exc_info=True)
            return
        while not self._stop.is_set():
            with self._lock:
                self._query_started_at = time.monotonic()
            try:
                memory = self._read_memory()
            except Exception as error:
                with self._lock:
                    self._query_error = repr(error)
                logger.warning("[LWD][mem] device_query_error role=%s rank=%d error=%r", self.role, self.rank, error)
            else:
                with self._lock:
                    self._memory = memory
                    self._query_error = "none"
            finally:
                with self._lock:
                    self._query_started_at = 0.0
            self._stop.wait(_SAMPLE_INTERVAL_SECONDS)

    def _read_memory(self) -> _LwdMemorySample:
        free_bytes, total_bytes = torch.npu.mem_get_info(self.device)
        stats = torch.npu.memory_stats(self.device)
        return _LwdMemorySample(
            sampled_at=time.monotonic(),
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            allocated_bytes=stats.get("allocated_bytes.all.current", -1),
            reserved_bytes=stats.get("reserved_bytes.all.current", -1),
            active_bytes=stats.get("active_bytes.all.current", -1),
            inactive_split_bytes=stats.get("inactive_split_bytes.all.current", -1),
            peak_allocated_bytes=stats.get("allocated_bytes.all.peak", -1),
            allocation_retries=stats.get("num_alloc_retries", -1),
            ooms=stats.get("num_ooms", -1),
        )
