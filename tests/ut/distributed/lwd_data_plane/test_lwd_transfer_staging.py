# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only tests for segmented host staging in the LWD data plane."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.lwd_data_plane import lwd_channels
from vllm_ascend.distributed.lwd_data_plane import lwd_transfer_queue as queue_module
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_queue import LwdTransferQueue
from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
    LwdChannelType,
    LwdTransferRequest,
    LwdTransferStatus,
)


class _FakeEvent:
    def __init__(self, runtime: _NpuRuntime) -> None:
        self.runtime = runtime
        self.ready = False
        self.recorded_stream: _FakeStream | None = None

    def record(self) -> None:
        self.recorded_stream = self.runtime.current_stream
        self.runtime.operations.append(("record", self))

    def query(self) -> bool:
        return self.ready


class _FakeStream:
    def __init__(self, runtime: _NpuRuntime, name: str) -> None:
        self.runtime = runtime
        self.name = name
        self.waits: list[_FakeEvent] = []

    def wait_event(self, event: _FakeEvent) -> None:
        self.waits.append(event)
        self.runtime.operations.append(("copy_wait" if self.name.startswith("copy") else "recv_wait", event))

    def wait_stream(self, stream: _FakeStream) -> None:
        self.runtime.operations.append(("stream_wait", stream))


class _NpuRuntime:
    def __init__(self) -> None:
        self.operations: list[tuple[Any, ...]] = []
        self.events: list[_FakeEvent] = []
        self.copy_streams: list[_FakeStream] = []
        self.channel_stream = _FakeStream(self, "channel")
        self.current_stream = _FakeStream(self, "producer")

    def create_event(self) -> _FakeEvent:
        event = _FakeEvent(self)
        self.events.append(event)
        return event

    def create_stream(self) -> _FakeStream:
        stream = _FakeStream(self, f"copy-{len(self.copy_streams)}")
        self.copy_streams.append(stream)
        return stream

    @contextmanager
    def activate_stream(self, stream: _FakeStream) -> Iterator[None]:
        previous = self.current_stream
        self.current_stream = stream
        try:
            yield
        finally:
            self.current_stream = previous

    def receive(self, tensor: torch.Tensor, **kwargs: Any) -> Any:
        self.operations.append(("recv", tensor))
        handle = MagicMock()
        handle.wait.side_effect = lambda: self.operations.append(("hccl_wait", tensor))
        return handle


@pytest.fixture
def npu_runtime(monkeypatch: pytest.MonkeyPatch) -> _NpuRuntime:
    runtime = _NpuRuntime()
    monkeypatch.setattr(queue_module.torch.npu, "Event", runtime.create_event)
    monkeypatch.setattr(queue_module.torch.npu, "Stream", runtime.create_stream)
    monkeypatch.setattr(queue_module.torch.npu, "stream", runtime.activate_stream)
    monkeypatch.setattr(queue_module.torch.npu, "current_stream", lambda: runtime.current_stream)
    monkeypatch.setattr(queue_module, "_lwd_scramble_seed_for", lambda request: None)
    monkeypatch.setattr(lwd_channels, "lwd_get_channel_stream", lambda channel, dp_idx: runtime.channel_stream)
    monkeypatch.setattr(lwd_channels, "lwd_get_channel_device_group", lambda channel, dp_idx: object())
    monkeypatch.setattr(lwd_channels, "lwd_get_channel_peer", lambda channel, dp_idx: 1)
    monkeypatch.setattr(queue_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(queue_module.dist, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(queue_module.dist, "irecv", runtime.receive)
    return runtime


def _request(num_elements: int = 6, **overrides: Any) -> LwdTransferRequest:
    values: dict[str, Any] = {
        "channel": LwdChannelType.DOWN,
        "op": "recv",
        "num_elements": num_elements,
    }
    values.update(overrides)
    return LwdTransferRequest(**values)


def _queue() -> LwdTransferQueue:
    return LwdTransferQueue(dp_idx=0, channel_type=LwdChannelType.DOWN, op="recv")


def _destination(runtime: _NpuRuntime, numel: int) -> MagicMock:
    dst = MagicMock()
    dst.numel.return_value = numel

    def copy(source: torch.Tensor, non_blocking: bool) -> None:
        runtime.operations.append(("copy", dst, source, non_blocking))

    dst.copy_.side_effect = copy
    return dst


def test_external_slot_and_segmented_d2h_order(npu_runtime: _NpuRuntime) -> None:
    slot = torch.arange(6)
    first_dst = _destination(npu_runtime, 2)
    second_dst = _destination(npu_runtime, 2)
    request = _request(
        recv_buffer=slot,
        stage_segments=[(first_dst, 1, 2), (second_dst, 4, 2)],
    )

    future = _queue().submit(request)

    assert future.result().tensor is slot
    copies = [operation for operation in npu_runtime.operations if operation[0] == "copy"]
    assert [operation[2].tolist() for operation in copies] == [[1, 2], [4, 5]]
    assert all(operation[3] is True for operation in copies)
    assert [operation[0] for operation in npu_runtime.operations] == [
        "recv",
        "hccl_wait",
        "record",
        "copy_wait",
        "copy",
        "copy",
        "record",
    ]
    assert future._done_event is not None
    assert future._staged_event is not None
    assert npu_runtime.copy_streams[0].waits == [future._done_event]
    assert future._done_event.recorded_stream is npu_runtime.channel_stream
    assert future._staged_event.recorded_stream is npu_runtime.copy_streams[0]


def test_d2h_waits_for_post_descramble_event(
    npu_runtime: _NpuRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.distributed.lwd_data_plane import lwd_payload_scrambler as scrambler_module

    scrambler = MagicMock()
    scrambler.descramble.side_effect = lambda tensor, seed: npu_runtime.operations.append(("descramble", seed))
    monkeypatch.setattr(queue_module, "_lwd_scramble_seed_for", lambda request: 17)
    monkeypatch.setattr(scrambler_module, "lwd_get_payload_scrambler", lambda: scrambler)

    future = _queue().submit(
        _request(
            recv_buffer=torch.arange(6),
            stage_segments=[(_destination(npu_runtime, 6), 0, 6)],
        )
    )

    assert [operation[0] for operation in npu_runtime.operations] == [
        "recv",
        "hccl_wait",
        "record",
        "descramble",
        "record",
        "copy_wait",
        "copy",
        "record",
    ]
    assert npu_runtime.copy_streams[0].waits == [future._done_event]
    assert future._done_event is npu_runtime.events[1]


def test_next_recv_waits_for_hccl_not_staging(npu_runtime: _NpuRuntime) -> None:
    queue = _queue()
    first_slot = torch.arange(6)
    first = queue.submit(
        _request(
            recv_buffer=first_slot,
            stage_segments=[(_destination(npu_runtime, 6), 0, 6)],
        )
    )
    assert first._staged_event is not None
    assert not first.staged_done()

    queue.submit(_request(recv_buffer=torch.arange(6)))

    assert npu_runtime.channel_stream.waits == [first._done_event]
    assert first._staged_event not in npu_runtime.channel_stream.waits


def test_deferred_future_binds_staged_event(npu_runtime: _NpuRuntime) -> None:
    queue = _queue()
    deferred_slot = torch.arange(6)
    deferred = queue.submit(
        _request(
            seqno=1,
            recv_buffer=deferred_slot,
            stage_segments=[(_destination(npu_runtime, 6), 0, 6)],
        )
    )
    assert deferred._done_event is None
    assert deferred._staged_event is None

    queue.submit(_request(seqno=0, recv_buffer=torch.arange(6)))

    assert deferred._done_event is not None
    assert deferred._staged_event is not None
    assert deferred.result().tensor is deferred_slot
    assert not deferred.staged_done()
    deferred._staged_event.ready = True
    assert deferred.staged_done()


def test_release_device_tensor_only_applies_to_staged_recv(npu_runtime: _NpuRuntime) -> None:
    queue = _queue()
    staged_slot = torch.arange(6)
    staged = queue.submit(
        _request(
            recv_buffer=staged_slot,
            stage_segments=[(_destination(npu_runtime, 6), 0, 6)],
        )
    )
    normal_slot = torch.arange(6)
    normal = queue.submit(_request(recv_buffer=normal_slot))

    staged.release_device_tensor()
    normal.release_device_tensor()

    assert staged.result().tensor is None
    assert normal.result().tensor is normal_slot
    assert normal.staged_done()


def test_staging_error_is_reported_by_result(npu_runtime: _NpuRuntime) -> None:
    staged_event = npu_runtime.create_event()
    staged_event.query = MagicMock(side_effect=RuntimeError("D2H failed"))
    future = LwdTransferFuture(
        request=_request(),
        handles=[],
        done_event=npu_runtime.create_event(),
        tensor=torch.arange(6),
        staged_event=staged_event,
    )
    future._status = LwdTransferStatus.OK

    assert future.staged_done()
    with pytest.raises(RuntimeError, match="lwd-comm op failed") as exc_info:
        future.result()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "D2H failed"


def test_recv_buffer_numel_is_strict(npu_runtime: _NpuRuntime) -> None:
    with pytest.raises(RuntimeError, match="recv_buffer numel 5 != requested num_elements 6"):
        _queue().submit(_request(recv_buffer=torch.arange(5)))
    assert not [operation for operation in npu_runtime.operations if operation[0] == "recv"]


@pytest.mark.parametrize(
    ("src_offset", "numel", "dst_numel"),
    [(-1, 2, 2), (0, -1, 0), (5, 2, 2), (0, 2, 3)],
)
def test_invalid_stage_geometry_fails_before_receive(
    npu_runtime: _NpuRuntime,
    src_offset: int,
    numel: int,
    dst_numel: int,
) -> None:
    dst = _destination(npu_runtime, dst_numel)
    request = _request(recv_buffer=torch.arange(6), stage_segments=[(dst, src_offset, numel)])

    with pytest.raises(ValueError, match="stage"):
        _queue().submit(request)
    assert not [operation for operation in npu_runtime.operations if operation[0] == "recv"]


def test_normal_recv_still_allocates_device_buffer(npu_runtime: _NpuRuntime) -> None:
    allocated = torch.arange(6)
    with patch.object(queue_module.torch, "empty", return_value=allocated) as empty:
        future = _queue().submit(_request())

    empty.assert_called_once_with(6, dtype=torch.bfloat16, device="npu")
    assert future.result().tensor is allocated
    assert future._staged_event is None
    assert not npu_runtime.copy_streams
