# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host ordering tests with CPU tensors and simulated device completions."""

import importlib
import importlib.util
import sys
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/lwd_comm"
    spec = importlib.util.spec_from_file_location("lwd_transport_test", path / "__init__.py")
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, package)
    spec.loader.exec_module(package)
    modules = {
        name: importlib.import_module(spec.name + "." + name)
        for name in ("types", "future", "channel", "service", "lwd_wire")
    }
    # These completions model ordering/ownership only; they do not validate HCCL.
    events, waits = [], []

    class Event:
        def __init__(self) -> None:
            self.complete = True
            events.append(self)

        def record(self) -> None:
            pass

        def query(self) -> bool:
            return self.complete

    stream = SimpleNamespace(wait_event=waits.append, wait_stream=lambda other: None)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Event=Event,
            Stream=lambda: stream,
            current_stream=lambda: stream,
            stream=lambda value: nullcontext(),
            set_device=lambda device: None,
        ),
        raising=False,
    )
    calls = []
    wire = SimpleNamespace(
        key=(0, 0, 0),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        member=True,
        is_edge=True,
        timeout_s=0.05,
        ensure_joined=lambda: None,
        get_channel_stream=lambda channel: stream,
        p2p_send=lambda channel, tensor, peer: calls.append((channel, tensor.clone())),
        p2p_recv=lambda channel, tensor, peer: tensor.fill_(9),
        shutdown=lambda: None,
    )
    wire.get_lwd_channel_stream = wire.get_channel_stream
    wire.get_lwd_channel_peer = lambda channel: 1
    wire.get_lwd_channel_device_group = lambda channel: (wire, channel)

    def post(op: str, tensor: torch.Tensor, group: tuple[Any, Any], peer: int) -> Mock:
        getattr(group[0], "p2p_" + op)(group[1], tensor, peer)
        return Mock()

    monkeypatch.setattr(torch.distributed, "isend", lambda tensor, dst, group: post("send", tensor, group, dst))
    monkeypatch.setattr(torch.distributed, "irecv", lambda tensor, src, group: post("recv", tensor, group, src))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    group = SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        ranks=[0],
        cpu_group=object(),
        device_group=object(),
        broadcast_object=Mock(side_effect=lambda value, src: value),
    )
    parallel = ModuleType("vllm.distributed.parallel_state")
    parallel.get_tp_group = lambda: group
    parallel.get_lwd_bootstrap_world = lambda: group
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)
    return SimpleNamespace(**modules, wire=wire, calls=calls, events=events, waits=waits, group=group)


@pytest.fixture
def worker_transport(transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    t = transport
    mq = ModuleType("vllm.distributed.device_communicators.shm_broadcast")
    mq.Handle, mq.MessageQueue = object, Mock()
    monkeypatch.setitem(sys.modules, mq.__name__, mq)
    early = importlib.import_module(t.types.__package__ + ".early_recv")
    for name in ("types", "future", "service", "lwd_wire"):
        monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.lwd_comm." + name, getattr(t, name))
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.lwd_comm.early_recv", early)
    base = ModuleType("vllm_ascend.worker.worker")
    base.NPUWorker = type(
        "NPUWorker",
        (),
        {"init_device": Mock(), "shutdown": Mock(), "determine_available_memory": Mock(return_value=123)},
    )
    monkeypatch.setitem(sys.modules, base.__name__, base)
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/lwd_worker.py"
    spec = importlib.util.spec_from_file_location("lwd_worker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.LwdWorker()
    worker.lwd_wire, worker.device = t.wire, t.wire.device
    worker.comm_service = t.service.LwdCommService(t.wire)
    worker.model_config = SimpleNamespace(dtype=t.wire.dtype)
    worker._lwd_up_hidden_size = 8
    worker._early_recv_enabled = True
    worker._init_lwd_early_recv_state()
    # CPU CI exercises the real Worker and service methods, replacing only NPU operations.
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    return SimpleNamespace(**vars(t), worker=worker, module=module, early=early, base=base)


def send(t: SimpleNamespace, seqno: int, value: float = 1) -> Any:
    return t.types.LwdCommRequest(
        t.types.LwdChannelType.UP,
        "send",
        2,
        torch.full((2,), value, dtype=torch.bfloat16),
        seqno=seqno,
        connection_key=t.wire.key,
    )


def test_out_of_order_send_preserves_source_snapshot_and_fifo(transport: SimpleNamespace) -> None:
    t = transport
    service = t.service.LwdCommService(t.wire)
    second_request = send(t, 1, 2)
    second = service.submit_send(second_request)
    second_request.tensor.fill_(99)
    assert not second.done() and not t.calls
    service.submit_send(send(t, 0))
    assert [tensor.tolist() for _, tensor in t.calls] == [[1, 1], [2, 2]]
    assert second._keepalive[0].tolist() == [2, 2]
    assert service.poll_completions() == 2
    assert second._keepalive is None
    service.shutdown()


@pytest.mark.xfail(strict=True, reason="PR #63 does not advance a skipped hole reached while draining held batches")
def test_abort_holes_advance_across_alternating_held_batches(transport: SimpleNamespace) -> None:
    t = transport
    service = t.service.LwdCommService(t.wire)
    third = service.submit_send(send(t, 3, 3))
    service.skip_seqno(t.types.LwdChannelType.UP, 0, "send")
    canceled = service.submit_send(send(t, 2, 2))
    service.skip_seqno(t.types.LwdChannelType.UP, 2, "send")
    service.submit_send(send(t, 1, 1))
    assert [tensor.tolist() for _, tensor in t.calls] == [[1, 1], [3, 3]]
    assert third.done()
    with pytest.raises(RuntimeError, match="failed"):
        canceled.wait(0)
    with pytest.raises(RuntimeError, match="failed"):
        service.submit_send(send(t, 2)).wait(0)
    service.shutdown()


def test_up_and_down_have_independent_sequence_zero(transport: SimpleNamespace) -> None:
    t = transport
    service = t.service.LwdCommService(t.wire)
    up = service.submit_send(send(t, 0))
    down = service.submit_recv(
        t.types.LwdCommRequest(t.types.LwdChannelType.DOWN, "recv", 3, seqno=0, connection_key=t.wire.key)
    )
    assert up.wait(0).status is t.types.LwdCommStatus.OK
    assert down.wait(0).tensor.tolist() == [9, 9, 9]
    service.shutdown()


def test_work_is_bridged_before_recording_completion(
    transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    t, order = transport, []
    monkeypatch.setattr(
        torch.distributed, "isend", lambda *args, **kwargs: SimpleNamespace(wait=lambda: order.append("wait"))
    )
    monkeypatch.setattr(torch.npu.Event, "record", lambda event: order.append("event"))
    t.service.LwdCommService(t.wire).submit_send(send(t, 0))
    assert order == ["wait", "event"]


def test_up_aux_frame_keeps_source_order_and_dtype(transport: SimpleNamespace) -> None:
    t = transport
    request = replace(send(t, 0), aux_tensor=torch.tensor([3, 4, 5]), aux_num_elements=3)
    sent = t.service.LwdCommService(t.wire).submit_send(request)
    assert [tensor.tolist() for _, tensor in t.calls] == [[1, 1], [3, 4, 5]]
    assert t.calls[1][1].dtype is torch.int64
    assert sent.wait(0).status is t.types.LwdCommStatus.OK
    t.wire.is_edge = False
    received = (
        t.service.LwdCommService(t.wire).submit_recv(replace(request, op="recv", tensor=None, aux_tensor=None)).wait(0)
    )
    assert received.tensor.tolist() == [9, 9]
    assert received.aux_tensor.tolist() == [9, 9, 9]
    assert received.aux_tensor.dtype is torch.int64


def test_service_rejects_wrong_submission_operation(transport: SimpleNamespace) -> None:
    t = transport
    with pytest.raises(AssertionError, match="requires op"):
        t.service.LwdCommService(t.wire).submit_recv(send(t, 0))
    assert not t.calls


def test_pending_future_retains_payload_until_completion(transport: SimpleNamespace) -> None:
    t = transport
    service = t.service.LwdCommService(t.wire)
    future = service.submit_send(send(t, 0))
    future._done_event.complete = False
    with pytest.raises(TimeoutError, match="timed out.*seqno=0"):
        future.wait(0)
    assert future._keepalive is not None
    future._done_event.complete = True
    assert service.poll_completions() == 1
    assert future._keepalive is None
    service.shutdown(timeout=0)
    assert not service._channels


def test_service_shutdown_preserves_source_timeout_and_reset(transport: SimpleNamespace) -> None:
    t = transport
    service = t.service.LwdCommService(t.wire)
    service.submit_send(send(t, 0))._done_event.complete = False
    with pytest.raises(TimeoutError, match="timed out"):
        service.shutdown(timeout=0)
    assert not service._channels and not service._shutting_down


def test_missing_receive_sequence_is_reported_as_queued(transport: SimpleNamespace) -> None:
    t = transport
    service = t.service.LwdCommService(t.wire)
    request = t.types.LwdCommRequest(t.types.LwdChannelType.DOWN, "recv", 3, seqno=1, connection_key=t.wire.key)
    future = service.submit_recv(request)
    with pytest.raises(TimeoutError, match="timed out.*unbound"):
        future.wait(0)
    assert not t.events
    service.submit_recv(replace(request, seqno=0))
    assert future.wait(0).tensor.tolist() == [9, 9, 9]
    service.shutdown()


def test_enqueue_failure_is_raised_to_the_caller(transport: SimpleNamespace) -> None:
    t = transport

    def fail(*args: object) -> None:
        raise RuntimeError("device enqueue failed")

    t.wire.p2p_send = fail
    with pytest.raises(RuntimeError, match="device enqueue failed"):
        t.service.LwdCommService(t.wire).submit_send(send(t, 0))


@pytest.mark.parametrize("role,tp_rank", [("edge", 0), ("cloud", 0), ("cloud", 1)])
def test_all_ranks_create_two_groups_and_only_endpoints_warm_up(
    transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, role: str, tp_rank: int
) -> None:
    t, calls, transfers = transport, [], []
    Key = namedtuple("Key", "edge_id cloud_id dp_idx")
    link = SimpleNamespace(
        key=Key(0, 0, 0),
        dp_idx=0,
        edge=SimpleNamespace(addr="edge", ranks=(0,)),
        cloud=SimpleNamespace(addr="cloud", ranks=(1, 2)),
    )
    config = SimpleNamespace(role=role, connections=(link,))
    module = t.lwd_wire
    bootstrap = SimpleNamespace(device_group=object(), barrier=Mock())
    monkeypatch.setattr(module, "get_lwd_bootstrap_world", lambda: bootstrap)
    rank = 0 if role == "edge" else tp_rank + 1
    monkeypatch.setattr(module.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(module.dist, "get_backend", lambda group: "hccl")

    def new_group(ranks: list[int], **kwargs: object) -> object:
        group = object()
        calls.append((ranks, group))
        return group

    monkeypatch.setattr(module.dist, "new_group", new_group)

    def post(op: str, tensor: torch.Tensor, peer: int, group: object) -> Mock:
        transfers.append((op, peer, group))
        return Mock()

    monkeypatch.setattr(module.dist, "isend", lambda tensor, dst, group: post("send", tensor, dst, group))
    monkeypatch.setattr(module.dist, "irecv", lambda tensor, src, group: post("recv", tensor, src, group))
    monkeypatch.setattr(module.dist, "destroy_process_group", Mock())
    wire = module.LwdWire(config, tp_rank, torch.device("cpu"))
    wire.init_lwd_duplex_channels()
    assert [ranks for ranks, _ in calls] == [[0, 1], [0, 1]]
    assert calls[0][1] is not calls[1][1]
    assert len(transfers) == (2 if tp_rank == 0 else 0)
    if transfers:
        assert [item[0] for item in transfers] == (["send", "recv"] if role == "edge" else ["recv", "send"])
        assert transfers[0][2] is calls[0][1] and transfers[1][2] is calls[1][1]
    bootstrap.barrier.assert_called_once()
    wire.shutdown()


@pytest.mark.parametrize("endpoint", [True, False])
def test_worker_distributes_received_tensor_to_local_tp_peers(
    worker_transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, endpoint: bool
) -> None:
    t = worker_transport
    t.wire.is_edge, t.wire.member = False, endpoint
    t.group.world_size, t.group.rank_in_group = 2, 0 if endpoint else 1
    broadcasts = []

    def broadcast(tensor: torch.Tensor, **kwargs: object) -> Mock:
        if endpoint:
            assert tensor.tolist() == [9, 9, 9]
        else:
            tensor.fill_(9)
        broadcasts.append(kwargs)
        return Mock()

    monkeypatch.setattr(t.module.dist, "broadcast", broadcast)
    future = (
        t.worker.comm_service.submit_recv(t.types.LwdCommRequest(t.types.LwdChannelType.UP, "recv", 3, seqno=0))
        if endpoint
        else None
    )
    assert (future is not None) == endpoint
    tensor = t.worker.receive_lwd_tensor(future, 3)
    assert tensor.tolist() == [9, 9, 9]
    assert broadcasts == [{"src": 0, "group": t.group.device_group, "async_op": True}]


@pytest.mark.parametrize("endpoint", [True, False])
def test_receive_failure_reaches_tp_peers_before_tensor_broadcast(
    worker_transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, endpoint: bool
) -> None:
    t = worker_transport
    t.wire.member = endpoint
    t.group.world_size = 2
    broadcast = Mock()
    monkeypatch.setattr(t.module.dist, "broadcast", broadcast)
    future = Mock()
    future.wait.side_effect = TimeoutError("receive timed out")
    if not endpoint:
        t.group.broadcast_object.side_effect = None
        t.group.broadcast_object.return_value = "receive timed out"
    with pytest.raises(RuntimeError, match="receive timed out"):
        t.worker.receive_lwd_tensor(future if endpoint else None, 3)
    broadcast.assert_not_called()


def test_early_receive_reuses_future_and_retains_it_until_both_consumers_finish(
    worker_transport: SimpleNamespace,
) -> None:
    t = worker_transport
    t.wire.is_edge = False
    worker = t.worker
    worker._irecv_done_mq = Mock()
    hint = {"channel": t.types.LwdChannelType.UP, "seqno": 0, "num_tokens": 1, "aux_rows": 0}
    worker.start_early_irecv(hint)
    future = worker._early_recv_futures[0]
    worker.start_early_irecv(hint)
    assert worker.get_or_post_early_recv(0, 1, 0) is future
    assert worker._early_recv_futures[0] is future
    worker._report_irecv_completions()
    worker._irecv_done_mq.enqueue.assert_called_once_with((t.types.LwdChannelType.UP, 0))
    assert not worker._early_recv_futures


def test_early_receive_bounds_prefetch_and_does_not_report_failed_receives(worker_transport: SimpleNamespace) -> None:
    t = worker_transport
    t.wire.is_edge = False
    worker = t.worker
    worker._irecv_done_mq = Mock()
    for seqno in range(4):
        worker.start_early_irecv({"channel": t.types.LwdChannelType.UP, "seqno": seqno, "num_tokens": 1, "aux_rows": 0})
    assert len(worker._early_recv_futures) == 3 and list(worker._early_recv_pending) == [3]
    worker._report_irecv_completions()
    assert len(worker._early_recv_futures) == 3
    worker.get_or_post_early_recv(0, 1, 0)
    assert set(worker._early_recv_futures) == {1, 2, 3}
    worker._early_recv_futures[3]._finalize(error=RuntimeError("receive failed"))
    with pytest.raises(RuntimeError, match="failed"):
        worker._report_irecv_completions()
    assert worker._irecv_done_mq.enqueue.call_count == 3


@pytest.mark.parametrize("failure", [None, TimeoutError("join failed")])
def test_worker_finalizes_wire_before_native_memory_profiling(
    worker_transport: SimpleNamespace, failure: Exception | None
) -> None:
    t = worker_transport
    profile = t.base.NPUWorker.determine_available_memory
    t.wire.ensure_joined = Mock(side_effect=failure)
    if failure is not None:
        with pytest.raises(RuntimeError, match="join failed"):
            t.worker.determine_available_memory()
        profile.assert_not_called()
    else:
        profile.side_effect = lambda: t.wire.ensure_joined.assert_called_once() or 123
        assert t.worker.determine_available_memory() == 123


@pytest.mark.parametrize(
    "fault",
    ["scene", "links", "ranks", "pipeline_parallel_size", "data_parallel_size", "prefill_context_parallel_size"],
)
def test_worker_rejects_unsupported_layout_before_device_initialization(
    worker_transport: SimpleNamespace, fault: str
) -> None:
    t = worker_transport
    config = SimpleNamespace(
        topology=SimpleNamespace(deployment=SimpleNamespace(scene="single_instance")),
        connections=(object(),),
        instance=SimpleNamespace(dp=(SimpleNamespace(ranks=(0, 1)),)),
    )
    t.worker.vllm_config = SimpleNamespace(lwd_config=config)
    t.worker.parallel_config = SimpleNamespace(
        tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, prefill_context_parallel_size=1
    )
    if fault == "scene":
        config.topology.deployment.scene = "cloud_share"
    elif fault == "links":
        config.connections = (object(), object())
    elif fault == "ranks":
        config.instance.dp[0].ranks = (0,)
    else:
        setattr(t.worker.parallel_config, fault, 2)
    with pytest.raises(ValueError):
        t.worker.init_device()
    t.base.NPUWorker.init_device.assert_not_called()
