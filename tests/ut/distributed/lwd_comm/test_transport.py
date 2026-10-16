# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host ordering tests with CPU tensors and simulated device completions."""

import importlib
import importlib.util
import io
import json
import queue
import sys
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import copy
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
        domain_rank=0,
        timeout_s=0.05,
        ensure_joined=lambda: None,
        get_channel_stream=lambda channel: stream,
        p2p_send=lambda channel, tensor, peer: calls.append((channel, tensor.clone())),
        p2p_recv=lambda channel, tensor, peer: tensor.fill_(9),
        shutdown=lambda: None,
    )
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
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)
    return SimpleNamespace(**modules, wire=wire, calls=calls, events=events, waits=waits, group=group)


@pytest.fixture
def worker_transport(transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    t = transport
    mq = ModuleType("vllm.distributed.device_communicators.shm_broadcast")
    mq.Handle, mq.MessageQueue = object, Mock()
    monkeypatch.setitem(sys.modules, mq.__name__, mq)
    early = importlib.import_module(t.types.__package__ + ".early_recv")
    probe = importlib.import_module(t.types.__package__ + ".probe")
    for name in ("types", "future", "service", "lwd_wire"):
        monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.lwd_comm." + name, getattr(t, name))
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.lwd_comm.early_recv", early)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.lwd_comm.probe", probe)
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
    worker._lwd_probe_round = 0
    worker._lwd_probe_next = {channel: 0 for channel in t.types.LwdChannelType}
    worker._lwd_probe_error = None
    # CPU CI exercises the real Worker and service methods, replacing only NPU operations.
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    return SimpleNamespace(**vars(t), worker=worker, module=module, probe=probe, early=early, base=base)


def send(t: SimpleNamespace, seqno: int, value: float = 1) -> Any:
    return t.types.LwdCommRequest(
        t.types.LwdChannelType.UP,
        "send",
        2,
        torch.full((2,), value, dtype=torch.bfloat16),
        seqno=seqno,
        connection_key=t.wire.key,
    )


def test_out_of_order_send_preserves_snapshot_and_producer_event(transport: SimpleNamespace) -> None:
    t = transport
    service = t.service.LwdCommService(t.wire)
    second_request = send(t, 1, 2)
    second = service.submit_send(second_request)
    producer = second.producer_event
    second_request.tensor.fill_(99)
    assert not second.done() and not t.calls
    first = service.submit_send(send(t, 0))
    assert [tensor.tolist() for _, tensor in t.calls] == [[1, 1], [2, 2]]
    assert t.waits == [first.producer_event, producer]
    assert service.poll_completions() == 2
    assert second.request.tensor is None
    service.shutdown()


def test_abort_holes_advance_across_alternating_held_batches(transport: SimpleNamespace) -> None:
    t = transport
    service = t.service.LwdCommService(t.wire)
    service.skip_seqno(t.types.LwdChannelType.UP, 0, "send")
    third = service.submit_send(send(t, 3, 3))
    canceled = service.submit_send(send(t, 2, 2))
    service.skip_seqno(t.types.LwdChannelType.UP, 2, "send")
    service.submit_send(send(t, 1, 1))
    assert [tensor.tolist() for _, tensor in t.calls] == [[1, 1], [3, 3]]
    assert third.done()
    with pytest.raises(RuntimeError, match="failed"):
        canceled.wait(0)
    with pytest.raises(ValueError, match="duplicate or skipped"):
        service.submit_send(send(t, 2))
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


@pytest.mark.parametrize(
    "change",
    [
        {"connection_key": (9, 0, 0)},
        {"num_elements": 3},
        {"seqno": -1},
        {"src_dst": 8},
    ],
)
def test_invalid_requests_never_touch_the_wire(transport: SimpleNamespace, change: dict) -> None:
    t = transport
    with pytest.raises(ValueError):
        t.service.LwdCommService(t.wire).submit_send(replace(send(t, 0), **change))
    assert not t.calls and not t.events


def test_pending_dma_retains_its_buffer_after_timeout(transport: SimpleNamespace) -> None:
    t = transport
    service = t.service.LwdCommService(t.wire)
    future = service.submit_send(send(t, 0))
    future._done_event.complete = False
    with pytest.raises(TimeoutError, match="seqno=0 timed out stage=device_pending"):
        future.wait(0)
    with pytest.raises(RuntimeError, match="could not drain"):
        service.shutdown(timeout=0)
    assert future.request.tensor is not None
    future._done_event.complete = True
    service.shutdown(timeout=0.1)
    assert future.request.tensor is None
    with pytest.raises(RuntimeError, match="closed"):
        service.submit_send(send(t, 1))


def test_missing_receive_sequence_is_reported_as_queued(transport: SimpleNamespace) -> None:
    t = transport
    service = t.service.LwdCommService(t.wire)
    request = t.types.LwdCommRequest(t.types.LwdChannelType.DOWN, "recv", 3, seqno=1, connection_key=t.wire.key)
    future = service.submit_recv(request)
    with pytest.raises(TimeoutError, match="lwd_down seqno=1 timed out stage=queued"):
        future.wait(0)
    assert not t.events
    service.submit_recv(replace(request, seqno=0))
    assert future.wait(0).tensor.tolist() == [9, 9, 9]
    service.shutdown()


@pytest.mark.parametrize("is_edge", [True, False])
@pytest.mark.parametrize("channel", ["up", "down"])
def test_probe_uses_the_same_bounded_transport_for_either_role(
    worker_transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, is_edge: bool, channel: str
) -> None:
    t = worker_transport
    probe = importlib.import_module(t.types.__package__ + ".probe")
    t.wire.is_edge, t.wire.domain_rank = is_edge, 0 if is_edge else 1
    plan = probe.LwdProbePlan(batches=7, tokens=(1, 4, 2), hidden_size=8, window=3, channel=channel)
    payloads, posts_at_wait = [], []

    def post(channel: object, tensor: torch.Tensor, peer: int | None) -> None:
        expected = probe.make_payload(len(payloads), tensor.numel())
        if is_edge == (plan.channel == "up"):
            assert torch.equal(tensor, expected)
        else:
            tensor.copy_(expected)
        payloads.append(tensor.clone())

    original_wait = t.future.LwdCommFuture.wait

    def wait(future: Any, timeout: float | None = None) -> Any:
        posts_at_wait.append(len(payloads))
        return original_wait(future, timeout)

    t.wire.p2p_send = t.wire.p2p_recv = post
    monkeypatch.setattr(t.future.LwdCommFuture, "wait", wait)
    service = t.worker.comm_service
    probe.run_batches(t.worker, plan)
    assert [tensor.numel() for tensor in payloads] == [8, 32, 16, 8, 32, 16, 8]
    assert posts_at_wait == [3, 4, 5, 6, 7, 7, 7]
    service.shutdown()


def test_probe_rejects_corrupt_or_reordered_payload(worker_transport: SimpleNamespace) -> None:
    t = worker_transport
    probe = importlib.import_module(t.types.__package__ + ".probe")
    t.wire.is_edge, t.wire.domain_rank = False, 1
    t.wire.p2p_recv = lambda channel, tensor, peer: tensor.copy_(probe.make_payload(1, tensor.numel()))
    with pytest.raises(ValueError, match="payload mismatch seqno=0"):
        probe.run_batches(t.worker, probe.LwdProbePlan(batches=1))


@pytest.mark.parametrize("peer_payload", ['{"code":"old","batches":1}', '{"code":"same","batches":2}'])
def test_probe_rejects_peer_code_or_plan_mismatch(transport: SimpleNamespace, peer_payload: str) -> None:
    probe = importlib.import_module(transport.types.__package__ + ".probe")
    store = SimpleNamespace(
        set=lambda key, value: None, wait=lambda keys, timeout: None, get=lambda key: peer_payload.encode()
    )
    with pytest.raises(ValueError, match="peer plan mismatch"):
        probe.exchange_check(store, "edge", "plan", '{"code":"same","batches":1}', 1.0)
    probe.exchange_check(store, "cloud", "plan", peer_payload, 1.0)


@pytest.mark.parametrize("invalid", [{"tokens": (0,)}, {"hidden_size": 4}, {"window": 0}, {"batches": 0}])
def test_probe_rejects_invalid_transfer_plan(transport: SimpleNamespace, invalid: dict[str, object]) -> None:
    probe = importlib.import_module(transport.types.__package__ + ".probe")
    with pytest.raises(ValueError):
        probe.LwdProbePlan(**invalid)


def test_post_failure_fails_waiters_without_releasing_possible_dma(transport: SimpleNamespace) -> None:
    t = transport
    service = t.service.LwdCommService(t.wire)
    held = service.submit_send(send(t, 1))

    def fail(*args: object) -> None:
        raise RuntimeError("device enqueue failed")

    t.wire.p2p_send = fail
    with pytest.raises(RuntimeError, match="device enqueue"):
        service.submit_send(send(t, 0))
    with pytest.raises(RuntimeError, match="failed"):
        held.wait(0)
    with pytest.raises(RuntimeError, match="failed"):
        service.poll_completions()
    channel = next(iter(service._channels.values()))
    assert channel._failed[0].request.tensor is not None


@pytest.mark.parametrize("role,domain_rank", [("edge", 0), ("cloud", 1)])
def test_join_uses_one_domain_for_both_directions_and_leaves_tp_ranks_local(
    transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, role: str, domain_rank: int
) -> None:
    t = transport
    calls = []
    module = t.lwd_wire
    # A hashable key with the same attributes as the actual config key.
    Key = namedtuple("Key", "edge_id cloud_id dp_idx")
    key = Key(0, 0, 0)
    link = SimpleNamespace(
        key=key,
        dp_idx=0,
        edge=SimpleNamespace(addr="edge", ranks=(5, 2)),
        cloud=SimpleNamespace(addr="cloud", ranks=(4, 0, 3, 1)),
    )
    cfg = SimpleNamespace(
        role=role,
        topology=SimpleNamespace(deployment=SimpleNamespace(scene="single_instance")),
        connections=(link,),
    )

    class PG:
        Options = SimpleNamespace

        def __init__(self, store: object, rank: int, size: int, options: object) -> None:
            calls.append((rank, size))

        def get_hccl_comm(self, device: object) -> int:
            return 123

    stub = ModuleType("torch_npu._C._distributed_c10d")
    stub.ProcessGroupHCCL = PG
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    hccl = ModuleType("lwd_transport_test.lwd_hccl")
    hccl.hcclComm_t = lambda value: value
    hccl.LwdHcclLibrary = lambda: SimpleNamespace(
        send=lambda tensor, peer, comm: None, recv=lambda tensor, peer, comm: tensor.fill_(1 + peer)
    )
    monkeypatch.setitem(sys.modules, hccl.__name__, hccl)
    store_calls = []

    def tcp_store(*args: object, **kwargs: object) -> object:
        store_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(module.dist, "TCPStore", tcp_store)
    monkeypatch.setattr(module.dist, "PrefixStore", lambda *args: object())
    wire = module.LwdWire(cfg, 0, torch.device("cpu"))
    wire.join_async()
    wire.ensure_joined(1)
    assert calls == [(domain_rank, 2)]
    args, kwargs = store_calls[0]
    assert args == ("edge", 29600, 2, role == "edge")
    assert kwargs["timeout"].total_seconds() == 1200
    assert len(wire._streams) == 2 and wire._comm == 123
    assert wire.connection.edge_ranks == (5, 2)
    other = module.LwdWire(cfg, 1, torch.device("cpu"))
    other.join_async()
    assert not other.member and other._thread is None
    with pytest.raises(RuntimeError, match="non-endpoint"):
        other.ensure_joined(0)
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
    future = t.worker.submit_lwd_recv(3, t.types.LwdChannelType.UP, 0)
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
    worker._early_recv_futures[3]._fail(RuntimeError("receive failed"))
    with pytest.raises(RuntimeError, match="failed"):
        worker._report_irecv_completions()
    assert worker._irecv_done_mq.enqueue.call_count == 3


def test_two_workers_reuse_transport_for_repeated_up_and_down_probes(
    worker_transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = worker_transport
    topology = SimpleNamespace(digest="x")
    t.worker.vllm_config = SimpleNamespace(lwd_config=SimpleNamespace(role="edge", topology=topology))
    cloud = copy(t.worker)
    cloud.vllm_config = SimpleNamespace(lwd_config=SimpleNamespace(role="cloud", topology=topology))
    cloud.lwd_wire = copy(t.wire)
    cloud.lwd_wire.is_edge, cloud.lwd_wire.domain_rank = False, 1
    cloud.comm_service = t.service.LwdCommService(cloud.lwd_wire)
    cloud._lwd_probe_next = dict(t.worker._lwd_probe_next)
    real_asdict = t.probe.asdict
    monkeypatch.setattr(
        t.probe,
        "asdict",
        lambda obj: real_asdict(obj)
        if isinstance(obj, t.probe.LwdProbePlan)
        else ({"digest": "x"} if obj is topology else {"edge_id": 0}),
    )
    values, condition, prefixes = {}, threading.Condition(), []
    pipes = {channel: queue.Queue() for channel in t.types.LwdChannelType}

    class Store:
        def __init__(self, prefix: str, store: object) -> None:
            self.prefix = prefix
            prefixes.append(prefix)

        def set(self, key: str, value: str) -> None:
            with condition:
                values[self.prefix + key] = value
                condition.notify_all()

        def wait(self, keys: list[str], timeout: object) -> None:
            with condition:
                assert condition.wait_for(lambda: all(self.prefix + key in values for key in keys), 2)

        def get(self, key: str) -> bytes:
            return values[self.prefix + key].encode()

    for worker in (t.worker, cloud):
        wire = worker.lwd_wire
        wire._store, wire.timeout_s = object(), 2
        wire.p2p_send = lambda channel, tensor, peer: pipes[channel].put(tensor.clone())
        wire.p2p_recv = lambda channel, tensor, peer: tensor.copy_(pipes[channel].get(timeout=2))
    monkeypatch.setattr(t.probe.dist, "PrefixStore", Store)
    with ThreadPoolExecutor(2) as executor:
        for round_id, (channel, start) in enumerate((("up", 0), ("up", 3), ("down", 0))):
            plan = json.dumps({"batches": 3, "hidden_size": 8, "channel": channel})
            calls = [executor.submit(worker.lwd_run_probe, plan) for worker in (t.worker, cloud)]
            results = [call.result(timeout=5) for call in calls]
            assert {result["role"] for result in results} == {"edge", "cloud"}
            assert all(result["round"] == round_id and result["start_seqno"] == start for result in results)
            assert all(result["status"] == "PASS" for result in results)
    assert sorted(prefixes) == [f"lwd_probe_round/{i}" for i in (0, 0, 1, 1, 2, 2)]
    assert all(pipe.empty() for pipe in pipes.values())
    assert not t.worker.comm_service._closed and not cloud.comm_service._closed
    t.worker.shutdown()
    cloud.shutdown()
    assert t.worker.comm_service._closed and cloud.comm_service._closed
    assert t.base.NPUWorker.shutdown.call_count == 2


def test_failed_probe_requires_restart_and_keeps_transport_buffers(
    worker_transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = worker_transport
    run = Mock(side_effect=TimeoutError("device pending"))
    monkeypatch.setattr(t.probe, "run_probe", run)
    with pytest.raises(TimeoutError):
        t.worker.lwd_run_probe()
    with pytest.raises(RuntimeError, match="restart after failed probe"):
        t.worker.lwd_run_probe()
    assert run.call_count == 1 and not t.worker.comm_service._closed


def test_probe_reports_completion_reaping_failure_to_local_tp_peers(worker_transport: SimpleNamespace) -> None:
    t = worker_transport
    t.worker.comm_service.poll_completions = Mock(side_effect=RuntimeError("later transfer failed"))
    with pytest.raises(RuntimeError, match="later transfer failed"):
        t.probe.run_batches(t.worker, t.probe.LwdProbePlan(batches=1, hidden_size=8))
    assert t.group.broadcast_object.call_args.args == ("later transfer failed",)


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


@pytest.mark.parametrize("result", [[{"status": "PASS", "tp_rank": 0}], [], [{"status": "FAIL"}]])
def test_client_waits_for_server_and_requires_all_worker_results(
    transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, result: list[dict]
) -> None:
    client = importlib.import_module(transport.types.__package__ + ".probe_client")
    calls = []

    def urlopen(request: object, timeout: float) -> io.BytesIO:
        calls.append(request)
        if len(calls) == 1:
            raise client.URLError("server starting")
        response = io.BytesIO(json.dumps({"results": result}).encode())
        response.status = 200
        return response

    monkeypatch.setattr(client, "urlopen", urlopen)
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    if result and result[0]["status"] == "PASS":
        assert client.run_client("http://127.0.0.1:8000", {"batches": 2}, 10) == result
    else:
        with pytest.raises(RuntimeError, match="Not all local Workers passed"):
            client.run_client("http://127.0.0.1:8000", {"batches": 2}, 10)
    assert calls[:2] == ["http://127.0.0.1:8000/health"] * 2
    assert json.loads(calls[2].data)["method"] == "lwd_run_probe"


def test_client_never_triggers_transfers_if_startup_times_out(
    transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = importlib.import_module(transport.types.__package__ + ".probe_client")
    request = Mock(side_effect=client.URLError("server absent"))
    monkeypatch.setattr(client, "urlopen", request)
    monkeypatch.setattr(client.time, "monotonic", Mock(side_effect=[0, 11]))
    with pytest.raises(TimeoutError, match="did not become healthy"):
        client.run_client("http://127.0.0.1:8000", {}, 10)
    assert request.call_count == 1
