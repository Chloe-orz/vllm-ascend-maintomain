# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host ordering tests with CPU tensors and simulated device completions."""

import ast
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
        domain_rank=0,
        timeout_s=0.05,
        ensure_joined=lambda: None,
        get_channel_stream=lambda channel: stream,
        p2p_send=lambda channel, tensor, peer: calls.append((channel, tensor.clone())),
        p2p_recv=lambda channel, tensor, peer: tensor.fill_(9),
        shutdown=lambda: None,
    )
    return SimpleNamespace(**modules, wire=wire, calls=calls, events=events, waits=waits)


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
    transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, is_edge: bool, channel: str
) -> None:
    t = transport
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
    service = t.service.LwdCommService(t.wire)
    probe.run_batches(service, t.wire, plan)
    assert [tensor.numel() for tensor in payloads] == [8, 32, 16, 8, 32, 16, 8]
    assert posts_at_wait == [3, 4, 5, 6, 7, 7, 7]
    service.shutdown()


def test_probe_rejects_corrupt_or_reordered_payload(transport: SimpleNamespace) -> None:
    t = transport
    probe = importlib.import_module(t.types.__package__ + ".probe")
    t.wire.is_edge, t.wire.domain_rank = False, 1
    t.wire.p2p_recv = lambda channel, tensor, peer: tensor.copy_(probe.make_payload(1, tensor.numel()))
    with pytest.raises(ValueError, match="payload mismatch seqno=0"):
        probe.run_batches(t.service.LwdCommService(t.wire), t.wire, probe.LwdProbePlan(batches=1))


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


@pytest.fixture
def startup_probe(transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    probe = importlib.import_module(transport.types.__package__ + ".probe")
    config = SimpleNamespace(
        role="edge",
        topology=SimpleNamespace(deployment=SimpleNamespace(scene="single_instance")),
        connections=(object(),),
        instance=SimpleNamespace(dp=(SimpleNamespace(ranks=(0, 1)),)),
    )
    parallel = SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1, prefill_context_parallel_size=1)
    group = SimpleNamespace(world_size=2, rank_in_group=0, broadcast_object=Mock(side_effect=lambda value, src: value))
    run = Mock(return_value=transport.wire)
    monkeypatch.setattr(probe, "run_probe", run)
    return SimpleNamespace(probe=probe, config=config, parallel=parallel, group=group, run=run, wire=transport.wire)


@pytest.mark.parametrize("role,size", [("edge", 2), ("cloud", 8)])
def test_startup_probe_uses_parsed_config_once_per_tp_group(
    startup_probe: SimpleNamespace, role: str, size: int
) -> None:
    t = startup_probe
    t.config.role = role
    t.config.instance.dp[0].ranks = tuple(range(size))
    t.group.world_size = size
    device = torch.device("cpu")
    wire = t.probe.run_startup_probe(t.config, t.parallel, device, t.group)
    assert wire is t.wire
    t.run.assert_called_once_with(t.config, device, t.probe.LwdProbePlan())
    t.group.broadcast_object.assert_called_once_with(None, src=0)
    t.group.rank_in_group = 1
    assert t.probe.run_startup_probe(t.config, t.parallel, device, t.group) is None
    assert t.run.call_count == 1
    assert t.group.broadcast_object.call_count == 2


@pytest.mark.parametrize("failure", [TimeoutError("peer absent"), ValueError("payload mismatch")])
def test_startup_probe_failure_stops_all_local_tp_workers(startup_probe: SimpleNamespace, failure: Exception) -> None:
    t = startup_probe
    t.run.side_effect = failure
    with pytest.raises(RuntimeError, match=str(failure)) as caught:
        t.probe.run_startup_probe(t.config, t.parallel, torch.device("cpu"), t.group)
    assert caught.value.__cause__ is failure
    t.group.broadcast_object.assert_called_once_with(str(failure), src=0)
    t.group.rank_in_group = 1
    t.group.broadcast_object.side_effect = None
    t.group.broadcast_object.return_value = str(failure)
    with pytest.raises(RuntimeError, match=str(failure)):
        t.probe.run_startup_probe(t.config, t.parallel, torch.device("cpu"), t.group)
    assert t.run.call_count == 1


@pytest.mark.parametrize(
    "fault",
    ["pipeline_parallel_size", "data_parallel_size", "prefill_context_parallel_size", "ranks", "scene", "links"],
)
def test_startup_probe_rejects_ambiguous_endpoints_before_join(startup_probe: SimpleNamespace, fault: str) -> None:
    t = startup_probe
    if fault == "ranks":
        t.config.instance.dp[0].ranks = (0,)
    elif fault == "scene":
        t.config.topology.deployment.scene = "cloud_share"
    elif fault == "links":
        t.config.connections = (object(), object())
    else:
        setattr(t.parallel, fault, 2)
    with pytest.raises(ValueError):
        t.probe.run_startup_probe(t.config, t.parallel, torch.device("cpu"), t.group)
    t.run.assert_not_called()
    t.group.broadcast_object.assert_not_called()


@pytest.mark.parametrize("mode", ["centralized", "lwd", "failed"])
def test_worker_startup_only_continues_after_required_wire_check(
    startup_probe: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    t = startup_probe
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/worker.py"
    worker_cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in worker_cls.body if isinstance(node, ast.FunctionDef) and node.name == "init_device")
    # Exercise the real startup method without loading the NPU backend on CPU CI.
    module = ast.Module(body=[method], type_ignores=[])
    runner = Mock()
    scope = {
        "get_tp_group": lambda: t.group,
        "init_workspace_manager": Mock(),
        "NPUModelRunner": runner,
        "report_usage_stats": Mock(),
    }
    exec(compile(module, str(path), "exec"), scope)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.lwd_comm.probe", t.probe)
    worker = SimpleNamespace(
        _init_device=lambda: torch.device("cpu"),
        vllm_config=SimpleNamespace(lwd_config=None if mode == "centralized" else t.config),
        parallel_config=t.parallel,
        use_v2_model_runner=False,
        rank=0,
    )
    if mode == "failed":
        t.run.side_effect = TimeoutError("peer absent")
        with pytest.raises(RuntimeError, match="peer absent"):
            scope["init_device"](worker)
        runner.assert_not_called()
    else:
        scope["init_device"](worker)
        runner.assert_called_once_with(worker.vllm_config, worker.device)
        assert t.run.call_count == (0 if mode == "centralized" else 1)
        if mode == "lwd":
            assert worker._lwd_probe_wire is t.wire
