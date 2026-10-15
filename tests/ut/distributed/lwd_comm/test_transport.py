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


def test_join_uses_one_domain_for_both_directions_and_leaves_tp_ranks_local(
    transport: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
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
        role="edge",
        topology=SimpleNamespace(deployment=SimpleNamespace(scene="single_instance")),
        connections=(link,),
        transport=SimpleNamespace(
            post_out_host=None, wire_store_port=29600, wire_timeout_s=1, endpoints=lambda link: {}
        ),
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
    monkeypatch.setattr(module.dist, "TCPStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(module.dist, "PrefixStore", lambda *args: object())
    wire = module.LwdWire(cfg, 0, torch.device("cpu"))
    wire.join_async()
    wire.ensure_joined(1)
    assert calls == [(0, 2)]
    assert len(wire._streams) == 2 and wire._comm == 123
    assert wire.connection.edge_ranks == (5, 2)
    other = module.LwdWire(cfg, 1, torch.device("cpu"))
    other.join_async()
    assert not other.member and other._thread is None
    with pytest.raises(RuntimeError, match="non-endpoint"):
        other.ensure_joined(0)
    wire.shutdown()


def test_early_receive_reports_failures_instead_of_advancing(transport: SimpleNamespace) -> None:
    pre_recv = importlib.import_module(transport.types.__package__ + ".pre_recv")
    pre_recv.register_hint_sender(lambda hint: None)
    assert pre_recv.hint_fields(pre_recv.make_recv_hint(0, 4)) == (0, 4, 0)
    for seqno, tokens, aux in [(-1, 4, 0), (0, 0, 0), (0, 4, 1)]:
        with pytest.raises(ValueError):
            pre_recv.make_recv_hint(seqno, tokens, aux)
    with pytest.raises(RuntimeError, match="broken receive"):
        pre_recv.record_irecv_completions([pre_recv.LwdRecvFailure("broken receive")])
    assert not pre_recv.is_irecv_complete(transport.types.LwdChannelType.UP, 0)
    pre_recv.record_irecv_completions([(transport.types.LwdChannelType.UP, 0)])
    assert pre_recv.is_irecv_complete(transport.types.LwdChannelType.UP, 0)


def test_receive_slots_live_until_both_report_and_consumption(transport: SimpleNamespace) -> None:
    import ast
    import threading

    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/lwd_cloud/lwd_cloud_worker.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LwdCloudWorker")
    pre_recv = importlib.import_module(transport.types.__package__ + ".pre_recv")
    symbols = {
        "torch": torch,
        "threading": threading,
        "LwdWorker": object,
        "LwdCloudModelRunner": object,
        "LwdChannelType": transport.types.LwdChannelType,
        "hint_fields": pre_recv.hint_fields,
        "_LWD_EARLY_RECV_MAX_INFLIGHT": 3,
    }
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    exec(compile(module, str(path), "exec"), symbols)
    worker = symbols["LwdCloudWorker"]()
    worker._init_lwd_early_recv_state()
    worker._lwd_build_up_recv_request = lambda seqno, tokens, aux: seqno
    submitted, reports = [], []

    def submit(seqno: int) -> SimpleNamespace:
        submitted.append(seqno)
        return SimpleNamespace(done=lambda: True, result=lambda: None)

    worker.comm_service = SimpleNamespace(submit_recv=submit)
    worker._irecv_done_mq = SimpleNamespace(enqueue=reports.append)
    for seqno in range(4):
        worker.start_early_irecv(pre_recv.make_recv_hint(seqno, 4))
    assert submitted == [0, 1, 2]
    worker._report_irecv_completions()
    assert len(reports) == 3
    assert submitted == [0, 1, 2]  # Receiving alone must not free the buffer.
    worker.get_or_post_early_recv(0, 4, 0)
    assert submitted == [0, 1, 2, 3]
    worker.start_early_irecv(pre_recv.make_recv_hint(0, 4))
    assert submitted == [0, 1, 2, 3]  # A late duplicate cannot post another receive.
