# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Check the LWD transport during ordinary centralized worker startup."""

from __future__ import annotations

import json
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from vllm.logger import logger

from .types import LwdChannelType, LwdCommRequest

if TYPE_CHECKING:
    from torch.distributed import Store
    from vllm.config import ParallelConfig
    from vllm.config.lwd import LwdConfig
    from vllm.distributed.parallel_state import GroupCoordinator

    from .future import LwdCommFuture
    from .lwd_wire import LwdWire
    from .service import LwdCommService

_SEQUENCE_BYTES = 8


@dataclass(frozen=True)
class LwdProbePlan:
    batches: int = 200
    tokens: tuple[int, ...] = (1, 4, 2, 8)
    hidden_size: int = 5120
    window: int = 4
    channel: str = "up"

    def __post_init__(self) -> None:
        if not 0 < self.batches <= 256**_SEQUENCE_BYTES or self.window < 1:
            raise ValueError("batches must fit the sequence header; window must be positive")
        if not self.tokens or min(self.tokens) < 1 or self.hidden_size < _SEQUENCE_BYTES:
            raise ValueError("positive token counts and hidden_size >= 8 are required")
        if self.channel not in ("up", "down"):
            raise ValueError("channel must be up or down")

    def num_elements(self, seqno: int) -> int:
        return self.tokens[seqno % len(self.tokens)] * self.hidden_size


def make_payload(seqno: int, num_elements: int) -> torch.Tensor:
    # Byte values are exact in bfloat16; the header detects reordered packets
    # even when successive transfers use the same shape.
    payload = (torch.arange(num_elements, dtype=torch.int64) + seqno % 256) % 256
    payload[:_SEQUENCE_BYTES] = torch.tensor(list(seqno.to_bytes(_SEQUENCE_BYTES, "little")))
    return payload.to(torch.bfloat16)


def source_digest(directory: Path) -> str:
    digest = sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def source_revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def exchange_check(store: Store, role: str, phase: str, payload: str, timeout: float) -> None:
    peer = "cloud" if role == "edge" else "edge"
    key = f"lwd_probe/{phase}/{peer}"
    store.set(f"lwd_probe/{phase}/{role}", payload)
    store.wait([key], timedelta(seconds=timeout))
    actual = store.get(key).decode()
    if actual != payload:
        raise ValueError(f"[LWD][wire-test] peer {phase} mismatch: local={payload} peer={actual}")


def run_batches(service: LwdCommService, wire: LwdWire, plan: LwdProbePlan) -> None:
    channel = LwdChannelType.UP if plan.channel == "up" else LwdChannelType.DOWN
    sending = wire.is_edge == (channel is LwdChannelType.UP)
    pending: deque[LwdCommFuture] = deque()

    def complete(future: LwdCommFuture) -> None:
        result = future.wait(wire.timeout_s)
        request = future.request
        assert request.seqno is not None
        if not sending:
            expected = make_payload(request.seqno, request.num_elements)
            if result.tensor is None or not torch.equal(result.tensor.cpu(), expected):
                raise ValueError(f"[LWD][wire-test] payload mismatch seqno={request.seqno}")
        logger.info(
            "[LWD][wire-test] %s seqno=%d num_elements=%d %s",
            request.op,
            request.seqno,
            request.num_elements,
            "complete" if sending else "verified",
        )
        service.poll_completions()

    for seqno in range(plan.batches):
        num_elements = plan.num_elements(seqno)
        tensor = make_payload(seqno, num_elements).to(wire.device) if sending else None
        request = LwdCommRequest(
            channel, "send" if sending else "recv", num_elements, tensor, seqno=seqno, connection_key=wire.key
        )
        future = service.submit_send(request) if sending else service.submit_recv(request)
        pending.append(future)
        if len(pending) >= plan.window:
            complete(pending.popleft())
    while pending:
        complete(pending.popleft())


def run_probe(config: LwdConfig, device: torch.device, plan: LwdProbePlan) -> LwdWire:
    import vllm.config.lwd as lwd

    from .lwd_wire import LwdWire
    from .service import LwdCommService

    wire = LwdWire(config, tp_rank=0, device=device)
    service = LwdCommService(wire)
    topology = asdict(config.topology)
    topology.pop("digest")
    agreement = {
        "ascend_commit": source_revision(Path(__file__).resolve().parents[3]),
        "vllm_commit": source_revision(Path(lwd.__file__).resolve().parents[3]),
        "transport_sha256": source_digest(Path(__file__).parent),
        "config_sha256": source_digest(Path(lwd.__file__).parent),
        "topology": topology,
        "connection": asdict(wire.key),
        "plan": asdict(plan),
    }
    logger.info("[LWD][wire-test] config=%s device=%s", json.dumps(asdict(config)), device)
    logger.info("[LWD][wire-test] agreement=%s", json.dumps(agreement, sort_keys=True))
    wire.join_async()
    wire.ensure_joined()
    # Reuse the rendezvous store for test agreement and final verification;
    # no inference control-plane sockets or per-packet notifications are used.
    exchange_check(wire._store, config.role, "plan", json.dumps(agreement, sort_keys=True), wire.timeout_s)
    logger.info("[LWD][wire-test] peer code/config/plan matched; starting transfers")
    run_batches(service, wire, plan)
    exchange_check(wire._store, config.role, "verified", str(plan.batches), wire.timeout_s)
    # The cloud's last store operation acknowledges consuming the result.
    # Only the store-owning edge waits, avoiding a shutdown/read race.
    if wire.is_edge:
        wire._store.wait(["lwd_probe/cloud_done"], timedelta(seconds=wire.timeout_s))
    else:
        wire._store.set("lwd_probe/cloud_done", "1")
    service.shutdown(timeout=wire.timeout_s)
    logger.info("[LWD][wire-test] PASS role=%s channel=%s batches=%d", config.role, plan.channel, plan.batches)
    return wire


def run_startup_probe(
    config: LwdConfig, parallel: ParallelConfig, device: torch.device, tp_group: GroupCoordinator
) -> LwdWire | None:
    if config.topology.deployment.scene != "single_instance" or len(config.connections) != 1:
        raise ValueError("[LWD][wire-test] startup test requires single_instance DP=1")
    if any(
        size != 1
        for size in (
            parallel.pipeline_parallel_size,
            parallel.data_parallel_size,
            parallel.prefill_context_parallel_size,
        )
    ):
        raise ValueError("[LWD][wire-test] startup test requires PP=DP=PCP=1")
    if len(config.instance.dp[0].ranks) != tp_group.world_size:
        raise ValueError("[LWD][wire-test] YAML rank count must match the local tensor-parallel size")

    wire = None
    failure = None
    if tp_group.rank_in_group == 0:
        logger.info("[LWD][wire-test] starting before centralized model loading, role=%s", config.role)
        try:
            wire = run_probe(config, device, LwdProbePlan())
        except Exception as exc:
            failure = exc
    # Other TP workers wait here, so none enter model collectives before the
    # endpoint finishes; a failed endpoint stops every local worker.
    error = tp_group.broadcast_object(str(failure) if failure is not None else None, src=0)
    if error is not None:
        raise RuntimeError(f"[LWD][wire-test] startup test failed: {error}") from failure
    if tp_group.rank_in_group == 0:
        logger.info("[LWD][wire-test] continuing centralized model loading")
    return wire
