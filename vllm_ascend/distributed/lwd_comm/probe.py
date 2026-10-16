# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise the persistent Worker data plane after the API server is ready."""

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
import torch.distributed as dist
import vllm
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger

from .types import LwdChannelType

if TYPE_CHECKING:
    from torch.distributed import Store

    from vllm_ascend.worker.lwd_worker import LwdWorker

    from .future import LwdCommFuture


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


def check_local_result(error: str | None) -> None:
    tp = get_tp_group()
    errors = [error]
    if tp.world_size > 1:
        errors = [None] * tp.world_size
        dist.all_gather_object(errors, error, group=tp.cpu_group)
    if any(item is not None for item in errors):
        raise ValueError(f"[LWD][wire-test] local TP verification failed: {errors}")


def run_batches(worker: LwdWorker, plan: LwdProbePlan, start_seqno: int = 0) -> None:
    wire = worker.lwd_wire
    channel = LwdChannelType.UP if plan.channel == "up" else LwdChannelType.DOWN
    sending = wire.is_edge == (channel is LwdChannelType.UP)
    pending: deque[tuple[int, int, LwdCommFuture | None]] = deque()

    def complete(item: tuple[int, int, LwdCommFuture | None]) -> None:
        seqno, numel, future = item
        if sending:
            worker.wait_lwd_send(future)
        else:
            tensor = worker.receive_lwd_tensor(future, numel)
            error = None
            try:
                expected = make_payload(seqno, numel).to(dtype=wire.dtype)
                if not torch.equal(tensor.cpu(), expected):
                    error = f"payload mismatch seqno={seqno}"
            except Exception as exc:
                error = str(exc)
            check_local_result(error)
        error = None
        if wire.member:
            logger.info(
                "[LWD][wire-test] %s seqno=%d num_elements=%d %s",
                "send" if sending else "recv",
                seqno,
                numel,
                "complete" if sending else "verified",
            )
            try:
                worker.comm_service.poll_completions()
            except Exception as exc:
                error = str(exc)
        worker.lwd_sync_error(error)

    for offset in range(plan.batches):
        seqno, numel = start_seqno + offset, plan.num_elements(offset)
        future, error = None, None
        try:
            if sending:
                tensor = make_payload(seqno, numel).to(device=wire.device, dtype=wire.dtype) if wire.member else None
                future = worker.send_lwd_tensor(tensor, channel, seqno)
            else:
                future = worker.submit_lwd_recv(numel, channel, seqno)
        except Exception as exc:
            error = str(exc)
        worker.lwd_sync_error(error)
        pending.append((seqno, numel, future))
        if len(pending) >= plan.window:
            complete(pending.popleft())
    while pending:
        complete(pending.popleft())


def run_probe(worker: LwdWorker, plan: LwdProbePlan) -> dict[str, str | int]:
    config, wire = worker.vllm_config.lwd_config, worker.lwd_wire
    channel = LwdChannelType.UP if plan.channel == "up" else LwdChannelType.DOWN
    round_id, start_seqno = worker._lwd_probe_round, worker._lwd_probe_next[channel]
    if start_seqno + plan.batches > 256**_SEQUENCE_BYTES:
        raise ValueError("[LWD][wire-test] sequence exceeds payload header capacity")
    topology = asdict(config.topology)
    topology.pop("digest")
    vllm_root = Path(vllm.__file__).resolve().parents[1]
    agreement = {
        "ascend_commit": source_revision(Path(__file__).resolve().parents[3]),
        "vllm_commit": source_revision(vllm_root),
        "transport_sha256": source_digest(Path(__file__).parent),
        "worker_sha256": sha256(Path(__file__).parents[2].joinpath("worker/lwd_worker.py").read_bytes()).hexdigest(),
        "config_sha256": source_digest(vllm_root / "vllm/config/lwd"),
        "topology": topology,
        "connection": asdict(wire.key),
        "dtype": str(wire.dtype),
        "plan": asdict(plan),
        "start_seqno": start_seqno,
    }
    store, error = None, None
    if wire.member:
        try:
            wire.ensure_joined()
            # Per-round keys prevent an earlier PASS from satisfying a repeated test.
            store = dist.PrefixStore(f"lwd_probe_round/{round_id}", wire._store)
            exchange_check(store, config.role, "plan", json.dumps(agreement, sort_keys=True), wire.timeout_s)
            logger.info(
                "[LWD][wire-test] peer code/config/plan matched; starting round=%d channel=%s", round_id, plan.channel
            )
        except Exception as exc:
            error = str(exc)
    worker.lwd_sync_error(error)
    run_batches(worker, plan, start_seqno)
    error = None
    if wire.member:
        try:
            exchange_check(store, config.role, "verified", str(plan.batches), wire.timeout_s)
        except Exception as exc:
            error = str(exc)
    worker.lwd_sync_error(error)
    worker._lwd_probe_next[channel] += plan.batches
    worker._lwd_probe_round += 1
    result = {
        "status": "PASS",
        "role": config.role,
        "channel": plan.channel,
        "batches": plan.batches,
        "round": round_id,
        "start_seqno": start_seqno,
        "tp_rank": get_tp_group().rank_in_group,
    }
    logger.info(
        "[LWD][wire-test] PASS role=%s channel=%s batches=%d round=%d tp_rank=%d; transport remains open",
        config.role,
        plan.channel,
        plan.batches,
        round_id,
        result["tp_rank"],
    )
    return result
