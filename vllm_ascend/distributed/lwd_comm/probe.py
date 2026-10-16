# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise the persistent Worker data plane after the API server is ready."""

from __future__ import annotations

import json
import subprocess
from collections import deque
from contextlib import ExitStack
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
    from .probe_early_recv import LwdProbeEarlyRecv


_SEQUENCE_BYTES = 8


@dataclass(frozen=True)
class LwdProbePlan:
    batches: int = 200
    tokens: tuple[int, ...] = (1, 4, 2, 8)
    hidden_size: int = 5120
    window: int = 4
    channel: str = "up"
    label: str = ""
    early_recv: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.batches <= 256**_SEQUENCE_BYTES or self.window < 1:
            raise ValueError("batches must fit the sequence header; window must be positive")
        if not self.tokens or min(self.tokens) < 1 or self.hidden_size < _SEQUENCE_BYTES:
            raise ValueError("positive token counts and hidden_size >= 8 are required")
        if self.channel not in ("up", "down", "mixed"):
            raise ValueError("channel must be up, down or mixed")
        if self.channel == "mixed" and self.window < 2:
            raise ValueError("mixed traffic must post both directions before waiting")
        if self.early_recv and self.channel != "up":
            raise ValueError("early receive is an UP path")

    @property
    def channels(self) -> tuple[LwdChannelType, ...]:
        if self.channel == "mixed":
            return (LwdChannelType.UP, LwdChannelType.DOWN)
        return (LwdChannelType.UP if self.channel == "up" else LwdChannelType.DOWN,)

    @property
    def name(self) -> str:
        return self.label or self.channel

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


@dataclass
class LwdProbeTransfer:
    channel: LwdChannelType
    seqno: int
    numel: int
    sending: bool
    future: LwdCommFuture | None = None


def post_transfer(worker: LwdWorker, item: LwdProbeTransfer, early: LwdProbeEarlyRecv | None) -> None:
    wire, error = worker.lwd_wire, None
    try:
        if item.sending:
            tensor = (
                make_payload(item.seqno, item.numel).to(device=wire.device, dtype=wire.dtype) if wire.member else None
            )
            item.future = worker.send_lwd_tensor(tensor, item.channel, item.seqno)
        elif early is not None:
            item.future = early.post(item.seqno, item.numel)
        else:
            item.future = worker.submit_lwd_recv(item.numel, item.channel, item.seqno)
    except Exception as exc:
        error = str(exc)
    worker.lwd_sync_error(error)


def complete_transfer(worker: LwdWorker, item: LwdProbeTransfer, early: LwdProbeEarlyRecv | None) -> None:
    wire = worker.lwd_wire
    if item.sending:
        worker.wait_lwd_send(item.future)
    else:
        tensor = worker.receive_lwd_tensor(item.future, item.numel)
        error = None
        try:
            expected = make_payload(item.seqno, item.numel).to(dtype=wire.dtype)
            if not torch.equal(tensor.cpu(), expected):
                error = f"payload mismatch seqno={item.seqno}"
        except Exception as exc:
            error = str(exc)
        check_local_result(error)
    error = None
    if wire.member:
        try:
            if early is not None:
                early.verify_report(item.seqno)
            worker.comm_service.poll_completions()
            logger.info(
                "[LWD][wire-test] %s channel=%s seqno=%d num_elements=%d %s",
                "send" if item.sending else "recv",
                item.channel.value,
                item.seqno,
                item.numel,
                "complete" if item.sending else "verified",
            )
        except Exception as exc:
            error = str(exc)
    worker.lwd_sync_error(error)


def run_batches(
    worker: LwdWorker,
    plan: LwdProbePlan,
    start_seqnos: dict[LwdChannelType, int] | None = None,
    early: LwdProbeEarlyRecv | None = None,
) -> None:
    starts = start_seqnos or {channel: 0 for channel in plan.channels}
    pending: deque[LwdProbeTransfer] = deque()
    for offset in range(plan.batches):
        # Post UP then DOWN without waiting for the preceding direction to finish.
        for channel in plan.channels:
            item = LwdProbeTransfer(
                channel,
                starts[channel] + offset,
                plan.num_elements(offset),
                worker.lwd_wire.is_edge == (channel is LwdChannelType.UP),
            )
            post_transfer(worker, item, early)
            pending.append(item)
            if len(pending) >= plan.window:
                complete_transfer(worker, pending.popleft(), early)
    while pending:
        complete_transfer(worker, pending.popleft(), early)


def run_probe_transfers(worker: LwdWorker, plan: LwdProbePlan, starts: dict[LwdChannelType, int]) -> None:
    from .probe_early_recv import LwdProbeEarlyRecv

    wire = worker.lwd_wire
    error = None
    try:
        with ExitStack() as stack:
            early, setup_error = None, None
            if plan.early_recv and wire.member and not wire.is_edge:
                try:
                    early = stack.enter_context(LwdProbeEarlyRecv(worker))
                except Exception as exc:
                    setup_error = str(exc)
            worker.lwd_sync_error(setup_error)
            run_batches(worker, plan, starts, early)
        if early is not None:
            logger.info(
                "[LWD][wire-test] early receive verified: hints=%d reports=%d retained=0",
                early.preposted,
                early.verified_reports,
            )
    except Exception as exc:
        error = str(exc)
    check_local_result(error)


def run_probe(worker: LwdWorker, plan: LwdProbePlan) -> dict[str, str | int]:
    config, wire = worker.vllm_config.lwd_config, worker.lwd_wire
    round_id = worker._lwd_probe_round
    starts = {channel: worker._lwd_probe_next[channel] for channel in plan.channels}
    if plan.early_recv and (not worker._early_recv_enabled or plan.hidden_size != worker._lwd_up_hidden_size):
        raise ValueError("[LWD][wire-test] early receive requires enabled YAML flag and model hidden size")
    if any(start + plan.batches > 256**_SEQUENCE_BYTES for start in starts.values()):
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
        "start_seqnos": {channel.value: start for channel, start in starts.items()},
    }
    store, error = None, None
    if wire.member:
        try:
            wire.ensure_joined()
            # Per-round keys prevent an earlier PASS from satisfying a repeated test.
            store = dist.PrefixStore(f"lwd_probe_round/{round_id}", wire._store)
            exchange_check(store, config.role, "plan", json.dumps(agreement, sort_keys=True), wire.timeout_s)
            logger.info(
                "[LWD][wire-test] peer code/config/plan matched; case=%s round=%d channel=%s",
                plan.name,
                round_id,
                plan.channel,
            )
        except Exception as exc:
            error = str(exc)
    worker.lwd_sync_error(error)
    run_probe_transfers(worker, plan, starts)
    error = None
    if wire.member:
        try:
            exchange_check(store, config.role, "verified", str(plan.batches), wire.timeout_s)
        except Exception as exc:
            error = str(exc)
    worker.lwd_sync_error(error)
    for channel in plan.channels:
        worker._lwd_probe_next[channel] += plan.batches
    worker._lwd_probe_round += 1
    result = {
        "status": "PASS",
        "role": config.role,
        "channel": plan.channel,
        "batches": plan.batches,
        "round": round_id,
        "start_seqno": starts[plan.channels[0]],
        "case": plan.name,
        "transfers": plan.batches * len(plan.channels),
        "tp_rank": get_tp_group().rank_in_group,
    }
    logger.info(
        "[LWD][wire-test] PASS role=%s channel=%s batches=%d round=%d tp_rank=%d "
        "case=%s transfers=%d; transport remains open",
        config.role,
        plan.channel,
        plan.batches,
        round_id,
        result["tp_rank"],
        plan.name,
        result["transfers"],
    )
    return result
