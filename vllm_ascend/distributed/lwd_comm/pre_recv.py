# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Early-receive hints and completion watermarks from source 7bfc919a.

Hints use an independent queue so a running model step cannot delay receives.
Queue handles are passed by initialization RPC, without environment variables.
"""

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from .types import LwdChannelType

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    from vllm.v1.executor.abstract import Executor

LWD_SIDEBAND_CHUNK_BYTES = 1024
LWD_SIDEBAND_CHUNKS = 64


class LwdRecvHint(TypedDict):
    channel: LwdChannelType
    seqno: int
    num_tokens: int
    aux_rows: int


@dataclass(frozen=True)
class LwdRecvFailure:
    message: str


def make_recv_hint(seqno: int, num_tokens: int, aux_rows: int = 0) -> LwdRecvHint:
    hint = LwdRecvHint(channel=LwdChannelType.UP, seqno=seqno, num_tokens=num_tokens, aux_rows=aux_rows)
    if hint_fields(hint) is None:
        raise ValueError("[LWD] invalid early-receive hint")
    return hint


def hint_fields(hint: LwdRecvHint) -> tuple[int, int, int] | None:
    values = (hint.get("seqno"), hint.get("num_tokens"), hint.get("aux_rows", 0))
    if hint.get("channel") != LwdChannelType.UP or any(type(value) is not int for value in values):
        return None
    seqno, num_tokens, aux_rows = values
    if seqno < 0 or num_tokens <= 0 or aux_rows != 0:
        return None
    return seqno, num_tokens, aux_rows


def create_hint_mq() -> "MessageQueue":
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue(1, 1, max_chunk_bytes=LWD_SIDEBAND_CHUNK_BYTES, max_chunks=LWD_SIDEBAND_CHUNKS)


def create_report_mq() -> "MessageQueue":
    return create_hint_mq()


def connect_recv_queues(executor: "Executor") -> tuple["MessageQueue", "MessageQueue"]:
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    hints = create_hint_mq()
    handles = executor.collective_rpc("lwd_start_early_recv", args=(hints.export_handle(),))
    endpoints = [handle for handle in handles if handle is not None]
    if len(endpoints) != 1:
        hints.shutdown()
        raise RuntimeError("[LWD] early receive requires exactly one local TP endpoint")
    done = MessageQueue.create_from_handle(endpoints[0], 0)
    hints.wait_until_ready()
    done.wait_until_ready()
    return hints, done


_hint_sender: Callable[[LwdRecvHint], None] | None = None
_watermarks: dict[LwdChannelType, int] = {}
_lock = threading.Lock()


def register_hint_sender(sender: Callable[[LwdRecvHint], None]) -> None:
    global _hint_sender
    with _lock:
        _watermarks.clear()
        _hint_sender = sender


def post_irecv_hint(hint: LwdRecvHint) -> None:
    if _hint_sender is None:
        raise RuntimeError("[LWD] early-receive sender is not initialized")
    _hint_sender(hint)


def record_irecv_completions(
    items: Iterable[tuple[LwdChannelType, int] | LwdRecvFailure],
) -> list[tuple[LwdChannelType, int]]:
    advanced = []
    with _lock:
        for item in items:
            if isinstance(item, LwdRecvFailure):
                raise RuntimeError(f"[LWD] receive worker failed: {item.message}")
            channel, seqno = item
            # Each channel records completion events on one ordered NPU stream.
            if seqno > _watermarks.get(channel, -1):
                _watermarks[channel] = seqno
                advanced.append((channel, seqno))
    return advanced


def watermark(channel: LwdChannelType) -> int:
    with _lock:
        return _watermarks.get(channel, -1)


def is_irecv_complete(channel: LwdChannelType, seqno: int) -> bool:
    return seqno <= watermark(channel)
