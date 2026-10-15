# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""One independent HCCL domain shared by the UP and DOWN directions."""

from __future__ import annotations

import threading
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from vllm.logger import logger

from .topology import bind_lwd_worker_connections
from .types import LwdChannelType

if TYPE_CHECKING:
    from vllm.config.lwd import LwdConfig

    from .lwd_hccl import LwdHcclLibrary

LWD_HCCL_BUFFER_SIZE_MB = 200
LWD_WARMUP_ELEMENTS = 8
LWD_WIRE_STORE_PORT_DEFAULT = 29600
LWD_WIRE_TIMEOUT_S_DEFAULT = 1200.0


class LwdWire:
    def __init__(
        self, config: LwdConfig, tp_rank: int, device: torch.device, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        if config.topology.deployment.scene != "single_instance" or len(config.connections) != 1:
            raise ValueError("[LWD] wire runtime currently requires single_instance DP=1")
        bindings = bind_lwd_worker_connections(config, tp_rank)
        self.connection = next(iter(bindings.values()))
        self.key = self.connection.key
        self.is_edge = config.role == "edge"
        ranks = self.connection.edge_ranks if self.is_edge else self.connection.cloud_ranks
        self.logical_rank = ranks[tp_rank]
        self.member = self.logical_rank in self.connection.endpoint_ranks
        self.domain_rank = self.connection.domain_rank_for(self.logical_rank) if self.member else None
        self.device, self.dtype = device, dtype
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("[LWD] unsupported hidden-state dtype")
        # Keep the no-PP bootstrap separate from the YAML control-plane socket.
        self.store_host = config.connections[0].edge.addr
        self.store_port = LWD_WIRE_STORE_PORT_DEFAULT
        self.timeout_s = LWD_WIRE_TIMEOUT_S_DEFAULT
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._ready = False
        self._closed = False
        self._pg: Any = None
        self._store: Any = None
        self._comm: Any = None
        self._library: LwdHcclLibrary | None = None
        self._streams: dict[LwdChannelType, Any] = {}
        # Retain self-check buffers even after a device timeout until process exit.
        self._warmup_buffers: list[torch.Tensor] = []

    def join_async(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("[LWD] wire is closed")
            if not self.member or self._thread is not None:
                return
            self._thread = threading.Thread(target=self._join_and_check, name="lwd-wire-join", daemon=True)
            self._thread.start()
            threading.Thread(target=self._watchdog, name="lwd-wire-watchdog", daemon=True).start()

    def _fail(self, error: Exception) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
            self._ready = False
            self._done.set()

    def _join_and_check(self) -> None:
        try:
            from torch_npu._C._distributed_c10d import ProcessGroupHCCL

            from .lwd_hccl import LwdHcclLibrary, hcclComm_t

            # A fresh thread needs an explicit ACL context, even if the process
            # already selected this NPU on its compute thread.
            torch.npu.set_device(self.device)
            self._library = LwdHcclLibrary()
            logger.info(
                "[LWD][wire] joining connection=%s logical_rank=%s domain_rank=%s device=%s store=%s:%s",
                self.key,
                self.logical_rank,
                self.domain_rank,
                self.device,
                self.store_host,
                self.store_port,
            )
            self._store = dist.TCPStore(
                self.store_host,
                self.store_port,
                2,
                self.is_edge,
                timeout=timedelta(seconds=self.timeout_s),
                wait_for_workers=False,
            )
            prefix = f"lwd_{self.key.edge_id}_{self.key.cloud_id}_{self.key.dp_idx}"
            store = dist.PrefixStore(prefix, self._store)
            options = ProcessGroupHCCL.Options()
            options.hccl_config = {"group_name": prefix, "hccl_buffer_size": LWD_HCCL_BUFFER_SIZE_MB}
            # No init_process_group/new_group: native model TP/PP groups stay local.
            self._pg = ProcessGroupHCCL(store, self.domain_rank, 2, options)
            self._comm = hcclComm_t(self._pg.get_hccl_comm(self.device.index))
            self._streams = {channel: torch.npu.Stream() for channel in LwdChannelType}
            self._self_check()
            with self._lock:
                if self._closed or self._error is not None:
                    return
                self._ready = True
                self._done.set()
            self._warmup_buffers.clear()
            logger.info("[LWD][wire] ready connection=%s domain_size=2 UP/DOWN self-check passed", self.key)
        except Exception as exc:
            self._fail(exc)
            logger.exception("[LWD][wire] join failed connection=%s", self.key)

    def _self_check(self) -> None:
        assert self._library is not None and self.domain_rank is not None
        operations = ("send", "recv") if self.is_edge else ("recv", "send")
        for channel, op in zip(LwdChannelType, operations):
            peer = 1 - self.domain_rank
            payload = torch.full((LWD_WARMUP_ELEMENTS,), 1.0 + self.domain_rank, device=self.device, dtype=self.dtype)
            self._warmup_buffers.append(payload)
            stream = self._streams[channel]
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                getattr(self._library, op)(payload, peer, self._comm)
                event = torch.npu.Event()
                event.record()
            deadline = time.monotonic() + self.timeout_s
            while not event.query():
                if self._closed or self._error is not None:
                    raise RuntimeError("[LWD] wire self-check interrupted")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"[LWD] wire {channel.value} self-check timed out")
                time.sleep(0.01)
            if op == "recv" and not torch.all(payload.cpu() == 1.0 + peer).item():
                raise RuntimeError(f"[LWD] wire {channel.value} self-check payload mismatch")

    def _watchdog(self) -> None:
        if not self._done.wait(self.timeout_s):
            error = TimeoutError(f"[LWD] wire join timed out connection={self.key}")
            self._fail(error)
            logger.error("%s", error)

    def wire_ready(self) -> bool:
        return self._ready and not self._closed

    def wire_error(self) -> Exception | None:
        return self._error

    def ensure_joined(self, timeout: float | None = None) -> None:
        if not self.member:
            raise RuntimeError("[LWD] non-endpoint TP worker cannot post wire operations")
        if self._closed:
            raise RuntimeError("[LWD] wire is closed")
        if self._thread is None:
            raise RuntimeError("[LWD] join_async must be called before posting")
        if not self._done.wait(self.timeout_s if timeout is None else timeout):
            raise TimeoutError(f"[LWD] wire not ready connection={self.key}")
        if self._error is not None:
            raise RuntimeError(f"[LWD] wire failed connection={self.key}") from self._error
        if not self.wire_ready():
            raise RuntimeError("[LWD] wire closed during join")

    def get_channel_stream(self, channel: LwdChannelType) -> Any:
        self.ensure_joined()
        return self._streams[channel]

    def p2p_send(self, channel: LwdChannelType, tensor: torch.Tensor, peer_rank: int | None = None) -> None:
        self._p2p("send", channel, tensor, peer_rank)

    def p2p_recv(self, channel: LwdChannelType, tensor: torch.Tensor, peer_rank: int | None = None) -> None:
        self._p2p("recv", channel, tensor, peer_rank)

    def _p2p(self, op: str, channel: LwdChannelType, tensor: torch.Tensor, peer_rank: int | None) -> None:
        self.ensure_joined()
        assert self.domain_rank is not None and self._library is not None
        expected = "send" if (channel == LwdChannelType.UP) == self.is_edge else "recv"
        peer = 1 - self.domain_rank
        if op != expected or (peer_rank is not None and peer_rank != peer):
            raise ValueError("[LWD] P2P direction or domain peer does not match connection")
        getattr(self._library, op)(tensor, peer, self._comm)

    def shutdown(self) -> None:
        self._closed = True
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise TimeoutError("[LWD] native HCCL join is still running; stop the worker process")
        # The PG owns the communicator. Never raw-destroy its HCCL handle.
        # Keep references until process teardown, including timed-out device ops.
        self._ready = False
