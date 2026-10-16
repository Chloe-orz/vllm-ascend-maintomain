# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""PR #63's UP/DOWN groups, bound to one YAML connection per Worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_lwd_bootstrap_world
from vllm.logger import logger

from .topology import bind_lwd_worker_connections
from .types import LwdChannelType

if TYPE_CHECKING:
    from vllm.config.lwd import LwdConfig

LWD_WIRE_TIMEOUT_S_DEFAULT = 1200.0
LWD_WARMUP_ELEMENTS = 8


class LwdWire:
    def __init__(
        self, config: LwdConfig, tp_rank: int, device: torch.device, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        bindings = bind_lwd_worker_connections(config, tp_rank)
        if len(bindings) != 1:
            raise ValueError("[LWD] no-PP wire requires one single-instance connection")
        self.connection = next(iter(bindings.values()))
        self.key = self.connection.key
        self.is_edge = config.role == "edge"
        ranks = self.connection.edge_ranks if self.is_edge else self.connection.cloud_ranks
        self.logical_rank = ranks[tp_rank]
        self.member = self.logical_rank in self.connection.endpoint_ranks
        self.device, self.dtype = device, dtype
        self.timeout_s = LWD_WIRE_TIMEOUT_S_DEFAULT
        self._groups: dict[LwdChannelType, dist.ProcessGroup] = {}
        self._streams: dict[LwdChannelType, Any] = {}
        self._ready = False

    def init_lwd_duplex_channels(self) -> None:
        if self._ready:
            return
        bootstrap = get_lwd_bootstrap_world()
        if dist.get_rank() != self.logical_rank:
            raise RuntimeError("[LWD] distributed rank does not match the YAML allocation")
        backend = dist.get_backend(bootstrap.device_group)
        endpoints = list(self.connection.endpoint_ranks)
        # All bootstrap ranks create both groups in the same order, including
        # non-endpoint cloud TP ranks. PP remains a singleton on every rank.
        for channel in LwdChannelType:
            self._groups[channel] = dist.new_group(endpoints, backend=backend)
            self._streams[channel] = torch.npu.Stream()
        self.warmup_lwd_duplex_channels()
        self._ready = True
        logger.info(
            "[LWD][wire] ready connection=%s endpoints=%s UP/DOWN groups warmed up",
            self.key,
            endpoints,
        )

    def warmup_lwd_duplex_channels(self) -> None:
        for channel in LwdChannelType:
            if not self.member:
                continue
            payload = torch.zeros(LWD_WARMUP_ELEMENTS, dtype=self.dtype, device=self.device)
            group = self._groups[channel]
            peer = self.connection.peer_for(self.logical_rank)
            if (channel is LwdChannelType.UP) == self.is_edge:
                handle = dist.isend(payload, dst=peer, group=group)
            else:
                handle = dist.irecv(payload, src=peer, group=group)
            handle.wait()
            logger.info(
                "[LWD][wire] warmup complete channel=%s rank=%d peer=%d", channel.value, self.logical_rank, peer
            )
        get_lwd_bootstrap_world().barrier()

    def ensure_joined(self) -> None:
        if not self._ready:
            raise RuntimeError("[LWD] initialize and warm UP/DOWN groups before posting")

    def get_lwd_channel_device_group(self, channel: LwdChannelType) -> dist.ProcessGroup:
        self.ensure_joined()
        return self._groups[channel]

    def get_lwd_channel_peer(self, channel: LwdChannelType) -> int:
        self.ensure_joined()
        return self.connection.peer_for(self.logical_rank)

    def get_lwd_channel_stream(self, channel: LwdChannelType) -> Any:
        self.ensure_joined()
        return self._streams[channel]

    def shutdown(self) -> None:
        for channel, group in self._groups.items():
            try:
                dist.destroy_process_group(group)
            except Exception:
                logger.warning("[LWD] failed to destroy group for %s", channel.value)
        self._groups.clear()
        self._streams.clear()
        self._ready = False
