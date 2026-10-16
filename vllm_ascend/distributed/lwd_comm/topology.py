# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Map logical YAML ranks to independent edge/cloud transport endpoints."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config.lwd import LwdConfig
    from vllm.config.lwd.topology import LwdConnectionKey, LwdTopology
    from vllm.v1.lwd.types import LwdBatch


@dataclass(frozen=True)
class LwdWireConnection:
    key: "LwdConnectionKey"
    edge_ranks: tuple[int, ...]
    cloud_ranks: tuple[int, ...]

    @property
    def edge_rank(self) -> int:
        return self.edge_ranks[0]

    @property
    def cloud_leader_rank(self) -> int:
        return self.cloud_ranks[0]

    @property
    def endpoint_ranks(self) -> tuple[int, int]:
        return self.edge_rank, self.cloud_leader_rank

    def peer_for(self, rank: int) -> int:
        if rank == self.edge_rank:
            return self.cloud_leader_rank
        if rank == self.cloud_leader_rank:
            return self.edge_rank
        raise ValueError(f"[LWD] rank={rank} is not a P2P endpoint of {self.key}")

    def includes(self, rank: int) -> bool:
        return rank in self.edge_ranks or rank in self.cloud_ranks


def build_lwd_wire_plan(topology: "LwdTopology") -> tuple[LwdWireConnection, ...]:
    """Keep separate links even when they share physical endpoint ranks."""
    return tuple(
        LwdWireConnection(link.key, link.edge.ranks, link.cloud.ranks)
        for edge in sorted(topology.edges, key=lambda item: item.id)
        for link in sorted(
            topology.connections("edge", edge.id),
            key=lambda item: (item.cloud_id, item.dp_idx),
        )
    )


def bind_lwd_worker_connections(
    config: "LwdConfig", tp_rank: int, dp_idx: int = 0
) -> dict["LwdConnectionKey", LwdWireConnection]:
    """Bind using the worker's local TP rank, never a global torch rank."""
    bindings = {}
    for link in config.connections:
        if link.dp_idx != dp_idx:
            continue
        ranks = link.edge.ranks if config.role == "edge" else link.cloud.ranks
        if not 0 <= tp_rank < len(ranks):
            raise ValueError("[LWD] TP rank is outside the selected YAML allocation")
        bindings[link.key] = LwdWireConnection(link.key, link.edge.ranks, link.cloud.ranks)
    if not bindings:
        raise ValueError("[LWD] Worker has no connection for its instance/DP")
    return bindings


def resolve_lwd_batch_connection(
    batch: "LwdBatch", bindings: dict["LwdConnectionKey", LwdWireConnection]
) -> "LwdConnectionKey":
    """Require the scheduler's explicit identity before posting any tensor."""
    key = batch.connection_key
    if key is None or key not in bindings:
        raise ValueError(f"[LWD] Batch connection {key!r} is not bound to this worker")
    return key


def select_lwd_wire_connection(
    plan: tuple[LwdWireConnection, ...],
    rank: int,
    key: "LwdConnectionKey | None" = None,
) -> LwdWireConnection:
    candidates = [item for item in plan if item.includes(rank) and (key is None or item.key == key)]
    if len(candidates) != 1:
        raise ValueError(f"[LWD] rank={rank} connection_key={key}: expected one connection, found {len(candidates)}")
    return candidates[0]
