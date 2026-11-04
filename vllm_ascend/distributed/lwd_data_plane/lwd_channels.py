# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UP and DOWN channels of the LWD edge-cloud data plane.

Owns the physical channels (UP edge->cloud, DOWN cloud->edge):
  * channel HCCL communicators: two dedicated ``new_group`` groups per DP
    connection, over its endpoint pair (``LwdChannelEndpoints``: the first
    edge rank and the first cloud rank);
  * one NPU stream per channel of each DP connection the local rank
    endpoints (channel ops are bridged onto it, keeping the compute stream
    free);
  * init-time warmup of every connection's channels in a fixed global
    order (moves the first-use HCCL rendezvous out of the pipeline, where
    the two sides could otherwise arrive on different channels and
    deadlock).

In a single_instance topology the edge process serves every DP
connection, so its endpoint rank holds the channel pairs of all of them;
a cloud endpoint rank holds exactly its own DP's pair.

Design: docs/design/lwd_data_plane.md in the vLLM repository.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed.lwd import LwdChannelEndpoints, LwdParallelGroupsSpec
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger

from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

_LWD_WARMUP_NUM_ELEMENTS = 8

# (dp_idx, channel) -> (device_group, peer_global_rank), endpoint ranks only
_LWD_CHANNEL_GROUPS: dict[tuple[int, LwdChannelType], tuple[dist.ProcessGroup, int]] = {}
_LWD_CHANNEL_STREAMS: dict[tuple[int, LwdChannelType], torch.npu.Stream] = {}
_LWD_ENDPOINTS: tuple[LwdChannelEndpoints, ...] = ()
_LWD_CHANNELS_INITIALIZED = False


def lwd_init_channels() -> None:
    """Create the UP and DOWN channels of every DP connection. Idempotent.

    Called once by every worker of the shared world after its distributed
    environment is up. Group creation is a world collective, so every rank
    creates the groups of every connection in the order of
    ``LwdParallelGroupsSpec``; only the two endpoint ranks keep them.
    """
    global _LWD_CHANNELS_INITIALIZED, _LWD_ENDPOINTS
    if _LWD_CHANNELS_INITIALIZED:
        return
    lwd_config = get_current_vllm_config().lwd_config
    assert lwd_config is not None, "the data plane requires lwd_config"
    spec = LwdParallelGroupsSpec.from_topology(lwd_config.topology)
    backend = dist.get_backend(get_world_group().device_group)
    my_rank = dist.get_rank()
    for endpoints in spec.channel_endpoints:
        ranks = [endpoints.edge_rank, endpoints.cloud_rank]
        for channel in LwdChannelType:
            # Both channels span the same rank pair; the direction is given
            # by who sends.
            group = dist.new_group(ranks, backend=backend)
            if my_rank in ranks:
                peer = endpoints.cloud_rank if my_rank == endpoints.edge_rank else endpoints.edge_rank
                _LWD_CHANNEL_GROUPS[(endpoints.dp_idx, channel)] = (group, peer)
                _LWD_CHANNEL_STREAMS[(endpoints.dp_idx, channel)] = torch.npu.Stream()
    _LWD_ENDPOINTS = spec.channel_endpoints_of(my_rank)
    # The cloud-internal embeds broadcast does not build its own group: a
    # dedicated new_group delivered nothing under torch_npu in testing
    # (suspected group creation order mismatch), so it uses the framework TP
    # device_group, whose creation order the framework guarantees. The
    # broadcast is issued on the compute stream at injection time, FIFO with
    # the forward TP collectives, so the two cannot interleave.
    _LWD_CHANNELS_INITIALIZED = True
    logger.info(
        "[LWD] data-plane channels created for %s (my_rank=%d, endpoints=%s)",
        spec.channel_endpoints,
        my_rank,
        _LWD_ENDPOINTS,
    )
    # Every connection's two channels, always warmed up: the first-use HCCL
    # rendezvous is moved to init time, where both sides are guaranteed to
    # arrive in the same fixed order.
    lwd_warm_up_channels()


def lwd_warm_up_channels() -> None:
    """Pre-establish every connection's channel P2P links at init time.

    A tiny payload is exchanged on each channel in a fixed global order
    (connection order, UP then DOWN), so both sides rendezvous identically
    regardless of when the first real message is posted. Non-endpoint ranks
    only join the closing world barrier.
    """
    if _LWD_ENDPOINTS:
        my_rank = dist.get_rank()
        for endpoints in _LWD_ENDPOINTS:
            for channel in (LwdChannelType.UP, LwdChannelType.DOWN):
                group, peer = _LWD_CHANNEL_GROUPS[(endpoints.dp_idx, channel)]
                sender = endpoints.edge_rank if channel is LwdChannelType.UP else endpoints.cloud_rank
                payload = torch.zeros(_LWD_WARMUP_NUM_ELEMENTS, dtype=torch.bfloat16, device="npu")
                if my_rank == sender:
                    handle = dist.isend(payload, dst=peer, group=group)
                else:
                    handle = dist.irecv(payload, src=peer, group=group)
                handle.wait()
                logger.info(
                    "[LWD] data-plane warmup done: dp_idx=%d channel=%s my_rank=%d peer=%d",
                    endpoints.dp_idx,
                    channel,
                    my_rank,
                    peer,
                )
    get_world_group().barrier()


def lwd_channels_initialized() -> bool:
    return _LWD_CHANNELS_INITIALIZED


def _lwd_channel(dp_idx: int, channel: LwdChannelType) -> tuple[dist.ProcessGroup, int]:
    try:
        return _LWD_CHANNEL_GROUPS[(dp_idx, channel)]
    except KeyError:
        raise RuntimeError(
            f"[LWD] rank {dist.get_rank()} holds no {channel.value} channel of "
            f"dp_idx {dp_idx}; only the endpoint ranks of that DP connection do"
        ) from None


def lwd_get_channel_device_group(channel: LwdChannelType, dp_idx: int = 0) -> dist.ProcessGroup:
    group, _ = _lwd_channel(dp_idx, channel)
    return group


def lwd_get_channel_peer(channel: LwdChannelType, dp_idx: int = 0) -> int:
    """Global rank of this process's P2P peer on the DP connection's channel."""
    _, peer = _lwd_channel(dp_idx, channel)
    return peer


def lwd_get_channel_stream(channel: LwdChannelType, dp_idx: int = 0) -> torch.npu.Stream:
    _lwd_channel(dp_idx, channel)
    return _LWD_CHANNEL_STREAMS[(dp_idx, channel)]


def lwd_destroy_channels() -> None:
    global _LWD_CHANNELS_INITIALIZED, _LWD_ENDPOINTS
    for (dp_idx, channel), (group, _) in list(_LWD_CHANNEL_GROUPS.items()):
        try:
            dist.destroy_process_group(group)
        except Exception:
            logger.warning("[LWD] failed to destroy the data-plane group for dp%d %s", dp_idx, channel.value)
    _LWD_CHANNEL_GROUPS.clear()
    _LWD_CHANNEL_STREAMS.clear()
    _LWD_ENDPOINTS = ()
    _LWD_CHANNELS_INITIALIZED = False
