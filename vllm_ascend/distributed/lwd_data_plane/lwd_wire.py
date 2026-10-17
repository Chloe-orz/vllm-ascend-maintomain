# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only duplex data-plane wire layer.

Owns the two physical channels (UP edge->cloud, DOWN cloud->edge):
  * channel HCCL communicators: two dedicated ``new_group`` groups over
    the edge/cloud endpoint pair (edge rank 0 and the first cloud rank);
  * one NPU stream per channel (wire ops are bridged onto it, keeping
    the compute stream free);
  * init-time warmup of both channels in both directions in a fixed
    global order (moves the first-use HCCL rendezvous out of the
    pipeline, where the two sides could otherwise arrive on different
    channels and deadlock).

Only prefill_only mode builds these groups; other modes never reach
here (gated by ``LwdConfig.is_prefill_only``).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger

from vllm_ascend.distributed.lwd_data_plane.types import LwdChannelType

# channel -> (device_group, peer_global_rank)
_LWD_CHANNEL_GROUPS: dict[LwdChannelType, tuple[dist.ProcessGroup, int]] = {}
_LWD_CHANNEL_STREAMS: dict[LwdChannelType, torch.npu.Stream] = {}
_LWD_ENDPOINTS: tuple[int, int] | None = None  # (edge_global_rank, cloud_global_rank)
_INITIALIZED = False


def init_lwd_duplex_channels() -> None:
    """Create the two duplex channels for the edge/cloud endpoint pair.

    Called once per worker process in prefill_only mode. Idempotent.

    Ranks follow the contiguous edge-first layout derived from
    ``--edge-npu-count`` / ``--cloud-npu-count``: edge [0, E), cloud
    [E, E + C), endpoints (0, E). Only the two endpoints ever touch the
    channels; the other ranks join the group creation (a world
    collective) with peer=-1.
    """
    global _INITIALIZED, _LWD_ENDPOINTS
    if _INITIALIZED:
        return
    from vllm.config import get_current_vllm_config

    lwd_cfg = get_current_vllm_config().parallel_config.lwd_config
    if lwd_cfg.edge_npu_count <= 0 or lwd_cfg.cloud_npu_count <= 0:
        raise RuntimeError(
            "[Lwd] prefill_only duplex channels require positive "
            "--edge-npu-count and --cloud-npu-count, got "
            f"edge={lwd_cfg.edge_npu_count}, cloud={lwd_cfg.cloud_npu_count}"
        )
    edge_rank, cloud_rank = 0, lwd_cfg.edge_npu_count
    ranks = [edge_rank, cloud_rank]
    backend = dist.get_backend(get_world_group().device_group)
    my_rank = dist.get_rank()
    for channel in LwdChannelType:
        # Both channels span the same rank pair; direction is given by
        # who sends.  new_group is a world collective — every rank must
        # participate in every creation, in the same order.
        group = dist.new_group(ranks, backend=backend)
        # P2P peer = the other rank of the pair, defined only on the two
        # endpoint ranks; other ranks (e.g. extra cloud TP ranks) never
        # touch the channels.
        if my_rank == edge_rank:
            peer = cloud_rank
        elif my_rank == cloud_rank:
            peer = edge_rank
        else:
            peer = -1
        _LWD_CHANNEL_GROUPS[channel] = (group, peer)
        _LWD_CHANNEL_STREAMS[channel] = torch.npu.Stream()
    _LWD_ENDPOINTS = (edge_rank, cloud_rank)
    # embeds 组内广播不再自建通信域(自建 new_group 在 torch_npu 下实测
    # 零交付,疑建组顺序错位),改用框架 TP device_group(demo 同款,
    # 建组顺序由框架保证);广播在注入时刻于计算流上发起,与前向 TP
    # 集合同流 FIFO,天然不串台。
    _INITIALIZED = True
    logger.info(
        "[lwd-wire] duplex channels created over ranks=%s (my_rank=%d)",
        ranks,
        my_rank,
    )
    # Exactly two channels, always warmed up: the first-use HCCL
    # rendezvous is moved to init time, where both sides are guaranteed
    # to arrive in the same fixed order.
    warmup_lwd_duplex_channels()


def warmup_lwd_duplex_channels() -> None:
    """Pre-establish both channels' P2P links at init time.

    A tiny payload is exchanged on each channel in a fixed global order
    (UP then DOWN), so both sides rendezvous identically regardless of
    when the first real message is posted.
    """
    my_rank = dist.get_rank()
    assert _LWD_ENDPOINTS is not None
    edge_rank, cloud_rank = _LWD_ENDPOINTS
    for channel in (LwdChannelType.UP, LwdChannelType.DOWN):
        group, peer = _LWD_CHANNEL_GROUPS[channel]
        am_sender = (channel is LwdChannelType.UP and my_rank == edge_rank) or (
            channel is LwdChannelType.DOWN and my_rank == cloud_rank
        )
        am_receiver = (channel is LwdChannelType.UP and my_rank == cloud_rank) or (
            channel is LwdChannelType.DOWN and my_rank == edge_rank
        )
        # Non-endpoint ranks participate in new_group/barrier (world
        # collectives) but skip the P2P ops entirely.
        if not (am_sender or am_receiver):
            continue
        payload = torch.zeros(8, dtype=torch.bfloat16, device="npu")
        if am_sender:
            logger.info(
                "[lwd-warmup] SEND post channel=%s my_rank=%d peer=%d group_ranks=%s",
                channel,
                my_rank,
                peer,
                dist.get_process_group_ranks(group),
            )
            handle = dist.isend(payload, dst=peer, group=group)
        else:
            logger.info(
                "[lwd-warmup] RECV post channel=%s my_rank=%d src=%d group_ranks=%s",
                channel,
                my_rank,
                peer,
                dist.get_process_group_ranks(group),
            )
            handle = dist.irecv(payload, src=peer, group=group)
        handle.wait()
        logger.info("[lwd-warmup] DONE channel=%s my_rank=%d", channel, my_rank)
    get_world_group().barrier()
    logger.info("[lwd-wire] duplex channels warmed up (UP + DOWN)")


def lwd_channels_initialized() -> bool:
    return _INITIALIZED


def get_lwd_channel_device_group(channel: LwdChannelType) -> dist.ProcessGroup:
    group, _ = _LWD_CHANNEL_GROUPS[channel]
    return group


def get_lwd_channel_peer(channel: LwdChannelType) -> int:
    """Global rank of this process's P2P peer on the channel."""
    _, peer = _LWD_CHANNEL_GROUPS[channel]
    return peer


def get_lwd_channel_stream(channel: LwdChannelType) -> torch.npu.Stream:
    return _LWD_CHANNEL_STREAMS[channel]


def destroy_lwd_duplex_channels() -> None:
    global _INITIALIZED, _LWD_ENDPOINTS
    for channel, (group, _) in list(_LWD_CHANNEL_GROUPS.items()):
        try:
            dist.destroy_process_group(group)
        except Exception:
            logger.warning("[lwd-wire] failed to destroy group for %s", channel.value)
    _LWD_CHANNEL_GROUPS.clear()
    _LWD_CHANNEL_STREAMS.clear()
    _LWD_ENDPOINTS = None
    _INITIALIZED = False
