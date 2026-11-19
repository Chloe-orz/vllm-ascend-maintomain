# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Print NPU memory directly after LWD allocation and release operations."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest

_MIB = 2**20


def log_lwd_memory(
    stage: str,
    *,
    device: torch.device | str | int | None = None,
    request: LwdCommRequest | None = None,
    role: str = "unknown",
) -> None:
    """Read memory at the call site without adding a device synchronization."""
    channel = request.channel.value if request is not None else "none"
    operation = request.op if request is not None else "none"
    engine = request.engine_index if request is not None else None
    seqno = request.seqno if request is not None else None
    numel = request.num_elements if request is not None else None
    if request is not None:
        is_edge = (request.channel is LwdChannelType.UP) == (operation == "send")
        role = "edge" if is_edge else "cloud"
    started = time.monotonic()
    try:
        rank = dist.get_rank() if dist.is_initialized() else -1
        if device is None:
            device = torch.npu.current_device()
        free_bytes, total_bytes = torch.npu.mem_get_info(device)
        stats = torch.npu.memory_stats(device)
    except Exception as error:
        logger.warning(
            "[LWD][mem] query_error stage=%s role=%s device=%s channel=%s op=%s engine=%s seqno=%s error=%r ts=%.6f",
            stage,
            role,
            device,
            channel,
            operation,
            engine,
            seqno,
            error,
            time.monotonic(),
        )
        return
    logger.info(
        "[LWD][mem] event stage=%s role=%s rank=%d device=%s channel=%s "
        "op=%s engine=%s seqno=%s numel=%s hbm_free_mib=%.3f hbm_total_mib=%.3f "
        "allocated_mib=%.3f reserved_mib=%.3f active_mib=%.3f "
        "peak_allocated_mib=%.3f allocation_retries=%d ooms=%d "
        "query_ms=%.3f wall=%.3f ts=%.6f",
        stage,
        role,
        rank,
        device,
        channel,
        operation,
        engine,
        seqno,
        numel,
        free_bytes / _MIB,
        total_bytes / _MIB,
        stats.get("allocated_bytes.all.current", -_MIB) / _MIB,
        stats.get("reserved_bytes.all.current", -_MIB) / _MIB,
        stats.get("active_bytes.all.current", -_MIB) / _MIB,
        stats.get("allocated_bytes.all.peak", -_MIB) / _MIB,
        stats.get("num_alloc_retries", -1),
        stats.get("num_ooms", -1),
        (time.monotonic() - started) * 1000,
        time.time(),
        time.monotonic(),
    )
