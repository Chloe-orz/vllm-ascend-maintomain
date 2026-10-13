# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Requests and results for ordered UP/DOWN communication."""

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch
    from vllm.config.lwd.topology import LwdConnectionKey


class LwdChannelType(enum.Enum):
    # Directions share one domain but keep separate streams and sequence numbers.
    UP = "lwd_up"
    DOWN = "lwd_down"


class LwdCommStatus(enum.Enum):
    PENDING = "pending"
    OK = "ok"
    ERROR = "error"


@dataclass
class LwdCommRequest:
    channel: LwdChannelType
    op: Literal["send", "recv"]
    num_elements: int
    tensor: "torch.Tensor | None" = None
    seqno: int | None = None
    src_dst: int | None = None
    connection_key: "LwdConnectionKey | None" = None


@dataclass
class LwdCommResult:
    status: LwdCommStatus
    tensor: "torch.Tensor | None" = None
    error: BaseException | None = None
