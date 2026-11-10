# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host (RoCE) backend of the LWD data plane.

The package's contract is 'host RoCE transport' — the vocabulary of the
topology's ``data_plane_comm: host``. UCX is the current implementation
detail: the transport core lives in the vllm repo (``vllm/v1/lwd_comm/data_comm``,
pure Python ctypes over libucp), and this package assembles it for NPU:

  * ``lwd_host_channels`` — identity-driven bootstrap + registry (the public
    ``lwd_*_host_*`` entry points);
  * ``lwd_host_transfer_queue`` — ``LwdHostTransferQueue``, the
    ``LwdTransferQueue`` implementation the data-plane factory selects
    for host comm;
  * ``ucx_io`` / ``ucx_block_policy`` — UCX-specific internals (link IO
    + D2H/H2D orchestration, block auto-sizing).
"""

from vllm_ascend.distributed.lwd_data_plane.host.lwd_host_channels import (
    lwd_data_plane_comm,
    lwd_data_plane_is_host,
    lwd_destroy_host_channels,
    lwd_get_host_io,
    lwd_host_channels_initialized,
    lwd_init_host_channels,
    lwd_warm_up_host_channels,
)
from vllm_ascend.distributed.lwd_data_plane.host.lwd_host_transfer_queue import (
    LwdHostTransferQueue,
)

__all__ = [
    "LwdHostTransferQueue",
    "lwd_data_plane_comm",
    "lwd_data_plane_is_host",
    "lwd_destroy_host_channels",
    "lwd_get_host_io",
    "lwd_init_host_channels",
    "lwd_host_channels_initialized",
    "lwd_warm_up_host_channels",
]
