# SPDX-License-Identifier: Apache-2.0
"""Queue selection of the LWD data plane: device vs host transport.

The transport is chosen once at channel-init time and cached; the request
path must never fall back to the HCCL queue when a host topology fails to
load its UCX queue, because that would surface later as a misleading
"holds no channel" error.
"""

from __future__ import annotations

import sys

import pytest

from vllm_ascend.distributed.lwd_data_plane import lwd_data_plane as plane
from vllm_ascend.distributed.lwd_data_plane.host import lwd_host_channels
from vllm_ascend.distributed.lwd_data_plane.host.lwd_host_transfer_queue import (
    LwdHostTransferQueue,
)
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_queue import LwdTransferQueue

HOST_QUEUE_MODULE = "vllm_ascend.distributed.lwd_data_plane.host.lwd_host_transfer_queue"


def test_device_transport_keeps_the_hccl_queue(monkeypatch) -> None:
    monkeypatch.setattr(lwd_host_channels, "lwd_data_plane_comm", lambda: "device")
    assert plane._queue_class() is LwdTransferQueue


def test_host_transport_selects_the_ucx_queue(monkeypatch) -> None:
    monkeypatch.setattr(lwd_host_channels, "lwd_data_plane_comm", lambda: "host")
    assert plane._queue_class() is LwdHostTransferQueue


def test_host_transport_fails_loudly_when_its_queue_cannot_load(monkeypatch) -> None:
    """No silent downgrade to HCCL: the host topology must raise instead."""
    monkeypatch.setattr(lwd_host_channels, "lwd_data_plane_comm", lambda: "host")
    monkeypatch.setitem(sys.modules, HOST_QUEUE_MODULE, None)
    with pytest.raises(ImportError, match="lwd_host_transfer_queue"):
        plane._queue_class()
