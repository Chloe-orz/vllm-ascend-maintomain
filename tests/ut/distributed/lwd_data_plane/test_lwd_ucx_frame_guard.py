# SPDX-License-Identifier: Apache-2.0
"""Frame-size fast-fail for the UCX NPU send path."""

from __future__ import annotations

import queue

import pytest

from vllm_ascend.distributed.lwd_data_plane.host.ucx_io import LwdNpuUcxChannelIo


class _Tensor:
    def __init__(self, numel: int, element_size: int = 2) -> None:
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class _Future:
    pass


def _send_io(block_bytes: int, env: str):
    io = object.__new__(LwdNpuUcxChannelIo)
    io.tag = "lwd_up/send"
    io._tr_submit_send = False
    io._block_bytes = block_bytes
    io._block_env = env
    io._sq = queue.Queue()
    return io


def test_send_frame_over_block_raises_before_queueing() -> None:
    io = _send_io(128 << 20, "data_plane_host.up_buffer_size")
    # 20,480 tokens x 4,168 bf16 elements/token = 162.8 MiB.
    tensor = _Tensor(20_480 * 4_168)

    with pytest.raises(RuntimeError, match="data_plane_host.up_buffer_size"):
        io.submit_send(2, tensor, None, None, _Future())

    assert io._sq.empty()


def test_send_frame_within_block_is_queued() -> None:
    io = _send_io(128 << 20, "data_plane_host.up_buffer_size")
    tensor = _Tensor(8_000 * 4_168)

    io.submit_send(2, tensor, None, None, _Future())

    assert io._sq.qsize() == 1
