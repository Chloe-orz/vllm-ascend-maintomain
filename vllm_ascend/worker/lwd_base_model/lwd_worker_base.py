# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared bring-up for the LWD edge/cloud workers (worker layer only).

Both subclasses are selected by ``platform.py`` only for LWD deployments.
They diverge after ``init_device``: the edge hijacks ``execute_model`` for
embed/unembed direct calls (no runner forward, no KV cache), the cloud
delegates to a swapped-in ``LwdCloudModelRunner`` and owns the UP/DOWN
data plane. This base only owns what the two workers used to duplicate
verbatim: the LWD distributed bring-up and the duplex-channel handshake,
plus the shared comm-service handle.
"""

from __future__ import annotations

from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized

from vllm_ascend.batch_invariant import init_batch_invariance
from vllm_ascend.distributed.lwd_comm.lwd_parallel_init import (
    init_lwd_ascend_model_parallel,
)
from vllm_ascend.distributed.lwd_comm.service import get_lwd_comm_service
from vllm_ascend.worker.worker import NPUWorker


class LwdWorkerBase(NPUWorker):
    """Common bring-up shared by :class:`LwdEdgeWorker` / :class:`LwdCloudWorker`."""

    def init_device(self) -> None:
        super().init_device()
        # Shared handle to the LWD comm service singleton; both workers talk
        # to the UP/DOWN channels through it.
        self.comm_service = get_lwd_comm_service()

    def _init_worker_distributed_environment(self) -> None:
        """覆写原生入口(worker.py):ascend 侧并行组按 Lwd 布局构建。

        vllm 侧分组由 parallel_state.initialize_model_parallel 的 Lwd
        分支完成;ascend 侧原生 init_ascend_model_parallel 按 (dp, pp,
        pcp, tp) 均匀网格切分,表达不了非对称边云拓扑,故以
        init_lwd_ascend_model_parallel 替代(构建 MC2 等组后注入)。
        其余步骤与原生 worker.py 保持一致。"""
        init_batch_invariance()
        init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            "hccl",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.prefill_context_parallel_size,
            self.parallel_config.decode_context_parallel_size,
        )
        init_lwd_ascend_model_parallel(self.parallel_config)
        ensure_ec_transfer_initialized(self.vllm_config)

    def _init_lwd_duplex_channels(self) -> None:
        # The duplex channels MUST be built here, never in ``__init__``.
        # ``init_lwd_duplex_channels`` creates the two HCCL process groups
        # (``dist.new_group``) and then warms them up with a two-sided P2P
        # exchange (world barrier plus the edge/cloud isend/irecv pair), so it
        # needs the distributed environment -- world group and PP group --
        # which the distributed init above only builds inside
        # ``init_device``. The executor constructs the worker first and
        # calls ``init_device`` afterwards, so running it from ``__init__``
        # would trip the "group is not initialized" assertions; the two-sided
        # warmup additionally requires both peers to reach it in the same
        # fixed order, which only holds once every rank is past its
        # distributed init. Subclasses call this at exactly the position they
        # previously built the channels inline.
        from vllm_ascend.distributed import lwd_wire

        lwd_wire.init_lwd_duplex_channels()
