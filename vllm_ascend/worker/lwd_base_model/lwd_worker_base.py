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
        """Override of the native entry (worker.py): ascend-side parallel
        groups are built per the LWD layout.

        The vllm-side grouping is done by the LWD branch of
        parallel_state.initialize_model_parallel; the ascend-side native
        init_ascend_model_parallel splits ranks on a uniform (dp, pp,
        pcp, tp) grid, which cannot express the asymmetric edge-cloud
        topology — hence init_lwd_ascend_model_parallel replaces it
        (building the MC2 etc. groups and injecting them). Every other
        step matches the native worker.py verbatim."""
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
