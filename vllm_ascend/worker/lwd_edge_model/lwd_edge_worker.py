#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""LWD (layerwise disaggregated) prefill_only mode edge worker.

Dispatcher only: the edge data-plane compute (embed/unembed forward,
MM cache, tid2eid tables) lives on LwdEdgeModelRunner — same
worker↔runner division as native. The native per-step runner pipeline
stays dormant on the edge (LWD hijacks execute_model at the worker).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.v1.core.sched.output import LwdBatchType
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.worker.lwd_base_model.lwd_worker_base import LwdWorkerBase
from vllm_ascend.worker.lwd_edge_model.lwd_edge_model_runner import (
    LwdEdgeModelRunner,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class LwdEdgeWorker(LwdWorkerBase):
    """LWD edge worker: batch dispatch only (embed/unembed on the runner)."""

    def init_device(self) -> None:
        # comm_service and group setup belong to the base init_device; see
        # LwdWorkerBase._init_lwd_duplex_channels。
        super().init_device()
        # Swap in the edge runner (same pattern as the cloud side): embed/unembed compute lives in the runner.
        self.model_runner = LwdEdgeModelRunner(
            self.vllm_config, self.device, worker=self,
        )
        self._init_lwd_duplex_channels()
        logger.info(
            "[lwd-edge] worker ready: duplex channels (UP/DOWN) initialized "
            "on global rank=%d",
            self.rank,
        )

    def compile_or_warm_up_model(self) -> "CompilationTimes":
        # The LWD edge's runtime forward is hijacked by the LWD flow
        # (execute_model only dispatches to the runner's embed/unembed
        # direct calls), so model-level warmup and cudagraph capture are
        # never used; and under the 0-layer topology a full forward would
        # fail unpacking at the final norm. Skip entirely.
        from vllm.v1.worker.worker_base import CompilationTimes

        logger.info("[lwd-edge] skip model warmup/capture (forward is hijacked by LWD)")
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> "ModelRunnerOutput | None":
        lwd_batch = scheduler_output.lwd_batch
        # Multimodal cache lifecycle: when a request finishes (abort
        # included) drop its mm feature/encoder-output caches to prevent
        # cross-request leaks (the healthy path already removed them once
        # the chunk window passed all mm segments; this is the backstop).
        if scheduler_output.finished_req_ids:
            self.model_runner.mm_cache.release_finished(
                scheduler_output.finished_req_ids
            )
        if lwd_batch is None:
            logger.debug("[lwd-edge] step carries no LWD batch; nothing to do")
            return None

        batch_meta = lwd_batch.batch_meta
        if lwd_batch.batch_type == LwdBatchType.LWD_EMBED:
            logger.info(
                "[Lwd][edge-worker] EMBED seqno=%d reqs=%d tokens=%d "
                "has_mrope=%s",
                lwd_batch.seqno,
                len(batch_meta.req_ids),
                sum(len(token_ids) for token_ids in batch_meta.token_ids),
                batch_meta.has_mrope,
            )
            return self.model_runner.execute_lwd_embed(
                lwd_batch.seqno, batch_meta, scheduler_output
            )
        if lwd_batch.batch_type == LwdBatchType.LWD_UNEMBED:
            logger.info(
                "[Lwd][edge-worker] UNEMBED seqno=%d reqs=%d accepted=%d "
                "num_elements=%d",
                lwd_batch.seqno,
                len(batch_meta.req_ids),
                sum(batch_meta.num_accept_tokens),
                batch_meta.recv_num_elements,
            )
            result = self.model_runner.execute_lwd_unembed(
                lwd_batch.seqno, batch_meta
            )
            # [Lwd][eos-trace] Point C: log rows that declared accepted>0
            # yet received no token backfill (paired with edge-side point B
            # by down_seqno; accepted=0 rows legitimately produce no token
            # and are not counted as missing)
            LwdControlDiag.trace_worker_missing_rows(
                lwd_batch.seqno,
                list(batch_meta.req_ids),
                list(batch_meta.num_accept_tokens),
                list(getattr(result, "req_ids", None) or []),
                [list(t) for t in (getattr(result, "sampled_token_ids", None) or [])],
            )
            return result

        logger.debug(
            "[lwd-edge] unknown LWD batch type %r; nothing to do",
            lwd_batch.batch_type,
        )
        return None
