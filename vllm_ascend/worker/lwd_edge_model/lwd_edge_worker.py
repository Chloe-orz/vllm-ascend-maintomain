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
        # comm_service 与组网由基类 init_device 负责;双工通道时序注释见
        # LwdWorkerBase._init_lwd_duplex_channels。
        super().init_device()
        # 换装边侧 runner(与云侧同款模式):embed/unembed 计算住 runner。
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
        # LWD 边侧运行期 forward 被 LWD 流程劫持(execute_model 只分派给
        # runner 的 embed/unembed 直调),模型级 warmup 与 cudagraph 捕获
        # 均用不到;且 0 层拓扑下 full-forward 会在 final norm 解包失败。
        # 整体跳过。
        from vllm.v1.worker.worker_base import CompilationTimes

        logger.info("[lwd-edge] skip model warmup/capture (forward is hijacked by LWD)")
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> "ModelRunnerOutput | None":
        lwd_batch = scheduler_output.lwd_batch
        # 多模态缓存生命周期:请求完结(含 abort)即摘除其 mm 特征/
        # encoder 输出缓存,防跨请求泄漏(正常路径在 chunk 越过全部 mm
        # 窗口后已提前摘除,此为兜底)。
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
            # [Lwd][eos-trace] C 点:声明 accepted>0 却无 token 回填的行
            # 留痕(与边侧 B 点按 down_seqno 配对;accepted=0 的行不产
            # token 属合法,不计缺)
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
