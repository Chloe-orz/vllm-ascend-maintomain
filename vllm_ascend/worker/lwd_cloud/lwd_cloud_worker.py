# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cloud execution and early receive, migrated from 7bfc919a and 76be9e00."""

from __future__ import annotations

import threading
from collections.abc import Iterable

import torch
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.logger import logger
from vllm.v1.core.sched.output import GrammarOutput, LwdBatchType, LwdEmbedBatch, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

# Register device operations before importing the model runner.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.distributed.lwd_comm.future import LwdCommFuture
from vllm_ascend.distributed.lwd_comm.pre_recv import (
    LwdRecvFailure,
    LwdRecvHint,
    create_report_mq,
    hint_fields,
)
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest
from vllm_ascend.worker.lwd_cloud.lwd_cloud_model_runner import LwdCloudModelRunner
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import hash_layer_count, hash_payload_numel
from vllm_ascend.worker.lwd_worker import LwdWorker

# Matches the control scheduler's maximum number of outstanding prompt chunks.
_LWD_EARLY_RECV_MAX_INFLIGHT = 3


class LwdAsyncModelRunnerOutput(AsyncModelRunnerOutput):
    """Keep each carrier alive and await its own D2H event before serialization."""

    def __init__(
        self,
        output: ModelRunnerOutput | AsyncModelRunnerOutput,
        carrier: tuple[torch.Tensor, list[str], int, int],
        ready: torch.npu.Event,
    ) -> None:
        self.output = output
        self.carrier = carrier
        self.ready = ready

    def get_output(self) -> ModelRunnerOutput:
        output = self.output.get_output() if isinstance(self.output, AsyncModelRunnerOutput) else self.output
        self.ready.synchronize()
        # UniProc shares objects with the engine. A copy prevents the next
        # runner step from overwriting metadata still queued at the scheduler.
        pinned, req_ids, numel, seqno = self.carrier
        output.lwd_down_carrier = (pinned.clone(), req_ids, numel, seqno)
        return output


class LwdCloudWorker(LwdWorker):
    model_runner_cls = LwdCloudModelRunner

    def init_device(self) -> None:
        self.model_config.enable_prompt_embeds = True
        super().init_device()
        self.model_runner.worker = self
        self._lwd_down_next_seqno = 0
        self._lwd_up_recv_futures: dict[int, tuple[LwdCommFuture | None, LwdEmbedBatch]] = {}
        self._lwd_up_hidden_size = self.model_config.get_hidden_size()
        self._early_recv_enabled = self.vllm_config.lwd_config.topology.feature_ctrl.enable_early_recv
        self._init_lwd_early_recv_state()

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        self._lwd_up_post_recvs(scheduler_output)
        if scheduler_output.finished_req_ids:
            self._lwd_cloud_flush_finished(scheduler_output.finished_req_ids)
        self.comm_service.poll_completions()
        if not scheduler_output.total_num_scheduled_tokens and scheduler_output.lwd_batch is not None:
            # Every rank consumes this control step; only the endpoint drains P2P.
            for future, _meta in self._lwd_up_recv_futures.values():
                if future is not None:
                    future.wait(timeout=60.0)
            self._lwd_up_recv_futures.clear()
        return super().execute_model(scheduler_output)

    def _lwd_up_post_recvs(self, scheduler_output: SchedulerOutput) -> None:
        batch = scheduler_output.lwd_batch
        if batch is None:
            return
        self.validate_lwd_batch(batch)
        if batch.batch_type is not LwdBatchType.LWD_EMBED or not isinstance(batch.batch_meta, LwdEmbedBatch):
            raise ValueError("[LWD] cloud expects an embedding batch")
        meta = batch.batch_meta
        num_tokens = sum(len(tokens) for tokens in meta.token_ids)
        future = None
        if self.lwd_wire.member:
            if self._early_recv_enabled:
                future = self.get_or_post_early_recv(batch.seqno, num_tokens, 0)
            else:
                future = self.comm_service.submit_recv(self._lwd_build_up_recv_request(batch.seqno, num_tokens, 0))
        self._lwd_up_recv_futures[batch.seqno] = (future, meta)

    def _init_lwd_early_recv_state(self) -> None:
        self._early_recv_futures: dict[int, LwdCommFuture] = {}
        self._early_recv_pending: dict[int, LwdRecvHint] = {}
        self._early_recv_reported: set[int] = set()
        self._early_recv_consumed: set[int] = set()
        self._last_consumed_seqno = -1
        self._early_recv_lock = threading.Lock()
        self._early_recv_stop = threading.Event()
        self._early_recv_error: Exception | None = None
        self._irecv_hint_mq: MessageQueue | None = None
        self._irecv_done_mq: MessageQueue | None = None
        self._lwd_early_recv_thread: threading.Thread | None = None

    def _lwd_build_up_recv_request(self, seqno: int, num_tokens: int, aux_rows: int) -> LwdCommRequest:
        if num_tokens <= 0 or aux_rows:
            raise ValueError("[LWD] UP requires prompt rows; mRoPE travels on the control plane")
        config = self.model_config.hf_config
        return LwdCommRequest(
            connection_key=self.lwd_wire.key,
            channel=LwdChannelType.UP,
            op="recv",
            num_elements=hash_payload_numel(
                num_tokens,
                self._lwd_up_hidden_size,
                hash_layer_count(config),
                getattr(config, "num_experts_per_tok", 0),
            ),
            seqno=seqno,
        )

    def lwd_start_early_recv(self, handle: Handle) -> Handle | None:
        """Exchange queue handles once, before model requests can be scheduled."""
        if not self.lwd_wire.member:
            return None
        if not self._early_recv_enabled or self._lwd_early_recv_thread is not None:
            raise RuntimeError("[LWD] invalid early-receive initialization")
        self._irecv_hint_mq = MessageQueue.create_from_handle(handle, 0)
        self._irecv_done_mq = create_report_mq()
        self._lwd_early_recv_thread = threading.Thread(
            target=self._lwd_early_recv_loop,
            name="lwd-early-recv",
            daemon=True,
        )
        self._lwd_early_recv_thread.start()
        return self._irecv_done_mq.export_handle()

    def _lwd_early_recv_loop(self) -> None:
        hints, done = self._irecv_hint_mq, self._irecv_done_mq
        assert hints is not None and done is not None
        try:
            # NPU device selection is thread-local, including communication threads.
            hints.wait_until_ready()
            done.wait_until_ready()
            torch.npu.set_device(self.device)
            logger.info("[LWD] cloud receive thread ready: connection=%s", self.lwd_wire.key)
            while not self._early_recv_stop.is_set():
                try:
                    hint = hints.dequeue(timeout=0.0005)
                except TimeoutError:
                    pass
                else:
                    self.start_early_irecv(hint)
                self._report_irecv_completions()
                self._early_recv_stop.wait(0.002)
        except Exception as exc:
            if not self._early_recv_stop.is_set():
                self._early_recv_error = exc
                logger.exception("[LWD] cloud early receive failed")
                done.enqueue(LwdRecvFailure(str(exc)))

    def start_early_irecv(self, hint: LwdRecvHint) -> None:
        fields = hint_fields(hint)
        if fields is None:
            raise ValueError(f"[LWD] invalid receive hint: {hint!r}")
        seqno = fields[0]
        with self._early_recv_lock:
            if seqno <= self._last_consumed_seqno or seqno in self._early_recv_futures:
                return
            self._early_recv_pending[seqno] = hint
        self._early_recv_pump()

    def _early_recv_pump(self) -> None:
        with self._early_recv_lock:
            while self._early_recv_pending and len(self._early_recv_futures) < _LWD_EARLY_RECV_MAX_INFLIGHT:
                seqno, hint = next(iter(self._early_recv_pending.items()))
                del self._early_recv_pending[seqno]
                fields = hint_fields(hint)
                assert fields is not None
                self._early_recv_futures[seqno] = self.comm_service.submit_recv(
                    self._lwd_build_up_recv_request(*fields)
                )

    def _report_irecv_completions(self) -> None:
        assert self._irecv_done_mq is not None
        with self._early_recv_lock:
            pending = [
                (seqno, future)
                for seqno, future in self._early_recv_futures.items()
                if seqno not in self._early_recv_reported
            ]
        for seqno, future in pending:
            if not future.done():
                continue
            future.result()  # A failed receive must never advance the scheduler's watermark.
            self._irecv_done_mq.enqueue((LwdChannelType.UP, seqno))
            with self._early_recv_lock:
                self._early_recv_reported.add(seqno)
                self._release_early_recv(seqno)
        self._early_recv_pump()

    def _release_early_recv(self, seqno: int) -> None:
        # Both consumers must finish: the engine needs the completion report,
        # and the runner still needs the tensor after that report unblocks it.
        if seqno in self._early_recv_reported and seqno in self._early_recv_consumed:
            self._early_recv_futures.pop(seqno)
            self._early_recv_reported.remove(seqno)
            self._early_recv_consumed.remove(seqno)

    def get_or_post_early_recv(self, seqno: int, num_tokens: int, aux_rows: int) -> LwdCommFuture:
        if self._early_recv_error is not None:
            raise RuntimeError("[LWD] receive thread failed") from self._early_recv_error
        with self._early_recv_lock:
            future = self._early_recv_futures.get(seqno)
            if future is None:
                self._early_recv_pending.pop(seqno, None)
                future = self.comm_service.submit_recv(self._lwd_build_up_recv_request(seqno, num_tokens, aux_rows))
                self._early_recv_futures[seqno] = future
            self._last_consumed_seqno = max(self._last_consumed_seqno, seqno)
            self._early_recv_consumed.add(seqno)
            self._release_early_recv(seqno)
        self._early_recv_pump()
        return future

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput | None) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        output = self.model_runner.sample_tokens(grammar_output)
        payload = self.model_runner.take_lwd_pending_down_payload()
        if payload is None:
            return output
        if output is None or not self.lwd_wire.member:
            raise RuntimeError("[LWD] DOWN payload requires the cloud output endpoint")
        hidden, pinned, ready, req_ids, _accepted = payload
        seqno = self._lwd_next_down_seqno()
        self.comm_service.submit_send(
            LwdCommRequest(
                connection_key=self.lwd_wire.key,
                channel=LwdChannelType.DOWN,
                op="send",
                num_elements=hidden.numel(),
                tensor=hidden,
                seqno=seqno,
            )
        )
        if seqno == 0:
            logger.info(
                "[LWD] first DOWN submitted: connection=%s rows=%d dtype=%s",
                self.lwd_wire.key,
                hidden.shape[0],
                hidden.dtype,
            )
        return LwdAsyncModelRunnerOutput(output, (pinned, req_ids, hidden.numel(), seqno), ready)

    def _lwd_cloud_flush_finished(self, finished_req_ids: Iterable[str]) -> None:
        embeds_map = self.model_runner.input_batch.req_prompt_embeds
        indices = self.model_runner.input_batch.req_id_to_index
        for req_id in finished_req_ids:
            if self.model_runner.lwd_hash_state is not None:
                self.model_runner.lwd_hash_state.discard(req_id)
            idx = indices.get(req_id)
            if idx is not None:
                embeds_map.pop(idx, None)

    def _lwd_next_down_seqno(self) -> int:
        seqno = self._lwd_down_next_seqno
        self._lwd_down_next_seqno += 1
        return seqno

    def shutdown(self) -> None:
        if hasattr(self, "_early_recv_stop"):
            self._early_recv_stop.set()
            if self._irecv_hint_mq is not None:
                self._irecv_hint_mq.shutdown()
            if self._lwd_early_recv_thread is not None:
                self._lwd_early_recv_thread.join(timeout=5)
            if self._irecv_done_mq is not None:
                self._irecv_done_mq.shutdown()
        super().shutdown()
