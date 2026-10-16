# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Own the no-PP data plane independently of the worker's native TP group.

Transport ownership and TP receive distribution come from d430200/982f776.
The centralized model runner and scheduler remain unchanged.
"""

from __future__ import annotations

import json

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger

from vllm_ascend.distributed.lwd_comm.early_recv import LwdEarlyRecv
from vllm_ascend.distributed.lwd_comm.future import LwdCommFuture
from vllm_ascend.distributed.lwd_comm.lwd_wire import LwdWire
from vllm_ascend.distributed.lwd_comm.service import LwdCommService
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest
from vllm_ascend.worker.worker import NPUWorker


class LwdWorker(LwdEarlyRecv, NPUWorker):
    def init_device(self) -> None:
        config, parallel = self.vllm_config.lwd_config, self.parallel_config
        if config is None or config.topology.deployment.scene != "single_instance" or len(config.connections) != 1:
            raise ValueError("[LWD] data-plane worker requires single_instance DP=1")
        if any(
            size != 1
            for size in (
                parallel.pipeline_parallel_size,
                parallel.data_parallel_size,
                parallel.prefill_context_parallel_size,
            )
        ):
            raise ValueError("[LWD] data-plane worker requires PP=DP=PCP=1")
        if len(config.instance.dp[0].ranks) != parallel.tensor_parallel_size:
            raise ValueError("[LWD] YAML rank count must match the local tensor-parallel size")
        super().init_device()
        self.lwd_wire = LwdWire(config, get_tp_group().rank_in_group, self.device, self.model_config.dtype)
        self.comm_service = LwdCommService(self.lwd_wire)
        self._lwd_up_hidden_size = self.model_config.get_hidden_size()
        self._early_recv_enabled = config.topology.feature_ctrl.enable_early_recv
        self._init_lwd_early_recv_state()
        self._lwd_probe_round = 0
        self._lwd_probe_next = {channel: 0 for channel in LwdChannelType}
        self._lwd_probe_error: str | None = None
        self.lwd_wire.join_async()
        logger.info(
            "[LWD][worker] role=%s tp_rank=%d connection=%s endpoint=%s; transport retained until shutdown",
            config.role,
            get_tp_group().rank_in_group,
            self.lwd_wire.key,
            self.lwd_wire.member,
        )

    def lwd_sync_error(self, error: str | None) -> None:
        # A failed endpoint must wake local TP peers before they enter a tensor broadcast.
        error = get_tp_group().broadcast_object(error, src=0)
        if error is not None:
            raise RuntimeError(f"[LWD] endpoint communication failed: {error}")

    def determine_available_memory(self) -> int:
        error = None
        if self.lwd_wire.member:
            try:
                self.lwd_wire.ensure_joined()
            except Exception as exc:
                error = str(exc)
        self.lwd_sync_error(error)
        # Include HCCL buffers in profiling before the native KV cache budget is fixed.
        return super().determine_available_memory()

    def send_lwd_tensor(self, tensor: torch.Tensor | None, channel: LwdChannelType, seqno: int) -> LwdCommFuture | None:
        if not self.lwd_wire.member:
            return None
        if tensor is None:
            raise ValueError("[LWD] sending endpoint requires a tensor")
        return self.comm_service.submit_send(
            LwdCommRequest(channel, "send", tensor.numel(), tensor, seqno=seqno, connection_key=self.lwd_wire.key)
        )

    def submit_lwd_recv(self, numel: int, channel: LwdChannelType, seqno: int) -> LwdCommFuture | None:
        if not self.lwd_wire.member:
            return None
        return self.comm_service.submit_recv(
            LwdCommRequest(channel, "recv", numel, seqno=seqno, connection_key=self.lwd_wire.key)
        )

    def wait_lwd_send(self, future: LwdCommFuture | None) -> None:
        error = None
        if self.lwd_wire.member:
            try:
                if future is None:
                    raise RuntimeError("[LWD] endpoint is missing its send")
                future.wait(timeout=self.lwd_wire.timeout_s)
            except Exception as exc:
                error = str(exc)
        self.lwd_sync_error(error)

    def receive_lwd_tensor(self, future: LwdCommFuture | None, numel: int) -> torch.Tensor:
        tensor, error = None, None
        if self.lwd_wire.member:
            try:
                if future is None:
                    raise RuntimeError("[LWD] endpoint is missing its receive")
                tensor = future.wait(timeout=self.lwd_wire.timeout_s).tensor
                if tensor is None or tensor.numel() != numel:
                    raise RuntimeError("[LWD] received tensor does not match the announced size")
            except Exception as exc:
                error = str(exc)
        self.lwd_sync_error(error)
        if not self.lwd_wire.member:
            tensor = torch.empty(numel, dtype=self.model_config.dtype, device=self.device)
        assert tensor is not None
        tp = get_tp_group()
        if tp.world_size > 1:
            # Wait for P2P completion before TP broadcast on the consuming stream.
            dist.broadcast(tensor, src=tp.ranks[0], group=tp.device_group, async_op=True).wait()
        tensor.record_stream(torch.npu.current_stream())
        return tensor

    def _lwd_build_up_recv_request(self, seqno: int, num_tokens: int, aux_rows: int) -> LwdCommRequest:
        if num_tokens <= 0 or aux_rows:
            raise ValueError("[LWD] UP requires hidden-state rows without auxiliary model payloads")
        return LwdCommRequest(
            channel=LwdChannelType.UP,
            op="recv",
            num_elements=num_tokens * self._lwd_up_hidden_size,
            seqno=seqno,
            connection_key=self.lwd_wire.key,
        )

    def lwd_run_probe(self, plan_json: str = "{}") -> dict[str, str | int]:
        from vllm_ascend.distributed.lwd_comm.probe import LwdProbePlan, run_probe

        if self._lwd_probe_error is not None:
            raise RuntimeError(f"[LWD] restart after failed probe: {self._lwd_probe_error}")
        plan = LwdProbePlan(**json.loads(plan_json))
        try:
            return run_probe(self, plan)
        except Exception as exc:
            # A timed-out device operation may still own buffers; do not reuse its sequence.
            self._lwd_probe_error = str(exc)
            raise

    def shutdown(self) -> None:
        try:
            if hasattr(self, "_early_recv_stop"):
                self.shutdown_early_recv()
            if hasattr(self, "comm_service"):
                self.comm_service.shutdown(timeout=5.0)
        finally:
            super().shutdown()
