# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Own the no-PP data plane independently of the worker's native TP group."""

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.v1.lwd.types import LwdBatch

from vllm_ascend.distributed.lwd_comm.future import LwdCommFuture
from vllm_ascend.distributed.lwd_comm.lwd_wire import LwdWire
from vllm_ascend.distributed.lwd_comm.service import LwdCommService
from vllm_ascend.worker.worker import NPUWorker


class LwdWorker(NPUWorker):
    def validate_lwd_batch(self, batch: LwdBatch) -> None:
        if batch.connection_key != self.lwd_wire.key:
            raise ValueError(f"[LWD] batch connection {batch.connection_key} != worker {self.lwd_wire.key}")

    def receive_lwd_tensor(self, future: LwdCommFuture | None, numel: int) -> torch.Tensor:
        if self.lwd_wire.member:
            if future is None:
                raise RuntimeError("[LWD] endpoint is missing its receive")
            tensor = future.wait(timeout=60.0).tensor
            if tensor is None or tensor.numel() != numel:
                raise RuntimeError("[LWD] received tensor does not match the announced size")
        else:
            tensor = torch.empty(numel, dtype=self.model_config.dtype, device=self.device)
        tp = get_tp_group()
        if tp.world_size > 1:
            # Wait for P2P completion before TP broadcast. Launch and wait on
            # the consuming stream: separate HCCL streams cannot imply ordering.
            dist.broadcast(tensor, src=tp.ranks[0], group=tp.device_group, async_op=True).wait()
        tensor.record_stream(torch.npu.current_stream())
        return tensor

    def init_device(self) -> None:
        if self.use_v2_model_runner:
            raise ValueError("[LWD] the migrated data plane requires the V1 model runner")
        super().init_device()
        self.lwd_wire = LwdWire(
            self.vllm_config.lwd_config,
            get_tp_group().rank_in_group,
            self.device,
            self.model_config.dtype,
        )
        self.comm_service = LwdCommService(self.lwd_wire)
        self.lwd_wire.join_async()
        logger.info(
            "[LWD][worker] role=%s tp_rank=%d connection=%s endpoint=%s",
            self.vllm_config.lwd_config.role,
            get_tp_group().rank_in_group,
            self.lwd_wire.key,
            self.lwd_wire.member,
        )

    def shutdown(self) -> None:
        try:
            if hasattr(self, "comm_service"):
                self.comm_service.shutdown(timeout=5.0)
        finally:
            super().shutdown()
