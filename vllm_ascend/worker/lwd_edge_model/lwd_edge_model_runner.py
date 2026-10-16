# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD edge data-plane compute host (moved verbatim from LwdEdgeWorker).

The edge worker only dispatches EMBED/UNEMBED batches; the compute —
embedding forward + Hash expert lookup + UP send, DOWN receive + lm_head
+ rank replay, the MM cache and the tid2eid tables — lives here. The
native per-step runner pipeline stays dormant (no _prepare_inputs /
_model_forward / sample_tokens): the edge hijack semantics are
unchanged, this only restores the native worker↔runner division.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from vllm.logger import logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.v1.core.sched.output import LwdEmbedBatch, LwdUnembedBatch
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend import envs
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest
from vllm_ascend.worker.lwd_base_model.lwd_base_model_runner import LwdBaseModelRunner
from vllm_ascend.worker.lwd_edge_model.lwd_edge_mm_cache import LwdEdgeMmCache
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import pack_hash_payload
from vllm_ascend.worker.lwd_hash.lwd_hash_tables import load_tid2eid_tables

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheSpec


def select_token_batch(
    logits: torch.Tensor, ranks: list[int]
) -> list[int]:
    """Batched rank→token lookup: one sort, one D2H sync for all rows.

    logits [R,V] rows sorted together (NPU sorts rows in parallel);
    sorted_idx[i, ranks[i]] is the token id at rank ranks[i] for row i.
    """
    _, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    row = torch.arange(len(ranks), device=logits.device)
    return sorted_idx[row, torch.tensor(ranks, device=logits.device)].tolist()


class LwdEdgeModelRunner(LwdBaseModelRunner):
    """Embed + unembed compute for the LWD edge (no native step pipeline)."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: "torch.device",
        worker: "LwdWorkerBase | None" = None,
    ) -> None:
        super().__init__(vllm_config, device, worker)
        self.mm_cache = LwdEdgeMmCache()
        self.tid2eid_tables: list[torch.Tensor] = []

    def load_model(self) -> None:
        """加载边侧模型,并为 DeepSeek V4 从权重文件读取 CPU Hash 路由表。"""
        super().load_model()
        # [layer-trace] 对拍插桩(env 门控,缺省零开销):边侧安装点,
        # 与云侧/集中式共用同一份代码(见 lwd_debug.lwd_layer_trace)。
        from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
            install_lwd_layer_trace,
        )

        install_lwd_layer_trace(self.get_model())
        config = self.model_config.hf_config
        if getattr(config, "model_type", None) != "deepseek_v4":
            return
        num_hash_layers = config.num_hash_layers
        if not 0 <= num_hash_layers <= config.num_hidden_layers:
            raise ValueError(f"Invalid DeepSeek V4 num_hash_layers: {num_hash_layers}")
        if num_hash_layers == 0:
            logger.info("[lwd-edge] tid2eid loading skipped: num_hash_layers=0")
            return
        logger.info("[lwd-edge] loading tid2eid tables for %d Hash MoE layers", num_hash_layers)
        # Reuse vLLM's checkpoint resolution (revision, cache and shard filtering).
        loader = DefaultModelLoader(self.vllm_config.load_config)
        _, weight_files, use_safetensors = loader._prepare_weights(
            self.model_config.model,
            subfolder=None,
            revision=self.model_config.revision,
            fall_back_to_pt=False,
            allow_patterns_overrides=None,
        )
        if not use_safetensors:
            raise ValueError("LWD DeepSeek V4 tid2eid loading requires safetensors weights")
        self.tid2eid_tables = load_tid2eid_tables(
            weight_files, num_hash_layers, config.vocab_size,
            config.num_experts_per_tok, config.n_routed_experts,
        )

    def get_kv_cache_spec(self) -> "dict[str, KVCacheSpec]":
        """The LWD edge runs no attention/transformer, so it needs no KV cache."""
        return {}

    def _lwd_embed_ids_h2d(self, flat_token_ids: list[int]) -> torch.Tensor:
        """token ids 异步上卡:pinned 环(4 槽,与派发队列深度同阶)+
        non_blocking,替代 torch.tensor(list, device=...) 的 pageable
        同步拷贝——后者在主流上等设备排空(含上一块 chunk 的 UP send
        配对等待),把跨侧锁步传导进边 worker 主线程。同流序保证 embed
        kernel 读到的一定是本 chunk 的值(H2D 先于 embed 入队)。"""
        return self._lwd_staging(
            "edge_ids", dtype=torch.int64, min_capacity=8192,
        ).stage_to_device(flat_token_ids, len(flat_token_ids))

    @torch.inference_mode()
    def execute_lwd_embed(
        self,
        seqno: int,
        batch_meta: LwdEmbedBatch,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        model = self.get_model()
        if not batch_meta.token_ids:
            return

        # 登记新到请求的 mm 特征(首 chunk 随 scheduled_new_reqs 携带
        # data;后续 chunk 走缓存)。注意登记不过滤 data:None 的项留到
        # encoder 执行点 fail-fast(见 LwdEdgeMmCache.encoder_outputs)
        # ——静默跳过会让图像 embeds 缺失、占位 token 按文本嵌入,即错算。
        for req_data in scheduler_output.scheduled_new_reqs:
            feats = [
                f
                for f in req_data.mm_features
                if f.modality != "prompt_embeds"
            ]
            if feats:
                self.mm_cache.register(req_data.req_id, feats)

        # Flatten all requests' prompt tokens into one batch (order = req_ids).
        flat_token_ids = [tid for token_ids in batch_meta.token_ids for tid in token_ids]
        logger.debug(
            "[Lwd][edge-worker] embed token_ids=%s", flat_token_ids
        )
        token_ids_tensor = self._lwd_embed_ids_h2d(flat_token_ids)
        _t0 = time.monotonic()
        # 多模态 merge:批内任一请求带 mm 数据即按本 chunk 窗口收集
        # 视觉塔 embeds + 构造 is_multimodal 掩码,走模型原生
        # embed_input_ids 的 merge 分支;纯文本批走原直达路径。
        mm_embeds, is_mm = self.mm_cache.gather_chunk(
            batch_meta, model, self.device, self.pin_memory,
        )
        if mm_embeds:
            embeds = model.embed_input_ids(
                token_ids_tensor,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm,
            )
        else:
            embeds = model.embed_input_ids(token_ids_tensor)  # (total_N, H)
        _t_fwd = time.monotonic()
        payload = embeds
        if self.tid2eid_tables:
            # The original prompt IDs stay on the edge; transmit only lookups.
            cpu_ids = torch.tensor(flat_token_ids, dtype=torch.int64)
            expert_ids = torch.stack([table[cpu_ids] for table in self.tid2eid_tables], dim=1)
            payload = pack_hash_payload(embeds, expert_ids)
            logger.info(
                "[lwd-edge] UP Hash MoE seqno=%d expert_shape=%s payload_numel=%d",
                seqno, tuple(expert_ids.shape), payload.numel(),
            )

        # mrope positions 已随 RangeNotify 走控制面(wire_version>=2),
        # 数据面只发 embeds 主帧;aux 帧机制保留在通道层供 deepstack
        # 等扩展,mrope 不再使用。
        request = LwdCommRequest(
            channel=LwdChannelType.UP,
            op="send",
            num_elements=payload.numel(),
            tensor=payload,
            seqno=seqno,
        )
        self.worker.comm_service.submit_send(request)
        # [Lwd][perf] TTFT 探针:forward=embed 前向;submit_send=快照clone+
        # 广播提交(含 bridge 的 handle.wait)——若 UP 世界广播在等云侧
        # 入队,此段会显著变大(chunk 级锁步的直接证据)
        logger.info(
            "[Lwd][perf] embed seqno=%s forward=%.2f submit_send=%.2f "
            "total=%.2fms tokens=%d",
            seqno, (_t_fwd - _t0) * 1000,
            (time.monotonic() - _t_fwd) * 1000,
            (time.monotonic() - _t0) * 1000,
            len(flat_token_ids),
        )

    @torch.inference_mode()
    def execute_lwd_unembed(
        self, seqno: int, batch_meta: LwdUnembedBatch
    ) -> ModelRunnerOutput:
        model = self.get_model()
        if not batch_meta.req_ids:
            return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])

        _t0 = time.monotonic()
        recv_future = self.worker.comm_service.submit_recv(
            LwdCommRequest(
                channel=LwdChannelType.DOWN,
                op="recv",
                num_elements=batch_meta.recv_num_elements,  # int = rows_total * hidden_size
                seqno=seqno,
            )
        )
        _t_post = time.monotonic()
        # wait_for_comm:device 序等待,主机不阻塞
        recv_future.wait_for_comm()
        result = recv_future.result()
        _t_tensor = time.monotonic()
        hidden_size = self.model_config.get_hidden_size()
        hidden_states = result.tensor.view(-1, hidden_size)  # (rows_total, H)
        # 诊断旁路(VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE=1):DOWN 张量照收
        # (传输成本保留在测量内),但跳过 lm_head + rank-replay,直接交付
        # 云侧 c2e 通告捎带的 token ids——性能恢复即瓶颈在边侧 unembed。
        if envs.VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE and batch_meta.token_ids:
            assert len(batch_meta.token_ids) == len(batch_meta.req_ids)
            sampled_token_ids: list[list[int]] = [
                list(t) for t in batch_meta.token_ids
            ]
            _t_lm = time.monotonic()
        else:
            # lm_head 不允许直接调用(ParallelLMHead.forward 强制经 sampler);
            # compute_logits 是标准接口,包装层/单体模型都有。
            logits = model.compute_logits(hidden_states)   # (rows_total, V)
            _t_lm = time.monotonic()
            # L5 对拍:首行 top-20,与云侧 sampler 入口的 [layer-trace]
            # top20 逐位对照——排名换位即重放漂移的直接视图。
            try:
                from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
                    lwd_layer_trace_enabled,
                )

                if lwd_layer_trace_enabled() and logits.dim() == 2:
                    row = logits[0].detach().float()
                    vals, tids = torch.topk(row, min(20, row.numel()))
                    logger.info(
                        "[layer-trace] edge lm_head logits shape=%s row0 l2=%.4f "
                        "top20=%s",
                        tuple(logits.shape), row.norm().item(),
                        list(zip(tids.tolist(), [round(v, 3) for v in vals.tolist()])),
                    )
            except Exception:  # noqa: BLE001
                pass

            flat_ids = select_token_batch(
                logits, [r for ths in batch_meta.top_id_ths for r in ths]
            )
            sampled_token_ids = []
            off = 0
            for accept, ths in zip(
                batch_meta.num_accept_tokens, batch_meta.top_id_ths
            ):
                sampled_token_ids.append(flat_ids[off : off + accept])
                off += len(ths)
        _t_sel = time.monotonic()
        # [Lwd][perf] 临时探针:单请求慢的归因分段——
        # post_recv=挂 irecv;wait_tensor=等张量落卡(网络);lm_head=词表
        # 前向(权重带宽);select=argsort 取名(.item 同步);total=边尾段全长
        logger.info(
            "[Lwd][perf] unembed seqno=%s post_recv=%.2f wait_tensor=%.2f "
            "lm_head=%.2f select=%.2f total=%.2fms reqs=%d",
            seqno,
            (_t_post - _t0) * 1000,
            (_t_tensor - _t_post) * 1000,
            (_t_lm - _t_tensor) * 1000,
            (_t_sel - _t_lm) * 1000,
            (_t_sel - _t0) * 1000,
            len(batch_meta.req_ids),
        )
        req_id_to_index = {
            req_id: index for index, req_id in enumerate(batch_meta.req_ids)
        }
        return ModelRunnerOutput(
            req_ids=batch_meta.req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
        )
