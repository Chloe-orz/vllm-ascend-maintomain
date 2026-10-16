# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only LWD model runner subclass (cloud side).

All LWD runner-side logic lives here (moved out of
``model_runner_v1.py``): the per-step cloud-side collection of the
DOWN payload (hidden-only tensor) plus the step metadata (global ranks
/ num_accepted / req_ids) that rides back to the scheduler on
``ModelRunnerOutput.lwd_c2e_meta``.

Selected via ``LwdCloudWorker`` (worker_cls), which swaps the model
runner class at init_device when ``lwd_config`` enables prefill_only.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend import envs
from vllm_ascend.worker.lwd_base_model.lwd_pinned_staging import LwdPinnedStaging
from vllm_ascend.worker.lwd_cloud_model.lwd_mtp_proposer import LwdMTPProposer
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import (
    LwdHashRoutingState,
    hash_layer_count,
    hash_payload_numel,
    unpack_hash_payload,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import LwdEmbedBatch
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm_ascend.distributed.lwd_comm.future import LwdCommFuture


# Endpoint host-gate timeout for the UP harvest wait: exceeding it means the
# matching isend never arrived (edge never posted or the wire stalled), so
# fail loudly instead of waiting forever.
_LWD_UP_HOST_GATE_TIMEOUT_S = 60.0


class LwdUpStagedBatch(NamedTuple):
    """Per-batch harvest record flowing stage -> tensor phase.

    ``t_wait`` carries the probe start so the tensor phase's
    up-recv-wait window matches the pre-split single loop; the inject
    pass receives ``rows`` from the window check instead of recomputing
    indices (the duplicate existence check used to run twice per
    request).
    """

    seqno: int
    meta: LwdEmbedBatch
    future: LwdCommFuture | None
    num_tokens: int
    t_wait: float


class LwdDownSelection(NamedTuple):
    """Row/rank selection product of one DOWN collect step."""

    hidden_packet: torch.Tensor
    ranks: list[torch.Tensor]
    counts_dev: torch.Tensor | None
    seg_lens_host: list[int] | None


class LwdCloudModelRunner(NPUModelRunner):
    """NPUModelRunner + prefill_only LWD cloud-side runner logic."""

    # Config-derived constants, computed once at first use; lwd_config and
    # the HF config are immutable at runtime. Class-level defaults keep
    # instances built via __new__ (tests) from tripping on missing attrs.
    _lwd_hash_dims_cached: tuple[int, int] | None = None

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: "torch.device",
        worker: "LwdCloudWorker | None" = None,
    ) -> None:
        super().__init__(vllm_config, device)
        self.worker = worker  # LwdCloudWorker ref (UP recv futures live there)
        self._lwd_pending_down_payload = None
        # Stage per-request prompt lengths for device-side mask refresh.
        if self.use_async_spec_decode and self.enable_prompt_embeds:
            self._lwd_prompt_lens_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            self._lwd_prompt_lens_gpu = None
        self.lwd_hash_state = None
        # 用空值表示尚未准备真实批次，区分 dummy 前向。
        self._lwd_hash_step_tokens = None
        config = self.model_config.hf_config
        num_hash_layers = hash_layer_count(config)
        if num_hash_layers and vllm_config.lwd_config.mode == "prefill_only":
            # UP currently broadcasts one edge's batch to one cloud TP group.
            # CP/DP need additional request-to-rank layouts before supporting V4.
            parallel = vllm_config.parallel_config
            if (parallel.data_parallel_size != 1 or parallel.prefill_context_parallel_size != 1
                    or parallel.decode_context_parallel_size != 1):
                raise ValueError("LWD V4 Hash routing currently requires DP=PCP=DCP=1 (TP/EP are supported)")
            self.lwd_hash_state = LwdHashRoutingState(
                num_hash_layers, config.num_experts_per_tok, config.n_routed_experts,
            )
            # Persistent buffers are populated before model execution, so graph
            # capture and replay see the same addresses for prompt and decode.
            self._lwd_hash_experts = torch.zeros(
                self.max_num_tokens, num_hash_layers, config.num_experts_per_tok,
                dtype=torch.int32, device=device,
            )
            self._lwd_hash_prompt_mask = torch.zeros(self.max_num_tokens, dtype=torch.bool, device=device)
            self._lwd_hash_token_ids = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)

    def _get_drafter(self) -> "SpecDecodeBaseProposer | None":
        """Use the LWD MTP embedding adapter only in prefill_only mode."""
        if (
            self.vllm_config.lwd_config.mode == "prefill_only"
            and self.speculative_config.method == "mtp"
            and not self.speculative_config.use_step3p5_mtp()
        ):
            return LwdMTPProposer(self.vllm_config, self.device, self)
        return super()._get_drafter()

    # ------------------------------------------------------------------ #
    # Remote embeds injection (cloud input has NO token ids — the prompt   #
    # embeddings arrive over the UP channel and must drive forward)        #
    # ------------------------------------------------------------------ #

    def _prepare_inputs(
        self, scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: "np.ndarray",
    ) -> "tuple[torch.Tensor, SpecDecodeMetadata | None]":
        """UP 收割与 mrope 写入在 super() 之前,embeds 覆写在之后。

        mrope 时序契约(多模态迁移复核发现的源分支缺陷修复):super()
        内部的 _calc_mrope_positions 当步即把 req_state.mrope_positions
        的窗口行拷上 GPU(model_runner_v1.py:1033-1045)——mrope 写入
        若晚于 super()(源实现形态),当步 chunk 的 mrope 行永远不被
        自己的前向看到(所有 chunk 用 ids=None 直通初始化的 arange
        值,图像行必错)。故收割(端点 host 门 + TP 组内现场广播)与
        mrope/delta 写入提到 super() 之前。

        embeds 保持在 super() 之后:基类填充循环 + copy_to_gpu 只看到
        零缓冲,随后把收到的 UP embeds 直写 inputs_embeds.gpu 当步调度
        窗口(后写为权威),经 future 完成事件在流上有序——非阻塞,
        recv 等待另有 [Lwd][perf] up-recv-wait 计时可见。
        """
        harvested = self._lwd_harvest_up_batches(num_scheduled_tokens)
        # 跳过基类填充循环对 LWD 占位行的逐行拷贝(gpu_model_runner
        # .py:952-996):该循环把 CPU 装配缓冲的当步窗口(此刻恒为
        # 零值)做 84MB 主机 memcpy + H2D 灌入 inputs_embeds,随后
        # 注入覆写同一窗口(后写为权威)——纯浪费,实测 super 段
        # ~17-20ms/chunk。清空映射即整体跳过;注入/释放逻辑在
        # super() 返回后恢复引用。LWD 云侧请求全是占位行,无真实
        # prompt-embeds 消费者被误伤。
        embeds_map = self.input_batch.req_prompt_embeds
        self.input_batch.req_prompt_embeds = {}
        try:
            out = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        finally:
            self.input_batch.req_prompt_embeds = embeds_map
        self._lwd_rebuild_is_token_ids_mask(num_scheduled_tokens)
        self._lwd_release_consumed_prompt_embeds()
        self._lwd_inject_remote_embeds(harvested, num_scheduled_tokens)
        if self.lwd_hash_state is not None:
            self._lwd_prepare_hash_batch(num_scheduled_tokens)
        return out

    def _lwd_prepare_hash_batch(self, num_scheduled_tokens: "np.ndarray") -> None:
        """Skip remote expert staging for decode, preserving graph buffers."""
        num_reqs = self.input_batch.num_reqs
        counts = num_scheduled_tokens[:num_reqs]
        computed = self.input_batch.num_computed_tokens_cpu[:num_reqs]
        prompt_lengths = self.input_batch.num_prompt_tokens[:num_reqs]
        self._lwd_hash_step_tokens = sum(int(n) for n in counts)
        has_prompt = any(
            int(count) > 0 and int(start) < int(prompt_len)
            for count, start, prompt_len in zip(counts, computed, prompt_lengths)
        )
        # Clear the full mask, including graph padding and previous larger
        # batches. Keep its address and the captured merge graph unchanged.
        self._lwd_hash_prompt_mask.zero_()
        if not has_prompt:
            # The merge selects local lookup for every row. Stale expert IDs
            # are only unselected values, never indices into an expert table.
            return
        batch, mask = self.lwd_hash_state.build_batch(
            self.input_batch.req_ids, counts, computed, prompt_lengths,
        )
        self._lwd_hash_experts.zero_()
        self._lwd_hash_experts[:batch.shape[0]].copy_(batch)
        self._lwd_hash_prompt_mask[:mask.shape[0]].copy_(mask)

    def _lwd_rebuild_is_token_ids_mask(
        self, num_scheduled_tokens: "np.ndarray"
    ) -> None:
        """重建 prompt-embeds 分支的扁平 is_token_ids 掩码。

        根因(async spec decode 乐观值):runner 的 num_computed_tokens_cpu
        假设上步草稿全被接受(比真实提交位置多 r),而原生扁平化
        (model_runner_v1.py 的 token_indices = positions + idx*M)用乐观
        positions 从二维掩码取值——decode 行整体右移 r 位,每段末尾
        r 行越出 True 标记区读到 False/陈旧值,被 prompt-embeds 分支
        当作 embeds 行而不做本地嵌入,保留 inputs_embeds.gpu 的陈旧
        垃圾;这些行恰是被采样行(logits_indices 取段末 1+k 行),
        垃圾 logits 直接污染已提交 token。原生/phase/单请求不受影响
        (掩码只在 prompt-embeds 分支被消费,那些形态走 MM 分支)。

        LWD 语义里只有注入行(prompt 段)应是 embeds 行(False),
        decode/草稿行恒为 token id(True)。此处按调度状态确定性重建,
        不依赖任何乐观值修正:decode 请求的 computed 乐观只会偏大,
        prefill_rows 恒为 0,天然安全。保留 CPU 视图修复供父类路径使用
        (.np 与 .cpu 同存储)；异步路径在 _preprocess 中按修正后的
        设备位置重新生成 GPU 掩码，并直接消费设备 embedding。
        """
        if not self.input_batch.req_prompt_embeds:
            # 批内无 embeds 缓冲(MM/text 分支,掩码无人读)
            return
        mask = self.is_token_ids.np
        computed = self.input_batch.num_computed_tokens_cpu
        num_prompt = self.input_batch.num_prompt_tokens
        n_reqs = len(num_scheduled_tokens)
        total = int(np.sum(num_scheduled_tokens[:n_reqs]))
        mask[:total] = True
        off = 0
        for i in range(n_reqs):
            n = int(num_scheduled_tokens[i])
            prefill_rows = min(
                n, max(int(num_prompt[i]) - int(computed[i]), 0)
            )
            if prefill_rows > 0:
                mask[off : off + prefill_rows] = False
            off += n

    def _preprocess(
        self, scheduler_output: "SchedulerOutput", num_input_tokens: int,
        intermediate_tensors: "IntermediateTensors | None" = None,
    ) -> tuple[
        "torch.Tensor | None",
        "torch.Tensor | None",
        "torch.Tensor",
        "IntermediateTensors | None",
        "dict[str, Any]",
        "ECConnectorOutput | None",
    ]:
        """在异步投机位置修正后刷新 decode 掩码，再准备模型输入。"""
        if (self.use_async_spec_decode and self.enable_prompt_embeds
                and intermediate_tensors is None and self.pcp_size == 1):
            self._lwd_refresh_decode_embedding_mask(scheduler_output)
            if (get_pp_group().is_first_rank
                    and not self.model_config.is_encoder_decoder
                    and (not self.supports_mm_inputs or self.input_batch.req_prompt_embeds)):
                return self._lwd_preprocess_prompt_embeds(scheduler_output, num_input_tokens)
            # Other model paths keep the parent's preprocessing, which may
            # still consume the CPU mask (for example encoder-decoder models).
            count = scheduler_output.total_num_scheduled_tokens
            self.is_token_ids.np[:count] = self.is_token_ids.gpu[:count].cpu().numpy()
        return super()._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)

    def _resolve_prompt_embeds(
        self, new_req_data: "NewRequestData"
    ) -> "torch.Tensor | None":
        """LWD: the edge-cloud transport carries shape only for all-zero
        embeds buffers, so rebuild the local placeholder here (same
        semantics as the engine-side torch.zeros)."""
        if (
            new_req_data.prompt_embeds is None
            and new_req_data.prompt_embeds_shape is not None
        ):
            return torch.zeros(
                new_req_data.prompt_embeds_shape,
                dtype=new_req_data.prompt_embeds_dtype,
            )
        return new_req_data.prompt_embeds

    def _should_run_mm_encoder(
        self, is_first_rank: bool, is_encoder_decoder: bool
    ) -> bool:
        """LWD: batches carrying edge-provided embeds must not run the MM
        encoder, or embeddings recomputed from input_ids would overwrite
        the remote ones wholesale."""
        if self.enable_prompt_embeds and bool(self.input_batch.req_prompt_embeds):
            return False
        return super()._should_run_mm_encoder(is_first_rank, is_encoder_decoder)

    def _lwd_preprocess_prompt_embeds(
        self, scheduler_output: "SchedulerOutput", num_input_tokens: int
    ) -> tuple[
        "torch.Tensor | None",
        "torch.Tensor | None",
        "torch.Tensor",
        "IntermediateTensors | None",
        "dict[str, Any]",
        "ECConnectorOutput | None",
    ]:
        """Consume the corrected device mask on the LWD cloud decoder path."""
        count = scheduler_output.total_num_scheduled_tokens
        is_token_ids = self.is_token_ids.gpu[:count]
        # Remote prompt rows do not carry meaningful token IDs.
        token_ids = self.input_ids.gpu[:count].masked_fill(~is_token_ids, 0)
        tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
        embeds = self.inputs_embeds.gpu[:count]
        embeds.copy_(torch.where(is_token_ids.unsqueeze(-1), tokens_to_embeds, embeds))

        # Match the parent's position layout and clear padding for graph replay.
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]
            if num_input_tokens > count:
                self.positions[count:num_input_tokens].zero_()
        return (
            None,
            self.inputs_embeds.gpu[:num_input_tokens],
            positions,
            None,
            self._init_model_kwargs(),
            None,
        )

    def _lwd_refresh_decode_embedding_mask(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """根据设备上的实际位置重建掩码，避免 decode 行复用旧 embedding。

        异步上传各请求的 prompt 长度，复用设备上的请求索引展开到 token 行。
        掩码直接写入设备缓冲区，由 LWD prompt-embeds 路径在设备上消费；
        不回传 positions，也不刷新 CPU 掩码。
        """
        num_reqs = len(self.input_batch.req_ids)
        count = scheduler_output.total_num_scheduled_tokens
        prompt_lens_gpu = self._lwd_prompt_lens_gpu
        assert prompt_lens_gpu is not None
        prompt_lens_gpu[:num_reqs].copy_(
            self.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )
        prompt_lens_rows = prompt_lens_gpu[self.req_indices.gpu[:count]]
        self.is_token_ids.gpu[:count] = self.positions[:count] >= prompt_lens_rows

    # 在云侧模拟前向前清除真实批次标记，防止复用上次请求路由。
    def _dummy_run(self, *args, **kwargs) -> None:
        """清除真实批次的 Hash 路由标记，再执行模拟前向。"""
        # 用空值表示尚未准备真实批次，区分 dummy 前向。
        self._lwd_hash_step_tokens = None
        return super()._dummy_run(*args, **kwargs)

    def _model_forward(self, num_tokens_padded: int,
                       input_ids: "torch.Tensor | None" = None,
                       positions: "torch.Tensor | None" = None,
                       intermediate_tensors: "IntermediateTensors | None" = None,
                       inputs_embeds: "torch.Tensor | None" = None,
                       **model_kwargs) -> "torch.Tensor":
        """按 SP 布局切分 embedding，并将完整 Hash 路由数据写入前向上下文。"""
        context = get_forward_context()
        if (inputs_embeds is not None and intermediate_tensors is None
                and context.flash_comm_v1_enabled and not self.supports_mm_inputs):
            # 直接传入 embedding 会绕过原生嵌入层，需要在此补上序列切分。
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                if inputs_embeds.shape[0] != num_tokens_padded:
                    raise ValueError("LWD SP expects full-token inputs_embeds before slicing")
                if context.pad_size:
                    inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, context.pad_size))
                inputs_embeds = inputs_embeds.chunk(tp_group.world_size, dim=0)[tp_group.rank_in_group]
            # 位置、注意力元数据和 Hash 路由表保持完整 token 布局。
        if self.lwd_hash_state is not None:
            # 先初始化路由索引缓存，确保补齐行使用合法零索引。
            self._lwd_hash_token_ids.zero_()
            if self._lwd_hash_step_tokens is None:
                # Synthetic forwards have no remote prompt; route dummy token 0.
                self._lwd_hash_prompt_mask.zero_()
            else:
                count = self._lwd_hash_step_tokens
                if count > num_tokens_padded:
                    raise ValueError("LWD Hash routing rows exceed model input rows")
                # Native input preparation retains real decode IDs even when
                # model input_ids=None because the model consumes embeddings.
                self._lwd_hash_token_ids[:count].copy_(self.input_ids.gpu[:count])
            context.input_ids = self._lwd_hash_token_ids[:num_tokens_padded]
            context.lwd_hash_expert_ids = self._lwd_hash_experts[:num_tokens_padded]
            context.lwd_hash_prompt_mask = self._lwd_hash_prompt_mask[:num_tokens_padded]
        return super()._model_forward(
            num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs,
        )

    def _lwd_hash_dims(self) -> tuple[int, int]:
        """(num_hash_layers, experts_per_tok), cached at first use."""
        dims = self._lwd_hash_dims_cached
        if dims is None:
            config = self.model_config.hf_config
            dims = (
                hash_layer_count(config),
                getattr(config, "num_experts_per_tok", 0),
            )
            self._lwd_hash_dims_cached = dims
        return dims

    def _lwd_release_consumed_prompt_embeds(self) -> None:
        """Release assembly buffers whose prompt was fully consumed.

        A buffer becomes releasable once the request's computed tokens
        cover the whole prompt: the last chunk's rows were consumed by
        the previous step's fill loop, and the draft proposer read its
        window during that step's sampling (which precedes this call).
        Decode steps never read prompt embeds, so dropping the entry
        here cannot corrupt anything.  Entries follow native index
        compaction (condense/swap), so keying by idx stays correct.

        Also drops orphan entries whose slot no longer holds a live
        request (preemption removes the request without finishing it, so
        the flush hook never fires): a stale buffer at a recycled index
        would otherwise be mistaken for the new occupant's embeds.
        """
        if self.lwd_hash_state is not None:
            for req_id in list(self.lwd_hash_state.requests):
                request = self.requests.get(req_id)
                if request is None or request.num_computed_tokens >= request.num_prompt_tokens:
                    self.lwd_hash_state.discard(req_id)
        embeds_map = self.input_batch.req_prompt_embeds
        if not embeds_map:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        slot_req_ids = self.input_batch.req_ids
        for idx in list(embeds_map.keys()):
            if idx >= len(computed):
                embeds_map.pop(idx)
                continue
            if idx >= len(slot_req_ids) or slot_req_ids[idx] is None:
                embeds_map.pop(idx)
                continue
            if computed[idx] >= num_prompt[idx]:
                embeds_map.pop(idx)

    def lwd_release_finished(self, finished_req_ids: list[str]) -> None:
        """完结请求的采集簿记释放(释放路径唯一 owner):Hash chunk 表
        + prompt-embeds 装配缓冲。由 worker 在引擎 finished_req_ids
        信号处调用(abort 中途 prefill 的兜底;正常 prefill 完结已在
        _lwd_release_consumed_prompt_embeds 释放)。"""
        embeds_map = self.input_batch.req_prompt_embeds
        req_id_to_index = self.input_batch.req_id_to_index
        for req_id in finished_req_ids:
            if self.lwd_hash_state is not None:
                self.lwd_hash_state.discard(req_id)
            idx = req_id_to_index.get(req_id)
            if idx is not None:
                embeds_map.pop(idx, None)

    def _lwd_harvest_up_batches(
        self, num_scheduled_tokens: "np.ndarray"
    ) -> list[tuple[int, LwdEmbedBatch, torch.Tensor, list[tuple[str, int, int]]]]:
        """UP 批收割(super()._prepare_inputs 之前调用,时序契约见
        _prepare_inputs)。逐 seqno 三段交错,事件序与拆分前的单循环
        逐点一致:长度核验(_lwd_stage_up_batch)→ 端点 host 门 + TP
        广播(_lwd_harvest_tensors_one)→ 窗口核验 + mrope 写入
        (_lwd_check_up_windows)。

        返回 [(seqno, meta, up_flat, rows)] 供 super() 后的注入段复用
        已核验批位。张量段整体后移到 super() 之后与宿主工作重叠属
        暂缓项 D1(见 docs/refactor/lwd_worker_packages_refactor_plan.md),
        届时只需把 _lwd_harvest_tensors_one 的调用点挪出本循环。"""
        worker = self.worker
        if worker is None:
            return []
        posted = getattr(worker, "_lwd_up_recv_futures", None)
        if not posted:
            return []
        harvested = []
        for batch_seqno in list(posted.keys()):
            item = worker._lwd_up_recv_futures.pop(batch_seqno, None)
            if item is None:
                continue
            future, meta = item
            staged = self._lwd_stage_up_batch(batch_seqno, future, meta)
            up_flat = self._lwd_harvest_tensors_one(staged)
            if up_flat is None:
                continue
            rows = self._lwd_check_up_windows(
                batch_seqno, meta, num_scheduled_tokens,
            )
            harvested.append((batch_seqno, meta, up_flat, rows))
        return harvested

    def _lwd_stage_up_batch(
        self,
        batch_seqno: int,
        future: LwdCommFuture | None,
        meta: LwdEmbedBatch,
    ) -> LwdUpStagedBatch:
        """收割登记段:计算批 token 数并核验 meta 长度契约(拆分前位于
        host 门之前,顺序保持)。计时起点随记录携带,张量段的
        up-recv-wait 探针窗口与拆分前一致。"""
        t_wait = time.monotonic()
        num_tokens = sum(len(t) for t in meta.token_ids)
        if not (len(meta.req_ids) == len(meta.token_ids)
                == len(meta.prompt_offsets) == len(meta.has_mrope)
                == len(meta.mrope_positions)):
            raise ValueError("LWD requires one prompt offset and mrope flag per request")
        return LwdUpStagedBatch(
            seqno=batch_seqno, meta=meta, future=future,
            num_tokens=num_tokens, t_wait=t_wait,
        )

    def _lwd_check_up_windows(
        self,
        batch_seqno: int,
        meta: LwdEmbedBatch,
        num_scheduled_tokens: np.ndarray,
    ) -> list[tuple[str, int, int]]:
        """逐请求窗口核验 + mrope 行写入;返回每行 (req_id, 批位, 行数)。

        三类 fail-fast 与拆分前逐字一致:请求不在本步 batch / chunk 行数
        不等于调度窗口 / chunk 偏移与 computed 不符——继续走下去都会以
        未注入的脏 embeds 解码(静默乱码),必须当场暴露。

        mrope positions 自 wire_version 2 起经控制面随 meta 到达(逐
        请求 [3,n] int64 字节,frombuffer 零转置直写 req_state),不再
        经数据面 aux 帧与 TP 组二次广播;纯文本请求空字节、零路径
        变化。"""
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        mrope_rows = 0
        _mrope_cpu_ms = _mrope_delta_ms = 0.0
        rows: list[tuple[str, int, int]] = []
        for k, (req_id, token_ids) in enumerate(
            zip(meta.req_ids, meta.token_ids)
        ):
            n = len(token_ids)
            idx = self.input_batch.req_id_to_index.get(req_id)
            if idx is None:
                # fail-fast:embeds 已收齐但请求不在本步 batch——继续
                # 走下去该请求将用未注入的脏 embeds 解码(静默乱码),
                # 必须当场暴露而非丢弃。出现即调度/数据面时序契约
                # 被破坏,需要修的是上游而不是这里。
                raise RuntimeError(
                    f"[Lwd][cloud-runner] INJECT req={req_id} seqno="
                    f"{batch_seqno} rows={n}: req not in input_batch "
                    f"(batch={self.input_batch.req_ids})"
                )
            if n > 0:
                # fail-fast:本 chunk 行数必须恰等于该请求本步调度 token
                # 数。不等(调度窗口 ≠ 边侧 chunk,如预算挤压截断)时
                # 注入会越窗踩邻请求行/留下未注入尾巴——静默乱码源,
                # 直接暴露。
                scheduled = (
                    int(num_scheduled_tokens[idx])
                    if idx < len(num_scheduled_tokens) else 0
                )
                if n != scheduled:
                    raise RuntimeError(
                        f"[Lwd][cloud-runner] INJECT window mismatch "
                        f"req={req_id} seqno={batch_seqno}: chunk rows="
                        f"{n} != scheduled={scheduled} (computed="
                        f"{int(computed[idx])} prompt="
                        f"{int(num_prompt[idx])})"
                    )
                start = int(meta.prompt_offsets[k])
                prompt_len = int(num_prompt[idx])
                if start != int(computed[idx]) or not 0 <= start <= start + n <= prompt_len:
                    raise ValueError(
                        f"LWD prompt chunk offset mismatch for {req_id}: "
                        f"start={start}, computed={int(computed[idx])}, rows={n}, prompt={prompt_len}"
                    )
                if meta.has_mrope[k]:
                    # positions 字节随 meta 经控制面到达([3,n] int64),
                    # frombuffer 成 CPU 张量零转置直写 req_state;
                    # 空字节 = 边云版本错配,fail-fast
                    payload = meta.mrope_positions[k]
                    if not payload:
                        raise RuntimeError(
                            f"[Lwd][cloud-runner] mrope payload missing "
                            f"req={req_id} seqno={batch_seqno}: "
                            f"has_mrope=True 但控制面 positions 为空"
                            f"(边云版本错配,须同版本部署)"
                        )
                    cpu_ms, delta_ms = self._lwd_inject_mrope_positions(
                        req_id, start, n,
                        torch.frombuffer(
                            bytearray(payload), dtype=torch.int64
                        ).view(3, n),
                        int(num_prompt[idx]),
                    )
                    _mrope_cpu_ms += cpu_ms
                    _mrope_delta_ms += delta_ms
                    mrope_rows += n
            rows.append((req_id, idx, n))
        if mrope_rows > 0:
            # [Lwd][perf] mrope 注入探针:cpu=positions 字节 frombuffer
            # +CPU 直写 req_state(控制面送达,无设备往返);delta=末
            # chunk max() 自推,逐请求累加
            logger.info(
                "[Lwd][perf] cloud mrope-inject seqno=%s rows=%d "
                "cpu=%.2f delta=%.2fms",
                batch_seqno, mrope_rows,
                _mrope_cpu_ms, _mrope_delta_ms,
            )
        return rows

    def _lwd_harvest_tensors_one(
        self, staged: LwdUpStagedBatch,
    ) -> torch.Tensor | None:
        """收割张量段(单批):端点过 host 门 + TP 组内现场广播收主帧
        embeds(each frame its own launch+wait atomic pair on the current
        stream — the prefill_only_demo_br discipline; 跨流排序不可依赖
        是此前广播段交付残缺的根因)。端点 res.tensor 为 None 时返回
        None(拆分前同款 continue 语义)。"""
        # get_tp_group 保持函数内导入:v4 迁移测试经 sys.modules 按调用
        # 打桩 parallel_state,模块级名字在裁剪命名空间里不可见。
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        is_endpoint = self.worker._lwd_early_recv_endpoint
        hidden_size = self.model_config.get_hidden_size()
        num_hash_layers, top_k = self._lwd_hash_dims()
        numel = hash_payload_numel(
            staged.num_tokens, hidden_size, num_hash_layers, top_k,
        )
        if is_endpoint:
            # 端点:先过 host 就绪门(demo 的 wait gate):done_event
            # 轮询通过即 P2P 数据已完整落 buffer;超时显式报错而非
            # 静默乱码。门通过后再广播,广播读到的必然是完整数据。
            res = staged.future.wait(timeout=_LWD_UP_HOST_GATE_TIMEOUT_S)
            up_flat = res.tensor
            if up_flat is None:
                return None
        else:
            # 非端点:现场分配接收 buffer,等端点广播转发。
            up_flat = torch.empty(numel, dtype=torch.bfloat16, device="npu")
        if tp.world_size > 1:
            # demo 同款:直接用框架 TP 通信域(建组顺序由框架保证,
            # PD 分离路径已验证),不再使用自建 _LWD_EMBED_BCAST_GROUP。
            work = dist.broadcast(
                up_flat,
                src=tp.ranks[0],
                group=tp.device_group,
                async_op=True,
            )
            work.wait()  # 桥接广播完成到当前(计算)流,后续 copy 有序
        logger.info(
            "[Lwd][perf] cloud up-recv-wait seqno=%s dur=%.2fms",
            staged.seqno, (time.monotonic() - staged.t_wait) * 1000,
        )
        logger.info(
            "[Lwd][cloud-runner] UP embeds harvested seqno=%s rows=%d "
            "reqs=%s mrope_reqs=%s",
            staged.seqno, staged.num_tokens, staged.meta.req_ids,
            sum(staged.meta.has_mrope),
        )
        return up_flat

    def _lwd_inject_remote_embeds(
        self,
        harvested: list[
            tuple[int, LwdEmbedBatch, torch.Tensor, list[tuple[str, int, int]]]
        ] | None,
        num_scheduled_tokens: "np.ndarray",
    ) -> None:
        """Device-side inject(super() 之后):把收割到的 UP embeds 直写
        ``inputs_embeds.gpu`` 各请求的当步调度窗口。

        The flattened output offsets reproduce the native fill loop's
        accumulation (per-request scheduled segment start), so rows land
        exactly where the prompt-embeds branch expects them.  The CPU
        draft first-pass provider reads the SAME window straight from
        ``inputs_embeds.gpu`` (llm_base_proposer), so the CPU assembly
        buffer is no longer written at all — the per-chunk 84MB D2H is
        gone.  The base fill loop is skipped for LWD placeholder rows
        (see ``_prepare_inputs``); our device write is authoritative.

        批位直接复用收割段携带的 rows(重复的"req not in batch"防线
        已收敛到收割段一处;此处索引必为已核验值)。
        """
        if not harvested:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        hidden_size = self.model_config.get_hidden_size()
        gpu_embeds = self.inputs_embeds.gpu

        # Flattened output offset per request (native fill loop 同款累计):
        # 每个请求的调度段在扁平 token 序列中的起点,按批位索引。
        offsets: list[int] = []
        off = 0
        for i in range(len(self.input_batch.req_ids)):
            offsets.append(off)
            off += int(num_scheduled_tokens[i]) if i < len(num_scheduled_tokens) else 0

        num_hash_layers, top_k = self._lwd_hash_dims()
        for batch_seqno, meta, up_flat, rows in harvested:
            num_tokens = sum(len(t) for t in meta.token_ids)
            expert_ids = None
            if num_hash_layers:
                embeds, expert_ids = unpack_hash_payload(
                    up_flat, num_tokens, hidden_size, num_hash_layers,
                    top_k,
                )
            else:
                embeds = up_flat.view(num_tokens, hidden_size)
            logger.info(
                "[Lwd][cloud-runner] UP embeds injected seqno=%s rows=%s "
                "reqs=%s",
                batch_seqno, embeds.shape[0], meta.req_ids,
            )
            row = 0
            for meta_idx, (req_id, idx, n) in enumerate(rows):
                assert idx >= 0, f"LWD inject index missing for {req_id}"
                if n > 0:
                    start = int(meta.prompt_offsets[meta_idx])
                    if self.lwd_hash_state is not None:
                        if expert_ids is None:
                            raise ValueError(f"Missing LWD V4 expert payload for {req_id}")
                        self.lwd_hash_state.add_chunk(
                            req_id, int(num_prompt[idx]), start, expert_ids[row:row + n],
                        )
                    out = offsets[idx]
                    # stream 上直接写 inputs_embeds.gpu 的调度窗口
                    gpu_embeds[out : out + n].copy_(
                        embeds[row : row + n], non_blocking=True
                    )
                    # 不再 D2H 写 CPU 装配缓冲:draft provider 已改为
                    # 直读 inputs_embeds.gpu 同窗口(本行即权威副本),
                    # 消除每 chunk 84MB 的 D2H 同步长尾
                    self.input_batch.is_token_ids[idx, start : start + n] = False
                row += n
            # 跨流生命周期登记:端点 recv buffer 由通道流分配与复用
            # (同尺寸 chunk 下分配器几乎总给同一块),本步 copy 在计算流。
            # 不登记 record_stream,分配器可在本批 copy 尚未执行时把块交给
            # 下一 seqno 的 irecv 覆写——深异步队列下表现为跨 seqno 串
            # 数据(多请求乱码)。登记后复用即被正确排序。
            up_flat.record_stream(torch.npu.current_stream())

    # ------------------------------------------------------------------ #
    # Collect probe (measurement only; output discarded, protocol        #
    # still returns token_ids directly)                                   #
    # ------------------------------------------------------------------ #

    def _lwd_inject_mrope_positions(
        self,
        req_id: str,
        start: int,
        n: int,
        chunk_positions: torch.Tensor,
        prompt_len: int,
    ) -> tuple[float, float]:
        """写入一个 chunk 的 mrope positions([3,n] int64,CPU——控制面
        RangeItem.mrope_positions 字节 frombuffer 而来)到请求缓存的
        [3, prompt] 窗口;末 chunk 落地时自推 delta(=
        positions.max()+1-prompt_len,与边侧逐值一致)供 decode 期
        原生 _calc_mrope_positions 现算 completion 段位置。

        req_state.mrope_positions 由 ids=None 直通初始化(arange 纯文
        本位置)——多模态请求的窗口被逐 chunk 覆盖,纯文本请求永不走
        本路径(边侧不给它发 positions)。

        返回 (cpu_ms, delta_ms):CPU 直写与末 chunk delta 自推的耗时
        打点,由调用方按 seqno 聚合。"""
        req_state = self.requests.get(req_id)
        assert chunk_positions.shape == (3, n), (
            f"mrope payload shape {tuple(chunk_positions.shape)} != "
            f"(3, {n}) (req={req_id})"
        )
        _t_cpu = time.monotonic()
        req_state.mrope_positions[:, start : start + n] = chunk_positions
        cpu_ms = (time.monotonic() - _t_cpu) * 1000
        delta_ms = 0.0
        if start + n >= prompt_len:
            positions = req_state.mrope_positions[:, :prompt_len]
            req_state.mrope_position_delta = (
                int(positions.max().item()) + 1 - prompt_len
            )
            logger.info(
                "[Lwd][cloud-runner] mrope complete: req=%s prompt=%d "
                "delta=%d",
                req_id, prompt_len, req_state.mrope_position_delta,
            )
        return cpu_ms, delta_ms

    def _pp_broadcast_prev_sampled_token_ids(
        self, sampled_token_ids: "torch.Tensor"
    ) -> None:
        """LWD: token flow rides the lwd DOWN channel, skip the native PP broadcast."""

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:
        """LWD: token flow rides the lwd UP channel, skip the native PP receive."""

    def _sample(
        self, logits: "torch.Tensor | None",
        spec_decode_metadata: "SpecDecodeMetadata | None",
    ) -> "SamplerOutput":
        """Capture the sampler output for the collect probe
        (the base sample_tokens clears execute_model_state on return)."""
        sampler_output = super()._sample(logits, spec_decode_metadata)
        self._lwd_captured_sampler_output = sampler_output
        return sampler_output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: "GrammarOutput | None") -> ModelRunnerOutput:
        from vllm.distributed.parallel_state import get_tp_group

        # 只有云 TP 组首卡(= DOWN 通道端点 rank)采集/发送;
        # 其余 rank 无通道 peer,采集即弃也一并省掉(8 卡冗余)。
        wire_endpoint = get_tp_group().is_first_rank
        captured = None
        if wire_endpoint and self.execute_model_state is not None:
            # ExecuteModelState layout (see model_runner_v1.sample_tokens):
            # (scheduler_output, logits, spec_decode_metadata,
            #  spec_decode_common_attn_metadata, hidden_states,
            #  sample_hidden_states, ...)
            state = self.execute_model_state
            # 批序/掩码也一并快照:batch queue 重叠时,下一步的
            # _prepare_inputs 可能在 collect 前重排 input_batch,
            # 现读会拿到批序B去切批序A的张量(高并发错位根因)
            captured = (
                state[5], state[1], state[2], state[0],
                list(self.input_batch.req_ids),
                self.discard_request_mask.np.copy(),
            )
        self._lwd_captured_sampler_output = None
        output = super().sample_tokens(grammar_output)
        # VLLM_ASCEND_LWD_DISABLE_DOWN=1 诊断:不做 collect(云引擎改走
        # token_ids 直通通告),payload 为 None 时 worker 自然跳过 DOWN send
        if (captured is not None
                and self._lwd_captured_sampler_output is not None
                and not envs.VLLM_ASCEND_LWD_DISABLE_DOWN):
            self._lwd_pending_down_payload = self._lwd_collect_down_payload(
                captured[0], captured[1], captured[2], captured[3],
                captured[4], captured[5], self._lwd_captured_sampler_output,
            )
        return output

    def take_lwd_pending_down_payload(
        self,
    ) -> "tuple[torch.Tensor, torch.Tensor, None, list[str], None] | None":
        """worker 层取走本步 DOWN payload(单槽覆盖写,每步必被取走)。"""
        payload = getattr(self, "_lwd_pending_down_payload", None)
        self._lwd_pending_down_payload = None
        return payload

    @torch.inference_mode()
    def _lwd_collect_down_payload(
        self,
        sample_hidden_states: "torch.Tensor | None",
        logits: "torch.Tensor | None",
        spec_decode_metadata: "SpecDecodeMetadata | None",
        scheduler_output: "SchedulerOutput",
        batch_req_ids: "list[str]",
        discard_mask_np: "np.ndarray",
        sampler_output: "SamplerOutput",
    ) -> "tuple[torch.Tensor, torch.Tensor, None, list[str], None] | None":
        """生产级 DOWN 采集(rank-replay):hidden 组包 + 全局秩 + num_accepted。

        编排:行选择/秩计算按 spec 与否分流(_lwd_pack_plain_down_rows /
        _lwd_pack_spec_down_rows),meta 拼包与就绪事件在
        _lwd_stage_down_meta;计算关键路径零新增同步,批序/掩码取捕获
        时刻快照(防 batch queue 重叠重排)。返回:
          (hidden_packet, pinned_view, None, req_ids, None)
        或 None(本步无在途请求)。
        """
        _t0 = time.monotonic()
        sampled = sampler_output.sampled_token_ids
        if sampled is None or sampled.dim() != 2 or logits is None:
            return None
        # batch_req_ids/discard_mask 来自捕获时刻快照(与 logits/hidden
        # 同批序),不读活的 input_batch
        valid = ~discard_mask_np[: len(batch_req_ids)]
        if spec_decode_metadata is None:
            packed = self._lwd_pack_plain_down_rows(
                sample_hidden_states, logits, sampled, valid, batch_req_ids,
            )
        else:
            packed = self._lwd_pack_spec_down_rows(
                sample_hidden_states, logits, sampled, spec_decode_metadata,
                valid, batch_req_ids,
            )
        if packed is None:
            return None
        hidden_packet, ranks_list, counts_dev, seg_lens_host = packed
        if seg_lens_host is not None:
            meta_dev = torch.cat(ranks_list + [counts_dev])
            # seg_lens 不上设备(host 直写 pinned 尾段):与 D2H 前缀
            # 区域不相交,且引擎读 pinned 发生在 ready_event 同步
            # (更晚)之后,无竞态。
        else:
            n_req = sum(1 for i in range(len(batch_req_ids)) if valid[i])
            counts_dev = torch.ones(n_req, dtype=torch.int32,
                                    device=logits.device)
            meta_dev = torch.cat(ranks_list + [counts_dev, counts_dev])
        selection_done = time.monotonic()
        pinned_view, _n_meta = self._lwd_stage_down_meta(
            meta_dev, seg_lens_host,
        )
        # [Lwd][perf] 采集分段:rank=秩+行收集;pinned=meta 拼包+拷贝
        logger.info(
            "[Lwd][perf] collect rows=%d rank=%.2f pinned=%.2f total=%.2fms",
            hidden_packet.shape[0],
            (selection_done - _t0) * 1000,
            (time.monotonic() - selection_done) * 1000,
            (time.monotonic() - _t0) * 1000,
        )
        return (
            hidden_packet,
            pinned_view,
            None,
            [r for i, r in enumerate(batch_req_ids) if valid[i]],
            None,
        )

    def _lwd_pack_plain_down_rows(
        self,
        sample_hidden_states: torch.Tensor | None,
        logits: torch.Tensor,
        sampled: torch.Tensor,
        valid: np.ndarray,
        batch_req_ids: list[str],
    ) -> LwdDownSelection | None:
        """非 spec 路径的行选择与秩计算(对当关系:等价于原
        _lwd_collect_down_payload 的 not is_spec 分支)。

        廉价计算:bf16 直比(logits 原生 bf16,与 cast 后逐位等价)、
        批全覆盖时整行直接算(不做高级索引取材)。"""
        idx = [i for i in range(len(batch_req_ids)) if valid[i]]
        if not idx:
            return None
        full_cover = all(valid[: len(batch_req_ids)])
        lg_sel = logits if full_cover else logits[idx]
        sm_sel = sampled[:, 0] if full_cover else sampled[idx][:, 0]
        thresh = lg_sel.gather(1, sm_sel.long().unsqueeze(1))
        ranks = [(lg_sel > thresh).sum(dim=1).to(torch.int32)]
        hidden_packet = torch.cat(
            [sample_hidden_states[i : i + 1] for i in idx]
        )
        return LwdDownSelection(
            hidden_packet=hidden_packet, ranks=ranks,
            counts_dev=None, seg_lens_host=None,
        )

    def _lwd_pack_spec_down_rows(
        self,
        sample_hidden_states: torch.Tensor,
        logits: torch.Tensor,
        sampled: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata,
        valid: np.ndarray,
        batch_req_ids: list[str],
    ) -> LwdDownSelection | None:
        """全零同步 spec 路径(向量化)的行选择与秩计算(对当关系:等价
        于原 _lwd_collect_down_payload 的 is_spec 分支)。

        段长按 host 侧 scheduled_spec_decode_tokens 推(1+draft_len),
        按完整段打包(含被拒行,边侧按 num_accepted 取有效前缀);
        段长取 spec_decode_metadata.num_draft_tokens(host list,
        运行期真实布局,与 sample_hidden_states 段结构一致);
        不用 scheduler_output.scheduled_spec_decode_tokens
        (调度输入,可能与实际运行不一致)。
        counts 从同一个 sampled 张量 device 推导(与秩/行同源,
        天然按批位对齐);seg_lens/counts 只收 valid 请求,
        保证 [ranks|counts|seg_lens] 三段长度一致。
        host 一趟建索引,替代逐请求 device 循环(原实现每请求
        ~6 次小 kernel 发射,rank 段随并发涨到 20+ms;向量化后
        发射数与 N 无关)。索引语义与原循环逐位等价:
        vidx/vlen = 有效请求批位/实际行数(按总段长封顶,
        = 原 rows_i);req_idx/pos_idx = 每行的请求位/行内位
        (原 sampled[i, :rows_i]);row_sel = 有效行全局行号。"""
        seg_lens = [
            d + 1 for d in spec_decode_metadata.num_draft_tokens
        ]
        counts_all = (sampled != -1).sum(dim=1)
        seg_lens_np = np.asarray(seg_lens, dtype=np.int32)
        seg_starts = np.cumsum(seg_lens_np) - seg_lens_np
        total_rows = sample_hidden_states.shape[0]
        rows_per_req = np.minimum(
            seg_lens_np, np.maximum(total_rows - seg_starts, 0)
        )
        vidx_np = np.nonzero(
            valid[: len(batch_req_ids)] & (rows_per_req >= 1)
        )[0].astype(np.int64)
        if vidx_np.size == 0:
            return None
        vlen_np = rows_per_req[vidx_np].astype(np.int64)
        starts_v = seg_starts[vidx_np].astype(np.int64)
        n_rows = int(vlen_np.sum())
        req_idx_np = np.repeat(vidx_np, vlen_np)
        pos_idx_np = np.arange(n_rows, dtype=np.int64) - np.repeat(
            np.cumsum(vlen_np) - vlen_np, vlen_np
        )
        row_sel_np = np.repeat(starts_v, vlen_np) + pos_idx_np
        # 全部索引一次 pinned 异步 H2D(pageable 同步 H2D 会在
        # 主流上等设备排空,见 seg_lens 同款教训)
        buf = self._lwd_h2d_stage_i64(
            np.concatenate(
                [req_idx_np, pos_idx_np, row_sel_np, vidx_np, vlen_np]
            )
        )
        n_valid = vidx_np.size
        req_d, pos_d = buf[:n_rows], buf[n_rows : 2 * n_rows]
        sel_d = buf[2 * n_rows : 3 * n_rows]
        vidx_d = buf[3 * n_rows : 3 * n_rows + n_valid]
        vlen_d = buf[3 * n_rows + n_valid :]
        sampled_rows = sampled[req_d, pos_d]
        if n_rows == total_rows:
            # 全 valid 无封顶:行即连续前缀,免 gather
            lg_rows = logits[:n_rows]
            hidden_packet = sample_hidden_states[:n_rows]
        else:
            lg_rows = logits[sel_d]
            hidden_packet = sample_hidden_states[sel_d]
        thresh = lg_rows.gather(1, sampled_rows.long().unsqueeze(1))
        ranks = [(lg_rows > thresh).sum(dim=1).to(torch.int32)]
        # 过期 sampled 位(spec 未运行的请求可能残留上个 spec 步的
        # token):accepted/秩/段长一律按实际 hidden 行数封顶——
        # 行数才是真实采样位置的真相
        counts_dev = torch.minimum(counts_all[vidx_d], vlen_d).to(
            torch.int32
        )
        return LwdDownSelection(
            hidden_packet=hidden_packet, ranks=ranks,
            counts_dev=counts_dev, seg_lens_host=vlen_np.tolist(),
        )

    def _lwd_stage_down_meta(
        self,
        meta_dev: torch.Tensor,
        seg_lens_host: list[int] | None,
    ) -> tuple[torch.Tensor, int]:
        """meta 拼包与就绪事件:pinned 拷贝排主流末尾(异步),紧随记录
        就绪事件(经 worker.note_lwd_meta_ready);由 worker 响应入队处
        在发送前 synchronize——"响应发出 ⟹ pinned 就绪"成为硬保证
        (同步在输出线程,不在计算关键路径)。返回 (pinned 视图, 长度)。"""
        n_dev = meta_dev.numel()
        n_meta = n_dev + (len(seg_lens_host) if seg_lens_host is not None else 0)
        pinned = self._lwd_meta_pinned(n_meta)
        pinned[:n_dev].copy_(meta_dev, non_blocking=True)
        if seg_lens_host is not None:
            pinned.numpy()[n_dev:n_meta] = seg_lens_host
        self.worker.note_lwd_meta_ready()
        return pinned[:n_meta], n_meta

    def _lwd_meta_pinned(self, n: int) -> "torch.Tensor":
        """轮换 pinned 缓冲(深度 4 > batch_queue 深度 2 + 引擎滞后 1):
        避免下一/N+2 步的边流拷贝覆盖引擎尚未读完的上一步 meta。
        对当关系:LwdPinnedStaging.next_pinned ≍ 原手写 4 槽列表轮换
        (容量下限 4096 同旧值)。"""
        ring = getattr(self, "_lwd_meta_ring", None)
        if ring is None:
            ring = LwdPinnedStaging(dtype=torch.int32, min_capacity=4096)
            self._lwd_meta_ring = ring
        return ring.next_pinned(n)

    def _lwd_h2d_stage_i64(self, arr: "np.ndarray") -> torch.Tensor:
        """int64 host 索引数组的异步 H2D 暂存(pinned 4 槽轮换 +
        non_blocking):替代 torch.tensor(arr, device=...) 的 pageable
        同步拷贝——后者在主流上等设备排空(MTP 下即 draft propose
        尾巴)。环深与 meta ring 同理:覆盖 batch_queue 深度 + 引擎滞后,
        防止下步 host 覆写时上步拷贝尚未执行。
        对当关系:LwdPinnedStaging.stage_to_device ≍ 原手写 pinned+device 4 槽对
        轮换(容量下限 1024 同旧值)。"""
        ring = getattr(self, "_lwd_h2d_ring", None)
        if ring is None:
            ring = LwdPinnedStaging(
                dtype=torch.int64, min_capacity=1024, device=self.device,
            )
            self._lwd_h2d_ring = ring
        return ring.stage_to_device(arr, arr.size)
