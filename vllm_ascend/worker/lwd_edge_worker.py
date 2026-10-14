#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""LWD (layerwise disaggregated) prefill_only mode edge worker."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.multimodal.utils import group_and_batch_mm_kwargs
from vllm.v1.core.sched.output import (
    LwdBatchType,
    LwdEmbedBatch,
    LwdUnembedBatch,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes

from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import pack_hash_payload
from vllm_ascend.worker.lwd_hash.lwd_hash_tables import load_tid2eid_tables
from vllm_ascend.worker.lwd_worker import LwdWorker

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheSpec


_LWD_MM_HASH_CACHE_MAX_ITEMS = 32


# ---- token recovery / logit-rank lookup ----
def select_token_batch(logits: torch.Tensor, ranks: list[int]) -> list[int]:
    """Batched rank→token lookup: one sort, one D2H sync for all rows.

    logits [R,V] rows sorted together (NPU sorts rows in parallel);
    sorted_idx[i, ranks[i]] is the token id at rank ranks[i] for row i.
    """

    _, sorted_idx = torch.sort(logits, dim=-1, descending=True, stable=True)
    row = torch.arange(len(ranks), device=logits.device)
    return sorted_idx[row, torch.tensor(ranks, device=logits.device)].tolist()


def compute_top_id_th(logits: torch.Tensor, token_id: int) -> int:
    """Look up the descending-logits ordinal of ``token_id`` (cloud side)."""
    logits = logits.reshape(-1)

    order = torch.argsort(logits, descending=True, stable=True)
    return int((order == token_id).nonzero()[0].item())


# ---- edge worker ----
class LwdEdgeWorker(LwdWorker):
    """LWD edge worker: embed (prefill) + unembed (token recovery) only."""

    def load_model(self) -> None:
        """Load edge embedding/head weights and any CPU Hash MoE tables."""
        super().load_model()
        self.tid2eid_tables: list[torch.Tensor] = []
        config = self.model_config.hf_config
        if getattr(config, "model_type", None) != "deepseek_v4":
            return
        num_hash_layers = config.num_hash_layers
        if not 0 <= num_hash_layers <= config.num_hidden_layers:
            raise ValueError(f"Invalid DeepSeek V4 num_hash_layers: {num_hash_layers}")
        if num_hash_layers == 0:
            logger.debug("[LWD][edge] tid2eid loading skipped: num_hash_layers=0")
            return
        logger.debug("[LWD][edge] loading tid2eid tables for %d Hash MoE layers", num_hash_layers)
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
            weight_files,
            num_hash_layers,
            config.vocab_size,
            config.num_experts_per_tok,
            config.n_routed_experts,
        )

    def init_device(self) -> None:
        super().init_device()
        self._skip_sample = LwdControlDiag.token_passthrough_enabled()
        self._lwd_mm_features_dict: dict[str, list[MultiModalFeatureSpec]] = {}
        self._lwd_mm_hash_embeds: dict[str, torch.Tensor] = {}

    def compile_or_warm_up_model(self) -> CompilationTimes:
        logger.debug("[LWD][edge] skip model warmup/capture (forward is hijacked by LWD)")
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """The LWD edge runs no attention/transformer, so it needs no KV cache."""
        return {}

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        lwd_batch = scheduler_output.lwd_batch

        if scheduler_output.finished_req_ids:
            for req_id in scheduler_output.finished_req_ids:
                self._lwd_mm_release_request(req_id)
        if lwd_batch is None:
            logger.debug("[LWD][edge] step carries no LWD batch; nothing to do")
            return None

        self.validate_lwd_batch(lwd_batch)
        self.comm_service.poll_completions()
        batch_meta = lwd_batch.batch_meta
        if lwd_batch.batch_type == LwdBatchType.LWD_EMBED:
            logger.debug(
                "[LWD][edge-worker] EMBED seqno=%d reqs=%d tokens=%d has_mrope=%s",
                lwd_batch.seqno,
                len(batch_meta.req_ids),
                sum(len(token_ids) for token_ids in batch_meta.token_ids),
                batch_meta.has_mrope,
            )
            return self._execute_lwd_embed(lwd_batch.seqno, batch_meta, scheduler_output)
        if lwd_batch.batch_type == LwdBatchType.LWD_UNEMBED:
            logger.debug(
                "[LWD][edge-worker] UNEMBED seqno=%d reqs=%d accepted=%d num_elements=%d",
                lwd_batch.seqno,
                len(batch_meta.req_ids),
                sum(batch_meta.num_accept_tokens),
                batch_meta.recv_num_elements,
            )
            return self._execute_lwd_unembed(lwd_batch.seqno, batch_meta)

        logger.debug(
            "[LWD][edge] unknown LWD batch type %r; nothing to do",
            lwd_batch.batch_type,
        )
        return None

    def _lwd_embed_ids_h2d(self, flat_token_ids: list[int]) -> torch.Tensor:
        """Stage token IDs in a pinned ring without synchronizing prior wire sends."""
        n = len(flat_token_ids)
        ring = getattr(self, "_lwd_ids_ring", None)
        if ring is None or ring[0][0].numel() < n:
            m = max(n, 8192)
            ring = [
                (
                    torch.empty(m, dtype=torch.int64, pin_memory=True),
                    torch.empty(m, dtype=torch.int64, device=self.model_runner.device),
                )
                for _ in range(4)
            ]
            self._lwd_ids_ring = ring
            self._lwd_ids_ring_idx = 0
        idx = self._lwd_ids_ring_idx
        self._lwd_ids_ring_idx = (idx + 1) % len(ring)
        stage, dev = ring[idx]
        stage.numpy()[:n] = flat_token_ids
        dev[:n].copy_(stage[:n], non_blocking=True)
        return dev[:n]

    @torch.inference_mode()
    def _execute_lwd_embed(self, seqno: int, batch_meta: LwdEmbedBatch, scheduler_output: SchedulerOutput) -> None:
        model = self.model_runner.get_model()
        if not batch_meta.token_ids:
            return

        for req_data in scheduler_output.scheduled_new_reqs:
            feats = [f for f in req_data.mm_features if f.modality != "prompt_embeds"]
            if feats:
                self._lwd_mm_features_dict[req_data.req_id] = feats

        # Flatten all requests' prompt tokens into one batch (order = req_ids).
        flat_token_ids = [tid for token_ids in batch_meta.token_ids for tid in token_ids]
        logger.debug("[LWD][edge-worker] embed token_ids=%s", flat_token_ids)
        token_ids_tensor = self._lwd_embed_ids_h2d(flat_token_ids)
        _t0 = time.monotonic()

        mm_embeds, is_mm = self._lwd_gather_chunk_mm(batch_meta)
        if mm_embeds:
            embeds = model.embed_input_ids(
                token_ids_tensor,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm,
            )
        else:
            embeds = model.embed_input_ids(token_ids_tensor)  # (total_N, H)
        _t_fwd = time.monotonic()
        if not self.lwd_wire.member:
            return
        payload = embeds
        if self.tid2eid_tables:
            # The original prompt IDs stay on the edge; transmit only lookups.
            cpu_ids = torch.tensor(flat_token_ids, dtype=torch.int64)
            expert_ids = torch.stack([table[cpu_ids] for table in self.tid2eid_tables], dim=1)
            payload = pack_hash_payload(embeds, expert_ids)
            logger.debug(
                "[LWD][edge] UP Hash MoE seqno=%d expert_shape=%s payload_numel=%d",
                seqno,
                tuple(expert_ids.shape),
                payload.numel(),
            )

        request = LwdCommRequest(
            connection_key=self.lwd_wire.key,
            channel=LwdChannelType.UP,
            op="send",
            num_elements=payload.numel(),
            tensor=payload,
            seqno=seqno,
        )
        self.comm_service.submit_send(request)
        if seqno == 0:
            logger.info(
                "[LWD] first UP submitted: connection=%s rows=%d dtype=%s",
                self.lwd_wire.key,
                len(flat_token_ids),
                payload.dtype,
            )

        logger.debug(
            "[LWD][perf] embed seqno=%s forward=%.2f submit_send=%.2f total=%.2fms tokens=%d",
            seqno,
            (_t_fwd - _t0) * 1000,
            (time.monotonic() - _t_fwd) * 1000,
            (time.monotonic() - _t0) * 1000,
            len(flat_token_ids),
        )

    # ------------------------------------------------------------------ #
    # Multimodal (image) merge helpers                                     #
    # ------------------------------------------------------------------ #

    def _lwd_mm_encode_missing(self, req_ids: list[str]) -> None:
        """Batch missing image features by modality and cache their encoder outputs."""
        missing_features = []
        missing_data = []
        for req_id in req_ids:
            for f in self._lwd_mm_features_dict.get(req_id) or []:
                if f.identifier not in self._lwd_mm_hash_embeds:
                    if f.data is None:
                        missing_data.append((req_id, f.identifier))
                    else:
                        missing_features.append(f)
        if missing_data:
            raise RuntimeError(
                f"[LWD] mm feature data missing: {missing_data}; "
                "cannot compute image embeddings; "
                "check mm processor/receiver cache chain"
            )
        if not missing_features:
            return
        _t_enc = time.monotonic()
        model = self.model_runner.get_model()
        mm_kwargs = [(f.modality, f.data) for f in missing_features]
        outputs_by_modality: dict[str, list[torch.Tensor]] = {}
        num_outputs = 0
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs,
            device=self.model_runner.device,
            pin_memory=getattr(self.model_runner, "pin_memory", False),
        ):
            batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
            assert len(batch_outputs) == num_items, (
                f"encoder outputs {len(batch_outputs)} != items {num_items} (modality={modality})"
            )
            outputs_by_modality.setdefault(modality, []).extend(batch_outputs)
            num_outputs += len(batch_outputs)
        assert num_outputs == len(missing_features), (
            f"encoder outputs {num_outputs} != missing features {len(missing_features)}"
        )
        for feature in missing_features:
            self._lwd_mm_hash_embeds[feature.identifier] = outputs_by_modality[feature.modality].pop(0)

        while len(self._lwd_mm_hash_embeds) > _LWD_MM_HASH_CACHE_MAX_ITEMS:
            self._lwd_mm_hash_embeds.pop(next(iter(self._lwd_mm_hash_embeds)))

        logger.debug(
            "[LWD][perf] mm-encoder reqs=%d items=%d dur=%.2fms",
            len(req_ids),
            len(missing_features),
            (time.monotonic() - _t_enc) * 1000,
        )

    def _lwd_mm_encoder_outputs(self, req_id: str) -> list[tuple[PlaceholderRange, torch.Tensor]]:
        """Return this request's encoded multimodal features in original order."""
        features = self._lwd_mm_features_dict.get(req_id)
        if not features:
            return []
        missing = [f.identifier for f in features if f.identifier not in self._lwd_mm_hash_embeds]
        if missing:
            raise RuntimeError(f"[LWD] mm embeds missing from batch encode: req={req_id} (mm_hashes={missing})")
        return [(f.mm_position, self._lwd_mm_hash_embeds[f.identifier]) for f in features]

    def _lwd_mm_release_request(self, req_id: str) -> None:
        """Release request references to cached image embeddings."""
        self._lwd_mm_features_dict.pop(req_id, None)

    def _lwd_gather_chunk_mm(self, batch_meta: LwdEmbedBatch) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Select only the image rows overlapping this prompt chunk."""
        total = sum(len(t) for t in batch_meta.token_ids)
        is_mm = torch.zeros(total, dtype=torch.bool)
        mm_embeds: list[torch.Tensor] = []
        if len(batch_meta.prompt_offsets) != len(batch_meta.req_ids):
            raise ValueError("[LWD] each prompt chunk requires an absolute offset")
        self._lwd_mm_encode_missing(batch_meta.req_ids)
        req_start = 0
        for req_id, token_ids, p_offset in zip(batch_meta.req_ids, batch_meta.token_ids, batch_meta.prompt_offsets):
            n = len(token_ids)
            cached = self._lwd_mm_encoder_outputs(req_id)
            max_end = 0
            for pos_info, item_embeds in cached:
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length
                max_end = max(max_end, start_pos + num_encoder_tokens)
                start_idx = max(p_offset - start_pos, 0)
                end_idx = min(p_offset - start_pos + n, num_encoder_tokens)
                if start_idx >= end_idx:
                    continue
                req_start_pos = req_start + start_pos - p_offset
                if (is_embed := pos_info.is_embed) is not None:
                    s, e = pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                    if s == e:
                        continue
                    rows = item_embeds[s:e]
                    is_mm[req_start_pos + start_idx : req_start_pos + end_idx] |= is_embed[start_idx:end_idx]
                else:
                    rows = item_embeds[start_idx:end_idx]
                    is_mm[req_start_pos + start_idx : req_start_pos + end_idx] = True
                mm_embeds.append(rows)
            if cached and p_offset + n >= max_end:
                self._lwd_mm_release_request(req_id)
            req_start += n
        return mm_embeds, is_mm

    @torch.inference_mode()
    def _execute_lwd_unembed(self, seqno: int, batch_meta: LwdUnembedBatch) -> ModelRunnerOutput | None:
        model = self.model_runner.get_model()
        if not batch_meta.req_ids:
            return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])

        _t0 = time.monotonic()
        recv_future = None
        if self.lwd_wire.member:
            recv_future = self.comm_service.submit_recv(
                LwdCommRequest(
                    connection_key=self.lwd_wire.key,
                    channel=LwdChannelType.DOWN,
                    op="recv",
                    num_elements=batch_meta.recv_num_elements,
                    seqno=seqno,
                )
            )
        _t_post = time.monotonic()
        hidden_states = self.receive_lwd_tensor(recv_future, batch_meta.recv_num_elements)
        hidden_states = hidden_states.view(-1, self.model_config.get_hidden_size())
        _t_tensor = time.monotonic()

        if self._skip_sample and batch_meta.token_ids:
            assert len(batch_meta.token_ids) == len(batch_meta.req_ids)
            sampled_token_ids: list[list[int]] = [list(t) for t in batch_meta.token_ids]
            _t_lm = time.monotonic()
        else:
            logits = model.compute_logits(hidden_states)  # (rows_total, V)
            _t_lm = time.monotonic()
            if logits is None:
                # Non-output TP ranks still participate in the vocab collective.
                return None

            flat_ids = select_token_batch(logits, [r for ranks in batch_meta.top_id_ths for r in ranks])
            sampled_token_ids = []
            off = 0
            for accept, ranks in zip(batch_meta.num_accept_tokens, batch_meta.top_id_ths):
                sampled_token_ids.append(flat_ids[off : off + accept])
                off += len(ranks)
        _t_sel = time.monotonic()

        logger.debug(
            "[LWD][perf] unembed seqno=%s post_recv=%.2f wait_tensor=%.2f "
            "lm_head=%.2f select=%.2f total=%.2fms reqs=%d",
            seqno,
            (_t_post - _t0) * 1000,
            (_t_tensor - _t_post) * 1000,
            (_t_lm - _t_tensor) * 1000,
            (_t_sel - _t_lm) * 1000,
            (_t_sel - _t0) * 1000,
            len(batch_meta.req_ids),
        )
        if not get_tp_group().is_first_rank:
            return None
        req_id_to_index = {req_id: index for index, req_id in enumerate(batch_meta.req_ids)}
        return ModelRunnerOutput(
            req_ids=batch_meta.req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
        )
