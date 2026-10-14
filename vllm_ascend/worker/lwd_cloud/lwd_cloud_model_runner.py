# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Receive prompt chunks and return hidden rows with token-rank metadata."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, LwdEmbedBatch, SchedulerOutput
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput, SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.worker.lwd_cloud.lwd_draft_cache import LwdDraftCache
from vllm_ascend.worker.lwd_cloud.lwd_mtp_proposer import LwdMTPProposer
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import (
    LwdHashRoutingState,
    hash_layer_count,
    hash_payload_numel,
    unpack_hash_payload,
)
from vllm_ascend.worker.lwd_token_rank import LwdTokenRankEncoder
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

if TYPE_CHECKING:
    from vllm_ascend.worker.lwd_cloud.lwd_cloud_worker import LwdCloudWorker

LwdDownPayload = tuple[torch.Tensor, torch.Tensor, torch.npu.Event, list[str], None]


class LwdCloudModelRunner(NPUModelRunner):
    """NPUModelRunner + prefill_only LWD cloud-side runner logic."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, worker: LwdCloudWorker | None = None) -> None:
        super().__init__(vllm_config, device)
        self._disable_down = LwdControlDiag.disable_down_enabled()
        self.worker = worker  # LwdCloudWorker ref (UP recv futures live there)
        self._lwd_pending_down_payload = None
        self._lwd_token_rank_encoder = LwdTokenRankEncoder()
        self._lwd_draft_cache = (
            LwdDraftCache()
            if self.use_async_scheduling
            and self.speculative_config is not None
            and self.speculative_config.method == "mtp"
            and self.pcp_size == 1
            and self.dcp_size == 1
            else None
        )
        # Stage per-request prompt lengths for device-side mask refresh.
        if self.use_async_spec_decode and self.enable_prompt_embeds:
            self._lwd_prompt_lens_gpu = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
        else:
            self._lwd_prompt_lens_gpu = None
        self.lwd_hash_state = None

        self._lwd_hash_step_tokens = None
        config = self.model_config.hf_config
        num_hash_layers = hash_layer_count(config)
        lwd_config = getattr(vllm_config, "lwd_config", None)
        if num_hash_layers and lwd_config is not None:
            # UP currently broadcasts one edge's batch to one cloud TP group.
            # CP/DP need additional request-to-rank layouts before supporting V4.
            parallel = vllm_config.parallel_config
            if (
                parallel.data_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
            ):
                raise ValueError("LWD V4 Hash routing currently requires DP=PCP=DCP=1 (TP/EP are supported)")
            self.lwd_hash_state = LwdHashRoutingState(
                num_hash_layers,
                config.num_experts_per_tok,
                config.n_routed_experts,
            )
            # Persistent buffers are populated before model execution, so graph
            # capture and replay see the same addresses for prompt and decode.
            self._lwd_hash_experts = torch.zeros(
                self.max_num_tokens,
                num_hash_layers,
                config.num_experts_per_tok,
                dtype=torch.int32,
                device=device,
            )
            self._lwd_hash_prompt_mask = torch.zeros(self.max_num_tokens, dtype=torch.bool, device=device)
            self._lwd_hash_token_ids = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        for request in scheduler_output.scheduled_new_reqs:
            shape = request.lwd_prompt_embeds_shape
            if shape is not None:
                if request.lwd_prompt_embeds_dtype is None:
                    raise ValueError("[LWD] prompt placeholder is missing its dtype")
                # Native bookkeeping needs the shape. Only received device rows
                # are consumed; the inherited CPU embedding copy is bypassed.
                request.prompt_embeds = torch.empty(shape, dtype=request.lwd_prompt_embeds_dtype)
        # Called inside synchronize_input_prep(), before remove_request()
        # destroys the previous mapping and before any new graph replay.
        if self._lwd_draft_cache is not None:
            self._lwd_draft_cache.preserve(
                scheduler_output,
                self.requests,
                self.input_batch.prev_req_id_to_index,
                self._draft_token_ids,
            )
        return super()._update_states(scheduler_output)

    def _prepare_input_ids(
        self,
        scheduler_output: SchedulerOutput,
        num_reqs: int,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        super()._prepare_input_ids(
            scheduler_output,
            num_reqs,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )
        self._lwd_restore_drafts(scheduler_output, num_reqs, cu_num_tokens)

    def _lwd_restore_drafts(self, scheduler_output: SchedulerOutput, num_reqs: int, cu_num_tokens: np.ndarray) -> None:
        cache = self._lwd_draft_cache
        if cache is None or not cache.entries:
            return
        rows, indices = [], []
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            if req_id not in cache.entries:
                continue
            tokens = scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            row = cache.take(req_id, self.requests[req_id], len(tokens))
            # Consume the generation even when the scheduler drops all drafts.
            if not tokens or self.prev_positions.np[idx] >= 0:
                continue
            if row is None:
                raise RuntimeError(f"LWD draft snapshot is invalid for {req_id}")
            end = int(cu_num_tokens[idx])
            begin = int(cu_num_tokens[idx - 1]) if idx else 0
            if end - begin < len(tokens) + 1:
                raise RuntimeError(f"LWD draft segment is too short for {req_id}")
            rows.append(row)
            indices.extend(range(end - len(tokens), end))
        if rows:
            # Same stream as native input assembly and subsequent metadata /
            # embedding construction. No device-to-host copy or host wait.
            dst = torch.tensor(indices, dtype=torch.int64, pin_memory=self.pin_memory).to(
                self.device,
                non_blocking=True,
            )
            self.input_ids.gpu.scatter_(
                0,
                dst,
                torch.cat(rows).to(dtype=self.input_ids.gpu.dtype),
            )

    def _get_drafter(self) -> LwdMTPProposer:
        """Use the LWD MTP embedding adapter only in prefill_only mode."""
        lwd_config = getattr(self.vllm_config, "lwd_config", None)
        if (
            lwd_config is not None
            and self.speculative_config.method == "mtp"
            and not self.speculative_config.use_step3p5_mtp()
        ):
            return LwdMTPProposer(self.vllm_config, self.device, self)
        raise ValueError("[LWD] the migrated speculative path supports standard MTP only")

    # ------------------------------------------------------------------ #
    # Remote embeds injection (cloud input has NO token ids — the prompt   #
    # embeddings arrive over the UP channel and must drive forward)        #
    # ------------------------------------------------------------------ #

    def _prepare_inputs(
        self, scheduler_output: SchedulerOutput, num_scheduled_tokens: np.ndarray
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None, int]:
        """Install mRoPE positions before native position preparation, then inject embeddings.

        The native CPU placeholder copy is bypassed; received device rows are
        authoritative for the target and MTP first pass."""
        harvested = self._lwd_harvest_up_batches(num_scheduled_tokens)
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

    def _lwd_prepare_hash_batch(self, num_scheduled_tokens: np.ndarray) -> None:
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
            self.input_batch.req_ids,
            counts,
            computed,
            prompt_lengths,
        )
        self._lwd_hash_experts.zero_()
        self._lwd_hash_experts[: batch.shape[0]].copy_(batch)
        self._lwd_hash_prompt_mask[: mask.shape[0]].copy_(mask)

    def _lwd_rebuild_is_token_ids_mask(self, num_scheduled_tokens: np.ndarray) -> None:
        """Identify prompt rows from request lengths instead of optimistic draft positions.

        Async decode may overestimate the previous acceptance count. Those rows
        must still be embedded from token IDs instead of using stale buffers."""
        if not self.input_batch.req_prompt_embeds:
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
            prefill_rows = min(n, max(int(num_prompt[i]) - int(computed[i]), 0))
            if prefill_rows > 0:
                mask[off : off + prefill_rows] = False
            off += n

    def _preprocess(
        self,
        scheduler_output: SchedulerOutput,
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None, torch.Tensor | None, torch.Tensor, IntermediateTensors | None, dict[str, object], object
    ]:
        """Refresh decode masks after native async position correction."""
        if (
            self.use_async_spec_decode
            and self.enable_prompt_embeds
            and intermediate_tensors is None
            and self.pcp_size == 1
        ):
            self._lwd_refresh_decode_embedding_mask(scheduler_output)
            if (
                get_pp_group().is_first_rank
                and not self.model_config.is_encoder_decoder
                and (not self.supports_mm_inputs or self.input_batch.req_prompt_embeds)
            ):
                return self._lwd_preprocess_prompt_embeds(scheduler_output, num_input_tokens)
            # Other model paths keep the parent's preprocessing, which may
            # still consume the CPU mask (for example encoder-decoder models).
            count = scheduler_output.total_num_scheduled_tokens
            self.is_token_ids.np[:count] = self.is_token_ids.gpu[:count].cpu().numpy()
        return super()._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)

    def _lwd_preprocess_prompt_embeds(
        self, scheduler_output: SchedulerOutput, num_input_tokens: int
    ) -> tuple[None, torch.Tensor, torch.Tensor, None, dict[str, object], None]:
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

    def _lwd_refresh_decode_embedding_mask(self, scheduler_output: SchedulerOutput) -> None:
        """Build the actual prompt/decode mask on device, avoiding a positions D2H copy."""
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

    def _dummy_run(self, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        """Clear request routing state before synthetic forwards."""

        self._lwd_hash_step_tokens = None
        return super()._dummy_run(*args, **kwargs)

    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Provide full-token Hash routing data; native models own SP slicing."""
        context = get_forward_context()
        if self.lwd_hash_state is not None:
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
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )

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

    def _lwd_harvest_up_batches(
        self, num_scheduled_tokens: np.ndarray
    ) -> list[tuple[int, LwdEmbedBatch, torch.Tensor]]:
        """Receive the complete batch, broadcast within cloud TP and check prompt windows."""
        worker = self.worker
        if worker is None:
            return []
        posted = getattr(worker, "_lwd_up_recv_futures", None)
        if not posted:
            return []
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        hidden_size = self.model_config.get_hidden_size()

        harvested = []
        for batch_seqno in list(posted.keys()):
            item = worker._lwd_up_recv_futures.pop(batch_seqno, None)
            if item is None:
                continue
            future, meta = item
            _t_wait = time.monotonic()
            num_tokens = sum(len(t) for t in meta.token_ids)
            config = self.model_config.hf_config
            numel = hash_payload_numel(
                num_tokens,
                hidden_size,
                hash_layer_count(config),
                getattr(config, "num_experts_per_tok", 0),
            )
            if not (
                len(meta.req_ids)
                == len(meta.token_ids)
                == len(meta.prompt_offsets)
                == len(meta.has_mrope)
                == len(meta.mrope_positions)
            ):
                raise ValueError("LWD requires one prompt offset and mrope flag per request")
            up_flat = worker.receive_lwd_tensor(future, numel)
            logger.debug(
                "[LWD][perf] cloud up-recv-wait seqno=%s dur=%.2fms",
                batch_seqno,
                (time.monotonic() - _t_wait) * 1000,
            )
            logger.debug(
                "[LWD][cloud-runner] UP embeds harvested seqno=%s rows=%d reqs=%s mrope_reqs=%s",
                batch_seqno,
                num_tokens,
                meta.req_ids,
                sum(meta.has_mrope),
            )

            mrope_rows = 0
            _mrope_cpu_ms = _mrope_delta_ms = 0.0
            for k, (req_id, token_ids) in enumerate(zip(meta.req_ids, meta.token_ids)):
                n = len(token_ids)
                if req_id in meta.discard_req_ids:
                    continue
                idx = self.input_batch.req_id_to_index.get(req_id)
                if idx is None:
                    raise RuntimeError(
                        f"[LWD][cloud-runner] INJECT req={req_id} seqno="
                        f"{batch_seqno} rows={n}: req not in input_batch "
                        f"(batch={self.input_batch.req_ids})"
                    )
                if n > 0:
                    scheduled = int(num_scheduled_tokens[idx]) if idx < len(num_scheduled_tokens) else 0
                    if n != scheduled:
                        raise RuntimeError(
                            f"[LWD][cloud-runner] INJECT window mismatch "
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
                        payload = meta.mrope_positions[k]
                        if not payload:
                            raise RuntimeError(
                                f"[LWD][cloud-runner] mrope payload missing "
                                f"req={req_id} seqno={batch_seqno}: "
                                f"has_mrope=True but control positions are empty"
                                f"(edge/cloud protocol mismatch)"
                            )
                        cpu_ms, delta_ms = self._lwd_inject_mrope_positions(
                            req_id,
                            start,
                            n,
                            torch.frombuffer(bytearray(payload), dtype=torch.int64).view(3, n),
                            int(num_prompt[idx]),
                        )
                        _mrope_cpu_ms += cpu_ms
                        _mrope_delta_ms += delta_ms
                        mrope_rows += n
            if mrope_rows > 0:
                logger.debug(
                    "[LWD][perf] cloud mrope-inject seqno=%s rows=%d cpu=%.2f delta=%.2fms",
                    batch_seqno,
                    mrope_rows,
                    _mrope_cpu_ms,
                    _mrope_delta_ms,
                )
            harvested.append((batch_seqno, meta, up_flat))
        return harvested

    def _lwd_inject_remote_embeds(
        self, harvested: list[tuple[int, LwdEmbedBatch, torch.Tensor]], num_scheduled_tokens: np.ndarray
    ) -> None:
        """Copy each received chunk into its scheduled device window.

        Record the consuming stream so the allocator cannot reuse a P2P buffer
        while that stream is still copying it. MTP reads this same device window."""
        if not harvested:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        hidden_size = self.model_config.get_hidden_size()
        gpu_embeds = self.inputs_embeds.gpu

        out_offset: dict[str, int] = {}
        off = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            out_offset[req_id] = off
            off += int(num_scheduled_tokens[i]) if i < len(num_scheduled_tokens) else 0

        for batch_seqno, meta, up_flat in harvested:
            num_tokens = sum(len(t) for t in meta.token_ids)
            config = self.model_config.hf_config
            num_hash_layers = hash_layer_count(config)
            expert_ids = None
            if num_hash_layers:
                embeds, expert_ids = unpack_hash_payload(
                    up_flat,
                    num_tokens,
                    hidden_size,
                    num_hash_layers,
                    config.num_experts_per_tok,
                )
            else:
                embeds = up_flat.view(num_tokens, hidden_size)
            logger.debug(
                "[LWD][cloud-runner] UP embeds injected seqno=%s rows=%s reqs=%s",
                batch_seqno,
                embeds.shape[0],
                meta.req_ids,
            )
            row = 0
            for meta_idx, (req_id, token_ids) in enumerate(zip(meta.req_ids, meta.token_ids)):
                n = len(token_ids)
                if req_id in meta.discard_req_ids:
                    row += n
                    continue
                idx = self.input_batch.req_id_to_index.get(req_id)
                if idx is None:
                    raise RuntimeError(
                        f"[LWD][cloud-runner] INJECT req={req_id} seqno="
                        f"{batch_seqno} rows={n}: req not in input_batch "
                        f"(batch={self.input_batch.req_ids})"
                    )
                if n > 0:
                    start = int(meta.prompt_offsets[meta_idx])
                    if self.lwd_hash_state is not None:
                        if expert_ids is None:
                            raise ValueError(f"Missing LWD V4 expert payload for {req_id}")
                        self.lwd_hash_state.add_chunk(
                            req_id,
                            int(num_prompt[idx]),
                            start,
                            expert_ids[row : row + n],
                        )
                    out = out_offset.get(req_id, 0)

                    gpu_embeds[out : out + n].copy_(embeds[row : row + n], non_blocking=True)

                    self.input_batch.is_token_ids[idx, start : start + n] = False
                row += n

            up_flat.record_stream(torch.npu.current_stream())

    # ------------------------------------------------------------------ #
    # Hidden-state and rank collection                                    #
    # ------------------------------------------------------------------ #

    def _lwd_inject_mrope_positions(
        self, req_id: str, start: int, n: int, chunk_positions: torch.Tensor, prompt_len: int
    ) -> tuple[float, float]:
        """Install chunk positions and derive the decode delta from the final prompt chunk."""
        req_state = self.requests.get(req_id)
        assert chunk_positions.shape == (3, n), (
            f"mrope payload shape {tuple(chunk_positions.shape)} != (3, {n}) (req={req_id})"
        )
        _t_cpu = time.monotonic()
        req_state.mrope_positions[:, start : start + n] = chunk_positions
        cpu_ms = (time.monotonic() - _t_cpu) * 1000
        delta_ms = 0.0
        if start + n >= prompt_len:
            positions = req_state.mrope_positions[:, :prompt_len]
            req_state.mrope_position_delta = int(positions.max().item()) + 1 - prompt_len
            logger.debug(
                "[LWD][cloud-runner] mrope complete: req=%s prompt=%d delta=%d",
                req_id,
                prompt_len,
                req_state.mrope_position_delta,
            )
        return cpu_ms, delta_ms

    def _sample(self, logits: torch.Tensor, spec_decode_metadata: SpecDecodeMetadata | None) -> SamplerOutput:
        """Capture the sampler output for the collect probe
        (the base sample_tokens clears execute_model_state on return)."""
        sampler_output = super()._sample(logits, spec_decode_metadata)
        self._lwd_captured_sampler_output = sampler_output
        return sampler_output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput | None) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        wire_endpoint = get_tp_group().is_first_rank
        captured = None
        if wire_endpoint and self.execute_model_state is not None:
            # ExecuteModelState layout (see model_runner_v1.sample_tokens):
            # (scheduler_output, logits, spec_decode_metadata,
            #  spec_decode_common_attn_metadata, hidden_states,
            #  sample_hidden_states, ...)
            state = self.execute_model_state

            captured = (
                state[5],
                state[1],
                state[2],
                state[0],
                list(self.input_batch.req_ids),
                self.discard_request_mask.np.copy(),
            )
        self._lwd_captured_sampler_output = None
        output = super().sample_tokens(grammar_output)

        if captured is not None and self._lwd_captured_sampler_output is not None and not self._disable_down:
            self._lwd_pending_down_payload = self._lwd_collect_down_payload(
                captured[0],
                captured[1],
                captured[2],
                captured[3],
                captured[4],
                captured[5],
                self._lwd_captured_sampler_output,
            )
        return output

    def take_lwd_pending_down_payload(self) -> LwdDownPayload | None:
        """Transfer this step's pending payload to the worker once."""
        payload = getattr(self, "_lwd_pending_down_payload", None)
        self._lwd_pending_down_payload = None
        return payload

    @torch.inference_mode()
    def _lwd_collect_down_payload(
        self,
        sample_hidden_states: torch.Tensor,
        logits: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None,
        scheduler_output: SchedulerOutput,
        batch_req_ids: list[str],
        discard_mask_np: np.ndarray,
        sampler_output: SamplerOutput,
    ) -> LwdDownPayload | None:
        """Encode token ranks and acceptance counts alongside the corresponding hidden rows.

        Capture request order before native sampling can advance batch state.
        Rejected draft rows retain their wire slots; acceptance counts determine
        which prefix is returned. Metadata D2H completion belongs to this step."""
        _t0 = time.monotonic()
        sampled = sampler_output.sampled_token_ids
        if sampled is None or sampled.dim() != 2 or logits is None:
            return None

        valid = ~discard_mask_np[: len(batch_req_ids)]
        is_spec = spec_decode_metadata is not None

        rows_list, ranks_list, accepted = [], [], []
        hidden_packet: torch.Tensor | None = None
        lg = logits
        full_cover = all(valid[: len(batch_req_ids)])
        if not is_spec:
            idx = [i for i in range(len(batch_req_ids)) if valid[i]]
            if not idx:
                return None
            lg_sel = lg if full_cover else lg[idx]
            sm_sel = sampled[:, 0] if full_cover else sampled[idx][:, 0]

            ranks_list.append(self._lwd_token_rank_encoder.encode(lg_sel, sm_sel))
            rows_list.extend(sample_hidden_states[i : i + 1] for i in idx)
            accepted.extend([1] * len(idx))
        else:
            seg_lens = [d + 1 for d in spec_decode_metadata.num_draft_tokens]

            counts_all = (sampled != -1).sum(dim=1)

            seg_lens_np = np.asarray(seg_lens, dtype=np.int32)
            seg_starts = np.cumsum(seg_lens_np) - seg_lens_np
            total_rows = sample_hidden_states.shape[0]
            rows_per_req = np.minimum(seg_lens_np, np.maximum(total_rows - seg_starts, 0))
            vidx_np = np.nonzero(valid[: len(batch_req_ids)] & (rows_per_req >= 1))[0].astype(np.int64)
            if vidx_np.size == 0:
                return None
            vlen_np = rows_per_req[vidx_np].astype(np.int64)
            starts_v = seg_starts[vidx_np].astype(np.int64)
            n_rows = int(vlen_np.sum())
            req_idx_np = np.repeat(vidx_np, vlen_np)
            pos_idx_np = np.arange(n_rows, dtype=np.int64) - np.repeat(np.cumsum(vlen_np) - vlen_np, vlen_np)
            row_sel_np = np.repeat(starts_v, vlen_np) + pos_idx_np

            buf = self._lwd_h2d_stage_i64(np.concatenate([req_idx_np, pos_idx_np, row_sel_np, vidx_np, vlen_np]))
            n_valid = vidx_np.size
            req_d, pos_d = buf[:n_rows], buf[n_rows : 2 * n_rows]
            sel_d = buf[2 * n_rows : 3 * n_rows]
            vidx_d = buf[3 * n_rows : 3 * n_rows + n_valid]
            vlen_d = buf[3 * n_rows + n_valid :]
            sampled_rows = sampled[req_d, pos_d]
            if n_rows == total_rows:
                lg_rows = lg[:n_rows]
                hidden_packet = sample_hidden_states[:n_rows]
            else:
                lg_rows = lg[sel_d]
                hidden_packet = sample_hidden_states[sel_d]

            ranks_list.append(self._lwd_token_rank_encoder.encode(lg_rows, sampled_rows))

            counts_dev = torch.minimum(counts_all[vidx_d], vlen_d).to(torch.int32)
            seg_lens_list = vlen_np.tolist()
        if hidden_packet is None:
            if not rows_list:
                return None
            hidden_packet = torch.cat(rows_list)
        n_req = sum(1 for i in range(len(batch_req_ids)) if valid[i])
        seg_lens_host: list[int] | None = None
        if is_spec:
            meta_dev = torch.cat(ranks_list + [counts_dev])

            seg_lens_host = seg_lens_list
        else:
            counts_dev = torch.ones(n_req, dtype=torch.int32, device=logits.device)
            seg_lens_dev = counts_dev
            meta_dev = torch.cat(ranks_list + [counts_dev, seg_lens_dev])

        _t_pack = time.monotonic()
        n_dev = meta_dev.numel()
        n_meta = n_dev + (len(seg_lens_host) if seg_lens_host is not None else 0)
        pinned = self._lwd_meta_pinned(n_meta)
        pinned[:n_dev].copy_(meta_dev, non_blocking=True)
        if seg_lens_host is not None:
            pinned.numpy()[n_dev:n_meta] = seg_lens_host
        ready = torch.npu.Event()
        ready.record()

        logger.debug(
            "[LWD][perf] collect rows=%d rank=%.2f pinned=%.2f total=%.2fms",
            hidden_packet.shape[0],
            (_t_pack - _t0) * 1000,
            (time.monotonic() - _t_pack) * 1000,
            (time.monotonic() - _t0) * 1000,
        )
        return (
            hidden_packet,
            pinned[:n_meta],
            ready,
            [r for i, r in enumerate(batch_req_ids) if valid[i]],
            None,
        )

    def _lwd_meta_pinned(self, n: int) -> torch.Tensor:
        """Rotate four pinned buffers, covering the bounded native batch queue."""
        ring = getattr(self, "_lwd_pinned_ring", None)
        if ring is None or ring[0].numel() < n:
            ring = [torch.empty(max(n, 4096), dtype=torch.int32, pin_memory=True) for _ in range(4)]
            self._lwd_pinned_ring = ring
            self._lwd_pinned_ring_idx = 0
        idx = self._lwd_pinned_ring_idx
        self._lwd_pinned_ring_idx = (idx + 1) % len(ring)
        return ring[idx]

    def _lwd_h2d_stage_i64(self, arr: np.ndarray) -> torch.Tensor:
        """Stage gather indices in a pinned ring without blocking the compute stream."""
        n = arr.size
        ring = getattr(self, "_lwd_h2d_ring", None)
        if ring is None or ring[0][0].numel() < n:
            m = max(n, 1024)
            ring = [
                (
                    torch.empty(m, dtype=torch.int64, pin_memory=True),
                    torch.empty(m, dtype=torch.int64, device=self.device),
                )
                for _ in range(4)
            ]
            self._lwd_h2d_ring = ring
            self._lwd_h2d_ring_idx = 0
        idx = self._lwd_h2d_ring_idx
        self._lwd_h2d_ring_idx = (idx + 1) % len(ring)
        stage, dev = ring[idx]
        stage.numpy()[:n] = arr
        dev[:n].copy_(stage[:n], non_blocking=True)
        return dev[:n]
