# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Edge-side multimodal cache (moved verbatim from LwdEdgeWorker).

Two registries with different lifecycles: per-request feature
bookkeeping (position info; released when the request's chunks pass all
MM windows or the request finishes) and content-hash shared encoder
outputs (FIFO eviction, NOT request-scoped — the first reference-count
design evicted shared entries before later requests arrived and the
same image was re-encoded 30-way under concurrency).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import LwdEmbedBatch

# mm_hash 编码缓存容量(条目数;1080p 单条约 26MB HBM)
_LWD_MM_HASH_CACHE_MAX_ITEMS = 32


class LwdEdgeMmCache:
    """Per-request MM feature bookkeeping + hash-shared encoder outputs."""

    def __init__(self) -> None:
        self._features: dict[str, list["MultiModalFeatureSpec"]] = {}
        self._hash_embeds: dict[str, object] = {}

    def register(self, req_id: str, feats: "list[object]") -> None:
        """Record a new request's MM features (first chunk carries data)."""
        self._features[req_id] = feats

    def release_finished(self, finished_req_ids: "list[str]") -> None:
        """完结兜底:请求完结(含 abort)即摘除其 mm 特征/位置登记
        (正常路径在 chunk 越过全部 mm 窗口后已提前摘除)。"""
        for req_id in finished_req_ids:
            self.release_request(req_id)

    def encode_missing(
        self,
        req_ids: "list[str]",
        model: "torch.nn.Module",
        device: "torch.device",
        pin_memory: bool,
    ) -> None:
        """批级编码:聚合本批全部请求的缓存未命中特征,按 modality 一次
        组批过视觉塔,结果回填 mm_hash 缓存。

        跨请求组批与集中式 _execute_mm_encoder 同构(整批一次前向而
        非逐请求 N 次):ViT eager 下单次调用的固定开销(逐算子
        dispatch/H2D/同步)占比高,逐请求串行会把它支付 N 倍,且这些
        串行调用压在 worker 的 embed 执行里、堵住后续 unembed 交付。
        缓存键(identifier)与容量淘汰语义不变;回填顺序 = 收集顺序
        (req_ids 序 × 请求内特征序),组批输出按同序消费——错序即
        张冠李戴错算,总数断言兜底。"""
        missing_features = []
        missing_data = []
        for req_id in req_ids:
            for f in self._features.get(req_id) or []:
                if f.identifier not in self._hash_embeds:
                    if f.data is None:
                        # data 缺失即无法计算图像 embeds——静默跳过 =
                        # 图像 token 按文本嵌入错算,fail-fast。
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
        from vllm.multimodal.utils import group_and_batch_mm_kwargs

        mm_kwargs = [(f.modality, f.data) for f in missing_features]
        outputs_by_modality: dict[str, list[torch.Tensor]] = {}
        num_outputs = 0
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs,
            device=device,
            pin_memory=pin_memory,
        ):
            batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
            assert len(batch_outputs) == num_items, (
                f"encoder outputs {len(batch_outputs)} != items "
                f"{num_items} (modality={modality})"
            )
            outputs_by_modality.setdefault(modality, []).extend(batch_outputs)
            num_outputs += len(batch_outputs)
        assert num_outputs == len(missing_features), (
            f"encoder outputs {num_outputs} != missing features "
            f"{len(missing_features)}"
        )
        for feature in missing_features:
            self._hash_embeds[feature.identifier] = (
                outputs_by_modality[feature.modality].pop(0)
            )
        # 容量淘汰(FIFO):超容量摘最旧条目,被淘汰顶多重编一次
        while len(self._hash_embeds) > _LWD_MM_HASH_CACHE_MAX_ITEMS:
            self._hash_embeds.pop(next(iter(self._hash_embeds)))
        # [Lwd][perf] 视觉塔耗时(批级,仅缓存未命中计次;同图复用不重编)
        logger.info(
            "[Lwd][perf] mm-encoder reqs=%d items=%d dur=%.2fms",
            len(req_ids), len(missing_features),
            (time.monotonic() - _t_enc) * 1000,
        )

    def encoder_outputs(
        self, req_id: str
    ) -> "list[tuple[PlaceholderRange, torch.Tensor]]":
        """视觉塔输出的只读视图:[(mm_position, embeds)]。

        编码已前移到批级(encode_missing,整批一次组批);本处只读
        缓存,未命中即批级编码漏编,fail-fast。编码结果按内容哈希
        (mm_hash/identifier)跨请求共享——与集中式 encoder cache 同
        粒度,同图复用不重编;位置信息(mm_position)仍按请求取,只有
        编码结果共享;容量淘汰(FIFO),不随请求生命周期释放。
        release_request 在 chunk 越过全部 mm 窗口或 finished_req_ids
        兜底时摘除请求的位置特征(不动编码缓存)。"""
        features = self._features.get(req_id)
        if not features:
            return []
        missing = [
            f.identifier for f in features
            if f.identifier not in self._hash_embeds
        ]
        if missing:
            raise RuntimeError(
                f"[LWD] mm embeds missing from batch encode: req={req_id} "
                f"(mm_hashes={missing})"
            )
        return [
            (f.mm_position, self._hash_embeds[f.identifier])
            for f in features
        ]

    def release_request(self, req_id: str) -> None:
        """按请求释放 mm 位置特征;hash 编码结果不随请求释放
        (容量淘汰,见 encoder_outputs)。"""
        self._features.pop(req_id, None)

    def gather_chunk(
        self,
        batch_meta: "LwdEmbedBatch",
        model: "torch.nn.Module",
        device: "torch.device",
        pin_memory: bool,
    ) -> "tuple[list[torch.Tensor], torch.Tensor]":
        """按本 chunk 窗口收集 mm embeds 行 + 构造 is_multimodal 掩码。

        编码在切片之前批级完成(整批一次组批,见 encode_missing),
        之后逐请求只读缓存切片。窗口语义逐字对齐原生
        _gather_mm_embeddings:feature 的占位区间 [offset, offset+length)
        与 chunk 窗口 [p_offset, p_offset+n) 求交,相交段切 embeds 行、
        置掩码位;图像跨 chunk 时逐段收集。同时完成越过全部 mm 窗口
        请求的特征摘除。"""
        total = sum(len(t) for t in batch_meta.token_ids)
        is_mm = torch.zeros(total, dtype=torch.bool)
        mm_embeds: list[torch.Tensor] = []
        if len(batch_meta.prompt_offsets) != len(batch_meta.req_ids):
            return mm_embeds, is_mm
        self.encode_missing(batch_meta.req_ids, model, device, pin_memory)
        req_start = 0
        for req_id, token_ids, p_offset in zip(
            batch_meta.req_ids, batch_meta.token_ids, batch_meta.prompt_offsets
        ):
            n = len(token_ids)
            cached = self.encoder_outputs(req_id)
            max_end = 0
            for pos_info, item_embeds in cached:
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length
                max_end = max(max_end, start_pos + num_encoder_tokens)
                start_idx = max(p_offset - start_pos, 0)
                end_idx = min(
                    p_offset - start_pos + n, num_encoder_tokens
                )
                if start_idx >= end_idx:
                    continue
                req_start_pos = req_start + start_pos - p_offset
                if (is_embed := pos_info.is_embed) is not None:
                    s, e = pos_info.get_embeds_indices_in_range(
                        start_idx, end_idx
                    )
                    if s == e:
                        continue
                    rows = item_embeds[s:e]
                    is_mm[req_start_pos + start_idx : req_start_pos + end_idx] |= (
                        is_embed[start_idx:end_idx]
                    )
                else:
                    rows = item_embeds[start_idx:end_idx]
                    is_mm[req_start_pos + start_idx : req_start_pos + end_idx] = True
                mm_embeds.append(rows)
            if cached and p_offset + n >= max_end:
                self.release_request(req_id)
            req_start += n
        return mm_embeds, is_mm
