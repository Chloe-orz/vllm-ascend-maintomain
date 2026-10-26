# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Lossless LWD expert-ID transport and request-position bookkeeping."""

import torch


def hash_layer_count(config: "PretrainedConfig") -> int:
    """Return the Hash MoE layer count for DeepSeek V4; zero for other models."""
    return config.num_hash_layers if getattr(config, "model_type", None) == "deepseek_v4" else 0


def hash_payload_numel(num_tokens: int, hidden_size: int, num_layers: int, top_k: int) -> int:
    """Compute the transport payload element count: embeddings plus expert IDs encoded as four bytes each."""
    # Each int32 expert ID is encoded as four exactly representable BF16 bytes.
    return num_tokens * (hidden_size + 4 * num_layers * top_k)


def pack_hash_payload(embeds: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """Pack embeddings and [tokens, hash_layers, top_k] expert IDs into a 1-D BF16 payload."""
    if embeds.dtype != torch.bfloat16:
        raise ValueError("LWD V4 expert transport requires bfloat16 embeddings")
    if expert_ids.ndim != 3 or expert_ids.shape[0] != embeds.shape[0]:
        raise ValueError("LWD expert IDs must have shape [tokens, hash_layers, top_k]")
    # Numeric BF16 conversion of IDs would round values above 256 (e.g. Pro).
    # Encode byte values, rather than interpreting integer bits as BF16 NaNs.
    encoded = expert_ids.to(torch.int32).contiguous().view(torch.uint8).flatten().to(embeds.dtype)
    return torch.cat((embeds.flatten(), encoded.to(embeds.device)))


def unpack_hash_payload(
    payload: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    num_layers: int,
    top_k: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Validate and unpack the payload; returns on-device embeddings and CPU int32 expert IDs."""
    expected = hash_payload_numel(num_tokens, hidden_size, num_layers, top_k)
    if payload.numel() != expected:
        raise ValueError(f"LWD V4 payload size mismatch: got {payload.numel()}, expected {expected}")
    flat = payload.flatten()
    embed_end = num_tokens * hidden_size
    embeds = flat[:embed_end].view(num_tokens, hidden_size)
    # Request-position bookkeeping lives on CPU; decode the wire bytes there.
    expert_ids = flat[embed_end:].cpu().to(torch.uint8).contiguous().view(torch.int32)
    return embeds, expert_ids.view(num_tokens, num_layers, top_k)


def merge_hash_expert_ids(
    tid2eid: torch.Tensor,
    token_ids: torch.Tensor,
    prompt_experts: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> torch.Tensor:
    """Use remote IDs for prompt rows and checkpoint lookup for decode rows."""
    # The token buffer's prompt slots are intentionally unspecified. Never use
    # them as table indices, even in the unselected branch of torch.where.
    safe_token_ids = torch.where(prompt_mask | (token_ids == -1), 0, token_ids).to(torch.int64)
    decode_experts = tid2eid[safe_token_ids]
    return torch.where(prompt_mask.unsqueeze(-1), prompt_experts, decode_experts).to(torch.int32)


class LwdHashRoutingState:
    """CPU prompt tables keyed by request ID, with explicit received-row masks."""

    def __init__(self, num_layers: int, top_k: int, num_experts: int) -> None:
        """Store the Hash routing dimensions and initialize the per-request prompt expert-ID cache."""
        self.num_layers = num_layers
        self.top_k = top_k
        self.num_experts = num_experts
        self.requests: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def discard(self, req_id: str) -> None:
        """Release a request's expert IDs and validity mask; a missing request is a no-op."""
        self.requests.pop(req_id, None)

    def add_chunk(
        self, req_id: str, prompt_len: int, start: int, expert_ids: torch.Tensor
    ) -> None:
        """Validate a remote expert-ID chunk, write it into the request's absolute prompt position and mark it valid."""
        ids = expert_ids.to(device="cpu", dtype=torch.int32)
        if ids.ndim != 3 or tuple(ids.shape[1:]) != (self.num_layers, self.top_k):
            raise ValueError(f"Invalid LWD expert shape for {req_id}: {tuple(ids.shape)}")
        end = start + ids.shape[0]
        if not 0 <= start <= end <= prompt_len:
            raise ValueError(f"Invalid LWD expert chunk for {req_id}: [{start}, {end})/{prompt_len}")
        if torch.any(ids < 0) or torch.any(ids >= self.num_experts):
            raise ValueError(f"LWD expert IDs out of range for {req_id}")
        entry = self.requests.get(req_id)
        if entry is None or entry[0].shape[0] != prompt_len:
            entry = (
                torch.zeros(prompt_len, self.num_layers, self.top_k, dtype=torch.int32),
                torch.zeros(prompt_len, dtype=torch.bool),
            )
            self.requests[req_id] = entry
        entry[0][start:end].copy_(ids)
        entry[1][start:end] = True

    def build_batch(
        self,
        req_ids: "Sequence[str]",
        counts: "Sequence[int]",
        computed: "Sequence[int]",
        prompt_lengths: "Sequence[int]",
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Assemble CPU expert IDs and the prompt mask by request order and computed positions; raise when prompt data is missing."""
        if not len(req_ids) == len(counts) == len(computed) == len(prompt_lengths):
            raise ValueError("LWD Hash routing batch metadata lengths do not match")
        total = sum(int(n) for n in counts)
        batch = torch.zeros(total, self.num_layers, self.top_k, dtype=torch.int32)
        mask = torch.zeros(total, dtype=torch.bool)
        row = 0
        for req_id, count, start, prompt_len in zip(req_ids, counts, computed, prompt_lengths):
            count, start, prompt_len = int(count), int(start), int(prompt_len)
            prompt_count = max(0, min(count, prompt_len - start))
            if prompt_count:
                entry = self.requests.get(req_id)
                end = start + prompt_count
                if entry is None or end > entry[0].shape[0] or not entry[1][start:end].all():
                    raise ValueError(f"Missing LWD Hash MoE expert IDs for {req_id} positions [{start}, {end})")
                batch[row : row + prompt_count].copy_(entry[0][start:end])
                mask[row : row + prompt_count] = True
            row += count
        return batch, mask
