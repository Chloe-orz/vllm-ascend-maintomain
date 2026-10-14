# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD lwd token rank."""

import torch


class LwdTokenRankEncoder:
    """Encode descending-logit rank, breaking ties by ascending token ID."""

    def __init__(self) -> None:
        self.vocab_ids = None

    def encode(self, logits: torch.Tensor, selected_ids: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.shape[-1]
        if self.vocab_ids is None or self.vocab_ids.numel() != vocab_size or self.vocab_ids.device != logits.device:
            self.vocab_ids = torch.arange(vocab_size, device=logits.device, dtype=torch.int64)
        selected_ids = selected_ids.long()

        valid = selected_ids != -1
        safe_ids = selected_ids.masked_fill(~valid, 0)
        selected_logits = logits.gather(1, safe_ids.unsqueeze(1))

        before = (logits > selected_logits) | ((logits == selected_logits) & (self.vocab_ids < safe_ids.unsqueeze(1)))
        ranks = before.sum(dim=1, dtype=torch.int32)
        return ranks.masked_fill(~valid, 0)
