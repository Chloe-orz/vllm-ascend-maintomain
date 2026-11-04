# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Table-driven tests for LwdEdgeWorker.select_token_batch (edge rank replay).

The replay contract: for row ``i``, the returned token id is the
``ranks[i]``-th largest logit of that row (rank 0 = argmax, rank V-1 = the
smallest logit). Ties are resolved by ascending token ID, matching the cloud
rank encoder.

Execution needs the training stack; skipped on hosts without it.
Rerun condition: python>=3.10 with vllm / vllm-ascend installed (CPU tensors
suffice, no NPU required).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip(
    "torch",
    reason="select_token_batch needs torch; rerun on a host with the training stack",
)

from vllm_ascend.worker.lwd_edge_model.lwd_edge_model_runner import (  # noqa: E402
    select_token_batch,
)  # noqa: E402


@pytest.mark.parametrize(
    ("logits", "ranks", "expected"),
    [
        # rank 0 picks the argmax token id.
        ([[10, 30, 20]], [0], [1]),
        # rank V-1 picks the smallest logit.
        ([[10, 30, 20]], [2], [0]),
        # Ranks are resolved independently per row.
        ([[10, 30, 20], [3, 1, 2]], [1, 2], [2, 1]),
        # Smallest logit sits at the last column here.
        ([[5, 9, 1]], [2], [2]),
        # Four rows, mixed ranks, distinct values in every row.
        (
            [[7, 5, 3], [1, 2, 3], [9, 0, 5], [4, 6, 8]],
            [1, 0, 2, 1],
            [1, 2, 1, 1],
        ),
        # Equal logits are ordered by ascending token ID.
        ([[10, 10, 10], [5, 5, 5]], [0, 2], [0, 2]),
        ([[10, 10, 0], [10, 10, 0]], [1, 2], [1, 2]),
    ],
    ids=[
        "rank0-argmax",
        "rank-vocab-minus-one",
        "per-row-ranks",
        "min-at-last-column",
        "four-rows-mixed",
        "all-equal",
        "tied-max-and-lower-score",
    ],
)
def test_select_token_batch(logits: list[list[int]], ranks: list[int], expected: list[int]) -> None:
    assert select_token_batch(torch.tensor(logits), ranks) == expected
