# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Source-scan guard tests for the LWD worker/runner modules.

Locks the mechanically checkable AGENTS.md style rules for the LWD worker
and model_runner files:

- every named parameter and every return has a type annotation
- no bare generic annotations (``dict``/``list``/``tuple``/``Callable``
  without type parameters)
- function bodies stay within the 100-line budget (the former raised caps
  for ``_lwd_harvest_up_batches``/``_lwd_collect_down_payload`` were
  dropped once the S2 split landed)

Pure AST scan over source files: no torch/NPU needed, runs on any checkout.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

LWD_WORKER_FILES = [
    "vllm_ascend/worker/lwd_edge_model/lwd_edge_worker.py",
    "vllm_ascend/worker/lwd_cloud_model/lwd_cloud_worker.py",
    "vllm_ascend/worker/lwd_cloud_model/lwd_cloud_model_runner.py",
    "vllm_ascend/worker/lwd_cloud_model/lwd_mtp_proposer.py",
    "vllm_ascend/worker/lwd_base_model/lwd_worker_base.py",
    "vllm_ascend/worker/lwd_base_model/lwd_pinned_staging.py",
    "vllm_ascend/worker/lwd_hash/lwd_hash_routing.py",
    "vllm_ascend/worker/lwd_hash/lwd_hash_tables.py",
    "vllm_ascend/worker/lwd_debug/lwd_layer_trace.py",
]

_BARE_GENERIC = re.compile(
    r"\b(Dict|List|Set|FrozenSet|Tuple|Callable|dict|list|set|frozenset|tuple)\b"
    r"(?!\[)"
)


def _parse(rel_path: str) -> ast.Module:
    return ast.parse((_REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def _functions(
    tree: ast.Module,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


@pytest.mark.parametrize("rel_path", LWD_WORKER_FILES)
def test_params_and_returns_annotated(rel_path: str) -> None:
    problems: list[str] = []
    for fn in _functions(_parse(rel_path)):
        for arg in fn.args.posonlyargs + fn.args.args:
            if arg.arg != "self" and arg.annotation is None:
                problems.append(f"{fn.name}(... {arg.arg})")
        if fn.returns is None:
            problems.append(f"{fn.name} return")
    assert not problems, "unannotated params/returns: " + "; ".join(problems)


@pytest.mark.parametrize("rel_path", LWD_WORKER_FILES)
def test_no_bare_generic_annotations(rel_path: str) -> None:
    src = (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
    problems: list[str] = []
    for node in ast.walk(_parse(rel_path)):
        annotations: list[ast.expr] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations = [
                a.annotation
                for a in node.args.posonlyargs + node.args.args
                if a.arg != "self" and a.annotation is not None
            ]
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            annotations = [node.annotation]
        for ann in annotations:
            seg = ast.get_source_segment(src, ann) or ""
            if _BARE_GENERIC.search(seg):
                problems.append(f"{rel_path}:{node.lineno} bare generic: {seg}")
    assert not problems, "; ".join(problems)


@pytest.mark.parametrize("rel_path", LWD_WORKER_FILES)
def test_function_length_within_limit(rel_path: str) -> None:
    problems: list[str] = []
    for fn in _functions(_parse(rel_path)):
        span = fn.end_lineno - fn.lineno + 1
        if span > 100:
            problems.append(f"{fn.name} {span}>100")
    assert not problems, "; ".join(problems)
