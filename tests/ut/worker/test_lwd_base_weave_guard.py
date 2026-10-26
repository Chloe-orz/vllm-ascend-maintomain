# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""AST weave guard: the shared vllm-ascend base classes must stay LWD-free.

Scans ``vllm_ascend/worker/worker.py`` and ``model_runner_v1.py`` for LWD
symbols (identifiers, attributes, import aliases, getattr strings). Every
allowlist entry is a known weave-in scheduled for removal by the base-class
sinking batches; the test fails on

- any NEW lwd symbol appearing in the base files, and
- any STALE allowlist entry whose symbol is already gone,

so the allowlist only ever shrinks. Pure AST scan, no torch/NPU needed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

BASE_FILES = [
    "vllm_ascend/worker/worker.py",
    "vllm_ascend/worker/model_runner_v1.py",
]

# Allowlist pins the currently restored weave-ins (see ALLOWED below).
# Both ascend base files are fully LWD-free. The spec recovery chain
# (draft stash + accepted persist) was dead code in pdmix deployments
# (LwdCloudScheduler defers prefill instead of evicting decode; the
# PhaseScheduler experiment arm is the only consumer and is not used in
# production). Any new weave-in fails immediately.
ALLOWED = {
    "vllm_ascend/worker/worker.py": set(),
    "vllm_ascend/worker/model_runner_v1.py": set(),
}


def _scan(path: Path) -> set[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[tuple[str, str]] = set()

    def visit(node: ast.AST, func: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_func = func
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_func = child.name
            if isinstance(child, ast.Name) and "lwd" in child.id.lower():
                found.add((func, child.id))
            elif isinstance(child, ast.Attribute) and "lwd" in child.attr.lower():
                found.add((func, child.attr))
            elif isinstance(child, ast.alias) and "lwd" in child.name.lower():
                found.add((func, child.name.split(".")[-1]))
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "getattr"
                and len(child.args) >= 2
                and isinstance(child.args[1], ast.Constant)
                and isinstance(child.args[1].value, str)
                and "lwd" in child.args[1].value.lower()
            ):
                found.add((func, f"getattr:{child.args[1].value}"))
            visit(child, child_func)

    visit(tree, "<module>")
    return found


@pytest.mark.parametrize("rel_path", BASE_FILES)
def test_base_files_free_of_new_lwd_weave(rel_path: str) -> None:
    found = _scan(_REPO_ROOT / rel_path)
    allowed = ALLOWED.get(rel_path, set())
    unexpected = found - allowed
    stale = allowed - found
    assert not unexpected, f"new lwd weave-ins in {rel_path}: {sorted(unexpected)}"
    assert not stale, f"stale allowlist entries in {rel_path}: {sorted(stale)}"
