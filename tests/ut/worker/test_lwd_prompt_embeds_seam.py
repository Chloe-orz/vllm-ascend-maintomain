# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cloud-side seam tests for the prompt-embeds sinking batch.

Companion to vllm ``tests/v1/lwd_control/test_prompt_embeds_seam.py``:
the LWD halves of the two seams live ONLY here --

- ``_resolve_prompt_embeds``: rebuild an all-zeros placeholder when the
  edge-cloud transport carried shape only,
- ``_should_run_mm_encoder``: skip the encoder when the batch carries
  edge-provided embeds (recomputed embeddings would overwrite them),

while the shared bases (``model_runner_v1.py``) stay protocol-free.
Source-level assertions run everywhere; the runtime half re-runs on the
NPU env (needs torch + the vllm import chain).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
CLOUD = (
    _REPO_ROOT / "vllm_ascend" / "worker" / "lwd_cloud_model" / "lwd_cloud_model_runner.py"
)
BASE_MR = _REPO_ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"


def _method(path: Path, name: str) -> ast.FunctionDef:
    src = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found in {path}")


def test_cloud_overrides_exist() -> None:
    src = CLOUD.read_text(encoding="utf-8")
    assert "def _resolve_prompt_embeds(" in src
    assert "def _should_run_mm_encoder(" in src


def test_cloud_placeholder_rebuild_uses_shape_protocol() -> None:
    fn = _method(CLOUD, "_resolve_prompt_embeds")
    seg = ast.get_source_segment(CLOUD.read_text(encoding="utf-8"), fn)
    assert seg is not None
    assert "prompt_embeds_shape" in seg
    assert "torch.zeros" in seg


def test_cloud_mm_gate_skips_embeds_then_delegates() -> None:
    fn = _method(CLOUD, "_should_run_mm_encoder")
    seg = ast.get_source_segment(CLOUD.read_text(encoding="utf-8"), fn)
    assert seg is not None
    assert "req_prompt_embeds" in seg
    assert "super()._should_run_mm_encoder(" in seg


def test_ascend_base_runner_stays_protocol_free() -> None:
    src = BASE_MR.read_text(encoding="utf-8")
    assert "prompt_embeds_shape" not in src
    assert "has_prompt_embeds" not in src


def _cloud_cls():
    pytest.importorskip(
        "torch", reason="re-run on NPU env: python>=3.10 + torch + vllm"
    )
    from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_model_runner import (
        LwdCloudModelRunner,
    )

    return LwdCloudModelRunner


def test_runtime_placeholder_rebuild() -> None:
    cls = _cloud_cls()
    import torch

    runner = cls.__new__(cls)

    class _ShapeOnly:
        prompt_embeds = None
        prompt_embeds_shape = (2, 3, 4)
        prompt_embeds_dtype = torch.float16

    out = runner._resolve_prompt_embeds(_ShapeOnly)
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (2, 3, 4)
    assert out.dtype == torch.float16
    assert float(out.abs().sum()) == 0.0


def test_runtime_passthrough_when_embeds_present() -> None:
    cls = _cloud_cls()
    runner = cls.__new__(cls)

    class _Carried:
        prompt_embeds = "EMBEDS"
        prompt_embeds_shape = None
        prompt_embeds_dtype = None

    assert runner._resolve_prompt_embeds(_Carried) == "EMBEDS"


def _gate(runner, req_prompt_embeds, supports_mm=True):
    runner.enable_prompt_embeds = True
    runner.supports_mm_inputs = supports_mm

    class _IB:
        pass

    runner.input_batch = _IB()
    runner.input_batch.req_prompt_embeds = req_prompt_embeds
    return runner._should_run_mm_encoder(is_first_rank=True, is_encoder_decoder=False)


def test_runtime_mm_gate_skips_when_embeds_in_batch() -> None:
    cls = _cloud_cls()
    assert _gate(cls.__new__(cls), req_prompt_embeds=["e"]) is False


def test_runtime_mm_gate_delegates_without_embeds() -> None:
    cls = _cloud_cls()
    assert _gate(cls.__new__(cls), req_prompt_embeds=[]) is True
    assert _gate(cls.__new__(cls), req_prompt_embeds=[], supports_mm=False) is False
