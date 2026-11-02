# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Zero-base-weave seam tests for the LWD cloud runner (ascend side).

The vllm base ``gpu_model_runner.py`` carries NO LWD seams anymore
(locked by the vllm-side companion test): the shape-only placeholder
rebuild, the MM-encoder gate and the mrope prompt-embeds-only branch are
all presented by ``LwdCloudModelRunner`` overriding the native methods
and wrapping ``super()``:

- ``_update_states`` / ``_update_streaming_request`` rebuild the
  all-zeros placeholder in-place on the incoming request data before the
  base builds/updates ``CachedRequestState`` (base passthrough of
  ``new_req_data.prompt_embeds`` keeps native semantics);
- ``_init_mrope_positions`` constructs arange positions directly for
  prompt-embeds-only requests, otherwise defers to the base scan;
- ``supports_mm_inputs`` is a dynamic property: inject batches (remote
  embeds present) report False so the native MM gate skips the encoder,
  everything else reports the model capability.

Source-level assertions run everywhere; the runtime half re-runs on the
NPU env (python>=3.10 + torch + vllm import chain), mirroring the
vllm-side companion.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

_REPO_ROOT = Path(__file__).resolve().parents[3]
CLOUD = _REPO_ROOT / "vllm_ascend" / "worker" / "lwd_cloud_model" / "lwd_cloud_model_runner.py"


def test_zero_weave_overrides_present() -> None:
    src = CLOUD.read_text(encoding="utf-8")
    for marker in (
        "def _update_states(",
        "def _update_streaming_request(",
        "def _init_mrope_positions(",
        "@property",
        "def supports_mm_inputs(",
        "@supports_mm_inputs.setter",
    ):
        assert marker in src, f"zero-weave override missing: {marker}"
    for stale in ("_resolve_prompt_embeds", "_should_run_mm_encoder"):
        assert stale not in src, f"stale base hook reference: {stale}"


def test_zero_weave_overrides_delegate_to_super() -> None:
    """All three wraps must end/delegate via super(), never replicating base implementations."""
    src = CLOUD.read_text(encoding="utf-8")
    assert "return super()._update_states(scheduler_output)" in src
    assert "return super()._update_streaming_request(req_id, new_req_data)" in src
    assert "super()._init_mrope_positions(req_state)" in src


def test_placeholder_wash_uses_meta_protocol() -> None:
    """The placeholder wash keys off is_meta (the engine attaches meta
    tensors): the shape/dtype transport fields must stay gone from the
    cloud runner."""
    src = CLOUD.read_text(encoding="utf-8")
    assert "is_meta" in src
    for stale in ("prompt_embeds_shape", "prompt_embeds_dtype"):
        assert stale not in src, f"stale transport field reference: {stale}"


def test_runtime_meta_placeholder_materialization(
    monkeypatch,
) -> None:
    """A meta placeholder entering _update_states / _update_streaming
    request must come out as a local all-zeros tensor with identical
    shape/dtype, and the base must still run (super delegated)."""
    cls = _cloud_cls()
    import torch

    def fake_base_update_states(self, scheduler_output):
        self._base_saw = scheduler_output
        return "BASE_RET"

    def fake_base_update_streaming(self, req_id, new_req_data):
        self._base_streaming_saw = (req_id, new_req_data)
        return "STREAM_RET"

    runner = cls.__new__(cls)
    runner.worker = NS(_lwd_early_recv_endpoint=False)
    base_cls = type(runner).__mro__[1]
    monkeypatch.setattr(base_cls, "_update_states", fake_base_update_states)
    monkeypatch.setattr(base_cls, "_update_streaming_request", fake_base_update_streaming)
    meta_embeds = torch.empty(4, 8, dtype=torch.float32, device="meta")
    assert meta_embeds.is_meta
    new_req = NS(prompt_embeds=meta_embeds)
    cached = NS(req_ids=[], num_computed_tokens=[])
    so = NS(scheduled_new_reqs=[new_req], scheduled_cached_reqs=cached)

    ret = runner._update_states(so)
    assert ret == "BASE_RET" and runner._base_saw is so
    washed = new_req.prompt_embeds
    assert washed is not None and not washed.is_meta
    assert tuple(washed.shape) == (4, 8) and washed.dtype == torch.float32
    assert torch.equal(washed, torch.zeros(4, 8, dtype=torch.float32))

    streaming_req = NS(prompt_embeds=meta_embeds.clone())
    ret = runner._update_streaming_request("rid", streaming_req)
    assert ret == "STREAM_RET"
    assert runner._base_streaming_saw[0] == "rid"
    assert not streaming_req.prompt_embeds.is_meta
    assert torch.equal(streaming_req.prompt_embeds, torch.zeros(4, 8, dtype=torch.float32))


def _cloud_cls():
    pytest_import = __import__("pytest")
    pytest_import.importorskip("torch", reason="re-run on NPU env: python>=3.10 + torch + vllm")
    from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_model_runner import (
        LwdCloudModelRunner,
    )

    return LwdCloudModelRunner


def test_runtime_mrope_direct_construction() -> None:
    cls = _cloud_cls()
    import torch

    runner = cls.__new__(cls)
    state = NS(
        prompt_token_ids=None,
        num_prompt_tokens=4,
        mrope_positions=None,
        mrope_position_delta=None,
    )
    runner._init_mrope_positions(state)
    assert state.mrope_position_delta == 0
    assert torch.equal(
        state.mrope_positions,
        torch.arange(4).unsqueeze(0).expand(3, 4),
    )


def test_runtime_supports_mm_inputs_property() -> None:
    cls = _cloud_cls()
    runner = cls.__new__(cls)
    runner._model_supports_mm_inputs = True
    runner.input_batch = NS(req_prompt_embeds={0: object()})
    assert runner.supports_mm_inputs is False  # injected batch: gate closed
    runner.input_batch = NS(req_prompt_embeds={})
    assert runner.supports_mm_inputs is True  # empty batch: model true value
