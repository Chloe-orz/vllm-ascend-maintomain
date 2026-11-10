# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure UCX block policy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_POLICY = (
    Path(__file__).resolve().parents[4]
    / "vllm_ascend"
    / "distributed"
    / "lwd_data_plane"
    / "host"
    / "ucx_block_policy.py"
)
_spec = importlib.util.spec_from_file_location("ucx_block_policy", _POLICY)
assert _spec is not None and _spec.loader is not None
_policy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _policy
_spec.loader.exec_module(_policy)

BLOCK_ALIGN_MB = _policy.BLOCK_ALIGN_MB
FALLBACK_UP_BLOCK_MB = _policy.FALLBACK_UP_BLOCK_MB
MIN_UP_BLOCK_MB = _policy.MIN_UP_BLOCK_MB
LwdUcxBlockPolicyError = _policy.LwdUcxBlockPolicyError
resolve = _policy.resolve


def _dsv4(max_tokens: int, **kwargs):
    return resolve(
        model_type="deepseek_v4",
        hidden_size=4096,
        num_hash_layers=3,
        num_experts_per_tok=6,
        max_num_batched_tokens=max_tokens,
        **kwargs,
    )


def test_dsv4_16388_uses_160mib():
    plan = _dsv4(16_388)
    assert plan.up_block_mb == 160
    assert plan.source == "auto"
    assert plan.auto_up_block_mb == 160
    assert plan.raw_mib == pytest.approx(130.29, abs=0.01)
    assert plan.per_token_elements == 4168
    assert plan.warnings == ()


def test_dsv4_20480_uses_192mib():
    plan = _dsv4(20_480)
    assert plan.up_block_mb == 192
    assert plan.auto_up_block_mb == 192


def test_qwen_7168_uses_288mib_without_hash_tail():
    plan = resolve(
        model_type="qwen3",
        hidden_size=7168,
        max_num_batched_tokens=20_480,
    )
    assert plan.up_block_mb == 288
    assert plan.per_token_elements == 7168
    assert plan.auto_up_block_mb == 288


def test_small_frame_keeps_minimum_block():
    plan = resolve(
        model_type="qwen3",
        hidden_size=4096,
        max_num_batched_tokens=8192,
    )
    assert plan.up_block_mb == MIN_UP_BLOCK_MB
    assert plan.auto_up_block_mb == MIN_UP_BLOCK_MB


def test_alignment_is_32mib():
    # 67,108,865 bf16 elements = 128MiB + 2 bytes -> next 32MiB boundary.
    plan = resolve(
        model_type="qwen3",
        hidden_size=67_108_865,
        max_num_batched_tokens=1,
    )
    assert plan.up_block_mb == 160
    assert plan.up_block_mb % BLOCK_ALIGN_MB == 0


def test_explicit_value_is_respected_and_warned_when_low():
    plan = _dsv4(20_480, explicit_up_block_mb=128)
    assert plan.up_block_mb == 128
    assert plan.source == "explicit"
    assert plan.auto_up_block_mb == 192
    assert len(plan.warnings) == 1
    assert "auto_required=192" in plan.warnings[0]


def test_explicit_value_above_auto_has_no_warning():
    plan = _dsv4(20_480, explicit_up_block_mb=256)
    assert plan.up_block_mb == 256
    assert plan.warnings == ()


def test_unknown_model_falls_back_to_128():
    plan = resolve(
        model_type="unknown_model",
        hidden_size=7168,
        max_num_batched_tokens=20_480,
    )
    assert plan.up_block_mb == FALLBACK_UP_BLOCK_MB
    assert plan.source == "fallback"
    assert plan.auto_up_block_mb == 0
    assert "fallback up_buffer_size=128" in plan.warnings[0]
    assert "submit_send fast-fail" in plan.warnings[0]


def test_unknown_model_honours_explicit():
    plan = resolve(
        model_type="unknown_model",
        hidden_size=None,
        max_num_batched_tokens=20_480,
        explicit_up_block_mb=512,
    )
    assert plan.up_block_mb == 512
    assert plan.source == "explicit"
    assert "using explicit up_buffer_size=512" in plan.warnings[0]


def test_known_model_missing_hidden_fails():
    with pytest.raises(LwdUcxBlockPolicyError, match="hidden_size"):
        resolve(
            model_type="deepseek_v4",
            hidden_size=None,
            max_num_batched_tokens=20_480,
        )


def test_bad_inputs_fail():
    with pytest.raises(LwdUcxBlockPolicyError, match="max_num_batched_tokens"):
        _dsv4(0)
    with pytest.raises(LwdUcxBlockPolicyError, match="explicit UP block"):
        _dsv4(20_480, explicit_up_block_mb=0)
    with pytest.raises(LwdUcxBlockPolicyError, match="fallback UP block"):
        _dsv4(20_480, fallback_up_block_mb=0)
