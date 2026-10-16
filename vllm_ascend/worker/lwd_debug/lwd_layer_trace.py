#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""LWD edge-cloud vs centralized per-layer parity instrumentation (debug
facility, shared by both modes).

Design constraints:
  * Centralized and edge-cloud MUST run the identical instrumentation —
    the instrumentation itself may not introduce divergence — so the
    install points live in the per-side load_model (edge/cloud arms are
    automatic; see LwdEdgeWorker / LwdCloudWorker), gated by the
    VLLM_ASCEND_LWD_LAYER_TRACE=1 env var, zero overhead when off;
  * The centralized comparison arm is manual opt-in: set the same env
    var and call install_lwd_layer_trace(model) from the driver — the
    NPUModelRunner base stays free of LWD symbols and the weave-guard
    allowlist stays empty;
  * Log checksums only (numel/sum/abs_sum/max_abs/l2/head4), never full
    tensors;
  * Trace only the first _MAX_STEPS model forwards (prefill plus a few
    decode steps) so the decode loop cannot flood the log.
Log prefix is uniformly [layer-trace], coexisting with [Lwd][DUMP] for
easy grep.
"""

import os
from collections.abc import Callable
from typing import Any

import torch

_LWD_TRACE_ENV = "VLLM_ASCEND_LWD_LAYER_TRACE"
_MAX_STEPS = 3
_state: dict[str, int] = {"step": 0}


def lwd_layer_trace_enabled() -> bool:
    return os.getenv(_LWD_TRACE_ENV, "") == "1"


def lwd_tensor_checksum(tag: str, tensor: Any) -> str:
    """One-line tensor checksum description; non-tensors and empty tensors return a placeholder."""
    if tensor is None:
        return f"{tag}=None"
    if not isinstance(tensor, torch.Tensor):
        return f"{tag}=type({type(tensor).__name__})"
    if tensor.numel() == 0:
        return f"{tag}=empty{tuple(tensor.shape)}"
    f = tensor.detach().float()
    head4 = [round(v, 4) for v in f.flatten()[:4].cpu().tolist()]
    return (
        f"{tag} shape={tuple(tensor.shape)} numel={tensor.numel()} "
        f"sum={f.sum().item():.4f} abs={f.abs().sum().item():.4f} "
        f"max={f.abs().max().item():.4f} l2={f.norm().item():.4f} "
        f"head4={head4}"
    )


def _hidden_from(x: Any) -> Any:
    """Extract the hidden tensor from module inputs/outputs (tolerates tuple/None wrapping)."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and len(x) > 0:
        return _hidden_from(x[0])
    return None


def _make_hook(label: str) -> Callable[..., None]:
    from vllm.logger import init_logger

    log = init_logger(__name__)

    def hook(
        module: "torch.nn.Module",
        inputs: "tuple[torch.Tensor, ...]",
        output: "torch.Tensor | tuple[torch.Tensor, ...]",
    ) -> None:
        if _state["step"] > _MAX_STEPS:
            return
        log.info(
            "[layer-trace] step=%d %s in[%s] out[%s]",
            _state["step"], label,
            lwd_tensor_checksum("h", _hidden_from(inputs)),
            lwd_tensor_checksum("h", _hidden_from(output)),
        )

    return hook


def _make_step_hook() -> Callable[..., None]:
    from vllm.logger import init_logger

    log = init_logger(__name__)

    def pre_hook(
        module: "torch.nn.Module", args: "tuple[torch.Tensor, ...]"
    ) -> None:
        _state["step"] += 1
        if _state["step"] <= _MAX_STEPS:
            log.info("[layer-trace] ==== forward step=%d ====", _state["step"])

    return pre_hook


def _find_text_backbone(
    model: "torch.nn.Module",
) -> "tuple[torch.nn.Module | None, torch.nn.ModuleList | None]":
    """Locate the text backbone: handles plain structures (model.model.layers)
    and VL wrappers (model.language_model[.model].layers, top level being
    visual+language_model). Returns (backbone, layers) or (None, None)."""
    candidates = (
        getattr(model, "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(model, "language_model", None),
    )
    for holder in candidates:
        if holder is None:
            continue
        layers = (
            getattr(holder, "layers", None)
            or getattr(holder, "decoder_layers", None)
        )
        if layers is not None:
            return holder, layers
    return None, None


def install_lwd_layer_trace(model: "torch.nn.Module") -> None:
    """L0+L2: layer-count/structure log + per-layer forward hooks (idempotent,
    env-gated).

    Covers embed_tokens / decoder layers / final norm / lm_head; both
    centralized and edge-cloud use this installer during parity runs, and
    the layer list itself is the adjudicating evidence for half-model
    suspicions."""
    if not lwd_layer_trace_enabled():
        return
    if getattr(model, "_lwd_layer_traced", False):
        return
    model._lwd_layer_traced = True
    from vllm.logger import init_logger

    log = init_logger(__name__)
    backbone, layers = _find_text_backbone(model)
    if layers is None:
        log.warning(
            "[layer-trace] no decoder layers found on %r; top modules=%s",
            type(model).__name__,
            [n for n, _ in model.named_children()][:12],
        )
        return
    log.info(
        "[layer-trace] model=%s decoder_layers=%d first=%s last=%s "
        "norm=%s lm_head=%s",
        type(model).__name__, len(layers),
        list(backbone.named_children())[0][0] if layers else "-",
        f"layer{len(layers) - 1}",
        hasattr(backbone, "norm"), hasattr(model, "lm_head"),
    )
    backbone.register_forward_pre_hook(_make_step_hook())
    embed = getattr(backbone, "embed_tokens", None)
    if embed is not None:
        embed.register_forward_hook(_make_hook("embed_tokens"))
    for i, layer in enumerate(layers):
        layer.register_forward_hook(_make_hook(f"layer{i:02d}"))
    if hasattr(backbone, "norm"):
        backbone.norm.register_forward_hook(_make_hook("final_norm"))
    if hasattr(model, "lm_head"):
        model.lm_head.register_forward_hook(_make_hook("lm_head"))
