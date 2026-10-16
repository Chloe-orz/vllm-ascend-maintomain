#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Lwd 边云 vs 集中式的逐层对拍插桩(调试设施,双模式共用)。

设计约束:
  * 集中式与边云必须走同一份插桩代码,插桩本身不得引入差异——
    安装点在边/云各自的 load_model(边云臂自动;见 LwdEdgeWorker /
    LwdCloudWorker),环境变量 VLLM_ASCEND_LWD_LAYER_TRACE=1 门控,
    缺省零开销;
  * 集中式对照臂为手动 opt-in:设同一环境变量后由驱动方调用
    install_lwd_layer_trace(model)——基类 NPUModelRunner 保持零
    LWD 符号,weave guard 白名单维持为空;
  * 只打校验和(numel/sum/abs_sum/max_abs/l2/head4),不打全量张量;
  * 只追踪前 _MAX_STEPS 个 model forward(prefill + 若干 decode),
    防止 decode 循环刷屏。
日志前缀统一 [layer-trace],与 [Lwd][DUMP] 并存便于 grep。
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
    """张量校验和单行描述;非张量/空张量返回占位。"""
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
    """从模块入参/出参里取 hidden 张量(容 tuple/None 包裹)。"""
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
    """定位文本 backbone:兼容普通结构(model.model.layers)与 VL 包装
    结构(model.language_model[.model].layers,顶层是 visual+language_model)。
    返回 (backbone, layers);找不到返回 (None, None)。"""
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
    """L0+L2:层数/结构日志 + 逐层 forward hook(幂等,env 门控)。

    覆盖 embed_tokens / decoder layers / final norm / lm_head;对拍时
    集中式与边云同用本安装器,层列表本身即半模型嫌疑的裁决证据。"""
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
