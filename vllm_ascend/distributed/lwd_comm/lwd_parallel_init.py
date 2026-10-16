# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend groups from PR #63, with counts read from the current YAML schema."""

import torch.distributed as dist
from vllm.config import VllmConfig
from vllm.distributed.lwd import get_lwd_rank_groups
from vllm.distributed.parallel_state import get_world_group, init_model_parallel_group
from vllm.logger import logger

import vllm_ascend.distributed.parallel_state as ascend_parallel_state
from vllm_ascend.ascend_config import get_ascend_config


def init_lwd_ascend_model_parallel(config: VllmConfig) -> None:
    if ascend_parallel_state.model_parallel_initialized():
        return
    assert config.lwd_config is not None
    groups = get_lwd_rank_groups(config.lwd_config)
    assert dist.get_world_size() == sum(map(len, groups))
    world = get_world_group()
    backend = dist.get_backend(world.device_group)
    ascend = get_ascend_config()
    finegrained = ascend.finegrained_tp_config
    unsupported = [
        name
        for name, size in (
            ("oproj", finegrained.oproj_tensor_parallel_size),
            ("lmhead", finegrained.lmhead_tensor_parallel_size),
            ("embedding", finegrained.embedding_tensor_parallel_size),
            ("mlp", finegrained.mlp_tensor_parallel_size),
        )
        if size and size > 0
    ]
    # New-main KVPP needs extra groups that were absent from the source PR.
    if ascend.kvpp_config.size > 1:
        unsupported.append("kvpp")
    if unsupported:
        raise NotImplementedError("[LWD] unsupported no-PP parallel groups: " + ", ".join(unsupported))
    ascend_parallel_state._MC2 = init_model_parallel_group(groups, world.local_rank, backend, group_name="mc2")
    if ascend.eplb_config.dynamic_eplb:
        ascend_parallel_state._DYNAMIC_EPLB = init_model_parallel_group(
            groups, world.local_rank, backend, group_name="dynamic_eplb"
        )
    # The old flashcomm2/layer_sharding/FC3 group options were removed upstream.
    logger.info("[LWD] Ascend no-PP groups ready: MC2=%s", groups)
