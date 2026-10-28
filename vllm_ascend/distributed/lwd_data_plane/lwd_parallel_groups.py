#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Ascend-side groups of the HCCL world shared by the LWD edge and cloud.

The native ``init_ascend_model_parallel`` slices the ranks on a uniform
(dp, pp, pcp, tp) grid, which cannot express the different edge and cloud
rank counts; the groups come from ``LwdParallelGroupsSpec`` instead.
"""

from vllm.config import get_current_vllm_config
from vllm.config.lwd import LwdConfig
from vllm.distributed.lwd import LwdParallelGroupsSpec

from vllm_ascend.ascend_config import get_ascend_config


def lwd_mc2_group_ranks(lwd_config: LwdConfig) -> list[list[int]]:
    """MC2 group ranks.

    These match the vLLM-side EP groups, as the native MC2 groups do: one
    group per edge instance, one group spanning every DP's ranks of each
    cloud instance. A MoE cloud shards its experts dp x tp over all its
    ranks, so the MC2 all2all of an engine must reach every cloud rank of
    its instance -- grouping per engine would let tokens reach only a
    fraction of the experts. Without MoE (or one DP per instance) the EP
    groups equal the TP groups.
    """
    _lwd_check_ascend_features()
    spec = LwdParallelGroupsSpec.from_topology(lwd_config.topology)
    config = get_current_vllm_config()
    is_moe = config is None or config.model_config is None or config.model_config.is_moe
    groups = spec.ep_groups if is_moe else spec.tp_groups
    return [list(group) for group in groups]


# TODO(lwd): move this check to NPUPlatform.check_and_update_config, so that
# an unsupported config fails at startup, before any worker joins the world.
def _lwd_check_ascend_features() -> None:
    ascend_config = get_ascend_config()
    finegrained = ascend_config.finegrained_tp_config
    unsupported = [
        name
        for name, enabled in (
            ("KV layer parallelism", ascend_config.kvpp_config.size > 1),
            ("PD head ratio > 1", ascend_config.pd_head_ratio > 1),
            ("finegrained oproj TP", finegrained.oproj_tensor_parallel_size > 0),
            ("finegrained lmhead TP", finegrained.lmhead_tensor_parallel_size > 0),
            ("finegrained embedding TP", finegrained.embedding_tensor_parallel_size > 0),
            ("finegrained mlp TP", finegrained.mlp_tensor_parallel_size > 0),
        )
        if enabled
    ]
    if unsupported:
        raise NotImplementedError(f"[LWD] not supported in edge-cloud mode: {', '.join(unsupported)}")
