#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Ascend-side parallel group initialization for LWD edge-cloud mode.

The native ``init_ascend_model_parallel`` splits global ranks on a
uniform (dp, pp, pcp, tp) grid, which cannot express the asymmetric
edge/cloud topology. Rank sets come from the YAML LwdTopology carried
by ``VllmConfig.lwd_config`` (the main-branch config generation):
each instance link pairs an edge DP with a cloud DP and lists both
rank sets explicitly. This module builds the ascend-side groups
(MC2 etc.) from those rank sets and injects them into
``vllm_ascend.distributed.parallel_state``; it is invoked by the
``_init_worker_distributed_environment`` override in LwdCloudWorker /
LwdEdgeWorker, replacing the native entry point.
"""

import torch
from vllm.distributed.parallel_state import (
    get_world_group,
    init_model_parallel_group,
)
from vllm.lwd_logger import lwd_logger

import vllm_ascend.distributed.parallel_state as ascend_parallel_state
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import flashcomm2_enable



def init_lwd_ascend_model_parallel(vllm_config) -> None:
    """Build the ascend-side communication groups per the LWD layout.

    Mirrors the feature-free path of native
    ``init_ascend_model_parallel``: MC2 only, plus the dynamic_eplb /
    multistream_overlap_gate variants that share its grouping.
    """
    if ascend_parallel_state.model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    lwd = vllm_config.lwd_config
    if lwd is None:
        raise RuntimeError(
            "init_lwd_ascend_model_parallel requires VllmConfig.lwd_config "
            "(the YAML LwdTopology generation on main)")
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)

    # MC2 groups per instance link: segment 0 holds the edge ranks,
    # segment 1 the paired cloud ranks, both read verbatim from the
    # topology (no arithmetic rank layout is assumed).
    mc2_groups: list[list[int]] = []
    covered: set[int] = set()
    for link in lwd.topology.instance_links:
        edge_inst = lwd.topology.edges[link.edge]
        cloud_inst = lwd.topology.clouds[link.cloud]
        if link.dp_idx >= len(edge_inst.dp) or link.dp_idx >= len(cloud_inst.dp):
            raise ValueError(
                f"Lwd instance link dp_idx {link.dp_idx} out of range: "
                f"edge dp={len(edge_inst.dp)} cloud dp={len(cloud_inst.dp)}")
        mc2_groups.append(list(edge_inst.dp[link.dp_idx].ranks))
        mc2_groups.append(list(cloud_inst.dp[link.dp_idx].ranks))
        covered.update(edge_inst.dp[link.dp_idx].ranks)
        covered.update(cloud_inst.dp[link.dp_idx].ranks)
    if len(covered) != world_size:
        raise ValueError(
            "Lwd topology ranks do not cover the comm world: covered "
            f"{sorted(covered)} of world_size={world_size}")
    ascend_parallel_state._MC2 = init_model_parallel_group(
        mc2_groups,
        get_world_group().local_rank,
        backend,
        group_name="mc2",
    )

    ascend_config = get_ascend_config()
    if ascend_config.eplb_config.dynamic_eplb:
        ascend_parallel_state._DYNAMIC_EPLB = init_model_parallel_group(
            mc2_groups,
            get_world_group().local_rank,
            backend,
            group_name="dynamic_eplb",
        )
    if ascend_config.multistream_overlap_gate:
        ascend_parallel_state._FC3_QUANT_X = init_model_parallel_group(
            mc2_groups,
            get_world_group().local_rank,
            backend,
            group_name="fc3_quant_x",
        )

    # The features below depend on uniformly-gridded communication groups
    # and are not yet adapted to the asymmetric LWD layout; fail
    # explicitly when enabled so a missing group never surfaces at
    # runtime.
    finegrained = ascend_config.finegrained_tp_config
    enabled_finegrained = [
        name
        for name, size in (
            ("oproj", finegrained.oproj_tensor_parallel_size),
            ("lmhead", finegrained.lmhead_tensor_parallel_size),
            ("embedding", finegrained.embedding_tensor_parallel_size),
            ("mlp", finegrained.mlp_tensor_parallel_size),
        )
        if size and size > 0
    ]
    unsupported: list[str] = []
    if enabled_finegrained:
        unsupported.append(f"finegrained tp ({', '.join(enabled_finegrained)})")
    if flashcomm2_enable():
        unsupported.append("flashcomm2")
    if ascend_config.layer_sharding is not None:
        unsupported.append("layer_sharding")
    if unsupported:
        raise NotImplementedError(
            "features not yet supported in Lwd edge-cloud mode: "
            + ", ".join(unsupported)
        )

    lwd_logger.info(
        "Lwd ascend model parallel initialized: world_size=%s, MC2 groups=%s",
        world_size,
        mc2_groups,
    )
