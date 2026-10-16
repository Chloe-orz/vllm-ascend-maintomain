#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Ascend-side parallel group initialization for LWD edge-cloud mode.

The native ``init_ascend_model_parallel`` splits global ranks on a
uniform (dp, pp, pcp, tp) grid, which cannot express the asymmetric
"edge edge_npu_count + cloud cloud_npu_count" topology. This module
builds the ascend-side communication groups (MC2 etc.) per the LWD
layout and injects them into
``vllm_ascend.distributed.parallel_state``; it is invoked by the
``_init_worker_distributed_environment`` override in LwdCloudWorker /
LwdEdgeWorker, replacing the native entry point.
"""

import torch
from vllm.distributed.parallel_state import (
    get_world_group,
    init_model_parallel_group,
)
from vllm.logger import init_logger

import vllm_ascend.distributed.parallel_state as ascend_parallel_state
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import flashcomm2_enable

logger = init_logger(__name__)


def init_lwd_ascend_model_parallel(parallel_config) -> None:
    """Build the ascend-side communication groups per the LWD layout.

    Mirrors the feature-free path of native
    ``init_ascend_model_parallel``: MC2 only, plus the dynamic_eplb /
    multistream_overlap_gate variants that share its grouping.
    """
    if ascend_parallel_state.model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    lwd = parallel_config.lwd_config
    edge_count = lwd.edge_npu_count
    cloud_count = lwd.cloud_npu_count
    world_per_dp = edge_count + cloud_count
    data_parallel_size = parallel_config.data_parallel_size
    world_size = torch.distributed.get_world_size()
    assert world_size == data_parallel_size * world_per_dp, (
        f"Lwd world size mismatch: got {world_size}, expected "
        f"{data_parallel_size} * ({edge_count} + {cloud_count})")
    backend = torch.distributed.get_backend(get_world_group().device_group)

    # MC2 groups: split by pp segment — segment 0 holds the edge ranks,
    # segment 1 this instance's cloud ranks, aligned with the vllm-side
    # PP grouping ([edge0, cloud0] two-stage chain + singletons) link
    # semantics.
    mc2_groups: list[list[int]] = []
    for dp_idx in range(data_parallel_size):
        base = dp_idx * world_per_dp
        mc2_groups.append([base + r for r in range(edge_count)])
        mc2_groups.append([base + edge_count + r for r in range(cloud_count)])
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

    logger.info(
        "Lwd ascend model parallel initialized: world_size=%s, MC2 groups=%s",
        world_size,
        mc2_groups,
    )
