# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise rank/connection selection without importing the NPU runtime."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/lwd_comm/topology.py"
spec = importlib.util.spec_from_file_location("lwd_wire_topology_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_endpoint_peers_keep_global_yaml_ranks() -> None:
    link = module.LwdWireConnection((0, 0, 0), (5, 2), (4, 0, 3, 1))
    assert link.endpoint_ranks == (5, 4)
    assert link.peer_for(5) == 4
    assert link.peer_for(4) == 5
    assert link.includes(2)
    with pytest.raises(ValueError, match="not a P2P endpoint"):
        link.peer_for(2)


def test_shared_endpoint_requires_an_explicit_connection() -> None:
    first = module.LwdWireConnection((0, 0, 0), (0,), (2, 3))
    second = module.LwdWireConnection((1, 0, 0), (1,), (2, 3))
    plan = (first, second)
    with pytest.raises(ValueError, match="found 2"):
        module.select_lwd_wire_connection(plan, 2)
    assert module.select_lwd_wire_connection(plan, 2, second.key) == second
    with pytest.raises(ValueError, match="found 0"):
        module.select_lwd_wire_connection(plan, 0, second.key)


def test_nonleader_edge_worker_keeps_its_connection_and_rejects_foreign_batches() -> None:
    key = (0, 0, 0)
    link = SimpleNamespace(
        key=key,
        dp_idx=0,
        edge=SimpleNamespace(ranks=(5, 2)),
        cloud=SimpleNamespace(ranks=(4, 0, 3, 1)),
    )
    cfg = SimpleNamespace(role="edge", connections=(link,))
    bindings = module.bind_lwd_worker_connections(cfg, tp_rank=1)
    assert bindings[key].edge_ranks == (5, 2)
    assert module.resolve_lwd_batch_connection(SimpleNamespace(connection_key=key), bindings) == key
    for invalid in (None, (0, 1, 0), (0, 0, 1)):
        with pytest.raises(ValueError, match="not bound"):
            module.resolve_lwd_batch_connection(SimpleNamespace(connection_key=invalid), bindings)
    with pytest.raises(ValueError, match="TP rank"):
        module.bind_lwd_worker_connections(cfg, tp_rank=2)
    with pytest.raises(ValueError, match="no connection"):
        module.bind_lwd_worker_connections(cfg, tp_rank=0, dp_idx=1)
