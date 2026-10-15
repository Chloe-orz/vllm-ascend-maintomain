# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU checks for wire payloads and runner handoff; these do not emulate HCCL."""

import ast
import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
WORKER = ROOT / "vllm_ascend/worker"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_method(path: Path, class_name: str, method_name: str, **symbols: Any) -> Callable[..., Any]:
    """Execute the real method without importing the hardware plugin."""
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    scope = {"torch": torch, **symbols}
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    return scope[method_name]


@pytest.mark.parametrize("pcp", [1, 2])
def test_cloud_initialization_reads_pcp_from_parallel_config(pcp: int) -> None:
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        use_async_scheduling=True,
        speculative_config=SimpleNamespace(method="mtp"),
        dcp_size=1,
        use_async_spec_decode=False,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )
    cache = object()
    initialize = load_method(
        WORKER / "lwd_cloud/lwd_cloud_model_runner.py",
        "LwdCloudModelRunner",
        "__init__",
        super=lambda: SimpleNamespace(__init__=lambda *_: None),
        LwdControlDiag=SimpleNamespace(disable_down_enabled=lambda: False),
        LwdTokenRankEncoder=lambda: None,
        LwdDraftCache=lambda: cache,
        hash_layer_count=lambda _: 0,
    )
    initialize(runner, SimpleNamespace(lwd_config=None), torch.device("cpu"))
    assert runner._lwd_draft_cache is (cache if pcp == 1 else None)


@pytest.mark.parametrize("pcp", [1, 2])
def test_cloud_preprocess_reads_pcp_without_legacy_runner_attribute(pcp: int) -> None:
    refreshed = []
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        use_async_spec_decode=True,
        enable_prompt_embeds=True,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        supports_mm_inputs=False,
        _lwd_refresh_decode_embedding_mask=refreshed.append,
        _lwd_preprocess_prompt_embeds=lambda *_: "remote prompt",
    )
    preprocess = load_method(
        WORKER / "lwd_cloud/lwd_cloud_model_runner.py",
        "LwdCloudModelRunner",
        "_preprocess",
        super=lambda: SimpleNamespace(_preprocess=lambda *_: "native"),
        get_pp_group=lambda: SimpleNamespace(is_first_rank=True),
    )
    scheduled = object()
    result = preprocess(runner, scheduled, 4)
    assert result == ("remote prompt" if pcp == 1 else "native")
    assert refreshed == ([scheduled] if pcp == 1 else [])


def test_rank_replay_uses_identical_tie_order_and_rejected_rows() -> None:
    rank = load_module("lwd_ranks", WORKER / "lwd_token_rank.py").LwdTokenRankEncoder()
    logits = torch.tensor([[2.0, 3.0, 3.0, 1.0], [4.0, 4.0, 0.0, 4.0], [0.0, 1.0, 2.0, 3.0]])
    tokens = torch.tensor([2, 3, -1])
    encoded = rank.encode(logits, tokens)
    order = torch.argsort(logits, descending=True, stable=True)
    assert encoded.tolist() == [1, 2, 0]
    assert order[torch.arange(2), encoded[:2].long()].tolist() == [2, 3]


def test_hash_payload_is_lossless_above_bfloat16_integer_precision() -> None:
    routing = load_module("lwd_hash", WORKER / "lwd_hash/lwd_hash_routing.py")
    embeds = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
    ids = torch.tensor([[[257, 1023]], [[513, 1024]]], dtype=torch.int32)
    payload = routing.pack_hash_payload(embeds, ids)
    decoded, decoded_ids = routing.unpack_hash_payload(payload, 2, 4, 1, 2)
    assert torch.equal(decoded, embeds)
    assert torch.equal(decoded_ids, ids)
    with pytest.raises(ValueError, match="payload size"):
        routing.unpack_hash_payload(payload[:-1], 2, 4, 1, 2)


def test_hash_prompt_requires_received_rows_but_decode_uses_real_tokens() -> None:
    routing = load_module("lwd_hash", WORKER / "lwd_hash/lwd_hash_routing.py")
    state = routing.LwdHashRoutingState(1, 2, 8)
    state.add_chunk("p", 4, 0, torch.tensor([[[6, 7]], [[4, 5]]]))
    with pytest.raises(ValueError, match="Missing|missing"):
        state.build_batch(["p"], [2], [2], [4])
    remote, mask = state.build_batch(["p", "d"], [2, 1], [0, 5], [4, 4])
    table = torch.arange(16).view(8, 2) % 8
    merged = routing.merge_hash_expert_ids(table, torch.tensor([999, 999, 3]), remote[:, 0], mask)
    assert merged.tolist() == [[6, 7], [4, 5], [6, 7]]
    with pytest.raises(ValueError, match="out of range"):
        state.add_chunk("p", 4, 2, torch.tensor([[[8, 0]]]))


def make_mtp_request(prompt_ids: list[int] | None, prompt_len: int, output_ids: list[int]) -> SimpleNamespace:
    spec = importlib.util.find_spec("vllm")
    assert spec is not None and spec.origin is not None
    get_token_id = load_method(
        Path(spec.origin).parent / "v1/worker/gpu_input_batch.py", "CachedRequestState", "get_token_id"
    )
    request = SimpleNamespace(prompt_token_ids=prompt_ids, num_prompt_tokens=prompt_len, output_token_ids=output_ids)
    request.get_token_id = MethodType(get_token_id, request)
    return request


def test_mtp_backup_handles_chunked_prompt_and_mixed_decode_batch() -> None:
    path = ROOT / "vllm_ascend/spec_decode/llm_base_proposer.py"
    prepare = load_method(
        path,
        "AscendSpecDecodeBaseProposer",
        "prepare_next_token_ids_padded",
        np=np,
        DeviceOperator=SimpleNamespace(index_fill=torch.index_fill),
    )
    backup = load_method(WORKER / "lwd_cloud/lwd_mtp_proposer.py", "LwdMTPProposer", "_backup_token_id")
    requests = {
        "chunk": make_mtp_request(None, 1572, []),
        "tokens": make_mtp_request([41, 42], 2, []),
        "decode": make_mtp_request(None, 2, [60]),
        "completed": make_mtp_request(None, 2, [70]),
    }
    buffer = SimpleNamespace(np=np.empty(4, dtype=np.int64), gpu=torch.empty(4, dtype=torch.int64))
    buffer.copy_to_gpu = lambda n: buffer.gpu[:n].copy_(torch.from_numpy(buffer.np[:n]))
    proposer = SimpleNamespace(backup_next_token_ids=buffer)
    proposer._backup_token_id = MethodType(backup, proposer)
    batch = SimpleNamespace(
        num_reqs=4, req_ids=list(requests), num_tokens_no_spec=np.array([1572, 2, 3, 3]), vocab_size=100
    )
    sampled = torch.tensor([[99, -1, -1, -1], [98, -1, -1, -1], [51, 52, -1, -1], [70, -1, -1, -1]])
    next_ids, valid_counts = prepare(proposer, sampled, requests, batch, torch.tensor([0, 1]), 2)
    assert buffer.np.tolist() == [0, 42, 60, 70]
    assert next_ids.tolist() == [0, 42, 52, 70]
    assert valid_counts.tolist() == [0, 0, 2, 1]


@pytest.mark.parametrize("lwd", [False, True])
def test_mtp_backup_preserves_token_lookup_and_invalid_indices(lwd: bool) -> None:
    path, cls = (
        (WORKER / "lwd_cloud/lwd_mtp_proposer.py", "LwdMTPProposer")
        if lwd
        else (ROOT / "vllm_ascend/spec_decode/llm_base_proposer.py", "AscendSpecDecodeBaseProposer")
    )
    backup = load_method(path, cls, "_backup_token_id")
    tokens = make_mtp_request([11, 12], 2, [13])
    assert [backup(None, tokens, i) for i in range(4)] == [11, 12, 13, -1]
    embeds = make_mtp_request(None, 2, [13])
    assert backup(None, embeds, 2) == 13
    with pytest.raises(ValueError, match="ID is unknown"):
        backup(None, embeds, -1)
    if not lwd:
        with pytest.raises(ValueError, match="ID is unknown"):
            backup(None, embeds, 1)


def test_mtp_mixed_batch_shifts_remote_prompt_only() -> None:
    method = load_method(
        WORKER / "lwd_cloud/lwd_mtp_proposer.py", "LwdMTPProposer", "_prepare_first_pass_inputs_embeds"
    )
    target_embeds = torch.tensor([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [99.0, 99.0], [99.0, 99.0]])
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_computed_tokens_cpu=[0, 9], num_prompt_tokens=[8, 4], num_reqs=2),
        inputs_embeds=SimpleNamespace(gpu=target_embeds),
    )
    proposer = SimpleNamespace(
        runner=runner,
        inputs_embeds=torch.empty(8, 2),
        input_ids=torch.tensor([0, 0, 7, 3, 4]),
        model=SimpleNamespace(embed_input_ids=lambda ids: ids[:, None].expand(-1, 2).float()),
    )
    cad = SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 3, 5]))
    actual = method(proposer, None, 5, 6, cad)
    assert actual.tolist() == [[11.0, 11.0], [12.0, 12.0], [7.0, 7.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]]


def test_output_waits_for_its_metadata_and_owns_the_result() -> None:
    class AsyncOutput:
        def get_output(self) -> SimpleNamespace:
            return SimpleNamespace()

    method = load_method(
        WORKER / "lwd_cloud/lwd_cloud_worker.py",
        "LwdAsyncModelRunnerOutput",
        "get_output",
        AsyncModelRunnerOutput=AsyncOutput,
    )
    pinned = torch.zeros(3, dtype=torch.int32)
    ready = SimpleNamespace(synchronize=lambda: pinned.fill_(7))
    wrapper = SimpleNamespace(output=AsyncOutput(), carrier=(pinned, ["r"], 12, 0), ready=ready)
    result = method(wrapper)
    pinned.fill_(99)
    assert result.lwd_down_carrier[0].tolist() == [7, 7, 7]


@pytest.mark.parametrize("endpoint", [False, True])
def test_every_edge_tp_rank_embeds_but_only_endpoint_sends(endpoint: bool) -> None:
    import time

    calls = []
    logger = SimpleNamespace(debug=lambda *args: None, info=lambda *args: None)
    method = load_method(
        WORKER / "lwd_edge_worker.py",
        "LwdEdgeWorker",
        "_execute_lwd_embed",
        logger=logger,
        time=time,
        LwdCommRequest=lambda **kw: SimpleNamespace(**kw),
        LwdChannelType=SimpleNamespace(UP="up"),
    )

    def embed(ids: torch.Tensor) -> torch.Tensor:
        calls.append("embed")
        return ids[:, None].expand(-1, 2).float()

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(get_model=lambda: SimpleNamespace(embed_input_ids=embed)),
        _lwd_embed_ids_h2d=torch.tensor,
        _lwd_gather_chunk_mm=lambda meta: ([], None),
        _lwd_mm_features_dict={},
        tid2eid_tables=[],
        lwd_wire=SimpleNamespace(member=endpoint, key=(2, 7, 0)),
        comm_service=SimpleNamespace(submit_send=lambda req: calls.append(req)),
    )
    method(worker, 0, SimpleNamespace(token_ids=[[1, 2]]), SimpleNamespace(scheduled_new_reqs=[]))
    assert calls[0] == "embed"
    assert len(calls) == (2 if endpoint else 1)
    if endpoint:
        assert calls[1].connection_key == (2, 7, 0)
        assert calls[1].tensor.tolist() == [[1.0, 1.0], [2.0, 2.0]]
