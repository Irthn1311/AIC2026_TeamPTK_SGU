from __future__ import annotations

import pytest

from triage_eg.fs1.contracts import FS1Settings
from triage_eg.fs1.event_graph import Edge, EventGraph, Node
from triage_eg.fs1.fusion import assert_protected_prefix, fuse_tail, reciprocal_rank_fusion
from triage_eg.fs1.io import PreGTGate, write_jsonl
from triage_eg.fs1.model_registry import SequentialModelRegistry
from triage_eg.fs1.qa import GroundingCandidate, bounded_grounding_candidates, parse_qwen_output
from triage_eg.fs1.router import route_events, route_query


def row(rank: int, source: str = "V") -> dict:
    return {"query_id": "Q", "video_id": f"{source}{rank}", "frame_id": rank, "rank": rank}


def test_frozen_settings_reject_tuning() -> None:
    with pytest.raises(ValueError):
        FS1Settings(rrf_k=61)


def test_router_is_deterministic_and_availability_bounded() -> None:
    a = route_query("KIS", "người nói và biển chữ", available={"asr", "ocr"})
    assert a == route_query("KIS", "người nói và biển chữ", available={"ocr", "asr"})
    assert a.modalities == ("b0_visual", "ocr", "asr")
    assert route_query("KIS", "người nói", available=set()).modalities == ("b0_visual",)


def test_trake_routes_per_event() -> None:
    routes = route_events("TRAKE", ["read a sign", "then running"], available={"ocr", "action"})
    assert [route.event_index for route in routes] == [0, 1]
    assert "ocr" in routes[0].modalities and "action" in routes[1].modalities


def test_rrf60_formula_and_stable_tie() -> None:
    fused = reciprocal_rank_fusion([[row(1, "A")], [row(1, "B")]])
    assert fused[0]["fs1_rrf_score"] == pytest.approx(1 / 61)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=50)


@pytest.mark.parametrize("task", ["KIS", "TRAKE"])
def test_protected_top5_exact(task: str) -> None:
    b0 = [row(rank) for rank in range(1, 101)]
    out = fuse_tail(task, b0, [[row(rank, "X") for rank in range(1, 101)]])
    assert out[:5] == b0[:5]
    assert len(out) == 100
    assert_protected_prefix(task, b0, out)


def test_qa_has_no_protected_prefix_and_is_bounded() -> None:
    b0 = [row(rank) for rank in range(1, 101)]
    answer = [{**row(1, "Q"), "answer": "red"}]
    assert fuse_tail("QA", b0, [answer])[0]["video_id"] in {"Q1", "V1"}
    candidates = [GroundingCandidate("V", index, 30 - index, {}) for index in range(30)]
    assert len(bounded_grounding_candidates(candidates)) == 20


def test_qwen_cannot_create_candidate_ids_and_invalid_falls_back() -> None:
    candidate = GroundingCandidate("SAFE", 7, 1, {})
    parsed = parse_qwen_output(
        '{"video_id":"EVIL","frame_id":999,"answer":" red car ","evidence_sufficient":true}',
        candidate,
    )
    assert parsed["video_id"] == "SAFE" and parsed["frame_id"] == 7
    assert parse_qwen_output("not json", candidate) is None


def test_model_registry_is_sequential_and_optional_failure_disables() -> None:
    registry = SequentialModelRegistry()
    registry.load("A", object)
    with pytest.raises(RuntimeError):
        registry.load("B", object)
    registry.unload()
    status = registry.repair_optional("xclip", lambda: (_ for _ in ()).throw(RuntimeError("bad")))
    assert not status.enabled and status.status == "XCLIP_DISABLED"


def test_graph_requires_provenance_and_one_revision() -> None:
    with pytest.raises(ValueError):
        Node("e", "QueryEvent", {})
    graph = EventGraph("Q")
    graph.add_node(Node("e", "QueryEvent", {"query": "Q"}))
    graph.add_node(Node("c", "EventCandidate", {"source": "B0"}))
    graph.add_edge(Edge("c", "e", "SUPPORTS", {"rank": 1}))
    graph.revise_once("EXPLOIT", "e", lambda current, event: None)
    with pytest.raises(RuntimeError):
        graph.revise_once("EXPLORE", "e", lambda current, event: None)


def test_gt_gate_requires_all_six_prediction_hashes(tmp_path) -> None:
    gate = PreGTGate()
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, [row(1)])
    for benchmark in ("cross", "l21"):
        for arm in ("B0", "M0", "M1"):
            if (benchmark, arm) != ("l21", "M1"):
                gate.finalize(benchmark, arm, path)
    with pytest.raises(RuntimeError):
        gate.open_gt()
    gate.finalize("l21", "M1", path)
    gate.open_gt()
    assert gate.gt_opened
