from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.e2e1 import (
    CanonicalTriagePipeline,
    E2E1Settings,
    QueryPlan,
    materialize_inference_only,
    numeric_tokens,
    plan_query,
    route_intent,
)
from triage_eg.e2e1.pipeline import PredictionResult, _renumber, _strictly_increasing
from triage_eg.e2e1.qa import VOCABULARIES, dynamic_object_candidates
from triage_eg.e2e1.runner import extract_development_bundle, run_prediction_variant


def kis_query() -> dict:
    return {"query_id": "K1", "task": "KIS", "query": "a red car", "language": "en"}


def qa_query() -> dict:
    return {
        "query_id": "Q1",
        "task": "QA",
        "query": "a car on a road",
        "question": "What color is the car?",
        "language": "en",
    }


def trake_query() -> dict:
    return {
        "query_id": "T1",
        "task": "TRAKE",
        "query": "three events",
        "event_count": 3,
        "event_descriptions": [
            {"event_id": "E1", "description": "person enters"},
            {"event_id": "E2", "description": "person sits"},
            {"event_id": "E3", "description": "person leaves"},
        ],
        "language": "en",
    }


def test_01_query_adapter_kis() -> None:
    plan = plan_query(kis_query())
    assert plan.task == "KIS" and plan.events == (("E1", "a red car"),)


def test_02_query_adapter_qa() -> None:
    plan = plan_query(qa_query())
    assert plan.grounding_text == "a car on a road" and plan.question == "What color is the car?"


def test_03_query_adapter_trake_uses_structured_events() -> None:
    plan = plan_query(trake_query())
    assert [text for _, text in plan.events] == ["person enters", "person sits", "person leaves"]


def test_04_trake_event_count_consistency() -> None:
    query = trake_query()
    query["event_count"] = 2
    with pytest.raises(ValueError, match="TRAKE_EVENT_COUNT_MISMATCH"):
        plan_query(query)


def test_05_trake_unsupported_event_count() -> None:
    query = trake_query()
    query["event_count"] = 5
    query["event_descriptions"] *= 2
    query["event_descriptions"] = query["event_descriptions"][:5]
    with pytest.raises(ValueError, match="UNSUPPORTED_EVENT_COUNT"):
        plan_query(query)


def test_06_inference_api_has_no_gt_parameter() -> None:
    parameters = inspect.signature(CanonicalTriagePipeline.predict_query).parameters
    assert not {"gt", "accepted_intervals", "accepted_answers", "event_intervals"} & set(parameters)


@pytest.mark.parametrize(
    "field", ["gt", "accepted_intervals", "correct_video", "accepted_answers", "event_intervals"]
)
def test_07_to_11_gt_fields_rejected(field: str) -> None:
    query = {**kis_query(), field: "forbidden"}
    with pytest.raises(ValueError, match="GT_FIELDS_FORBIDDEN"):
        plan_query(query)


def test_12_inference_directory_contains_queries_only(tmp_path: Path) -> None:
    source = tmp_path / "benchmark"
    source.mkdir()
    (source / "queries.jsonl").write_text(json.dumps(kis_query()) + "\n", encoding="utf-8")
    (source / "gt.jsonl").write_text("{}\n", encoding="utf-8")
    output = materialize_inference_only(source, tmp_path / "inference")
    assert [path.name for path in output.iterdir()] == ["queries.jsonl"]


def test_13_rank_normalization_is_one_based_contiguous() -> None:
    assert [row["rank"] for row in _renumber([{"x": 1}, {"x": 2}])] == [1, 2]


def test_14_strict_temporal_order() -> None:
    assert _strictly_increasing((1, 2, 4))
    assert not _strictly_increasing((1, 1, 4))


def test_15_qa_answer_vocab_is_finite_and_not_gt_derived() -> None:
    assert "COLOR" in VOCABULARIES
    assert all(
        "accepted_answers" not in candidate.canonical_id for candidate in VOCABULARIES["COLOR"]
    )


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Màu gì?", "COLOR"),
        ("What color is it?", "COLOR"),
        ("Ở đâu?", "LOCATION"),
        ("Where is this?", "LOCATION"),
        ("Con số nào?", "OCR_NUMERIC"),
        ("What text is shown?", "OCR_TEXT"),
    ],
)
def test_16_to_21_bilingual_intent_router(question: str, intent: str) -> None:
    assert route_intent(question) == intent


def test_22_dynamic_object_candidates_ignore_numeric_labels() -> None:
    values = dynamic_object_candidates(["/m/person", "84", "traffic_light"])
    assert [value.en_output for value in values] == ["person", "traffic light"]


def test_23_object_candidate_contract_has_no_bbox_input() -> None:
    assert "box" not in inspect.signature(dynamic_object_candidates).parameters


def test_24_numeric_ocr_parser_is_generic() -> None:
    assert numeric_tokens("Score 12 and 3.5, then -7") == ["12", "3.5", "-7"]


def test_25_frozen_configuration_rejects_m3() -> None:
    with pytest.raises(ValueError, match="frozen"):
        E2E1Settings(use_m3=True)


def test_26_frozen_configuration_rejects_event_graph() -> None:
    with pytest.raises(ValueError, match="frozen"):
        E2E1Settings(use_event_graph=True)


def test_27_frozen_configuration_rejects_tuning() -> None:
    with pytest.raises(ValueError, match="frozen"):
        E2E1Settings(max_predictions=99)


def test_28_prediction_result_preserves_original_frame_coordinate() -> None:
    result = PredictionResult({}, ({"frame_id": 321},), (), 0.1)
    assert result.predictions[0]["frame_id"] == 321


def test_29_no_timestamp_fps_reconstruction_in_pipeline_source() -> None:
    source = inspect.getsource(CanonicalTriagePipeline)
    assert "timestamp *" not in source and "pts_time *" not in source


def test_30_sealed_archive_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "dev.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("SEALED_FINAL_30/queries.jsonl", "")
    with pytest.raises(RuntimeError, match="SEALED_CONTENT_REJECTED"):
        extract_development_bundle(archive, tmp_path / "out")


def test_31_qa_grounding_plan_excludes_question() -> None:
    plan = plan_query(qa_query())
    assert plan.grounding_text != plan.question


def test_32_m3_event_graph_vlm_agent_nvdec_all_disabled() -> None:
    settings = E2E1Settings()
    assert not any(
        (
            settings.use_m3,
            settings.use_event_graph,
            settings.use_vlm,
            settings.use_agent,
            settings.use_nvdec_default,
        )
    )


def test_33_predict_queries_dispatches_without_gt(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = object.__new__(CanonicalTriagePipeline)
    monkeypatch.setattr(
        pipeline,
        "predict_query",
        lambda query, variant="P1_CANONICAL": SimpleNamespace(query=query, variant=variant),
    )
    output = pipeline.predict_queries([kis_query()], "P0_COARSE")
    assert output[0].variant == "P0_COARSE"


def _plan(task: str = "KIS") -> QueryPlan:
    return QueryPlan(
        "X1", task, "en", "scene", "what object?" if task == "QA" else None, (("E1", "scene"),)
    )


def _grounded(count: int, *, duplicate: bool = False) -> list[dict]:
    rows = []
    for index in range(count):
        frame_id = 1 if duplicate else index + 1
        rows.append(
            {
                "video_id": "L01_V001",
                "frame_id": frame_id,
                "coarse_frame_id": frame_id,
                "global_row": index,
                "original_frame_idx": frame_id,
            }
        )
    return rows


def test_34_kis_tuple_deduplication(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = object.__new__(CanonicalTriagePipeline)
    pipeline.settings = E2E1Settings()
    monkeypatch.setattr(
        pipeline, "_ground_single", lambda plan, variant: (_grounded(3, duplicate=True), [])
    )
    result = pipeline.predict_kis(_plan(), "P0_COARSE")
    assert len(result.predictions) == 1 and result.predictions[0]["rank"] == 1


def test_35_kis_output_is_bounded_to_100(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = object.__new__(CanonicalTriagePipeline)
    pipeline.settings = E2E1Settings()
    monkeypatch.setattr(pipeline, "_ground_single", lambda plan, variant: (_grounded(150), []))
    assert len(pipeline.predict_kis(_plan(), "P0_COARSE").predictions) == 100


def test_36_p0_never_calls_m1(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = object.__new__(CanonicalTriagePipeline)
    pipeline.settings = E2E1Settings()
    pipeline._single_pool_cache = {}
    candidate = {
        "video_id": "L01_V001",
        "global_row": 0,
        "original_frame_idx": 7,
        "score": 1.0,
    }
    monkeypatch.setattr(pipeline, "_scores", lambda *args: (object(), object(), {}))
    monkeypatch.setattr(pipeline, "_single_event_pool", lambda *args: (candidate,))
    monkeypatch.setattr(pipeline, "_refine", lambda *args: pytest.fail("P0 called M1"))
    rows, _ = pipeline._ground_single(_plan(), "P0_COARSE")
    assert rows[0]["frame_id"] == 7 and not rows[0]["m1_applied"]


def test_37_p1_m1_budget_is_exactly_top10(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = object.__new__(CanonicalTriagePipeline)
    pipeline.settings = E2E1Settings()
    candidates = tuple(
        {
            "video_id": "L01_V001",
            "global_row": index,
            "original_frame_idx": index + 1,
            "score": 1.0 - index / 100,
        }
        for index in range(12)
    )
    calls = []
    monkeypatch.setattr(pipeline, "_scores", lambda *args: (object(), object(), {}))
    monkeypatch.setattr(pipeline, "_single_event_pool", lambda *args: candidates)
    monkeypatch.setattr(
        pipeline,
        "_refine",
        lambda video, frame, text, vector: calls.append(frame) or {"refined_frame_idx": frame},
    )
    rows, _ = pipeline._ground_single(_plan(), "P1_CANONICAL")
    assert len(calls) == 10 and sum(row["m1_applied"] for row in rows) == 10


def test_38_qa_answer_is_never_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = object.__new__(CanonicalTriagePipeline)
    pipeline.settings = E2E1Settings()
    monkeypatch.setattr(pipeline, "_ground_single", lambda plan, variant: (_grounded(1), []))
    monkeypatch.setattr(pipeline, "_answer_row", lambda *args: ("unknown", {"intent": "OBJECT"}))
    result = pipeline.predict_qa(_plan("QA"), "P0_COARSE")
    assert result.predictions[0]["answer"] == "unknown"


def test_39_ocr_unavailable_fallback_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    import triage_eg.e2e1.pipeline as pipeline_module

    pipeline = object.__new__(CanonicalTriagePipeline)
    pipeline.settings = E2E1Settings()
    pipeline.ocr = SimpleNamespace(status="UNAVAILABLE")
    monkeypatch.setattr(pipeline, "_object_names", lambda *args: ((), 10))
    monkeypatch.setattr(pipeline, "_frame_embedding", lambda row: np.ones(512, dtype=np.float32))
    monkeypatch.setattr(
        pipeline,
        "_answer_embeddings",
        lambda intent, candidates: np.ones((len(candidates), 512), dtype=np.float32),
    )
    monkeypatch.setattr(pipeline_module, "score_answers", lambda *args: ("unknown", 0.0, 0.0))
    answer, diagnostic = pipeline._answer_row(_plan("QA"), _grounded(1)[0], "OCR_TEXT", 1)
    assert answer and diagnostic["answer_fallback_reason"] == "OCR_UNAVAILABLE"


def test_40_qa_frame_embedding_cache_hit() -> None:
    pipeline = object.__new__(CanonicalTriagePipeline)
    pipeline._frame_embedding_cache = {("L01_V001", 3): np.ones(512, dtype=np.float32)}
    pipeline.qa_frame_cache_hits = 0
    value = pipeline._frame_embedding({"video_id": "L01_V001", "frame_id": 3})
    assert value.shape == (512,) and pipeline.qa_frame_cache_hits == 1


def test_41_m1_nonmonotonic_chain_falls_back_entire_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = QueryPlan("T1", "TRAKE", "en", "events", None, (("E1", "a"), ("E2", "b"), ("E3", "c")))
    pipeline = object.__new__(CanonicalTriagePipeline)
    pipeline.settings = E2E1Settings()
    chain = {"video_id": "L01_V001", "frame_ids": (10, 20, 30), "score": 1.0}
    monkeypatch.setattr(pipeline, "_trake_chains", lambda plan: ([chain], [None] * 3, [{}] * 3))
    refined = iter((11, 25, 24))
    monkeypatch.setattr(pipeline, "_refine", lambda *args: {"refined_frame_idx": next(refined)})
    result = pipeline.predict_trake(plan, "P1_CANONICAL")
    assert result.predictions[0]["frame_ids"] == [10, 20, 30]
    assert result.diagnostics[0]["M1_ORDER_FALLBACK_TO_COARSE"] is True


def test_42_t3_top5_order_unchanged_before_m1(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = QueryPlan("T1", "TRAKE", "en", "events", None, (("E1", "a"), ("E2", "b")))
    pipeline = object.__new__(CanonicalTriagePipeline)
    pipeline.settings = E2E1Settings()
    chains = [
        {"video_id": "L01_V001", "frame_ids": (rank, rank + 10), "score": 100 - rank}
        for rank in range(1, 8)
    ]
    monkeypatch.setattr(pipeline, "_trake_chains", lambda plan: (chains, [None] * 2, [{}] * 2))
    monkeypatch.setattr(
        pipeline, "_refine", lambda video, frame, *args: {"refined_frame_idx": frame}
    )
    result = pipeline.predict_trake(plan, "P1_CANONICAL")
    assert [row["frame_ids"] for row in result.predictions[:5]] == [
        [rank, rank + 10] for rank in range(1, 6)
    ]


def test_43_trake_output_frame_count_equals_event_count(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = QueryPlan("T1", "TRAKE", "en", "events", None, (("E1", "a"), ("E2", "b")))
    pipeline = object.__new__(CanonicalTriagePipeline)
    pipeline.settings = E2E1Settings()
    chain = {"video_id": "L01_V001", "frame_ids": (1, 2), "score": 1.0}
    monkeypatch.setattr(pipeline, "_trake_chains", lambda plan: ([chain], [None] * 2, [{}] * 2))
    result = pipeline.predict_trake(plan, "P0_COARSE")
    assert len(result.predictions[0]["frame_ids"]) == len(plan.events)


def test_44_trake_filters_catalog_monotonic_raw_frame_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = QueryPlan("T1", "TRAKE", "en", "events", None, (("E1", "a"), ("E2", "b")))
    pipeline = object.__new__(CanonicalTriagePipeline)
    pipeline.settings = E2E1Settings()
    pipeline.groups = [SimpleNamespace(video_id="L01_V001", rows=np.arange(3, dtype=np.int64))]
    pipeline.runtime = SimpleNamespace(
        catalog=SimpleNamespace(
            map_row=lambda row: {"original_frame_idx": (10, 10, 20)[int(row)]}
        )
    )
    pipeline.trake_non_strict_coarse_paths_dropped = 0
    monkeypatch.setattr(
        pipeline,
        "_scores_many",
        lambda *args: (
            [np.ones(2, dtype=np.float32)] * 2,
            np.asarray(((0.9, 0.8, 0.7), (0.8, 0.9, 0.7)), dtype=np.float32),
            [{}, {}],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_fps_and_frames",
        lambda group: (1.0, np.asarray((10, 10, 20), dtype=np.int64)),
    )

    chains, _, _ = pipeline._trake_chains(plan)

    assert [chain["frame_ids"] for chain in chains] == [(10, 20)]
    assert pipeline.trake_non_strict_coarse_paths_dropped == 1


def test_45_prediction_runner_rejects_non_isolated_directory(tmp_path: Path) -> None:
    root = tmp_path / "inference"
    root.mkdir()
    (root / "queries.jsonl").write_text(json.dumps(kis_query()) + "\n", encoding="utf-8")
    (root / "gt.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="GT_UNAVAILABLE"):
        run_prediction_variant(SimpleNamespace(), root, "DEV_CROSS_60", "P0_COARSE", tmp_path)
