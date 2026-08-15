from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.e2e1 import E2E1Settings
from triage_eg.e2e1.contracts import QueryPlan
from triage_eg.e2e1.pipeline import CanonicalTriagePipeline, PredictionResult
from triage_eg.e2eg1 import (
    E2EG1Settings,
    SafeCoveragePipeline,
    coverage_order,
    filter_machine_ids,
    g0_order,
    rank_video_hypotheses,
    safe_alternative_order,
)
from triage_eg.e2eg1.contracts import VARIANTS
from triage_eg.e2eg1.pipeline import is_opaque_machine_id
from triage_eg.e2eg1.runner import create_bundle, materialize_inference_only, run_prediction_variant
from triage_eg.experiments.moment_m1 import M1Settings
from triage_eg.experiments.t3_diverse_temporal import POOL_LIMIT, REGION_RADIUS_SECONDS


def _candidate(video: int, region: int, score: float, *, frame: int | None = None) -> dict:
    frame_id = frame if frame is not None else video * 100 + region * 10
    return {
        "video_id": f"L01_V{video:03d}",
        "original_frame_idx": frame_id,
        "global_row": video * 100 + region,
        "score": score,
        "event_region_id": f"V{video}:R{region}",
        "mapping_fps": 25.0,
        "n": region,
    }


def _pool(video_count: int = 6, regions: int = 3) -> list[dict]:
    values = [
        _candidate(video, region, 1.0 - region * 0.1 - video * 0.001)
        for video in range(1, video_count + 1)
        for region in range(1, regions + 1)
    ]
    return sorted(values, key=lambda row: (-row["score"], row["global_row"]))


def _rows_with_frames(rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "frame_id": row["original_frame_idx"],
            "coarse_frame_id": row["original_frame_idx"],
            "hypothesis_kind": "COARSE",
        }
        for row in rows
    ]


def _trake_plan() -> QueryPlan:
    return QueryPlan(
        "T1",
        "TRAKE",
        "en",
        "events",
        None,
        (("E1", "first"), ("E2", "second"), ("E3", "third")),
    )


def _configure_g2_pipeline(monkeypatch: pytest.MonkeyPatch, refined: list[tuple[int, ...]]):
    pipeline = object.__new__(SafeCoveragePipeline)
    pipeline.settings = E2EG1Settings()
    pipeline.refined_alternative_count = 0
    pipeline.refined_duplicate_dropped_count = 0
    pipeline.refined_order_invalid_dropped_count = 0
    coarse = [
        {
            "query_id": "T1",
            "video_id": f"L01_V{rank:03d}",
            "frame_ids": [rank * 10, rank * 10 + 3, rank * 10 + 6],
            "coarse_rank": rank,
            "hypothesis_kind": "COARSE",
            "chain": {
                "score": 100 - rank,
                "global_rows": (rank, rank + 1, rank + 2),
                "event_scores": (0.9, 0.8, 0.7),
                "region_ids": ("r1", "r2", "r3"),
            },
        }
        for rank in range(1, 8)
    ]
    monkeypatch.setattr(
        pipeline,
        "_coarse_trake_records",
        lambda plan: (coarse, [np.ones(2)] * 3, [{}, {}, {}]),
    )
    values = iter(value for chain in refined for value in chain)
    monkeypatch.setattr(
        pipeline,
        "_refine",
        lambda *args: {"refined_frame_idx": next(values), "refined_score": 0.75},
    )
    return pipeline, coarse


def test_01_g0_order_equals_e2e1_coarse_tuple_order() -> None:
    pool = [*_pool(), _pool()[0]]
    ordered, _ = g0_order(pool, E2EG1Settings())
    expected = []
    seen = set()
    for row in pool:
        key = row["video_id"], row["original_frame_idx"]
        if key not in seen:
            seen.add(key)
            expected.append(key)
    assert [(row["video_id"], row["original_frame_idx"]) for row in ordered] == expected


def test_02_g1_preserves_exact_global_top5() -> None:
    g0, _ = g0_order(_pool(), E2EG1Settings())
    g1, _ = coverage_order(_pool(), E2EG1Settings())
    assert g1[:5] == [{**row, "was_in_coverage_block": False} for row in g0[:5]]


def test_03_video_ranking_uses_strongest_candidate() -> None:
    ranking = rank_video_hypotheses(
        [_candidate(1, 1, 0.8), _candidate(1, 2, 0.2), _candidate(2, 1, 0.7)]
    )
    assert [(row["video_id"], row["video_score"]) for row in ranking] == [
        ("L01_V001", 0.8),
        ("L01_V002", 0.7),
    ]


def test_04_coverage_round_robins_top_video_regions() -> None:
    g1, _ = coverage_order(_pool(regions=4), E2EG1Settings())
    assert [(row["video_id"], row["within_video_region_rank"]) for row in g1[5:10]] == [
        (f"L01_V{video:03d}", 2) for video in range(1, 6)
    ]


def test_05_within_video_order_remains_descending_semantic_order() -> None:
    g1, _ = coverage_order(_pool(regions=4), E2EG1Settings())
    for video in range(1, 6):
        rows = [row for row in g1 if row["video_id"] == f"L01_V{video:03d}"]
        assert [row["within_video_region_rank"] for row in rows] == sorted(
            row["within_video_region_rank"] for row in rows
        )


def test_06_no_temporal_anchor_is_fabricated() -> None:
    pool = _pool()
    g1, _ = coverage_order(pool, E2EG1Settings())
    source = {(row["video_id"], row["original_frame_idx"]) for row in pool}
    assert {(row["video_id"], row["original_frame_idx"]) for row in g1} <= source


def test_07_global_tail_keeps_original_order() -> None:
    g1, _ = coverage_order(_pool(video_count=7, regions=2), E2EG1Settings())
    tail = [
        row for row in g1 if not row["was_in_protected_prefix"] and not row["was_in_coverage_block"]
    ]
    assert [row["original_global_rank"] for row in tail] == sorted(
        row["original_global_rank"] for row in tail
    )


def test_08_exact_tuple_deduplication() -> None:
    pool = _pool()
    pool.insert(1, dict(pool[0]))
    g1, _ = coverage_order(pool, E2EG1Settings())
    keys = [(row["video_id"], row["original_frame_idx"]) for row in g1]
    assert len(keys) == len(set(keys))


def test_09_coverage_ranks_are_contiguous() -> None:
    g1, _ = coverage_order(_pool(), E2EG1Settings())
    assert [row["coverage_rank"] for row in g1] == list(range(1, len(g1) + 1))


def test_10_max_predictions_is_bounded() -> None:
    g1, _ = coverage_order(_pool(video_count=20, regions=10), E2EG1Settings())
    assert len(g1) == 100


def test_11_coverage_order_is_deterministic() -> None:
    assert coverage_order(_pool(), E2EG1Settings()) == coverage_order(_pool(), E2EG1Settings())


def test_12_duplicate_mapping_frame_does_not_change_coordinate() -> None:
    pool = [_candidate(1, 1, 0.9, frame=10), _candidate(1, 2, 0.8, frame=10)]
    g1, _ = coverage_order(pool, E2EG1Settings())
    assert [(row["video_id"], row["original_frame_idx"]) for row in g1] == [("L01_V001", 10)]


def test_13_m1_refined_result_is_new_hypothesis_and_14_source_remains() -> None:
    coverage, _ = coverage_order(_pool(), E2EG1Settings())
    rows = _rows_with_frames(coverage)
    source = rows[0]
    alternative = {
        **source,
        "frame_id": source["frame_id"] + 1,
        "original_frame_idx": source["frame_id"] + 1,
        "hypothesis_kind": "M1_REFINED_ALTERNATIVE",
    }
    output, _ = safe_alternative_order(
        rows, {(source["video_id"], source["coarse_frame_id"]): alternative}, E2EG1Settings()
    )
    assert output[0]["hypothesis_kind"] == "COARSE"
    assert alternative in output and source in output


def test_15_refined_equal_coarse_does_not_duplicate() -> None:
    coverage, _ = coverage_order(_pool(), E2EG1Settings())
    rows = _rows_with_frames(coverage)
    output, stats = safe_alternative_order(
        rows, {(rows[0]["video_id"], rows[0]["coarse_frame_id"]): dict(rows[0])}, E2EG1Settings()
    )
    assert len(output) == len(rows)
    assert stats["refined_duplicate_dropped_count"] == 1


def test_16_refined_duplicate_of_other_coarse_is_deduplicated() -> None:
    coverage, _ = coverage_order(_pool(), E2EG1Settings())
    rows = _rows_with_frames(coverage)
    alt = {
        **rows[0],
        "video_id": rows[1]["video_id"],
        "frame_id": rows[1]["frame_id"],
        "hypothesis_kind": "M1_REFINED_ALTERNATIVE",
    }
    output, stats = safe_alternative_order(
        rows, {(rows[0]["video_id"], rows[0]["coarse_frame_id"]): alt}, E2EG1Settings()
    )
    assert stats["refined_duplicate_dropped_count"] == 1
    assert sum(row["frame_id"] == rows[1]["frame_id"] for row in output) == 1


def test_17_g2_single_event_top5_remains_coarse() -> None:
    coverage, _ = coverage_order(_pool(), E2EG1Settings())
    rows = _rows_with_frames(coverage)
    alternatives = {
        (row["video_id"], row["coarse_frame_id"]): {
            **row,
            "frame_id": row["frame_id"] + 1,
            "hypothesis_kind": "M1_REFINED_ALTERNATIVE",
        }
        for row in rows[:10]
    }
    output, _ = safe_alternative_order(rows, alternatives, E2EG1Settings())
    assert output[:5] == rows[:5]


def test_18_m1_budget_is_bounded_to_ten_unique_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = object.__new__(SafeCoveragePipeline)
    pipeline.settings = E2EG1Settings()
    pipeline.refined_alternative_count = 0
    pipeline.refined_duplicate_dropped_count = 0
    pipeline.machine_ids_filtered = 0
    monkeypatch.setattr(
        pipeline,
        "_scores",
        lambda *args: (np.ones(2), np.ones(len(_pool())), {"route": "en"}),
    )
    monkeypatch.setattr(pipeline, "_single_event_pool", lambda *args: tuple(_pool()))
    calls = []

    def refine(video, frame, *args):
        calls.append((video, frame))
        return {"refined_frame_idx": frame + 1, "refined_score": 0.5}

    monkeypatch.setattr(pipeline, "_refine", refine)
    rows, diagnostics = pipeline._ground_single(
        QueryPlan("K1", "KIS", "en", "query", None, (("E1", "query"),)), "G2_SAFE_M1"
    )
    assert len(calls) == len(set(calls)) == 10
    assert rows[:5] and all(row["hypothesis_kind"] == "COARSE" for row in rows[:5])
    provenance = [
        row for row in diagnostics if row["diagnostic_type"] == "m1_alternative_provenance"
    ]
    assert len(provenance) == 10 and all(row["emitted_after_dedup"] for row in provenance)


def test_19_m1_parameters_are_frozen_and_20_inference_selection_has_no_gt() -> None:
    assert M1Settings() == M1Settings(
        local_window_seconds=6.0, coarse_stride_frames=12, dense_radius_frames=15
    )
    source = inspect.getsource(SafeCoveragePipeline._ground_single)
    assert not any(field in source for field in ("accepted_intervals", "correct_video", "gt"))


def test_21_g0_and_22_g1_trake_equal_current_coarse(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = PredictionResult(
        _trake_plan().as_dict(),
        ({"query_id": "T1", "rank": 1, "video_id": "L01_V001", "frame_ids": [1, 2, 3]},),
        ({"variant": "P0_COARSE", "t3_score": 1.0},),
        0.1,
    )
    monkeypatch.setattr(CanonicalTriagePipeline, "predict_trake", lambda *args: expected)
    pipeline = object.__new__(SafeCoveragePipeline)
    for variant in ("G0_E2E1_COARSE", "G1_COVERAGE_COARSE"):
        result = pipeline.predict_trake(_trake_plan(), variant)
        assert result.predictions == expected.predictions


def test_23_g2_trake_top5_equal_g0_and_24_valid_alternative_is_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refined = [(11, 14, 17), (21, 24, 27), (31, 34, 37), (41, 44, 47), (51, 54, 57)]
    pipeline, coarse = _configure_g2_pipeline(monkeypatch, refined)
    result = pipeline.predict_trake(_trake_plan(), "G2_SAFE_M1")
    assert [row["frame_ids"] for row in result.predictions[:5]] == [
        row["frame_ids"] for row in coarse[:5]
    ]
    assert [row["frame_ids"] for row in result.predictions[5:10]] == [list(row) for row in refined]


def test_25_invalid_refined_chain_dropped_and_26_coarse_source_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refined = [(11, 17, 14)] * 5
    pipeline, coarse = _configure_g2_pipeline(monkeypatch, refined)
    result = pipeline.predict_trake(_trake_plan(), "G2_SAFE_M1")
    assert [row["frame_ids"] for row in result.predictions[:5]] == [
        row["frame_ids"] for row in coarse[:5]
    ]
    assert pipeline.refined_order_invalid_dropped_count == 5


def test_27_duplicate_refined_chain_is_deduplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    refined = [(11, 14, 17)] * 5
    pipeline, coarse = _configure_g2_pipeline(monkeypatch, refined)
    for row in coarse:
        row["video_id"] = "L01_V999"
    result = pipeline.predict_trake(_trake_plan(), "G2_SAFE_M1")
    assert sum(row["frame_ids"] == [11, 14, 17] for row in result.predictions) == 1


def test_28_event_count_exact_and_29_final_frames_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    refined = [(11, 14, 17), (21, 24, 27), (31, 34, 37), (41, 44, 47), (51, 54, 57)]
    pipeline, _ = _configure_g2_pipeline(monkeypatch, refined)
    result = pipeline.predict_trake(_trake_plan(), "G2_SAFE_M1")
    assert all(len(row["frame_ids"]) == 3 for row in result.predictions)
    assert all(
        all(a < b for a, b in zip(row["frame_ids"], row["frame_ids"][1:], strict=False))
        for row in result.predictions
    )


def test_30_no_timestamp_fps_reconstruction() -> None:
    source = inspect.getsource(SafeCoveragePipeline)
    assert "pts_time *" not in source and "timestamp *" not in source


@pytest.mark.parametrize("value", ["/m/0h9mv", "/m/019jd", "0cgh4"])
def test_31_32_machine_ids_are_not_natural_language_candidates(value: str) -> None:
    allowed, filtered = filter_machine_ids([value])
    assert not allowed and filtered == (value,) and is_opaque_machine_id(value)


def test_33_machine_ids_remain_in_diagnostics_and_34_human_labels_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = object.__new__(SafeCoveragePipeline)
    pipeline.settings = E2EG1Settings()
    pipeline.machine_ids_filtered = 0
    pipeline.ocr = SimpleNamespace(status="UNAVAILABLE")
    monkeypatch.setattr(pipeline, "_object_names", lambda *args: (("/m/0h9mv", "fire truck"), 10))
    monkeypatch.setattr(pipeline, "_frame_embedding", lambda row: np.ones(512, dtype=np.float32))
    monkeypatch.setattr(
        pipeline,
        "_answer_embeddings",
        lambda intent, candidates: np.ones((len(candidates), 512), dtype=np.float32),
    )
    answer, diagnostic = pipeline._answer_row(
        QueryPlan("Q1", "QA", "en", "truck", "what object", (("E1", "truck"),)),
        {"video_id": "L01_V001", "frame_id": 10, "coarse_frame_id": 10},
        "OBJECT",
        1,
    )
    assert answer and "/m/0h9mv" in diagnostic["raw_object_class_evidence"]
    assert "fire truck" in diagnostic["human_readable_object_candidates"]
    assert diagnostic["OBJECT_MACHINE_ID_FILTERED"] is True


def test_35_machine_filter_has_no_gt_dependency_and_36_answer_nonempty() -> None:
    source = inspect.getsource(filter_machine_ids)
    assert "gt" not in source and "accepted" not in source
    allowed, _ = filter_machine_ids(["/m/019jd", "person"])
    assert allowed == ("person",)


def test_37_inference_directory_contains_queries_only(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    (benchmark / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    (benchmark / "gt.jsonl").write_text("{}\n", encoding="utf-8")
    inference = materialize_inference_only(benchmark, tmp_path / "inference")
    assert [path.name for path in inference.iterdir()] == ["queries.jsonl"]


def test_38_gt_unavailable_during_prediction(tmp_path: Path) -> None:
    inference = tmp_path / "inference"
    inference.mkdir()
    (inference / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    (inference / "gt.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="GT_UNAVAILABLE"):
        run_prediction_variant(SimpleNamespace(), inference, "DEV_CROSS_60", VARIANTS[0], tmp_path)


def test_39_sealed_content_is_rejected_from_bundle(tmp_path: Path) -> None:
    root = tmp_path / "output"
    (root / "diagnostics").mkdir(parents=True)
    (root / "diagnostics/sealed.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        create_bundle(root, tmp_path / "bundle.zip")


def test_40_benchmark_scores_remain_separate_by_contract() -> None:
    from triage_eg.e2eg1.runner import BENCHMARK_SLUGS

    assert BENCHMARK_SLUGS == {"DEV_CROSS_60": "cross", "DEV_L21_150": "l21"}


def test_41_e2e1_package_does_not_depend_on_e2eg1() -> None:
    root = Path("src/triage_eg/e2e1")
    assert not any("e2eg1" in path.read_text(encoding="utf-8") for path in root.glob("*.py"))


def test_42_stage1_exact_scoring_is_inherited_unchanged() -> None:
    assert SafeCoveragePipeline._scores is CanonicalTriagePipeline._scores
    assert SafeCoveragePipeline._scores_many is CanonicalTriagePipeline._scores_many


def test_43_t3_constants_and_44_m1_constants_unchanged() -> None:
    settings = E2EG1Settings()
    assert POOL_LIMIT == settings.coverage_regions_per_video == 10
    assert REGION_RADIUS_SECONDS == 3.0 and settings.t3_selected_delta == 0.05
    assert M1Settings() == M1Settings(
        local_window_seconds=6.0, coarse_stride_frames=12, dense_radius_frames=15
    )


def test_g0_pipeline_predictions_match_e2e1_p0(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = QueryPlan("K1", "KIS", "en", "query", None, (("E1", "query"),))
    pool = tuple(_pool())
    vector = np.ones(2)
    scores = np.ones(len(pool))

    base = object.__new__(CanonicalTriagePipeline)
    base.settings = E2E1Settings()
    monkeypatch.setattr(base, "_scores", lambda *args: (vector, scores, {}))
    monkeypatch.setattr(base, "_single_event_pool", lambda *args: pool)

    g1 = object.__new__(SafeCoveragePipeline)
    g1.settings = E2EG1Settings()
    monkeypatch.setattr(g1, "_scores", lambda *args: (vector, scores, {}))
    monkeypatch.setattr(g1, "_single_event_pool", lambda *args: pool)

    assert (
        g1.predict_kis(plan, "G0_E2E1_COARSE").predictions
        == base.predict_kis(plan, "P0_COARSE").predictions
    )


def test_bundle_contains_no_model_arrays(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (root / "README.md").write_text("ok", encoding="utf-8")
    bundle = create_bundle(root, tmp_path / "bundle.zip")
    with ZipFile(bundle) as archive:
        assert archive.namelist() == ["README.md"]


def test_trake_event_interval_normalization_uses_shared_contract() -> None:
    from triage_eg.e2eg1.runner import _event_interval

    assert _event_interval([9, 11]) == [(9, 11)]
    assert _event_interval({"start_frame": 19, "end_frame": 21}) == [(19, 21)]
