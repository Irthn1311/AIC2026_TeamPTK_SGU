from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from triage_eg.diagnostics.d1_grounding_attribution import (
    D1Settings,
    SemanticUnitSnapshot,
    audit_event_unit,
    audit_trake_query,
    blind_translation_rows,
    capture_inference_snapshot,
    classify_single_event,
    classify_trake,
    frame_distance_to_intervals,
    select_review_cases,
    strict_target_chain_exists,
    translation_review_instructions,
    translation_surface_checks,
    verify_historical_reproduction,
)
from triage_eg.diagnostics.d1_grounding_attribution.trake import _global_t3_chain_ranking
from triage_eg.e2e1.contracts import QueryPlan
from triage_eg.e2e1.pipeline import PredictionResult
from triage_eg.experiments.reference_rt1.scoring import VideoRows
from triage_eg.experiments.t3_diverse_temporal import POOL_LIMIT, REGION_RADIUS_SECONDS


def _catalog() -> SimpleNamespace:
    return SimpleNamespace(
        original_idx=np.asarray([0, 10, 10, 20, 5, 15, 25], dtype=np.int64),
        mapping_fps=np.ones(7, dtype=np.float64),
    )


def _groups() -> list[VideoRows]:
    return [
        VideoRows("L01_V001", np.asarray([0, 1, 2, 3]), True),
        VideoRows("L01_V002", np.asarray([4, 5, 6]), True),
    ]


def _unit(
    *,
    query_id: str = "K1",
    task: str = "KIS",
    event_id: str | None = None,
    scores: list[float] | None = None,
    source: str = "Một xe có chữ 'COVIDIA' số 2026",
    target: str = "A car with the text 'COVIDIA' number 2026",
) -> SemanticUnitSnapshot:
    return SemanticUnitSnapshot(
        unit_id=f"{query_id}:{event_id or 'E1'}",
        query_id=query_id,
        task=task,
        event_id=event_id,
        source_language="vi",
        source_text=source,
        embedding=np.ones(2, dtype=np.float32),
        scores=np.asarray(scores or [0.1, 0.9, 0.8, 0.7, 0.95, 0.6, 0.5], dtype=np.float32),
        encoding={
            "translated_text": target,
            "clip_input_text": target,
            "translation_applied": True,
        },
    )


def _event_audit(interval: list[list[int]] | None = None) -> dict:
    groups = _groups()
    return audit_event_unit(
        _unit(),
        correct_video="L01_V001",
        interval_value=interval or [[9, 11]],
        groups=groups,
        group_by_video={row.video_id: row for row in groups},
        catalog=_catalog(),
    )


def _taxonomy_row(**overrides) -> dict:
    row = {
        "g1_has_target": False,
        "has_btc_target": True,
        "t3_pool_has_target": True,
        "target_within_video_rank": 1,
        "entered_final_top100": True,
        "correct_video_rank": 1,
        "g0_rank": 1,
    }
    return {**row, **overrides}


def _trake_taxonomy_row(**overrides) -> dict:
    row = {
        "g1_top100_full_target_chain_exists": False,
        "btc_target_chain_exists": True,
        "t3_target_chain_exists": True,
        "events": [
            {
                "has_btc_target": True,
                "t3_pool_has_target": True,
                "target_within_video_rank": 1,
            },
            {
                "has_btc_target": True,
                "t3_pool_has_target": True,
                "target_within_video_rank": 2,
            },
        ],
    }
    return {**row, **overrides}


def test_01_d1_inference_phase_rejects_gt_fields() -> None:
    plan = QueryPlan("K1", "KIS", "vi", "text", None, (("E1", "text"),))
    result = PredictionResult(plan.as_dict(), (), (), 0.1)
    pipeline = SimpleNamespace(_encoded_text={}, _score_cache={}, _single_pool_cache={})
    with pytest.raises(RuntimeError, match="GT_FIELD"):
        capture_inference_snapshot(
            pipeline,
            {
                "sha256": "abc",
                "validation": {"status": "PASS"},
                "queries": [{"query_id": "K1", "task": "KIS", "correct_video": "L01_V001"}],
                "plans": [plan],
                "results": [result],
            },
        )


def test_02_predictions_must_be_hashed_before_snapshot() -> None:
    with pytest.raises(RuntimeError, match="HASHED"):
        capture_inference_snapshot(SimpleNamespace(), {"sha256": ""})


def test_03_04_05_translation_blind_rows_have_no_gt_or_outcome() -> None:
    row = blind_translation_rows({_unit().unit_id: _unit()})[0]
    forbidden = {
        "correct_video",
        "acceptable_intervals",
        "event_intervals",
        "retrieval_rank",
        "retrieval_score",
        "success",
        "failure",
    }
    assert not forbidden & set(row)


def test_06_post_gt_module_has_no_prediction_feedback_call() -> None:
    from triage_eg.diagnostics.d1_grounding_attribution.runner import run_post_gt_attribution

    source = inspect.getsource(run_post_gt_attribution)
    assert "predict_" not in source and "_scores(" not in source and "_encode(" not in source


def test_07_btc_target_detection_interval_hit() -> None:
    assert _event_audit()["has_btc_target"] is True


def test_08_duplicate_frame_rows_preserved() -> None:
    assert _event_audit()["target_btc_rows_inside_gt"] == 2


def test_09_unique_frame_count_differs_from_row_count() -> None:
    row = _event_audit()
    assert row["target_unique_frame_ids_inside_gt"] == 1
    assert row["target_unique_frame_ids_inside_gt"] != row["target_btc_rows_inside_gt"]


@pytest.mark.parametrize(("frame", "expected"), [(10, 0), (5, 4), (15, 4), (30, 19)])
def test_10_nearest_frame_distance(frame: int, expected: int) -> None:
    assert frame_distance_to_intervals(frame, [(9, 11)]) == expected


def test_11_12_no_timestamp_reconstruction_and_original_frame_idx_used() -> None:
    source = inspect.getsource(audit_event_unit)
    assert "timestamp" not in source and "pts_time" not in source
    assert "original_idx" in source


def test_13_within_video_rank_uses_frozen_stable_scores() -> None:
    assert _event_audit()["best_target_within_video_rank"] == 1


def test_14_global_target_rank_uses_stable_full_vector() -> None:
    assert _event_audit()["best_target_global_frame_rank"] == 2


def test_15_correct_video_rank_uses_strongest_frame() -> None:
    assert _event_audit()["correct_video_rank_by_best_frame"] == 2


def test_16_gt_is_used_only_after_score_snapshot_finalized() -> None:
    source = inspect.getsource(capture_inference_snapshot)
    assert "ground_truth" not in source and "interval" not in source


def test_17_score_vector_is_not_modified_by_audit() -> None:
    unit = _unit()
    before = unit.scores.copy()
    groups = _groups()
    audit_event_unit(
        unit,
        correct_video="L01_V001",
        interval_value=[[9, 11]],
        groups=groups,
        group_by_video={row.video_id: row for row in groups},
        catalog=_catalog(),
    )
    assert np.array_equal(unit.scores, before)


def test_18_d1_reuses_existing_t3_builder() -> None:
    source = inspect.getsource(audit_event_unit)
    assert "build_diverse_event_pool" in source


def test_19_20_t3_constants_frozen() -> None:
    settings = D1Settings()
    assert settings.t3_pool_limit == POOL_LIMIT == 10
    assert settings.t3_region_radius_seconds == REGION_RADIUS_SECONDS == 3.0


def test_21_t3_pool_target_hit_detection() -> None:
    assert _event_audit()["t3_pool_has_target"] is True


def test_22_nearest_t3_distance() -> None:
    assert _event_audit()["t3_best_distance_to_gt"] == 0


def test_23_t3_creates_no_non_mapping_anchor() -> None:
    row = _event_audit()
    frames = set(_catalog().original_idx[:4].tolist())
    assert {item["original_frame_idx"] for item in row["t3_pool"]} <= frames


def test_24_success_taxonomy_is_first() -> None:
    assert classify_single_event(_taxonomy_row(g1_has_target=True), D1Settings()) == (
        "SUCCESS_G1_TARGET_HIT"
    )


def test_25_no_btc_target_taxonomy() -> None:
    assert classify_single_event(_taxonomy_row(has_btc_target=False), D1Settings()) == (
        "BTC_REPRESENTATION_GAP"
    )


def test_26_weak_target_beyond_pool_taxonomy() -> None:
    row = _taxonomy_row(t3_pool_has_target=False, target_within_video_rank=11)
    assert classify_single_event(row, D1Settings()) == "TARGET_SEMANTIC_SCORE_WEAK"


def test_27_competitive_target_missing_t3_taxonomy() -> None:
    row = _taxonomy_row(t3_pool_has_target=False, target_within_video_rank=10)
    assert classify_single_event(row, D1Settings()) == "T3_REGION_REPRESENTATIVE_GAP"


def test_28_global_video_ranking_taxonomy() -> None:
    row = _taxonomy_row(entered_final_top100=False, correct_video_rank=6, g0_rank=101)
    assert classify_single_event(row, D1Settings()) == "GLOBAL_VIDEO_RANKING_GAP"


def test_29_g1_allocation_taxonomy() -> None:
    row = _taxonomy_row(entered_final_top100=False, correct_video_rank=2, g0_rank=90)
    assert classify_single_event(row, D1Settings()) == "G1_ALLOCATION_GAP"


def test_30_btc_target_chain_true() -> None:
    assert strict_target_chain_exists([[1, 3], [2, 4], [5]])


def test_31_btc_target_chain_false() -> None:
    assert not strict_target_chain_exists([[5], [4], [6]])


def test_32_33_t3_target_chain_requires_each_event_and_strict_order() -> None:
    assert strict_target_chain_exists([[1], [2]])
    assert not strict_target_chain_exists([[2], []])
    assert not strict_target_chain_exists([[2], [1]])


def test_34_individual_hits_without_monotonic_chain_taxonomy() -> None:
    row = _trake_taxonomy_row(t3_target_chain_exists=False)
    assert classify_trake(row, D1Settings()) == "MONOTONIC_COMPOSITION_GAP"


def test_35_target_chain_beyond_output_taxonomy() -> None:
    assert classify_trake(_trake_taxonomy_row(), D1Settings()) == ("GLOBAL_CHAIN_RANKING_GAP")


def test_36_event_id_order_is_preserved_in_snapshot_dataclass() -> None:
    units = [_unit(query_id="T1", task="TRAKE", event_id=value) for value in ("E1", "E2", "E3")]
    assert [unit.event_id for unit in units] == ["E1", "E2", "E3"]


def test_37_quoted_literal_checker() -> None:
    anomaly, reasons = translation_surface_checks(
        "Tiêu đề 'COVIDIA'", "The title disappeared", source_language="vi"
    )
    assert anomaly and "QUOTED_LITERAL_LOST_OR_CHANGED" in reasons


def test_38_number_preservation_checker() -> None:
    anomaly, reasons = translation_surface_checks("Năm 2026", "The year 2025", source_language="vi")
    assert anomaly and "NUMBER_LOST_OR_CHANGED" in reasons


def test_39_acronym_preservation_checker() -> None:
    anomaly, reasons = translation_surface_checks("Biển COVIDIA", "A sign", source_language="vi")
    assert anomaly and "ACRONYM_LOST_OR_CHANGED" in reasons


def test_40_empty_translation_checker() -> None:
    assert translation_surface_checks("một người", "", source_language="vi")[0]


def test_41_surface_anomaly_is_not_semantic_fail() -> None:
    row = blind_translation_rows({_unit(target="").unit_id: _unit(target="")})[0]
    assert row["translation_surface_anomaly"] is True
    assert "semantic" not in row and "verdict" not in row


def test_42_no_reference_translation_is_fabricated() -> None:
    row = blind_translation_rows({_unit().unit_id: _unit()})[0]
    assert "reference_translation" not in row
    assert "Only for `FAIL`" in translation_review_instructions()


def test_trake_global_ranking_reproduces_frozen_visible_output() -> None:
    groups, catalog = _groups(), _catalog()
    units = [
        _unit(
            query_id="T1",
            task="TRAKE",
            event_id="E1",
            scores=[0.1, 0.95, 0.8, 0.2, 0.4, 0.3, 0.1],
        ),
        _unit(
            query_id="T1",
            task="TRAKE",
            event_id="E2",
            scores=[0.1, 0.2, 0.1, 0.9, 0.2, 0.3, 0.4],
        ),
    ]
    _, visible = _global_t3_chain_ranking(units, groups, catalog)
    predictions = [
        {
            "query_id": "T1",
            "rank": row["rank"],
            "video_id": row["video_id"],
            "frame_ids": list(row["frame_ids"]),
        }
        for row in visible
    ]
    _, query = audit_trake_query(
        units,
        ground_truth={
            "query_id": "T1",
            "correct_video": "L01_V001",
            "event_intervals": [[9, 11], [19, 21]],
        },
        predictions=predictions,
        groups=groups,
        group_by_video={row.video_id: row for row in groups},
        catalog=catalog,
        settings=D1Settings(),
    )
    assert query["t3_global_ranking_reproduced"] is True
    assert query["best_correct_video_chain_global_rank"] is not None
    assert {
        "target_within_video_rank",
        "target_global_rank",
        "correct_video_rank",
        "nearest_btc_distance",
        "nearest_t3_distance",
    } <= set(query["events"][0])


def test_trake_global_ranking_mismatch_fails_closed() -> None:
    groups, catalog = _groups(), _catalog()
    units = [
        _unit(query_id="T1", task="TRAKE", event_id="E1"),
        _unit(query_id="T1", task="TRAKE", event_id="E2"),
    ]
    with pytest.raises(RuntimeError, match="GLOBAL_RANKING_REPRODUCTION_MISMATCH"):
        audit_trake_query(
            units,
            ground_truth={
                "query_id": "T1",
                "correct_video": "L01_V001",
                "event_intervals": [[9, 11], [19, 21]],
            },
            predictions=[
                {
                    "query_id": "T1",
                    "rank": 1,
                    "video_id": "L01_V999",
                    "frame_ids": [1, 2],
                }
            ],
            groups=groups,
            group_by_video={row.video_id: row for row in groups},
            catalog=catalog,
            settings=D1Settings(),
        )


def test_reproduction_not_checked_without_optional_artifact(tmp_path: Path) -> None:
    report = verify_historical_reproduction(
        {
            "predictions": [],
            "sha256": "frozen-prediction-hash",
            "validation": {"status": "PASS"},
        },
        None,
        tmp_path,
    )
    assert report["status"] == "NOT_CHECKED"


def test_review_selection_is_bounded_and_deterministic() -> None:
    single = [
        {
            "query_id": f"K{index}",
            "task": "KIS",
            "primary_failure_reason": "TARGET_SEMANTIC_SCORE_WEAK",
        }
        for index in range(30)
    ]
    assert select_review_cases(single, []) == select_review_cases(single, [])
    assert len(select_review_cases(single, [])) <= 18
