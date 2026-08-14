from __future__ import annotations

from aic2026_eval.scoring import evaluate
from aic2026_eval.validation import validate_predictions


def _queries() -> list[dict]:
    return [
        {"query_id": "K1", "task": "KIS", "query": "scene"},
        {"query_id": "Q1", "task": "QA", "query": "scene", "question": "what?"},
        {"query_id": "T1", "task": "TRAKE", "query": "events", "event_count": 2},
    ]


def _predictions() -> list[dict]:
    return [
        {"query_id": "K1", "rank": 1, "video_id": "L01_V001", "frame_id": 10},
        {
            "query_id": "Q1",
            "rank": 1,
            "video_id": "L01_V001",
            "frame_id": 10,
            "answer": "red",
        },
        {"query_id": "T1", "rank": 1, "video_id": "L01_V001", "frame_ids": [10, 20]},
    ]


def _gt() -> list[dict]:
    return [
        {"query_id": "K1", "correct_video": "L01_V001", "acceptable_intervals": [[5, 15]]},
        {
            "query_id": "Q1",
            "correct_video": "L01_V001",
            "acceptable_intervals": [[5, 15]],
            "accepted_answers": ["red"],
        },
        {
            "query_id": "T1",
            "correct_video": "L01_V001",
            "event_intervals": [[[5, 15]], [[16, 25]]],
        },
    ]


def test_valid_all_task_predictions_pass_shared_validator() -> None:
    summary, issues = validate_predictions(_queries(), _predictions())
    assert summary["status"] == "PASS" and not issues


def test_p0_and_p1_validate_independently() -> None:
    for _variant in ("P0_COARSE", "P1_CANONICAL"):
        summary, _ = validate_predictions(_queries(), _predictions())
        assert summary["status"] == "PASS"


def test_shared_evaluator_scores_three_tasks() -> None:
    summary, per_query, slices, _ = evaluate(_queries(), _predictions(), _gt())
    assert summary["final_score"] == 1.0
    assert len(per_query) == 3
    assert set(slices) >= {"task:KIS", "task:QA", "task:TRAKE"}


def test_development_scores_remain_separate() -> None:
    cross = evaluate(_queries(), _predictions(), _gt())[0]
    l21 = evaluate(_queries(), _predictions(), _gt())[0]
    output = {"DEV_CROSS_60": cross, "DEV_L21_150": l21}
    assert set(output) == {"DEV_CROSS_60", "DEV_L21_150"}
