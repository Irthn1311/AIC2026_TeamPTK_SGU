import pytest

from triage_eg.evaluation.official_metrics import (
    final_score,
    kis_rscore,
    qa_rscore,
    trake_rscore,
)


def test_kis_correct_and_incorrect_cases():
    ground_truth = {"video_id": "v1", "start_frame": 10, "end_frame": 20}
    assert kis_rscore({"video_id": "v1", "frame_id": 15}, ground_truth) == 1
    assert kis_rscore({"video_id": "v2", "frame_id": 15}, ground_truth) == 0
    assert kis_rscore({"video_id": "v1", "frame_id": 21}, ground_truth) == 0


def test_qa_normalized_answer():
    ground_truth = {
        "video_id": "v1",
        "start_frame": 10,
        "end_frame": 20,
        "accepted_answers": ["Ho Chi Minh City"],
    }
    assert (
        qa_rscore(
            {"video_id": "v1", "frame_id": 15, "answer": "  HO CHI   MINH CITY "}, ground_truth
        )
        == 1
    )
    assert qa_rscore({"video_id": "v1", "frame_id": 15, "answer": "Hanoi"}, ground_truth) == 0


def test_trake_three_of_four_events():
    ground_truth = {
        "video_id": "v1",
        "intervals": [[0, 10], [20, 30], [40, 50], [60, 70]],
    }
    prediction = {"video_id": "v1", "frame_ids": [5, 25, 45, 99]}
    assert trake_rscore(prediction, ground_truth) == 0.75


def test_final_score_from_cutoff_scores():
    assert final_score({1: 0.5, 5: 0.8, 20: 0.8, 50: 0.8, 100: 0.8}) == pytest.approx(0.74)
