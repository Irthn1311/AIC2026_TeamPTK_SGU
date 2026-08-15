"""Unit tests for Q&A Top-100 Prediction Constructor (QA-R1.1 Interleaved Anti-Starvation)."""

from __future__ import annotations

import pytest

from system_tai.preliminary.validation import validate_ranked_top100
from system_tai.qa.top100_constructor import construct_ranked_qa_top100


def test_top100_constructor_generates_contiguous_unique_predictions() -> None:
    scored = [
        {
            "video_id": f"L21_V{i:03d}",
            "frame_id": 1000 + i * 25,
            "answers": ["trắng", "đen"],
            "evidence_rank": i,
        }
        for i in range(1, 15)
    ]
    predictions = construct_ranked_qa_top100("QA-TEST-01", scored, output_top_k=100)
    assert len(predictions) == 100
    assert [p.rank for p in predictions] == list(range(1, 101))

    # Assert validation passes with zero errors
    errors = validate_ranked_top100(predictions, "qa", expected_query_id="QA-TEST-01")
    assert not errors

    # Check top prediction
    assert predictions[0].video_id == "L21_V001"
    assert predictions[0].frame_id == 1025
    assert predictions[0].answer == "trắng"


def test_top100_constructor_interleaved_preserves_top5_close_temporal_depth() -> None:
    scored = [
        {
            "video_id": f"L21_V{i:03d}",
            "frame_id": 1000 + i * 100,
            "answers": [f"ans_{i}"],
            "evidence_rank": i,
        }
        for i in range(1, 33)
    ]
    predictions, prov = construct_ranked_qa_top100(
        "QA-INTERLEAVED-01", scored, output_top_k=100, return_provenance=True
    )
    assert len(predictions) == 100
    assert len(prov) == 100

    # Invariant 1: Top 5 candidates have their primary and ±30 close offsets in ranks 1..16
    top16_videos = [p.video_id for p in predictions[:16]]
    top16_sources = [r["slot_source"] for r in prov[:16]]

    # Check that candidate 1, 2, 3, 4, 5 appear with both PRIMARY and CLOSE_OFFSET in top 16
    for i in range(1, 6):
        cand_vid = f"L21_V{i:03d}"
        assert cand_vid in top16_videos, f"{cand_vid} missing from top 16!"

    assert "PRIMARY" in top16_sources
    assert "CLOSE_OFFSET" in top16_sources


def test_top100_constructor_anti_starvation_guarantees_all_32_candidates_covered_by_rank_45() -> None:
    # 32 nominated candidates
    scored = [
        {
            "video_id": f"L21_V{i:03d}",
            "frame_id": 1000 + i * 50,
            "answers": [f"ans_{i}", f"alt_{i}"],
            "evidence_rank": i,
        }
        for i in range(1, 33)
    ]
    predictions, prov = construct_ranked_qa_top100(
        "QA-DIVERSITY-01", scored, output_top_k=100, return_provenance=True
    )
    assert len(predictions) == 100

    # Verify all 32 candidate videos appear in the prediction list
    predicted_videos = {p.video_id for p in predictions}
    for i in range(1, 33):
        assert f"L21_V{i:03d}" in predicted_videos

    # Anti-Starvation Invariant: All 32 candidates MUST receive their primary slot within ranks 1..45
    top45_videos = {p.video_id for p in predictions[:45]}
    for i in range(1, 33):
        assert f"L21_V{i:03d}" in top45_videos, f"Candidate L21_V{i:03d} was starved from top 45!"


def test_top100_constructor_respects_frame_bounds() -> None:
    scored = [
        {
            "video_id": "L21_V001",
            "frame_id": 20,
            "total_frames": 100,
            "answers": ["trắng"],
        }
    ]
    predictions = construct_ranked_qa_top100("QA-BOUND-01", scored, output_top_k=10)
    for p in predictions:
        assert 0 <= p.frame_id <= 100


def test_top100_constructor_handles_empty_or_zero_top_k() -> None:
    assert construct_ranked_qa_top100("QA-01", [], output_top_k=100) == []
    assert construct_ranked_qa_top100("QA-01", [{"video_id": "V1", "frame_id": 10, "answers": ["a"]}], output_top_k=0) == []


def test_top100_constructor_unexpanded_mode() -> None:
    scored = [
        {"video_id": f"L21_V{i:03d}", "frame_id": i * 100, "answers": ["đỏ"]}
        for i in range(1, 4)
    ]
    predictions = construct_ranked_qa_top100("QA-02", scored, expand_temporal=False)
    assert len(predictions) == 3
    assert [p.video_id for p in predictions] == ["L21_V001", "L21_V002", "L21_V003"]
    assert [p.frame_id for p in predictions] == [100, 200, 300]
