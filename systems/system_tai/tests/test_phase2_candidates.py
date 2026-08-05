from __future__ import annotations

import pytest

from system_tai.common.schemas import FrameRecord, RetrievalHit
from system_tai.retrieval.candidates import CandidateConstructor


def _frame(clip_row: int, frame_id: int) -> FrameRecord:
    return FrameRecord(
        video_id="L21_V001",
        actual_frame_id=frame_id,
        keyframe_order=clip_row + 10,
        clip_row=clip_row,
        pts_time=1.0,
        fps=30.0,
        mapping_version="btc-map",
        physical_row=clip_row,
    )


def test_candidate_constructor_preserves_frame_mapping_and_hit_fields() -> None:
    candidates = CandidateConstructor().build(
        "q",
        [RetrievalHit(clip_row=1, score=0.7, rank=1)],
        [_frame(0, 10), _frame(1, 999)],
    )
    candidate = candidates[0]
    assert (candidate.frame_id, candidate.clip_row, candidate.keyframe_order) == (999, 1, 11)
    assert (candidate.score, candidate.rank, candidate.source) == (0.7, 1, "clip_exact")


def test_candidate_constructor_rejects_missing_or_duplicate_mapping() -> None:
    hit = RetrievalHit(clip_row=1, score=0.7, rank=1)
    with pytest.raises(ValueError, match="missing FrameRecord"):
        CandidateConstructor().build("q", [hit], [_frame(0, 10)])
    with pytest.raises(ValueError, match="duplicate FrameRecord"):
        CandidateConstructor().build("q", [], [_frame(0, 10), _frame(0, 20)])
