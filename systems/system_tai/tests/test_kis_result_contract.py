import pytest
from system_tai.common.schemas import CandidateFrame, KISResult


def test_kis_result_contract():
    candidates = (
        CandidateFrame(
            video_id="L21_V001",
            frame_id=100,
            clip_row=0,
            keyframe_order=0,
            score=0.35,
            rank=1,
            source="clip_exact",
        ),
        CandidateFrame(
            video_id="L21_V002",
            frame_id=200,
            clip_row=1,
            keyframe_order=1,
            score=0.30,
            rank=2,
            source="clip_exact",
        ),
    )
    res = KISResult(query_id="QA-01", ranked_candidates=candidates)

    assert hasattr(res, "ranked_candidates")
    assert len(res.ranked_candidates) == 2
    assert res.ranked_candidates[0].video_id == "L21_V001"
    assert res.ranked_candidates[0].frame_id == 100
    assert res.ranked_candidates[0].score == 0.35
