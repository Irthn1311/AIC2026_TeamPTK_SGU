import pytest
from system_tai.retrieval.query_decomposition import QueryVariants
from system_tai.retrieval.multi_variant_fusion import (
    fuse_multi_variant_video_ranks,
    ChannelContribution,
    VideoCandidateWithProvenance,
)


def test_fuse_multi_variant_video_ranks():
    variants = QueryVariants(
        literal="man getting out of car",
        action_focused="man exiting vehicle",
    )
    channel_video_rankings = {
        "clip_b32:literal": [("L21_V001", 100, 0.28), ("L21_V002", 200, 0.25)],
        "clip_b32:action_focused": [("L21_V003", 300, 0.31), ("L21_V001", 105, 0.29)],
    }
    baseline_top_video_ids = ["L21_V001", "L21_V002"]

    res = fuse_multi_variant_video_ranks(
        query_id="QA-01",
        variants=variants,
        channel_video_rankings=channel_video_rankings,
        baseline_top_video_ids=baseline_top_video_ids,
        rrf_k=60.0,
    )

    assert len(res.ranked_videos) == 3
    # L21_V001 was ranked in both channels, so it should have highest RRF score
    assert res.ranked_videos[0].video_id == "L21_V001"
    assert len(res.ranked_videos[0].contributions) == 2

    # L21_V003 was not in baseline, so it must be in novel_rescue_videos
    novel_ids = [v.video_id for v in res.novel_rescue_videos]
    assert "L21_V003" in novel_ids
    assert "L21_V001" not in novel_ids
