"""
Unit test for TemporalCandidateGrouper and CandidateSegment
"""

import pytest
from system_tai.retrieval.temporal_grouper import (
    TemporalCandidateGrouper,
    CandidateSegment,
)


def test_temporal_grouper_basic_clustering():
    grouper = TemporalCandidateGrouper(window_seconds=2.5, max_duration_seconds=5.0)

    # 2 distinct clusters in L30_V046:
    # Cluster 1 around 97s (frames 2420, 2425, 2430)
    # Cluster 2 around 194s (frames 4860, 4865)
    candidates = [
        {"video_id": "L30_V046", "frame_id": 2420, "pts_time": 96.8, "score": 0.35},
        {"video_id": "L30_V046", "frame_id": 2425, "pts_time": 97.0, "score": 0.42},
        {"video_id": "L30_V046", "frame_id": 2430, "pts_time": 97.2, "score": 0.38},
        {"video_id": "L30_V046", "frame_id": 4860, "pts_time": 194.4, "score": 0.30},
        {"video_id": "L30_V046", "frame_id": 4865, "pts_time": 194.6, "score": 0.32},
        {"video_id": "L21_V003", "frame_id": 1739, "pts_time": 69.56, "score": 0.40},
    ]

    segments = grouper.group_candidates(candidates)

    # Expect 3 distinct segments:
    # 1. L30_V046 [96.8s - 97.2s]
    # 2. L21_V003 [69.56s]
    # 3. L30_V046 [194.4s - 194.6s]
    assert len(segments) == 3

    # Segment 1 should be the highest scoring (around 97s with score ~0.40)
    seg1 = segments[0]
    assert seg1.video_id == "L30_V046"
    assert seg1.representative_frame["frame_id"] == 2425
    assert len(seg1.member_frames) == 3
    assert seg1.start_sec == 96.8
    assert seg1.end_sec == 97.2

    # Segment 2
    seg2 = segments[1]
    assert seg2.video_id == "L21_V003"
    assert seg2.representative_frame["frame_id"] == 1739

    # Segment 3 (the 194s cluster)
    seg3 = segments[2]
    assert seg3.video_id == "L30_V046"
    assert seg3.representative_frame["frame_id"] == 4865
    assert len(seg3.member_frames) == 2
