"""
Temporal Candidate Grouper for KIS Retrieval
===========================================
Clusters raw candidate keyframes into temporally coherent video segments (shots/windows).
Maintains representative keyframe and full cluster evidence, allowing distinct action scenes
in the same video to be ranked and nominated independently.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass, field


@dataclass
class CandidateSegment:
    """A temporally contiguous cluster of keyframes within a video."""
    segment_id: str
    video_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    representative_frame: Dict[str, Any]
    member_frames: List[Dict[str, Any]] = field(default_factory=list)
    fusion_score: float = 0.0
    action_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        rep = dict(self.representative_frame)
        rep["segment_id"] = self.segment_id
        rep["video_id"] = self.video_id
        rep["start_sec"] = round(self.start_sec, 3)
        rep["end_sec"] = round(self.end_sec, 3)
        rep["duration_sec"] = round(self.duration_sec, 3)
        rep["cluster_size"] = len(self.member_frames)
        rep["member_frame_ids"] = [
            int(m.get("frame_id", m.get("frame_idx", 0))) for m in self.member_frames
        ]
        rep["member_pts"] = [
            round(float(m.get("pts_time", m.get("timestamp_sec", 0.0))), 2)
            for m in self.member_frames
        ]
        rep["fusion_score"] = round(float(self.fusion_score), 4)
        rep["final_score"] = round(float(self.final_score), 4)
        rep["rank"] = self.rank
        return rep


class TemporalCandidateGrouper:
    """
    Groups top candidates into coherent temporal segments based on frame proximity.
    """

    def __init__(
        self,
        window_seconds: float = 2.5,
        max_duration_seconds: float = 5.0,
    ):
        self.window_seconds = window_seconds
        self.max_duration_seconds = max_duration_seconds

    def group_candidates(
        self,
        raw_candidates: Sequence[Dict[str, Any]],
        action_variant_id: Optional[str] = None,
    ) -> List[CandidateSegment]:
        """
        Group raw candidates into CandidateSegment list.
        Preserves high-scoring moments while clustering nearby frames in the same video.
        """
        if not raw_candidates:
            return []

        # Partition candidates by video_id
        by_video: Dict[str, List[Dict[str, Any]]] = {}
        for c in raw_candidates:
            vid = str(c.get("video_id", "")).strip()
            if vid:
                by_video.setdefault(vid, []).append(c)

        segments: List[CandidateSegment] = []

        for vid, cands in by_video.items():
            # Sort chronologically for clustering
            sorted_cands = sorted(
                cands,
                key=lambda x: float(x.get("pts_time", x.get("timestamp_sec", (int(x.get("frame_id", 0)) / 25.0))))
            )

            current_group: List[Dict[str, Any]] = []

            for cand in sorted_cands:
                ts = float(cand.get("pts_time", cand.get("timestamp_sec", (int(cand.get("frame_id", 0)) / 25.0))))

                if not current_group:
                    current_group.append(cand)
                    continue

                prev_ts = float(current_group[-1].get("pts_time", current_group[-1].get("timestamp_sec", (int(current_group[-1].get("frame_id", 0)) / 25.0))))
                group_start_ts = float(current_group[0].get("pts_time", current_group[0].get("timestamp_sec", (int(current_group[0].get("frame_id", 0)) / 25.0))))
                group_span = ts - group_start_ts
                time_diff = ts - prev_ts

                # Check proximity window match
                if time_diff <= self.window_seconds and group_span <= self.max_duration_seconds:
                    current_group.append(cand)
                else:
                    segments.append(self._create_segment(vid, current_group, action_variant_id))
                    current_group = [cand]

            if current_group:
                segments.append(self._create_segment(vid, current_group, action_variant_id))

        # Sort all segments by final_score descending
        segments.sort(key=lambda s: s.final_score, reverse=True)
        for idx, seg in enumerate(segments, start=1):
            seg.rank = idx

        return segments

    def _create_segment(
        self,
        video_id: str,
        member_frames: List[Dict[str, Any]],
        action_variant_id: Optional[str] = None,
    ) -> CandidateSegment:
        """Create a CandidateSegment from member frames."""
        timestamps = [
            float(m.get("pts_time", m.get("timestamp_sec", (int(m.get("frame_id", 0)) / 25.0))))
            for m in member_frames
        ]
        start_sec = min(timestamps)
        end_sec = max(timestamps)
        duration_sec = max(0.04, end_sec - start_sec)

        # Representative frame: highest scoring candidate (or action-specific highest)
        def _score_fn(m: Dict[str, Any]) -> float:
            base_s = float(m.get("score", m.get("fusion_score", 0.0)))
            if action_variant_id and "scores_by_variant" in m and isinstance(m["scores_by_variant"], dict):
                var_s = float(m["scores_by_variant"].get(action_variant_id, 0.0))
                return 0.6 * base_s + 0.4 * var_s
            return base_s

        best_cand = max(member_frames, key=_score_fn)

        # Aggregate segment score (top-2 mean or max)
        scores = sorted([_score_fn(m) for m in member_frames], reverse=True)
        if len(scores) >= 2:
            agg_score = float((scores[0] + scores[1]) / 2.0)
        else:
            agg_score = float(scores[0])

        seg_id = f"{video_id}_{start_sec:.1f}s_{end_sec:.1f}s"

        return CandidateSegment(
            segment_id=seg_id,
            video_id=video_id,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=duration_sec,
            representative_frame=best_cand,
            member_frames=member_frames,
            fusion_score=agg_score,
            final_score=agg_score,
        )
