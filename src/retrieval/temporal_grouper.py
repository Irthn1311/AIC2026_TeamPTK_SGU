"""
Temporal Candidate Grouper for KIS Retrieval
===========================================
Groups raw candidate frames into temporally coherent video segments (shots/windows).
Maintains representative keyframe and full cluster evidence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class CandidateSegment:
    segment_id: str
    video_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    representative_frame: Dict[str, Any]
    member_frames: List[Dict[str, Any]] = field(default_factory=list)
    shot_id: int = -1
    fusion_score: float = 0.0
    rerank_score: Optional[float] = None
    final_score: float = 0.0
    rank: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        rep = dict(self.representative_frame)
        rep["segment_id"] = self.segment_id
        rep["start_sec"] = round(self.start_sec, 3)
        rep["end_sec"] = round(self.end_sec, 3)
        rep["duration_sec"] = round(self.duration_sec, 3)
        rep["cluster_size"] = len(self.member_frames)
        rep["member_frame_ids"] = [int(m.get("frame_idx", m.get("frame_id", 0))) for m in self.member_frames]
        rep["member_timestamps"] = [round(float(m.get("timestamp_seconds", m.get("timestamp_sec", 0.0))), 2) for m in self.member_frames]
        rep["fusion_score"] = round(float(self.fusion_score), 4)
        if self.rerank_score is not None:
            rep["rerank_score"] = round(float(self.rerank_score), 4)
        rep["final_score"] = round(float(self.final_score), 4)
        rep["rank"] = self.rank
        rep["evidence"] = self.evidence
        return rep


class TemporalCandidateGrouper:
    """
    Groups top-K candidates into coherent temporal segments based on shot boundaries
    and proximity windows.
    """

    def __init__(
        self,
        window_seconds: float = 1.5,
        max_duration_seconds: float = 3.0,
        prefer_shot_id: bool = True,
    ):
        self.window_seconds = window_seconds
        self.max_duration_seconds = max_duration_seconds
        self.prefer_shot_id = prefer_shot_id

    def group_candidates(self, raw_candidates: List[Dict[str, Any]]) -> List[CandidateSegment]:
        """
        Group raw candidates into CandidateSegment list.
        Preserves original ranking order while clustering nearby frames in the same video.
        """
        if not raw_candidates:
            return []

        # Partition candidates by video_id
        by_video: Dict[str, List[Dict[str, Any]]] = {}
        for c in raw_candidates:
            vid = str(c.get("video_id", "")).strip()
            by_video.setdefault(vid, []).append(c)

        segments: List[CandidateSegment] = []

        for vid, cands in by_video.items():
            # Sort chronologically for clustering
            sorted_cands = sorted(
                cands,
                key=lambda x: float(x.get("timestamp_seconds", x.get("timestamp_sec", 0.0)))
            )

            current_group: List[Dict[str, Any]] = []
            current_shot_id: int = -1

            for cand in sorted_cands:
                ts = float(cand.get("timestamp_seconds", cand.get("timestamp_sec", 0.0)))
                shot_id = int(cand.get("shot_id", -1)) if cand.get("shot_id") is not None and not math.isnan(float(cand.get("shot_id", -1))) else -1

                if not current_group:
                    current_group.append(cand)
                    current_shot_id = shot_id
                    continue

                prev_ts = float(current_group[-1].get("timestamp_seconds", current_group[-1].get("timestamp_sec", 0.0)))
                group_start_ts = float(current_group[0].get("timestamp_seconds", current_group[0].get("timestamp_sec", 0.0)))
                group_span = ts - group_start_ts
                time_diff = ts - prev_ts

                # Check grouping criteria:
                # 1. Shot match (if valid shot_id >= 0 and within max duration)
                shot_match = (
                    self.prefer_shot_id
                    and current_shot_id >= 0
                    and shot_id == current_shot_id
                    and group_span <= self.max_duration_seconds
                )

                # 2. Window match (within window_seconds and within max duration)
                window_match = (
                    time_diff <= self.window_seconds
                    and group_span <= self.max_duration_seconds
                )

                if shot_match or window_match:
                    current_group.append(cand)
                else:
                    # Finalize current group and start new one
                    segments.append(self._create_segment(vid, current_group))
                    current_group = [cand]
                    current_shot_id = shot_id

            if current_group:
                segments.append(self._create_segment(vid, current_group))

        # Score segments and sort by fusion_score descending
        for seg in segments:
            seg.final_score = seg.fusion_score

        segments.sort(key=lambda s: s.fusion_score, reverse=True)
        for idx, seg in enumerate(segments, start=1):
            seg.rank = idx

        return segments

    def _create_segment(self, video_id: str, member_frames: List[Dict[str, Any]]) -> CandidateSegment:
        """Create a CandidateSegment from a list of member frames."""
        timestamps = [
            float(m.get("timestamp_seconds", m.get("timestamp_sec", 0.0)))
            for m in member_frames
        ]
        start_sec = min(timestamps)
        end_sec = max(timestamps)
        duration_sec = max(0.1, end_sec - start_sec)

        # Representative frame: highest fusion score candidate
        best_cand = max(
            member_frames,
            key=lambda m: float(m.get("score", m.get("fused_score", 0.0)))
        )

        shot_ids = [
            int(m.get("shot_id", -1))
            for m in member_frames
            if m.get("shot_id") is not None and not math.isnan(float(m.get("shot_id", -1))) and int(m.get("shot_id", -1)) >= 0
        ]
        shot_id = shot_ids[0] if shot_ids else -1

        # Max fusion score in group
        fusion_score = float(best_cand.get("score", best_cand.get("fused_score", 0.0)))

        seg_id = f"{video_id}_{start_sec:.1f}s_{end_sec:.1f}s"

        return CandidateSegment(
            segment_id=seg_id,
            video_id=video_id,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=duration_sec,
            representative_frame=best_cand,
            member_frames=member_frames,
            shot_id=shot_id,
            fusion_score=fusion_score,
            final_score=fusion_score,
        )
