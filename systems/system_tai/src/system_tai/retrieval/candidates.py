"""Convert feature-row retrieval hits into original-frame candidates."""

from __future__ import annotations

from collections.abc import Sequence

from system_tai.common.schemas import CandidateFrame, FrameRecord, RetrievalHit


class CandidateConstructor:
    def build(
        self,
        query_id: str,
        hits: Sequence[RetrievalHit],
        frame_records: Sequence[FrameRecord],
    ) -> Sequence[CandidateFrame]:
        if not query_id.strip():
            raise ValueError("query_id must not be empty")
        by_row: dict[int, FrameRecord] = {}
        for record in frame_records:
            if record.clip_row in by_row:
                raise ValueError(f"duplicate FrameRecord for clip_row {record.clip_row}")
            by_row[record.clip_row] = record
        candidates: list[CandidateFrame] = []
        for hit in hits:
            try:
                frame = by_row[hit.clip_row]
            except KeyError as exc:
                raise ValueError(f"missing FrameRecord for clip_row {hit.clip_row}") from exc
            candidates.append(
                CandidateFrame(
                    video_id=frame.video_id,
                    frame_id=frame.actual_frame_id,
                    clip_row=hit.clip_row,
                    keyframe_order=(
                        frame.keyframe_order
                        if frame.keyframe_order is not None
                        else frame.physical_row
                    ),
                    score=hit.score,
                    rank=hit.rank,
                    source="clip_exact",
                    diagnostic_metadata={"mapping_version": frame.mapping_version},
                )
            )
        return tuple(candidates)
