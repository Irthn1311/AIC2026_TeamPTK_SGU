"""Skeleton for converting feature hits into original-frame candidates."""

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
        raise NotImplementedError("CandidateConstructor.build is not implemented")
