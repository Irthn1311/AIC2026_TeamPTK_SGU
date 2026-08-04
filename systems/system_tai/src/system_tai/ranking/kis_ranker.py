"""Skeleton for deterministic grouping, deduplication, and Top-100 ranking."""

from __future__ import annotations

from collections.abc import Sequence

from system_tai.common.schemas import CandidateFrame, RankedKISRecord


class KISRanker:
    def rank(
        self,
        candidates: Sequence[CandidateFrame],
        *,
        limit: int = 100,
    ) -> Sequence[RankedKISRecord]:
        raise NotImplementedError("KISRanker.rank is not implemented")
