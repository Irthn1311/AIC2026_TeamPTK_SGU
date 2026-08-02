"""Reciprocal Rank Fusion for multiple retrieval branches."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import replace

from triage_eg.common.schemas import CandidateFrame


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[CandidateFrame]],
    *,
    weights: Mapping[str, float] | None = None,
    k: int = 60,
    candidate_key: Callable[[CandidateFrame], Hashable] | None = None,
) -> list[CandidateFrame]:
    """Fuse ranked branches with ``sum(weight / (k + rank))``."""

    if k < 0:
        raise ValueError("k must be non-negative")
    key_function = candidate_key or (lambda item: item.frame_uid)
    totals: dict[Hashable, float] = {}
    representatives: dict[Hashable, CandidateFrame] = {}
    for branch, candidates in ranked_lists.items():
        weight = 1.0 if weights is None else weights.get(branch, 1.0)
        if weight < 0:
            raise ValueError("RRF weights must be non-negative")
        seen: set[Hashable] = set()
        for fallback_rank, candidate in enumerate(candidates, start=1):
            key = key_function(candidate)
            if key in seen:
                continue
            seen.add(key)
            rank = candidate.rank if candidate.rank > 0 else fallback_rank
            totals[key] = totals.get(key, 0.0) + weight / (k + rank)
            current = representatives.get(key)
            if current is None or candidate.score > current.score:
                representatives[key] = candidate
    ordered = sorted(totals, key=lambda key: (-totals[key], str(key)))
    return [
        replace(
            representatives[key],
            score=totals[key],
            rank=rank,
            source_branch="rrf",
        )
        for rank, key in enumerate(ordered, start=1)
    ]
