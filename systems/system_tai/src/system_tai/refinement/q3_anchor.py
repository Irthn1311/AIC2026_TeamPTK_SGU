"""Deterministic selection and same-slot integration for Q3 anchor refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from system_tai.common.schemas import KISResult
from system_tai.refinement.models import Phase3Candidate, RefinedCandidate, RefinementStatus
from system_tai.retrieval.video_restricted import VIDEO_CONDITIONED_KEYFRAME_DIVERSITY


@dataclass(frozen=True, slots=True)
class Q3AnchorSelection:
    eligible: tuple[Phase3Candidate, ...]
    selected: tuple[Phase3Candidate, ...]


@dataclass(frozen=True, slots=True)
class Q3AnchorIntegration:
    result: KISResult
    refined_count: int
    kept_original_count: int
    collision_skip_count: int
    failure_count: int


def _restricted_cosine(candidate: Phase3Candidate) -> float | None:
    value = candidate.retrieval_provenance.get("q3_restricted_cosine_score")
    if type(value) not in {int, float}:
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _is_authoritative_q3_anchor(
    candidate: Phase3Candidate,
    *,
    protected_prefix_rank: int,
) -> bool:
    provenance = candidate.retrieval_provenance
    restricted_rank = provenance.get("q3_restricted_rank")
    return (
        candidate.rank > protected_prefix_rank
        and provenance.get("q3_policy") == VIDEO_CONDITIONED_KEYFRAME_DIVERSITY
        and provenance.get("candidate_source")
        == "video_conditioned_keyframe_diversity"
        and _restricted_cosine(candidate) is not None
        and type(restricted_rank) is int
        and restricted_rank > 0
    )


def select_q3_anchor_candidates(
    candidates: tuple[Phase3Candidate, ...],
    *,
    protected_prefix_rank: int,
    max_extra_q3_anchors: int,
) -> Q3AnchorSelection:
    if type(protected_prefix_rank) is not int or protected_prefix_rank < 0:
        raise ValueError("protected_prefix_rank must be a non-negative integer")
    if type(max_extra_q3_anchors) is not int or max_extra_q3_anchors <= 0:
        raise ValueError("max_extra_q3_anchors must be a positive integer")

    eligible = tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if _is_authoritative_q3_anchor(
                    candidate,
                    protected_prefix_rank=protected_prefix_rank,
                )
            ),
            key=lambda candidate: (
                -float(_restricted_cosine(candidate)),
                candidate.rank,
                candidate.video_id,
                candidate.frame_id,
            ),
        )
    )
    selected: list[Phase3Candidate] = []
    selected_ranks: set[int] = set()
    selected_videos: set[str] = set()
    for candidate in eligible:
        if candidate.video_id in selected_videos:
            continue
        selected.append(candidate)
        selected_ranks.add(candidate.rank)
        selected_videos.add(candidate.video_id)
        if len(selected) >= max_extra_q3_anchors:
            return Q3AnchorSelection(eligible=eligible, selected=tuple(selected))
    for candidate in eligible:
        if candidate.rank in selected_ranks:
            continue
        selected.append(candidate)
        if len(selected) >= max_extra_q3_anchors:
            break
    return Q3AnchorSelection(eligible=eligible, selected=tuple(selected))


def integrate_q3_anchor_refinements(
    baseline: KISResult,
    records: tuple[RefinedCandidate, ...],
) -> Q3AnchorIntegration:
    output = list(baseline.ranked_candidates)
    rank_to_index: dict[int, int] = {}
    for index, candidate in enumerate(output):
        metadata = candidate.diagnostic_metadata or {}
        original_rank = metadata.get("original_candidate_rank", candidate.rank)
        if type(original_rank) is int:
            rank_to_index[original_rank] = index

    refined_count = 0
    kept_original_count = 0
    collision_skip_count = 0
    failure_count = 0
    for record in sorted(records, key=lambda item: item.original_candidate_rank):
        slot_index = rank_to_index.get(record.original_candidate_rank)
        if slot_index is None:
            kept_original_count += 1
            continue
        current = output[slot_index]
        if record.status is not RefinementStatus.REFINED or record.refined_frame_id is None:
            kept_original_count += 1
            failure_count += int(record.failure_reason is not None)
            continue
        if current.video_id != record.video_id:
            raise AssertionError("Q3 anchor refinement changed the surviving slot video")
        refined_identity = (record.video_id, record.refined_frame_id)
        identities_outside_slot = {
            (candidate.video_id, candidate.frame_id)
            for index, candidate in enumerate(output)
            if index != slot_index
        }
        if refined_identity in identities_outside_slot:
            kept_original_count += 1
            collision_skip_count += 1
            continue
        output[slot_index] = replace(current, frame_id=record.refined_frame_id)
        refined_count += 1

    result = KISResult(query_id=baseline.query_id, ranked_candidates=tuple(output))
    if [item.rank for item in result.ranked_candidates] != [
        item.rank for item in baseline.ranked_candidates
    ]:
        raise AssertionError("Q3 anchor refinement changed the rank sequence")
    if [item.video_id for item in result.ranked_candidates] != [
        item.video_id for item in baseline.ranked_candidates
    ]:
        raise AssertionError("Q3 anchor refinement changed the video sequence")
    identities = [(item.video_id, item.frame_id) for item in result.ranked_candidates]
    if len(identities) != len(set(identities)):
        raise AssertionError("Q3 anchor refinement introduced a duplicate identity")
    return Q3AnchorIntegration(
        result=result,
        refined_count=refined_count,
        kept_original_count=kept_original_count,
        collision_skip_count=collision_skip_count,
        failure_count=failure_count,
    )
