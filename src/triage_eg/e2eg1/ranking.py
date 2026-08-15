"""Deterministic metric-aware allocation over the frozen E2E-1 candidate pool."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .contracts import E2EG1Settings


def candidate_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["video_id"]), int(row["original_frame_idx"])


def canonical_coarse_candidates(
    pool: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Match E2E-1 P0 tuple deduplication while retaining raw global ranks."""

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw_rank, source in enumerate(pool, 1):
        key = candidate_key(source)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                **source,
                "original_global_rank": raw_rank,
                "g0_rank": len(output) + 1,
                "hypothesis_kind": "COARSE",
            }
        )
    return output


def rank_video_hypotheses(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank videos by strongest candidate evidence, with deterministic ties."""

    strongest: dict[str, float] = {}
    for row in candidates:
        video_id = str(row["video_id"])
        score = float(row["score"])
        strongest[video_id] = max(strongest.get(video_id, float("-inf")), score)
    return [
        {"video_id": video_id, "video_score": score, "video_hypothesis_rank": rank}
        for rank, (video_id, score) in enumerate(
            sorted(strongest.items(), key=lambda item: (-item[1], item[0])), 1
        )
    ]


def _allocation_metadata(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[tuple[str, int], int], list[dict[str, Any]]]:
    video_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        video_rows[str(row["video_id"])].append(row)
    video_ranking = rank_video_hypotheses(candidates)
    video_rank = {row["video_id"]: int(row["video_hypothesis_rank"]) for row in video_ranking}
    region_rank = {
        candidate_key(row): rank
        for video_id in sorted(video_rows)
        for rank, row in enumerate(video_rows[video_id], 1)
    }
    return video_rank, region_rank, video_ranking


def g0_order(
    pool: list[dict[str, Any]] | tuple[dict[str, Any], ...], settings: E2EG1Settings
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = canonical_coarse_candidates(pool)
    video_rank, region_rank, video_ranking = _allocation_metadata(candidates)
    output = []
    for rank, source in enumerate(candidates[: settings.max_predictions], 1):
        output.append(
            {
                **source,
                "coverage_rank": rank,
                "video_hypothesis_rank": video_rank[str(source["video_id"])],
                "within_video_region_rank": region_rank[candidate_key(source)],
                "was_in_protected_prefix": rank <= settings.protected_global_prefix,
                "was_in_coverage_block": False,
                "was_selected_for_m1": False,
            }
        )
    return output, video_ranking


def coverage_order(
    pool: list[dict[str, Any]] | tuple[dict[str, Any], ...], settings: E2EG1Settings
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Protect Top-5, round-robin full T3 regions for top videos, then append global tail."""

    candidates = canonical_coarse_candidates(pool)
    video_rank, region_rank, video_ranking = _allocation_metadata(candidates)
    per_video: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        per_video[str(row["video_id"])].append(row)
    top_videos = [row["video_id"] for row in video_ranking[: settings.coverage_video_limit]]
    emitted: set[tuple[str, int]] = set()
    staged: list[tuple[dict[str, Any], bool, bool]] = []

    for row in candidates[: settings.protected_global_prefix]:
        key = candidate_key(row)
        if key not in emitted:
            emitted.add(key)
            staged.append((row, True, False))
    for region_index in range(settings.coverage_regions_per_video):
        for video_id in top_videos:
            rows = per_video[video_id]
            if region_index >= len(rows):
                continue
            row = rows[region_index]
            key = candidate_key(row)
            if key not in emitted:
                emitted.add(key)
                staged.append((row, False, True))
    for row in candidates:
        key = candidate_key(row)
        if key not in emitted:
            emitted.add(key)
            staged.append((row, False, False))

    output = []
    for rank, (source, protected, coverage) in enumerate(staged[: settings.max_predictions], 1):
        output.append(
            {
                **source,
                "coverage_rank": rank,
                "video_hypothesis_rank": video_rank[str(source["video_id"])],
                "within_video_region_rank": region_rank[candidate_key(source)],
                "was_in_protected_prefix": protected,
                "was_in_coverage_block": coverage,
                "was_selected_for_m1": False,
            }
        )
    expected = [candidate_key(row) for row in candidates[: settings.protected_global_prefix]]
    actual = [candidate_key(row) for row in output[: settings.protected_global_prefix]]
    if actual != expected:
        raise RuntimeError("E2EG1_PROTECTED_GLOBAL_PREFIX_VIOLATION")
    return output, video_ranking


def safe_alternative_order(
    coverage: list[dict[str, Any]],
    alternatives: dict[tuple[str, int], dict[str, Any]],
    settings: E2EG1Settings,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep coarse sources and add only non-colliding M1 alternatives."""

    protected = coverage[: settings.protected_global_prefix]
    coarse_keys = {candidate_key(row) for row in coverage}
    emitted: set[tuple[str, int]] = set()
    output: list[dict[str, Any]] = []
    duplicate_alternatives = 0

    def emit(row: dict[str, Any]) -> None:
        key = str(row["video_id"]), int(row["frame_id"])
        if key not in emitted and len(output) < settings.max_predictions:
            emitted.add(key)
            output.append(row)

    def emit_alternative(source: dict[str, Any]) -> None:
        nonlocal duplicate_alternatives
        alternative = alternatives.get(candidate_key(source))
        if alternative is None:
            return
        key = str(alternative["video_id"]), int(alternative["frame_id"])
        if key in coarse_keys or key in emitted:
            duplicate_alternatives += 1
            return
        emit(alternative)

    for row in protected:
        emit(row)
    for row in protected:
        emit_alternative(row)
    for row in coverage[settings.protected_global_prefix :]:
        emit(row)
        emit_alternative(row)
        if len(output) == settings.max_predictions:
            break
    protected_keys = [candidate_key(row) for row in protected]
    actual = [(str(row["video_id"]), int(row["frame_id"])) for row in output[: len(protected)]]
    if actual != protected_keys or any(
        row["hypothesis_kind"] != "COARSE" for row in output[: len(protected)]
    ):
        raise RuntimeError("E2EG1_SAFE_M1_PROTECTED_PREFIX_VIOLATION")
    return output, {"refined_duplicate_dropped_count": duplicate_alternatives}


__all__ = [
    "candidate_key",
    "canonical_coarse_candidates",
    "coverage_order",
    "g0_order",
    "rank_video_hypotheses",
    "safe_alternative_order",
]
