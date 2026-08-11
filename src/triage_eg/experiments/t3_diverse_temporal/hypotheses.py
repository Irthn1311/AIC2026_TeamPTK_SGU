"""Ground-truth-free candidate pools and temporal hypothesis selection for T3."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from triage_eg.experiments.t2d_ceiling import stable_event_ranking

POOL_LIMIT = 10
REGION_RADIUS_SECONDS = 3.0
FINAL_PATH_LIMIT = 5
MAX_RAW_COMBINATIONS = POOL_LIMIT**4


@dataclass(frozen=True)
class EventCandidate:
    event_id: str
    event_region_id: str
    catalog_position: int
    original_frame_idx: int
    similarity: float


@dataclass(frozen=True)
class DiverseTemporalPath:
    score: float
    positions: tuple[int, ...]
    region_ids: tuple[str, ...]
    event_scores: tuple[float, ...]


def build_diverse_event_pool(
    event_id: str,
    scores: np.ndarray,
    original_frames: np.ndarray,
    fps: float,
    *,
    limit: int = POOL_LIMIT,
    radius_seconds: float = REGION_RADIUS_SECONDS,
) -> tuple[EventCandidate, ...]:
    """Greedily retain high-score anchors separated by more than the region radius."""

    values = np.asarray(scores, dtype=np.float32)
    frames = np.asarray(original_frames, dtype=np.int64)
    if values.ndim != 1 or frames.shape != values.shape or fps <= 0:
        raise ValueError("T3 event-pool inputs are invalid")
    if limit != POOL_LIMIT or radius_seconds != REGION_RADIUS_SECONDS:
        raise ValueError("T3 freezes pool limit=10 and temporal region radius=3 seconds")
    ranking = stable_event_ranking(values)
    retained: list[EventCandidate] = []
    for position_value in ranking:
        position = int(position_value)
        frame = int(frames[position])
        if all(
            abs(frame - candidate.original_frame_idx) / fps > radius_seconds
            for candidate in retained
        ):
            region_number = len(retained) + 1
            retained.append(
                EventCandidate(
                    event_id=event_id,
                    event_region_id=f"{event_id}:R{region_number:02d}:P{position:06d}",
                    catalog_position=position,
                    original_frame_idx=frame,
                    similarity=float(values[position]),
                )
            )
            if len(retained) == limit:
                break
    if not retained:
        raise RuntimeError(f"T3 event pool is empty: {event_id}")
    return tuple(retained)


def enumerate_feasible_paths(
    pools: tuple[tuple[EventCandidate, ...], ...],
) -> tuple[tuple[DiverseTemporalPath, ...], int]:
    """Enumerate every pool combination and retain exact strict-monotonic paths."""

    if not 2 <= len(pools) <= 4 or any(not pool or len(pool) > POOL_LIMIT for pool in pools):
        raise ValueError("T3 requires 2-4 non-empty event pools of at most ten candidates")
    raw_combination_count = int(np.prod([len(pool) for pool in pools], dtype=np.int64))
    if raw_combination_count > MAX_RAW_COMBINATIONS:
        raise RuntimeError("T3 raw candidate combinations exceeded the bounded maximum")
    unique: dict[tuple[int, ...], DiverseTemporalPath] = {}
    for combination in itertools.product(*pools):
        positions = tuple(candidate.catalog_position for candidate in combination)
        if any(left >= right for left, right in zip(positions[:-1], positions[1:], strict=True)):
            continue
        event_scores = tuple(candidate.similarity for candidate in combination)
        path = DiverseTemporalPath(
            score=float(sum(event_scores)),
            positions=positions,
            region_ids=tuple(candidate.event_region_id for candidate in combination),
            event_scores=event_scores,
        )
        existing = unique.get(positions)
        if existing is None or path.score > existing.score:
            unique[positions] = path
    paths = tuple(sorted(unique.values(), key=lambda item: (-item.score, item.positions)))
    return paths, raw_combination_count


def select_score_top_k(
    feasible_paths: tuple[DiverseTemporalPath, ...], k: int = FINAL_PATH_LIMIT
) -> tuple[DiverseTemporalPath, ...]:
    if k != FINAL_PATH_LIMIT:
        raise ValueError("T3 final hypothesis count is frozen at five")
    return tuple(sorted(feasible_paths, key=lambda item: (-item.score, item.positions))[:k])


def event_region_novelty(
    candidate: DiverseTemporalPath, selected: tuple[DiverseTemporalPath, ...]
) -> int:
    if not selected:
        return len(candidate.region_ids)
    if any(len(path.region_ids) != len(candidate.region_ids) for path in selected):
        raise ValueError("T3 novelty paths must have aligned event positions")
    return sum(
        candidate.region_ids[event_index]
        not in {path.region_ids[event_index] for path in selected}
        for event_index in range(len(candidate.region_ids))
    )


def relative_score_gap(best_score: float, path_score: float) -> float:
    return (best_score - path_score) / max(abs(best_score), 1e-12)


def select_coverage_aware(
    feasible_paths: tuple[DiverseTemporalPath, ...],
    delta: float,
    k: int = FINAL_PATH_LIMIT,
) -> tuple[DiverseTemporalPath, ...]:
    """Prefer per-event region novelty among competitive paths, then score-fill to K."""

    if delta not in {0.01, 0.03, 0.05} or k != FINAL_PATH_LIMIT:
        raise ValueError("T3 freezes delta grid=(0.01,0.03,0.05) and final K=5")
    ordered = tuple(sorted(feasible_paths, key=lambda item: (-item.score, item.positions)))
    if not ordered:
        return ()
    selected: list[DiverseTemporalPath] = [ordered[0]]
    selected_positions = {ordered[0].positions}
    while len(selected) < k:
        eligible = [
            path
            for path in ordered
            if path.positions not in selected_positions
            and relative_score_gap(ordered[0].score, path.score) <= delta
        ]
        if not eligible:
            break
        winner = min(
            eligible,
            key=lambda path: (
                -event_region_novelty(path, tuple(selected)),
                -path.score,
                path.positions,
            ),
        )
        selected.append(winner)
        selected_positions.add(winner.positions)
    if len(selected) < k:
        for path in ordered:
            if path.positions not in selected_positions:
                selected.append(path)
                selected_positions.add(path.positions)
                if len(selected) == k:
                    break
    output = tuple(selected)
    if len({path.positions for path in output}) != len(output):
        raise RuntimeError("T3 selector returned duplicate paths")
    return output


__all__ = [
    "DiverseTemporalPath",
    "EventCandidate",
    "FINAL_PATH_LIMIT",
    "MAX_RAW_COMBINATIONS",
    "POOL_LIMIT",
    "REGION_RADIUS_SECONDS",
    "build_diverse_event_pool",
    "enumerate_feasible_paths",
    "event_region_novelty",
    "relative_score_gap",
    "select_coverage_aware",
    "select_score_top_k",
]
