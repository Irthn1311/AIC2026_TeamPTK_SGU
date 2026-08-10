"""Deterministic bounded k-best strict-monotonic temporal solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_BEAM_WIDTH = 5


@dataclass(frozen=True)
class TemporalPath:
    score: float
    positions: tuple[int, ...]


def _top_unique(paths: list[TemporalPath], k: int) -> list[TemporalPath]:
    unique: dict[tuple[int, ...], TemporalPath] = {}
    for path in paths:
        existing = unique.get(path.positions)
        if existing is None or path.score > existing.score:
            unique[path.positions] = path
    return sorted(unique.values(), key=lambda item: (-item.score, item.positions))[:k]


def k_best_monotonic_paths(scores: np.ndarray, k: int) -> tuple[TemporalPath, ...]:
    """Return up to ``k`` unique paths with strict order and deterministic ties.

    Each dynamic-programming state retains the best ``k`` prefixes. A running prefix pool
    avoids the quadratic predecessor scan while remaining exact for the bounded top-k result.
    """

    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("scores must be a non-empty event-by-position matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("T2 scores must be finite")
    if not 1 <= k <= MAX_BEAM_WIDTH:
        raise ValueError(f"T2 k must be between 1 and {MAX_BEAM_WIDTH}")
    event_count, position_count = matrix.shape
    if position_count < event_count:
        return ()

    previous = [
        [TemporalPath(float(matrix[0, position]), (position,))]
        for position in range(position_count)
    ]
    for event_index in range(1, event_count):
        current: list[list[TemporalPath]] = [[] for _ in range(position_count)]
        prefix_pool: list[TemporalPath] = []
        for position in range(position_count):
            predecessor = position - 1
            if predecessor >= 0:
                prefix_pool = _top_unique(prefix_pool + previous[predecessor], k)
            if prefix_pool:
                event_score = float(matrix[event_index, position])
                current[position] = [
                    TemporalPath(path.score + event_score, path.positions + (position,))
                    for path in prefix_pool
                ]
        previous = current

    finalists: list[TemporalPath] = []
    for bucket in previous:
        finalists = _top_unique(finalists + bucket, k)
    for path in finalists:
        if any(
            left >= right
            for left, right in zip(path.positions[:-1], path.positions[1:], strict=True)
        ):
            raise RuntimeError("T2 solver produced a non-monotonic path")
    return tuple(finalists)


__all__ = ["MAX_BEAM_WIDTH", "TemporalPath", "k_best_monotonic_paths"]
