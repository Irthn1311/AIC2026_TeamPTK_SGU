"""Exact DANTE-style strict monotonic dynamic programming reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DanteAlignment:
    score: float
    positions: tuple[int, ...]


def dante_monotonic_dp(scores: np.ndarray, distance_lambda: float) -> DanteAlignment | None:
    """Solve strict event-to-keyframe alignment in O(events * keyframes)."""

    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("scores must be a non-empty event-by-keyframe matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("DANTE scores must be finite")
    if distance_lambda < 0:
        raise ValueError("distance_lambda must be non-negative")
    event_count, position_count = matrix.shape
    if position_count < event_count:
        return None
    previous = matrix[0].astype(np.float64)
    pointers = np.full((event_count, position_count), -1, dtype=np.int64)
    for event_index in range(1, event_count):
        current = np.full(position_count, -np.inf, dtype=np.float64)
        best_value = -np.inf
        best_position = -1
        for position in range(1, position_count):
            predecessor = position - 1
            candidate = previous[predecessor] + distance_lambda * predecessor
            if candidate > best_value:
                best_value = candidate
                best_position = predecessor
            if best_position >= 0:
                current[position] = (
                    float(matrix[event_index, position]) - distance_lambda * position + best_value
                )
                pointers[event_index, position] = best_position
        previous = current
    final_position = int(np.argmax(previous))
    if not np.isfinite(previous[final_position]):
        return None
    positions = [final_position]
    for event_index in range(event_count - 1, 0, -1):
        predecessor = int(pointers[event_index, positions[-1]])
        if predecessor < 0:
            raise RuntimeError("DANTE backtracking pointer is invalid")
        positions.append(predecessor)
    positions.reverse()
    if any(left >= right for left, right in zip(positions[:-1], positions[1:], strict=True)):
        raise RuntimeError("DANTE backtracking violated strict temporal order")
    return DanteAlignment(float(previous[final_position]), tuple(positions))


__all__ = ["DanteAlignment", "dante_monotonic_dp"]
