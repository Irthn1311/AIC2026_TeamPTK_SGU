from __future__ import annotations

import numpy as np


def temporal_score(candidate_frame: int, target_frame: int, window: int) -> float:
    if window <= 0:
        return 1.0 if candidate_frame == target_frame else 0.0
    return float(max(0.0, 1.0 - abs(candidate_frame - target_frame) / float(window)))


def representative_scores(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0,), dtype=np.float32)
    rep = embeddings.mean(axis=0, keepdims=True)
    rep_norm = np.linalg.norm(rep, axis=1, keepdims=True)
    rep_norm[rep_norm == 0] = 1.0
    rep = rep / rep_norm
    sims = embeddings @ rep.T
    return ((sims[:, 0] + 1.0) / 2.0).astype(np.float32)


def duplicate_penalty(embedding: np.ndarray, selected_embeddings: list[np.ndarray], soft_threshold: float) -> tuple[float, float, str]:
    if not selected_embeddings:
        return 0.0, 0.0, "unique"
    sims = [float(np.dot(embedding, prev)) for prev in selected_embeddings]
    max_sim = max(sims)
    penalty = max(0.0, (max_sim - soft_threshold) / max(1e-6, 1.0 - soft_threshold))
    status = "duplicate_soft" if penalty > 0 else "unique"
    return max_sim, float(min(1.0, penalty)), status


def final_score(row: dict, weights: dict) -> float:
    return float(
        float(weights["quality_weight"]) * row["quality_score"]
        + float(weights["representativeness_weight"]) * row["representative_score"]
        + float(weights["temporal_weight"]) * row["temporal_score"]
        - float(weights["duplicate_penalty_weight"]) * row["duplicate_penalty"]
    )
