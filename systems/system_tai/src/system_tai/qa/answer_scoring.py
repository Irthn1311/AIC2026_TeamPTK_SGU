from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .models import AnswerHypothesis, QAEvidenceCandidate


@runtime_checkable
class EvidenceAnswerScorer(Protocol):
    def score_answers(
        self,
        candidate: QAEvidenceCandidate,
        hypotheses: Sequence[AnswerHypothesis],
        image_embedding: np.ndarray | None = None,
        prompt_embeddings: dict[str, np.ndarray] | None = None,
    ) -> list[tuple[AnswerHypothesis, float]]:
        ...


class CosineEvidenceAnswerScorer:
    """Deterministic cosine similarity answer scorer for normalized embeddings.

    Prompt Aggregation Policy:
        For each AnswerHypothesis, the score against an image embedding is computed
        as the MAXIMUM cosine similarity over all prompt embeddings associated with
        that hypothesis's visual_prompts.

    Deterministic Tie-Breaking Policy:
        Results are sorted by score descending, then by canonical_answer ascending.
    """

    def __init__(self, default_score: float = 0.0) -> None:
        self.default_score = float(default_score)

    def _validate_normalized_embedding(self, vec: np.ndarray, name: str) -> None:
        if not isinstance(vec, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray")
        if vec.dtype != np.float32:
            raise TypeError(f"{name} must have dtype float32, got {vec.dtype}")
        if vec.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional, got shape {vec.shape}")
        if vec.size == 0:
            raise ValueError(f"{name} cannot be empty")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")

        norm = float(np.linalg.norm(vec))
        if norm <= 1e-7:
            raise ValueError(f"{name} has zero or near-zero norm")
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(
                f"{name} is not L2-normalized: norm is {norm:.6f}, expected ~1.0"
            )

    def score_answers(
        self,
        candidate: QAEvidenceCandidate,
        hypotheses: Sequence[AnswerHypothesis],
        image_embedding: np.ndarray | None = None,
        prompt_embeddings: dict[str, np.ndarray] | None = None,
    ) -> list[tuple[AnswerHypothesis, float]]:
        if not hypotheses:
            return []

        if image_embedding is not None:
            self._validate_normalized_embedding(image_embedding, "image_embedding")

        scored_results: list[tuple[AnswerHypothesis, float]] = []

        for hyp in hypotheses:
            max_sim = self.default_score
            if image_embedding is not None and prompt_embeddings is not None:
                for prompt in hyp.visual_prompts:
                    if prompt in prompt_embeddings:
                        p_emb = prompt_embeddings[prompt]
                        self._validate_normalized_embedding(
                            p_emb, f"prompt_embedding['{prompt}']"
                        )
                        if p_emb.shape != image_embedding.shape:
                            msg = (
                                f"Dimension mismatch: image shape {image_embedding.shape} "
                                f"vs prompt shape {p_emb.shape}"
                            )
                            raise ValueError(msg)
                        sim = float(np.dot(image_embedding, p_emb))
                        if not np.isfinite(sim):
                            sim = self.default_score
                        if sim > max_sim or max_sim == self.default_score:
                            max_sim = sim
            scored_results.append((hyp, max_sim))

        scored_results.sort(key=lambda x: (-x[1], x[0].canonical_answer))
        return scored_results
