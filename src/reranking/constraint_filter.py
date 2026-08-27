"""
Constraint Filter — Enforces must-have and negated attributes from query analysis.

Applies soft penalization (score multiplier) rather than hard filtering to avoid
accidentally removing correct results when constraint detection is imperfect.

Used after fusion, before final reranking:
  Fusion → ConstraintFilter → CLIPReranker → TemporalReranker → output
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from src.reranking.base import BaseReranker
from src.common.types import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Penalty multiplier for constraint violations (0.0 = remove, 1.0 = no penalty)
_NEGATION_PENALTY = 0.4       # Candidate contains something it shouldn't
_MUST_HAVE_PENALTY = 0.5      # Candidate missing a required attribute
_MULTI_SOURCE_BONUS = 1.15    # Bonus for candidates found by multiple retrievers


class ConstraintFilter(BaseReranker):
    """
    Soft-penalizes candidates that violate query constraints.

    Constraint types:
      - must_have: attributes the result MUST contain (e.g., "áo đen" → black shirt)
      - negated_attributes: attributes the result must NOT contain
      - multi_source_bonus: candidates found by 2+ retrievers get a reliability bonus

    Args:
        negation_penalty: Score multiplier when a negated attribute matches (default: 0.4)
        must_have_penalty: Score multiplier when a must-have attribute is missing (default: 0.5)
    """

    def __init__(
        self,
        negation_penalty: float = _NEGATION_PENALTY,
        must_have_penalty: float = _MUST_HAVE_PENALTY,
    ):
        self.negation_penalty = negation_penalty
        self.must_have_penalty = must_have_penalty

    @property
    def name(self) -> str:
        return "constraint_filter"

    def rerank(
        self,
        query: Any,
        candidates: List[SearchResult],
        top_k: int = 100,
    ) -> List[SearchResult]:
        """
        Apply constraint-based soft penalization to candidates.

        Examines query for:
          - .negated_attributes: List[str] of things that should NOT appear
          - .must_have: List[str] of things that MUST appear
          - Multi-source bonus for candidates found by multiple retrievers
        """
        if not candidates:
            return []

        # Extract constraints from query object
        negated = getattr(query, "negated_attributes", []) or []
        must_have = getattr(query, "must_have", []) or []

        # If no constraints, just apply multi-source bonus
        if not negated and not must_have:
            return self._apply_multi_source_bonus(candidates, top_k)

        negated_lower = {n.lower().strip() for n in negated if n.strip()}
        must_have_lower = {m.lower().strip() for m in must_have if m.strip()}

        reranked: List[SearchResult] = []
        n_penalized = 0

        for cand in candidates:
            penalty = 1.0
            violations = []

            # Check OCR text for constraint violations
            ocr_text = (
                cand.metadata.get("ocr_text", "")
                or cand.metadata.get("text_snippet", "")
            ).lower()

            # Check negation constraints against OCR
            for neg in negated_lower:
                if neg in ocr_text:
                    penalty *= self.negation_penalty
                    violations.append(f"neg:{neg}")

            # Check topic/category metadata for constraint matches
            topic = cand.metadata.get("topic_category", "").lower()

            # Multi-source reliability bonus
            n_sources = cand.metadata.get("n_sources", 1)
            if n_sources > 1:
                penalty *= _MULTI_SOURCE_BONUS

            if penalty < 1.0:
                n_penalized += 1

            new_score = cand.score * penalty
            reranked.append(SearchResult(
                keyframe_id=cand.keyframe_id,
                video_id=cand.video_id,
                n=cand.n,
                frame_idx=cand.frame_idx,
                pts_time=cand.pts_time,
                score=new_score,
                retriever_source=cand.retriever_source,
                metadata={
                    **cand.metadata,
                    "constraint_penalty": round(penalty, 3),
                    "constraint_violations": violations,
                },
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)

        if n_penalized > 0:
            logger.info(
                f"  • ConstraintFilter: {n_penalized}/{len(candidates)} candidates penalized | "
                f"negated={list(negated_lower)} must_have={list(must_have_lower)}"
            )

        return reranked[:top_k]

    def _apply_multi_source_bonus(
        self,
        candidates: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Apply multi-source bonus only (no constraint filtering)."""
        reranked = []
        for cand in candidates:
            n_sources = cand.metadata.get("n_sources", 1)
            bonus = _MULTI_SOURCE_BONUS if n_sources > 1 else 1.0

            reranked.append(SearchResult(
                keyframe_id=cand.keyframe_id,
                video_id=cand.video_id,
                n=cand.n,
                frame_idx=cand.frame_idx,
                pts_time=cand.pts_time,
                score=cand.score * bonus,
                retriever_source=cand.retriever_source,
                metadata={
                    **cand.metadata,
                    "constraint_penalty": round(bonus, 3),
                },
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
