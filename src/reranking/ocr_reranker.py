"""
OCR & Text Keywords Relevance Reranker.

Boosts candidates whose extracted OCR text or metadata contains exact string matches
for explicit OCR/Logo/Text keywords found in the query.
"""

from __future__ import annotations

import re
from typing import Any, List, Set

from src.reranking.base import BaseReranker
from src.common.types import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_text_keywords(query: Any) -> List[str]:
    """Extract explicit text/OCR keywords from query object or string."""
    if hasattr(query, "ocr_keywords") and getattr(query, "ocr_keywords"):
        return [kw.lower() for kw in getattr(query, "ocr_keywords") if len(kw.strip()) >= 2]
    
    query_str = getattr(query, "raw_text", str(query))
    # Extract quoted text or ALL CAPS words (station names like VTV1, HTV7)
    quoted = re.findall(r'["\']([\w\s]{2,})["\']', query_str)
    caps = re.findall(r'\b[A-Z][A-Z0-9]{1,}\b', query_str)
    
    keywords = [k.lower() for k in quoted + caps if k not in ("TV", "HD", "OK", "AI")]
    return list(dict.fromkeys(keywords))


class OCRRelevanceReranker(BaseReranker):
    """
    Reranks candidates by boosting items with exact OCR keyword matches.

    Args:
        ocr_match_boost: Score multiplier when an OCR keyword is matched (default: 0.35)
    """

    def __init__(self, ocr_match_boost: float = 0.35):
        self.ocr_match_boost = ocr_match_boost

    @property
    def name(self) -> str:
        return "ocr_reranker"

    def rerank(
        self,
        query: Any,
        candidates: List[SearchResult],
        top_k: int = 50,
    ) -> List[SearchResult]:
        if not candidates:
            return []

        keywords = _extract_text_keywords(query)
        if not keywords:
            return candidates[:top_k]

        reranked: List[SearchResult] = []
        for cand in candidates:
            ocr_text = (cand.metadata.get("ocr_text", "") or cand.metadata.get("text_snippet", "")).lower()
            
            matches = [kw for kw in keywords if kw in ocr_text]
            boost_factor = 1.0 + (len(matches) * self.ocr_match_boost)
            boosted_score = cand.score * boost_factor

            new_cand = SearchResult(
                keyframe_id=cand.keyframe_id,
                video_id=cand.video_id,
                n=cand.n,
                frame_idx=cand.frame_idx,
                pts_time=cand.pts_time,
                score=boosted_score,
                retriever_source=cand.retriever_source,
                metadata={
                    **cand.metadata,
                    "ocr_matches": matches,
                    "ocr_boost": round(boost_factor, 2),
                },
            )
            reranked.append(new_cand)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
