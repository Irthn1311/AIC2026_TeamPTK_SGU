"""
Gemini Vision-Language Semantic Reranker.

Uses Gemini Flash to verify top-N candidates by sending keyframe images
along with the query text and asking for a relevance judgment.

This is the FINAL reranking stage — applied after all other rerankers
to catch false positives that CLIP cosine similarity cannot distinguish.

Requirements:
  - GEMINI_API_KEY environment variable set
  - Keyframe images available on disk (datasets/artifacts/keyframe_btc_full/)
  - Gracefully skips if either requirement is unmet

Performance notes:
  - Only processes top_n_verify candidates (default: 15) to control latency
  - Each Gemini call takes ~0.5-1s, total ~8-15s for verification
  - Caches results per keyframe to avoid redundant API calls
"""

from __future__ import annotations

import base64
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.reranking.base import BaseReranker
from src.common.types import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default number of top candidates to verify with Gemini
_DEFAULT_TOP_N_VERIFY = 15
# Blend ratio: how much weight to give Gemini score vs pipeline score
_GEMINI_WEIGHT = 0.45
_PIPELINE_WEIGHT = 0.55


class GeminiReranker(BaseReranker):
    """
    Reranks top candidates using Gemini Flash vision-language understanding.

    Args:
        keyframe_image_root: Root directory containing keyframe images
        top_n_verify: Number of top candidates to verify (default: 15)
        gemini_weight: Weight for Gemini score in final blend (default: 0.45)
        pipeline_weight: Weight for existing pipeline score (default: 0.55)
    """

    def __init__(
        self,
        keyframe_image_root: str = "",
        top_n_verify: int = _DEFAULT_TOP_N_VERIFY,
        gemini_weight: float = _GEMINI_WEIGHT,
        pipeline_weight: float = _PIPELINE_WEIGHT,
    ):
        self.keyframe_image_root = Path(keyframe_image_root) if keyframe_image_root else None
        self.top_n_verify = top_n_verify
        self.gemini_weight = gemini_weight
        self.pipeline_weight = pipeline_weight

        self._client = None
        self._init_lock = threading.Lock()
        self._cache: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return "gemini_reranker"

    def _ensure_client(self) -> bool:
        """Lazily initialize Gemini client. Returns True if ready."""
        if self._client is not None:
            return True

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return False

        try:
            import google.genai as genai
            with self._init_lock:
                if self._client is None:
                    self._client = genai.Client(api_key=api_key)
                    logger.info("[GeminiReranker] Initialized Gemini client")
            return True
        except Exception as e:
            logger.warning(f"[GeminiReranker] Failed to initialize: {e}")
            return False

    def _find_keyframe_image(self, video_id: str, n: int) -> Optional[Path]:
        """Find the keyframe image file on disk."""
        if not self.keyframe_image_root:
            return None

        # Try common path patterns
        candidates = [
            self.keyframe_image_root / video_id / f"{n:03d}.jpg",
            self.keyframe_image_root / video_id / f"{n}.jpg",
            self.keyframe_image_root / video_id / f"{n:04d}.jpg",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _score_candidate(
        self,
        image_path: Path,
        query_text: str,
    ) -> float:
        """
        Score a single candidate image against the query using Gemini.

        Returns a relevance score in [0.0, 1.0].
        """
        cache_key = f"{image_path}::{query_text[:100]}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            import google.genai as genai
            from google.genai import types

            prompt = (
                "You are a visual relevance judge. Rate how well this image matches "
                "the following description on a scale of 0 to 10.\n\n"
                f"Description: {query_text}\n\n"
                "Consider:\n"
                "- Do the objects, people, and scene match the description?\n"
                "- Are the colors, clothing, and spatial relationships correct?\n"
                "- Is this the specific scene/moment described?\n\n"
                "Respond with ONLY a single number from 0 to 10. Nothing else."
            )

            response = self._client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type="image/jpeg",
                            ),
                            types.Part.from_text(text=prompt),
                        ],
                    )
                ],
            )

            # Parse numeric score from response
            text = (response.text or "").strip()
            # Extract first number from response
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if numbers:
                raw_score = float(numbers[0])
                normalized = min(raw_score / 10.0, 1.0)
                self._cache[cache_key] = normalized
                return normalized

        except Exception as e:
            logger.debug(f"[GeminiReranker] Scoring failed for {image_path.name}: {e}")

        return -1.0  # Indicates failure — will be ignored in blending

    def rerank(
        self,
        query: Any,
        candidates: List[SearchResult],
        top_k: int = 100,
    ) -> List[SearchResult]:
        """
        Verify top candidates using Gemini vision-language model.

        Only the top `top_n_verify` candidates are sent to Gemini.
        Remaining candidates keep their original scores.
        """
        if not candidates:
            return []

        # Check if Gemini is available
        if not self._ensure_client():
            logger.debug("[GeminiReranker] Skipped — no GEMINI_API_KEY")
            return candidates[:top_k]

        # Check if images are available
        if not self.keyframe_image_root or not self.keyframe_image_root.exists():
            logger.debug("[GeminiReranker] Skipped — no keyframe images on disk")
            return candidates[:top_k]

        # Extract query text
        query_text = getattr(query, "raw_text", "") or getattr(query, "clip_prompt", "")
        if not query_text:
            return candidates[:top_k]

        # Split: top N for Gemini verification, rest pass through
        verify_candidates = candidates[:self.top_n_verify]
        pass_through = candidates[self.top_n_verify:]

        # Normalize pipeline scores for the verify set
        if verify_candidates:
            max_pipeline = max(c.score for c in verify_candidates)
            min_pipeline = min(c.score for c in verify_candidates)
            pipeline_range = max_pipeline - min_pipeline
        else:
            pipeline_range = 0

        reranked: List[SearchResult] = []
        n_verified = 0
        n_boosted = 0

        for cand in verify_candidates:
            # Find image
            image_path = self._find_keyframe_image(cand.video_id, cand.n)
            if image_path is None:
                reranked.append(cand)
                continue

            # Get Gemini score
            gemini_score = self._score_candidate(image_path, query_text)
            n_verified += 1

            if gemini_score < 0:
                # Gemini call failed — keep original score
                reranked.append(cand)
                continue

            # Normalize pipeline score to [0, 1]
            if pipeline_range > 1e-8:
                norm_pipeline = (cand.score - min_pipeline) / pipeline_range
            else:
                norm_pipeline = 1.0

            # Blend scores
            blended = (
                self.gemini_weight * gemini_score +
                self.pipeline_weight * norm_pipeline
            )

            if gemini_score > 0.6:
                n_boosted += 1

            reranked.append(SearchResult(
                keyframe_id=cand.keyframe_id,
                video_id=cand.video_id,
                n=cand.n,
                frame_idx=cand.frame_idx,
                pts_time=cand.pts_time,
                score=blended,
                retriever_source=f"{cand.retriever_source}+gemini",
                metadata={
                    **cand.metadata,
                    "gemini_score": round(gemini_score, 3),
                    "pre_gemini_score": cand.score,
                },
            ))

        # Sort verified candidates by new blended score
        reranked.sort(key=lambda x: x.score, reverse=True)

        # Append pass-through candidates (already sorted by original score)
        # Scale them down slightly so verified candidates always rank higher
        if reranked and pass_through:
            min_verified_score = reranked[-1].score if reranked else 0
            for pt in pass_through:
                scaled_score = min_verified_score * 0.95 * (pt.score / (pass_through[0].score or 1))
                reranked.append(SearchResult(
                    keyframe_id=pt.keyframe_id,
                    video_id=pt.video_id,
                    n=pt.n,
                    frame_idx=pt.frame_idx,
                    pts_time=pt.pts_time,
                    score=scaled_score,
                    retriever_source=pt.retriever_source,
                    metadata=pt.metadata,
                ))
        else:
            reranked.extend(pass_through)

        if n_verified > 0:
            logger.info(
                f"  • GeminiReranker: Verified {n_verified}/{len(verify_candidates)} "
                f"candidates | {n_boosted} high-confidence (>0.6)"
            )

        return reranked[:top_k]
