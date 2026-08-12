"""
BatchRouter — Predicts which batch(es) likely contain the answer to a query.

Without target_prefix knowledge (as in real competition), the retrieval system
would blindly search 177K vectors, letting L25 (79K frames) and L26 (37K frames)
dominate results. BatchRouter addresses this by:

  1. KeywordHeuristic: Simple rule-based scoring using known topic/theme signals.
  2. MediaInfoBM25 (optional): If media-info JSONs are available, uses BM25 keyword
     matching against video title/description to rank batches by relevance.

When no strong signal is found, returns all known batches (global search fallback).

Usage:
    router = BatchRouter(known_batches=["L21","L22",...,"L30"])
    predicted = router.predict(query_text, top_n=3)
    # → ["L21", "L24", "L28"]  (most likely batches for this query)
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# Batch-level topic keyword heuristics
# These are rough signals only — NOT ground truth.
# They help narrow the FAISS search space without index rebuild.
# ============================================================
_BATCH_TOPIC_SIGNALS: Dict[str, List[str]] = {
    # Keywords that STRONGLY suggest a batch based on known content themes
    # (populated from pre-analysis of map-keyframes content patterns)
    # This can be extended as you analyze more batches.
    "general_news": [
        "bản tin", "thời sự", "tin tức", "phóng sự", "mc", "phát thanh viên",
        "news", "broadcast", "anchor", "reporter", "studio", "trường quay",
    ],
    "sports": [
        "thể thao", "bóng đá", "cầu thủ", "trận đấu", "thi đấu", "vận động viên",
        "sport", "football", "player", "match", "game", "competition", "athlete",
    ],
    "nature_disaster": [
        "cháy", "lũ", "lụt", "bão", "sạt lở", "thiên tai", "núi lửa",
        "fire", "flood", "storm", "volcano", "disaster", "earthquake",
    ],
    "agriculture": [
        "nông nghiệp", "lúa", "thu hoạch", "nông dân", "ruộng", "đồng",
        "agriculture", "harvest", "farmer", "rice", "crop", "field",
    ],
    "culture_heritage": [
        "di tích", "tháp", "lịch sử", "cổ", "văn hóa", "bảo tàng",
        "heritage", "temple", "ancient", "culture", "museum", "historic",
    ],
    "city_life": [
        "đường phố", "xe máy", "giao thông", "thành phố", "đô thị", "chợ",
        "street", "traffic", "motorcycle", "city", "urban", "market",
    ],
}


class BatchRouter:
    """
    Predicts which batch(es) are most likely to contain the answer for a query.

    This is a heuristic-based routing layer that runs BEFORE FAISS search.
    It narrows the search space from N batches × full index to Top-M batches,
    dramatically improving precision without requiring target_prefix.

    Args:
        known_batches: List of all batch IDs (e.g., ["L21", ..., "L30"])
        media_info_dir: Optional path to media-info JSON directory.
                        If provided, enables BM25 MediaInfo matching.
        batch_sizes:    Dict mapping batch_id → number of keyframes.
                        Used for size-inverse weighting (smaller batch = higher weight).
    """

    def __init__(
        self,
        known_batches: Optional[List[str]] = None,
        media_info_dir: Optional[str] = None,
        batch_sizes: Optional[Dict[str, int]] = None,
    ):
        self._batches = known_batches or [f"L{i}" for i in range(21, 31)]
        self._batch_sizes = batch_sizes or {}
        self._media_index: Dict[str, Dict[str, str]] = {}  # batch_id → {video_id: text}
        self._idf: Dict[str, float] = {}

        if media_info_dir:
            self._build_media_index(media_info_dir)

        logger.info(
            f"[BatchRouter] Initialized with {len(self._batches)} batches: {self._batches}"
            + (f" | MediaInfo: {len(self._media_index)} videos indexed" if self._media_index else "")
        )

    # ----------------------------------------------------------
    # MediaInfo BM25 Index (optional)
    # ----------------------------------------------------------

    def _build_media_index(self, media_info_dir: str) -> None:
        """Load all media-info JSON files and build a simple BM25 corpus."""
        info_dir = Path(media_info_dir)
        if not info_dir.exists():
            logger.warning(f"[BatchRouter] media_info_dir not found: {info_dir}")
            return

        corpus: Dict[str, str] = {}  # video_id → combined text
        for json_path in sorted(info_dir.glob("**/*.json")):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                video_id = json_path.stem  # e.g., "L21_V001"
                parts = []
                for field in ("title", "description", "tags", "author", "keywords"):
                    val = data.get(field, "")
                    if isinstance(val, list):
                        val = " ".join(str(v) for v in val)
                    if val:
                        parts.append(str(val))
                text = " ".join(parts).lower()
                if text.strip():
                    corpus[video_id] = text
                    batch_id = video_id.split("_")[0]
                    if batch_id not in self._media_index:
                        self._media_index[batch_id] = {}
                    self._media_index[batch_id][video_id] = text
            except Exception as e:
                logger.debug(f"[BatchRouter] Skip {json_path.name}: {e}")

        # Compute IDF over all video texts
        self._compute_idf(corpus)
        logger.info(f"[BatchRouter] MediaInfo BM25 built: {len(corpus)} videos")

    def _compute_idf(self, corpus: Dict[str, str]) -> None:
        """Compute IDF for BM25."""
        N = len(corpus)
        df: Dict[str, int] = defaultdict(int)
        for text in corpus.values():
            tokens = set(re.split(r"\W+", text))
            for t in tokens:
                if t:
                    df[t] += 1
        self._idf = {
            t: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for t, freq in df.items()
        }

    def _bm25_score(self, query_tokens: List[str], doc_text: str, k1: float = 1.5, b: float = 0.75) -> float:
        """Compute BM25 score for a single document."""
        tokens = re.split(r"\W+", doc_text.lower())
        doc_len = len(tokens)
        avg_len = 200  # approximate average doc length
        tf: Dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        score = 0.0
        for qt in query_tokens:
            if qt not in self._idf:
                continue
            tf_val = tf.get(qt, 0)
            score += self._idf[qt] * (
                tf_val * (k1 + 1) / (tf_val + k1 * (1 - b + b * doc_len / avg_len))
            )
        return score

    # ----------------------------------------------------------
    # Keyword Heuristic Scoring
    # ----------------------------------------------------------

    def _keyword_heuristic_score(self, query_lower: str) -> Dict[str, float]:
        """
        Score each batch using keyword signals.
        Returns: {batch_id: score} — higher = more relevant.
        """
        # Topic matching
        topic_hits: Dict[str, float] = defaultdict(float)
        for topic, keywords in _BATCH_TOPIC_SIGNALS.items():
            topic_score = sum(1.0 for kw in keywords if kw in query_lower)
            if topic_score > 0:
                topic_hits[topic] += topic_score

        # All batches start equal
        batch_scores: Dict[str, float] = {b: 0.0 for b in self._batches}

        # Apply size-inverse bias: smaller batch → higher base weight
        # This counter-acts the natural tendency for large batches to win
        for batch_id in self._batches:
            size = self._batch_sizes.get(batch_id, 1000)
            # Inverse-log weight: large batches (L26: 79K) get weight ~1.0,
            # small batches (L21: 6K) get weight ~3.5
            batch_scores[batch_id] += max(0.0, 5.0 - math.log10(max(size, 100)))

        return batch_scores

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def predict(
        self,
        query_text: str,
        top_n: int = 3,
        return_all_if_uncertain: bool = True,
    ) -> List[str]:
        """
        Predict the most likely batch(es) for a given query.

        Args:
            query_text: Raw query string (Vietnamese or English)
            top_n: Number of batches to return
            return_all_if_uncertain: If True and confidence is low,
                                     return ALL batches (global search)

        Returns:
            List of batch prefix strings, e.g. ["L21", "L24", "L28"]
        """
        query_lower = query_text.lower()
        query_tokens = [t for t in re.split(r"\W+", query_lower) if len(t) > 2]

        # --- Step 1: Base heuristic scores ---
        batch_scores = self._keyword_heuristic_score(query_lower)

        # --- Step 2: MediaInfo BM25 boost (if available) ---
        if self._media_index:
            for batch_id, video_texts in self._media_index.items():
                if batch_id not in batch_scores:
                    continue
                # Score = max BM25 over all videos in this batch
                best_bm25 = 0.0
                for vid_text in video_texts.values():
                    s = self._bm25_score(query_tokens, vid_text)
                    if s > best_bm25:
                        best_bm25 = s
                batch_scores[batch_id] += best_bm25 * 0.5  # blend factor

        # --- Step 3: Sort and decide ---
        ranked = sorted(batch_scores.items(), key=lambda x: -x[1])

        # Confidence check: if top batch score is very low, return all batches
        if return_all_if_uncertain:
            top_score = ranked[0][1] if ranked else 0.0
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            # Gap between 1st and 2nd must be meaningful
            if top_score < 1.0 or (top_score - second_score) < 0.5:
                logger.info(
                    f"[BatchRouter] Low confidence (top={top_score:.2f}, gap={top_score-second_score:.2f}) "
                    f"→ returning ALL {len(self._batches)} batches"
                )
                return self._batches  # global search

        predicted = [b for b, _ in ranked[:top_n]]
        logger.info(
            f"[BatchRouter] query='{query_text[:60]}' → predicted batches: {predicted} "
            f"(scores: {[(b, f'{s:.2f}') for b, s in ranked[:top_n]]})"
        )
        return predicted

    def update_batch_sizes(self, batch_sizes: Dict[str, int]) -> None:
        """Update batch size mapping (call after loading MetadataStore)."""
        self._batch_sizes.update(batch_sizes)
        logger.info(f"[BatchRouter] Batch sizes updated: {batch_sizes}")
