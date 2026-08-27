"""
In-Memory OCR Retriever — Fast local text search over extracted OCR JSON/JSONL files.

Supports searching OCR text without requiring a running Qdrant vector database.
Ideal for Kaggle notebooks and offline local execution.

Input: Directory containing:
  - L{XX}_{V}.json  — legacy format (single JSON object with "keyframes" list)
  - L{XX}_{V}.jsonl — v2 format (one JSON object per line, each = one keyframe record)
Output: Ranked list of SearchResult objects.

Fix (Trụ 4A): Now loads both *.json AND *.jsonl files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.retrieval.base import BaseRetriever
from src.common.types import SearchResult, TextualKISQuery
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _tokenize(text: str) -> List[str]:
    """Extract alphanumeric words (lowercase)."""
    return re.findall(r"\w+", text.lower())


class InMemoryOCRRetriever(BaseRetriever):
    """
    Retrieves keyframes by matching query keywords against extracted OCR JSON files.

    Args:
        ocr_dir: Directory containing per-video OCR JSON files (e.g., datasets/ocr)
        meta_store: Optional MetadataStore for accurate frame_idx/pts_time lookups

    Usage:
        retriever = InMemoryOCRRetriever(ocr_dir="datasets/ocr")
        retriever.load()
        results = retriever.retrieve("VTV1 bản tin", top_k=50)
    """

    def __init__(
        self,
        ocr_dir: str | Path,
        meta_store: Optional[Any] = None,
    ):
        self.ocr_dir = Path(ocr_dir)
        self.meta_store = meta_store
        # Storage: keyframe_id -> Dict record
        self._records: Dict[str, Dict[str, Any]] = {}
        self._is_loaded: bool = False

    @property
    def name(self) -> str:
        return "ocr_inmemory"

    @property
    def is_configured(self) -> bool:
        return self._is_loaded and len(self._records) > 0

    def load(self) -> "InMemoryOCRRetriever":
        """Load all per-video JSON and JSONL files in ocr_dir into memory.

        Supports two file formats:
          - *.json  — legacy: {"video_id": ..., "keyframes": [{"n": ..., "texts": [...]}]}
          - *.jsonl — v2: one record per line, each line = {"video_id": ..., "n": ..., "texts": [...]}
                     or: {"video_id": ..., "n": ..., "ocr_text": "..."}  (flat format)
        """
        if not self.ocr_dir.exists():
            logger.warning(f"[InMemoryOCRRetriever] Directory not found: {self.ocr_dir}")
            return self

        json_files  = sorted(self.ocr_dir.glob("*.json"))
        jsonl_files = sorted(self.ocr_dir.glob("*.jsonl"))
        total_files = len(json_files) + len(jsonl_files)
        logger.info(
            f"[InMemoryOCRRetriever] Loading OCR data from {self.ocr_dir} | "
            f"{len(json_files)} .json + {len(jsonl_files)} .jsonl files"
        )

        total_with_text = 0

        # ── Format A: legacy .json files ─────────────────────────────────────
        for json_path in json_files:
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)

                video_id = data.get("video_id", json_path.stem)
                keyframes = data.get("keyframes", [])

                for kf in keyframes:
                    n = int(kf.get("n", 0))
                    texts = kf.get("texts", [])
                    raw_text_str = " ".join(texts) if isinstance(texts, list) else str(texts)
                    clean_text = _normalize_text(raw_text_str)

                    if not clean_text:
                        continue

                    keyframe_id = f"{video_id}_n{n}"
                    frame_idx = int(kf.get("frame_idx", 0))
                    pts_time = float(kf.get("pts_time", 0.0))

                    if self.meta_store is not None:
                        meta = self.meta_store.get_by_keyframe_id(keyframe_id)
                        if meta:
                            frame_idx = meta.frame_idx
                            pts_time = meta.pts_time

                    self._records[keyframe_id] = {
                        "keyframe_id": keyframe_id,
                        "video_id": video_id,
                        "n": n,
                        "frame_idx": frame_idx,
                        "pts_time": pts_time,
                        "text": raw_text_str,
                        "clean_text": clean_text,
                        "tokens": set(_tokenize(clean_text)),
                    }
                    total_with_text += 1
            except Exception as e:
                logger.warning(f"[InMemoryOCRRetriever] Failed to parse {json_path.name}: {e}")

        # ── Format B: new .jsonl files (one JSON object per line) ─────────────
        for jsonl_path in jsonl_files:
            try:
                with open(jsonl_path, encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # JSONL record can be:
                        #  {"video_id": ..., "n": ..., "texts": [...], "frame_idx": ..., "pts_time": ...}
                        #  {"video_id": ..., "n": ..., "ocr_text": "...", ...}
                        vid = record.get("video_id", jsonl_path.stem)
                        n = int(record.get("n", record.get("frame_n", 0)))

                        # Extract text — support multiple field names
                        texts_field = record.get("texts", record.get("text_lines", []))
                        if isinstance(texts_field, list):
                            raw_text_str = " ".join(str(t) for t in texts_field if t)
                        else:
                            raw_text_str = str(texts_field)

                        # Fallback to flat ocr_text field
                        if not raw_text_str.strip():
                            raw_text_str = str(record.get("ocr_text", ""))

                        clean_text = _normalize_text(raw_text_str)
                        if not clean_text:
                            continue

                        keyframe_id = f"{vid}_n{n}"
                        frame_idx = int(record.get("frame_idx", 0))
                        pts_time = float(record.get("pts_time", record.get("timestamp", 0.0)))

                        if self.meta_store is not None:
                            meta = self.meta_store.get_by_keyframe_id(keyframe_id)
                            if meta:
                                frame_idx = meta.frame_idx
                                pts_time = meta.pts_time

                        # JSONL records can overwrite existing .json records for same keyframe
                        self._records[keyframe_id] = {
                            "keyframe_id": keyframe_id,
                            "video_id": vid,
                            "n": n,
                            "frame_idx": frame_idx,
                            "pts_time": pts_time,
                            "text": raw_text_str,
                            "clean_text": clean_text,
                            "tokens": set(_tokenize(clean_text)),
                        }
                        total_with_text += 1
            except Exception as e:
                logger.warning(f"[InMemoryOCRRetriever] Failed to parse {jsonl_path.name}: {e}")

        self._is_loaded = True
        logger.info(
            f"[InMemoryOCRRetriever] Loaded {total_with_text:,} keyframes with OCR text "
            f"across {total_files} files ({len(json_files)} json + {len(jsonl_files)} jsonl)."
        )
        return self

    def retrieve(
        self,
        query: object,
        top_k: int = 100,
        target_prefix: Optional[str] = None,  # kept for interface compatibility, not used
        max_per_video: int = 2,
        max_per_batch: int = 10,
    ) -> List[SearchResult]:
        """
        Search OCR text in memory for query matches.

        Args:
            query:         Text string or TextualKISQuery
            top_k:         Number of top results to return
            target_prefix: Ignored — disabled. Full database is always searched.
            max_per_video: Max keyframes per video_id (default: 2)
            max_per_batch: Max keyframes per batch prefix L21..L30 (default: 10)
        """
        if not self.is_configured:
            logger.debug("[InMemoryOCRRetriever] Not loaded or empty, returning empty list")
            return []

        # target_prefix intentionally disabled — always search full database
        target_prefix = None

        # Extract search strings & keywords from query object
        query_text = ""
        ocr_keywords: List[str] = []

        if isinstance(query, str):
            query_text = query
        elif hasattr(query, "raw_text"):
            query_text = getattr(query, "raw_text", "")
            ocr_keywords = getattr(query, "ocr_keywords", [])
            # Note: target_prefix from query object is intentionally ignored
        else:
            query_text = str(query)

        clean_query = _normalize_text(query_text)
        query_tokens = set(_tokenize(clean_query))

        # Filter out common stop words in Vietnamese & English for token scoring
        stop_words = {
            "co", "la", "trong", "o", "tren", "duoi", "va", "cung", "voi", "mot",
            "nguoi", "hinh", "canh", "video", "nhin", "thay", "cho", "dang", "duoc",
            "a", "an", "the", "in", "on", "at", "with", "is", "are", "and", "of",
        }
        substantive_tokens = query_tokens - stop_words
        if not substantive_tokens:
            substantive_tokens = query_tokens

        clean_ocr_keywords = [_normalize_text(kw) for kw in ocr_keywords if kw.strip()]

        scored_results: List[Tuple[float, Dict[str, Any]]] = []

        for kid, record in self._records.items():
            kf_text = record["clean_text"]
            kf_tokens = record["tokens"]

            score = 0.0

            # 1. Exact match for explicit OCR keywords (Highest Priority)
            for kw in clean_ocr_keywords:
                if kw in kf_text:
                    score += 3.0  # Significant boost for explicit keywords (e.g. "VTV1")
                elif any(word in kf_tokens for word in _tokenize(kw)):
                    score += 1.5

            # 2. Exact phrase/substring match of full query in OCR text
            if len(clean_query) >= 3 and clean_query in kf_text:
                score += 2.0

            # 3. Token overlap score
            if substantive_tokens:
                overlap = kf_tokens.intersection(substantive_tokens)
                if overlap:
                    ratio = len(overlap) / len(substantive_tokens)
                    score += ratio * 1.5

            if score > 0.0:
                scored_results.append((score, record))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Apply video-level + batch-level deduplication
        results: List[SearchResult] = []
        video_counts: Dict[str, int] = {}
        batch_counts: Dict[str, int] = {}

        for score, rec in scored_results:
            if len(results) >= top_k:
                break

            vid = rec["video_id"]
            batch = vid.split("_")[0]  # e.g. "L25"

            # Video-level cap
            if max_per_video > 0:
                if video_counts.get(vid, 0) >= max_per_video:
                    continue

            # Batch-level cap (prevents L25/L26 dominance when target_prefix is None)
            if max_per_batch > 0 and not target_prefix:
                if batch_counts.get(batch, 0) >= max_per_batch:
                    continue

            video_counts[vid] = video_counts.get(vid, 0) + 1
            batch_counts[batch] = batch_counts.get(batch, 0) + 1

            results.append(
                SearchResult(
                    keyframe_id=rec["keyframe_id"],
                    video_id=rec["video_id"],
                    n=rec["n"],
                    frame_idx=rec["frame_idx"],
                    pts_time=rec["pts_time"],
                    score=float(score),
                    retriever_source=self.name,
                    metadata={"ocr_text": rec["text"][:100]},
                )
            )

        logger.info(
            f"[{self.name}] '{query_text[:50]}' (full-db) -> {len(results)} results "
            f"from {len(video_counts)} videos / {len(batch_counts)} batches"
        )
        return results
