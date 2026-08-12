"""
Text Retriever — Qdrant-based search over captions, OCR, and ASR text.

Sprint 3+ version: fully wired to QdrantDB and BGEEncoder.
Gracefully returns empty list if Qdrant is not reachable or collection is empty.
"""

from __future__ import annotations

from typing import List, Optional

from src.retrieval.base import BaseRetriever
from src.common.types import SearchResult
from src.common.constants import (
    QDRANT_COLLECTION_CAPTIONS,
    QDRANT_COLLECTION_OCR,
    QDRANT_COLLECTION_ASR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextRetriever(BaseRetriever):
    """
    Retrieves keyframes via dense text search over Qdrant collections.

    Supports 3 text modalities:
      - "caption"  — auto-generated image captions (Qwen2.5-VL)
      - "ocr"      — extracted on-screen text (PaddleOCR)
      - "asr"      — speech transcripts (Faster-Whisper)

    Args:
        qdrant_db:  Connected QdrantDB instance (or None → stub mode)
        bge_encoder: Loaded BGEEncoder (or None → stub mode)
        modality:   Which collection to search: "caption" | "ocr" | "asr"
    """

    _COLLECTION_MAP = {
        "caption": QDRANT_COLLECTION_CAPTIONS,
        "ocr":     QDRANT_COLLECTION_OCR,
        "asr":     QDRANT_COLLECTION_ASR,
    }

    def __init__(
        self,
        modality: str = "caption",
        qdrant_db=None,      # QdrantDB instance
        bge_encoder=None,    # BGEEncoder instance
    ):
        if modality not in self._COLLECTION_MAP:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Choose from: {list(self._COLLECTION_MAP)}"
            )
        self.modality    = modality
        self._qdrant_db  = qdrant_db
        self._bge_encoder = bge_encoder

    @property
    def name(self) -> str:
        return f"text_{self.modality}"

    @property
    def collection(self) -> str:
        return self._COLLECTION_MAP[self.modality]

    @property
    def is_configured(self) -> bool:
        """Returns True if both Qdrant and BGE encoder are set and ready."""
        return self._qdrant_db is not None and self._bge_encoder is not None

    # ----------------------------------------------------------
    # Retrieve
    # ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 100,
        target_prefix: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search Qdrant for keyframes whose text (caption/ocr/asr) matches query.

        Returns empty list if Qdrant is not configured (graceful fallback).
        """
        if not self.is_configured:
            logger.debug(f"[{self.name}] Skipped — not configured (Qdrant/BGE not set)")
            return []

        # Encode query with BGE-M3
        try:
            query_vec = self._bge_encoder.encode(query, normalize=True)
        except Exception as e:
            logger.warning(f"[{self.name}] BGE encoding failed: {e}")
            return []

        # Search Qdrant
        try:
            hits = self._qdrant_db.search(
                query_vec=query_vec,
                collection=self.modality,
                top_k=top_k * 5 if target_prefix else top_k,
            )
        except Exception as e:
            logger.warning(f"[{self.name}] Qdrant search failed: {e}")
            return []

        results: List[SearchResult] = []
        for hit in hits:
            vid = hit.get("video_id", "")
            if target_prefix and not vid.startswith(target_prefix):
                continue

            results.append(SearchResult(
                keyframe_id=hit.get("keyframe_id", ""),
                video_id=vid,
                n=int(hit.get("n", 0)),
                frame_idx=int(hit.get("frame_idx", 0)),
                pts_time=float(hit.get("pts_time", 0.0)),
                score=float(hit.get("score", 0.0)),
                retriever_source=self.name,
                metadata={"text_snippet": hit.get("text", "")[:100]},
            ))
            if len(results) >= top_k:
                break

        logger.debug(f"[{self.name}] '{query[:50]}' (prefix={target_prefix}) → {len(results)} results")
        return results
