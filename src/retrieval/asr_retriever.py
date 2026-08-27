"""
ASR (Automatic Speech Recognition) Retriever — Trụ 4B.

Retrieves keyframes by searching transcribed speech content from videos.
Uses a pre-built FAISS index over ASR transcript embeddings for semantic search.

ASR data layout (datasets/artifacts/indexes/asr_v3/):
  - asr_corpus.parquet      — video_id, n, frame_idx, pts_time, transcript
  - asr_*.faiss             — FAISS flat-IP index of BGE embeddings

Falls back to keyword matching if FAISS index is not available.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.retrieval.base import BaseRetriever
from src.common.types import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Parquet field names in ASR corpus
_FIELD_VIDEO_ID   = "video_id"
_FIELD_N          = "n"
_FIELD_FRAME_IDX  = "frame_idx"
_FIELD_PTS_TIME   = "pts_time"
_FIELD_TRANSCRIPT = "transcript"


def _normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _tokenize(text: str) -> List[str]:
    """Extract alphanumeric words (lowercase)."""
    return re.findall(r"\w+", text.lower())


class ASRRetriever(BaseRetriever):
    """
    Retrieves keyframes by searching ASR transcript content.

    Two modes (auto-selected based on available data):
      1. FAISS semantic search — if BGE encoder + FAISS index are available
      2. Keyword matching     — fallback using tokenized transcript text

    Args:
        asr_dir:     Directory containing asr_corpus.parquet and optionally *.faiss
        meta_store:  Optional MetadataStore for frame_idx/pts_time lookups
        encoder:     Optional BGEEncoder for semantic search (if None, keyword mode)

    Usage:
        retriever = ASRRetriever(asr_dir="datasets/artifacts/indexes/asr_v3")
        retriever.load()
        results = retriever.retrieve("bản tin thời sự VTV1", top_k=50)
    """

    def __init__(
        self,
        asr_dir: str | Path,
        meta_store: Optional[Any] = None,
        encoder: Optional[Any] = None,     # BGEEncoder or compatible text encoder
    ):
        self.asr_dir   = Path(asr_dir)
        self.meta_store = meta_store
        self._encoder  = encoder

        # Keyword mode storage: keyframe_id → record
        self._records: Dict[str, Dict[str, Any]] = {}
        # FAISS mode storage
        self._faiss_index  = None
        self._faiss_corpus: List[Dict[str, Any]] = []  # parallel to FAISS vectors
        self._is_loaded: bool = False

    @property
    def name(self) -> str:
        return "asr_retriever"

    @property
    def is_configured(self) -> bool:
        return self._is_loaded and (
            len(self._records) > 0 or len(self._faiss_corpus) > 0
        )

    def load(self) -> "ASRRetriever":
        """
        Load ASR data from asr_dir.

        Tries to load:
          1. asr_corpus.parquet → keyword records
          2. Any *.faiss file   → FAISS index (if BGE encoder available)
        """
        if not self.asr_dir.exists():
            logger.warning(f"[ASRRetriever] Directory not found: {self.asr_dir}")
            return self

        # ── Load corpus parquet ────────────────────────────────────────────────
        parquet_files = sorted(self.asr_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"[ASRRetriever] No *.parquet files found in {self.asr_dir}")
        else:
            self._load_parquet(parquet_files[0])

        # ── Try to load FAISS index (if encoder available) ─────────────────────
        faiss_files = sorted(self.asr_dir.glob("*.faiss"))
        if faiss_files and self._encoder is not None:
            self._load_faiss(faiss_files[0])
        elif faiss_files:
            logger.info(
                f"[ASRRetriever] Found {len(faiss_files)} FAISS index(es) but no encoder — "
                f"will use keyword matching mode."
            )

        self._is_loaded = True
        mode = "FAISS semantic" if self._faiss_index is not None else "keyword matching"
        logger.info(
            f"[ASRRetriever] Ready: {len(self._records):,} transcript records | mode={mode}"
        )
        return self

    def _load_parquet(self, parquet_path: Path) -> None:
        """Load ASR corpus from parquet file into keyword-matching records."""
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            logger.info(
                f"[ASRRetriever] Loading parquet: {parquet_path.name} "
                f"({len(df):,} rows, cols={list(df.columns[:6])})"
            )
        except Exception as e:
            logger.warning(f"[ASRRetriever] Failed to load parquet {parquet_path}: {e}")
            return

        # Map column names flexibly
        col_map = {}
        for field, aliases in [
            (_FIELD_VIDEO_ID,   ["video_id", "vid", "video"]),
            (_FIELD_N,          ["n", "frame_n", "keyframe_n"]),
            (_FIELD_FRAME_IDX,  ["frame_idx", "frame_index", "idx"]),
            (_FIELD_PTS_TIME,   ["pts_time", "timestamp", "time_sec", "pts"]),
            (_FIELD_TRANSCRIPT, ["transcript", "text", "asr_text", "speech"]),
        ]:
            for alias in aliases:
                if alias in df.columns:
                    col_map[field] = alias
                    break

        if _FIELD_VIDEO_ID not in col_map or _FIELD_TRANSCRIPT not in col_map:
            logger.warning(
                f"[ASRRetriever] Parquet missing required columns. "
                f"Found: {list(df.columns)}. Need: video_id, transcript."
            )
            return

        loaded = 0
        for _, row in df.iterrows():
            try:
                vid   = str(row[col_map[_FIELD_VIDEO_ID]])
                n     = int(row[col_map[_FIELD_N]]) if _FIELD_N in col_map else 0
                fidx  = int(row[col_map[_FIELD_FRAME_IDX]]) if _FIELD_FRAME_IDX in col_map else 0
                pts   = float(row[col_map[_FIELD_PTS_TIME]]) if _FIELD_PTS_TIME in col_map else 0.0
                text  = str(row[col_map[_FIELD_TRANSCRIPT]] or "")

                clean_text = _normalize_text(text)
                if not clean_text:
                    continue

                keyframe_id = f"{vid}_n{n}"

                # Complement from MetadataStore if available
                if self.meta_store is not None:
                    meta = self.meta_store.get_by_keyframe_id(keyframe_id)
                    if meta:
                        fidx = meta.frame_idx
                        pts  = meta.pts_time

                self._records[keyframe_id] = {
                    "keyframe_id": keyframe_id,
                    "video_id": vid,
                    "n": n,
                    "frame_idx": fidx,
                    "pts_time": pts,
                    "text": text,
                    "clean_text": clean_text,
                    "tokens": set(_tokenize(clean_text)),
                }
                loaded += 1
            except Exception:
                continue

        logger.info(f"[ASRRetriever] Loaded {loaded:,} records with transcript text")

    def _load_faiss(self, faiss_path: Path) -> None:
        """Load pre-built FAISS index for ASR semantic search."""
        try:
            import faiss
            index = faiss.read_index(str(faiss_path))
            self._faiss_index = index
            # Build parallel corpus list aligned with FAISS vectors
            self._faiss_corpus = list(self._records.values())
            logger.info(
                f"[ASRRetriever] Loaded FAISS index: {faiss_path.name} "
                f"({index.ntotal:,} vectors, d={index.d})"
            )
        except Exception as e:
            logger.warning(f"[ASRRetriever] Failed to load FAISS index: {e}")
            self._faiss_index = None

    def retrieve(
        self,
        query: object,
        top_k: int = 100,
        target_prefix: Optional[str] = None,
        max_per_video: int = 3,
        max_per_batch: int = 10,
    ) -> List[SearchResult]:
        """
        Search ASR transcripts for the given query.

        Automatically chooses FAISS semantic search or keyword matching
        based on what's available.

        Args:
            query:         Text string or object with .raw_text attribute
            top_k:         Number of top results to return
            target_prefix: Optional batch filter (e.g. "L21")
            max_per_video: Max keyframes per video (default: 3)
            max_per_batch: Max keyframes per batch L21..L30 (default: 10)
        """
        if not self.is_configured:
            return []

        # Extract query text
        if isinstance(query, str):
            query_text = query
        elif hasattr(query, "raw_text"):
            query_text = getattr(query, "raw_text", "")
        else:
            query_text = str(query)

        if not query_text.strip():
            return []

        # Choose retrieval mode
        if self._faiss_index is not None and self._encoder is not None:
            return self._retrieve_faiss(
                query_text, top_k, target_prefix, max_per_video, max_per_batch
            )
        else:
            return self._retrieve_keyword(
                query_text, top_k, target_prefix, max_per_video, max_per_batch
            )

    def _retrieve_faiss(
        self,
        query_text: str,
        top_k: int,
        target_prefix: Optional[str],
        max_per_video: int,
        max_per_batch: int,
    ) -> List[SearchResult]:
        """FAISS semantic search over ASR embeddings."""
        try:
            import numpy as np
            query_vec = self._encoder.encode_text(query_text, normalize=True)
            query_vec = np.array([query_vec], dtype=np.float32)

            # Search with generous pool
            search_k = min(self._faiss_index.ntotal, top_k * 10)
            scores, faiss_ids = self._faiss_index.search(query_vec, search_k)

            results: List[SearchResult] = []
            video_counts: Dict[str, int] = {}
            batch_counts: Dict[str, int] = {}

            for score, fid in zip(scores[0], faiss_ids[0]):
                if fid < 0 or fid >= len(self._faiss_corpus):
                    continue

                rec = self._faiss_corpus[int(fid)]
                vid = rec["video_id"]
                batch = vid.split("_")[0]

                if target_prefix and not vid.startswith(target_prefix):
                    continue
                if max_per_video > 0 and video_counts.get(vid, 0) >= max_per_video:
                    continue
                if max_per_batch > 0 and not target_prefix:
                    if batch_counts.get(batch, 0) >= max_per_batch:
                        continue

                video_counts[vid] = video_counts.get(vid, 0) + 1
                batch_counts[batch] = batch_counts.get(batch, 0) + 1

                results.append(SearchResult(
                    keyframe_id=rec["keyframe_id"],
                    video_id=vid,
                    n=rec["n"],
                    frame_idx=rec["frame_idx"],
                    pts_time=rec["pts_time"],
                    score=float(score),
                    retriever_source=f"{self.name}_faiss",
                    metadata={"asr_text": rec["text"][:100]},
                ))
                if len(results) >= top_k:
                    break

            logger.info(
                f"[{self.name}] FAISS '{query_text[:50]}' → {len(results)} results "
                f"from {len(video_counts)} videos"
            )
            return results

        except Exception as e:
            logger.warning(f"[ASRRetriever] FAISS search failed: {e} — falling back to keyword")
            return self._retrieve_keyword(
                query_text, top_k, target_prefix, max_per_video, max_per_batch
            )

    def _retrieve_keyword(
        self,
        query_text: str,
        top_k: int,
        target_prefix: Optional[str],
        max_per_video: int,
        max_per_batch: int,
    ) -> List[SearchResult]:
        """Keyword matching over ASR transcripts."""
        clean_query = _normalize_text(query_text)
        query_tokens = set(_tokenize(clean_query))

        stop_words = {
            "co", "la", "trong", "o", "tren", "duoi", "va", "cung", "voi", "mot",
            "nguoi", "hinh", "canh", "video", "nhin", "thay", "cho", "dang", "duoc",
            "a", "an", "the", "in", "on", "at", "with", "is", "are", "and", "of",
        }
        substantive_tokens = query_tokens - stop_words or query_tokens

        scored: List[tuple] = []
        for kid, rec in self._records.items():
            if target_prefix and not rec["video_id"].startswith(target_prefix):
                continue

            kf_text   = rec["clean_text"]
            kf_tokens = rec["tokens"]
            score = 0.0

            # Phrase match
            if len(clean_query) >= 3 and clean_query in kf_text:
                score += 2.5

            # Token overlap
            overlap = kf_tokens.intersection(substantive_tokens)
            if overlap:
                score += len(overlap) / len(substantive_tokens) * 1.5

            if score > 0:
                scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResult] = []
        video_counts: Dict[str, int] = {}
        batch_counts: Dict[str, int] = {}

        for score, rec in scored:
            if len(results) >= top_k:
                break

            vid   = rec["video_id"]
            batch = vid.split("_")[0]

            if max_per_video > 0 and video_counts.get(vid, 0) >= max_per_video:
                continue
            if max_per_batch > 0 and not target_prefix:
                if batch_counts.get(batch, 0) >= max_per_batch:
                    continue

            video_counts[vid] = video_counts.get(vid, 0) + 1
            batch_counts[batch] = batch_counts.get(batch, 0) + 1

            results.append(SearchResult(
                keyframe_id=rec["keyframe_id"],
                video_id=vid,
                n=rec["n"],
                frame_idx=rec["frame_idx"],
                pts_time=rec["pts_time"],
                score=float(score),
                retriever_source=self.name,
                metadata={"asr_text": rec["text"][:100]},
            ))

        logger.info(
            f"[{self.name}] keyword '{query_text[:50]}' → {len(results)} results "
            f"from {len(video_counts)} videos / {len(batch_counts)} batches"
        )
        return results
