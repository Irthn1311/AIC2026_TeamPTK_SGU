"""
Visual Retriever — FAISS-based keyframe search using CLIP-32 vectors.

Query flow:
  1. Encode text query with CLIPEncoder → 512-dim vector
  2. Search FaissDB → (faiss_ids, cosine_scores)
  3. Lookup MetadataStore → KeyframeMeta for each result
  4. Return List[SearchResult] sorted by score descending

This retriever works entirely with the pre-extracted .npy features:
  no re-encoding of keyframe images is needed during retrieval.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from src.retrieval.base import BaseRetriever
from src.common.types import SearchResult
from src.database.faiss_db import FaissDB
from src.storage.metadata_store import MetadataStore
from src.embeddings.visual.clip import CLIPEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VisualRetriever(BaseRetriever):
    """
    Retrieves keyframes by visual similarity using CLIP text→vector→FAISS.

    Args:
        faiss_db:   Pre-loaded FaissDB instance
        meta_store: Pre-loaded MetadataStore instance
        encoder:    Pre-loaded CLIPEncoder (ViT-B/32)

    Usage:
        retriever = VisualRetriever(faiss_db, meta_store, encoder)
        results = retriever.retrieve("người dẫn mặc áo đỏ phát biểu ngoài trời", top_k=100)
    """

    def __init__(
        self,
        faiss_db: FaissDB,
        meta_store: MetadataStore,
        encoder: CLIPEncoder,
    ):
        self._faiss_db = faiss_db
        self._meta_store = meta_store
        self._encoder = encoder

    @property
    def name(self) -> str:
        return "visual_clip32"

    def retrieve(
        self,
        query: str,
        top_k: int = 100,
        target_prefix: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search FAISS for keyframes visually similar to the text query.

        Args:
            query: Natural language description (Vietnamese or English)
            top_k: Number of candidates to return
            target_prefix: e.g. "L21", "L23" (if specified, filter results to this prefix)

        Returns:
            List[SearchResult] sorted by cosine similarity score descending
        """
        # 1. Encode query text → CLIP vector
        query_vec = self._encoder.encode_text(query, normalize=True)

        # 2. FAISS ANN search (fetch extra candidates if filtering by prefix)
        search_k = max(top_k * 20, 2000) if target_prefix else top_k
        faiss_ids, scores = self._faiss_db.search(query_vec, top_k=search_k)

        # 3. Map faiss_id → KeyframeMeta → SearchResult
        results: List[SearchResult] = []
        for fid, score in zip(faiss_ids, scores):
            if fid < 0:  # FAISS returns -1 for empty slots
                continue
            meta = self._meta_store.get_by_faiss_id(int(fid))
            if meta is None:
                logger.warning(f"No metadata for faiss_id={fid}")
                continue

            if target_prefix and not meta.video_id.startswith(target_prefix):
                continue

            results.append(SearchResult(
                keyframe_id=meta.keyframe_id,
                video_id=meta.video_id,
                n=meta.n,
                frame_idx=meta.frame_idx,
                pts_time=meta.pts_time,
                score=float(score),
                retriever_source=self.name,
            ))
            if len(results) >= top_k:
                break

        logger.debug(f"[{self.name}] query='{query[:50]}' (prefix={target_prefix}) → {len(results)} results")
        return results

    def retrieve_by_vector(
        self,
        query_vec: np.ndarray,
        top_k: int = 100,
    ) -> List[SearchResult]:
        """
        Search using a pre-computed query vector (e.g., image query).

        Args:
            query_vec: L2-normalised CLIP vector shape (512,)
        """
        faiss_ids, scores = self._faiss_db.search(query_vec, top_k=top_k)

        results: List[SearchResult] = []
        for fid, score in zip(faiss_ids, scores):
            if fid < 0:
                continue
            meta = self._meta_store.get_by_faiss_id(int(fid))
            if meta is None:
                continue
            results.append(SearchResult(
                keyframe_id=meta.keyframe_id,
                video_id=meta.video_id,
                n=meta.n,
                frame_idx=meta.frame_idx,
                pts_time=meta.pts_time,
                score=float(score),
                retriever_source=self.name,
            ))
        return results

    def retrieve_within_video(
        self,
        query_vec: np.ndarray,
        video_id: str,
        top_k: int = 20,
    ) -> List[SearchResult]:
        """
        Restrict search to keyframes belonging to a specific video.
        Used in TRAKE Phase 2 per-event alignment.

        Args:
            query_vec: L2-normalised CLIP event description vector
            video_id:  e.g. "L21_V001"
            top_k:     Max results within this video
        """
        # Get all faiss_ids for this video
        video_faiss_ids = set(self._meta_store.faiss_ids_for_video(video_id))
        if not video_faiss_ids:
            logger.warning(f"No keyframes found for video_id={video_id}")
            return []

        # 1. Direct O(N_video) exact cosine similarity via vector reconstruction
        results: List[SearchResult] = []
        try:
            for fid in video_faiss_ids:
                try:
                    vec = self._faiss_db._index.reconstruct(int(fid))
                    sim = float(np.dot(vec, query_vec))
                    meta = self._meta_store.get_by_faiss_id(int(fid))
                    if meta:
                        results.append(SearchResult(
                            keyframe_id=meta.keyframe_id,
                            video_id=meta.video_id,
                            n=meta.n,
                            frame_idx=meta.frame_idx,
                            pts_time=meta.pts_time,
                            score=sim,
                            retriever_source=f"{self.name}_in_video",
                        ))
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"FAISS vector reconstruct failed: {e}")

        if results:
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        # 2. Fallback: Search globally with total_vectors, filter to video_id
        search_k = min(self._faiss_db.total_vectors, max(top_k * 10, 10000))
        faiss_ids, scores = self._faiss_db.search(query_vec, top_k=search_k)

        for fid, score in zip(faiss_ids, scores):
            if fid < 0 or int(fid) not in video_faiss_ids:
                continue
            meta = self._meta_store.get_by_faiss_id(int(fid))
            if meta is None:
                continue
            results.append(SearchResult(
                keyframe_id=meta.keyframe_id,
                video_id=meta.video_id,
                n=meta.n,
                frame_idx=meta.frame_idx,
                pts_time=meta.pts_time,
                score=float(score),
                retriever_source=f"{self.name}_in_video",
            ))
            if len(results) >= top_k:
                break

        return results
