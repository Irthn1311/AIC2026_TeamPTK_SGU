"""
FAISS Vector Database Wrapper for AIC Video Retrieval System.

Handles:
- Building FAISS HNSW index from pre-extracted CLIP-32 .npy files
- Saving / loading index to disk
- Nearest-neighbour search returning (faiss_ids, distances)

The FAISS integer IDs align 1:1 with faiss_id column in keyframe_master.parquet.
Use MetadataStore.get_by_faiss_id() to map results back to KeyframeMeta.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss not installed. Run: pip install faiss-cpu  "
        "(or faiss-gpu on CUDA environment)"
    )

from src.common.constants import (
    CLIP32_FEATURE_DIM, FAISS_HNSW_M, FAISS_HNSW_EF_SEARCH,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FaissDB:
    """
    HNSW-based FAISS index for visual keyframe retrieval.

    The index stores L2-normalised CLIP-32 vectors so that
    inner-product search is equivalent to cosine similarity.

    Index IDs (int64) correspond directly to faiss_id in
    keyframe_master.parquet, enabling O(1) metadata lookups.
    """

    def __init__(self, dim: int = CLIP32_FEATURE_DIM):
        self.dim = dim
        self._index: Optional[faiss.IndexHNSWFlat] = None
        self._total: int = 0

    # ----------------------------------------------------------
    # Build
    # ----------------------------------------------------------

    def build_from_npy_files(
        self,
        npy_dir: str,
        id_offset: int = 0,
        normalize: bool = True,
    ) -> "FaissDB":
        """
        Load all .npy files from npy_dir and add them to the index.

        Each .npy file: shape (num_keyframes, 512) for CLIP-32.
        Files are processed in sorted order to maintain consistent
        faiss_id ↔ keyframe_id mapping.

        Args:
            npy_dir:    Directory containing L{XX}_{V}.npy files
            id_offset:  Starting faiss_id (default 0)
            normalize:  L2-normalize vectors for cosine similarity
        """
        npy_dir = Path(npy_dir)
        npy_files = sorted(
            list(npy_dir.glob("*.npy")) + list(npy_dir.glob("*/*.npy")) + list(npy_dir.rglob("*.npy")),
            key=lambda p: p.stem
        )
        # Deduplicate paths preserving order
        seen_stems = set()
        unique_npy_files = []
        for p in npy_files:
            if p.stem not in seen_stems:
                seen_stems.add(p.stem)
                unique_npy_files.append(p)
        npy_files = unique_npy_files

        logger.info(f"Found {len(npy_files)} unique .npy files in {npy_dir}")

        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {npy_dir}")

        # --- Init HNSW index (IDMap wraps it so we can assign custom IDs) ---
        hnsw = faiss.IndexHNSWFlat(self.dim, FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efSearch = FAISS_HNSW_EF_SEARCH
        self._index = faiss.IndexIDMap(hnsw)

        current_id = id_offset
        total_vectors = 0

        for npy_path in npy_files:
            vectors = np.load(str(npy_path)).astype(np.float32)

            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            if vectors.shape[1] != self.dim:
                logger.warning(
                    f"Skipping {npy_path.name}: expected dim={self.dim}, "
                    f"got {vectors.shape[1]}"
                )
                continue

            if normalize:
                faiss.normalize_L2(vectors)

            ids = np.arange(current_id, current_id + len(vectors), dtype=np.int64)
            self._index.add_with_ids(vectors, ids)

            current_id += len(vectors)
            total_vectors += len(vectors)
            logger.debug(f"  {npy_path.name}: {len(vectors)} vectors (ids {ids[0]}..{ids[-1]})")

        self._total = total_vectors
        logger.info(f"FAISS index built: {total_vectors:,} vectors, dim={self.dim}")
        return self

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: np.ndarray,
        normalize: bool = True,
    ) -> None:
        """Add additional vectors with explicit IDs."""
        vectors = vectors.astype(np.float32)
        if normalize:
            faiss.normalize_L2(vectors)
        self._index.add_with_ids(vectors, ids.astype(np.int64))
        self._total += len(vectors)

    # ----------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------

    def save(self, index_path: str) -> None:
        """Save FAISS index to disk."""
        out = Path(index_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(out))
        logger.info(f"FAISS index saved → {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")

    def load(self, index_path: str) -> "FaissDB":
        """Load FAISS index from disk and restore efSearch parameter."""
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        self._index = faiss.read_index(str(path))
        self._total = self._index.ntotal
        
        # Restore HNSW efSearch parameter (faiss.read_index resets it to default 16)
        try:
            # If wrapped in IndexIDMap
            if isinstance(self._index, faiss.IndexIDMap):
                sub_idx = faiss.downcast_index(self._index.index)
                if hasattr(sub_idx, "hnsw"):
                    sub_idx.hnsw.efSearch = FAISS_HNSW_EF_SEARCH
            elif hasattr(self._index, "hnsw"):
                self._index.hnsw.efSearch = FAISS_HNSW_EF_SEARCH
            logger.info(f"FAISS index loaded: {self._total:,} vectors from {path} (efSearch={FAISS_HNSW_EF_SEARCH})")
        except Exception as e:
            logger.warning(f"Could not set efSearch on loaded index: {e}")
        return self

    # ----------------------------------------------------------
    # Search
    # ----------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top_k nearest keyframes.

        Args:
            query_vector: Shape (dim,) or (1, dim)
            top_k:        Number of results
            normalize:    L2-normalize query vector

        Returns:
            (faiss_ids, scores): Both shape (top_k,)
            faiss_ids align with keyframe_master.parquet faiss_id column.
            scores are cosine similarities in [0, 1].
        """
        if self._index is None:
            raise RuntimeError("Index not loaded. Call build_from_npy_files() or load() first.")

        vec = query_vector.astype(np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        if normalize:
            faiss.normalize_L2(vec)

        distances, ids = self._index.search(vec, top_k)
        # distances are inner products (cosine sim after L2 norm)
        return ids[0], distances[0]

    def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 100,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries.

        Returns:
            (ids, scores): Both shape (num_queries, top_k)
        """
        vecs = query_vectors.astype(np.float32)
        if normalize:
            faiss.normalize_L2(vecs)
        distances, ids = self._index.search(vecs, top_k)
        return ids, distances

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------

    @property
    def total_vectors(self) -> int:
        return self._total

    @property
    def is_loaded(self) -> bool:
        return self._index is not None
