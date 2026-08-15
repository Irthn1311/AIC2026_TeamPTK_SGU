"""Unit tests for FAISS Vector Index and Retriever."""

from __future__ import annotations

import numpy as np
import pytest

from system_tai.common.schemas import KISQuery
from system_tai.retrieval.faiss_index import FaissVectorIndex, VectorRecord


def test_faiss_vector_index_add_and_search() -> None:
    dim = 4
    index = FaissVectorIndex(dimension=dim, index_type="flat", force_numpy=True)
    assert index.dimension == 4
    assert index.total_vectors == 0

    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.7071, 0.7071, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    records = [
        VectorRecord("V1", 100, 0, 1),
        VectorRecord("V1", 200, 1, 2),
        VectorRecord("V2", 150, 0, 1),
        VectorRecord("V2", 250, 1, 2),
    ]

    index.add(matrix, records)
    assert index.total_vectors == 4

    # Search for vector close to [1, 0, 0, 0]
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = index.search(query, top_k=2)

    assert len(results) == 2
    top_rec, top_score = results[0]
    assert top_rec.video_id == "V1"
    assert top_rec.actual_frame_id == 100
    assert top_score > 0.9

    second_rec, second_score = results[1]
    assert second_rec.video_id == "V2"
    assert second_rec.actual_frame_id == 150


def test_faiss_vector_index_validation_errors() -> None:
    index = FaissVectorIndex(dimension=4)
    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search(np.array([1, 0, 0, 0]), top_k=0)

    with pytest.raises(ValueError, match="matrix shape mismatch"):
        index.add(np.zeros((2, 3), dtype=np.float32), [VectorRecord("V1", 1, 0, 1), VectorRecord("V1", 2, 1, 2)])

    with pytest.raises(ValueError, match="record count mismatch"):
        index.add(np.zeros((2, 4), dtype=np.float32), [VectorRecord("V1", 1, 0, 1)])
