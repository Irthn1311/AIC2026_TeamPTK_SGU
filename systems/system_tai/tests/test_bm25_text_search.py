"""Unit tests for BM25 Text Search and Document Retrieval Engine."""

from __future__ import annotations

import pytest

from system_tai.common.schemas import KISQuery
from system_tai.retrieval.bm25_search import (
    BM25Document,
    BM25Index,
    BM25TextRetriever,
    tokenize_text,
)


def test_tokenize_text_multilingual() -> None:
    assert tokenize_text("Xe cứu hỏa màu đỏ!") == ["xe", "cứu", "hỏa", "màu", "đỏ"]
    assert tokenize_text("A red fire truck, running fast.") == ["a", "red", "fire", "truck", "running", "fast"]
    assert tokenize_text("") == []


def test_bm25_index_and_search() -> None:
    index = BM25Index(k1=1.5, b=0.75)
    docs = [
        BM25Document("doc1", "L21_V001", 100, "Một chiếc xe cứu hỏa màu đỏ đang chạy trên đường"),
        BM25Document("doc2", "L21_V001", 200, "Người đàn ông áo xanh đang đi bộ qua cầu"),
        BM25Document("doc3", "L21_V002", 300, "Cảnh cháy rừng dữ dội vào ban đêm khói lửa ngút trời"),
        BM25Document("doc4", "L21_V002", 400, "Chiếc xe tải lớn màu trắng chở đầy hàng hóa"),
    ]
    index.add_documents(docs)
    assert index.total_documents == 4

    # Search for "xe cứu hỏa đỏ"
    results = index.search("xe cứu hỏa đỏ", top_k=2)
    assert len(results) >= 1
    top_doc, top_score = results[0]
    assert top_doc.video_id == "L21_V001"
    assert top_doc.frame_id == 100
    assert top_score > 0.0

    # Search for "cháy rừng"
    results_fire = index.search("cháy rừng", top_k=2)
    assert results_fire[0][0].video_id == "L21_V002"
    assert results_fire[0][0].frame_id == 300


def test_bm25_text_retriever_interface() -> None:
    retriever = BM25TextRetriever()
    retriever.add_documents([
        BM25Document("d1", "V1", 50, "Cột nước phun mạnh từ mặt đất"),
        BM25Document("d2", "V2", 80, "Người ngồi xổm bên đường"),
    ])

    query = KISQuery(query_id="KIS-01", text="cột nước phun", top_k=1)
    res = retriever.retrieve(query)

    assert res.query_id == "KIS-01"
    assert len(res.ranked_candidates) == 1
    assert res.ranked_candidates[0].video_id == "V1"
    assert res.ranked_candidates[0].frame_id == 50
    assert res.ranked_candidates[0].rank == 1
    assert res.ranked_candidates[0].source == "bm25_text"
