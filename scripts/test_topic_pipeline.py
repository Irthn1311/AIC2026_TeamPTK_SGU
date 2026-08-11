"""
Test script for Topic-Aware Soft-Scoring Pipeline with MediaInfo.
Runs fast unit checks without requiring large index files or GPU.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.types import MediaInfo, SearchResult
from src.storage.media_info_store import MediaInfoStore
from src.reasoning.topic_classifier import TopicClassifier
from src.reasoning.query_parser import QueryParser
from src.fusion.reciprocal_rank import ReciprocalRankFusion


def test_topic_classifier():
    print("\n--- 1. Testing TopicClassifier ---")
    classifier = TopicClassifier()

    sample_texts = {
        "60 Giay Sang - HTV Tin Tuc Moi Nhat 2024 - Thoi su 60s": "tin_tuc",
        "Huong dan nau mon pho bo truyen thong ngon tai nha": "nau_an",
        "Tran chung ket bong da Seagames ban thang phut chot": "the_thao",
        "Mua lan rong mung khai truong va ruoc den trung thu": "mua_lan",
        "Bai giang toan lop 10 mon dai so thay Nam day hoc": "day_hoc",
        "Tinh hinh ket xe giao thong tren duong Nguyen Hue": "giao_thong",
    }

    passed = 0
    for text, expected in sample_texts.items():
        res = classifier.classify_text(text)
        status = "PASSED" if res.topic == expected else "FAILED"
        if res.topic == expected:
            passed += 1
        print(f"[{status}] Text: '{text[:40]}...' -> Topic: {res.topic} (Expected: {expected}, Conf: {res.confidence})")

    print(f"Result: {passed}/{len(sample_texts)} tests passed.")
    assert passed == len(sample_texts), "TopicClassifier test failed!"


def test_query_parser_topic_extraction():
    print("\n--- 2. Testing QueryParser Topic Extraction ---")
    parser = QueryParser()

    queries = [
        "Tim doan clip dau bep dang xao thit trong bep",
        "Cau thu sut phat den trong tran dau the thao",
        "Ban tin thoi su 19h phat tren kenh truyen hinh",
    ]

    for q in queries:
        res = parser.extract_topic(q)
        print(f"Query: '{q}' -> Detected Topic: '{res.topic}' (Confidence: {res.confidence})")


def test_rrf_topic_soft_scoring():
    print("\n--- 3. Testing RRF Topic Soft-Scoring ---")
    rrf = ReciprocalRankFusion(k=60)

    # Simulated candidate 1 (Cooking topic)
    cand1 = SearchResult(
        keyframe_id="L21_V001_n1", video_id="L21_V001", n=1, frame_idx=0, pts_time=0.0,
        score=0.90, retriever_source="visual", metadata={"topic_category": "nau_an"}
    )
    # Simulated candidate 2 (News topic)
    cand2 = SearchResult(
        keyframe_id="L21_V002_n1", video_id="L21_V002", n=1, frame_idx=0, pts_time=0.0,
        score=0.92, retriever_source="visual", metadata={"topic_category": "tin_tuc"}
    )

    list1 = [cand2, cand1]  # cand2 initially ranked higher (rank 1 vs rank 2)

    # 1. Unboosted fusion
    fused_unboosted = rrf.fuse([list1], query_topic=None)
    print(f"Unboosted Top 1: {fused_unboosted[0].keyframe_id} (Score: {fused_unboosted[0].score:.4f})")

    # 2. Boosted fusion with query_topic="nau_an"
    fused_boosted = rrf.fuse([list1], query_topic="nau_an", topic_boost_weight=0.20)
    print(f"Boosted ('nau_an') Top 1: {fused_boosted[0].keyframe_id} (Score: {fused_boosted[0].score:.4f})")

    # Verify cand1 was boosted to top 1
    assert fused_boosted[0].keyframe_id == "L21_V001_n1", "Topic boost failed to rerank matching candidate!"
    print("PASSED: Topic Soft-Scoring successfully boosted matching candidate!")


def test_media_info_store():
    print("\n--- 4. Testing MediaInfoStore ---")
    store = MediaInfoStore("datasets/media-info")
    store.load()
    print(f"MediaInfoStore loaded successfully. Total videos: {store.total_videos}")


if __name__ == "__main__":
    print("=== Running Topic Pipeline Unit Tests ===")
    test_topic_classifier()
    test_query_parser_topic_extraction()
    test_rrf_topic_soft_scoring()
    test_media_info_store()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
