import pytest
from system_tai.qa.rescue_tail import RescueCandidate, merge_rescue_tail


def test_merge_rescue_tail_empty_rescue():
    champion = [{"video_id": f"L21_V{i:03d}", "frame_id": i * 10, "answer": "xe", "rank": i} for i in range(1, 101)]
    result = merge_rescue_tail(champion, [], prefix_k=95, max_rescue=5)
    assert len(result) == 95
    for i in range(1, 96):
        assert result[i - 1]["video_id"] == champion[i - 1]["video_id"]
        assert result[i - 1]["rank"] == i


def test_merge_rescue_tail_with_valid_rescue():
    champion = [{"video_id": f"L21_V{i:03d}", "frame_id": i * 10, "answer": "xe", "rank": i} for i in range(1, 101)]
    rescue = [
        RescueCandidate(video_id="L21_V999", frame_id=500, answer="người", rescue_score=0.9, rescue_source="query_expansion"),
        RescueCandidate(video_id="L21_V888", frame_id=600, answer="màu đỏ", rescue_score=0.8, rescue_source="multi_crop"),
    ]
    result = merge_rescue_tail(champion, rescue, prefix_k=95, max_rescue=5)
    assert len(result) == 97
    # First 95 are exact champion
    for i in range(1, 96):
        assert result[i - 1]["video_id"] == champion[i - 1]["video_id"]
    # 96 and 97 are rescue
    assert result[95]["video_id"] == "L21_V999"
    assert result[95]["rank"] == 96
    assert result[95]["slot_source"] == "RESCUE_TAIL_QUERY_EXPANSION"
    assert result[96]["video_id"] == "L21_V888"
    assert result[96]["rank"] == 97
    assert result[96]["slot_source"] == "RESCUE_TAIL_MULTI_CROP"


def test_merge_rescue_tail_rejects_duplicates_and_caps_at_5():
    champion = [{"video_id": f"L21_V{i:03d}", "frame_id": i * 10, "answer": "xe", "rank": i} for i in range(1, 101)]
    rescue = [
        RescueCandidate(video_id="L21_V001", frame_id=10, answer="xe", rescue_score=0.9, rescue_source="query_expansion"),
    ] + [
        RescueCandidate(video_id=f"L21_V{100+i}", frame_id=500, answer="test", rescue_score=0.8, rescue_source="ocr")
        for i in range(1, 7)
    ]
    result = merge_rescue_tail(champion, rescue, prefix_k=95, max_rescue=5)
    assert len(result) == 100
    # Duplicate was rejected, exactly 5 new items admitted
    assert result[95]["video_id"] == "L21_V101"
    assert result[99]["video_id"] == "L21_V105"


def test_merge_rescue_tail_short_champion():
    champion = [{"video_id": "L21_V001", "frame_id": 10, "answer": "xe", "rank": 1}]
    rescue = [RescueCandidate(video_id="L21_V002", frame_id=20, answer="người", rescue_score=0.5, rescue_source="query_expansion")]
    result = merge_rescue_tail(champion, rescue, prefix_k=95, max_rescue=5)
    assert len(result) == 2
    assert result[0]["rank"] == 1
    assert result[1]["rank"] == 2
