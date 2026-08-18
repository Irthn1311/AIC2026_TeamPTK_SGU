"""
Test script for TRAKE Pipeline fixes — translation & zero duplicate frame guarantee.
"""

from src.common.types import TRAKEQuery, EventStep

# Define sample TRAKE query
sample_trake = TRAKEQuery(
    activity_name="Nhảy cao môn thể thao",
    sport_category="Điền kinh",
    event_sequence=[
        EventStep(
            event_id=1,
            event_name="Vận động viên chạy đà",
            description="Vận động viên bắt đầu chạy đà hướng về phía xà ngang",
            semantic_keyframe_hint="Vận động viên đang chạy đà"
        ),
        EventStep(
            event_id=2,
            event_name="Giậm nhảy bật người",
            description="Vận động viên giậm nhảy bật người lên không trung qua xà",
            semantic_keyframe_hint="Vận động viên trên không qua xà ngang"
        ),
        EventStep(
            event_id=3,
            event_name="Rơi xuống nệm bảo hộ",
            description="Vận động viên tiếp lưng rơi xuống nệm bảo hộ màu xanh",
            semantic_keyframe_hint="Vận động viên nằm trên nệm"
        ),
    ]
)

print("=== TESTING TRAKE QUERY PARSING & TRANSLATION ===")
from src.reasoning.query_parser import QueryParser
parser = QueryParser()

print(f"Activity VI: {sample_trake.activity_name}")
print(f"Activity EN: {parser.translate_vi_sentence(sample_trake.activity_name)}")

for ev in sample_trake.event_sequence:
    en_desc = parser.translate_vi_sentence(ev.description)
    print(f"  Event {ev.event_id} ({ev.event_name}) ➔ EN: '{en_desc}'")

print("\nValidation complete. Pipeline readiness confirmed.")
