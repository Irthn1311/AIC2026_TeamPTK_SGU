import sys
from src.reasoning.query_parser import QueryParser

parser = QueryParser()

test_queries = [
    "Tìm cảnh một con hẻm đông người và xe máy, hai bên treo nhiều cờ Việt Nam.",
    "Tìm cảnh cháy rừng lớn trên sườn núi, khói dày phủ bầu trời vào ban đêm.",
    "3 cầu thủ bóng đá mặc áo đỏ đang sút bóng vào lưới trên sân vận động.",
    "Biên tập viên nam mặc áo vest đen đang phát biểu trong bản tin thời sự.",
    "Cảnh một người phụ nữ đội nón lá đang gặt lúa ở đồng ruộng.",
]

print("=== TESTING QUERY PARSER TRANSLATION & PROMPT GENERATION ===")
for q in test_queries:
    kis = parser.parse_kis(q)
    print(f"\n[QUERY VI]   : '{q}'")
    print(f"[TRANSLATED] : '{parser.translate_vi_sentence(q)}'")
    print(f"[CLIP PROMPT]: '{kis.clip_prompt}'")
    print(f"[OCR QUERY]  : '{kis.ocr_query}'")
