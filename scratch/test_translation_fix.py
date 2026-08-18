import sys
from src.reasoning.query_parser import QueryParser

parser = QueryParser()

test_queries = [
    # User's 3 specific failing queries:
    "Tìm cảnh núi lửa đang phun, tạo cột khói rất lớn trên nền trời xanh.",
    "Tìm cảnh một người đàn ông đội mũ rơm được phỏng vấn cạnh bờ biển nhiều đá.",
    "Tìm cảnh một phụ nữ mặc áo xanh họa tiết được phỏng vấn, phía sau có các gói sản phẩm và giấy chứng nhận.",
    
    # Other complex queries:
    "Tìm cảnh một con hẻm đông người và xe máy, hai bên treo nhiều cờ Việt Nam.",
    "Tìm cảnh cháy rừng lớn trên sườn núi, khói dày phủ bầu trời vào ban đêm.",
    "3 cầu thủ bóng đá mặc áo đỏ đang sút bóng vào lưới trên sân vận động.",
    "Biên tập viên nam mặc áo vest đen đang phát biểu trong bản tin thời sự.",
    "Cảnh một người phụ nữ đội nón lá đang gặt lúa ở đồng ruộng.",
]

print("=== TESTING MASTER QUERY TRANSLATOR (BẢN THỂ 5) ===")
for i, q in enumerate(test_queries, 1):
    kis = parser.parse_kis(q)
    print(f"\n--- [QUERY {i}] ---")
    print(f"  VI GỐC     : '{q}'")
    print(f"  TRANSLATED : '{parser.translate_vi_sentence(q)}'")
    print(f"  CLIP PROMPT: '{kis.clip_prompt}'")
