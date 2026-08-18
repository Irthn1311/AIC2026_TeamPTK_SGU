import sys
from src.reasoning.query_parser import QueryParser

parser = QueryParser()

test_queries = [
    # Category 1: Real User Failures
    "Tìm cảnh núi lửa đang phun, tạo cột khói rất lớn trên nền trời xanh.",
    "Tìm cảnh một người đàn ông đội mũ rơm được phỏng vấn cạnh bờ biển nhiều đá.",
    "Tìm cảnh một phụ nữ mặc áo xanh họa tiết được phỏng vấn, phía sau có các gói sản phẩm và giấy chứng nhận.",

    # Category 2: Historic Ruins & Architecture
    "Tìm cảnh một tháp cổ bằng gạch đã xuống cấp, cây xanh mọc trên phần thân công trình.",

    # Category 3: Extreme Sports & Skateparks
    "Tìm cảnh một khu BMX/skatepark ngoài trời vào ban đêm với nhiều dốc trượt và hình vẽ xe đạp trên tường.",

    # Category 4: Countryside & Water Jets
    "Tìm cảnh một người đàn ông ngồi xổm bên một cột nước trắng phun mạnh thẳng lên từ mặt đất ở khu vực nông thôn.",

    # Category 5: Public Officials & Federal Reserve
    "Tìm cảnh một người đàn ông tóc bạc, đeo kính, phát biểu tại bục với cờ Mỹ và biểu tượng ngân hàng trung ương ở phía sau.",

    # Category 6: Hydroelectric Graphics & Data
    "Tìm đồ họa nền xanh liệt kê số cửa xả đang mở của các hồ thủy điện Hòa Bình, Sơn La và Tuyên Quang.",

    # Category 7: Fisheries & Animals
    "Tìm cận cảnh một thau/chậu tròn chứa rất nhiều cá nhỏ màu bạc.",
    "Bữa tiệc sinh nhật trong bản tin được tổ chức cho con hà mã.",

    # Category 8: Vintage Vehicles & Crowds
    "Tìm cảnh một chiếc xe mui trần cổ màu đỏ chở nhiều người đi qua đám đông.",

    # Category 9: TV Anchors & Outfits
    "Tìm cảnh nữ người dẫn chương trình mặc áo màu be/hồng nhạt đứng một mình trong trường quay.",

    # Category 10: Culinary Markets & Stalls
    "Tìm cảnh một khu chợ/không gian ẩm thực đông người, nhiều quầy chế biến món ăn dưới mái che lớn.",

    # Category 11: Safety Warning Signs & Rescue Vehicles
    "Biển cảnh báo sạt lở màu vàng đỏ nguy hiểm.",
    "Chiếc xe cứu trợ hoặc xe chữ thập đỏ di chuyển vào ban đêm.",

    # Category 12: High Swings & Agricultural Harvester
    "Người đang ở trên cao phía trên thành phố ngồi trên xích đu.",
    "Cây trồng đang được máy thu hoạch trên đồng."
]

print("=== MASSIVE MASTER TRANSLATION MATRIX (BẢN THỂ 6.0) TEST ===")
for i, q in enumerate(test_queries, 1):
    kis = parser.parse_kis(q)
    print(f"\n--- [QUERY {i:02d}] ---")
    print(f"  VI GỐC     : '{q}'")
    print(f"  TRANSLATED : '{parser.translate_vi_sentence(q)}'")
    print(f"  CLIP PROMPT: '{kis.clip_prompt}'")
