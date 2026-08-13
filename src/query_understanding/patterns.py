"""
Lexical Patterns, Regex Extractors, and Vocabulary for Query Understanding
==========================================================================
Maintains curated Vietnamese retrieval patterns for OCR, ASR, Action, Scene,
Temporal, and Visual clues without duplicating the Object Index SYNONYMS_MAP.
"""

from __future__ import annotations
import re
from typing import List, Pattern, Set

# ==============================================================================
# 1. OCR PATTERNS & REGEX EXTRACTORS
# ==============================================================================
OCR_MARKER_KEYWORDS: Set[str] = {
    "dòng chữ",
    "chữ",
    "ghi chữ",
    "ghi",
    "hiển thị",
    "xuất hiện chữ",
    "trên màn hình",
    "màn hình ghi",
    "biển báo",
    "biển hiệu",
    "bảng hiệu",
    "bảng tên",
    "bảng quảng cáo",
    "logo",
    "tiêu đề",
    "headline",
    "ticker",
    "phụ đề",
    "banner",
    "thương hiệu",
    "khẩu hiệu",
    "dòng tin",
    "ký tự",
}

# Regex patterns to isolate on-screen text payload
OCR_EXTRACTION_PATTERNS: List[Pattern] = [
    # Quoted text: có chữ "Bộ Y tế", có dòng chữ "..."
    re.compile(r'(?:dòng chữ|chữ|logo|biển hiệu|bảng hiệu|tiêu đề|ticker|ghi)\s*[:"]+([^"\'\n]+)["\']?', re.IGNORECASE),
    # Dòng chữ X: dòng chữ Bộ Y tế, có chữ Chợ Rẫy
    re.compile(r'(?:dòng chữ|có chữ|xuất hiện chữ|ghi chữ|màn hình ghi)\s+([A-ZÀ-Ỹa-zà-ỹ0-9\s\-_/]+?)(?:\s+(?:ở|trên|tại|phía|trong|kèm|với)|$)', re.IGNORECASE),
    # Logo X: logo HTV9, logo VTV1
    re.compile(r'logo\s+([A-ZÀ-Ỹa-zà-ỹ0-9\-_]+)', re.IGNORECASE),
    # Biển hiệu / Biển báo X
    re.compile(r'(?:biển hiệu|bảng hiệu|bảng tên|biển báo)\s+([A-ZÀ-Ỹa-zà-ỹ0-9\s\-_]+?)(?:\s+(?:ở|trên|tại|phía)|$)', re.IGNORECASE),
]

# ==============================================================================
# 2. ASR / SPEECH PATTERNS & REGEX EXTRACTORS
# ==============================================================================
SPEECH_MARKER_KEYWORDS: Set[str] = {
    "nói rằng",
    "nói là",
    "nói về",
    "nói chuyện",
    "nói",
    "phát biểu rằng",
    "phát biểu về",
    "phát biểu",
    "cho biết rằng",
    "cho biết",
    "chia sẻ rằng",
    "chia sẻ",
    "trả lời phỏng vấn",
    "trả lời",
    "phỏng vấn",
    "thông báo rằng",
    "thông báo",
    "tuyên bố rằng",
    "tuyên bố",
    "cho rằng",
    "đề cập đến",
    "đề cập",
    "theo lời",
    "nhắc đến",
    "cho hay",
    "đọc bản tin",
    "đọc",
    "giới thiệu",
    "thuyết minh",
    "bình luận",
    "giải thích",
    "khẳng định",
    "nhấn mạnh",
    "cảnh báo",
    "kêu gọi",
    "yêu cầu",
}

# Regex patterns to isolate spoken propositional payload
SPEECH_EXTRACTION_PATTERNS: List[Pattern] = [
    # nói rằng X, cho biết rằng X, tuyên bố rằng X
    re.compile(r'(?:nói rằng|phát biểu rằng|cho biết rằng|thông báo rằng|tuyên bố rằng|chia sẻ rằng|cho rằng|cho hay|khẳng định rằng)\s+(.+)', re.IGNORECASE),
    # nói về X, phát biểu về X
    re.compile(r'(?:nói về|phát biểu về|chia sẻ về|đề cập đến|bình luận về)\s+(.+)', re.IGNORECASE),
]

# ==============================================================================
# 3. ACTION LEXICON (Lexical video action verbs & motion)
# ==============================================================================
ACTION_MULTIWORD_LEXICON: List[str] = [
    # Compound action verbs (matched first by length)
    "chạy xe", "đi xe", "lái xe", "cất cánh", "hạ cánh", "bước xuống",
    "bước lên", "bước đi", "đi bộ", "vẫy tay", "bắt tay", "vỗ tay",
    "chỉ tay", "chụp ảnh", "quay phim", "nói chuyện", "trò chuyện",
    "thảo luận", "trình bày", "làm việc", "tập thể dục", "di chuyển",
    "rời khỏi", "tiến vào", "leo lên", "nhảy xuống", "đổ bộ",
]

ACTION_SINGLEWORD_LEXICON: Set[str] = {
    "đi", "chạy", "đứng", "ngồi", "nằm", "cầm", "mang", "mặc", "đội",
    "bước", "nhảy", "lái", "bay", "ăn", "uống", "nói", "mở", "đóng",
    "đưa", "nhận", "trao", "chỉ", "nhìn", "chơi", "bơi", "cắt", "nấu",
    "hát", "chào", "khám", "chữa", "vẽ", "viết", "đọc", "nghe",
}

# ==============================================================================
# 4. SCENE / CONTEXT LEXICON
# ==============================================================================
SCENE_MULTIWORD_LEXICON: List[str] = [
    "đường phố", "sân bay", "bệnh viện", "trường học", "sân vận động",
    "phòng họp", "hội trường", "sân khấu", "công viên", "ngoài trời",
    "trong nhà", "bờ sông", "bãi biển", "bến tàu", "công trường",
    "văn phòng", "phòng studio", "trường quay", "khu dân cư", "ngã tư",
    "trạm xe", "nhà ga", "trung tâm thương mại", "siêu thị",
]

SCENE_SINGLEWORD_LEXICON: Set[str] = {
    "sông", "biển", "đường", "nhà", "tòa nhà", "chợ", "ruộng", "núi",
    "rừng", "hồ", "cảng", "cầu", "hẻm", "phố", "vườn", "suối", "thác",
}

# ==============================================================================
# 5. TEMPORAL LEXICON (Chronological connectives)
# ==============================================================================
TEMPORAL_PATTERNS: List[str] = [
    "trước khi", "sau khi", "sau đó", "tiếp theo", "đầu tiên", "cuối cùng",
    "ban đầu", "rồi", "trước", "sau", "trong lúc", "khi", "đồng thời",
    "cùng lúc", "ngay sau đó", "vừa lúc", "lúc đầu", "kết thúc",
]

# ==============================================================================
# 6. VISUAL APPEARANCE & COMPOSITION CLUES
# ==============================================================================
VISUAL_CLUES_LEXICON: Set[str] = {
    # Colors
    "màu đỏ", "màu xanh", "màu vàng", "màu trắng", "màu đen", "màu cam",
    "màu tím", "màu hồng", "màu nâu", "màu xám", "màu lục", "màu lam",
    "áo đỏ", "áo xanh", "áo trắng", "áo đen", "áo vàng", "áo cam",
    "váy đỏ", "váy trắng", "váy đen", "quần jean", "áo vest", "áo thun",
    # Visual descriptors & accessories
    "đeo kính", "mặc áo", "đội mũ", "đội nón", "cà vạt", "khẩu trang",
    # Scene composition & camera layout
    "góc màn hình", "bên trái", "bên phải", "ở giữa", "phía trước", "phía sau",
    "toàn cảnh", "cận cảnh", "trung cảnh", "nền phía sau", "hậu cảnh",
}

# ==============================================================================
# 7. CONSERVATIVE ENTITY / OBJECT CANDIDATES
# (Sorted descending by length for greedy match)
# ==============================================================================
KNOWN_OBJECT_ENTITIES: List[str] = [
    # Multi-word humans & roles
    "người đàn ông", "người phụ nữ", "cô gái", "bé gái", "cậu bé", "bé trai",
    "con người", "người đi bộ", "phóng viên", "bác sĩ", "công an", "cảnh sát",
    "bảo vệ", "công nhân", "học sinh", "sinh viên", "người dân", "người phát biểu",
    # Watercraft
    "thuyền máy", "con thuyền", "cái thuyền", "tàu thủy", "cano", "thuyền", "tàu",
    # Land vehicles
    "xe hơi", "ô tô", "xe con", "xe máy", "xe đạp", "xe tải", "xe buýt", "xe cộ", "xe",
    # Aviation
    "máy bay", "phi cơ", "trực thăng",
    # Buildings & structures
    "tòa nhà cao tầng", "tòa nhà", "ngôi nhà", "nhà cao tầng", "nhà dân",
    # Nature & flora
    "cây cối", "cây xanh", "cây dừa", "hoa", "cây",
    # Objects & artifacts
    "cột cờ", "lá cờ", "mũ bảo hiểm", "vali", "túi xách", "ba lô",
    "điện thoại", "máy tính", "micro", "bàn ghế", "ghế", "bàn",
    # Animals
    "chó", "mèo", "ngựa", "chim", "cá",
]
