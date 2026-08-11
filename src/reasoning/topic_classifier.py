"""
TopicClassifier for AIC Video Retrieval System.

Categorizes videos (via media-info text) and queries (via query text)
into semantic topic categories for soft-scoring fusion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.common.types import MediaInfo
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Default Topic Taxonomy definition with characteristic keyword triggers
DEFAULT_TOPIC_TAXONOMY: Dict[str, List[str]] = {
    "tin_tuc": [
        "tin tức", "thời sự", "60s", "60 giây", "htv", "vtv", "bản tin", "thời sự 19h",
        "tin tức mới nhất", "báo chí", "truyền hình", "news", "report"
    ],
    "the_thao": [
        "thể thao", "bóng đá", "đua xe", "cầu lông", "bóng rổ", "bơi lội", "điền kinh",
        "võ thuật", "boxing", "chạy bộ", "xe đạp", "đua xe f1", "world cup", "seagames",
        "trận đấu", "bàn thắng", "giải đấu", "sport", "match", "goal"
    ],
    "nau_an": [
        "nấu ăn", "ẩm thực", "món ăn", "nhà bếp", "nấu nướng", "công thức", "hướng dẫn nấu",
        "đầu bếp", "bếp", "món ngon", "cooking", "recipe", "food", "chef", "kitchen"
    ],
    "mua_lan": [
        "múa lân", "múa rồng", "lân sư rồng", "trống lân", "trung thu", "lễ hội lân",
        "múa lân rồng", "đội lân", "lion dance", "dragon dance"
    ],
    "day_hoc": [
        "dạy học", "giảng dạy", "lớp học", "học sinh", "giáo viên", "trường học", "bài giảng",
        "giáo dục", "ôn thi", "tiết học", "thầy giáo", "cô giáo", "school", "education", "teacher"
    ],
    "giao_thong": [
        "giao thông", "đường phố", "xe cộ", "xe máy", "ô tô", "kẹt xe", "tai nạn giao thông",
        "xe buýt", "phương tiện", "đường bộ", "traffic", "street", "road"
    ],
    "am_nhac": [
        "âm nhạc", "ca nhạc", "bài hát", "mv", "nhạc sống", "hát", "ca sĩ", "nhạc cụ",
        "guitar", "piano", "concert", "music", "song", "singer"
    ],
    "du_lich": [
        "du lịch", "phong cảnh", "khám phá", "checkin", "địa điểm", "vlog du lịch",
        "thăm quan", "bãi biển", "vùng đất", "travel", "tourism", "landscape"
    ],
    "giai_tri": [
        "giải trí", "hài hước", "game show", "truyền hình thực tế", "phim ngắn", "kịch",
        "phim hài", "showbiz", "entertainment", "comedy", "show"
    ],
}


def remove_accents(text: str) -> str:
    """Strip Vietnamese accent diacritics for flexible text matching."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text)
    res = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return res.replace("đ", "d").replace("Đ", "D")


@dataclass
class TopicClassificationResult:
    topic: str              # Category ID (e.g. "nau_an", "tin_tuc")
    confidence: float       # Score between 0.0 and 1.0
    matched_keywords: List[str]


class TopicClassifier:
    """
    Classifies text (query or media-info metadata) into topic categories.
    Supports both Vietnamese accented and unaccented text matching.
    """

    def __init__(self, taxonomy: Optional[Dict[str, List[str]]] = None):
        self.taxonomy = taxonomy or DEFAULT_TOPIC_TAXONOMY

    def classify_text(self, text: str) -> TopicClassificationResult:
        """
        Classify text by keyword density and match scores.

        Args:
            text: Query description or concatenated media-info text

        Returns:
            TopicClassificationResult with best topic and confidence.
        """
        if not text or not text.strip():
            return TopicClassificationResult(topic="khac", confidence=0.0, matched_keywords=[])

        text_lower = text.lower()
        text_no_acc = remove_accents(text_lower)

        topic_scores: Dict[str, Tuple[float, List[str]]] = {}

        for topic, keywords in self.taxonomy.items():
            matches = []
            score = 0.0
            for kw in keywords:
                kw_lower = kw.lower()
                kw_no_acc = remove_accents(kw_lower)

                # Match in original text
                pattern = r'\b' + re.escape(kw_lower) + r'\b'
                occ = len(re.findall(pattern, text_lower))

                # Match in unaccented text if not matched in original
                if occ == 0 and kw_no_acc != kw_lower:
                    pattern_no_acc = r'\b' + re.escape(kw_no_acc) + r'\b'
                    occ = len(re.findall(pattern_no_acc, text_no_acc))

                if occ > 0:
                    matches.append(kw)
                    score += occ * (1.0 + 0.2 * len(kw.split()))

            if score > 0:
                topic_scores[topic] = (score, matches)

        if not topic_scores:
            return TopicClassificationResult(topic="khac", confidence=0.0, matched_keywords=[])

        # Find topic with highest score
        best_topic = max(topic_scores.keys(), key=lambda t: topic_scores[t][0])
        best_score, best_matches = topic_scores[best_topic]

        # Normalize confidence score
        confidence = min(1.0, round(best_score / 3.0, 2))

        return TopicClassificationResult(
            topic=best_topic,
            confidence=confidence,
            matched_keywords=best_matches,
        )

    def classify_media_info(self, media_info: MediaInfo) -> TopicClassificationResult:
        """Classify a video using its MediaInfo (author + keywords + description)."""
        combined_text = media_info.get_combined_text()
        res = self.classify_text(combined_text)
        media_info.topic_category = res.topic
        media_info.topic_confidence = res.confidence
        return res
