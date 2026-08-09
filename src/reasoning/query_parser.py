"""
Query Parser for AIC Video Retrieval System (v2 — Accuracy-Optimized).

Changes from v1:
- Extended Vi→En vocabulary: 80+ terms across colors, scenes, sports, clothing, objects.
- Handles "xanh lá" vs "xanh dương" ambiguity correctly.
- build_qa_retrieval_text(): New method combining event_description + question keywords
  → better recall for QA candidates (was using only event_description).
- build_retrieval_text(): Adds English paraphrase suffix for stronger CLIP alignment.
- infer_answer_type(): Extracted as a public helper for QAPipeline.
- Expanded object detection: news broadcast, ceremony, sports equipment.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.common.types import TextualKISQuery, QAQuery, TRAKEQuery, EventStep
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# Color keywords (vi + en) — handles compound colors first
# ============================================================
# Order matters: check longer compound terms before single words
_COLORS_VI_COMPOUND = [
    ("xanh lá", "green"), ("xanh lá cây", "green"), ("xanh dương", "blue"),
    ("xanh da trời", "blue"), ("xanh nước biển", "navy blue"),
    ("đỏ tươi", "bright red"), ("vàng kim", "gold"), ("đen tuyền", "jet black"),
    ("trắng tinh", "pure white"), ("hồng đậm", "deep pink"), ("cam đỏ", "orange red"),
]
_COLORS_VI_SIMPLE = [
    "đỏ", "xanh", "vàng", "trắng", "đen", "tím", "hồng", "cam", "nâu", "xám",
    "bạc", "vàng kim", "be", "kem",
]
_COLORS_EN = [
    "red", "blue", "green", "yellow", "white", "black", "purple",
    "pink", "orange", "brown", "grey", "gray", "silver", "gold", "navy",
    "turquoise", "beige", "cream",
]

# ============================================================
# Scene / environment keywords
# Priority: specific domain scenes FIRST, then generic outdoor/indoor
# ============================================================
_SCENE_KEYWORDS = {
    # Specific contexts first (higher specificity → checked first)
    "sport": [
        "thể thao", "thi đấu", "vận động viên", "cầu thủ", "sút bóng",
        "nhảy cao", "chạy đà", "bơi lội", "điền kinh",
        "sport", "athletic", "race", "jump", "competition", "match", "game",
        "tournament", "athlete", "player",
    ],
    "news": [
        "bản tin", "thời sự", "tin tức", "phóng sự",
        "news", "broadcast", "anchor", "reporter",
    ],
    "press_conference": ["họp báo", "press conference", "briefing"],
    "ceremony": ["lễ trao giải", "award ceremony", "trao giải", "khai mạc", "bế mạc"],
    # Generic location contexts last
    "outdoor": [
        "ngoài trời", "đường phố", "quảng trường", "bãi biển", "công viên",
        "outdoor", "outside", "street", "beach", "park",
    ],
    "indoor": [
        "trong nhà", "phòng họp", "hội trường", "studio", "phòng quay",
        "hội nghị", "sàn diễn",
        "indoor", "inside", "hall", "room",
    ],
}

# ============================================================
# Object keywords — expanded for AIC domain
# ============================================================
_OBJECT_PATTERNS = {
    "person": (
        r"\b(người|diễn giả|phát ngôn viên|vận động viên|cầu thủ|người phát biểu"
        r"|phóng viên|biên tập viên|nghệ sĩ|ca sĩ|quan chức|lãnh đạo|bộ trưởng"
        r"|speaker|athlete|player|person|man|woman|reporter|anchor|official)\b"
    ),
    "vehicle": r"\b(xe|ô tô|xe buýt|xe tải|xe máy|car|bus|truck|motorcycle|vehicle)\b",
    "flag": r"\b(cờ|flag|banner|biểu ngữ)\b",
    "screen": r"\b(màn hình|bảng|screen|board|sign|logo|biểu bảng)\b",
    "microphone": r"\b(micro|microphone|mic|bục phát biểu|podium)\b",
    "ball": r"\b(bóng|ball|quả bóng)\b",
    "medal": r"\b(huy chương|medal|cúp|cup|trophy|giải thưởng)\b",
    "crowd": r"\b(đám đông|khán giả|cổ động viên|crowd|audience|fans|spectators)\b",
}

# ============================================================
# Spatial relationship keywords
# ============================================================
_SPATIAL_PATTERNS = [
    r"(phía\s+\w+|bên\s+\w+|góc\s+\w+)",        # Vietnamese: phía sau, bên trái, góc trên
    r"(behind|in front of|to the (?:left|right)|above|below|next to|in the background)",
]

# ============================================================
# Vietnamese → English translation maps
# ============================================================
_VI_TO_EN_COLORS = {
    "xanh lá": "green", "xanh lá cây": "green", "xanh dương": "blue",
    "xanh da trời": "sky blue", "xanh nước biển": "navy blue",
    "xanh": "blue",  # fallback for ambiguous "xanh"
    "đỏ": "red", "vàng": "yellow", "trắng": "white", "đen": "black",
    "tím": "purple", "hồng": "pink", "cam": "orange", "nâu": "brown",
    "xám": "gray", "bạc": "silver", "vàng kim": "gold", "be": "beige",
    "kem": "cream",
}

_VI_TO_EN_SCENES = {
    "ngoài trời": "outdoor", "trong nhà": "indoor", "sân khấu": "stage",
    "sân vận động": "stadium", "phòng họp": "conference room",
    "đường phố": "street", "họp báo": "press conference",
    "lễ trao giải": "award ceremony", "bản tin": "news broadcast",
    "phòng quay": "TV studio", "hội trường": "auditorium",
    "nhà thi đấu": "sports arena", "quảng trường": "public square",
}

_VI_TO_EN_SPATIAL = {
    "phía sau": "in the background", "phía trước": "in the foreground",
    "bên trái": "on the left side", "bên phải": "on the right side",
    "phía trên": "at the top", "phía dưới": "at the bottom",
    "bên cạnh": "next to", "góc trên trái": "top-left corner",
    "góc trên phải": "top-right corner", "góc dưới trái": "bottom-left corner",
    "trung tâm": "in the center", "giữa": "in the middle",
}

_VI_TO_EN_ACTIONS = {
    "đang phát biểu": "is speaking", "đang trình bày": "is presenting",
    "đang thi đấu": "is competing", "đang chạy": "is running",
    "đang nhảy": "is jumping", "đang bơi": "is swimming",
    "đứng": "standing", "ngồi": "sitting", "cầm": "holding",
    "mặc": "wearing", "đội": "wearing on head",
}

# ============================================================
# Answer type keywords
# ============================================================
_COUNT_KEYWORDS = ["bao nhiêu", "how many", "mấy", "số lượng", "đếm", "tổng số"]
_NAME_KEYWORDS = ["ai ", "who ", "tên ", "tên của", "name of", "người nào"]
_YES_NO_KEYWORDS = ["có không", "yes or no", "có phải", "is it", "có đúng", "phải không"]
_COLOR_KEYWORDS = ["màu gì", "màu sắc", "what color", "mặc màu", "màu áo"]


class QueryParser:
    """
    Converts raw query text into structured query objects.
    Optimized for CLIP cross-lingual retrieval (Vi→En expansion).

    Usage:
        parser = QueryParser()
        kis_query = parser.parse_kis("Tìm video người dẫn mặc áo đỏ...")
        qa_query  = parser.parse_qa("Trong video lễ trao giải...", "Có bao nhiêu người?")
    """

    def parse_kis(
        self,
        raw_text: str,
        top_k: int = 100,
    ) -> TextualKISQuery:
        """
        Parse a Textual KIS query with rich entity extraction.
        """
        text_lower = raw_text.lower()

        # --- Extract colors (compound-first to avoid partial match) ---
        colors: List[str] = []
        processed_text = text_lower
        for vi_compound, _ in _COLORS_VI_COMPOUND:
            if vi_compound in processed_text:
                colors.append(vi_compound)
                processed_text = processed_text.replace(vi_compound, "")
        # Then simple colors on what remains
        for vi_color in _COLORS_VI_SIMPLE:
            if vi_color in processed_text and vi_color not in [c.split()[0] for c in colors]:
                colors.append(vi_color)
        for en_color in _COLORS_EN:
            if re.search(rf"\b{en_color}\b", text_lower):
                colors.append(en_color)
        colors = list(dict.fromkeys(colors))  # deduplicate, preserve order

        # --- Extract objects ---
        objects: List[str] = []
        for obj_label, pattern in _OBJECT_PATTERNS.items():
            if re.search(pattern, raw_text, re.IGNORECASE):
                objects.append(obj_label)

        # --- Extract scene ---
        scene = ""
        for scene_label, keywords in _SCENE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                scene = scene_label
                break

        # --- Extract OCR hints ---
        ocr_hints: List[str] = []
        quoted = re.findall(r'["\']([\w\s]{2,})["\']', raw_text)
        ocr_hints.extend(quoted)
        # ALL CAPS words (logos, channel names like VTV1, VNPT...)
        caps_words = re.findall(r'\b[A-Z][A-Z0-9]{1,}\b', raw_text)
        ocr_hints.extend([w for w in caps_words if w not in ("TV", "HD", "OK", "AI")])
        ocr_hints = list(dict.fromkeys(ocr_hints))

        # --- Extract spatial hints ---
        spatial_hints: List[str] = []
        for pat in _SPATIAL_PATTERNS:
            matches = re.findall(pat, text_lower)
            for m in matches:
                hint = m if isinstance(m, str) else " ".join(m).strip()
                if hint:
                    spatial_hints.append(hint)
        spatial_hints = list(dict.fromkeys(spatial_hints))

        query = TextualKISQuery(
            raw_text=raw_text,
            parsed_objects=list(dict.fromkeys(objects)),
            parsed_scene=scene,
            parsed_colors=colors,
            ocr_keywords=ocr_hints,
            spatial_hints=spatial_hints,
            top_k=top_k,
        )

        logger.debug(
            f"[QueryParser] KIS: objects={query.parsed_objects}, "
            f"colors={query.parsed_colors}, scene='{query.parsed_scene}', "
            f"ocr={query.ocr_keywords}"
        )
        return query

    def parse_qa(
        self,
        event_description: str,
        question: str,
        answer_language: str = "auto",
        top_k: int = 20,
    ) -> QAQuery:
        """
        Parse a Q&A query with answer type inference.
        """
        answer_type = self.infer_answer_type(question)

        return QAQuery(
            event_description=event_description,
            question=question,
            answer_type=answer_type,
            answer_language=answer_language,
            top_k=top_k,
        )

    def infer_answer_type(self, question: str) -> str:
        """
        Infer answer type from question text.
        Returns: "count" | "color" | "name" | "yes_no" | "description"
        """
        q_lower = question.lower()

        if any(kw in q_lower for kw in _COUNT_KEYWORDS):
            return "count"
        elif any(kw in q_lower for kw in _COLOR_KEYWORDS):
            return "color"
        elif any(kw in q_lower for kw in _NAME_KEYWORDS):
            return "name"
        elif any(kw in q_lower for kw in _YES_NO_KEYWORDS):
            return "yes_no"
        else:
            return "description"

    # ============================================================
    # Build CLIP retrieval text (KIS)
    # ============================================================

    def build_retrieval_text(self, kis_query: "TextualKISQuery") -> str:
        """
        Build retrieval prompt for CLIP (v3 — English-First Strategy).

        Puts high-precision English visual terms FIRST to ensure they fall within
        CLIP's 77-token context limit, followed by concise raw Vietnamese text.
        """
        en_parts = []

        # 1. Scene translation (English)
        raw_lower = kis_query.raw_text.lower()
        if kis_query.parsed_scene:
            scene_en = kis_query.parsed_scene
            for vi, en in _VI_TO_EN_SCENES.items():
                if vi in raw_lower:
                    scene_en = en
                    break
            en_parts.append(scene_en)

        # 2. Action translation (English)
        for vi_action, en_action in _VI_TO_EN_ACTIONS.items():
            if vi_action in raw_lower:
                en_parts.append(en_action)

        # 3. Object labels (English)
        if kis_query.parsed_objects:
            en_parts.append(", ".join(kis_query.parsed_objects))

        # 4. Colors (English)
        color_en_terms = []
        for vi_color in kis_query.parsed_colors:
            en = _VI_TO_EN_COLORS.get(vi_color, vi_color)
            if en not in color_en_terms:
                color_en_terms.append(en)
        if color_en_terms:
            en_parts.append("wearing " + " and ".join(color_en_terms))

        # 5. Spatial hints (English)
        spatial_en = []
        for hint in kis_query.spatial_hints:
            translated = hint
            for vi, en in _VI_TO_EN_SPATIAL.items():
                if vi in hint:
                    translated = en
                    break
            spatial_en.append(translated)
        if spatial_en:
            en_parts.append(", ".join(spatial_en))

        # 6. OCR keywords
        if kis_query.ocr_keywords:
            en_parts.append("text: " + " ".join(kis_query.ocr_keywords))

        # Combine: English visual terms FIRST, followed by concise raw_text
        prefix = ". ".join(en_parts)
        if prefix:
            return f"{prefix}. {kis_query.raw_text}"
        return kis_query.raw_text

    # ============================================================
    # Build CLIP retrieval text (Q&A)
    # ============================================================

    def build_qa_retrieval_text(self, qa_query: "QAQuery") -> str:
        """
        Build retrieval text for Q&A queries.
        Combines event_description + question visual keywords for better recall.

        Strategy:
          - Start with event_description (the scene context)
          - Extract visual keywords from the question (colors, objects, etc.)
          - Append English translation for CLIP
        """
        # Parse the event description as a KIS query to get structured fields
        kis = self.parse_kis(qa_query.event_description, top_k=100)
        base_text = self.build_retrieval_text(kis)

        # Extract visual keywords from the question itself
        q_lower = qa_query.question.lower()
        question_visual_parts = []

        # Extract colors from question
        for vi_color, en_color in _VI_TO_EN_COLORS.items():
            if vi_color in q_lower:
                question_visual_parts.append(f"{en_color} color")
                break

        # Extract objects from question
        for obj_label, pattern in _OBJECT_PATTERNS.items():
            if re.search(pattern, qa_query.question, re.IGNORECASE):
                if obj_label not in base_text:
                    question_visual_parts.append(obj_label)

        if question_visual_parts:
            return base_text + ". " + ", ".join(question_visual_parts)
        return base_text
