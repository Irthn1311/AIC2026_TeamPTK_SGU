"""
Deep Entity Extractor — Extracts 12 types of visual entities from Vi/En queries.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# ── Color tables ──────────────────────────────────────────────────────────────
_COLOR_COMPOUND_VI: List[tuple] = [
    ("xanh lá cây", "green"), ("xanh lá", "green"), ("xanh dương", "blue"),
    ("xanh da trời", "sky blue"), ("xanh nước biển", "navy blue"),
    ("đỏ tươi", "bright red"), ("vàng kim", "gold"), ("đen tuyền", "jet black"),
    ("trắng tinh", "pure white"), ("hồng đậm", "deep pink"), ("cam đỏ", "orange red"),
    ("xanh rêu", "olive green"), ("xanh ngọc", "teal"),
]
_COLOR_SIMPLE_VI: Dict[str, str] = {
    "đỏ": "red", "xanh": "blue", "vàng": "yellow", "trắng": "white",
    "đen": "black", "tím": "purple", "hồng": "pink", "cam": "orange",
    "nâu": "brown", "xám": "gray", "bạc": "silver", "be": "beige", "kem": "cream",
}
_COLOR_EN = {
    "red", "blue", "green", "yellow", "white", "black", "purple", "pink",
    "orange", "brown", "grey", "gray", "silver", "gold", "navy", "teal",
    "turquoise", "beige", "cream", "olive",
}

# ── Clothing targets ──────────────────────────────────────────────────────────
_CLOTHING_TARGETS_VI: Dict[str, str] = {
    "áo vest": "suit jacket", "áo khoác": "jacket", "áo sơ mi": "dress shirt",
    "áo thun": "t-shirt", "áo phông": "t-shirt", "áo dài": "ao dai",
    "áo blouse": "blouse", "áo": "shirt/top", "quần tây": "dress pants",
    "quần jean": "jeans", "quần short": "shorts", "quần": "pants",
    "váy": "skirt/dress", "đầm": "dress", "vest": "suit",
    "cà vạt": "tie", "khăn quàng": "scarf", "mũ": "hat", "nón": "hat",
    "kính": "glasses", "kính mắt": "eyeglasses",
}

# ── Person roles ───────────────────────────────────────────────────────────────
_PERSON_ROLES: Dict[str, str] = {
    "phát thanh viên": "news anchor", "biên tập viên": "TV editor",
    "người dẫn chương trình": "TV host", "mc": "MC/host",
    "phóng viên": "reporter", "diễn giả": "speaker",
    "vận động viên": "athlete", "cầu thủ": "player",
    "quan chức": "official", "lãnh đạo": "official",
    "bộ trưởng": "minister", "chủ tịch": "chairman",
    "giáo viên": "teacher", "học sinh": "student",
    "đầu bếp": "chef", "bác sĩ": "doctor",
    "anchor": "news anchor", "reporter": "reporter",
    "athlete": "athlete", "player": "player", "official": "official",
    "speaker": "speaker", "host": "host",
}
_GENDER_VI: Dict[str, str] = {
    "phụ nữ": "female", "nữ": "female", "cô": "female", "bà": "female",
    "nam": "male", "đàn ông": "male", "ông": "male", "anh": "male",
    "bé gái": "girl", "bé trai": "boy", "trẻ em": "child",
    "woman": "female", "female": "female", "girl": "female",
    "man": "male", "male": "male", "boy": "male",
}

# ── Scene types ────────────────────────────────────────────────────────────────
_SCENE_MAP: Dict[str, tuple] = {
    # (scene_type_en, specific_vi_terms)
    "news_studio":      ("news studio",      ["bản tin", "thời sự", "tin tức", "phòng quay"]),
    "press_conference": ("press conference", ["họp báo", "press conference", "briefing"]),
    "ceremony":         ("award ceremony",   ["lễ trao giải", "trao giải", "khai mạc", "bế mạc"]),
    "stadium":          ("stadium",          ["sân vận động", "nhà thi đấu", "stadium", "arena"]),
    "classroom":        ("classroom",        ["lớp học", "trường học", "phòng học", "classroom"]),
    "kitchen":          ("kitchen",          ["nhà bếp", "bếp", "kitchen"]),
    "street":           ("street",           ["đường phố", "phố", "street", "road"]),
    "outdoor":          ("outdoor",          ["ngoài trời", "outdoor", "outside", "park", "beach"]),
    "indoor":           ("indoor",           ["trong nhà", "hội trường", "phòng họp", "indoor"]),
    "concert":          ("concert",          ["concert", "sân khấu âm nhạc", "biểu diễn"]),
    "hospital":         ("hospital",         ["bệnh viện", "phòng khám", "hospital", "clinic"]),
    "restaurant":       ("restaurant",       ["nhà hàng", "quán ăn", "restaurant", "cafe"]),
    "forest":           ("forest/woods",     ["cháy rừng", "rừng", "cánh rừng", "rừng cây", "forest", "woods"]),
    "mountain":         ("mountain/hill",    ["núi", "ngọn núi", "sườn đồi", "đồi", "mountain", "hill"]),
    "waterfront":       ("river/sea",        ["sông", "bờ sông", "biển", "bờ biển", "hồ", "river", "lake", "sea"]),
}

# ── Object patterns ────────────────────────────────────────────────────────────
_OBJECT_PATTERNS: Dict[str, str] = {
    "volcano":     r"\b(núi lửa|núi lửa phun|núi lửa đang phun|volcano|erupting volcano)\b",
    "fire":        r"\b(cháy|cháy rừng|lửa|ngọn lửa|đám cháy|hỏa hoạn|bốc cháy|fire|wildfire|flames)\b",
    "smoke":       r"\b(khói|khói mù|làn khói|khói dày|cột khói|smoke|smoke plume)\b",
    "interview":   r"\b(phỏng vấn|được phỏng vấn|trả lời phỏng vấn|interview|interviewed)\b",
    "straw_hat":   r"\b(mũ rơm|nón rơm|straw hat)\b",
    "products":    r"\b(sản phẩm|gói sản phẩm|hàng hóa|packaged products|products)\b",
    "certificate": r"\b(giấy chứng nhận|bằng khen|giấy khen|chứng nhận|certificate|diploma)\b",
    "patterned":   r"\b(họa tiết|áo họa tiết|hoa văn|patterned)\b",
    "hillside":    r"\b(sườn đồi|quả đồi|đồi|sườn núi|hillside|hill)\b",
    "mountain":    r"\b(núi|ngọn núi|dãy núi|mountain)\b",
    "forest":      r"\b(rừng|cánh rừng|rừng cây|thảm thực vật|forest|woods|trees)\b",
    "sky":         r"\b(bầu trời|trời|bầu trời đêm|nền trời|sky)\b",
    "water":       r"\b(sông|biển|hồ|nước|dòng sông|bờ biển|bãi biển|river|sea|ocean|lake|beach|seashore|water)\b",
    "flood":       r"\b(ngập|ngập lụt|lũ|lũ lụt|lũ quét|flood|flooding)\b",
    "vehicle":     r"\b(xe|ô tô|xe buýt|xe tải|xe máy|xe cứu hỏa|car|bus|truck|vehicle|fire truck|motorbike)\b",
    "aircraft":    r"\b(máy bay|máy bay trực thăng|trực thăng|máy bay cứu hỏa|airplane|plane|helicopter)\b",
    "ship":        r"\b(con thuyền|con tàu|thuyền|tàu|canô|ship|boat)\b",
    "building":    r"\b(tòa nhà|ngôi nhà|căn nhà|nhà|bệnh viện|trường học|building|house|structure)\b",
    "firefighter": r"\b(lính cứu hỏa|cảnh sát PCCC|cứu hỏa|firefighter)\b",
    "animal":      r"\b(chó|mèo|chim|cá|con vật|động vật|đàn cá|con trâu|con bò|dog|cat|bird|fish|animal)\b",
    "bridge":      r"\b(cây cầu|cầu|bridge)\b",
    "microphone":  r"\b(micro|microphone|mic|bục phát biểu|podium)\b",
    "flag":        r"\b(cờ|flag|banner|biểu ngữ)\b",
    "screen":      r"\b(màn hình|bảng|screen|board|sign|logo|biểu bảng)\b",
    "ball":        r"\b(bóng|ball|quả bóng)\b",
    "medal":       r"\b(huy chương|medal|cúp|cup|trophy|giải thưởng)\b",
    "crowd":       r"\b(đám đông|khán giả|cổ động viên|crowd|audience|fans)\b",
    "camera":      r"\b(máy quay|camera|máy ảnh)\b",
    "table":       r"\b(bàn|bàn làm việc|table|desk)\b",
    "chair":       r"\b(ghế|chair|seat)\b",
    "computer":    r"\b(máy tính|laptop|computer|màn hình máy tính)\b",
}

# ── Action / verb patterns ─────────────────────────────────────────────────────
_ACTION_MAP: Dict[str, str] = {
    "được phỏng vấn": "being interviewed", "trả lời phỏng vấn": "answering interview",
    "phỏng vấn": "interviewed", "đang phun": "erupting", "phun trào": "erupting",
    "đang gặt lúa": "harvesting rice", "gặt lúa": "harvesting rice",
    "đang phát biểu": "speaking", "đang trình trình bày": "presenting",
    "đang thi đấu": "competing", "đang chạy": "running",
    "đang nhảy": "jumping", "đang bơi": "swimming",
    "đang sút bóng": "kicking the ball", "đang cầm": "holding",
    "đang đứng": "standing", "đang ngồi": "sitting",
    "đang cười": "smiling/laughing", "đang khóc": "crying",
    "đang vẫy tay": "waving", "đang chỉ tay": "pointing",
    "đang đọc": "reading", "đang viết": "writing",
    "đứng": "standing", "ngồi": "sitting", "mặc": "wearing",
    "cầm": "holding", "đội": "wearing (on head)", "đeo": "wearing (accessory)",
    "speaking": "speaking", "running": "running", "jumping": "jumping",
    "holding": "holding", "standing": "standing", "sitting": "sitting",
    "wearing": "wearing", "walking": "walking", "pointing": "pointing",
}

# ── Temporal cues ──────────────────────────────────────────────────────────────
_TEMPORAL_CUES: List[tuple] = [
    ("đầu tiên", "first", 1), ("lúc đầu", "initially", 1),
    ("tiếp theo", "next", 2), ("sau đó", "after that", 2),
    ("rồi", "then", 2), ("then", "then", 2),
    ("cuối cùng", "finally", 3), ("kết thúc", "at the end", 3),
    ("finally", "finally", 3), ("first", "first", 1),
    ("before", "before", 1), ("after", "after", 2),
    ("simultaneously", "simultaneously", 2), ("đồng thời", "simultaneously", 2),
]

# ── Spatial patterns ───────────────────────────────────────────────────────────
_SPATIAL_MAP: Dict[str, str] = {
    "phía sau": "in the background", "phía trước": "in the foreground",
    "bên trái": "on the left side", "bên phải": "on the right side",
    "phía trên": "at the top", "phía dưới": "at the bottom",
    "bên cạnh": "next to", "góc trên trái": "top-left corner",
    "góc trên phải": "top-right corner", "góc dưới trái": "bottom-left corner",
    "góc dưới phải": "bottom-right corner", "trung tâm": "in the center",
    "giữa": "in the middle", "nền": "background",
    "background": "in the background", "foreground": "in the foreground",
    "left": "on the left", "right": "on the right",
    "top": "at the top", "bottom": "at the bottom",
    "center": "in the center", "middle": "in the middle",
}

# ── Emotion patterns ───────────────────────────────────────────────────────────
_EMOTIONS: Dict[str, str] = {
    "vui": "happy", "vui vẻ": "happy", "cười": "smiling",
    "buồn": "sad", "khóc": "crying", "tức giận": "angry",
    "ngạc nhiên": "surprised", "lo lắng": "worried",
    "hào hứng": "excited", "căng thẳng": "tense",
    "happy": "happy", "sad": "sad", "angry": "angry",
    "excited": "excited", "surprised": "surprised",
}

# ── Weather/Lighting ───────────────────────────────────────────────────────────
_LIGHTING: Dict[str, str] = {
    "ban ngày": "daytime", "ban đêm": "nighttime", "buổi sáng": "morning",
    "buổi tối": "evening", "nắng": "sunny", "mưa": "rainy",
    "trong nhà": "indoor lighting", "ánh sáng": "lit",
    "day": "daytime", "night": "nighttime", "morning": "morning",
    "sunny": "sunny", "rainy": "rainy", "dark": "dark",
}

# ── OCR indicator patterns ─────────────────────────────────────────────────────
_OCR_PATTERNS = [
    r'"([^"]{2,40})"',                    # "quoted text"
    r"'([^']{2,40})'",                    # 'quoted text'
    r"\b([A-Z][A-Z0-9]{1,})\b",          # ALL-CAPS acronyms: VTV1, HTV, VNPT
    r"\b(\d{1,2}:\d{2})\b",              # Time: 19:00
    r"\b(\d{1,2}[-/]\d{1,2})\b",         # Score: 2-1, 3/0
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b",  # Proper Names: Hà Nội
]
_OCR_EXCLUDED = {"TV", "HD", "OK", "AI", "ID", "VR", "AR", "PC", "IT"}

# ── Quantity patterns ──────────────────────────────────────────────────────────
_QUANTITY_NOUN_VI = [
    "người", "cầu thủ", "vận động viên", "chiếc", "cái", "con", "em",
    "đội", "lần", "bàn thắng", "điểm", "huy chương", "học sinh", "giáo viên",
]
_QUANTITY_NOUN_EN = [
    "people", "person", "player", "athlete", "team", "times", "goals",
    "points", "medals", "students", "teachers",
]


@dataclass
class ExtractedEntities:
    """All extracted entities from a query."""
    persons: List[Dict[str, Any]] = field(default_factory=list)
    colors: List[Dict[str, str]] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    quantities: List[Dict[str, Any]] = field(default_factory=list)
    scene_type: str = ""
    scene_specific: str = ""
    actions: List[Dict[str, str]] = field(default_factory=list)
    temporal_cues: List[Dict[str, Any]] = field(default_factory=list)
    ocr_hints: List[str] = field(default_factory=list)
    spatial: List[Dict[str, str]] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    lighting: str = ""
    clothing_details: List[Dict[str, str]] = field(default_factory=list)
    raw_text: str = ""
    language_mix: Dict[str, float] = field(default_factory=dict)


class DeepEntityExtractor:
    """
    Extracts 12 categories of visual entities from bilingual (Vi/En) queries.

    Usage:
        extractor = DeepEntityExtractor()
        entities = extractor.extract("3 cầu thủ mặc áo xanh dương đang sút bóng bên trái")
    """

    def extract(self, text: str, language_mix: Optional[Dict[str, float]] = None) -> ExtractedEntities:
        entities = ExtractedEntities(raw_text=text, language_mix=language_mix or {})
        tl = text.lower()

        entities.persons       = self._extract_persons(tl, text)
        entities.colors        = self._extract_colors(tl)
        entities.clothing_details = self._extract_clothing(tl)
        entities.objects       = self._extract_objects(text)
        entities.quantities    = self._extract_quantities(tl, text)
        st, ss                 = self._extract_scene(tl)
        entities.scene_type    = st
        entities.scene_specific = ss
        entities.actions       = self._extract_actions(tl)
        entities.temporal_cues = self._extract_temporal(tl)
        entities.ocr_hints     = self._extract_ocr(text)
        entities.spatial       = self._extract_spatial(tl)
        entities.emotions      = self._extract_emotions(tl)
        entities.lighting      = self._extract_lighting(tl)

        return entities

    # ── Persons ───────────────────────────────────────────────────────────────
    def _extract_persons(self, tl: str, text: str) -> List[Dict]:
        persons = []
        role_en, gender = "", ""

        for vi_role, en_role in _PERSON_ROLES.items():
            if vi_role in tl:
                role_en = en_role
                break

        for vi_gender, gender_en in _GENDER_VI.items():
            if vi_gender in tl:
                gender = gender_en
                break

        if role_en or gender:
            persons.append({"role_vi": "", "role_en": role_en, "gender": gender})

        return persons

    # ── Colors ────────────────────────────────────────────────────────────────
    def _extract_colors(self, tl: str) -> List[Dict]:
        colors = []
        processed = tl

        # Compound first
        for vi, en in _COLOR_COMPOUND_VI:
            if vi in processed:
                target = self._find_color_target(tl, vi)
                colors.append({"vi": vi, "en": en, "target": target})
                processed = processed.replace(vi, " ")

        # Then simple
        for vi, en in _COLOR_SIMPLE_VI.items():
            if re.search(r"(?<!\w)" + re.escape(vi) + r"(?!\w)", processed):
                if not any(c["vi"] == vi for c in colors):
                    target = self._find_color_target(tl, vi)
                    colors.append({"vi": vi, "en": en, "target": target})

        # English colors
        for en in _COLOR_EN:
            if re.search(r"\b" + en + r"\b", tl):
                if not any(c["en"] == en or c["vi"] == en for c in colors):
                    colors.append({"vi": en, "en": en, "target": "unspecified"})

        return colors

    def _find_color_target(self, tl: str, color_vi: str) -> str:
        """Find what object the color applies to (áo, quần, nền, ...)."""
        idx = tl.find(color_vi)
        if idx == -1:
            return "unspecified"
        # Look N chars before the color
        before = tl[max(0, idx - 20):idx]
        for vi_cloth, en_cloth in _CLOTHING_TARGETS_VI.items():
            if vi_cloth in before:
                return en_cloth
        if "nền" in before or "phông" in before or "background" in before:
            return "background"
        return "unspecified"

    # ── Clothing ──────────────────────────────────────────────────────────────
    def _extract_clothing(self, tl: str) -> List[Dict]:
        found = []
        for vi_cloth, en_cloth in _CLOTHING_TARGETS_VI.items():
            if vi_cloth in tl:
                # Find associated color
                color = ""
                idx = tl.find(vi_cloth)
                around = tl[max(0, idx-15):idx+15]
                for vi_c, en_c in _COLOR_SIMPLE_VI.items():
                    if vi_c in around:
                        color = en_c
                        break
                found.append({"vi": vi_cloth, "en": en_cloth, "color": color})
        return found

    # ── Objects ───────────────────────────────────────────────────────────────
    def _extract_objects(self, text: str) -> List[str]:
        found = []
        for label, pattern in _OBJECT_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                found.append(label)
        return found

    # ── Quantities ────────────────────────────────────────────────────────────
    def _extract_quantities(self, tl: str, text: str) -> List[Dict]:
        quantities = []
        all_nouns = _QUANTITY_NOUN_VI + _QUANTITY_NOUN_EN
        noun_pattern = "|".join(re.escape(n) for n in all_nouns)

        # Pattern: digit followed by count noun
        for m in re.finditer(r"\b(\d+)\s+(" + noun_pattern + r")\b", tl):
            quantities.append({"value": int(m.group(1)), "entity": m.group(2), "source": "digit"})

        # Pattern: count noun preceded by digit
        for m in re.finditer(r"\b(" + noun_pattern + r")\s+(\d+)\b", tl):
            quantities.append({"value": int(m.group(2)), "entity": m.group(1), "source": "digit_after"})

        return list({f"{q['value']}_{q['entity']}": q for q in quantities}.values())

    # ── Scene ─────────────────────────────────────────────────────────────────
    def _extract_scene(self, tl: str) -> tuple:
        for scene_key, (scene_en, keywords) in _SCENE_MAP.items():
            for kw in keywords:
                if kw in tl:
                    return scene_en, kw
        return "", ""

    # ── Actions ───────────────────────────────────────────────────────────────
    def _extract_actions(self, tl: str) -> List[Dict]:
        found = []
        sorted_actions = sorted(_ACTION_MAP.keys(), key=len, reverse=True)
        for vi_act in sorted_actions:
            if vi_act in tl:
                found.append({"vi": vi_act, "en": _ACTION_MAP[vi_act]})
        return found

    # ── Temporal cues ─────────────────────────────────────────────────────────
    def _extract_temporal(self, tl: str) -> List[Dict]:
        found = []
        for vi, en, order in _TEMPORAL_CUES:
            if vi in tl:
                found.append({"marker_vi": vi, "marker_en": en, "order": order})
        return sorted(found, key=lambda x: x["order"])

    # ── OCR hints ─────────────────────────────────────────────────────────────
    def _extract_ocr(self, text: str) -> List[str]:
        hints = []
        for pat in _OCR_PATTERNS:
            for m in re.finditer(pat, text):
                val = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
                val = val.strip()
                if val and val not in _OCR_EXCLUDED and len(val) >= 2:
                    hints.append(val)
        return list(dict.fromkeys(hints))

    # ── Spatial ───────────────────────────────────────────────────────────────
    def _extract_spatial(self, tl: str) -> List[Dict]:
        found = []
        sorted_spatial = sorted(_SPATIAL_MAP.keys(), key=len, reverse=True)
        for vi_sp in sorted_spatial:
            if vi_sp in tl:
                found.append({"vi": vi_sp, "en": _SPATIAL_MAP[vi_sp]})
        return found

    # ── Emotions ──────────────────────────────────────────────────────────────
    def _extract_emotions(self, tl: str) -> List[str]:
        found = []
        for vi_em, en_em in _EMOTIONS.items():
            if vi_em in tl:
                found.append(en_em)
        return list(dict.fromkeys(found))

    # ── Lighting ─────────────────────────────────────────────────────────────
    def _extract_lighting(self, tl: str) -> str:
        for vi_lt, en_lt in _LIGHTING.items():
            if vi_lt in tl:
                return en_lt
        return ""
