"""
ViEnTranslator — Multi-Tier Vietnamese → English Translation Engine (v2 Production).

Translation strategy (priority order):
  Tier 1: Gemini Flash (via google.genai) — if GEMINI_API_KEY env var is set.
           Produces the highest quality, context-aware, CLIP-optimized translations.
  Tier 2: Google Translate Free (via deep-translator) — no API key needed.
           Covers ~95% of cases accurately via the public Google Translate endpoint.
  Tier 3: Dictionary fallback — the legacy phrase-substitution approach.
           Used only when both Tier 1 and Tier 2 are unavailable (network issues, etc.)

All tiers are thread-safe and lazily initialized.
Results are cached (LRU, max 512 entries) to avoid redundant API calls.

Usage:
    from src.preprocessing.vi_en_translator import ViEnTranslator
    translator = ViEnTranslator()
    english = translator.translate("Hai người phụ nữ đang cho dê ăn trong trại")
    # → "Two women feeding goats on a farm"
"""
from __future__ import annotations

import os
import re
import functools
import threading
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Vietnamese diacritic character set ────────────────────────────────────────
_VI_CHARS = set("àáảãạăắặằẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")


def _has_vi_diacritics(text: str) -> bool:
    return any(c in _VI_CHARS for c in text.lower())


# ── LRU translation cache (shared across all instances) ───────────────────────
@functools.lru_cache(maxsize=512)
def _cached_translate(text: str, tier: str) -> Optional[str]:
    """Internal cached wrapper — keyed by (text, tier)."""
    return None  # Filled by actual translation logic below


class ViEnTranslator:
    """
    Production-grade Vietnamese → English translator with 3-tier fallback.

    Thread-safe: each tier is initialized lazily under a lock.
    """

    _instance_lock = threading.Lock()

    def __init__(self, prefer_tier: str = "auto"):
        """
        Args:
            prefer_tier: "gemini" | "google" | "dict" | "auto" (default)
                         "auto" → try Tier 1, fall back to Tier 2, then Tier 3.
        """
        self.prefer_tier = prefer_tier
        self._gemini_client = None
        self._gemini_ready = False
        self._google_ready = False
        self._init_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def translate(self, text: str) -> str:
        """
        Translate Vietnamese text to English for CLIP prompt generation.

        Returns:
            English string, cleaned and ready for use in CLIP prompts.
            Falls back gracefully through tiers if any tier fails.
        """
        if not text or not text.strip():
            return ""

        # Skip translation if already English
        if not _has_vi_diacritics(text) and self._is_mostly_english(text):
            return text.strip()

        # Check LRU cache first
        cache_key = text.strip()
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        result = None

        if self.prefer_tier in ("auto", "gemini"):
            result = self._translate_gemini(text)

        if not result and self.prefer_tier in ("auto", "google"):
            result = self._translate_google(text)

        if not result:
            result = self._translate_dict(text)

        if result:
            self._set_cache(cache_key, result)
        return result or text

    # ── Tier 1: Gemini Flash ──────────────────────────────────────────────────

    def _translate_gemini(self, text: str) -> Optional[str]:
        """Translate using Gemini Flash via google.genai SDK."""
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None  # No key configured — skip tier

        try:
            import google.genai as genai

            if self._gemini_client is None:
                with self._init_lock:
                    if self._gemini_client is None:
                        self._gemini_client = genai.Client(api_key=api_key)
                        logger.info("[ViEnTranslator] Gemini Tier 1 initialized.")

            prompt = (
                "You are a Vietnamese-to-English translator specialized in producing "
                "concise, visually descriptive English sentences for image retrieval. "
                "Translate the following Vietnamese text to English. "
                "Output ONLY the English translation — no explanations, no quotes, no markdown.\n\n"
                f"Vietnamese: {text.strip()}"
            )

            response = self._gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            translated = response.text.strip() if response.text else ""
            translated = self._clean_translation(translated)

            if translated and len(translated) > 3:
                logger.debug(f"[Gemini] '{text[:60]}' → '{translated[:80]}'")
                return translated

        except Exception as e:
            logger.warning(f"[ViEnTranslator] Gemini tier failed: {e}")

        return None

    # ── Tier 2: Google Translate (free, via deep-translator) ──────────────────

    def _translate_google(self, text: str) -> Optional[str]:
        """Translate using Google Translate free endpoint via deep-translator."""
        try:
            from deep_translator import GoogleTranslator

            # deep-translator is stateless, no init needed
            translated = GoogleTranslator(source="vi", target="en").translate(text.strip())
            translated = self._clean_translation(translated or "")

            if translated and len(translated) > 3:
                logger.debug(f"[GoogleFree] '{text[:60]}' → '{translated[:80]}'")
                return translated

        except Exception as e:
            logger.warning(f"[ViEnTranslator] Google free tier failed: {e}")

        return None

    # ── Tier 3: Dictionary Fallback ───────────────────────────────────────────

    def _translate_dict(self, text: str) -> str:
        """
        Legacy dictionary-based phrase substitution (v6 Master Matrix).
        Guaranteed to return something (drops untranslated Vi tokens).
        """
        _VI2EN_DICT = [
            # Natural Phenomena, Volcanoes, Fires & Water
            ("cột nước trắng phun mạnh thẳng lên từ mặt đất", "white water jet spraying strongly straight up from ground"),
            ("cột nước trắng phun mạnh", "strong white water jet spraying"),
            ("cột nước trắng", "white water column"),
            ("phun mạnh thẳng lên từ mặt đất", "spraying strongly straight up from ground"),
            ("núi lửa đang phun", "erupting volcano"),
            ("núi lửa phun trào", "erupting volcano"),
            ("núi lửa", "volcano"),
            ("cột khói rất lớn", "massive column of smoke"),
            ("cột khói lớn", "large smoke plume"),
            ("cột khói", "column of smoke"),
            ("bầu trời xanh", "blue sky"),
            ("bầu trời đêm", "night sky"),
            ("bầu trời", "sky"),
            ("cháy rừng lớn", "large wildfire"),
            ("cháy rừng", "wildfire"),
            ("đám cháy lớn", "large fire"),
            ("đám cháy", "fire"),
            ("ngọn lửa", "flames"),
            ("hỏa hoạn", "fire disaster"),
            ("khói dày", "thick smoke"),
            ("khói mù", "dense smoke"),
            # Architecture
            ("tháp cổ bằng gạch", "ancient brick tower"),
            ("tháp cổ", "ancient tower"),
            ("đã xuống cấp", "dilapidated"),
            # Sports
            ("vận động viên", "athlete"),
            ("nhảy cao", "high jump"),
            ("điền kinh", "track and field athletics"),
            ("sân vận động", "stadium"),
            ("cầu thủ", "player"),
            # Animals
            ("con hà mã", "hippopotamus"),
            ("hà mã", "hippopotamus"),
            ("con hổ", "tiger"),
            ("đàn hổ", "tiger group"),
            ("hổ con", "tiger cub"),
            ("con dê", "goat"),
            ("đàn dê", "herd of goats"),
            ("cho dê ăn", "feeding goats"),
            ("con chó", "dog"),
            ("dắt chó", "walking dog"),
            # People & Roles
            ("phi hành gia", "astronaut"),
            ("phát thanh viên", "news anchor"),
            ("biên tập viên nam", "male news anchor"),
            ("biên tập viên nữ", "female news anchor"),
            ("biên tập viên", "news anchor"),
            ("người dẫn chương trình", "TV host"),
            ("phóng viên", "reporter"),
            ("nữ người dẫn chương trình", "female TV host"),
            ("người phụ nữ", "woman"),
            ("phụ nữ", "women"),
            ("đàn ông", "men"),
            ("bé gái", "girl"),
            ("bé trai", "boy"),
            ("trẻ em", "children"),
            # Space & Technology
            ("tàu vũ trụ tư nhân", "private spacecraft"),
            ("tàu vũ trụ", "spacecraft"),
            ("vũ trụ", "space"),
            ("phóng tàu vũ trụ", "spacecraft launch"),
            ("ánh sáng cực quang", "aurora borealis light"),
            ("cực quang", "aurora borealis"),
            ("vùng cực", "polar region"),
            # Clothing
            ("áo thun trắng", "white t-shirt"),
            ("áo thun đen", "black t-shirt"),
            ("áo thun", "t-shirt"),
            ("áo đỏ", "red shirt"),
            ("áo xanh dương", "blue shirt"),
            ("áo xanh lá", "green shirt"),
            ("áo dài tay kẻ sọc tím", "long-sleeved purple striped shirt"),
            ("áo dài tay kẻ sọc", "long-sleeved striped shirt"),
            ("áo dài tay", "long-sleeved shirt"),
            ("áo sọc", "striped shirt"),
            ("áo vest", "suit jacket"),
            ("áo khoác", "jacket"),
            ("áo sơ mi", "dress shirt"),
            ("áo", "shirt"),
            ("quàng áo đỏ trên vai", "red shirt draped over shoulder"),
            ("quàng trên vai", "draped over shoulder"),
            ("quần tây", "dress pants"),
            ("quần jean", "jeans"),
            ("quần short", "shorts"),
            ("quần", "pants"),
            ("váy", "skirt"),
            ("đầm", "dress"),
            ("mặc áo màu be", "wearing beige shirt"),
            ("mặc áo màu hồng nhạt", "wearing light pink shirt"),
            ("mặc áo đen", "wearing black shirt"),
            ("mặc áo trắng", "wearing white shirt"),
            ("mặc áo", "wearing"),
            ("đeo kính", "wearing glasses"),
            # Farm / Agriculture
            ("trại dê", "goat farm"),
            ("mái che bằng tôn", "metal roof canopy"),
            ("hàng rào gỗ", "wooden fence"),
            ("chuồng dê", "goat pen"),
            ("trại nuôi", "farm"),
            ("nuôi nhốt", "kept in enclosure"),
            ("được máy thu hoạch", "harvested by machinery"),
            ("thu hoạch", "harvesting"),
            ("máy nông nghiệp", "agricultural machinery"),
            ("cây trồng", "crops"),
            ("ruộng lúa", "rice paddy field"),
            ("đồng lúa", "rice paddy field"),
            # Locations & Scenes
            ("miền Nam", "southern region"),
            ("miền Bắc", "northern region"),
            ("khu vực nông thôn", "rural countryside"),
            ("địa phương", "local area"),
            ("thành phố", "city"),
            ("trường quay", "TV studio"),
            ("phòng quay", "news studio"),
            ("trong trường quay", "in news studio"),
            ("đứng một mình", "standing alone"),
            ("đường phố đông người", "crowded city street"),
            ("con hẻm nhỏ", "narrow alleyway"),
            ("con hẻm", "narrow alley"),
            ("hẻm", "alley"),
            ("hai bên treo nhiều cờ việt nam", "hanging many Vietnamese flags on both sides"),
            ("cờ việt nam", "Vietnamese flag"),
            ("cờ đỏ sao vàng", "Vietnamese flag with yellow star"),
            ("công viên", "park"),
            ("sân trường", "schoolyard"),
            # News & Media
            ("bản tin thời sự", "news broadcast"),
            ("bản tin", "news broadcast"),
            ("giới thiệu việc", "introduction to"),
            ("mẩu tin", "news story"),
            ("phóng sự", "news report"),
            ("đoạn clip", "video clip"),
            ("đoạn video", "video clip"),
            ("đang phát biểu", "speaking"),
            ("phát biểu tại bục", "speaking at podium"),
            ("bục phát biểu", "speech podium"),
            ("phỏng vấn", "interview"),
            ("được phỏng vấn", "being interviewed"),
            # Emotions & Expressions
            ("mỉm cười", "smiling"),
            ("tỏ vẻ thích thú", "appearing interested"),
            ("vui vẻ", "happy"),
            ("tươi cười", "smiling happily"),
            ("hào hứng", "enthusiastic"),
            # Colors
            ("xanh lá cây", "green"),
            ("xanh dương", "blue"),
            ("xanh da trời", "sky blue"),
            ("xanh nước biển", "navy blue"),
            ("xanh rêu", "olive green"),
            ("xanh ngọc", "teal"),
            ("xanh lá", "green"),
            ("đỏ tươi", "bright red"),
            ("vàng kim", "gold"),
            ("đen tuyền", "jet black"),
            ("trắng tinh", "pure white"),
            ("hồng đậm", "deep pink"),
            ("hồng nhạt", "light pink"),
            ("màu be", "beige"),
            ("đỏ", "red"),
            ("xanh", "blue"),
            ("vàng", "yellow"),
            ("trắng", "white"),
            ("đen", "black"),
            ("tím", "purple"),
            ("hồng", "pink"),
            ("cam", "orange"),
            ("nâu", "brown"),
            ("xám", "gray"),
            ("bạc", "silver"),
            # Numbers
            ("hai người", "two people"),
            ("ba người", "three people"),
            ("bốn người", "four people"),
            ("năm người", "five people"),
            ("một người", "one person"),
            # Structural filler words
            ("tìm cảnh một", ""),
            ("tìm cảnh", ""),
            ("tìm đoạn", ""),
            ("tìm video", ""),
            ("hình ảnh về", ""),
            ("đoạn clip cần tìm là cảnh", "scene where"),
            ("đoạn clip cần tìm", "the scene"),
            ("đây là phần giới thiệu", "this is an introduction to"),
            ("đây là", "this is"),
            ("đang", ""),
            ("và", "and"),
            ("với", "with"),
            ("trên", "on"),
            ("ở", "in"),
            ("tại", "at"),
            ("một", "a"),
            ("các", ""),
            ("có các", ""),
            ("có", "with"),
        ]

        _SINGLE_WORD_MAP = {
            "hẻm": "alley", "ngõ": "alley", "đường": "street", "phố": "street",
            "cờ": "flag", "xe": "vehicle", "máy": "motorbike",
            "người": "people", "đông": "crowded", "treo": "hanging", "bên": "side",
            "hai": "two", "nhiều": "many", "cháy": "fire",
            "lửa": "flames", "khói": "smoke", "rừng": "forest", "núi": "mountain",
            "đồi": "hill", "sông": "river", "biển": "sea", "hồ": "lake",
            "cầu": "bridge", "nhà": "house", "tòa": "building", "áo": "shirt",
            "quần": "pants", "váy": "skirt", "nón": "hat", "mũ": "hat",
            "đỏ": "red", "xanh": "blue", "vàng": "yellow", "trắng": "white",
            "đen": "black", "tím": "purple", "hồng": "pink", "cam": "orange",
            "nâu": "brown", "xám": "gray", "trái": "left", "phải": "right",
            "trên": "top", "dưới": "bottom", "giữa": "center",
            "nam": "male", "nữ": "female", "trai": "boy", "gái": "girl",
            "ruộng": "paddy field", "lúa": "rice", "đồng": "field", "đá": "rocks",
            "tháp": "tower", "gạch": "brick", "cổ": "ancient", "cây": "tree",
            "dốc": "ramp", "tường": "wall", "bục": "podium", "cá": "fish",
            "chậu": "basin", "thau": "basin",
            "nhảy": "jumping", "chạy": "running", "nệm": "mat",
            "con": "", "cái": "", "chiếc": "", "bức": "", "tấm": "", "đoạn": "", "bản": "",
        }

        _VI_UNACCENTED_LEAK_WORDS = {
            "phun", "sau", "phia", "tao", "duoc", "cung", "nhung", "cac", "nguoi",
            "vua", "qua", "mang", "cho", "lay", "xem", "lam", "ra", "vao", "theo",
            "nhieu", "hay", "voi", "va", "tren", "duoi", "trai", "phai", "giua",
            "ngoai", "trong", "tai", "den", "tu", "con", "cai", "chiec", "buc", "tam",
            "do", "dang", "rat", "mot", "co", "la", "vinh", "tuan", "lan", "di",
            "chuyen", "nguy", "hiem", "thu", "hoach", "ng", "vang",
        }

        sorted_dict = sorted(_VI2EN_DICT, key=lambda x: len(x[0]), reverse=True)
        lowered = text.strip().lower()

        for vi_phrase, en_phrase in sorted_dict:
            if vi_phrase in lowered:
                lowered = lowered.replace(vi_phrase, f" {en_phrase} ")

        tokens = lowered.split()
        clean_tokens = []
        for tok in tokens:
            word = tok.strip(".,!?:;\"'()")
            if not word:
                continue
            if word in _SINGLE_WORD_MAP:
                trans = _SINGLE_WORD_MAP[word]
                if trans:
                    clean_tokens.append(trans)
                continue
            if word in _VI_UNACCENTED_LEAK_WORDS:
                continue
            if any(c in _VI_CHARS for c in word):
                continue
            clean_tokens.append(word)

        translated = " ".join(clean_tokens)
        cleaned = re.sub(r'\s+', ' ', translated).strip(' .,!?:;')
        cleaned = re.sub(r'\b(in|on|at|with|and|of|for)\s*$', '', cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_mostly_english(text: str) -> bool:
        """Return True if >50% of words are pure ASCII alphabetic (English)."""
        words = text.split()
        if not words:
            return True
        en_count = sum(1 for w in words if w.isalpha() and w.isascii())
        return (en_count / len(words)) >= 0.5

    @staticmethod
    def _clean_translation(text: str) -> str:
        """Remove common LLM artifacts from translations."""
        text = text.strip()
        # Remove leading/trailing quotes
        text = re.sub(r'^["\']|["\']$', '', text).strip()
        # Remove "English:" prefix if model outputs it
        text = re.sub(r'^(English|Translation|Translated):\s*', '', text, flags=re.IGNORECASE).strip()
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ── Cache helpers ─────────────────────────────────────────────────────────
    _cache: dict = {}
    _cache_lock = threading.Lock()

    def _get_cache(self, key: str) -> Optional[str]:
        with self._cache_lock:
            return self._cache.get(key)

    def _set_cache(self, key: str, value: str) -> None:
        with self._cache_lock:
            if len(self._cache) >= 512:
                # Remove oldest entry
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = value


# ── Module-level singleton ─────────────────────────────────────────────────────
_default_translator: Optional[ViEnTranslator] = None
_singleton_lock = threading.Lock()


def get_translator() -> ViEnTranslator:
    """Get or create the module-level singleton translator."""
    global _default_translator
    if _default_translator is None:
        with _singleton_lock:
            if _default_translator is None:
                _default_translator = ViEnTranslator()
    return _default_translator
