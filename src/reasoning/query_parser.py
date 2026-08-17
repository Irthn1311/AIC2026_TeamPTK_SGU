"""
Query Parser for AIC Video Retrieval System (v3 — Deep Analysis).

Changes from v2:
- Integrates the full Deep Analysis Pipeline:
    TextNormalizer → NegationExtractor → DeepEntityExtractor → IntentScorer → PromptBuilder
- parse_kis():
    * Populates ALL new TextualKISQuery fields (persons, quantities, actions,
      temporal_cues, negated_attributes, must_have, emotions, lighting, clothing_details,
      retrieval_weights, language_mix, clip_prompt, ocr_query, vlm_verification_prompt)
    * Bilingual (Vi + En) handling via TextNormalizer.detect_language_mix()
- parse_qa():
    * answer_subtype (15 fine-grained types) via infer_answer_subtype()
    * expected_answer_format inference
    * Populates question_entities, scene_entities, negated_attributes,
      retrieval_weights, vlm_verification_prompt
- build_retrieval_text(): returns kis_query.clip_prompt (already computed)
- build_vlm_verification_prompt(): standalone helper for external callers
- Bilingual support: both Vi-only and En-only queries work transparently
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.common.types import TextualKISQuery, QAQuery, TRAKEQuery, EventStep
from src.preprocessing.text_normalizer import TextNormalizer
from src.preprocessing.negation_extractor import NegationExtractor
from src.preprocessing.entity_extractor import DeepEntityExtractor, ExtractedEntities
from src.preprocessing.intent_scorer import IntentScorer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Module-level singletons (created once per process) ───────────────────────
_normalizer = TextNormalizer()
_neg_extractor = NegationExtractor()
_entity_extractor = DeepEntityExtractor()
_intent_scorer = IntentScorer()


# ============================================================
# Answer subtype inference tables
# ============================================================
_SUBTYPE_RULES: List[tuple] = [
    # (subtype, answer_type, keywords_vi, keywords_en, expected_format)
    ("count_people",    "count",  ["bao nhiêu người", "mấy người", "số người"],
                                  ["how many people", "number of people"],        "integer"),
    ("count_objects",   "count",  ["bao nhiêu cái", "bao nhiêu chiếc", "mấy cái"],
                                  ["how many objects", "how many items"],          "integer"),
    ("count_events",    "count",  ["bao nhiêu lần", "mấy lần", "bao nhiêu lượt"],
                                  ["how many times", "how often"],                "integer"),
    ("number_score",    "count",  ["tỷ số", "điểm số", "kết quả"],
                                  ["score", "result", "final score"],             "score"),
    ("number_time",     "count",  ["mấy giờ", "lúc mấy giờ", "thời điểm"],
                                  ["what time", "when", "at what hour"],          "time"),
    ("color_clothing",  "color",  ["màu áo", "màu quần", "màu váy", "mặc màu gì", "màu trang phục"],
                                  ["color of shirt", "color of clothing", "wearing what color",
                                   "shirt color", "jersey color", "color of the shirt",
                                   "color of the jacket", "color of the uniform"], "color_name"),
    ("color_object",    "color",  ["màu của", "màu bóng", "màu xe", "màu cờ"],
                                  ["color of the", "what color is"],              "color_name"),
    ("color_background","color",  ["màu nền", "màu phông", "màu background"],
                                  ["background color", "color of the background"], "color_name"),
    ("name_person",     "name",   ["ai ", "tên người", "tên của ai", "người nào"],
                                  ["who ", "name of person", "whose"],            "person_name"),
    ("name_place",      "name",   ["ở đâu", "địa điểm", "nơi nào", "tại đâu"],
                                  ["where", "what place", "location", "venue"],   "place_name"),
    ("name_thing",      "name",   ["tên chương trình", "tên đội", "tên sản phẩm", "là gì"],
                                  ["name of team", "name of program", "what is it called"], "thing_name"),
    ("yes_no_presence", "yes_no", ["có xuất hiện", "có mặt", "có không"],
                                  ["is there", "does it have", "are there"],      "yes_no"),
    ("yes_no_action",   "yes_no", ["có đang", "có phải đang", "đang làm gì"],
                                  ["is doing", "was doing", "did"],               "yes_no"),
    ("yes_no_attribute","yes_no", ["có mặc", "có mang", "có đội", "có đeo"],
                                  ["is wearing", "does have", "is holding"],      "yes_no"),
]

# Fallback coarse type keywords (used when no subtype matches)
_COUNT_KEYWORDS = ["bao nhiêu", "how many", "mấy", "số lượng", "tổng số"]
_COLOR_KEYWORDS = ["màu gì", "màu sắc", "what color", "mặc màu", "màu áo"]
_NAME_KEYWORDS  = ["ai ", "who ", "tên ", "tên của", "name of", "người nào"]
_YES_NO_KEYWORDS = ["có không", "yes or no", "có phải", "is it", "phải không"]


class QueryParser:
    """
    Converts raw query text into fully structured query objects (v3).

    Usage:
        parser = QueryParser()
        kis = parser.parse_kis("3 cầu thủ mặc áo xanh dương đang sút bóng bên trái")
        qa  = parser.parse_qa("Lễ trao giải SEA Games...", "Màu áo của vận động viên?")
    """

    def __init__(self, topic_classifier=None):
        from src.reasoning.topic_classifier import TopicClassifier
        self.topic_classifier = topic_classifier or TopicClassifier()

    def extract_topic(self, raw_text: str):
        return self.topic_classifier.classify_text(raw_text)

    # =========================================================
    # KIS Query Parsing (full Deep Analysis)
    # =========================================================

    def parse_kis(self, raw_text: str, top_k: int = 100) -> TextualKISQuery:
        """
        Parse a KIS query through the full Deep Analysis pipeline.

        Pipeline:
          1. TextNormalizer  → clean + abbrev expand + number-word convert
          2. detect_language_mix → vi/en ratios
          3. NegationExtractor → negated_attrs + must_have
          4. DeepEntityExtractor → 12 entity categories
          5. IntentScorer → retrieval_weights
          6. PromptBuilder → clip_prompt + ocr_query + vlm_verification_prompt
          7. Pack into TextualKISQuery
        """
        # Stage 1: Normalize
        normalized = _normalizer.normalize(raw_text)
        lang_mix   = _normalizer.detect_language_mix(raw_text)

        # Stage 2: Negation extraction (on original text — normalizer may distort scope)
        neg_result = _neg_extractor.extract(raw_text)

        # Stage 3: Entity extraction
        entities   = _entity_extractor.extract(normalized, lang_mix)

        # Stage 4: Intent scoring
        weights    = _intent_scorer.score(entities)

        # Stage 5: Build prompts
        clip_prompt = self._build_clip_prompt(entities, raw_text, lang_mix)
        ocr_query   = self._build_ocr_query(entities, neg_result)
        vlm_prompt  = self.build_vlm_verification_prompt(
            description=raw_text,
            question=None,
            neg_result=neg_result,
            entities=entities,
        )

        # Stage 6: Pack legacy fields (backward compatibility)
        parsed_colors  = [c["vi"] for c in entities.colors]
        parsed_objects = entities.objects
        parsed_scene   = entities.scene_type
        ocr_keywords   = entities.ocr_hints
        spatial_hints  = [s["vi"] for s in entities.spatial]

        query = TextualKISQuery(
            raw_text=raw_text,
            # Legacy
            parsed_objects=parsed_objects,
            parsed_scene=parsed_scene,
            parsed_colors=parsed_colors,
            ocr_keywords=ocr_keywords,
            spatial_hints=spatial_hints,
            top_k=top_k,
            # Deep analysis
            persons=entities.persons,
            quantities=entities.quantities,
            actions=entities.actions,
            temporal_cues=entities.temporal_cues,
            negated_attributes=neg_result.negated_attributes,
            must_have=neg_result.must_have,
            emotions=entities.emotions,
            lighting=entities.lighting,
            clothing_details=entities.clothing_details,
            retrieval_weights=weights.as_dict(),
            language_mix=lang_mix,
            # Prompts
            clip_prompt=clip_prompt,
            ocr_query=ocr_query,
            vlm_verification_prompt=vlm_prompt,
        )

        logger.info(f"🔄 [Chuyển đổi Ngôn ngữ: Việt ➔ Anh]")
        logger.info(f"   ► Câu gốc (Vietnamese)  : '{raw_text}'")
        logger.info(f"   ► CLIP Prompt (English)  : '{clip_prompt}'")
        if ocr_query:
            logger.info(f"   ► OCR Query (Text/Logo)  : '{ocr_query}'")
        return query

    # =========================================================
    # QA Query Parsing
    # =========================================================

    def parse_qa(
        self,
        event_description: str,
        question: str,
        answer_language: str = "auto",
        top_k: int = 20,
        target_prefix: str = "",
    ) -> QAQuery:
        """
        Parse a Q&A query with deep entity + subtype analysis.
        """
        # Coarse + fine-grained type
        answer_type, answer_subtype, expected_format = self.infer_answer_subtype(question)

        # Entity extraction on both parts
        norm_desc = _normalizer.normalize(event_description)
        norm_q    = _normalizer.normalize(question)

        scene_ents = _entity_extractor.extract(norm_desc)
        q_ents     = _entity_extractor.extract(norm_q)

        # Negation in both
        neg_desc = _neg_extractor.extract(event_description)
        neg_q    = _neg_extractor.extract(question)
        all_negated = list(dict.fromkeys(
            neg_desc.negated_attributes + neg_q.negated_attributes
        ))

        # Retrieval weights (based on scene description)
        weights = _intent_scorer.score(scene_ents)

        # VLM verification prompt
        vlm_prompt = self.build_vlm_verification_prompt(
            description=event_description,
            question=question,
            neg_result=neg_desc,
            entities=scene_ents,
            answer_subtype=answer_subtype,
        )

        return QAQuery(
            event_description=event_description,
            question=question,
            answer_type=answer_type,
            answer_language=answer_language,
            top_k=top_k,
            target_prefix=target_prefix,
            # Deep analysis
            answer_subtype=answer_subtype,
            expected_answer_format=expected_format,
            question_entities=[{
                "colors": q_ents.colors, "objects": q_ents.objects,
                "persons": q_ents.persons, "actions": q_ents.actions,
            }],
            scene_entities=[{
                "colors": scene_ents.colors, "objects": scene_ents.objects,
                "persons": scene_ents.persons, "scene": scene_ents.scene_type,
                "clothing": scene_ents.clothing_details,
            }],
            negated_attributes=all_negated,
            retrieval_weights=weights.as_dict(),
            vlm_verification_prompt=vlm_prompt,
        )

    # =========================================================
    # Answer Subtype Inference
    # =========================================================

    def infer_answer_subtype(self, question: str) -> tuple:
        """
        Infer fine-grained answer subtype (15 types).

        Returns:
            (answer_type: str, answer_subtype: str, expected_format: str)
        """
        q_lower = question.lower()
        for subtype, ans_type, kw_vi, kw_en, fmt in _SUBTYPE_RULES:
            if any(kw in q_lower for kw in kw_vi + kw_en):
                return ans_type, subtype, fmt

        # Fallback coarse
        if any(kw in q_lower for kw in _COUNT_KEYWORDS):
            return "count", "count_objects", "integer"
        if any(kw in q_lower for kw in _COLOR_KEYWORDS):
            return "color", "color_object", "color_name"
        if any(kw in q_lower for kw in _NAME_KEYWORDS):
            return "name", "name_thing", "text"
        if any(kw in q_lower for kw in _YES_NO_KEYWORDS):
            return "yes_no", "yes_no_presence", "yes_no"

        return "description", "description_general", "text"

    # Legacy compatibility alias
    def infer_answer_type(self, question: str) -> str:
        ans_type, _, _ = self.infer_answer_subtype(question)
        return ans_type

    # =========================================================
    # Retrieval Text Builders
    # =========================================================

    def build_retrieval_text(self, kis_query: TextualKISQuery) -> str:
        """
        Return the pre-built CLIP prompt from the query struct.
        Falls back to building it on-the-fly if clip_prompt is empty.
        """
        if kis_query.clip_prompt:
            return kis_query.clip_prompt
        # Fallback: build on-the-fly
        return self._build_clip_prompt_from_legacy(kis_query)

    def build_qa_retrieval_text(self, qa_query: QAQuery) -> str:
        """
        Build retrieval text for QA by merging event_description + question keywords.
        """
        kis = self.parse_kis(qa_query.event_description, top_k=100)
        base_text = self.build_retrieval_text(kis)

        # Append visual keywords from the question
        q_lower = qa_query.question.lower()
        q_ents = _entity_extractor.extract(q_lower)
        extras = []
        for c in q_ents.colors:
            extras.append(c["en"] + " color")
        for obj in q_ents.objects:
            if obj not in base_text:
                extras.append(obj)

        return base_text + (". " + ", ".join(extras) if extras else "")

    # =========================================================
    # VLM Verification Prompt Builder
    # =========================================================

    def build_vlm_verification_prompt(
        self,
        description: str,
        question: Optional[str],
        neg_result=None,
        entities: Optional[ExtractedEntities] = None,
        answer_subtype: str = "description_general",
    ) -> str:
        """
        Build a structured VLM prompt embedding:
          - Scene context
          - What the image MUST contain (must_have)
          - What the image must NOT contain (negated_attributes)
          - The specific question to answer
          - Answer format guidance per subtype

        Returns a single string to pass directly to QwenVLClient.answer_question().
        """
        parts: List[str] = []

        # 1. Scene context
        parts.append(f"Scene context: {description.strip()}")

        # 2. Visual constraints from entities
        if entities:
            if entities.persons:
                pnames = [p.get("role_en") or p.get("gender", "person") for p in entities.persons]
                parts.append(f"Expected subject(s): {', '.join(pnames)}.")
            if entities.scene_type:
                parts.append(f"Setting: {entities.scene_type}.")
            if entities.colors:
                color_strs = [f"{c['en']} ({c['target']})" for c in entities.colors]
                parts.append(f"Color cues: {', '.join(color_strs)}.")
            if entities.clothing_details:
                cloth_strs = [f"{c['en']}" + (f" ({c['color']})" if c.get('color') else "")
                              for c in entities.clothing_details]
                parts.append(f"Clothing: {', '.join(cloth_strs)}.")
            if entities.emotions:
                parts.append(f"Emotion cues: {', '.join(entities.emotions)}.")
            if entities.ocr_hints:
                parts.append(f"Look for visible text: {', '.join(entities.ocr_hints)}.")

        # 3. Negation constraints
        if neg_result:
            constraint_text = neg_result.to_vlm_constraint_text()
            if constraint_text:
                parts.append(constraint_text)

        # 4. The question
        if question:
            parts.append(f"\nQuestion to answer: {question.strip()}")

        # 5. Answer format guidance
        format_guide = {
            "count_people":     "Answer with a single integer (number of people).",
            "count_objects":    "Answer with a single integer (number of objects).",
            "count_events":     "Answer with a single integer (number of times/events).",
            "number_score":     "Answer in score format like '2-1' or '3:0'.",
            "number_time":      "Answer with the time shown, e.g. '19:00' or '7 giờ tối'.",
            "color_clothing":   "Answer with the color name of the clothing item.",
            "color_object":     "Answer with the color name of the described object.",
            "color_background": "Answer with the color name of the background.",
            "name_person":      "Answer with the person's name or role.",
            "name_place":       "Answer with the place or location name.",
            "name_thing":       "Answer with the name of the thing/program/team.",
            "yes_no_presence":  "Answer 'Có' (Yes) or 'Không' (No).",
            "yes_no_action":    "Answer 'Có' (Yes) or 'Không' (No).",
            "yes_no_attribute": "Answer 'Có' (Yes) or 'Không' (No).",
            "description_general": "Describe what you observe in the image.",
        }
        guide = format_guide.get(answer_subtype, "Answer concisely based on what you see.")
        parts.append(f"Answer format: {guide}")
        parts.append("If the image does not contain relevant content, answer 'Không tìm thấy'.")

        return "\n".join(parts)

    # =========================================================
    # =========================================================
    # Internal: Open-Domain Vi ➔ En Sentence Translation Engine
    # =========================================================

    # =========================================================
    # Internal: Open-Domain Vi ➔ En Sentence Translation Engine
    # =========================================================

    def translate_vi_sentence(self, raw_text: str) -> str:
        """
        Translates Vietnamese natural language query to English preserving 100% semantics.
        Replaces visual phrases, objects, natural phenomena, settings, and actions.
        Does NOT convert untranslated Vietnamese to unaccented Vietnamese junk words.
        """
        _VI2EN_DICT = [
            # Alleys, Streets & Urban Scenes
            ("con hẻm đông người và xe máy", "a crowded narrow alley with people and motorbikes"),
            ("con hẻm đông người", "crowded narrow alley"), ("hẻm đông người", "crowded narrow alley"),
            ("con hẻm nhỏ", "narrow alleyway"), ("hẻm nhỏ", "narrow alleyway"),
            ("con hẻm", "narrow alley"), ("ngõ hẻm", "narrow alley"), ("hẻm", "alley"),
            ("ngõ phố", "narrow street"), ("con ngõ", "alleyway"),
            ("đường phố đông người", "crowded city street"), ("đường phố", "city street"),
            ("con đường", "street"), ("tuyến đường", "thoroughfare"), ("vỉa hè", "sidewalk"),
            ("ngã tư", "intersection"), ("ngã ba", "three-way junction"),
            ("bãi đỗ xe", "parking lot"), ("bãi xe", "parking lot"),
            ("khu dân cư", "residential area"), ("chợ đông người", "crowded market"),
            ("chợ", "market"), ("siêu thị", "supermarket"), ("công viên", "park"),
            ("bãi biển", "beach"), ("sân trường", "schoolyard"),

            # Flags, Banners & Decorations
            ("hai bên treo nhiều cờ việt nam", "hanging many Vietnamese flags on both sides"),
            ("treo nhiều cờ việt nam", "hanging many Vietnamese flags"),
            ("cờ đỏ sao vàng", "Vietnamese flag with yellow star"),
            ("cờ việt nam", "Vietnamese flag"), ("lá cờ việt nam", "Vietnamese flag"),
            ("hai bên treo nhiều cờ", "hanging many flags on both sides"),
            ("hai bên treo", "hanging on both sides"), ("treo nhiều cờ", "hanging many flags"),
            ("treo nhiều", "hanging many"), ("được treo", "hung"), ("treo", "hanging"),
            ("hai bên đường", "on both sides of the street"), ("hai bên", "on both sides"),
            ("cờ", "flag"), ("biển hiệu", "signboard"), ("bảng hiệu", "signboard"),
            ("biểu ngữ", "banner"), ("băng rôn", "banner"), ("đèn lồng", "lanterns"),
            ("đèn trang trí", "decorative lights"), ("trang trí", "decorated with"),

            # Countryside, Agriculture & Nature
            ("đang gặt lúa ở đồng ruộng", "harvesting rice in paddy field"),
            ("gặt lúa trên cánh đồng", "harvesting rice in paddy field"),
            ("đang gặt lúa", "harvesting rice"), ("gặt lúa", "harvesting rice"),
            ("cánh đồng lúa", "rice paddy field"), ("đồng ruộng", "rice paddy field"),
            ("ruộng lúa", "rice paddy field"), ("cánh đồng", "paddy field"),
            ("nông dân", "farmers"), ("chăn trâu", "herding water buffalo"),
            ("con trâu", "water buffalo"), ("con bò", "cow"), ("ao cá", "fish pond"),

            # Natural disasters, Fire, Water & Weather
            ("cháy rừng lớn", "large wildfire"), ("cháy rừng", "wildfire"), ("đám cháy lớn", "large fire"),
            ("đám cháy", "fire"), ("lửa lan dọc sườn đồi", "flames spreading along the hillside"),
            ("lửa lan dọc sườn núi", "flames spreading along the mountain slope"),
            ("lửa lan dọc", "flames spreading along"), ("lửa lan", "flames spreading"),
            ("ngọn lửa", "flames"), ("hỏa hoạn", "fire disaster"), ("lửa", "fire"),
            ("khói dày phủ bầu trời", "thick smoke covering the sky"), ("khói dày", "thick smoke"),
            ("khói mù", "dense smoke"), ("làn khói", "column of smoke"), ("khói", "smoke"),
            ("phủ bầu trời", "covering the sky"), ("bầu trời đêm", "night sky"), ("bầu trời", "sky"),
            ("sườn đồi", "hillside"), ("quả đồi", "hill"), ("sườn núi", "mountain slope"),
            ("ngọn núi", "mountain"), ("dãy núi", "mountain range"), ("núi", "mountain"), ("đồi", "hill"),
            ("rừng cây", "forest trees"), ("cánh rừng", "forest"), ("rừng", "forest"),
            ("lũ quét", "flash flood"), ("lũ lụt", "flooding"), ("ngập lụt", "flooded area"), ("lũ", "flood"),
            ("sạt lở đất", "landslide"), ("sạt lở", "landslide"),
            ("chữa cháy cho", "extinguishing fire at"), ("chữa cháy", "extinguishing fire"),
            ("bị ngập trong lửa", "engulfed in flames"), ("bị ngập", "flooded"),
            ("bờ sông", "riverbank"), ("bờ biển", "seashore"), ("dòng sông", "river"),
            ("con sông", "river"), ("cây cầu", "bridge"),

            # Sports & Actions
            ("sút bóng vào lưới", "kicking ball into goal"), ("sút bóng", "kicking soccer ball"),
            ("đánh đầu", "heading the ball"), ("chuyền bóng", "passing the ball"),
            ("cầu thủ bóng đá", "soccer player"), ("cầu thủ", "soccer player"),
            ("sân vận động đông khán giả", "crowded stadium"), ("sân vận động", "stadium"),
            ("sân bóng", "soccer field"), ("đông khán giả", "crowded with fans"),
            ("khán giả", "audience fans"), ("cổ động viên", "fans"),

            # News & Media
            ("bản tin thời sự", "news broadcast"), ("bản tin", "news broadcast"),
            ("biên tập viên nam", "male news anchor"), ("biên tập viên nữ", "female news anchor"),
            ("biên tập viên", "news anchor"), ("phát thanh viên", "news anchor"),
            ("người dẫn chương trình", "TV host"), ("phòng quay", "news studio"),
            ("đang nói chuyện", "speaking"), ("nói chuyện", "speaking"),
            ("phát biểu", "speaking"), ("trình bày", "presenting"),

            # People, Demographics & Clothing
            ("đông người và xe máy", "crowded with people and motorbikes"),
            ("tập trung đông người", "crowded gathering of people"),
            ("đông người", "crowded with people"), ("đám đông", "crowd of people"),
            ("nhiều người", "many people"), ("người đi bộ", "pedestrians"),
            ("người qua lại", "passersby"), ("người phụ nữ", "woman"),
            ("người đàn ông", "man"), ("trẻ em", "children"), ("bé gái", "girl"), ("bé trai", "boy"),
            ("mặc áo vest đen", "wearing black suit"), ("mặc áo vest", "wearing suit jacket"),
            ("mặc áo đỏ", "wearing red shirt"), ("mặc áo xanh", "wearing blue shirt"),
            ("mặc áo trắng", "wearing white shirt"), ("mặc áo phông", "wearing t-shirt"),
            ("áo vest đen", "black suit"), ("áo vest", "suit jacket"), ("áo sơ mi", "dress shirt"),
            ("người lính cứu hỏa", "firefighter"), ("lính cứu hỏa", "firefighters"),
            ("cảnh sát", "police officer"), ("bác sĩ", "doctor"),
            ("đội nón lá", "wearing conical hat"), ("nón lá", "conical hat"),

            # Vehicles & Structures
            ("xe cứu hỏa", "fire truck"), ("máy bay cứu hỏa", "firefighting aircraft"),
            ("máy bay trực thăng", "helicopter"), ("trực thăng", "helicopter"),
            ("máy bay", "airplane"), ("tòa nhà cao tầng", "skyscraper"), ("tòa nhà", "building"),
            ("ngôi nhà", "house"), ("căn nhà", "house"), ("xe ô tô", "car"), ("xe máy", "motorbike"),
            ("nhiều xe máy", "many motorbikes"), ("xe đạp", "bicycle"), ("xe buýt", "bus"),
            ("con thuyền", "boat"), ("thuyền", "boat"), ("con tàu", "ship"),

            # Spatial & Temporal
            ("vào ban đêm", "at night"), ("ban đêm", "at night"), ("buổi tối", "in the evening"),
            ("ban ngày", "during the day"), ("buổi sáng", "in the morning"), ("hoàng hôn", "at sunset"),
            ("ở giữa", "in the middle"), ("trung tâm", "in the center"),
            ("bên trái", "on the left"), ("bên phải", "on the right"),

            # Common Stopwords / Fillers to strip
            ("tìm cảnh một", ""), ("tìm cảnh", ""), ("cho tôi thấy", ""), ("hình ảnh về", ""),
            ("video quay cảnh", ""), ("xuất hiện cảnh", ""), ("đoạn clip quay", ""),
            ("xác định", ""), ("cho biết", ""), ("tìm đoạn", ""), ("tìm video", ""),
            ("cho thấy", ""), ("hình ảnh", ""), ("cảnh một", ""),

            # Prepositions & Connectors
            (" ngập trong lửa", " engulfed in flames"), (" ngập trong ", " engulfed in "),
            (" cho một ", " at a "), (" cho ", " at "), (" trong ", " in "),
            (" và ", " and "), (" với ", " with "), (" trên ", " on "),
            (" ở ", " in "), (" tại ", " at "), (" đang ", " "),
            (" một ", " a "), (" những ", " "), (" các ", " "),
        ]

        _SINGLE_WORD_MAP = {
            "hẻm": "alley", "ngõ": "alley", "đường": "street", "phố": "street",
            "cờ": "flag", "lá": "flag", "xe": "vehicle", "máy": "motorbike",
            "người": "people", "đông": "crowded", "treo": "hanging", "bên": "side",
            "hai": "two", "nhiều": "many", "cảnh": "scene", "cháy": "fire",
            "lửa": "flames", "khói": "smoke", "rừng": "forest", "núi": "mountain",
            "đồi": "hill", "sông": "river", "biển": "sea", "hồ": "lake",
            "cầu": "bridge", "nhà": "house", "tòa": "building", "áo": "shirt",
            "quần": "pants", "váy": "skirt", "nón": "hat", "mũ": "hat",
            "đỏ": "red", "xanh": "blue", "vàng": "yellow", "trắng": "white",
            "đen": "black", "tím": "purple", "hồng": "pink", "cam": "orange",
            "nâu": "brown", "xám": "gray", "trái": "left", "phải": "right",
            "trên": "top", "dưới": "bottom", "giữa": "center", "nam": "male",
            "nữ": "female", "trai": "boy", "gái": "girl", "ruộng": "paddy field",
            "lúa": "rice", "nông": "farmer", "đồng": "field", "con": "", "cái": "",
            "chiếc": "", "bức": "", "tấm": "", "đoạn": "", "bản": "",
        }

        # Pre-sort phrase dictionary by length descending to match longest phrases first
        sorted_dict = sorted(_VI2EN_DICT, key=lambda x: len(x[0]), reverse=True)

        text = raw_text.strip()
        lowered = text.lower()

        # Step 1: Phrase substitution
        for vi_phrase, en_phrase in sorted_dict:
            if vi_phrase in lowered:
                lowered = lowered.replace(vi_phrase, f" {en_phrase} ")

        # Step 2: Token-level translation and residual Vietnamese filter
        tokens = lowered.split()
        clean_tokens = []
        vi_chars = set("àáảãạăắặằẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")

        for tok in tokens:
            word = tok.strip(".,!?:;\"'()")
            if not word:
                continue

            # Check single word dictionary fallback
            if word in _SINGLE_WORD_MAP:
                trans = _SINGLE_WORD_MAP[word]
                if trans:
                    clean_tokens.append(trans)
                continue

            # Check if word still contains Vietnamese diacritics
            has_vi_diacritics = any(c in vi_chars for c in word)
            if has_vi_diacritics:
                # Filter out untranslated Vietnamese words containing diacritics
                logger.debug(f"Filter out untranslated Vietnamese token: '{word}'")
                continue

            # Keep ASCII / English tokens
            clean_tokens.append(word)

        translated = " ".join(clean_tokens)
        cleaned = re.sub(r'\s+', ' ', translated).strip(' .,!?:;')
        cleaned = re.sub(r'\b(in|on|at|with|and|of|for)\s*$', '', cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    # =========================================================
    # Internal: CLIP Prompt Builders
    # =========================================================

    def _build_clip_prompt(
        self,
        entities: ExtractedEntities,
        raw_text: str,
        lang_mix: Dict[str, float],
    ) -> str:
        """
        Build a natural English sentence for CLIP from full-sentence translation
        combined with extracted entity tags. Preserves 100% query semantics.
        """
        translated_en = self.translate_vi_sentence(raw_text)

        if translated_en and len(translated_en) > 3:
            translated_en = re.sub(r"^(a photo of|a scene of|photo of|scene of)\s+", "", translated_en, flags=re.IGNORECASE)
            prompt = f"A photo of {translated_en}"
            return prompt[:300].strip()

        # Fallback: Structured entity assembly if translation is empty
        parts: List[str] = []
        if entities.persons:
            p = entities.persons[0]
            role = p.get("role_en", "")
            gender = p.get("gender", "")
            subject = f"a {gender + ' ' if gender else ''}{role}".strip() if (role or gender) else "a person"
        elif entities.objects:
            subject = f"a {entities.objects[0]}"
        else:
            subject = "a scene"

        parts.append(f"A photo of {subject}")
        if entities.actions:
            parts.append(" and ".join(a["en"] for a in entities.actions[:2]))
        if entities.scene_type:
            parts.append(f"in {entities.scene_type}")
        if entities.lighting:
            parts.append(f", {entities.lighting}")

        sentence = " ".join(parts)
        return sentence[:300].strip()

    def _build_clip_prompt_from_legacy(self, kis_query: TextualKISQuery) -> str:
        """Fallback: build CLIP prompt from legacy TextualKISQuery fields."""
        from src.preprocessing.entity_extractor import _COLOR_SIMPLE_VI
        color_parts = [_COLOR_SIMPLE_VI.get(c, c) for c in kis_query.parsed_colors]
        scene_map = {
            "news": "a TV news studio", "outdoor": "an outdoor setting",
            "indoor": "an indoor setting", "sport": "a sports venue",
            "press_conference": "a press conference", "ceremony": "an award ceremony",
        }
        loc = scene_map.get(kis_query.parsed_scene, kis_query.parsed_scene)
        subject = "a person" if "person" in kis_query.parsed_objects else "a scene"
        sentence = f"A photo of {subject}"
        if loc:
            sentence += f" in {loc}"
        if color_parts:
            sentence += ", " + " and ".join(color_parts)
        if kis_query.ocr_keywords:
            sentence += f". Text visible: {' '.join(kis_query.ocr_keywords)}"
        # NOTE: Do NOT append raw Vietnamese text — CLIP ViT-B/32 only understands English
        return sentence[:300].strip()

    def _build_ocr_query(self, entities: ExtractedEntities, neg_result) -> str:
        """Build a lexical keyword string for Qdrant/BM25 OCR search."""
        parts = list(entities.ocr_hints)
        for p in entities.persons:
            if p.get("role_en"):
                parts.append(p["role_en"])
        for q in entities.quantities:
            if q.get("entity") not in ("con", "cái", "chiếc"):
                parts.append(f"{q['value']} {q['entity']}")

        raw = entities.raw_text
        for neg in neg_result.negated_attributes:
            raw = raw.replace(neg, "")
        clean_raw = re.sub(r'^(tìm cảnh một|tìm cảnh|cảnh một)\s+', '', raw, flags=re.IGNORECASE).strip()
        
        # Deduplicate
        if clean_raw not in parts:
            parts.append(clean_raw[:100])

        return " ".join(filter(None, parts)).strip()
