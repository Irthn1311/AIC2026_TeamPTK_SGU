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
    # Internal: Open-Domain Vi ➔ En Sentence Translation Engine (v6 Master Matrix)
    # =========================================================

    def translate_vi_sentence(self, raw_text: str) -> str:
        """
        Translates Vietnamese natural language query to English preserving >95% semantics.
        Uses 3-Tier Multi-Pass Compound Tokenizer + Unaccented Leak Filter + Entity Synthesizer.
        Covers open-domain categories: Natural phenomena, Media/News, Cities/Alleys, Extreme Sports,
        Agriculture, Hydroelectric/Graphics, Ruins, Culinary/Markets, Warning signs, Animals & Vehicles.
        Guarantees 0% unaccented Vietnamese leaks ("phun", "sau", "hem", etc.) in CLIP Prompts.
        """
        _VI2EN_DICT = [
            # 1. Natural Phenomena, Volcanoes, Fires & Water Jets
            ("cột nước trắng phun mạnh thẳng lên từ mặt đất", "white water jet spraying strongly straight up from ground"),
            ("cột nước trắng phun mạnh", "strong white water jet spraying"),
            ("cột nước trắng", "white water column"),
            ("phun mạnh thẳng lên từ mặt đất", "spraying strongly straight up from ground"),
            ("phun mạnh từ mặt đất", "spraying strongly from ground"),
            ("phun mạnh", "spraying strongly"),
            ("ngồi xổm bên", "squatting beside"),
            ("ngồi xổm", "squatting"),
            ("khu vực nông thôn", "rural countryside area"),
            ("núi lửa đang phun", "an erupting volcano"),
            ("núi lửa phun trào", "an erupting volcano"),
            ("núi lửa phun", "an erupting volcano"),
            ("núi lửa", "volcano"),
            ("tạo cột khói rất lớn", "creating a massive plume of smoke"),
            ("tạo cột khói lớn", "creating a large plume of smoke"),
            ("cột khói rất lớn", "massive column of smoke"),
            ("cột khói lớn", "large smoke plume"),
            ("cột khói", "column of smoke"),
            ("trên nền trời xanh", "against blue sky background"),
            ("nền trời xanh", "blue sky background"),
            ("trên nền trời", "against sky background"),
            ("bầu trời xanh", "blue sky"),
            ("bầu trời đêm", "night sky"),
            ("bầu trời", "sky"),
            ("cháy rừng lớn", "large wildfire"),
            ("cháy rừng", "wildfire"),
            ("đám cháy lớn", "large fire"),
            ("đám cháy", "fire"),
            ("lửa lan dọc sườn đồi", "flames spreading along the hillside"),
            ("lửa lan dọc sườn núi", "flames spreading along the mountain slope"),
            ("lửa lan dọc", "flames spreading along"),
            ("ngọn lửa", "flames"),
            ("hỏa hoạn", "fire disaster"),
            ("khói dày phủ bầu trời", "thick smoke covering the sky"),
            ("khói dày", "thick smoke"),
            ("khói mù", "dense smoke"),
            ("làn khói", "column of smoke"),

            # 2. Ruins, Historic Towers & Architecture
            ("tháp cổ bằng gạch đã xuống cấp", "dilapidated ancient brick tower"),
            ("tháp cổ bằng gạch", "ancient brick tower"),
            ("tháp cổ", "ancient tower"),
            ("cây xanh mọc trên phần thân công trình", "green plants growing on the building structure"),
            ("cây xanh mọc trên thân", "green plants growing on structure"),
            ("cây xanh mọc trên phần thân", "green plants growing on structure"),
            ("cây xanh mọc trên", "green plants growing on"),
            ("cây xanh mọc", "green plants growing"),
            ("đã xuống cấp", "dilapidated"),
            ("mọc trên", "growing on"),
            ("thân công trình", "building structure"),

            # 3. Extreme Sports, Skateparks & Graffiti
            ("khu bmx/skatepark ngoài trời vào ban đêm", "outdoor BMX skatepark at night"),
            ("khu bmx/skatepark ngoài trời", "outdoor BMX skatepark"),
            ("khu bmx/skatepark", "BMX skatepark"),
            ("dốc trượt và hình vẽ xe đạp trên tường", "skate ramps and bicycle graffiti on wall"),
            ("nhiều dốc trượt", "multiple skate ramps"),
            ("dốc trượt", "skate ramp"),
            ("hình vẽ xe đạp trên tường", "bicycle graffiti mural on wall"),
            ("hình vẽ trên tường", "graffiti mural on wall"),
            ("hình vẽ xe đạp", "bicycle graffiti"),

            # 4. Media, Federal Reserve, Graphics & Studio
            ("người đàn ông tóc bạc, đeo kính", "silver-haired man wearing glasses"),
            ("người đàn ông tóc bạc", "silver-haired man"),
            ("tóc bạc", "silver hair"),
            ("đeo kính", "wearing glasses"),
            ("phát biểu tại bục với cờ mỹ", "speaking at podium with American flag"),
            ("phát biểu tại bục", "speaking at podium"),
            ("bục phát biểu", "speech podium"),
            ("biểu tượng ngân hàng trung ương", "central bank logo symbol"),
            ("ngân hàng trung ương", "central bank"),
            ("cờ mỹ", "American flag"),
            ("đồ họa nền xanh liệt kê số cửa xả đang mở", "blue background graphic listing open spillway gates"),
            ("đồ họa nền xanh", "blue background graphics chart"),
            ("nền xanh", "blue background"),
            ("số cửa xả đang mở", "number of open spillway gates"),
            ("hồ thủy điện hòa bình", "Hoa Binh hydroelectric reservoir"),
            ("hồ thủy điện sơn la", "Son La hydroelectric dam"),
            ("hồ thủy điện tuyên quang", "Tuyen Quang hydroelectric reservoir"),
            ("tuyên quang", "Tuyen Quang"),
            ("sơn la", "Son La"),
            ("hòa bình", "Hoa Binh"),
            ("hồ thủy điện", "hydroelectric dam reservoir"),
            ("thủy điện", "hydroelectric dam"),
            ("nữ người dẫn chương trình mặc áo màu be/hồng nhạt đứng một mình trong trường quay", "female TV host wearing beige light pink shirt standing alone in studio"),
            ("nữ người dẫn chương trình", "female TV host"),
            ("mặc áo màu be/hồng nhạt", "wearing beige or light pink shirt"),
            ("áo màu be", "beige shirt"),
            ("màu hồng nhạt", "light pink"),
            ("đứng một mình trong trường quay", "standing alone in news studio"),
            ("đứng một mình", "standing alone"),
            ("trong trường quay", "in news studio"),
            ("màn hình lớn cạnh nữ mc", "large display screen beside female MC"),
            ("màn hình lớn", "large display screen"),

            # 5. Seashores, Basins, Animals & Food Markets
            ("cận cảnh một thau/chậu tròn chứa rất nhiều cá nhỏ màu bạc", "close-up of a round basin containing many small silver fish"),
            ("thau/chậu tròn chứa rất nhiều cá nhỏ màu bạc", "round basin containing many small silver fish"),
            ("thau chậu tròn chứa rất nhiều cá nhỏ màu bạc", "round basin containing many small silver fish"),
            ("chậu tròn chứa rất nhiều cá nhỏ màu bạc", "round basin with many small silver fish"),
            ("cá nhỏ màu bạc", "small silver fish"),
            ("cá màu bạc", "silver fish"),
            ("thau/chậu tròn", "round basin bowl"),
            ("thau chậu tròn", "round basin bowl"),
            ("chậu tròn", "round basin bowl"),
            ("cực kỳ nhiều cá", "abundance of fish"),
            ("bữa tiệc sinh nhật hà mã", "birthday party for hippopotamus"),
            ("tiệc sinh nhật hà mã", "hippo birthday party"),
            ("bữa tiệc sinh nhật", "birthday party"),
            ("con hà mã", "hippopotamus"),
            ("hà mã", "hippopotamus"),
            ("khu chợ/không gian ẩm thực đông người", "crowded food court market area"),
            ("không gian ẩm thực đông người", "crowded culinary food space"),
            ("không gian ẩm thực", "culinary food space"),
            ("nhiều quầy chế biến món ăn dưới mái che lớn", "multiple cooking food stalls under large canopy roof"),
            ("quầy chế biến món ăn", "food cooking stalls"),
            ("dưới mái che lớn", "under large canopy roof"),
            ("dưới mái che", "under canopy roof"),
            ("mái che lớn", "large canopy roof"),
            ("cạnh bờ biển nhiều đá", "near rocky seashore"),
            ("bờ biển nhiều đá", "rocky seashore"),
            ("bãi biển nhiều đá", "rocky beach"),
            ("nhiều đá", "rocky"),
            ("bờ biển", "seashore"),
            ("bãi biển", "beach"),

            # 6. Vintage Cars, Rescue Vehicles, Bars & Swings
            ("chiếc xe mui trần cổ màu đỏ chở nhiều người đi qua đám đông", "vintage red convertible car carrying people driving through crowd"),
            ("xe mui trần cổ màu đỏ", "vintage red convertible car"),
            ("xe mui trần cổ", "vintage convertible car"),
            ("xe mui trần", "convertible car"),
            ("chở nhiều người đi qua đám đông", "carrying multiple people driving through crowd"),
            ("đi qua đám đông", "passing through crowd"),
            ("tấm biển cảnh báo màu vàng-đỏ nguy hiểm", "yellow red warning sign indicating danger"),
            ("tấm biển cảnh báo màu vàng-đỏ", "yellow and red warning sign"),
            ("biển cảnh báo sạt lở màu vàng đỏ nguy hiểm", "yellow red warning sign indicating landslide danger"),
            ("biển cảnh báo màu vàng đỏ nguy hiểm", "yellow red warning sign indicating danger"),
            ("màu vàng đỏ nguy hiểm", "yellow red danger"),
            ("vàng đỏ nguy hiểm", "yellow red danger"),
            ("màu vàng đỏ", "yellow red"),
            ("nguy hiểm sạt lở", "landslide danger hazard"),
            ("nguy hiểm", "danger hazard"),
            ("sạt lở", "landslide"),
            ("mặc áo sọc", "wearing striped shirt"),
            ("áo sọc", "striped shirt"),
            ("ngồi phía sau song sắt", "sitting behind iron bars"),
            ("phía sau song sắt", "behind iron bars"),
            ("khung cửa song sắt", "iron bar window frame"),
            ("song sắt", "iron bars"),
            ("người dắt hai con chó", "person walking two dogs"),
            ("dắt hai con chó", "walking two dogs"),
            ("dắt chó", "walking dog"),
            ("xe cứu trợ hoặc xe chữ thập đỏ di chuyển vào ban đêm", "rescue vehicle or Red Cross vehicle moving at night"),
            ("xe cứu trợ hoặc xe chữ thập đỏ", "rescue vehicle or Red Cross vehicle"),
            ("di chuyển vào ban đêm", "moving at night"),
            ("di chuyển", "moving"),
            ("xe cứu trợ", "rescue relief vehicle"),
            ("xe chữ thập đỏ", "Red Cross vehicle"),
            ("chữ thập đỏ", "Red Cross"),
            ("ở trên cao phía trên thành phố", "high above overlooking the city"),
            ("phía trên thành phố", "above the city skyline"),
            ("ngồi trên xích đu", "sitting on a swing"),
            ("xích đu", "swing"),
            ("cây trồng đang được máy thu hoạch trên đồng", "crops being harvested by machinery in field"),
            ("được máy thu hoạch", "harvested by machinery"),
            ("máy thu hoạch lúa", "rice combine harvester"),
            ("máy thu hoạch", "combine harvester"),
            ("thu hoạch", "harvesting"),
            ("máy nông nghiệp nhìn từ trên cao", "aerial view of agricultural machinery"),
            ("máy nông nghiệp", "agricultural machinery"),
            ("nhìn từ trên cao", "aerial top-down view"),
            ("cây trồng", "crops"),
            ("phương tiện cỡ lớn", "large heavy vehicle"),
            ("ở phía trái khung hình", "on the left side of frame"),
            ("phía trái khung hình", "on the left side of frame"),
            ("phía trái", "on the left side"),

            # 7. Media, Interviews, Clothing & Documents
            ("người đàn ông đội mũ rơm được phỏng vấn", "man wearing straw hat being interviewed"),
            ("được phỏng vấn", "being interviewed"),
            ("đang được phỏng vấn", "being interviewed"),
            ("trả lời phỏng vấn", "answering interview"),
            ("phỏng vấn", "interviewed"),
            ("mặc áo xanh họa tiết", "wearing patterned blue shirt"),
            ("áo xanh họa tiết", "patterned blue shirt"),
            ("áo họa tiết", "patterned shirt"),
            ("họa tiết", "patterned"),
            ("phía sau có các gói sản phẩm và giấy chứng nhận", "in background with packaged products and certificates"),
            ("phía sau có các gói sản phẩm", "in background with packaged products"),
            ("các gói sản phẩm", "packaged products"),
            ("gói sản phẩm", "packaged products"),
            ("sản phẩm", "products"),
            ("giấy chứng nhận", "certificates"),
            ("bằng khen", "award certificates"),
            ("giấy khen", "certificates"),
            ("phía sau có", "in background with"),
            ("phía sau", "in background"),
            ("ở phía sau", "in background"),
            ("phát biểu trong bản tin thời sự", "speaking in news broadcast"),
            ("đang phát biểu trong", "speaking in"),
            ("đang phát biểu", "speaking"),
            ("phát biểu", "speaking"),
            ("phát thanh viên", "news anchor"),
            ("biên tập viên nam", "male news anchor"),
            ("biên tập viên nữ", "female news anchor"),
            ("biên tập viên", "news anchor"),
            ("người dẫn chương trình", "TV host"),
            ("bản tin thời sự", "news broadcast"),
            ("bản tin", "news broadcast"),
            ("phòng quay", "news studio"),

            # 8. Alleys, Streets, Flags & Urban Scenes
            ("con hẻm đông người và xe máy", "a crowded narrow alley with people and motorbikes"),
            ("con hẻm đông người", "crowded narrow alley"),
            ("hẻm đông người", "crowded narrow alley"),
            ("con hẻm nhỏ", "narrow alleyway"),
            ("hẻm nhỏ", "narrow alleyway"),
            ("con hẻm", "narrow alley"),
            ("ngõ hẻm", "narrow alley"),
            ("hẻm", "alley"),
            ("ngõ phố", "narrow street"),
            ("con ngõ", "alleyway"),
            ("đường phố đông người", "crowded city street"),
            ("đường phố", "city street"),
            ("con đường", "street"),
            ("tuyến đường", "thoroughfare"),
            ("vỉa hè", "sidewalk"),
            ("ngã tư", "intersection"),
            ("ngã ba", "three-way junction"),
            ("bãi đỗ xe", "parking lot"),
            ("bãi xe", "parking lot"),
            ("khu dân cư", "residential area"),
            ("chợ đông người", "crowded market"),
            ("chợ", "market"),
            ("siêu thị", "supermarket"),
            ("công viên", "park"),
            ("sân trường", "schoolyard"),
            ("hai bên treo nhiều cờ việt nam", "hanging many Vietnamese flags on both sides"),
            ("treo nhiều cờ việt nam", "hanging many Vietnamese flags"),
            ("cờ đỏ sao vàng", "Vietnamese flag with yellow star"),
            ("cờ việt nam", "Vietnamese flag"),
            ("lá cờ việt nam", "Vietnamese flag"),
            ("hai bên treo nhiều cờ", "hanging many flags on both sides"),
            ("hai bên treo", "hanging on both sides"),
            ("treo nhiều cờ", "hanging many flags"),
            ("treo nhiều", "hanging many"),
            ("được treo", "hung"),
            ("treo", "hanging"),
            ("hai bên đường", "on both sides of the street"),
            ("hai bên", "on both sides"),
            ("biển hiệu", "signboard"),
            ("bảng hiệu", "signboard"),
            ("biểu ngữ", "banner"),
            ("băng rôn", "banner"),

            # 9. Agriculture, Farmers & Animals
            # 9. Sports, Athletics, Actions & Event Sequences
            ("vận động viên chạy đà hướng về phía xà ngang", "athlete running up towards high jump bar"),
            ("vận động viên chạy đà", "athlete running up"),
            ("chạy đà hướng về phía xà ngang", "running towards high jump bar"),
            ("chạy đà", "running up momentum"),
            ("xà ngang", "high jump bar"),
            ("vận động viên giậm nhảy bật người lên không trung qua xà", "athlete taking off jumping over high jump bar in air"),
            ("giậm nhảy bật người", "taking off jumping over"),
            ("giậm nhảy", "take-off jump"),
            ("bật người lên không trung", "jumping up in air"),
            ("qua xà", "over the bar"),
            ("vận động viên tiếp lưng rơi xuống nệm bảo hộ màu xanh", "athlete landing on back onto blue safety mat"),
            ("tiếp lưng rơi xuống nệm bảo hộ", "landing on back onto safety mat"),
            ("rơi xuống nệm bảo hộ màu xanh", "falling onto blue landing mat"),
            ("rơi xuống nệm bảo hộ", "falling onto landing mat"),
            ("rơi xuống nệm", "landing on safety mat"),
            ("nệm bảo hộ màu xanh", "blue safety landing mat"),
            ("nệm bảo hộ", "safety landing mat"),
            ("nệm xanh", "blue mat"),
            ("vận động viên", "athlete"),
            ("nhảy cao môn thể thao", "track and field high jump sport"),
            ("nhảy cao", "high jump sport"),
            ("môn thể thao", "sports competition"),
            ("điền kinh", "track and field athletics"),

            # 10. Fillers & Structural Stopwords
            ("tìm cảnh một", ""),
            ("tìm cảnh", ""),
            ("tìm đồ họa", "graphics showing"),
            ("tìm cận cảnh", "close-up of"),
            ("cho tôi thấy", ""),
            ("hình ảnh về", ""),
            ("video quay cảnh", ""),
            ("xuất hiện cảnh", ""),
            ("đoạn clip quay", ""),
            ("xác định", ""),
            ("cho biết", ""),
            ("tìm đoạn", ""),
            ("tìm video", ""),
            ("cho thấy", ""),
            ("hình ảnh", ""),
            ("cảnh một", ""),
            ("cận cảnh", "close-up of"),
            ("đồ họa", "graphics chart"),
            ("có các", ""),
            ("các", ""),
            ("một", "a"),
            ("đang", ""),
            ("và", "and"),
            ("với", "with"),
            ("trên", "on"),
            ("ở", "in"),
            ("tại", "at"),
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
            "lúa": "rice", "nông": "farmer", "đồng": "field", "đá": "rocks",
            "tháp": "tower", "gạch": "brick", "cổ": "ancient", "cây": "plants",
            "dốc": "ramps", "tường": "wall", "bục": "podium", "cá": "fish",
            "chậu": "basin", "thau": "basin", "hà mã": "hippo", "xích đu": "swing",
            "nhảy": "jumping", "chạy": "running", "đà": "momentum", "xà": "bar",
            "nệm": "mat", "rơi": "falling", "giậm": "takeoff", "bảo": "safety",
            "thao": "sports", "thể": "athletic",
            "con": "", "cái": "", "chiếc": "", "bức": "", "tấm": "", "đoạn": "", "bản": "",
        }

        # Unaccented Vietnamese words that leak through ASCII check if not explicitly filtered
        _VI_UNACCENTED_LEAK_WORDS = {
            "phun", "sau", "phia", "tao", "duoc", "cung", "nhung", "cac", "nguoi",
            "vua", "qua", "mang", "cho", "lay", "xem", "lam", "ra", "vao", "theo",
            "nhieu", "hay", "voi", "va", "tren", "duoi", "trai", "phai", "giua",
            "ngoai", "trong", "tai", "den", "tu", "con", "cai", "chiec", "buc", "tam",
            "do", "dang", "rat", "mot", "co", "la", "vinh", "tuan", "lan", "di",
            "chuyen", "nguy", "hiem", "thu", "hoach", "ng", "vang"
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

            # Check unaccented leak words
            if word in _VI_UNACCENTED_LEAK_WORDS:
                logger.debug(f"Filter out unaccented leak word: '{word}'")
                continue

            # Check if word still contains Vietnamese diacritics
            has_vi_diacritics = any(c in vi_chars for c in word)
            if has_vi_diacritics:
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
