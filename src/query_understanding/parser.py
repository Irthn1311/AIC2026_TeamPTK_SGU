"""
Query Understanding Parser Implementation
==========================================
Provides BaseQueryParser interface and RuleBasedQueryParser implementation.
Performs deterministic, ultra-fast (<2ms) rule parsing, entity extraction,
OCR/ASR clue routing, and likelihood scoring without external network or LLM calls.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple

from src.query_understanding.patterns import (
    ACTION_MULTIWORD_LEXICON,
    ACTION_SINGLEWORD_LEXICON,
    KNOWN_OBJECT_ENTITIES,
    OCR_EXTRACTION_PATTERNS,
    OCR_MARKER_KEYWORDS,
    SCENE_MULTIWORD_LEXICON,
    SCENE_SINGLEWORD_LEXICON,
    SPEECH_EXTRACTION_PATTERNS,
    SPEECH_MARKER_KEYWORDS,
    TEMPORAL_PATTERNS,
    VISUAL_CLUES_LEXICON,
)
from src.query_understanding.schemas import IntentEnum, QueryUnderstandingResult

# Try importing SYNONYMS_MAP without hard duplication
try:
    from src.retrieval.object_index import SYNONYMS_MAP
except ImportError:
    SYNONYMS_MAP = {}


class BaseQueryParser(ABC):
    """Abstract Base Class for Multimodal Query Understanding."""

    @abstractmethod
    def parse(self, query: str) -> QueryUnderstandingResult:
        """Parse natural language query into structured decomposition."""
        pass


class RuleBasedQueryParser(BaseQueryParser):
    """
    High-speed deterministic rule-based query parser.
    Executes in < 5ms on CPU using compiled regex and keyword set lookups.
    """

    def __init__(self):
        # Build consolidated known object vocabulary from KNOWN_OBJECT_ENTITIES and SYNONYMS_MAP
        self.known_objects: List[str] = list(KNOWN_OBJECT_ENTITIES)
        for key in SYNONYMS_MAP.keys():
            k_str = str(key).strip().lower()
            if k_str and k_str not in self.known_objects and len(k_str) >= 2:
                self.known_objects.append(k_str)
        # Sort descending by length for greedy substring matching
        self.known_objects.sort(key=lambda s: len(s), reverse=True)

    def parse(self, query: str) -> QueryUnderstandingResult:
        """Parse a natural language query into multimodal intent and extracted components."""
        started_at = time.time()
        
        # 1. Input validation and text normalization
        raw_query = str(query or "").strip()
        if not raw_query:
            return QueryUnderstandingResult(
                original_query="",
                intent=IntentEnum.MIXED,
                confidence=0.0,
                visual_query="",
                visual_likelihood=0.40,
                ocr_likelihood=0.25,
                asr_likelihood=0.25,
                object_likelihood=0.10,
                matched_rules=["empty_query_fallback"],
            )

        # Normalized lowercased string for rule matching (preserves Vietnamese accents)
        norm_query = re.sub(r"\s+", " ", raw_query.lower())
        matched_rules: List[str] = []

        # 2. Extract OCR Clues & OCR Payload
        ocr_likelihood, ocr_query, ocr_markers_found = self._extract_ocr(norm_query, raw_query)
        if ocr_markers_found:
            for m in ocr_markers_found:
                matched_rules.append(f"ocr_marker:{m}")

        # 3. Extract ASR / Speech Clues & ASR Payload
        asr_likelihood, asr_query, speech_markers_found = self._extract_asr(norm_query, raw_query)
        if speech_markers_found:
            for m in speech_markers_found:
                matched_rules.append(f"speech_marker:{m}")

        # 4. Extract Object / Entity Terms (Conservative)
        object_terms = self._extract_objects(norm_query)
        for obj in object_terms:
            matched_rules.append(f"object:{obj}")

        # 5. Extract Lexical Actions & Motion Verbs
        actions = self._extract_actions(norm_query)
        for act in actions:
            matched_rules.append(f"action:{act}")

        # 6. Extract Scene & Background Terms
        scene_terms = self._extract_scenes(norm_query)
        for sc in scene_terms:
            matched_rules.append(f"scene:{sc}")

        # 7. Extract Temporal Connectives
        temporal_terms = self._extract_temporal(norm_query)
        for tmp in temporal_terms:
            matched_rules.append(f"temporal:{tmp}")

        # 8. Extract Visual Descriptors & Composition Clues
        visual_descriptors = self._extract_visual_descriptors(norm_query)
        for vd in visual_descriptors:
            matched_rules.append(f"visual_clue:{vd}")

        # 9. Compute Likelihoods
        # Object Likelihood
        if object_terms:
            obj_base = 0.50 + min(0.35, len(object_terms) * 0.15)
            # High specificity multiplier for watercraft/vehicles/aviation/people
            object_likelihood = min(0.95, obj_base)
        else:
            object_likelihood = 0.10

        # Visual Likelihood
        vis_score = 0.50
        if actions:
            vis_score += min(0.25, len(actions) * 0.12)
        if object_terms:
            vis_score += min(0.20, len(object_terms) * 0.10)
        if visual_descriptors:
            vis_score += min(0.25, len(visual_descriptors) * 0.15)
        if scene_terms:
            vis_score += 0.15

        # Penalize visual if it is a pure on-screen text query without visual cues
        if ocr_likelihood >= 0.85 and not actions and not visual_descriptors and len(object_terms) <= 1:
            vis_score = max(0.20, vis_score - 0.40)
        # Penalize visual if it is a pure speech proposition without visual description
        if asr_likelihood >= 0.85 and not actions and not visual_descriptors and len(object_terms) <= 1:
            vis_score = max(0.25, vis_score - 0.35)

        visual_likelihood = min(0.95, max(0.15, vis_score))

        # Visual Query defaults to original raw query
        visual_query = raw_query

        # 10. Classify Intent & Calculate Classification Confidence
        intent, confidence = self._classify_intent(
            ocr_likelihood=ocr_likelihood,
            asr_likelihood=asr_likelihood,
            visual_likelihood=visual_likelihood,
            object_likelihood=object_likelihood,
            object_terms=object_terms,
            actions=actions,
            scene_terms=scene_terms,
            ocr_query=ocr_query,
            asr_query=asr_query,
            visual_descriptors=visual_descriptors,
        )

        return QueryUnderstandingResult(
            original_query=raw_query,
            intent=intent,
            confidence=round(confidence, 3),
            visual_query=visual_query,
            ocr_query=ocr_query,
            asr_query=asr_query,
            object_terms=object_terms,
            actions=actions,
            scene_terms=scene_terms,
            temporal_terms=temporal_terms,
            visual_likelihood=round(visual_likelihood, 3),
            ocr_likelihood=round(ocr_likelihood, 3),
            asr_likelihood=round(asr_likelihood, 3),
            object_likelihood=round(object_likelihood, 3),
            matched_rules=matched_rules,
        )

    # --------------------------------------------------------------------------
    # Sub-Extractors
    # --------------------------------------------------------------------------
    def _extract_ocr(self, norm_text: str, raw_text: str) -> Tuple[float, Optional[str], List[str]]:
        """Identify OCR markers and extract text payload."""
        found_markers: List[str] = []
        for kw in OCR_MARKER_KEYWORDS:
            if kw in norm_text:
                found_markers.append(kw)

        if not found_markers:
            return 0.05, None, []

        # Attempt to extract precise target text payload
        extracted_text: Optional[str] = None
        for pattern in OCR_EXTRACTION_PATTERNS:
            match = pattern.search(raw_text)
            if match:
                payload = match.group(1).strip()
                # Discard common prepositions/filler
                if len(payload) >= 2 and payload.lower() not in {"ở", "trên", "tại", "có", "xuất hiện"}:
                    extracted_text = payload
                    break

        # If strong marker exists but regex couldn't isolate snippet, fallback gracefully
        if not extracted_text:
            extracted_text = raw_text

        # Likelihood scoring
        strong_markers = {"dòng chữ", "chữ", "logo", "biển hiệu", "bảng hiệu", "tiêu đề", "ticker", "headline"}
        if any(m in strong_markers for m in found_markers):
            likelihood = 0.95
        else:
            likelihood = 0.70

        return likelihood, extracted_text, found_markers

    def _extract_asr(self, norm_text: str, raw_text: str) -> Tuple[float, Optional[str], List[str]]:
        """Identify speech/verbal markers and extract proposition payload."""
        found_markers: List[str] = []
        for kw in SPEECH_MARKER_KEYWORDS:
            if kw in norm_text:
                found_markers.append(kw)

        if not found_markers:
            return 0.05, None, []

        # Attempt to extract speech proposition payload
        extracted_proposition: Optional[str] = None
        for pattern in SPEECH_EXTRACTION_PATTERNS:
            match = pattern.search(raw_text)
            if match:
                payload = match.group(1).strip()
                if len(payload) >= 3:
                    extracted_proposition = payload
                    break

        if not extracted_proposition:
            extracted_proposition = raw_text

        # Strong speech markers like "nói rằng", "phát biểu rằng", "cho biết rằng", "tuyên bố"
        strong_speech = {"nói rằng", "phát biểu rằng", "cho biết rằng", "tuyên bố rằng", "thông báo rằng", "chia sẻ rằng"}
        if any(m in strong_speech for m in found_markers):
            likelihood = 0.90
        elif any(m in {"nói", "phát biểu", "cho biết", "phỏng vấn", "nói chuyện", "chia sẻ"} for m in found_markers):
            likelihood = 0.75
        else:
            likelihood = 0.60

        return likelihood, extracted_proposition, found_markers

    def _extract_objects(self, norm_text: str) -> List[str]:
        """Extract valid entity/object terms conservatively."""
        matched: List[str] = []
        covered_spans: List[Tuple[int, int]] = []

        for entity in self.known_objects:
            pattern = r"\b" + re.escape(entity) + r"\b"
            for match in re.finditer(pattern, norm_text):
                start, end = match.span()
                # Check overlap with existing greedy span
                if not any(s <= start and end <= e for s, e in covered_spans):
                    matched.append(entity)
                    covered_spans.append((start, end))

        return list(dict.fromkeys(matched))

    def _extract_actions(self, norm_text: str) -> List[str]:
        """Extract lexical action/motion verbs."""
        actions_found: List[str] = []
        covered_spans: List[Tuple[int, int]] = []

        # 1. Multi-word actions first
        for act in ACTION_MULTIWORD_LEXICON:
            pattern = r"\b" + re.escape(act) + r"\b"
            for match in re.finditer(pattern, norm_text):
                start, end = match.span()
                actions_found.append(act)
                covered_spans.append((start, end))

        # 2. Single-word actions
        for act in ACTION_SINGLEWORD_LEXICON:
            pattern = r"\b" + re.escape(act) + r"\b"
            for match in re.finditer(pattern, norm_text):
                start, end = match.span()
                if not any(s <= start and end <= e for s, e in covered_spans):
                    actions_found.append(act)
                    covered_spans.append((start, end))

        return list(dict.fromkeys(actions_found))

    def _extract_scenes(self, norm_text: str) -> List[str]:
        """Extract scene and background context terms."""
        scenes: List[str] = []
        for sc in SCENE_MULTIWORD_LEXICON:
            if re.search(r"\b" + re.escape(sc) + r"\b", norm_text):
                scenes.append(sc)
        for sc in SCENE_SINGLEWORD_LEXICON:
            if re.search(r"\b" + re.escape(sc) + r"\b", norm_text) and sc not in scenes:
                scenes.append(sc)
        return scenes

    def _extract_temporal(self, norm_text: str) -> List[str]:
        """Extract chronological connectives and time markers."""
        temporals: List[str] = []
        for pat in TEMPORAL_PATTERNS:
            if re.search(r"\b" + re.escape(pat) + r"\b", norm_text):
                temporals.append(pat)
        return temporals

    def _extract_visual_descriptors(self, norm_text: str) -> List[str]:
        """Extract colors, clothing descriptors, and composition layout."""
        descriptors: List[str] = []
        for desc in VISUAL_CLUES_LEXICON:
            if desc in norm_text:
                descriptors.append(desc)
        return descriptors

    # --------------------------------------------------------------------------
    # Deterministic Intent Classifier
    # --------------------------------------------------------------------------
    def _classify_intent(
        self,
        ocr_likelihood: float,
        asr_likelihood: float,
        visual_likelihood: float,
        object_likelihood: float,
        object_terms: List[str],
        actions: List[str],
        scene_terms: List[str],
        ocr_query: Optional[str],
        asr_query: Optional[str],
        visual_descriptors: List[str],
    ) -> Tuple[IntentEnum, float]:
        """Classify multimodal intent based on clear, deterministic rule priorities."""
        
        # Rule 0: Multi-modal Co-occurrence (e.g. "người nói chuyện trước màn hình có dòng chữ COVID-19")
        if ocr_likelihood >= 0.60 and asr_likelihood >= 0.60:
            return IntentEnum.MIXED, 0.85

        # Rule 1: Dominant OCR Text (e.g. "trên màn hình có dòng chữ Bộ Y tế", "logo HTV9 xuất hiện ở góc màn hình")
        if ocr_likelihood >= 0.85 and asr_likelihood < 0.40:
            if visual_descriptors or (actions and visual_likelihood >= 0.65):
                return IntentEnum.VISUAL_OCR, 0.90
            return IntentEnum.OCR_TEXT, 0.95

        # Rule 2: Dominant Speech / ASR (e.g. "phóng viên nói rằng mưa sẽ tiếp tục kéo dài")
        if asr_likelihood >= 0.85 and ocr_likelihood < 0.40:
            if visual_descriptors or len(actions) >= 2:
                return IntentEnum.VISUAL_ASR, 0.88
            return IntentEnum.SPEECH_ASR, 0.92

        # Rule 3: Visual Person/Scene + Speech Action (e.g. "người phụ nữ đang phát biểu trước tòa nhà", "người đàn ông nói chuyện")
        if asr_likelihood >= 0.60 and (visual_likelihood >= 0.45 or len(object_terms) > 0 or len(scene_terms) > 0):
            return IntentEnum.VISUAL_ASR, 0.85

        # Rule 4: Visual Scene/Person + OCR Text (e.g. "phóng viên đứng trước bệnh viện có chữ Chợ Rẫy")
        if ocr_likelihood >= 0.60 and (visual_likelihood >= 0.45 or len(object_terms) > 0 or len(scene_terms) > 0):
            return IntentEnum.VISUAL_OCR, 0.88

        # Rule 5: Visual Object + Action (e.g. "thuyền máy chạy trên sông", "máy bay đang cất cánh", "người đàn ông mặc áo đỏ bước xuống xe")
        if len(object_terms) > 0 and len(actions) > 0 and visual_likelihood >= 0.50:
            return IntentEnum.VISUAL_OBJECT_ACTION, 0.90

        # Rule 6: Visual Action dominant (e.g. "đang bơi lội và chạy nhảy")
        if len(actions) > 0 and visual_likelihood >= 0.55 and len(object_terms) == 0:
            return IntentEnum.VISUAL_ACTION, 0.80

        # Rule 7: Visual Object dominant (e.g. "chiếc ô tô màu trắng và con thuyền")
        if len(object_terms) > 0 and visual_likelihood >= 0.50 and len(actions) == 0:
            # Generic single person query like 'một người ở ngoài' is ambiguous
            if object_terms == ["người"] and not visual_descriptors and not scene_terms:
                return IntentEnum.MIXED, 0.50
            if object_terms == ["người"] and len(scene_terms) > 0:
                return IntentEnum.VISUAL_SCENE, 0.60
            return IntentEnum.VISUAL_OBJECT, 0.82

        # Rule 8: Visual Scene dominant (e.g. "bờ sông hoàng hôn")
        if len(scene_terms) > 0 and visual_likelihood >= 0.50 and len(object_terms) == 0:
            return IntentEnum.VISUAL_SCENE, 0.78

        # Rule 9: Mixed / Ambiguous (e.g. "một người ở ngoài", vague query)
        # Margin check between top likelihood and secondary
        scores = [visual_likelihood, ocr_likelihood, asr_likelihood, object_likelihood]
        sorted_scores = sorted(scores, reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]
        
        if margin < 0.15:
            return IntentEnum.MIXED, 0.50

        # If visual is marginally higher than others
        if visual_likelihood > max(ocr_likelihood, asr_likelihood, object_likelihood):
            return IntentEnum.VISUAL_SCENE, 0.55

        return IntentEnum.MIXED, 0.50
