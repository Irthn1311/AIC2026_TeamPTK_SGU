"""
Intent Scorer — Computes optimal retrieval engine weights from extracted entities.

Given ExtractedEntities, scores the relative importance of:
  - visual_weight:  CLIP/FAISS visual similarity
  - ocr_weight:     Qdrant/BM25 text/OCR search
  - caption_weight: Caption-based semantic search

Output weights are normalized to sum = 1.0.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from src.preprocessing.entity_extractor import ExtractedEntities


@dataclass
class EngineWeights:
    visual: float = 0.6
    ocr: float = 0.25
    caption: float = 0.15

    def as_dict(self) -> Dict[str, float]:
        return {"visual": self.visual, "ocr": self.ocr, "caption": self.caption}


class IntentScorer:
    """
    Scores retrieval engine weights from entities.

    Strategy:
      - OCR score  += for each strong OCR hint (quoted text, ALL CAPS, scores, times)
      - Visual     += for colors, clothing, persons, scene context
      - Caption    += for actions, temporal cues, emotions, complex descriptions

    Usage:
        scorer = IntentScorer()
        weights = scorer.score(entities)
        # EngineWeights(visual=0.50, ocr=0.35, caption=0.15)
    """

    def score(self, entities: ExtractedEntities) -> EngineWeights:
        visual_score = 1.0   # baseline
        ocr_score    = 0.0
        caption_score = 0.0

        # OCR signals
        n_ocr = len(entities.ocr_hints)
        if n_ocr > 0:
            ocr_score += 0.4 + (n_ocr - 1) * 0.15
        # Quantities in context of scores/times boost OCR
        for q in entities.quantities:
            entity = q.get("entity", "")
            if any(w in entity for w in ("điểm", "bàn", "tỷ số", "goals", "points")):
                ocr_score += 0.25
                break

        # Visual signals
        visual_score += len(entities.colors) * 0.15
        visual_score += len(entities.clothing_details) * 0.10
        visual_score += len(entities.persons) * 0.10
        if entities.scene_type:
            visual_score += 0.10
        visual_score += len(entities.objects) * 0.05
        if entities.lighting:
            visual_score += 0.05

        # Caption signals
        caption_score += len(entities.actions) * 0.20
        caption_score += len(entities.temporal_cues) * 0.15
        caption_score += len(entities.emotions) * 0.10
        # Long raw text → more likely needs caption understanding
        word_count = len(entities.raw_text.split())
        if word_count > 20:
            caption_score += 0.15

        # Normalize
        total = visual_score + ocr_score + caption_score
        if total == 0:
            return EngineWeights()

        return EngineWeights(
            visual=round(visual_score / total, 3),
            ocr=round(ocr_score / total, 3),
            caption=round(caption_score / total, 3),
        )
