"""
Query Understanding Schemas & Intent Definitions
================================================
Defines structured intent types, parsing results, fusion weights,
and dynamic routing decisions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IntentEnum(str, Enum):
    """Supported multimodal query intents for AI Challenge 2026."""
    VISUAL_SCENE = "visual_scene"
    VISUAL_OBJECT = "visual_object"
    VISUAL_ACTION = "visual_action"
    OCR_TEXT = "ocr_text"
    SPEECH_ASR = "speech_asr"
    VISUAL_OCR = "visual_ocr"
    VISUAL_ASR = "visual_asr"
    VISUAL_OBJECT_ACTION = "visual_object_action"
    MIXED = "mixed"


class QueryUnderstandingResult(BaseModel):
    """Structured representation of parsed natural language query."""
    original_query: str = Field(..., description="Raw natural language input query")
    intent: IntentEnum = Field(default=IntentEnum.MIXED, description="Classified multimodal intent")
    confidence: float = Field(default=0.50, ge=0.0, le=1.0, description="Confidence in intent classification [0, 1]")
    
    # Sub-query payloads for specialized branches
    visual_query: Optional[str] = Field(default=None, description="Visual description payload for CLIP encoder")
    ocr_query: Optional[str] = Field(default=None, description="Extracted on-screen text snippet for OCR branch")
    asr_query: Optional[str] = Field(default=None, description="Extracted spoken proposition for ASR branch")
    
    # Clue extractions
    object_terms: List[str] = Field(default_factory=list, description="Extracted entity/object terms for Object branch")
    actions: List[str] = Field(default_factory=list, description="Extracted lexical action/motion verbs")
    scene_terms: List[str] = Field(default_factory=list, description="Extracted background/scene context words")
    temporal_terms: List[str] = Field(default_factory=list, description="Extracted chronological/temporal connective clues")
    
    # Branch evidence likelihoods in range [0.0, 1.0] (Not normalized to sum 1.0)
    visual_likelihood: float = Field(default=0.50, ge=0.0, le=1.0, description="Evidence score for visual retrieval")
    ocr_likelihood: float = Field(default=0.10, ge=0.0, le=1.0, description="Evidence score for on-screen text")
    asr_likelihood: float = Field(default=0.10, ge=0.0, le=1.0, description="Evidence score for speech/ASR content")
    object_likelihood: float = Field(default=0.10, ge=0.0, le=1.0, description="Evidence score for object detection")
    
    # Audit trail
    matched_rules: List[str] = Field(default_factory=list, description="Rule identifiers triggered during parsing")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard Python dictionary with stringified enum."""
        data = self.dict()
        data["intent"] = self.intent.value if isinstance(self.intent, IntentEnum) else str(self.intent)
        return data


class FusionWeights(BaseModel):
    """4-Branch Late Fusion weights (visual + ocr + asr + object = 1.0)."""
    visual: float = Field(default=0.40, ge=0.0, le=1.0)
    ocr: float = Field(default=0.25, ge=0.0, le=1.0)
    asr: float = Field(default=0.25, ge=0.0, le=1.0)
    object: float = Field(default=0.10, ge=0.0, le=1.0)

    def normalized(self) -> FusionWeights:
        """Return a strictly normalized instance where sum of weights equals 1.0."""
        v = max(0.0, float(self.visual))
        o = max(0.0, float(self.ocr))
        a = max(0.0, float(self.asr))
        obj = max(0.0, float(self.object))
        total = v + o + a + obj
        if total <= 1e-9:
            return FusionWeights(visual=0.40, ocr=0.25, asr=0.25, object=0.10)
        return FusionWeights(
            visual=round(v / total, 4),
            ocr=round(o / total, 4),
            asr=round(a / total, 4),
            object=round(obj / total, 4),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "visual": float(self.visual),
            "ocr": float(self.ocr),
            "asr": float(self.asr),
            "object": float(self.object),
        }


class DynamicFusionDecision(BaseModel):
    """Complete routing decision containing baseline, dynamic policy, and blended weights."""
    fusion_mode: str = Field(default="dynamic", description="'dynamic' or 'static'")
    intent: IntentEnum = Field(default=IntentEnum.MIXED, description="Assigned query intent")
    parser_confidence: float = Field(default=0.50, ge=0.0, le=1.0, description="Confidence of query parser")
    
    dynamic_weights: FusionWeights = Field(..., description="Raw intent policy weights")
    baseline_weights: FusionWeights = Field(..., description="Baseline fallback weights")
    final_weights: FusionWeights = Field(..., description="Confidence-blended final weights (sum=1.0)")
    
    blend_factor: float = Field(default=0.0, ge=0.0, le=1.0, description="Interpolation alpha [0: baseline, 1: dynamic]")
    dominant_branch: str = Field(default="visual", description="Branch with highest final weight")
    secondary_branch: str = Field(default="ocr", description="Branch with second highest final weight")
    
    routing_reason: List[str] = Field(default_factory=list, description="Explanatory audit items for debugging")
    fallback_used: bool = Field(default=False, description="True if forced static or fallback to baseline")

    def to_dict(self) -> Dict[str, Any]:
        data = self.dict()
        data["intent"] = self.intent.value if isinstance(self.intent, IntentEnum) else str(self.intent)
        data["dynamic_weights"] = self.dynamic_weights.to_dict()
        data["baseline_weights"] = self.baseline_weights.to_dict()
        data["final_weights"] = self.final_weights.to_dict()
        return data
