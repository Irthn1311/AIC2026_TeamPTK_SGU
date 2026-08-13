"""
Query Router & Dynamic Multimodal Fusion Policy
===============================================
Maps structured QueryUnderstandingResult to calibrated 4-branch retrieval weights
using config-driven intent policies, confidence-based interpolation, and guardrails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.query_understanding.schemas import (
    DynamicFusionDecision,
    FusionWeights,
    IntentEnum,
    QueryUnderstandingResult,
)

logger = logging.getLogger(__name__)

# Root fallback directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class QueryRouter:
    """
    Config-driven Query Router.
    Calculates dynamic multimodal fusion weights without modifying retrieval engines.
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config_path = self._resolve_config_path(config_path)
        self.baseline_weights: Dict[str, float] = {"visual": 0.40, "ocr": 0.25, "asr": 0.25, "object": 0.10}
        self.intent_weights: Dict[str, Dict[str, float]] = {
            "visual_scene": {"visual": 0.65, "ocr": 0.08, "asr": 0.07, "object": 0.20},
            "visual_object": {"visual": 0.50, "ocr": 0.05, "asr": 0.05, "object": 0.40},
            "visual_action": {"visual": 0.70, "ocr": 0.05, "asr": 0.10, "object": 0.15},
            "ocr_text": {"visual": 0.15, "ocr": 0.70, "asr": 0.08, "object": 0.07},
            "speech_asr": {"visual": 0.15, "ocr": 0.08, "asr": 0.70, "object": 0.07},
            "visual_ocr": {"visual": 0.40, "ocr": 0.45, "asr": 0.07, "object": 0.08},
            "visual_asr": {"visual": 0.38, "ocr": 0.07, "asr": 0.45, "object": 0.10},
            "visual_object_action": {"visual": 0.60, "ocr": 0.05, "asr": 0.05, "object": 0.30},
            "mixed": {"visual": 0.40, "ocr": 0.25, "asr": 0.25, "object": 0.10},
        }
        self.low_threshold: float = 0.55
        self.full_dynamic_threshold: float = 0.85
        self.mixed_max_alpha: float = 0.25
        self.min_visual_weight: float = 0.10
        self.max_object_weight: float = 0.45
        self.max_refinement_delta: float = 0.08

        # Load YAML configuration if available
        self._load_config()

    def _resolve_config_path(self, config_path: Optional[str | Path]) -> Path:
        if config_path is not None:
            return Path(config_path)
        candidates = [
            PROJECT_ROOT / "configs" / "query_routing.yaml",
            Path("configs/query_routing.yaml"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return PROJECT_ROOT / "configs" / "query_routing.yaml"

    def _load_config(self) -> None:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if "baseline_weights" in cfg:
                    self.baseline_weights = {k: float(v) for k, v in cfg["baseline_weights"].items()}
                if "intent_weights" in cfg:
                    self.intent_weights = {
                        str(k): {bk: float(bv) for bk, bv in v.items()}
                        for k, v in cfg["intent_weights"].items()
                    }
                conf_cfg = cfg.get("confidence_thresholds", {})
                self.low_threshold = float(conf_cfg.get("low_threshold", self.low_threshold))
                self.full_dynamic_threshold = float(conf_cfg.get("full_dynamic_threshold", self.full_dynamic_threshold))
                self.mixed_max_alpha = float(conf_cfg.get("mixed_max_alpha", self.mixed_max_alpha))
                guard_cfg = cfg.get("guardrails", {})
                self.min_visual_weight = float(guard_cfg.get("min_visual_weight", self.min_visual_weight))
                self.max_object_weight = float(guard_cfg.get("max_object_weight", self.max_object_weight))
                self.max_refinement_delta = float(guard_cfg.get("max_refinement_delta", self.max_refinement_delta))
            except Exception as exc:
                logger.warning(f"Could not parse routing config {self.config_path}: {exc}. Using internal defaults.")

    def route(
        self,
        analysis: QueryUnderstandingResult,
        force_mode: Optional[str] = None,
    ) -> DynamicFusionDecision:
        """
        Route structured query understanding analysis to 4-branch fusion weights.
        """
        base_fw = FusionWeights(
            visual=self.baseline_weights.get("visual", 0.40),
            ocr=self.baseline_weights.get("ocr", 0.25),
            asr=self.baseline_weights.get("asr", 0.25),
            object=self.baseline_weights.get("object", 0.10),
        ).normalized()

        # Handle explicit static override
        if force_mode == "static":
            return DynamicFusionDecision(
                fusion_mode="static",
                intent=analysis.intent,
                parser_confidence=analysis.confidence,
                dynamic_weights=base_fw,
                baseline_weights=base_fw,
                final_weights=base_fw,
                blend_factor=0.0,
                dominant_branch="visual",
                secondary_branch="ocr",
                routing_reason=["forced_static_override", "baseline_weights_applied"],
                fallback_used=True,
            )

        intent_key = analysis.intent.value if isinstance(analysis.intent, IntentEnum) else str(analysis.intent)
        raw_dyn = self.intent_weights.get(intent_key, self.baseline_weights).copy()
        reasons: List[str] = [f"intent={intent_key}"]

        # Multi-signal co-occurrence in MIXED intent (e.g. speech + on-screen text)
        if analysis.intent == IntentEnum.MIXED and analysis.ocr_likelihood >= 0.60 and analysis.asr_likelihood >= 0.60:
            raw_dyn = {"visual": 0.35, "ocr": 0.35, "asr": 0.25, "object": 0.05}
            reasons.append("multi_signal_cooccurrence_ocr_asr_visual")

        # Likelihood-aware bounded refinement (safe delta <= 0.06)
        if analysis.intent == IntentEnum.VISUAL_ASR:
            if analysis.asr_likelihood >= 0.85:
                raw_dyn["asr"] = raw_dyn.get("asr", 0.45) + 0.05
                raw_dyn["visual"] = max(self.min_visual_weight, raw_dyn.get("visual", 0.38) - 0.05)
                reasons.append("boost_speech_asr_evidence")
        elif analysis.intent == IntentEnum.VISUAL_OCR:
            if analysis.ocr_likelihood >= 0.85:
                raw_dyn["ocr"] = raw_dyn.get("ocr", 0.45) + 0.05
                raw_dyn["visual"] = max(self.min_visual_weight, raw_dyn.get("visual", 0.40) - 0.05)
                reasons.append("boost_ocr_text_evidence")
        elif analysis.intent == IntentEnum.VISUAL_OBJECT_ACTION:
            if analysis.object_likelihood >= 0.80:
                raw_dyn["object"] = min(self.max_object_weight, raw_dyn.get("object", 0.30) + 0.03)
                raw_dyn["visual"] = max(self.min_visual_weight, raw_dyn.get("visual", 0.60) - 0.03)
                reasons.append("boost_object_evidence")

        # Apply guardrails
        raw_dyn["visual"] = max(self.min_visual_weight, raw_dyn.get("visual", 0.10))
        raw_dyn["object"] = min(self.max_object_weight, raw_dyn.get("object", 0.45))

        dyn_fw = FusionWeights(
            visual=raw_dyn.get("visual", 0.40),
            ocr=raw_dyn.get("ocr", 0.25),
            asr=raw_dyn.get("asr", 0.25),
            object=raw_dyn.get("object", 0.10),
        ).normalized()

        # Compute continuous interpolation blend factor (alpha)
        conf = max(0.0, min(1.0, float(analysis.confidence)))
        if analysis.intent == IntentEnum.MIXED:
            # Conservative blending for mixed/ambiguous intents
            if conf <= self.low_threshold:
                alpha = 0.0
            else:
                interp = (conf - self.low_threshold) / max(1e-6, self.full_dynamic_threshold - self.low_threshold)
                alpha = min(self.mixed_max_alpha, max(0.0, min(1.0, interp)) * self.mixed_max_alpha)
            reasons.append("mixed_intent_conservative_blend")
        else:
            if conf <= self.low_threshold:
                alpha = 0.0
                reasons.append("confidence_below_low_threshold_baseline_used")
            elif conf >= self.full_dynamic_threshold:
                alpha = 1.0
                reasons.append("high_confidence_full_dynamic_applied")
            else:
                alpha = (conf - self.low_threshold) / (self.full_dynamic_threshold - self.low_threshold)
                reasons.append(f"interpolated_blend_alpha={alpha:.2f}")

        # Compute blended final weights
        final_v = alpha * dyn_fw.visual + (1.0 - alpha) * base_fw.visual
        final_o = alpha * dyn_fw.ocr + (1.0 - alpha) * base_fw.ocr
        final_a = alpha * dyn_fw.asr + (1.0 - alpha) * base_fw.asr
        final_obj = alpha * dyn_fw.object + (1.0 - alpha) * base_fw.object

        final_fw = FusionWeights(visual=final_v, ocr=final_o, asr=final_a, object=final_obj).normalized()

        # Identify dominant and secondary branches
        sorted_branches = sorted(
            [
                ("visual", final_fw.visual),
                ("ocr", final_fw.ocr),
                ("asr", final_fw.asr),
                ("object", final_fw.object),
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        dominant_branch = sorted_branches[0][0]
        secondary_branch = sorted_branches[1][0]

        if analysis.object_terms:
            reasons.append(f"detected_objects={len(analysis.object_terms)}")
        if analysis.actions:
            reasons.append(f"detected_actions={len(analysis.actions)}")
        if analysis.ocr_query:
            reasons.append("ocr_query_extracted")
        if analysis.asr_query:
            reasons.append("asr_query_extracted")

        return DynamicFusionDecision(
            fusion_mode="dynamic",
            intent=analysis.intent,
            parser_confidence=conf,
            dynamic_weights=dyn_fw,
            baseline_weights=base_fw,
            final_weights=final_fw,
            blend_factor=round(alpha, 4),
            dominant_branch=dominant_branch,
            secondary_branch=secondary_branch,
            routing_reason=reasons,
            fallback_used=(alpha == 0.0),
        )
