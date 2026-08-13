"""
Segment Evidence Aggregator for Multimodal Video Retrieval
=========================================================
Aggregates multimodal evidence across all keyframes in a candidate segment:
- OCR text (union & dedup)
- ASR transcripts (union & dedup)
- Object labels (union & confidence ranking)
- Frame scores and metadata
Constructs rich text context for Cross-Encoder Reranking.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set
from src.retrieval.temporal_grouper import CandidateSegment


class SegmentEvidenceAggregator:
    """
    Aggregates OCR, ASR, Object, and visual metadata across all member frames
    of a temporal candidate segment.
    """

    def __init__(self, max_context_chars: int = 512):
        self.max_context_chars = max_context_chars

    def aggregate(self, segment: CandidateSegment) -> CandidateSegment:
        """
        Populates segment.evidence and computes full_text_context.
        """
        ocr_snippets: List[str] = []
        asr_snippets: List[str] = []
        object_labels_set: Set[str] = set()
        seen_ocr: Set[str] = set()
        seen_asr: Set[str] = set()

        max_visual_score = 0.0
        max_ocr_score = 0.0
        max_asr_score = 0.0
        max_object_score = 0.0

        for m in segment.member_frames:
            # Branch scores
            sc = m.get("scores", {})
            if isinstance(sc, dict):
                max_visual_score = max(max_visual_score, float(sc.get("visual", 0.0)))
                max_ocr_score = max(max_ocr_score, float(sc.get("ocr", 0.0)))
                max_asr_score = max(max_asr_score, float(sc.get("asr", 0.0)))
                max_object_score = max(max_object_score, float(sc.get("object", 0.0)))

            # OCR text
            raw_ocr = str(m.get("ocr_text", "")).strip()
            if raw_ocr and raw_ocr.lower() not in ("none", "nan", "null", ""):
                cleaned_ocr = self._clean_text(raw_ocr)
                if cleaned_ocr and cleaned_ocr not in seen_ocr:
                    seen_ocr.add(cleaned_ocr)
                    ocr_snippets.append(cleaned_ocr)

            # ASR text
            raw_asr = str(m.get("asr_text", "")).strip()
            if raw_asr and raw_asr.lower() not in ("none", "nan", "null", ""):
                # Remove timestamp header if present like [12.0s - 15.0s]:
                cleaned_asr = re.sub(r"^\[\d+(\.\d+)?s\s*-\s*\d+(\.\d+)?s\]:\s*", "", raw_asr).strip()
                cleaned_asr = self._clean_text(cleaned_asr)
                if cleaned_asr and cleaned_asr not in seen_asr:
                    seen_asr.add(cleaned_asr)
                    asr_snippets.append(cleaned_asr)

            # Objects
            objs = m.get("objects", [])
            if isinstance(objs, list):
                for obj in objs:
                    if isinstance(obj, dict):
                        lbl = str(obj.get("label", "")).strip()
                        if lbl and lbl.lower() not in ("none", "nan", "null"):
                            object_labels_set.add(lbl)
                    elif isinstance(obj, str) and obj.strip():
                        object_labels_set.add(obj.strip())

        combined_ocr = " ; ".join(ocr_snippets)
        combined_asr = " ; ".join(asr_snippets)
        sorted_objects = sorted(list(object_labels_set))

        # Build structured text for Cross-Encoder (Query + Candidate Evidence)
        context_parts = [f"Video {segment.video_id} ({segment.start_sec:.1f}s-{segment.end_sec:.1f}s)"]
        if combined_ocr:
            context_parts.append(f"[OCR] {combined_ocr}")
        if combined_asr:
            context_parts.append(f"[ASR] {combined_asr}")
        if sorted_objects:
            context_parts.append(f"[Objects] {', '.join(sorted_objects)}")

        full_text = " | ".join(context_parts)
        if len(full_text) > self.max_context_chars:
            full_text = full_text[: self.max_context_chars].rsplit(" ", 1)[0] + "..."

        has_text = bool(combined_ocr or combined_asr or sorted_objects)

        segment.evidence = {
            "has_text_evidence": has_text,
            "ocr_text": combined_ocr,
            "asr_text": combined_asr,
            "object_labels": sorted_objects,
            "full_text_context": full_text,
            "max_branch_scores": {
                "visual": round(max_visual_score, 4),
                "ocr": round(max_ocr_score, 4),
                "asr": round(max_asr_score, 4),
                "object": round(max_object_score, 4),
            },
            "num_keyframes": len(segment.member_frames),
        }

        return segment

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text
