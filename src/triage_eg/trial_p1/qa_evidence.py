"""Answer-type-aware QA evidence sufficiency and bounded Qwen contracts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from triage_eg.e2e1.qa import (
    SHORT_SEMANTIC_TYPES,
    TEXT_PRESERVING_TYPES,
    garbage_reason,
    normalize_answer_text,
)
from triage_eg.fs1_v11.contracts import QWEN_ID, QWEN_REVISION

GENERIC_OBJECT_FALLBACKS = frozenset(
    {
        "người",
        "ghế",
        "bàn",
        "sách",
        "thức ăn",
        "dụng cụ",
        "person",
        "chair",
        "table",
        "book",
        "food",
        "tool",
        "không đủ bằng chứng",
        "insufficient evidence",
    }
)


@dataclass(frozen=True)
class BoundedEvidencePackage:
    video_id: str
    frame_id: int
    grounding_rank: int
    grounding_sources: tuple[str, ...]
    visual_context_present: bool = False
    ocr_lines: tuple[dict[str, Any], ...] = ()
    asr_spans: tuple[dict[str, Any], ...] = ()
    object_observations: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class EvidenceAssessment:
    answer: str
    answer_type: str
    answer_policy: str
    syntax_pass: bool
    syntax_reason: str | None
    evidence_sources: tuple[str, ...]
    evidence_sufficient: bool
    ocr_confidence: float | None
    reconstructed_ocr_line: str | None
    asr_span_confidence: float | None
    asr_provenance: dict[str, Any] | None
    qwen_verified: bool
    grounding_source: tuple[str, ...]
    grounding_rank: int
    insufficiency_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plain(value: str) -> str:
    return re.sub(r"[^\w]+", " ", normalize_answer_text(value).casefold()).strip()


def _contains_answer(evidence: str, answer: str) -> bool:
    left, right = _plain(evidence), _plain(answer)
    return bool(right and (right in left or left in right))


def _contextual_ocr(
    answer: str, answer_type: str, lines: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any] | None, str | None]:
    kind = answer_type.upper()
    for line in lines:
        text = normalize_answer_text(line.get("text", ""))
        confidence = float(line.get("confidence", -1))
        if confidence < 50 or not _contains_answer(text, answer):
            continue
        words = re.findall(r"\w+", text, re.UNICODE)
        if len(words) < 2:
            continue
        if kind == "LOCATION_NAME" and not re.search(
            r"\b(xã|phường|huyện|tỉnh|thành phố)\b", text, re.I
        ):
            continue
        if kind == "TITLE" and len(words) < 3:
            continue
        if kind == "QUOTE_OR_VISIBLE_TEXT" and len(words) < 4:
            continue
        return line, None
    return None, "NO_CONTEXTUAL_OCR_PHRASE_WITH_CONFIDENCE"


def _supporting_asr(
    answer: str, answer_type: str, spans: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any] | None, str | None]:
    for span in spans:
        text = normalize_answer_text(span.get("text", span.get("normalized_text", "")))
        confidence = span.get("confidence")
        provenance = span.get("provenance")
        if not isinstance(provenance, dict) or not provenance.get("video_id"):
            continue
        if confidence is not None and float(confidence) < 0.5:
            continue
        if not _contains_answer(text, answer):
            continue
        if answer_type == "LOCATION_NAME" and not re.search(
            r"\b(xã|phường|huyện|tỉnh|thành phố)\b", text, re.I
        ):
            continue
        return span, None
    return None, "NO_LOCAL_ASR_SPAN_MATCHING_SEMANTIC_ROLE"


def assess_answer_evidence(
    answer: str,
    answer_type: str,
    package: BoundedEvidencePackage,
    *,
    qwen_result: dict[str, Any] | None = None,
) -> EvidenceAssessment:
    kind = str(answer_type).upper()
    normalized = normalize_answer_text(answer)
    syntax_reason = garbage_reason(normalized, kind)
    policy = "TEXT_PRESERVING" if kind in TEXT_PRESERVING_TYPES else "SHORT_SEMANTIC"
    sources: list[str] = []
    reasons: list[str] = []
    ocr_row = asr_row = None
    qwen_verified = False
    if syntax_reason:
        reasons.append(f"SYNTAX:{syntax_reason}")
    if normalized.casefold() in GENERIC_OBJECT_FALLBACKS:
        reasons.append("GENERIC_OBJECT_OR_INSUFFICIENT_EVIDENCE_FALLBACK")
    if kind in TEXT_PRESERVING_TYPES:
        ocr_row, ocr_reason = _contextual_ocr(normalized, kind, package.ocr_lines)
        if ocr_row:
            sources.append("OCR_CONTEXTUAL_PHRASE")
        elif ocr_reason:
            reasons.append(ocr_reason)
        asr_row, asr_reason = _supporting_asr(normalized, kind, package.asr_spans)
        if asr_row:
            sources.append("ASR_LOCAL_SPAN")
        elif asr_reason:
            reasons.append(asr_reason)
    elif kind in SHORT_SEMANTIC_TYPES:
        if package.object_observations:
            sources.append("VISUAL_OR_OBJECT_EVIDENCE")
        numeric = any(
            _contains_answer(line.get("text", ""), normalized) for line in package.ocr_lines
        )
        if kind == "COUNT" and numeric:
            sources.append("OCR_NUMERIC_EVIDENCE")
    if qwen_result is not None:
        qwen_answer = normalize_answer_text(qwen_result.get("answer", ""))
        bounded_evidence_present = bool(
            package.visual_context_present
            or package.ocr_lines
            or package.asr_spans
            or package.object_observations
        )
        qwen_verified = bool(
            qwen_result.get("evidence_sufficient") is True
            and qwen_answer == normalized
            and garbage_reason(qwen_answer, kind) is None
            and bounded_evidence_present
        )
        if qwen_verified:
            sources.append("QWEN_BOUNDED_EVIDENCE_VERIFICATION")
        else:
            reasons.append("QWEN_DID_NOT_VERIFY_EXACT_ANSWER")
    sufficient = bool(sources) and syntax_reason is None
    if normalized.casefold() in GENERIC_OBJECT_FALLBACKS:
        sufficient = False
    return EvidenceAssessment(
        answer=normalized,
        answer_type=kind,
        answer_policy=policy,
        syntax_pass=syntax_reason is None,
        syntax_reason=syntax_reason,
        evidence_sources=tuple(dict.fromkeys(sources)),
        evidence_sufficient=sufficient,
        ocr_confidence=float(ocr_row["confidence"]) if ocr_row else None,
        reconstructed_ocr_line=str(ocr_row["text"]) if ocr_row else None,
        asr_span_confidence=(
            float(asr_row["confidence"])
            if asr_row and asr_row.get("confidence") is not None
            else None
        ),
        asr_provenance=dict(asr_row["provenance"]) if asr_row else None,
        qwen_verified=qwen_verified,
        grounding_source=package.grounding_sources,
        grounding_rank=package.grounding_rank,
        insufficiency_reasons=tuple(dict.fromkeys(reasons)),
    )


def build_qwen_prompt(question: str, answer_type: str, package: BoundedEvidencePackage) -> str:
    evidence = {
        "ocr_lines": list(package.ocr_lines)[:10],
        "asr_spans": list(package.asr_spans)[:5],
        "object_observations": list(package.object_observations)[:10],
    }
    return (
        "Answer only from this bounded evidence package. Return JSON with exactly answer and "
        "evidence_sufficient. Do not invent or return video/frame IDs. For text-preserving "
        "answers, copy faithfully without explanation or paraphrase. Answer <=100 characters. "
        f"Compiled answer_type={answer_type}. Question={question}. Evidence="
        + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    )


def parse_bounded_qwen_output(
    raw: str, answer_type: str, package: BoundedEvidencePackage
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        match = re.search(r"\{.*\}", str(raw), re.S)
        parsed = json.loads(match.group(0) if match else raw)
        answer = normalize_answer_text(parsed["answer"])
        sufficient = parsed["evidence_sufficient"]
        forbidden_identifier = answer in {package.video_id, str(package.frame_id)}
        if (
            not isinstance(sufficient, bool)
            or garbage_reason(answer, answer_type)
            or forbidden_identifier
        ):
            raise ValueError("invalid Qwen answer contract")
        result = {
            "video_id": package.video_id,
            "frame_id": package.frame_id,
            "answer": answer,
            "answer_type": answer_type,
            "evidence_sufficient": sufficient,
        }
        assessment = assess_answer_evidence(
            answer, answer_type, package, qwen_result=result
        ).as_dict()
        return result, assessment
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, AttributeError):
        return None, {
            "video_id": package.video_id,
            "frame_id": package.frame_id,
            "answer_type": answer_type,
            "evidence_sufficient": False,
            "reason": "QWEN_OUTPUT_CONTRACT_INVALID",
        }


class BoundedQwenExecutor:
    """Model-agnostic executor enforcing the pinned, bounded Qwen contract."""

    def __init__(self, generate: Any) -> None:
        if not callable(generate):
            raise TypeError("generate must be callable")
        self.generate = generate

    def execute(
        self,
        question: str,
        answer_type: str,
        package: BoundedEvidencePackage,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if not (
            package.visual_context_present
            or package.ocr_lines
            or package.asr_spans
            or package.object_observations
        ):
            return None, {
                "video_id": package.video_id,
                "frame_id": package.frame_id,
                "answer_type": answer_type,
                "evidence_sufficient": False,
                "reason": "QWEN_BOUNDED_EVIDENCE_PACKAGE_EMPTY",
                "model_id": QWEN_ID,
                "model_revision": QWEN_REVISION,
            }
        prompt = build_qwen_prompt(question, answer_type, package)
        raw = self.generate(prompt, package)
        result, diagnostic = parse_bounded_qwen_output(raw, answer_type, package)
        diagnostic.update(
            {
                "model_id": QWEN_ID,
                "model_revision": QWEN_REVISION,
                "bounded_input_video_id": package.video_id,
                "bounded_input_frame_id": package.frame_id,
            }
        )
        return result, diagnostic


def rank_full_qa(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """FULL QA deliberately has no protected visual prefix."""

    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row.get("grounding_plausibility", 0.0)),
            0 if row.get("evidence_sufficient") else 1,
            -len(set(row.get("evidence_sources", []))),
            int(row.get("grounding_rank", 10**9)),
            str(row.get("video_id", "")),
            int(row.get("frame_id", 0)),
            str(row.get("answer", "")),
        ),
    )
    return [{**row, "rank": rank} for rank, row in enumerate(ordered[:100], 1)]


__all__ = [
    "BoundedEvidencePackage",
    "BoundedQwenExecutor",
    "EvidenceAssessment",
    "GENERIC_OBJECT_FALLBACKS",
    "assess_answer_evidence",
    "build_qwen_prompt",
    "parse_bounded_qwen_output",
    "rank_full_qa",
]
