"""Bounded structured visual verification for KIS timeline candidates.

This module is deliberately isolated from canonical CLIP retrieval.  It verifies a
small, automatically selected set of raw-video frames and never accepts target video,
timestamp, frame, benchmark label, or ground truth inputs.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol


class VisualVerificationError(RuntimeError):
    """The optional visual verifier could not produce a trustworthy result."""


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_NUMBER_PATTERN = "|".join((*_NUMBER_WORDS, r"\d+"))
_COUNT_PATTERN = re.compile(
    rf"\b(?P<qualifier>more than|over|at least|fewer than|less than|at most|"
    rf"exactly|only)?\s*(?P<number>{_NUMBER_PATTERN})\s+"
    r"(?P<subject>people|person|men|man|women|woman|children|child|students?|"
    r"workers?|players?|drivers?|employees?|animals?|objects?|items?)"
    r"(?:\s+(?P<attribute>wearing|with|holding|carrying)\s+"
    r"(?P<detail>[^,.;]+))?",
    flags=re.IGNORECASE,
)
_SPATIAL_TERMS = re.compile(
    r"\b(in (?:a |one )?(?:row|line)|lined up|next to|between|behind|in front of|"
    r"left of|right of|around|surrounding|above|below)\b",
    flags=re.IGNORECASE,
)
_ACTION_TERMS = re.compile(
    r"\b(doing|performing|touching|bending|exercising|walking|running|sitting|"
    r"standing|jumping|cutting|cooking|placing|pouring|stirring|pulling|holding|"
    r"carrying|climbing|driving|riding|speaking|teaching|weighing|picking|"
    r"showing|opening|closing|moving|turning|looking|observing)\b",
    flags=re.IGNORECASE,
)
_SYNCHRONIZATION_TERMS = re.compile(
    r"\b(together|simultaneously|at the same time|same action|in unison)\b",
    flags=re.IGNORECASE,
)
_VI_SYNCHRONIZATION_TERMS = re.compile(
    r"\b(cùng|đồng thời|cùng lúc|đồng loạt)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class VisualPredicateRequirement:
    """One stable, query-derived predicate shared by every candidate frame."""

    predicate_id: str
    requirement: str
    comparison: str | None = None
    expected_value: int | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.predicate_id):
            raise ValueError("visual predicate ID must be lower snake case")
        if not self.requirement.strip():
            raise ValueError("visual predicate requirement must not be empty")
        if self.comparison not in {None, "eq", "gt", "ge", "lt", "le"}:
            raise ValueError("unsupported visual predicate comparison")
        if (self.comparison is None) != (self.expected_value is None):
            raise ValueError("count comparison and expected value must be paired")

    def to_prompt(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "id": self.predicate_id,
                "requirement": self.requirement,
                "comparison": self.comparison,
                "expected_value": self.expected_value,
            }
        )


def _query_fragments(query_en: str) -> tuple[str, ...]:
    fragments = re.split(
        rf"[.;]|,\s+|\s+and\s+(?=(?:only\s+|exactly\s+)?(?:{_NUMBER_PATTERN})\b)",
        query_en,
        flags=re.IGNORECASE,
    )
    return tuple(" ".join(item.split()) for item in fragments if item.strip())


def _number_value(raw: str) -> int:
    normalized = raw.strip().casefold()
    if normalized.isdigit():
        return int(normalized)
    return _NUMBER_WORDS[normalized]


def _count_comparison(qualifier: str | None) -> str:
    normalized = (qualifier or "").strip().casefold()
    return {
        "more than": "gt",
        "over": "gt",
        "at least": "ge",
        "fewer than": "lt",
        "less than": "lt",
        "at most": "le",
    }.get(normalized, "eq")


def _comparison_matches(observed: int, comparison: str, expected: int) -> bool:
    return {
        "eq": observed == expected,
        "gt": observed > expected,
        "ge": observed >= expected,
        "lt": observed < expected,
        "le": observed <= expected,
    }[comparison]


def _non_negative_integer(value: Any, *, field_name: str) -> int:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"{field_name} must be a non-negative integer")


def _ordered_predicate_observation(
    *,
    requirement: VisualPredicateRequirement,
    state: str,
    observed: Any,
    image_number: Any,
) -> tuple[bool, bool, float, str, str]:
    """Convert a compact VLM observation into deterministic predicate state.

    The model supplies only visibility, a literal count/value, and an evidence image.
    Count satisfaction is always computed in code from the frozen query contract; a
    model-emitted ``Y`` can therefore never override an incorrect count.
    """

    if state not in {"Y", "N", "U"}:
        raise ValueError("ordered predicate state must be Y, N, or U")
    parsed_image_number = _non_negative_integer(
        image_number,
        field_name="ordered predicate image number",
    )
    if state == "U":
        parsed_observed = _non_negative_integer(
            observed,
            field_name="unknown ordered predicate value",
        )
        if parsed_observed != 0 or parsed_image_number != 0:
            raise ValueError(
                'unknown ordered predicate must be exactly ["U",0,0]'
            )
        return False, False, 0.0, "unknown", "not visible"

    # An ungrounded negative remains non-promotable instead of failing the whole
    # candidate. Positive claims must always cite a supporting image.
    if state == "Y" and parsed_image_number <= 0:
        raise ValueError("positive ordered predicate requires a positive image number")
    visible = parsed_image_number > 0
    evidence = (
        f"image {parsed_image_number}: verifier state {state}"
        if visible
        else "not visible"
    )
    if requirement.comparison is not None:
        parsed_observed = _non_negative_integer(
            observed,
            field_name="ordered count predicate value",
        )
        assert requirement.expected_value is not None
        satisfied = visible and _comparison_matches(
            parsed_observed,
            requirement.comparison,
            requirement.expected_value,
        )
        observed_value = str(parsed_observed)
    else:
        if isinstance(observed, (dict, list, tuple)):
            raise ValueError("ordered non-count predicate value must be scalar")
        satisfied = visible and state == "Y"
        observed_value = "satisfied" if satisfied else "not satisfied"
    return visible, satisfied, 1.0 if satisfied else 0.0, observed_value, evidence


def compile_visual_predicate_contract(
    *,
    query_vi: str,
    query_en: str,
    maximum_predicates: int = 12,
) -> tuple[VisualPredicateRequirement, ...]:
    """Compile stable predicates without target-video or ground-truth knowledge."""

    if not query_vi.strip() or not query_en.strip():
        raise ValueError("visual predicate compilation requires VI and EN query text")
    if type(maximum_predicates) is not int or maximum_predicates <= 0:
        raise ValueError("maximum_predicates must be a positive integer")
    requirements: list[VisualPredicateRequirement] = []
    category_counts: dict[str, int] = {}

    def add(
        category: str,
        requirement: str,
        *,
        comparison: str | None = None,
        expected_value: int | None = None,
    ) -> None:
        normalized = " ".join(requirement.split()).strip(" ,")
        identity = (category, normalized.casefold(), comparison, expected_value)
        if any(
            (
                item.predicate_id.rsplit("_", 1)[0],
                item.requirement.casefold(),
                item.comparison,
                item.expected_value,
            )
            == identity
            for item in requirements
        ):
            return
        category_counts[category] = category_counts.get(category, 0) + 1
        requirements.append(
            VisualPredicateRequirement(
                predicate_id=f"{category}_{category_counts[category]}",
                requirement=normalized,
                comparison=comparison,
                expected_value=expected_value,
            )
        )

    fragments = _query_fragments(query_en)
    action_fragments: list[str] = []
    for fragment in fragments:
        for match in _COUNT_PATTERN.finditer(fragment):
            detail = match.group("detail")
            category = "person_attribute_count" if detail else "subject_count"
            add(
                category,
                match.group(0),
                comparison=_count_comparison(match.group("qualifier")),
                expected_value=_number_value(match.group("number")),
            )
        if _SPATIAL_TERMS.search(fragment):
            add("spatial_layout", fragment)
        if _ACTION_TERMS.search(fragment) and "wearing" not in fragment.casefold():
            add("primary_action", fragment)
            action_fragments.append(fragment)

    if (
        _SYNCHRONIZATION_TERMS.search(query_en)
        or _VI_SYNCHRONIZATION_TERMS.search(query_vi)
    ):
        action_context = "; ".join(action_fragments) or query_en
        add(
            "synchronized_action",
            f"People perform together: {action_context}",
        )
    if not requirements:
        add("scene_conjunction", query_en)
    return tuple(requirements[:maximum_predicates])


@dataclass(frozen=True, slots=True)
class VisualVerificationInput:
    video_id: str
    absolute_frame_id: int
    timestamp_seconds: float
    images: tuple[Any, ...]
    image_frame_ids: tuple[int, ...] = ()
    image_timestamps_seconds: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("visual verification video_id must not be empty")
        if self.absolute_frame_id < 0:
            raise ValueError("visual verification frame ID must be non-negative")
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("visual verification timestamp must be finite and non-negative")
        if not self.images:
            raise ValueError("visual verification requires at least one image")
        if self.image_frame_ids:
            if len(self.image_frame_ids) != len(self.images):
                raise ValueError(
                    "visual verification image_frame_ids must align with images"
                )
            if any(frame_id < 0 for frame_id in self.image_frame_ids):
                raise ValueError(
                    "visual verification image frame IDs must be non-negative"
                )
            if len(set(self.image_frame_ids)) != len(self.image_frame_ids):
                raise ValueError("visual verification image frame IDs must be unique")
            if self.absolute_frame_id not in self.image_frame_ids:
                raise ValueError(
                    "visual verification candidate frame must be present in image_frame_ids"
                )
        if self.image_timestamps_seconds:
            if len(self.image_timestamps_seconds) != len(self.images):
                raise ValueError(
                    "visual verification image timestamps must align with images"
                )
            if any(
                not math.isfinite(timestamp) or timestamp < 0
                for timestamp in self.image_timestamps_seconds
            ):
                raise ValueError(
                    "visual verification image timestamps must be finite and non-negative"
                )

    @property
    def candidate_image_number(self) -> int:
        """Return the one-based position of the candidate image."""

        if self.image_frame_ids:
            return self.image_frame_ids.index(self.absolute_frame_id) + 1
        return len(self.images) // 2 + 1


@dataclass(frozen=True, slots=True)
class VisualPredicateScore:
    requirement: str
    score: float
    visible: bool
    evidence: str
    predicate_id: str = ""
    observed_value: str = ""
    satisfied: bool = False
    comparison: str | None = None
    expected_value: int | None = None

    def __post_init__(self) -> None:
        if not self.requirement.strip():
            raise ValueError("visual predicate requirement must not be empty")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("visual predicate score must be in [0, 1]")
        if type(self.visible) is not bool:
            raise ValueError("visual predicate visible must be boolean")
        if self.predicate_id and not re.fullmatch(
            r"[a-z][a-z0-9_]*", self.predicate_id
        ):
            raise ValueError("visual predicate ID must be lower snake case")
        if type(self.satisfied) is not bool:
            raise ValueError("visual predicate satisfied must be boolean")

    @property
    def evidence_grounded(self) -> bool:
        placeholders = {
            "",
            "n/a",
            "na",
            "none",
            "not visible",
            "unknown",
            "unclear",
            "cannot tell",
        }
        normalized_evidence = self.evidence.strip().casefold()
        return (
            normalized_evidence not in placeholders
            and re.search(r"\bimage\s+\d+\b", normalized_evidence) is not None
            and self.observed_value.strip().casefold() not in placeholders
        )

    @property
    def count_matches(self) -> bool:
        if self.comparison is None or self.expected_value is None:
            return True
        observed_match = re.search(
            rf"\b({_NUMBER_PATTERN})\b",
            self.observed_value,
            flags=re.IGNORECASE,
        )
        if observed_match is None:
            return False
        observed = _number_value(observed_match.group(1))
        expected = self.expected_value
        return {
            "eq": observed == expected,
            "gt": observed > expected,
            "ge": observed >= expected,
            "lt": observed < expected,
            "le": observed <= expected,
        }[self.comparison]

    @property
    def strictly_satisfied(self) -> bool:
        return (
            self.visible
            and self.satisfied
            and self.score >= 0.5
            and self.evidence_grounded
            and self.count_matches
        )

    @property
    def evidence_image_number(self) -> int | None:
        match = re.search(r"\bimage\s+(\d+)\b", self.evidence, re.IGNORECASE)
        return int(match.group(1)) if match is not None else None


@dataclass(frozen=True, slots=True)
class VisualVerificationResult:
    video_id: str
    absolute_frame_id: int
    match_score: float
    requirement_coverage: float
    all_visible_requirements_satisfied: bool
    predicates: tuple[VisualPredicateScore, ...]
    summary: str
    contract_validated: bool = False
    source_candidate_frame_id: int | None = None
    temporal_context_witness_frame_id: int | None = None
    temporal_action_witness_frame_id: int | None = None
    temporal_evidence_frame_ids: tuple[int, ...] = ()
    temporal_coherence_validated: bool | None = None

    def __post_init__(self) -> None:
        if not self.video_id.strip() or self.absolute_frame_id < 0:
            raise ValueError("invalid visual verification identity")
        for field_name, value in (
            ("match_score", self.match_score),
            ("requirement_coverage", self.requirement_coverage),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if type(self.all_visible_requirements_satisfied) is not bool:
            raise ValueError("all_visible_requirements_satisfied must be boolean")
        if not self.predicates:
            raise ValueError("visual verification must contain predicate scores")
        for field_name, frame_id in (
            ("source_candidate_frame_id", self.source_candidate_frame_id),
            (
                "temporal_context_witness_frame_id",
                self.temporal_context_witness_frame_id,
            ),
            ("temporal_action_witness_frame_id", self.temporal_action_witness_frame_id),
        ):
            if frame_id is not None and frame_id < 0:
                raise ValueError(f"{field_name} must be non-negative when provided")
        if any(frame_id < 0 for frame_id in self.temporal_evidence_frame_ids):
            raise ValueError("temporal evidence frame IDs must be non-negative")
        if type(self.temporal_coherence_validated) not in {bool, type(None)}:
            raise ValueError("temporal_coherence_validated must be boolean or None")

    @property
    def predicate_bottleneck_score(self) -> float:
        """Return the weakest independently verified requirement score.

        An invisible requirement is a hard zero.  This keeps conjunction ranking from
        rewarding a high broad-scene score when one required count, attribute, action,
        or relation is absent from the frame.
        """

        if self.contract_validated:
            return min(
                predicate.score if predicate.strictly_satisfied else 0.0
                for predicate in self.predicates
            )
        return min(
            predicate.score if predicate.visible else 0.0
            for predicate in self.predicates
        )

    @property
    def eligible_for_promotion(self) -> bool:
        if not self.contract_validated:
            return True
        if self.temporal_coherence_validated is False:
            return False
        return self.all_visible_requirements_satisfied and all(
            predicate.strictly_satisfied for predicate in self.predicates
        )

    @property
    def semantic_signature(self) -> tuple[Any, ...]:
        """Return a bounded signature for cross-candidate calibration.

        Small VLMs commonly quantize visual judgements to a few decimal values. When
        several different frames receive the same positive conjunction verdict, the
        verdict is non-discriminative and must not become independent ranking evidence.
        Evidence text is deliberately excluded: free-form wording must not make an
        otherwise identical score template appear unique.
        """

        return (
            self.all_visible_requirements_satisfied,
            self.temporal_coherence_validated,
            self.temporal_context_witness_frame_id
            == self.temporal_action_witness_frame_id,
            round(self.match_score, 2),
            round(self.requirement_coverage, 2),
            tuple(
                sorted(
                    (
                    predicate.predicate_id
                    or " ".join(predicate.requirement.casefold().split()),
                    round(predicate.score, 2),
                    predicate.visible,
                    predicate.satisfied,
                    " ".join(predicate.observed_value.casefold().split()),
                    )
                    for predicate in self.predicates
                )
            ),
        )

    def to_trace(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "video_id": self.video_id,
                "absolute_frame_id": self.absolute_frame_id,
                "match_score": self.match_score,
                "requirement_coverage": self.requirement_coverage,
                "all_visible_requirements_satisfied": (
                    self.all_visible_requirements_satisfied
                ),
                "predicate_bottleneck_score": self.predicate_bottleneck_score,
                "contract_validated": self.contract_validated,
                "eligible_for_promotion": self.eligible_for_promotion,
                "source_candidate_frame_id": self.source_candidate_frame_id,
                "temporal_coherence_validated": (
                    self.temporal_coherence_validated
                ),
                "temporal_context_witness_frame_id": (
                    self.temporal_context_witness_frame_id
                ),
                "temporal_action_witness_frame_id": (
                    self.temporal_action_witness_frame_id
                ),
                "temporal_evidence_frame_ids": list(
                    self.temporal_evidence_frame_ids
                ),
                "predicates": [
                    {
                        "predicate_id": predicate.predicate_id,
                        "requirement": predicate.requirement,
                        "score": predicate.score,
                        "visible": predicate.visible,
                        "satisfied": predicate.satisfied,
                        "observed_value": predicate.observed_value,
                        "evidence": predicate.evidence,
                        "comparison": predicate.comparison,
                        "expected_value": predicate.expected_value,
                        "count_matches": predicate.count_matches,
                        "strictly_satisfied": predicate.strictly_satisfied,
                    }
                    for predicate in self.predicates
                ],
                "summary": self.summary,
            }
        )


def bind_temporal_visual_evidence(
    result: VisualVerificationResult,
    *,
    candidate: VisualVerificationInput,
) -> VisualVerificationResult:
    """Bind compact image-number evidence to absolute raw-video coordinates.

    Count/layout predicates must share one context witness image. Action and
    synchronization predicates must share one action witness image. The two witnesses
    may differ inside the bounded temporal window, which models a short video moment
    without combining unrelated observations across frames.
    """

    if not candidate.image_frame_ids:
        return result
    if result.video_id != candidate.video_id:
        raise VisualVerificationError("visual verifier candidate video mismatch")

    frame_by_image_number = {
        index: frame_id
        for index, frame_id in enumerate(candidate.image_frame_ids, start=1)
    }
    contextual: list[VisualPredicateScore] = []
    actions: list[VisualPredicateScore] = []
    evidence_image_numbers: list[int] = []
    rebound_predicates: list[VisualPredicateScore] = []
    for predicate in result.predicates:
        image_number = predicate.evidence_image_number
        if image_number is not None:
            frame_id = frame_by_image_number.get(image_number)
            if frame_id is None:
                raise VisualVerificationError(
                    "visual verifier cited an image outside the bounded candidate window"
                )
            evidence_image_numbers.append(image_number)
            predicate = replace(
                predicate,
                evidence=(
                    f"{predicate.evidence}; absolute original-video frame {frame_id}"
                ),
            )
        predicate_id = predicate.predicate_id
        if predicate.comparison is not None or predicate_id.startswith(
            "spatial_layout_"
        ):
            contextual.append(predicate)
        if predicate_id.startswith(("primary_action_", "synchronized_action_")):
            actions.append(predicate)
        rebound_predicates.append(predicate)

    def coherent_witness(
        predicates: Sequence[VisualPredicateScore],
    ) -> int | None:
        if not predicates:
            return None
        if not all(predicate.strictly_satisfied for predicate in predicates):
            return None
        image_numbers = {
            predicate.evidence_image_number for predicate in predicates
        }
        if None in image_numbers or len(image_numbers) != 1:
            return None
        return frame_by_image_number[int(next(iter(image_numbers)))]

    context_witness = coherent_witness(contextual)
    action_witness = coherent_witness(actions)
    every_predicate_strict = all(
        predicate.strictly_satisfied for predicate in rebound_predicates
    )
    context_coherent = not contextual or context_witness is not None
    action_coherent = not actions or action_witness is not None
    temporal_coherent = (
        every_predicate_strict and context_coherent and action_coherent
    )

    representative_frame_id = candidate.absolute_frame_id
    if temporal_coherent:
        # The exported frame should show the requested action whenever an action
        # contract exists. Context-only queries use their coherent context witness.
        representative_frame_id = (
            action_witness or context_witness or candidate.absolute_frame_id
        )
        if action_witness is None and context_witness is None and evidence_image_numbers:
            representative_frame_id = frame_by_image_number[
                min(evidence_image_numbers)
            ]

    strict_coverage = sum(
        predicate.strictly_satisfied for predicate in rebound_predicates
    ) / len(rebound_predicates)
    return VisualVerificationResult(
        video_id=result.video_id,
        absolute_frame_id=representative_frame_id,
        match_score=strict_coverage,
        requirement_coverage=strict_coverage,
        all_visible_requirements_satisfied=temporal_coherent,
        predicates=tuple(rebound_predicates),
        summary=(
            "bounded temporal conjunction verified"
            if temporal_coherent
            else "bounded temporal conjunction not coherently verified"
        ),
        contract_validated=result.contract_validated,
        source_candidate_frame_id=candidate.absolute_frame_id,
        temporal_context_witness_frame_id=context_witness,
        temporal_action_witness_frame_id=action_witness,
        temporal_evidence_frame_ids=candidate.image_frame_ids,
        temporal_coherence_validated=temporal_coherent,
    )


def _visual_result_preference(
    result: VisualVerificationResult,
) -> tuple[int, int, float, float, float, int]:
    source_frame_id = (
        result.source_candidate_frame_id
        if result.source_candidate_frame_id is not None
        else result.absolute_frame_id
    )
    return (
        int(result.eligible_for_promotion),
        int(result.all_visible_requirements_satisfied),
        result.predicate_bottleneck_score,
        result.requirement_coverage,
        result.match_score,
        -source_frame_id,
    )


@dataclass(frozen=True, slots=True)
class VisualVerificationFailure:
    """Bounded diagnostic for one candidate that failed primary and retry attempts."""

    video_id: str
    absolute_frame_id: int
    primary_error: str
    retry_error: str

    def __post_init__(self) -> None:
        if not self.video_id.strip() or self.absolute_frame_id < 0:
            raise ValueError("invalid visual verification failure identity")

    def to_trace(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "video_id": self.video_id,
                "absolute_frame_id": self.absolute_frame_id,
                "attempt_count": 2,
                "primary_error": self.primary_error,
                "retry_error": self.retry_error,
            }
        )


class StructuredVisualVerifier(Protocol):
    identifiers: Mapping[str, Any]

    def verify(
        self,
        *,
        query_vi: str,
        query_en: str,
        candidates: Sequence[VisualVerificationInput],
    ) -> tuple[VisualVerificationResult, ...]: ...


def _bounded_error(exc: BaseException, *, limit: int = 500) -> str:
    rendered = f"{type(exc).__name__}: {exc}"
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def _bounded_generated_response(text: str, *, limit: int = 600) -> str:
    rendered = " ".join(text.split())
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VisualVerificationError(f"duplicate JSON key from visual verifier: {key}")
        result[key] = value
    return result


def parse_visual_verification_json(
    text: str,
    *,
    video_id: str,
    absolute_frame_id: int,
    predicate_contract: Sequence[VisualPredicateRequirement] | None = None,
) -> VisualVerificationResult:
    """Parse one bounded JSON object; markdown fences and surrounding prose are tolerated."""
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise VisualVerificationError("visual verifier response did not contain a JSON object")
    try:
        payload = json.loads(
            stripped[start : end + 1],
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise VisualVerificationError(f"invalid visual verifier JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VisualVerificationError("visual verifier response must be an object")
    def aliased_value(canonical: str, compact: str, *, required: bool = True) -> Any:
        if canonical in payload and compact in payload:
            raise VisualVerificationError(
                f"visual verifier response contains both {canonical!r} and {compact!r}"
            )
        if canonical in payload:
            return payload[canonical]
        if compact in payload:
            return payload[compact]
        if required:
            raise VisualVerificationError(
                f"visual verifier response is missing {canonical!r}"
            )
        return ""

    raw_predicates = aliased_value("predicates", "p")
    if not isinstance(raw_predicates, list) or not raw_predicates:
        raise VisualVerificationError("visual verifier predicates must be a non-empty list")
    predicates: list[VisualPredicateScore] = []
    compact_state_wire = False
    ordered_state_wire = False
    contract_by_id = {
        item.predicate_id: item for item in (predicate_contract or ())
    }
    if len(contract_by_id) != len(predicate_contract or ()):
        raise ValueError("visual predicate contract contains duplicate IDs")
    seen_predicate_ids: set[str] = set()
    for index, item in enumerate(raw_predicates, start=1):
        try:
            if predicate_contract is not None:
                if isinstance(item, list) and len(item) == 3:
                    if index > len(predicate_contract):
                        raise ValueError("ordered state wire contains too many predicates")
                    state, observed, image_number = item
                    expected = predicate_contract[index - 1]
                    predicate_id = expected.predicate_id
                    compact_state_wire = True
                    ordered_state_wire = True
                    (
                        visible,
                        satisfied,
                        score,
                        observed_value,
                        evidence,
                    ) = _ordered_predicate_observation(
                        requirement=expected,
                        state=state,
                        observed=observed,
                        image_number=image_number,
                    )
                    seen_predicate_ids.add(predicate_id)
                    predicates.append(
                        VisualPredicateScore(
                            requirement=expected.requirement,
                            score=score,
                            visible=visible,
                            evidence=evidence,
                            predicate_id=predicate_id,
                            observed_value=observed_value,
                            satisfied=satisfied,
                            comparison=expected.comparison,
                            expected_value=expected.expected_value,
                        )
                    )
                    continue
                if isinstance(item, dict):
                    def predicate_value(
                        canonical: str,
                        *aliases: str,
                        required: bool = True,
                    ) -> Any:
                        present = [
                            key for key in (canonical, *aliases) if key in item
                        ]
                        if len(present) > 1:
                            values = {str(item[key]) for key in present}
                            if len(values) > 1:
                                raise ValueError(
                                    f"conflicting predicate field aliases {present}"
                                )
                        if present:
                            return item[present[0]]
                        if required:
                            raise KeyError(canonical)
                        return None

                    if (
                        "id" in item
                        and "predicate_id" in item
                        and str(item["id"]) != str(item["predicate_id"])
                    ):
                        raise ValueError(
                            "conflicting predicate ID aliases 'id' and "
                            "'predicate_id'"
                        )
                    predicate_id = str(
                        predicate_value("id", "predicate_id", "i")
                    )
                    visible = predicate_value("visible", "v")
                    satisfied = predicate_value("satisfied", "x", "sat")
                    observed_value = predicate_value(
                        "observed_value",
                        "short_observed_fact",
                        "observed",
                        "o",
                    )
                    evidence = predicate_value(
                        "evidence",
                        "short_evidence",
                        "e",
                    )
                    reported_score = predicate_value(
                        "score",
                        "confidence",
                        "sc",
                        required=False,
                    )
                    try:
                        parsed_score = float(reported_score)
                    except (TypeError, ValueError):
                        parsed_score = math.nan
                    # The fixed contract is a boolean evidence gate, not a score
                    # calibration benchmark. Small VLMs often omit this score or
                    # emit a count/percentage. Preserve only a valid unit score.
                    score = (
                        parsed_score
                        if math.isfinite(parsed_score) and 0.0 <= parsed_score <= 1.0
                        else (1.0 if visible is True and satisfied is True else 0.0)
                    )
                elif isinstance(item, list) and len(item) == 6:
                    (
                        predicate_id,
                        _reported_score,
                        visible,
                        satisfied,
                        observed_value,
                        evidence,
                    ) = item
                    predicate_id = str(predicate_id)
                    try:
                        parsed_score = float(_reported_score)
                    except (TypeError, ValueError):
                        parsed_score = math.nan
                    score = (
                        parsed_score
                        if math.isfinite(parsed_score) and 0.0 <= parsed_score <= 1.0
                        else (1.0 if visible is True and satisfied is True else 0.0)
                    )
                elif isinstance(item, list) and len(item) == 5:
                    (
                        predicate_id,
                        visible,
                        satisfied,
                        observed_value,
                        evidence,
                    ) = item
                    predicate_id = str(predicate_id)
                    score = 1.0 if visible is True and satisfied is True else 0.0
                elif isinstance(item, list) and len(item) == 4:
                    predicate_id, state, observed_value, image_number = item
                    predicate_id = str(predicate_id)
                    if state not in {"Y", "N", "U"}:
                        raise ValueError("compact predicate state must be Y, N, or U")
                    compact_state_wire = True
                    visible = state in {"Y", "N"}
                    satisfied = state == "Y"
                    score = 1.0 if satisfied else 0.0
                    if state == "U":
                        observed_value = "unknown"
                        evidence = "not visible"
                    else:
                        if type(image_number) is not int or image_number <= 0:
                            raise ValueError(
                                "grounded compact predicate requires a positive image number"
                            )
                        observed_value = str(observed_value)
                        evidence = f"image {image_number}: {observed_value}"
                else:
                    raise TypeError(
                        "must be an object or compact "
                        "[id, visible, satisfied, observed_value, evidence] or "
                        "[id, state, observed_value, image_number] array"
                    )
                if predicate_id in seen_predicate_ids:
                    raise ValueError(f"duplicate predicate ID {predicate_id!r}")
                seen_predicate_ids.add(predicate_id)
                expected = contract_by_id.get(predicate_id)
                if expected is None:
                    raise ValueError(f"unexpected predicate ID {predicate_id!r}")
                predicates.append(
                    VisualPredicateScore(
                        requirement=expected.requirement,
                        score=float(score),
                        visible=visible,
                        evidence=str(evidence),
                        predicate_id=predicate_id,
                        observed_value=str(observed_value),
                        satisfied=satisfied,
                        comparison=expected.comparison,
                        expected_value=expected.expected_value,
                    )
                )
                continue
            if isinstance(item, dict):
                requirement = item["requirement"]
                score = item["score"]
                visible = item["visible"]
                evidence = item.get("evidence", "")
            elif isinstance(item, list) and len(item) in {3, 4}:
                requirement, score, visible = item[:3]
                evidence = item[3] if len(item) == 4 else ""
            else:
                raise TypeError(
                    "must be an object or a compact [requirement, score, visible] array"
                )
            predicates.append(
                VisualPredicateScore(
                    requirement=str(requirement),
                    score=float(score),
                    visible=visible,
                    evidence=str(evidence),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VisualVerificationError(f"invalid predicate {index}: {exc}") from exc
    try:
        if predicate_contract is not None and seen_predicate_ids != set(contract_by_id):
            missing = sorted(set(contract_by_id) - seen_predicate_ids)
            raise VisualVerificationError(
                f"visual verifier response is missing predicate IDs: {missing}"
            )
        strict_coverage = (
            sum(item.strictly_satisfied for item in predicates) / len(predicates)
            if predicate_contract is not None
            else 0.0
        )
        strict_all = all(item.strictly_satisfied for item in predicates)
        if ordered_state_wire and set(payload) != {"p"}:
            raise VisualVerificationError(
                "ordered state wire must contain only the root key 'p'"
            )
        if compact_state_wire:
            reported_all = strict_all
            reported_coverage = strict_coverage
            if ordered_state_wire:
                match_score = strict_coverage
            else:
                raw_match_score = aliased_value("match_score", "m", required=False)
                match_score = (
                    strict_coverage if raw_match_score == "" else float(raw_match_score)
                )
        else:
            reported_all = aliased_value(
                "all_visible_requirements_satisfied", "a"
            )
            if type(reported_all) is not bool:
                raise VisualVerificationError(
                    "all_visible_requirements_satisfied must be boolean"
                )
            reported_coverage = float(
                aliased_value("requirement_coverage", "c")
            )
            if predicate_contract is None:
                strict_coverage = reported_coverage
            match_score = float(aliased_value("match_score", "m"))
        return VisualVerificationResult(
            video_id=video_id,
            absolute_frame_id=absolute_frame_id,
            match_score=match_score,
            requirement_coverage=min(reported_coverage, strict_coverage),
            all_visible_requirements_satisfied=(
                reported_all and strict_all
                if predicate_contract is not None
                else reported_all
            ),
            predicates=tuple(predicates),
            summary=str(aliased_value("summary", "s", required=False)),
            contract_validated=predicate_contract is not None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VisualVerificationError(f"invalid visual verifier result: {exc}") from exc


class HuggingFaceStructuredVisualVerifier:
    """Optional local Hugging Face VLM adapter loaded once per operational session."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None,
        device: str,
        allow_model_download: bool,
        cache_dir: Path | None,
        max_new_tokens: int,
        max_image_pixels: int | None = None,
        execution_profile: str = "full",
        progress_callback: Callable[[str], None] | None = None,
        transformers_module: Any | None = None,
        torch_module: Any | None = None,
        image_module: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("visual verifier model_name must not be empty")
        if device not in {"cpu", "cuda"}:
            raise ValueError("visual verifier device must be cpu or cuda")
        if max_new_tokens <= 0:
            raise ValueError("visual verifier max_new_tokens must be positive")
        if max_image_pixels is not None and max_image_pixels <= 0:
            raise ValueError("visual verifier max_image_pixels must be positive")
        try:
            transformers = transformers_module or importlib.import_module("transformers")
            torch = torch_module or importlib.import_module("torch")
            image_api = image_module or importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise VisualVerificationError(
                f"visual verifier optional dependency unavailable: {exc}"
            ) from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise VisualVerificationError("CUDA visual verification requested but unavailable")
        processor_class = getattr(transformers, "AutoProcessor", None)
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            model_class = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
        if processor_class is None or model_class is None:
            raise VisualVerificationError(
                "installed transformers lacks an image-text generation model adapter"
            )
        load_kwargs: dict[str, Any] = {
            "revision": revision,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "local_files_only": not allow_model_download,
        }
        load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
        processor_kwargs = dict(load_kwargs)
        if max_image_pixels is not None:
            processor_kwargs["max_pixels"] = max_image_pixels
        try:
            self._processor = processor_class.from_pretrained(
                model_name,
                **processor_kwargs,
            )
            dtype = torch.float16 if device == "cuda" else torch.float32
            model_load_kwargs = {
                **load_kwargs,
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
            }
            if device == "cuda":
                # Dispatch directly while materializing the checkpoint.  Loading the
                # full VLM on CPU and only then calling ``to("cuda")`` temporarily
                # keeps two model-sized allocations alive and has repeatedly stalled
                # Kaggle T4 sessions during Qwen checkpoint materialization.
                model_load_kwargs["device_map"] = device
            self._model = model_class.from_pretrained(
                model_name,
                **model_load_kwargs,
            )
            if device == "cpu":
                self._model = self._model.to(device)
            self._model.eval()
        except Exception as exc:
            raise VisualVerificationError(f"visual verifier model load failed: {exc}") from exc
        self._torch = torch
        self._image_api = image_api
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._progress_callback = progress_callback
        self._last_failures: tuple[VisualVerificationFailure, ...] = ()
        self._last_recovered_retries: tuple[Mapping[str, Any], ...] = ()
        self._last_predicate_contract: tuple[VisualPredicateRequirement, ...] = ()
        self.identifiers: Mapping[str, Any] = MappingProxyType(
            {
                "provider": "huggingface-structured-visual-verifier",
                "model": model_name,
                "revision": revision,
                "device": device,
                "model_download_allowed": allow_model_download,
                "candidate_batching": "one-temporal-candidate-per-generation",
                "execution_profile": execution_profile,
                "max_new_tokens": max_new_tokens,
                "max_image_pixels": max_image_pixels,
                "wire_format": "compact-json-v10-temporal-witness-arrays",
            }
        )

    @property
    def last_failures(self) -> tuple[VisualVerificationFailure, ...]:
        """Candidate-local failures from the most recent verify call."""

        return self._last_failures

    @property
    def last_recovered_retries(self) -> tuple[Mapping[str, Any], ...]:
        """Successful bounded retries from the most recent verify call."""

        return self._last_recovered_retries

    @property
    def last_predicate_contract(self) -> tuple[VisualPredicateRequirement, ...]:
        return self._last_predicate_contract

    def verify(
        self,
        *,
        query_vi: str,
        query_en: str,
        candidates: Sequence[VisualVerificationInput],
    ) -> tuple[VisualVerificationResult, ...]:
        """Verify frames independently so one malformed result is not batch-fatal.

        Inputs retain absolute original-video frame IDs. The return value contains only
        successful results; bounded candidate failures and recovered retries are exposed
        through diagnostic properties for the caller's explicit policy handling.
        """

        self._last_failures = ()
        self._last_recovered_retries = ()
        if not query_vi.strip() or not query_en.strip():
            raise ValueError("visual verification requires Vietnamese and English query text")
        self._last_predicate_contract = compile_visual_predicate_contract(
            query_vi=query_vi,
            query_en=query_en,
        )
        results: list[VisualVerificationResult] = []
        failures: list[VisualVerificationFailure] = []
        recovered_retries: list[Mapping[str, Any]] = []
        total = len(candidates)
        for index, candidate in enumerate(candidates, start=1):
            started = time.perf_counter()
            self._progress(
                f"visual verifier candidate {index}/{total} started: "
                f"{candidate.video_id}/{candidate.absolute_frame_id} "
                f"images={len(candidate.images)}"
            )
            try:
                result = self._verify_candidate(
                    query_vi=query_vi,
                    query_en=query_en,
                    candidate=candidate,
                    images=candidate.images,
                    max_new_tokens=self._max_new_tokens,
                )
            except Exception as primary_exc:
                retry_index = candidate.candidate_image_number - 1
                retry_images = (candidate.images[retry_index],)
                # Keep the configured generation budget.  The former 192-token retry
                # cap could truncate otherwise valid JSON for conjunction-heavy queries.
                # The retry is bounded by one center image and the same caller-approved
                # token budget; the compact wire schema keeps its output small.
                retry_max_tokens = self._max_new_tokens
                self._progress(
                    f"visual verifier candidate {index}/{total} primary failed: "
                    f"{type(primary_exc).__name__}: {primary_exc}; retrying with "
                    f"images=1 max_new_tokens={retry_max_tokens}"
                )
                try:
                    result = self._verify_candidate(
                        query_vi=query_vi,
                        query_en=query_en,
                        candidate=candidate,
                        images=retry_images,
                        max_new_tokens=retry_max_tokens,
                    )
                except Exception as retry_exc:
                    failures.append(
                        VisualVerificationFailure(
                            video_id=candidate.video_id,
                            absolute_frame_id=candidate.absolute_frame_id,
                            primary_error=_bounded_error(primary_exc),
                            retry_error=_bounded_error(retry_exc),
                        )
                    )
                    self._progress(
                        f"visual verifier candidate {index}/{total} failed after "
                        f"retry in {time.perf_counter() - started:.2f}s: "
                        f"{type(retry_exc).__name__}: {retry_exc}"
                    )
                    continue
                recovered_retries.append(
                    MappingProxyType(
                        {
                            "video_id": candidate.video_id,
                            "absolute_frame_id": candidate.absolute_frame_id,
                            "primary_error": _bounded_error(primary_exc),
                            "retry_image_count": 1,
                            "retry_max_new_tokens": retry_max_tokens,
                        }
                    )
                )
                self._progress(
                    f"visual verifier candidate {index}/{total} recovered on retry"
                )
            results.append(result)
            self._progress(
                f"visual verifier candidate {index}/{total} completed in "
                f"{time.perf_counter() - started:.2f}s"
            )
        # Overlapping temporal windows can choose the same absolute witness frame.
        # Keep only the strongest deterministic observation for that shared identity.
        unique_results: dict[tuple[str, int], VisualVerificationResult] = {}
        for result in results:
            identity = (result.video_id, result.absolute_frame_id)
            existing = unique_results.get(identity)
            if existing is None or _visual_result_preference(result) > (
                _visual_result_preference(existing)
            ):
                unique_results[identity] = result
        self._last_failures = tuple(failures)
        self._last_recovered_retries = tuple(recovered_retries)
        return tuple(unique_results.values())

    def _verify_candidate(
        self,
        *,
        query_vi: str,
        query_en: str,
        candidate: VisualVerificationInput,
        images: Sequence[Any],
        max_new_tokens: int,
    ) -> VisualVerificationResult:
        predicate_contract = compile_visual_predicate_contract(
            query_vi=query_vi,
            query_en=query_en,
        )
        prompt = self._build_prompt(
            query_vi=query_vi,
            query_en=query_en,
            predicate_contract=predicate_contract,
            image_count=len(images),
            candidate_image_number=(
                candidate.candidate_image_number
                if len(images) == len(candidate.images)
                else 1
            ),
        )
        rgb_images = [self._to_rgb_image(image) for image in images]
        content = [{"type": "image"} for _ in rgb_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        try:
            rendered = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=[rendered],
                images=rgb_images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            inference_context = getattr(
                self._torch,
                "inference_mode",
                self._torch.no_grad,
            )
            with inference_context():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            input_length = int(inputs["input_ids"].shape[1])
            decoded = self._processor.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:
            raise VisualVerificationError(
                f"visual verification generation failed for "
                f"{candidate.video_id}/{candidate.absolute_frame_id}: {exc}"
            ) from exc
        try:
            parsed = parse_visual_verification_json(
                decoded,
                video_id=candidate.video_id,
                absolute_frame_id=candidate.absolute_frame_id,
                predicate_contract=predicate_contract,
            )
            if len(images) == len(candidate.images):
                return bind_temporal_visual_evidence(
                    parsed,
                    candidate=candidate,
                )
            if candidate.image_frame_ids:
                retry_candidate = VisualVerificationInput(
                    video_id=candidate.video_id,
                    absolute_frame_id=candidate.absolute_frame_id,
                    timestamp_seconds=candidate.timestamp_seconds,
                    images=tuple(images),
                    image_frame_ids=(candidate.absolute_frame_id,),
                    image_timestamps_seconds=(candidate.timestamp_seconds,),
                )
                return bind_temporal_visual_evidence(
                    parsed,
                    candidate=retry_candidate,
                )
            return parsed
        except VisualVerificationError as exc:
            response = _bounded_generated_response(decoded)
            raise VisualVerificationError(
                f"{exc}; bounded_generated_response={response!r}"
            ) from exc

    @staticmethod
    def _build_prompt(
        *,
        query_vi: str,
        query_en: str,
        predicate_contract: Sequence[VisualPredicateRequirement] | None = None,
        image_count: int | None = None,
        candidate_image_number: int | None = None,
    ) -> str:
        contract = tuple(
            predicate_contract
            or compile_visual_predicate_contract(
                query_vi=query_vi,
                query_en=query_en,
            )
        )
        contract_lines = []
        for index, item in enumerate(contract, start=1):
            comparison = (
                f"; visible count must be {item.comparison} {item.expected_value}"
                if item.comparison is not None
                else ""
            )
            contract_lines.append(
                f"{index}. {item.predicate_id}: {item.requirement}{comparison}"
            )
        predicate_requirements = "\n".join(contract_lines)
        safe_template = json.dumps(
            {
                "p": [
                    ["U", 0, 0]
                    for _item in contract
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        image_context = ""
        if image_count is not None:
            if image_count <= 0:
                raise ValueError("image_count must be positive when provided")
            if candidate_image_number is None or not (
                1 <= candidate_image_number <= image_count
            ):
                raise ValueError(
                    "candidate_image_number must identify one supplied image"
                )
            image_context = (
                f" There are exactly {image_count} images; image "
                f"{candidate_image_number} is the retrieval center."
            )
        return (
            "You are a strict visual evidence verifier for video known-item search. "
            "The images are neighboring frames from one automatically retrieved temporal "
            "candidate, ordered by time and numbered from 1."
            f"{image_context} Inspect every supplied image before answering. Score only "
            "the fixed predicate contract below. Never create, "
            "rename, merge, or omit an ID. Never infer a hidden person, "
            "attribute, count, or action. Exact counts and conjunctions matter. Return "
            "exactly one minified JSON object with no prose or markdown. The only root "
            "key is p. p must contain one three-value array per numbered requirement, in "
            "the exact listed order. Each array is exactly [STATE,VALUE,IMAGE]. STATE is Y "
            "when the visual fact is present, N when it is visibly absent, or U when "
            "uncertain/not visible. IMAGE is the one-based image number that literally "
            "supports the observation. For a count requirement, VALUE is the visible "
            "integer; code compares it with the required count, so never convert the "
            "comparison into Y/N yourself. For every other requirement, VALUE is 1 for Y "
            "and 0 for N. U must always be "
            "exactly [\"U\",0,0]. Do not emit IDs, scores, explanations, observations, "
            "field names, or any other values. Verify the exact "
            "pose or motion in the requirement, not a related exercise. All count, person-"
            "attribute, and spatial-layout requirements must cite the SAME single image. "
            "All action and synchronization requirements must cite the SAME single image. "
            "Those two witness images may differ because they belong to one bounded video "
            "moment. Never add people or attributes across images. Code rejects incoherent "
            "witnesses. Do not repeat these instructions or copy any query or requirement text. "
            "The numbered requirements below are read-only prose, not JSON.\n"
            f"Vietnamese query: {query_vi}\nEnglish translation: {query_en}\n"
            f"Required visual predicates:\n{predicate_requirements}\n"
            "Use the following complete fail-closed JSON as the output skeleton. Preserve "
            "its structure, order, and IDs. Change false/unknown/not visible only when the "
            "images literally support the replacement. Return the completed JSON now; "
            "this JSON skeleton must be the final and only object in the response:\n"
            f"{safe_template}"
        )

    def _progress(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)

    def _to_rgb_image(self, image: Any) -> Any:
        import numpy as np

        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 3:
            array = array[:, :, ::-1]
        return self._image_api.fromarray(np.asarray(array, dtype=np.uint8))
