"""Strict, query-conditioned visual answer ontology for open-entity QA."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .models import AnswerHypothesis
from .question_types import QuestionType


class VisualOntologyError(ValueError):
    """The visual ontology is malformed or cannot be loaded safely."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VisualOntologyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(without_marks.split())


def _strict_fields(
    payload: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise VisualOntologyError(
            f"{context} fields mismatch: missing={missing}, extra={extra}"
        )


def _strict_strings(value: Any, *, context: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise VisualOntologyError(f"{context} must be a non-empty list")
    resolved = tuple(value)
    if any(type(item) is not str or not item.strip() for item in resolved):
        raise VisualOntologyError(f"{context} must contain non-empty strings")
    if len(set(resolved)) != len(resolved):
        raise VisualOntologyError(f"{context} must not contain duplicates")
    return resolved


@dataclass(frozen=True, slots=True)
class VisualOntologyDomain:
    domain_id: str
    question_types: tuple[QuestionType, ...]
    activation_terms: tuple[str, ...]
    entries: tuple[AnswerHypothesis, ...]


@dataclass(frozen=True, slots=True)
class VisualAnswerOntology:
    schema_version: int
    ontology_id: str
    description: str
    domains: tuple[VisualOntologyDomain, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class VisualOntologyConfig:
    enabled: bool = False
    ontology_path: Path | None = None
    evidence_frame_budget: int = 100
    max_active_domains: int = 1

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if not 1 <= self.evidence_frame_budget <= 100:
            raise ValueError("evidence_frame_budget must be in [1, 100]")
        if self.max_active_domains <= 0:
            raise ValueError("max_active_domains must be positive")
        if self.enabled and self.ontology_path is None:
            raise ValueError("enabled visual ontology requires ontology_path")


def load_visual_answer_ontology(path: Path) -> VisualAnswerOntology:
    source = Path(path)
    raw = source.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise VisualOntologyError("visual ontology must be UTF-8 without BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise VisualOntologyError("visual ontology is not valid UTF-8") from exc
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise VisualOntologyError(f"visual ontology JSON is invalid: {exc}") from exc
    if type(payload) is not dict:
        raise VisualOntologyError("visual ontology root must be an object")
    _strict_fields(
        payload,
        {"schema_version", "ontology_id", "description", "domains"},
        context="visual ontology",
    )
    if payload["schema_version"] != 1:
        raise VisualOntologyError("visual ontology schema_version must be 1")
    ontology_id = payload["ontology_id"]
    description = payload["description"]
    if type(ontology_id) is not str or not ontology_id.strip():
        raise VisualOntologyError("ontology_id must be a non-empty string")
    if type(description) is not str or not description.strip():
        raise VisualOntologyError("description must be a non-empty string")
    domain_payloads = payload["domains"]
    if type(domain_payloads) is not list or not domain_payloads:
        raise VisualOntologyError("domains must be a non-empty list")

    domains: list[VisualOntologyDomain] = []
    seen_domains: set[str] = set()
    seen_answers: set[tuple[QuestionType, str]] = set()
    for domain_index, domain_payload in enumerate(domain_payloads):
        context = f"domains[{domain_index}]"
        if type(domain_payload) is not dict:
            raise VisualOntologyError(f"{context} must be an object")
        _strict_fields(
            domain_payload,
            {"domain_id", "question_types", "activation_terms", "entries"},
            context=context,
        )
        domain_id = domain_payload["domain_id"]
        if type(domain_id) is not str or not domain_id.strip():
            raise VisualOntologyError(f"{context}.domain_id must be non-empty")
        if domain_id in seen_domains:
            raise VisualOntologyError(f"duplicate domain_id: {domain_id}")
        seen_domains.add(domain_id)
        raw_types = _strict_strings(
            domain_payload["question_types"], context=f"{context}.question_types"
        )
        try:
            question_types = tuple(QuestionType(item) for item in raw_types)
        except ValueError as exc:
            raise VisualOntologyError(
                f"{context}.question_types contains an unsupported type"
            ) from exc
        activation_terms = _strict_strings(
            domain_payload["activation_terms"],
            context=f"{context}.activation_terms",
        )
        entry_payloads = domain_payload["entries"]
        if type(entry_payloads) is not list or not entry_payloads:
            raise VisualOntologyError(f"{context}.entries must be a non-empty list")
        entries: list[AnswerHypothesis] = []
        for entry_index, entry_payload in enumerate(entry_payloads):
            entry_context = f"{context}.entries[{entry_index}]"
            if type(entry_payload) is not dict:
                raise VisualOntologyError(f"{entry_context} must be an object")
            _strict_fields(
                entry_payload,
                {"canonical_answer", "aliases", "visual_prompts"},
                context=entry_context,
            )
            canonical_answer = entry_payload["canonical_answer"]
            if type(canonical_answer) is not str or not canonical_answer.strip():
                raise VisualOntologyError(
                    f"{entry_context}.canonical_answer must be non-empty"
                )
            aliases = _strict_strings(
                entry_payload["aliases"], context=f"{entry_context}.aliases"
            )
            visual_prompts = _strict_strings(
                entry_payload["visual_prompts"],
                context=f"{entry_context}.visual_prompts",
            )
            normalized_answer = _normalize_text(canonical_answer)
            for question_type in question_types:
                identity = (question_type, normalized_answer)
                if identity in seen_answers:
                    raise VisualOntologyError(
                        "canonical answer repeats across domains for question type: "
                        f"{question_type.value}/{canonical_answer}"
                    )
                seen_answers.add(identity)
            entries.append(
                AnswerHypothesis(
                    canonical_answer=canonical_answer,
                    aliases=aliases,
                    visual_prompts=visual_prompts,
                )
            )
        domains.append(
            VisualOntologyDomain(
                domain_id=domain_id,
                question_types=question_types,
                activation_terms=activation_terms,
                entries=tuple(entries),
            )
        )
    return VisualAnswerOntology(
        schema_version=1,
        ontology_id=ontology_id,
        description=description,
        domains=tuple(domains),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


class VisualOntologyAnswerCandidateProvider:
    """Select a bounded visual vocabulary from explicit question intent terms."""

    def __init__(
        self,
        ontology: VisualAnswerOntology,
        config: VisualOntologyConfig,
    ) -> None:
        self.ontology = ontology
        self.config = config
        self.identifiers: Mapping[str, Any] = MappingProxyType(
            {
                "provider": "visual-answer-ontology",
                "schema_version": ontology.schema_version,
                "ontology_id": ontology.ontology_id,
                "ontology_sha256": ontology.sha256,
                "evidence_frame_budget": config.evidence_frame_budget,
                "max_active_domains": config.max_active_domains,
            }
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def supports(self, question_type: QuestionType) -> bool:
        return self.enabled and any(
            question_type in domain.question_types for domain in self.ontology.domains
        )

    def get_candidates(self, question_type: QuestionType) -> tuple[AnswerHypothesis, ...]:
        if not self.supports(question_type):
            return ()
        return tuple(
            entry
            for domain in self.ontology.domains
            if question_type in domain.question_types
            for entry in domain.entries
        )

    def get_candidates_for_query(
        self,
        question_type: QuestionType,
        question_text: str,
    ) -> tuple[AnswerHypothesis, ...]:
        if not self.supports(question_type) or not question_text.strip():
            return ()
        normalized_question = _normalize_text(question_text)
        matches: list[tuple[int, str, VisualOntologyDomain]] = []
        for domain in self.ontology.domains:
            if question_type not in domain.question_types:
                continue
            matched_lengths = [
                len(normalized_term)
                for term in domain.activation_terms
                if (normalized_term := _normalize_text(term)) in normalized_question
            ]
            if matched_lengths:
                matches.append((-max(matched_lengths), domain.domain_id, domain))
        matches.sort(key=lambda item: (item[0], item[1]))
        selected = matches[: self.config.max_active_domains]
        return tuple(entry for _length, _domain_id, domain in selected for entry in domain.entries)

    def active_domain_ids(
        self,
        question_type: QuestionType,
        question_text: str,
    ) -> tuple[str, ...]:
        selected = self.get_candidates_for_query(question_type, question_text)
        identities = {candidate.canonical_answer for candidate in selected}
        return tuple(
            domain.domain_id
            for domain in self.ontology.domains
            if any(entry.canonical_answer in identities for entry in domain.entries)
        )
