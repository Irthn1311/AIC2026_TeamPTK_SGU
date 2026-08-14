from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from system_tai.kis.session_schema import SessionConfig
from system_tai.qa.engine import QABaselineEngine
from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
from system_tai.qa.models import AnswerHypothesis, QAEvidenceCandidate, QAQuery
from system_tai.qa.object_provider import ObjectAnswerProviderConfig
from system_tai.qa.question_types import QuestionType
from system_tai.qa.visual_ontology import (
    VisualAnswerOntology,
    VisualOntologyAnswerCandidateProvider,
    VisualOntologyConfig,
    VisualOntologyError,
    load_visual_answer_ontology,
)

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
FROZEN_ONTOLOGY = (
    SYSTEM_ROOT
    / "benchmarks"
    / "l21_150_diagnostic"
    / "qa_dev_visual_ontology.json"
)


def _provider() -> VisualOntologyAnswerCandidateProvider:
    config = VisualOntologyConfig(enabled=True, ontology_path=FROZEN_ONTOLOGY)
    return VisualOntologyAnswerCandidateProvider(
        load_visual_answer_ontology(FROZEN_ONTOLOGY), config
    )


def test_frozen_visual_ontology_is_target_blind_and_selects_crop_domain() -> None:
    raw = FROZEN_ONTOLOGY.read_text(encoding="utf-8")
    forbidden = (
        '"query_id"',
        '"video_id"',
        '"frame_id"',
        '"accepted_answers"',
        '"holdout"',
    )
    assert all(token not in raw.casefold() for token in forbidden)

    provider = _provider()
    candidates = provider.get_candidates_for_query(
        QuestionType.OBJECT_ENTITY,
        "What crop is being harvested by the machine?",
    )
    assert provider.active_domain_ids(
        QuestionType.OBJECT_ENTITY,
        "What crop is being harvested by the machine?",
    ) == ("crop",)
    assert {candidate.canonical_answer for candidate in candidates} == {
        "lúa",
        "ngô",
        "lúa mì",
        "mía",
        "đậu nành",
        "rau",
        "cỏ",
    }
    assert "xe tải" not in {candidate.canonical_answer for candidate in candidates}


@pytest.mark.parametrize(
    ("question", "expected_domain"),
    [
        ("Đoàn xe này là loại xe gì?", "vehicle_type"),
        ("Khán giả dùng thiết bị gì để chụp?", "imaging_device"),
    ],
)
def test_visual_ontology_domain_selection_is_query_conditioned(
    question: str,
    expected_domain: str,
) -> None:
    provider = _provider()
    assert provider.active_domain_ids(QuestionType.OBJECT_ENTITY, question) == (
        expected_domain,
    )


def test_visual_ontology_returns_no_candidates_for_unmatched_intent() -> None:
    provider = _provider()
    assert provider.get_candidates_for_query(
        QuestionType.OBJECT_ENTITY,
        "What unknown thing is shown?",
    ) == ()


def test_visual_ontology_loader_rejects_duplicate_keys_and_bom(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1}', encoding="utf-8"
    )
    with pytest.raises(VisualOntologyError, match="duplicate JSON key"):
        load_visual_answer_ontology(duplicate)

    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf{}")
    with pytest.raises(VisualOntologyError, match="without BOM"):
        load_visual_answer_ontology(bom)


def test_visual_ontology_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = json.loads(FROZEN_ONTOLOGY.read_text(encoding="utf-8"))
    payload["unknown"] = True
    destination = tmp_path / "unknown.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VisualOntologyError, match="fields mismatch"):
        load_visual_answer_ontology(destination)


def test_visual_ontology_fingerprint_matches_source_bytes() -> None:
    ontology = load_visual_answer_ontology(FROZEN_ONTOLOGY)
    assert ontology.sha256 == (
        "fc19f4ca1ce2e4960463ba054be2f9c351cf874867eb65a2ef9ce2252d644ddc"
    )


def test_engine_scores_visual_answer_from_explicit_candidate_provider() -> None:
    hypothesis = AnswerHypothesis(
        canonical_answer="lúa",
        aliases=("rice",),
        visual_prompts=("rice crop",),
    )
    query = QAQuery(
        query_id="QA-11",
        event_description="harvesting",
        question="what crop?",
        question_type=QuestionType.OBJECT_ENTITY,
    )
    evidence = (
        QAEvidenceCandidate(
            query_id="QA-11",
            rank=1,
            video_id="L21_V005",
            frame_id=537,
            retrieval_score=0.9,
        ),
    )
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    class StaticProvider:
        def get_candidates(
            self, question_type: QuestionType
        ) -> tuple[AnswerHypothesis, ...]:
            assert question_type is QuestionType.OBJECT_ENTITY
            return (hypothesis,)

    result = QABaselineEngine(candidate_provider=StaticProvider()).answer(
        query,
        evidence,
        image_embeddings={("L21_V005", 537): vector},
        prompt_embeddings={"rice crop": vector},
    )
    observed = [
        (item.rank, item.video_id, item.frame_id, item.answer)
        for item in result.predictions
    ]
    assert observed == [(1, "L21_V005", 537, "lúa")]


def test_visual_ontology_config_is_opt_in_and_bounded() -> None:
    assert VisualOntologyConfig().enabled is False
    with pytest.raises(ValueError, match="requires ontology_path"):
        VisualOntologyConfig(enabled=True)
    with pytest.raises(ValueError, match=r"\[1, 100\]"):
        VisualOntologyConfig(evidence_frame_budget=101)


def test_session_config_requires_grounding_and_rejects_object_conflict() -> None:
    visual = VisualOntologyConfig(enabled=True, ontology_path=FROZEN_ONTOLOGY)
    with pytest.raises(ValueError, match="video-conditioned"):
        SessionConfig(qa_visual_ontology_config=visual)
    with pytest.raises(ValueError, match="mutually exclusive"):
        SessionConfig(
            qa_video_conditioned_evidence_config=QAVideoConditionedEvidenceConfig(
                enabled=True
            ),
            qa_object_answer_provider_config=ObjectAnswerProviderConfig(enabled=True),
            qa_visual_ontology_config=visual,
        )


def test_provider_identity_contains_only_bounded_provenance() -> None:
    provider = _provider()
    assert provider.identifiers == {
        "provider": "visual-answer-ontology",
        "schema_version": 1,
        "ontology_id": "system-tai-l21-qa-dev-visual-ontology-v1",
        "ontology_sha256": (
            "fc19f4ca1ce2e4960463ba054be2f9c351cf874867eb65a2ef9ce2252d644ddc"
        ),
        "evidence_frame_budget": 100,
        "max_active_domains": 1,
    }


def test_visual_ontology_model_is_immutable() -> None:
    ontology = load_visual_answer_ontology(FROZEN_ONTOLOGY)
    assert isinstance(ontology, VisualAnswerOntology)
    with pytest.raises(AttributeError):
        ontology.ontology_id = "mutated"  # type: ignore[misc]


def test_visual_ontology_fallback_to_baseline_candidates_for_color_and_count() -> None:
    provider = _provider()
    assert provider.supports(QuestionType.COLOR) is True
    assert provider.supports(QuestionType.COUNT) is True
    assert provider.supports(QuestionType.YES_NO) is True
    assert provider.supports(QuestionType.DIRECTION) is True

    color_candidates = provider.get_candidates_for_query(
        QuestionType.COLOR, "What color is the shirt?"
    )
    assert len(color_candidates) == 11
    assert "trắng" in {c.canonical_answer for c in color_candidates}
    assert "xanh lá" in {c.canonical_answer for c in color_candidates}

    count_candidates = provider.get_candidates_for_query(
        QuestionType.COUNT, "How many dogs are visible?"
    )
    assert len(count_candidates) == 11
    assert "2" in {c.canonical_answer for c in count_candidates}


def test_distill_qa_scene_prompt_extracts_visual_query() -> None:
    from system_tai.qa.grounding import distill_qa_scene_prompt

    assert (
        distill_qa_scene_prompt(
            "In the scene with a yellow-red warning sign, "
            "what danger is the sign warning about?"
        )
        == "a yellow-red warning sign"
    )
    assert (
        distill_qa_scene_prompt(
            "In the scene where the man in a striped shirt is working "
            "with an officer, what is he sitting behind?"
        )
        == "the man in a striped shirt is working with an officer"
    )
    assert (
        distill_qa_scene_prompt(
            "In the scene with a person walking dogs, "
            "how many dogs are clearly visible?"
        )
        == "a person walking dogs"
    )
    assert (
        distill_qa_scene_prompt(
            "What color is the shirt worn by the woman being interviewed?"
        )
        == "shirt worn by the woman being interviewed"
    )
    assert (
        distill_qa_scene_prompt(
            "What type of vehicles mainly make up "
            "the convoy of white vehicles in the scene?"
        )
        == "the convoy of white vehicles"
    )


def test_question_classifier_recognizes_new_color_and_entity_patterns() -> None:
    from system_tai.qa.question_types import classify_question

    assert (
        classify_question("Nữ MC mặc trang phục màu gì?").question_type
        == QuestionType.COLOR
    )
    assert (
        classify_question("Biểu tượng hình cầu có hai màu gì?").question_type
        == QuestionType.COLOR
    )
    assert (
        classify_question("Cậu bé đang dùng dụng cụ gì để quan sát?").question_type
        == QuestionType.OBJECT_ENTITY
    )
    assert (
        classify_question("Cuộc đua trong bùn sử dụng con vật nào?").question_type
        == QuestionType.OBJECT_ENTITY
    )
