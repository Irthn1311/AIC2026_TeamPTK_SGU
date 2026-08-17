import numpy as np
import pytest

from system_tai.preliminary.schemas import QAPrediction
from system_tai.qa.engine import QABaselineEngine
from system_tai.qa.models import QAEvidenceCandidate, QAQuery, AnswerHypothesis
from system_tai.refinement.models import RefinedCandidate
from system_tai.qa.question_types import QuestionType
from system_tai.qa.top100_constructor import construct_ranked_qa_top100
from system_tai.qa.answer_candidates import AnswerCandidateProvider
from system_tai.qa.candidate_selectors import select_fourth_unique_primary_candidate


class MockCandidateProvider(AnswerCandidateProvider):
    def get_candidates(self, question_type: QuestionType) -> list[AnswerHypothesis]:
        return [
            AnswerHypothesis(canonical_answer="3", visual_prompts=("3",)),
            AnswerHypothesis(canonical_answer="2", visual_prompts=("2",)),
            AnswerHypothesis(canonical_answer="4", visual_prompts=("4",)),
        ]

    def get_candidates_for_query(self, question_type: QuestionType, query_text: str) -> list[AnswerHypothesis]:
        return self.get_candidates(question_type)


class MockCosineScorer:
    def score_answers(self, candidate, hypotheses, image_embedding=None, prompt_embeddings=None):
        # Deterministic dummy scoring: for frame 14631 return ['3', '2', '4'], for others return ['3', '4', '2']
        if candidate.frame_id == 14631:
            return [
                (AnswerHypothesis(canonical_answer="3"), 0.22),
                (AnswerHypothesis(canonical_answer="2"), 0.21),
                (AnswerHypothesis(canonical_answer="4"), 0.20),
            ]
        return [
            (AnswerHypothesis(canonical_answer="3"), 0.23),
            (AnswerHypothesis(canonical_answer="4"), 0.22),
            (AnswerHypothesis(canonical_answer="2"), 0.21),
        ]


def test_qa_count_far_alt_micro_default_off():
    scored_candidates = [
        {"video_id": f"V{i:02d}", "frame_id": 1000 + i * 100, "answers": ["3", "4", "2"], "scores": [0.3, 0.2, 0.1], "video_nomination_rank": i, "local_anchor_rank": 1}
        for i in range(1, 11)
    ]
    aux_candidates = [
        {"video_id": "V04", "frame_id": 1490, "answers": ["3", "2", "4"], "scores": [0.22, 0.21, 0.20], "candidate_rank": 4}
    ]

    preds_off = construct_ranked_qa_top100(
        query_id="TEST-01",
        scored_candidates=scored_candidates,
        count_far_alt_micro=False,
        auxiliary_count_far_candidates=aux_candidates,
    )
    # When OFF, no slot with TIER5_COUNT_FAR_ALT_MICRO is emitted
    assert not any(p.video_id == "V04" and p.frame_id == 1490 and p.answer == "2" for p in preds_off)


def test_qa_count_far_alt_micro_only_emits_answers_1():
    scored_candidates = [
        {"video_id": f"V{i:02d}", "frame_id": 1000 + i * 100, "answers": ["3", "4", "2"], "scores": [0.3, 0.2, 0.1], "video_nomination_rank": i, "local_anchor_rank": 1}
        for i in range(1, 11)
    ]
    aux_candidates = [
        {"video_id": "V04", "frame_id": 1490, "answers": ["3", "2", "4"], "scores": [0.22, 0.21, 0.20], "candidate_rank": 4}
    ]

    preds_on, prov = construct_ranked_qa_top100(
        query_id="TEST-01",
        scored_candidates=scored_candidates,
        secondary_temporal_micro_budget=True,
        count_far_alt_micro=True,
        auxiliary_count_far_candidates=aux_candidates,
        return_provenance=True,
    )

    micro_records = [p for p in prov if p.get("slot_source") == "TIER5_COUNT_FAR_ALT_MICRO"]
    assert len(micro_records) == 1
    assert micro_records[0]["video_id"] == "V04"
    assert micro_records[0]["frame_id"] == 1490
    assert micro_records[0]["answer"] == "2"  # Must be answers[1], NOT answers[0] ('3')


def test_qa_count_far_alt_micro_occupies_slot_before_tier5_phase_b():
    scored_candidates = [
        {"video_id": f"V{i:02d}", "frame_id": 1000 + i * 100, "answers": ["3", "4", "2"], "scores": [0.3, 0.2, 0.1], "video_nomination_rank": i, "local_anchor_rank": 1}
        for i in range(1, 11)
    ]
    for i in range(1, 11):
        scored_candidates.append(
            {"video_id": f"V{i:02d}", "frame_id": 5000 + i * 100, "answers": ["3", "4", "2"], "scores": [0.25, 0.15, 0.05], "video_nomination_rank": i, "local_anchor_rank": 2}
        )
    scored_candidates.append(
        {"video_id": "V11", "frame_id": 11000, "answers": ["3", "4", "2"], "scores": [0.2, 0.1, 0.05], "video_nomination_rank": 11, "local_anchor_rank": 1}
    )
    scored_candidates.append(
        {"video_id": "V12", "frame_id": 12000, "answers": ["3", "4", "2"], "scores": [0.19, 0.09, 0.04], "video_nomination_rank": 12, "local_anchor_rank": 1}
    )

    aux_candidates = [
        {"video_id": "V04", "frame_id": 1490, "answers": ["3", "2", "4"], "scores": [0.22, 0.21, 0.20], "candidate_rank": 4}
    ]

    preds, prov = construct_ranked_qa_top100(
        query_id="TEST-01",
        scored_candidates=scored_candidates,
        secondary_temporal_micro_budget=True,
        primary_11_12_micro_coverage=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=True,
        count_far_alt_micro=True,
        auxiliary_count_far_candidates=aux_candidates,
        return_provenance=True,
    )
    assert len(preds) == 100
    micro_record = [p for p in prov if p.get("slot_source") == "TIER5_COUNT_FAR_ALT_MICRO"]
    assert len(micro_record) == 1
    assert micro_record[0]["final_rank"] == 100
    assert micro_record[0]["answer"] == "2"
    assert micro_record[0]["frame_id"] == 1490


def test_non_count_queries_ignore_count_far_alt_micro():
    cands = [
        QAEvidenceCandidate(query_id="TEST-02", rank=1, video_id="V01", frame_id=1000, retrieval_score=0.9, provenance={"video_nomination_rank": 1, "local_anchor_rank": 1}),
        QAEvidenceCandidate(query_id="TEST-02", rank=2, video_id="V02", frame_id=2000, retrieval_score=0.8, provenance={"video_nomination_rank": 2, "local_anchor_rank": 1}),
        QAEvidenceCandidate(query_id="TEST-02", rank=3, video_id="V03", frame_id=3000, retrieval_score=0.7, provenance={"video_nomination_rank": 3, "local_anchor_rank": 1}),
        QAEvidenceCandidate(query_id="TEST-02", rank=4, video_id="V04", frame_id=14541, retrieval_score=0.6, provenance={"video_nomination_rank": 4, "local_anchor_rank": 1}),
    ]
    dummy_vec = np.ones(512, dtype=np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)
    image_embeddings = {
        ("V01", 1000): dummy_vec,
        ("V02", 2000): dummy_vec,
        ("V03", 3000): dummy_vec,
        ("V04", 14541): dummy_vec,
        ("V04", 14631): dummy_vec,
    }
    prompt_embeddings = {"3": dummy_vec, "2": dummy_vec, "4": dummy_vec}

    engine = QABaselineEngine(
        candidate_provider=MockCandidateProvider(),
        scorer=MockCosineScorer(),
        expand_temporal=True,
        secondary_temporal_micro_budget=True,
        count_far_alt_micro=True,
    )

    query = QAQuery(query_id="TEST-02", event_description="Event B", question="Chiếc xe có màu gì?")
    res = engine.answer(query, cands, image_embeddings=image_embeddings, prompt_embeddings=prompt_embeddings)

    v04_far = [p for p in res.predictions if p.video_id == "V04" and p.frame_id == 14631 and p.answer == "2"]
    assert len(v04_far) == 0


def test_engine_multiseed_interleaving_selects_fourth_primary_candidate():
    cands = [
        QAEvidenceCandidate(query_id="TEST-03", rank=1, video_id="V01", frame_id=1000, retrieval_score=0.9, provenance={"video_nomination_rank": 1, "local_anchor_rank": 1}),
        QAEvidenceCandidate(query_id="TEST-03", rank=2, video_id="V01", frame_id=1030, retrieval_score=0.85, provenance={"video_nomination_rank": 1, "local_anchor_rank": 2}),
        QAEvidenceCandidate(query_id="TEST-03", rank=3, video_id="V02", frame_id=2000, retrieval_score=0.8, provenance={"video_nomination_rank": 2, "local_anchor_rank": 1}),
        QAEvidenceCandidate(query_id="TEST-03", rank=4, video_id="V02", frame_id=2030, retrieval_score=0.75, provenance={"video_nomination_rank": 2, "local_anchor_rank": 2}),
        QAEvidenceCandidate(query_id="TEST-03", rank=5, video_id="V03", frame_id=3000, retrieval_score=0.7, provenance={"video_nomination_rank": 3, "local_anchor_rank": 1}),
        QAEvidenceCandidate(query_id="TEST-03", rank=6, video_id="V03", frame_id=3030, retrieval_score=0.65, provenance={"video_nomination_rank": 3, "local_anchor_rank": 2}),
        QAEvidenceCandidate(query_id="TEST-03", rank=7, video_id="V04", frame_id=14541, retrieval_score=0.6, provenance={"video_nomination_rank": 4, "local_anchor_rank": 1}),
        QAEvidenceCandidate(query_id="TEST-03", rank=8, video_id="V04", frame_id=14571, retrieval_score=0.55, provenance={"video_nomination_rank": 4, "local_anchor_rank": 2}),
    ]

    dummy_vec = np.ones(512, dtype=np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)
    image_embeddings = {
        (c.video_id, c.frame_id): dummy_vec for c in cands
    }
    image_embeddings[("V04", 14631)] = dummy_vec
    prompt_embeddings = {"3": dummy_vec, "2": dummy_vec, "4": dummy_vec}

    engine = QABaselineEngine(
        candidate_provider=MockCandidateProvider(),
        scorer=MockCosineScorer(),
        expand_temporal=True,
        secondary_temporal_micro_budget=True,
        count_far_alt_micro=True,
    )

    query = QAQuery(query_id="TEST-03", event_description="Event C", question="Có bao nhiêu người trong ảnh?")
    res = engine.answer(query, cands, image_embeddings=image_embeddings, prompt_embeddings=prompt_embeddings)

    # Prove that V04 @ 14631 with answer '2' was emitted (fourth primary candidate)
    v04_far = [p for p in res.predictions if p.video_id == "V04" and p.frame_id == 14631 and p.answer == "2"]
    assert len(v04_far) == 1

    # Prove that V02 @ 2120 (from physical candidate 4: 2030 + 90) was NOT emitted as micro-slot
    v02_far = [p for p in res.predictions if p.video_id == "V02" and p.frame_id == 2120 and p.answer == "2"]
    assert len(v02_far) == 0


def test_selector_fails_closed_on_missing_or_invalid_metadata():
    # Candidates with missing or invalid metadata should be strictly ignored
    candidates = [
        {"video_id": "V01", "frame_id": 100, "video_nomination_rank": 1, "local_anchor_rank": 1},
        {"video_id": "V02", "frame_id": 200, "video_nomination_rank": 2, "local_anchor_rank": "invalid"},  # invalid string
        {"video_id": "V03", "frame_id": 300, "video_nomination_rank": None, "local_anchor_rank": 1},       # missing nom rank
        {"video_id": "V04", "frame_id": 400, "local_anchor_rank": 1},                                      # missing key
        {"video_id": "V05", "frame_id": 500, "video_nomination_rank": 3, "local_anchor_rank": 2},          # secondary anchor
        {"video_id": "V06", "frame_id": 600, "video_nomination_rank": 4, "local_anchor_rank": 1},          # valid primary 2
        {"video_id": "V07", "frame_id": 700, "video_nomination_rank": 5, "local_anchor_rank": 1},          # valid primary 3
    ]
    # Only 3 valid primary candidates exist (V01@1, V06@4, V07@5) -> Not enough for 4th -> Must return None
    res = select_fourth_unique_primary_candidate(candidates)
    assert res is None


def test_selector_deduplicates_multiple_primaries_for_same_nominated_video():
    # If the candidate list contains duplicate primary records for the same video nomination rank,
    # only the first record should be counted.
    candidates = [
        {"video_id": "V01", "frame_id": 100, "video_nomination_rank": 1, "local_anchor_rank": 1},
        {"video_id": "V01", "frame_id": 105, "video_nomination_rank": 1, "local_anchor_rank": 1},  # duplicate V01
        {"video_id": "V02", "frame_id": 200, "video_nomination_rank": 2, "local_anchor_rank": 1},
        {"video_id": "V03", "frame_id": 300, "video_nomination_rank": 3, "local_anchor_rank": 1},
        {"video_id": "V04", "frame_id": 400, "video_nomination_rank": 4, "local_anchor_rank": 1},  # 4th unique primary
        {"video_id": "V05", "frame_id": 500, "video_nomination_rank": 5, "local_anchor_rank": 1},
    ]
    res = select_fourth_unique_primary_candidate(candidates)
    assert res is not None
    idx, cand, prov = res
    assert idx == 3
    assert cand["video_id"] == "V04"
    assert cand["frame_id"] == 400
    assert prov["video_nomination_rank"] == 4
    assert prov["source_key"] == "V04:400:4:1"


def test_runtime_engine_selector_provenance_key_parity():
    # Proves that runtime tuple representation and engine dict representation
    # select the exact same candidate and produce the identical source_key.
    raw_tuples = [
        (
            QAEvidenceCandidate(query_id="TEST", rank=1, video_id="V01", frame_id=1000, retrieval_score=0.9, provenance={"video_nomination_rank": 1, "local_anchor_rank": 1}),
            None,
        ),
        (
            QAEvidenceCandidate(query_id="TEST", rank=2, video_id="V01", frame_id=1030, retrieval_score=0.85, provenance={"video_nomination_rank": 1, "local_anchor_rank": 2}),
            None,
        ),
        (
            QAEvidenceCandidate(query_id="TEST", rank=3, video_id="V02", frame_id=2000, retrieval_score=0.8, provenance={"video_nomination_rank": 2, "local_anchor_rank": 1}),
            None,
        ),
        (
            QAEvidenceCandidate(query_id="TEST", rank=4, video_id="V02", frame_id=2030, retrieval_score=0.75, provenance={"video_nomination_rank": 2, "local_anchor_rank": 2}),
            None,
        ),
        (
            QAEvidenceCandidate(query_id="TEST", rank=5, video_id="V03", frame_id=3000, retrieval_score=0.7, provenance={"video_nomination_rank": 3, "local_anchor_rank": 1}),
            None,
        ),
        (
            QAEvidenceCandidate(query_id="TEST", rank=6, video_id="V04", frame_id=14541, retrieval_score=0.6, provenance={"video_nomination_rank": 4, "local_anchor_rank": 1}),
            None,
        ),
    ]

    scored_dicts = [
        {"video_id": "V01", "frame_id": 1000, "video_nomination_rank": 1, "local_anchor_rank": 1, "answers": ["3", "4", "2"]},
        {"video_id": "V01", "frame_id": 1030, "video_nomination_rank": 1, "local_anchor_rank": 2, "answers": ["3", "4", "2"]},
        {"video_id": "V02", "frame_id": 2000, "video_nomination_rank": 2, "local_anchor_rank": 1, "answers": ["3", "4", "2"]},
        {"video_id": "V02", "frame_id": 2030, "video_nomination_rank": 2, "local_anchor_rank": 2, "answers": ["3", "4", "2"]},
        {"video_id": "V03", "frame_id": 3000, "video_nomination_rank": 3, "local_anchor_rank": 1, "answers": ["3", "4", "2"]},
        {"video_id": "V04", "frame_id": 14541, "video_nomination_rank": 4, "local_anchor_rank": 1, "answers": ["3", "2", "4"]},
    ]

    rt_res = select_fourth_unique_primary_candidate(raw_tuples)
    eng_res = select_fourth_unique_primary_candidate(scored_dicts)

    assert rt_res is not None
    assert eng_res is not None
    assert rt_res[2]["source_key"] == eng_res[2]["source_key"]
    assert rt_res[2]["video_id"] == "V04"
    assert rt_res[2]["frame_id"] == 14541
    assert rt_res[2]["video_nomination_rank"] == 4
    assert rt_res[2]["local_anchor_rank"] == 1
