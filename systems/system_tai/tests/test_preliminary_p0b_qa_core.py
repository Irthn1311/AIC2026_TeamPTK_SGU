import numpy as np
import pytest

from system_tai.qa import (
    AnswerHypothesis,
    BaselineQuestionCandidateProvider,
    CosineEvidenceAnswerScorer,
    QABaselineEngine,
    QAEvidenceCandidate,
    QAQuery,
    QuestionType,
    classify_question_type,
)


def test_qa_query_validation():
    # A. QAQuery validation
    query = QAQuery("q1", "Xe chạy trên đường", "Chiếc xe có màu gì?")
    assert query.query_id == "q1"
    assert query.event_description == "Xe chạy trên đường"
    assert query.question == "Chiếc xe có màu gì?"

    with pytest.raises(ValueError, match="query_id"):
        QAQuery("", "desc", "question")

    with pytest.raises(ValueError, match="event_description"):
        QAQuery("q1", "", "question")

    with pytest.raises(ValueError, match="question"):
        QAQuery("q1", "desc", "")


def test_qa_evidence_candidate_immutability():
    # Provenance dict isolation and immutability check
    original_dict = {"source": "retriever", "score": 0.9}
    cand = QAEvidenceCandidate(
        query_id="q1",
        rank=1,
        video_id="V001",
        frame_id=100,
        retrieval_score=0.95,
        provenance=original_dict,
    )
    # Mutating original dict must NOT mutate candidate.provenance
    original_dict["source"] = "mutated"
    assert cand.provenance["source"] == "retriever"

    # Attempting direct mutation on candidate.provenance must fail with TypeError
    with pytest.raises(TypeError):
        cand.provenance["new_key"] = 123  # type: ignore


def test_qa_evidence_candidate_optional_metadata_and_retrieval_score():
    # Optional metadata defaults to None
    cand_default = QAEvidenceCandidate("q1", 1, "V001", 100, 0.5)
    assert cand_default.evidence_score is None
    assert cand_default.timestamp_seconds is None

    # Finite float/int accepted
    cand_valid = QAEvidenceCandidate(
        "q1", 1, "V001", 100, 0.5, evidence_score=0.8, timestamp_seconds=12.5
    )
    assert cand_valid.evidence_score == 0.8
    assert cand_valid.timestamp_seconds == 12.5

    # Negative timestamp rejected
    with pytest.raises(ValueError, match="timestamp_seconds"):
        QAEvidenceCandidate("q1", 1, "V001", 100, 0.5, timestamp_seconds=-1.0)

    # NaN retrieval score rejected
    with pytest.raises(ValueError, match="retrieval_score"):
        QAEvidenceCandidate("q1", 1, "V001", 100, float("nan"))

    # Inf retrieval score rejected
    with pytest.raises(ValueError, match="retrieval_score"):
        QAEvidenceCandidate("q1", 1, "V001", 100, float("inf"))


def test_question_classification():
    # B. COLOR classification VI
    assert classify_question_type("Chiếc xe có màu gì?") == QuestionType.COLOR

    # C. COLOR classification EN
    assert classify_question_type("What color is the car?") == QuestionType.COLOR

    # D. COUNT classification
    assert classify_question_type("Có bao nhiêu người?") == QuestionType.COUNT

    # E. unsupported open-ended
    assert classify_question_type("Sau đó người phụ nữ làm gì?") == QuestionType.UNSUPPORTED


def test_deterministic_answer_candidate_ordering():
    # F. deterministic answer candidate ordering
    provider = BaselineQuestionCandidateProvider()
    colors = provider.get_candidates(QuestionType.COLOR)
    assert len(colors) == 11
    assert colors[0].canonical_answer == "đỏ"
    assert colors[1].canonical_answer == "xanh dương"
    assert colors[2].canonical_answer == "xanh lá"


def test_scorer_strict_validation():
    scorer = CosineEvidenceAnswerScorer()
    cand = QAEvidenceCandidate("q1", 1, "V001", 100, 0.9)
    hyp = AnswerHypothesis("đỏ", ("đỏ",), ("red",))

    norm_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    prompts_valid = {"red": norm_vec}

    # A. float64 image embedding rejected
    with pytest.raises(TypeError, match="float32"):
        f64_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        scorer.score_answers(cand, [hyp], f64_vec, prompts_valid)

    # B. NaN rejected
    with pytest.raises(ValueError, match="non-finite"):
        nan_vec = np.array([float("nan"), 0.0, 0.0, 0.0], dtype=np.float32)
        scorer.score_answers(cand, [hyp], nan_vec, prompts_valid)

    # C. Inf rejected
    with pytest.raises(ValueError, match="non-finite"):
        inf_vec = np.array([float("inf"), 0.0, 0.0, 0.0], dtype=np.float32)
        scorer.score_answers(cand, [hyp], inf_vec, prompts_valid)

    # D. zero vector rejected
    with pytest.raises(ValueError, match="zero"):
        zero_vec = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        scorer.score_answers(cand, [hyp], zero_vec, prompts_valid)

    # E. non-L2-normalized vector rejected
    with pytest.raises(ValueError, match="L2-normalized"):
        unnorm_vec = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        scorer.score_answers(cand, [hyp], unnorm_vec, prompts_valid)

    # F. 2D embedding rejected
    with pytest.raises(ValueError, match="1-dimensional"):
        vec_2d = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        scorer.score_answers(cand, [hyp], vec_2d, prompts_valid)

    # G. image/prompt dimension mismatch rejected
    prompt_5d = {"red": np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    with pytest.raises(ValueError, match="Dimension mismatch"):
        scorer.score_answers(cand, [hyp], norm_vec, prompt_5d)

    # H. valid normalized float32 embeddings accepted
    res = scorer.score_answers(cand, [hyp], norm_vec, prompts_valid)
    assert res[0][1] == 1.0


def test_deterministic_tie_breaking():
    # H & I. deterministic tie: equal cosine scores => canonical_answer ascending
    prompt_embs = {
        "prompt_b": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "prompt_a": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    }
    img_emb = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    cand = QAEvidenceCandidate("q1", 1, "V001", 500, 0.95)
    hyps = (
        AnswerHypothesis("z_answer", ("z",), ("prompt_b",)),
        AnswerHypothesis("a_answer", ("a",), ("prompt_a",)),
    )
    scorer = CosineEvidenceAnswerScorer()
    results = scorer.score_answers(cand, hyps, img_emb, prompt_embs)
    assert results[0][0].canonical_answer == "a_answer"
    assert results[1][0].canonical_answer == "z_answer"


def test_answer_hypothesis_validation():
    # Non-empty canonical answer required
    with pytest.raises(ValueError, match="canonical_answer"):
        AnswerHypothesis("")

    # Non-empty strings in aliases required
    with pytest.raises(ValueError, match="aliases"):
        AnswerHypothesis("ans", aliases=("",))

    # Non-empty strings in visual_prompts required
    with pytest.raises(ValueError, match="visual_prompts"):
        AnswerHypothesis("ans", visual_prompts=("",))


def test_yes_no_evidence_boundary():
    # Section 8 Option B: YES_NO baseline predictions tagged EXPERIMENTAL
    engine = QABaselineEngine()
    query = QAQuery("q1", "Mô tả", "Chiếc xe có phải màu đỏ không?")
    cands = [QAEvidenceCandidate("q1", 1, "V001", 100, 0.9)]
    res = engine.answer(query, cands)
    assert res.question_type == QuestionType.YES_NO
    assert res.diagnostics["confidence_level"] == "EXPERIMENTAL"


def test_engine_evidence_query_id_mismatch():
    # Section 9: Candidate query_id mismatch raises ValueError
    engine = QABaselineEngine()
    query = QAQuery("q1", "Mô tả", "Chiếc xe có màu gì?")
    mismatched_cands = [QAEvidenceCandidate("q2", 1, "V001", 100, 0.9)]
    with pytest.raises(ValueError, match="query_id mismatch"):
        engine.answer(query, mismatched_cands)


def test_engine_evidence_rank_contract():
    engine = QABaselineEngine()
    query = QAQuery("q1", "Mô tả", "Chiếc xe có màu gì?")

    # Duplicate rank rejected
    dup_cands = [
        QAEvidenceCandidate("q1", 1, "V001", 100, 0.9),
        QAEvidenceCandidate("q1", 1, "V002", 200, 0.8),
    ]
    with pytest.raises(ValueError, match="Duplicate evidence candidate rank"):
        engine.answer(query, dup_cands)

    # Out of order input sequence (rank 3, rank 1, rank 2) sorted by declared rank ascending
    out_of_order = [
        QAEvidenceCandidate("q1", 3, "V003", 300, 0.7),
        QAEvidenceCandidate("q1", 1, "V001", 100, 0.9),
        QAEvidenceCandidate("q1", 2, "V002", 200, 0.8),
    ]
    res = engine.answer(query, out_of_order)
    ranks_out = [p.rank for p in res.predictions]
    assert ranks_out == [1, 2, 3]


def test_engine_max_100_behavior():
    engine = QABaselineEngine()
    query = QAQuery("q1", "Mô tả sự kiện", "Chiếc xe màu gì?")

    # Exactly 100 candidates -> VALID
    cands_100 = [
        QAEvidenceCandidate("q1", i, f"V_{i:03d}", i * 10, 0.9) for i in range(1, 101)
    ]
    res_100 = engine.answer(query, cands_100)
    assert len(res_100.predictions) == 100

    # >100 candidates -> P0-A validation failure
    cands_101 = [
        QAEvidenceCandidate("q1", i, f"V_{i:03d}", i * 10, 0.9) for i in range(1, 102)
    ]
    with pytest.raises(ValueError, match="Cannot exceed 100 predictions"):
        engine.answer(query, cands_101)
