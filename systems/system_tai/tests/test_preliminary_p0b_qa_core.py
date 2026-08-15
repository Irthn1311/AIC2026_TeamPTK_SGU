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
    original_dict = {"source": "retriever", "score": 0.9}
    cand = QAEvidenceCandidate(
        query_id="q1",
        rank=1,
        video_id="V001",
        frame_id=100,
        retrieval_score=0.95,
        provenance=original_dict,
    )
    original_dict["source"] = "mutated"
    assert cand.provenance["source"] == "retriever"

    with pytest.raises(TypeError):
        cand.provenance["new_key"] = 123  # type: ignore


def test_qa_evidence_candidate_optional_metadata_and_retrieval_score():
    cand_default = QAEvidenceCandidate("q1", 1, "V001", 100, 0.5)
    assert cand_default.evidence_score is None
    assert cand_default.timestamp_seconds is None

    cand_valid = QAEvidenceCandidate(
        "q1", 1, "V001", 100, 0.5, evidence_score=0.8, timestamp_seconds=12.5
    )
    assert cand_valid.evidence_score == 0.8
    assert cand_valid.timestamp_seconds == 12.5

    with pytest.raises(ValueError, match="timestamp_seconds"):
        QAEvidenceCandidate("q1", 1, "V001", 100, 0.5, timestamp_seconds=-1.0)

    with pytest.raises(ValueError, match="retrieval_score"):
        QAEvidenceCandidate("q1", 1, "V001", 100, float("nan"))

    with pytest.raises(ValueError, match="retrieval_score"):
        QAEvidenceCandidate("q1", 1, "V001", 100, float("inf"))


def test_question_classification_precedence():
    # Precedence: COUNT -> YES_NO -> DIRECTION -> COLOR -> UNSUPPORTED
    assert classify_question_type("Chiếc xe có màu gì?") == QuestionType.COLOR
    assert classify_question_type("What color is the car?") == QuestionType.COLOR
    assert classify_question_type("Có bao nhiêu người?") == QuestionType.COUNT

    # Precedence hotfix tests
    assert classify_question_type("Chiếc xe có màu đỏ không?") == QuestionType.YES_NO
    assert classify_question_type("Is the car red?") == QuestionType.YES_NO
    assert classify_question_type("Có bao nhiêu người không đội mũ?") == QuestionType.COUNT
    assert classify_question_type("Người đó ở bên nào?") == QuestionType.DIRECTION

    # Unsupported open-ended
    assert classify_question_type("Sau đó người phụ nữ làm gì?") == QuestionType.UNSUPPORTED


def test_baseline_engine_respects_runtime_preclassified_question_type() -> None:
    engine = QABaselineEngine()
    query = QAQuery(
        "q-preclassified",
        "Một cảnh không thuộc bộ phân loại legacy.",
        "Câu hỏi mở không thuộc bộ phân loại legacy.",
        question_type=QuestionType.COLOR,
    )
    candidate = QAEvidenceCandidate("q-preclassified", 1, "V001", 100, 0.9)
    vector = np.asarray([1.0, 0.0], dtype=np.float32)

    result = engine.answer(
        query,
        [candidate],
        image_embeddings={("V001", 100): vector},
        prompt_embeddings={"red": vector},
    )

    assert result.question_type is QuestionType.COLOR
    assert [(item.rank, item.answer) for item in result.predictions] == [(1, "đỏ")]


def test_deterministic_answer_candidate_ordering():
    provider = BaselineQuestionCandidateProvider()
    colors = provider.get_candidates(QuestionType.COLOR)
    assert len(colors) == 11
    assert colors[0].canonical_answer == "đỏ"
    assert colors[1].canonical_answer == "xanh dương"
    assert colors[2].canonical_answer == "xanh lá"


def test_max_cosine_aggregation_correctness():
    # Section 4: prompt A cos = 0.0, prompt B cos = -1.0
    img_emb = np.array([1.0, 0.0], dtype=np.float32)
    p_a = np.array([0.0, 1.0], dtype=np.float32)
    p_b = np.array([-1.0, 0.0], dtype=np.float32)

    scorer = CosineEvidenceAnswerScorer()
    cand = QAEvidenceCandidate("q1", 1, "V001", 100, 0.9)

    # Hypothesis with both prompts A (0.0) and B (-1.0)
    hyp_order1 = AnswerHypothesis("h1", ("h1",), ("prompt_a", "prompt_b"))
    prompts = {"prompt_a": p_a, "prompt_b": p_b}

    res1 = scorer.score_answers(cand, [hyp_order1], img_emb, prompts)
    assert len(res1) == 1
    assert res1[0][1] == 0.0

    # Reversed prompt iteration order: ("prompt_b", "prompt_a")
    hyp_order2 = AnswerHypothesis("h1", ("h1",), ("prompt_b", "prompt_a"))
    res2 = scorer.score_answers(cand, [hyp_order2], img_emb, prompts)
    assert len(res2) == 1
    assert res2[0][1] == 0.0


def test_missing_visual_evidence_handling():
    scorer = CosineEvidenceAnswerScorer()
    cand = QAEvidenceCandidate("q1", 1, "V001", 100, 0.9)
    hyp = AnswerHypothesis("đỏ", ("đỏ",), ("red",))

    norm_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    prompts_valid = {"red": norm_vec}

    # A. image_embedding is None => []
    assert scorer.score_answers(cand, [hyp], None, prompts_valid) == []

    # B. prompt_embeddings is None => []
    assert scorer.score_answers(cand, [hyp], norm_vec, None) == []

    # C. prompt_embeddings is empty => []
    assert scorer.score_answers(cand, [hyp], norm_vec, {}) == []

    # D. hypothesis prompts do not exist in prompt_embeddings => hypothesis skipped
    prompts_unrelated = {"blue": norm_vec}
    assert scorer.score_answers(cand, [hyp], norm_vec, prompts_unrelated) == []

    # E. engine receives candidates but image_embeddings is None => 0 predictions
    engine = QABaselineEngine()
    query = QAQuery("q1", "Mô tả", "Chiếc xe có màu gì?")
    cands = [QAEvidenceCandidate("q1", 1, "V001", 100, 0.9)]
    res_e = engine.answer(query, cands, image_embeddings=None, prompt_embeddings=prompts_valid)
    assert res_e.predictions == []

    # F. candidate key (video_id, frame_id) absent from image_embeddings => 0 predictions
    img_embs_mismatched = {("V999", 999): norm_vec}
    res_f = engine.answer(
        query, cands, image_embeddings=img_embs_mismatched, prompt_embeddings=prompts_valid
    )
    assert res_f.predictions == []

    # G. prompt bank has no usable prompt for any hypothesis => 0 predictions
    res_g = engine.answer(
        query, cands, image_embeddings={("V001", 100): norm_vec}, prompt_embeddings={}
    )
    assert res_g.predictions == []


def test_scorer_strict_validation():
    scorer = CosineEvidenceAnswerScorer()
    cand = QAEvidenceCandidate("q1", 1, "V001", 100, 0.9)
    hyp = AnswerHypothesis("đỏ", ("đỏ",), ("red",))

    norm_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    prompts_valid = {"red": norm_vec}

    # float64 image embedding rejected
    with pytest.raises(TypeError, match="float32"):
        f64_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        scorer.score_answers(cand, [hyp], f64_vec, prompts_valid)

    # NaN rejected
    with pytest.raises(ValueError, match="non-finite"):
        nan_vec = np.array([float("nan"), 0.0, 0.0, 0.0], dtype=np.float32)
        scorer.score_answers(cand, [hyp], nan_vec, prompts_valid)

    # Inf rejected
    with pytest.raises(ValueError, match="non-finite"):
        inf_vec = np.array([float("inf"), 0.0, 0.0, 0.0], dtype=np.float32)
        scorer.score_answers(cand, [hyp], inf_vec, prompts_valid)

    # zero vector rejected
    with pytest.raises(ValueError, match="zero"):
        zero_vec = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        scorer.score_answers(cand, [hyp], zero_vec, prompts_valid)

    # non-L2-normalized vector rejected
    with pytest.raises(ValueError, match="L2-normalized"):
        unnorm_vec = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        scorer.score_answers(cand, [hyp], unnorm_vec, prompts_valid)

    # 2D embedding rejected
    with pytest.raises(ValueError, match="1-dimensional"):
        vec_2d = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        scorer.score_answers(cand, [hyp], vec_2d, prompts_valid)

    # image/prompt dimension mismatch rejected
    prompt_5d = {"red": np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    with pytest.raises(ValueError, match="Dimension mismatch"):
        scorer.score_answers(cand, [hyp], norm_vec, prompt_5d)


def test_real_answer_selection_engine():
    # Section 6: Proven visual score selection engine test
    engine = QABaselineEngine()
    query = QAQuery("q1", "Mô tả", "Chiếc xe có màu gì?")
    cand = QAEvidenceCandidate("q1", 1, "V001", 500, 0.95)

    # Image embedding aligned with "red"
    img_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Prompts for red, blue, green
    prompts = {
        "red": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "blue": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "green": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    img_embs = {("V001", 500): img_emb}

    res = engine.answer(query, [cand], image_embeddings=img_embs, prompt_embeddings=prompts)
    assert len(res.predictions) == 1
    pred = res.predictions[0]
    assert pred.answer == "đỏ"
    assert pred.video_id == "V001"
    assert pred.frame_id == 500
    assert pred.rank == 1


def test_answer_hypothesis_validation():
    with pytest.raises(ValueError, match="canonical_answer"):
        AnswerHypothesis("")

    with pytest.raises(ValueError, match="aliases"):
        AnswerHypothesis("ans", aliases=("",))

    with pytest.raises(ValueError, match="visual_prompts"):
        AnswerHypothesis("ans", visual_prompts=("",))


def test_yes_no_evidence_boundary():
    engine = QABaselineEngine()
    query = QAQuery("q1", "Mô tả", "Chiếc xe có phải màu đỏ không?")
    cand = QAEvidenceCandidate("q1", 1, "V001", 100, 0.9)

    # Without visual embeddings => zero predictions, EXPERIMENTAL flag
    res_no_embs = engine.answer(query, [cand])
    assert res_no_embs.question_type == QuestionType.YES_NO
    assert res_no_embs.predictions == []
    assert res_no_embs.diagnostics["confidence_level"] == "EXPERIMENTAL"

    # With valid synthetic visual embeddings => returns prediction tagged EXPERIMENTAL
    img_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    prompts = {"yes": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    res_with_embs = engine.answer(
        query,
        [cand],
        image_embeddings={("V001", 100): img_emb},
        prompt_embeddings=prompts,
    )
    assert res_with_embs.question_type == QuestionType.YES_NO
    assert len(res_with_embs.predictions) == 1
    assert res_with_embs.diagnostics["confidence_level"] == "EXPERIMENTAL"


def test_engine_evidence_query_id_mismatch():
    engine = QABaselineEngine()
    query = QAQuery("q1", "Mô tả", "Chiếc xe có màu gì?")
    mismatched_cands = [QAEvidenceCandidate("q2", 1, "V001", 100, 0.9)]
    with pytest.raises(ValueError, match="query_id mismatch"):
        engine.answer(query, mismatched_cands)


def test_engine_evidence_rank_contract():
    engine = QABaselineEngine()
    query = QAQuery("q1", "Mô tả", "Chiếc xe có màu gì?")

    dup_cands = [
        QAEvidenceCandidate("q1", 1, "V001", 100, 0.9),
        QAEvidenceCandidate("q1", 1, "V002", 200, 0.8),
    ]
    with pytest.raises(ValueError, match="Duplicate evidence candidate rank"):
        engine.answer(query, dup_cands)

    out_of_order = [
        QAEvidenceCandidate("q1", 3, "V003", 300, 0.7),
        QAEvidenceCandidate("q1", 1, "V001", 100, 0.9),
        QAEvidenceCandidate("q1", 2, "V002", 200, 0.8),
    ]
    img_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    img_embs = {
        ("V001", 100): img_vec,
        ("V002", 200): img_vec,
        ("V003", 300): img_vec,
    }
    prompts = {"red": img_vec}

    res = engine.answer(
        query, out_of_order, image_embeddings=img_embs, prompt_embeddings=prompts
    )
    ranks_out = [p.rank for p in res.predictions]
    assert ranks_out == [1, 2, 3]


def test_engine_max_100_behavior():
    engine = QABaselineEngine()
    query = QAQuery("q1", "Mô tả sự kiện", "Chiếc xe màu gì?")
    img_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    prompts = {"red": img_vec}

    # Exactly 100 candidates -> VALID
    cands_100 = [
        QAEvidenceCandidate("q1", i, f"V_{i:03d}", i * 10, 0.9) for i in range(1, 101)
    ]
    img_embs_100 = {(f"V_{i:03d}", i * 10): img_vec for i in range(1, 101)}
    res_100 = engine.answer(
        query, cands_100, image_embeddings=img_embs_100, prompt_embeddings=prompts
    )
    assert len(res_100.predictions) == 100

    # >100 candidates -> P0-A validation failure
    cands_101 = [
        QAEvidenceCandidate("q1", i, f"V_{i:03d}", i * 10, 0.9) for i in range(1, 102)
    ]
    img_embs_101 = {(f"V_{i:03d}", i * 10): img_vec for i in range(1, 102)}
    with pytest.raises(ValueError, match="Cannot exceed 100 predictions"):
        engine.answer(
            query, cands_101, image_embeddings=img_embs_101, prompt_embeddings=prompts
        )


def test_capability_driven_unsupported_recovery_flag_off():
    # Flag OFF: UNSUPPORTED query must unconditionally return empty predictions
    engine = QABaselineEngine(allow_unsupported_provider_fallback=False)
    query = QAQuery("q_unsupp", "Mô tả sự kiện", "Câu hỏi lạ không có pattern?")
    cands = [QAEvidenceCandidate("q_unsupp", 1, "V001", 100, 0.9)]
    res = engine.answer(query, cands)
    assert len(res.predictions) == 0
    assert res.question_type == QuestionType.UNSUPPORTED
    assert res.diagnostics.get("confidence_level") == "UNSUPPORTED"


def test_capability_driven_unsupported_recovery_flag_on_success():
    # Flag ON: UNSUPPORTED query with provider hypotheses and scored evidence must produce valid predictions
    class CustomFallbackProvider:
        def get_candidates_for_query(self, qtype, text):
            return (
                AnswerHypothesis("xe hơi", ("car", "xe"), visual_prompts=("car", "xe")),
                AnswerHypothesis("xe buýt", ("bus", "xe bus"), visual_prompts=("bus", "xe bus")),
            )
        def get_candidates(self, qtype):
            return self.get_candidates_for_query(qtype, "")

    engine = QABaselineEngine(
        candidate_provider=CustomFallbackProvider(),
        allow_unsupported_provider_fallback=True,
    )
    query = QAQuery("q_unsupp", "Mô tả sự kiện", "Câu hỏi lạ không có pattern?", question_type=QuestionType.UNSUPPORTED)
    img_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cands = [
        QAEvidenceCandidate("q_unsupp", 1, "V001", 100, 0.9),
        QAEvidenceCandidate("q_unsupp", 2, "V002", 200, 0.8),
    ]
    img_embs = {
        ("V001", 100): img_vec,
        ("V002", 200): img_vec,
    }
    prompts = {"car": img_vec, "bus": img_vec}

    res = engine.answer(query, cands, image_embeddings=img_embs, prompt_embeddings=prompts)
    assert len(res.predictions) > 0
    assert res.question_type == QuestionType.UNSUPPORTED
    assert res.diagnostics.get("confidence_level") == "FALLBACK"
    assert res.predictions[0].rank == 1
    assert res.predictions[0].video_id == "V001"
    assert res.predictions[0].answer in ("xe hơi", "xe buýt")


def test_capability_driven_unsupported_recovery_flag_on_no_hypotheses():
    # Flag ON: UNSUPPORTED query with NO provider hypotheses must return empty predictions
    class EmptyProvider:
        def get_candidates_for_query(self, qtype, text):
            return ()
        def get_candidates(self, qtype):
            return ()

    engine = QABaselineEngine(
        candidate_provider=EmptyProvider(),
        allow_unsupported_provider_fallback=True,
    )
    query = QAQuery("q_unsupp", "Mô tả sự kiện", "Câu hỏi lạ không có pattern?", question_type=QuestionType.UNSUPPORTED)
    cands = [QAEvidenceCandidate("q_unsupp", 1, "V001", 100, 0.9)]
    res = engine.answer(query, cands)
    assert len(res.predictions) == 0
    assert res.question_type == QuestionType.UNSUPPORTED
    assert res.diagnostics.get("confidence_level") == "UNSUPPORTED"


def test_capability_driven_unsupported_recovery_flag_on_no_evidence():
    # Flag ON: UNSUPPORTED query with NO evidence candidates must return empty predictions
    class CustomFallbackProvider:
        def get_candidates_for_query(self, qtype, text):
            return (AnswerHypothesis("xe hơi", ("car",)),)
        def get_candidates(self, qtype):
            return self.get_candidates_for_query(qtype, "")

    engine = QABaselineEngine(
        candidate_provider=CustomFallbackProvider(),
        allow_unsupported_provider_fallback=True,
    )
    query = QAQuery("q_unsupp", "Mô tả sự kiện", "Câu hỏi lạ không có pattern?", question_type=QuestionType.UNSUPPORTED)
    res = engine.answer(query, [])
    assert len(res.predictions) == 0
    assert res.question_type == QuestionType.UNSUPPORTED
    assert res.diagnostics.get("confidence_level") == "UNSUPPORTED"


def test_capability_driven_unsupported_recovery_provider_error_fail_closed():
    # Flag ON: If provider raises an exception on UNSUPPORTED, must fail-closed and return empty predictions
    class BrokenProvider:
        def get_candidates_for_query(self, qtype, text):
            raise RuntimeError("Provider service exploded")
        def get_candidates(self, qtype):
            raise RuntimeError("Provider service exploded")

    engine = QABaselineEngine(
        candidate_provider=BrokenProvider(),
        allow_unsupported_provider_fallback=True,
    )
    query = QAQuery("q_unsupp", "Mô tả sự kiện", "Câu hỏi lạ không có pattern?", question_type=QuestionType.UNSUPPORTED)
    cands = [QAEvidenceCandidate("q_unsupp", 1, "V001", 100, 0.9)]
    res = engine.answer(query, cands)
    assert len(res.predictions) == 0
    assert res.question_type == QuestionType.UNSUPPORTED
    assert res.diagnostics.get("confidence_level") == "UNSUPPORTED"


def test_supported_question_type_provider_exception_propagates():
    # Supported types (e.g. COLOR) must NOT swallow provider exceptions (preserving legacy behavior)
    class BrokenProvider:
        def get_candidates_for_query(self, qtype, text):
            raise RuntimeError("Provider broken for supported type")
        def get_candidates(self, qtype):
            raise RuntimeError("Provider broken for supported type")

    engine = QABaselineEngine(
        candidate_provider=BrokenProvider(),
        allow_unsupported_provider_fallback=True,
    )
    query = QAQuery("q_color", "Mô tả sự kiện", "Chiếc xe màu gì?", question_type=QuestionType.COLOR)
    cands = [QAEvidenceCandidate("q_color", 1, "V001", 100, 0.9)]
    with pytest.raises(RuntimeError, match="Provider broken for supported type"):
        engine.answer(query, cands)


