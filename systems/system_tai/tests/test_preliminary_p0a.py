import pytest

from system_tai.preliminary import (
    KISGroundTruth,
    KISPrediction,
    NormalizedAliasAnswerMatcher,
    QAGroundTruth,
    QAPrediction,
    TRAKEGroundTruth,
    TRAKEPrediction,
    evaluate_dataset,
    evaluate_ranked_query,
    score_kis_prediction,
    score_qa_prediction,
    score_trake_prediction,
    validate_ranked_top100,
)


def test_kis_scoring():
    gt = KISGroundTruth("q1", "L01_V001", 500, 510)

    assert score_kis_prediction(KISPrediction("q1", 1, "L01_V001", 505), gt) == 1.0
    assert score_kis_prediction(KISPrediction("q1", 2, "L01_V001", 500), gt) == 1.0
    assert score_kis_prediction(KISPrediction("q1", 3, "L01_V001", 510), gt) == 1.0

    assert score_kis_prediction(KISPrediction("q1", 4, "L01_V001", 499), gt) == 0.0
    assert score_kis_prediction(KISPrediction("q1", 5, "L01_V001", 511), gt) == 0.0
    assert score_kis_prediction(KISPrediction("q1", 6, "wrong", 505), gt) == 0.0


def test_qa_scoring():
    gt = QAGroundTruth("q1", "L05_V005", 800, 900, ("màu xanh", "xanh"))
    matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)

    assert (
        score_qa_prediction(QAPrediction("q1", 1, "L05_V005", 850, "màu xanh"), gt, matcher) == 1.0
    )
    assert score_qa_prediction(QAPrediction("q1", 2, "L05_V005", 850, " xanh "), gt, matcher) == 1.0

    assert score_qa_prediction(QAPrediction("q1", 3, "L05_V005", 850, "đỏ"), gt, matcher) == 0.0
    assert score_qa_prediction(QAPrediction("q1", 4, "wrong", 850, "xanh"), gt, matcher) == 0.0
    assert score_qa_prediction(QAPrediction("q1", 5, "L05_V005", 901, "xanh"), gt, matcher) == 0.0


def test_trake_scoring_and_mismatch():
    gt = TRAKEGroundTruth("q1", "L10_V010", ((100, 110), (150, 160), (200, 210), (250, 260)))

    # 1. Valid prediction: exact match = 1.0
    assert (
        score_trake_prediction(TRAKEPrediction("q1", 1, "L10_V010", (105, 155, 205, 255)), gt)
        == 1.0
    )
    # 2. Valid prediction: wrong video with CORRECT event count => 0.0
    assert (
        score_trake_prediction(TRAKEPrediction("q1", 2, "wrong_video", (105, 155, 205, 255)), gt)
        == 0.0
    )
    # 3. Valid prediction: correct video with all event frames outside intervals => 0.0
    assert (
        score_trake_prediction(TRAKEPrediction("q1", 3, "L10_V010", (999, 999, 999, 999)), gt)
        == 0.0
    )
    # 4. Wrong event count => ValueError
    with pytest.raises(ValueError, match="TRAKE event count mismatch"):
        score_trake_prediction(TRAKEPrediction("q1", 4, "L10_V010", (105, 155)), gt)


def test_trake_frame_order_preservation():
    pred = TRAKEPrediction("q1", 1, "L10_V010", (205, 105))
    assert pred.frame_ids == (205, 105)


def test_r_at_k_final_score():
    gt = KISGroundTruth("q1", "L01_V001", 500, 510)
    preds = [
        KISPrediction("q1", 1, "L01_V001", 999),
        KISPrediction("q1", 2, "L01_V001", 998),
        KISPrediction("q1", 3, "L01_V001", 505),
    ]
    report = evaluate_ranked_query("q1", "kis", preds, gt, score_kis_prediction)

    assert report.r_at_1 == 0.0
    assert report.r_at_5 == 1.0
    assert report.r_at_20 == 1.0
    assert report.r_at_50 == 1.0
    assert report.r_at_100 == 1.0
    assert report.final_score == 0.8


def test_trake_fractional_example():
    gt = TRAKEGroundTruth("q1", "L10_V010", ((100, 110), (150, 160), (200, 210), (250, 260)))
    preds = [
        TRAKEPrediction("q1", 1, "L10_V010", (105, 999, 999, 999)),  # score = 0.25
        TRAKEPrediction("q1", 2, "L10_V010", (999, 999, 999, 999)),  # score = 0.0
        TRAKEPrediction("q1", 3, "L10_V010", (105, 155, 205, 999)),  # score = 0.75
    ]
    report = evaluate_ranked_query("q1", "trake", preds, gt, score_trake_prediction)

    assert report.r_at_1 == 0.25
    assert report.r_at_5 == 0.75
    assert report.r_at_20 == 0.75
    assert report.r_at_50 == 0.75
    assert report.r_at_100 == 0.75
    assert report.final_score == 0.65


def test_evaluator_validates_before_scoring():
    gt = TRAKEGroundTruth("q1", "L10_V010", ((100, 110), (150, 160)))
    # Invalid prediction (event count mismatch)
    preds = [TRAKEPrediction("q1", 1, "L10_V010", (105,))]
    with pytest.raises(ValueError, match="Validation failed"):
        evaluate_ranked_query("q1", "trake", preds, gt, score_trake_prediction)


def test_top100_boundary():
    # Exactly 100 predictions = VALID
    preds = [KISPrediction("q1", i, "vid", i) for i in range(1, 101)]
    assert not validate_ranked_top100(preds, "kis")

    # 101 predictions = INVALID
    preds.append(KISPrediction("q1", 101, "vid", 101))
    errors = validate_ranked_top100(preds, "kis")
    assert any("Cannot exceed 100" in e.message for e in errors)


def test_qa_normalization_duplicate_key():
    preds = [
        QAPrediction("q1", 1, "vid", 10, "  A   RED Bus!!! "),
        QAPrediction("q1", 2, "vid", 10, "a red bus"),
    ]
    errors = validate_ranked_top100(preds, "qa")
    assert any("Duplicate QA key" in e.message for e in errors)


def test_strict_integer_contract():
    # float rank rejected
    with pytest.raises(TypeError, match="must be an integer"):
        KISPrediction("q1", 1.5, "vid", 10)  # type: ignore

    # bool rank rejected
    with pytest.raises(TypeError, match="must be an integer"):
        KISPrediction("q1", True, "vid", 10)  # type: ignore

    # float frame_id rejected
    with pytest.raises(TypeError, match="must be an integer"):
        KISPrediction("q1", 1, "vid", 500.5)  # type: ignore

    # bool frame_id rejected
    with pytest.raises(TypeError, match="must be an integer"):
        KISPrediction("q1", 1, "vid", True)  # type: ignore

    # GT interval float rejected
    with pytest.raises(TypeError, match="must be an integer"):
        KISGroundTruth("q1", "vid", 100.5, 200)  # type: ignore

    # TRAKE frame float rejected
    with pytest.raises(TypeError, match="must be an integer"):
        TRAKEPrediction("q1", 1, "vid", (100, 200.5))  # type: ignore

    # TRAKE frame bool rejected
    with pytest.raises(TypeError, match="must be an integer"):
        TRAKEPrediction("q1", 1, "vid", (100, True))  # type: ignore


def test_query_identity_contract():
    gt = KISGroundTruth("q1", "vid", 100, 200)
    preds_q1 = [KISPrediction("q1", 1, "vid", 150)]
    preds_q2 = [KISPrediction("q2", 1, "vid", 150)]

    # GT q1 + evaluate query_id q1 + predictions q1 => PASS
    report = evaluate_ranked_query("q1", "kis", preds_q1, gt, score_kis_prediction)
    assert report.final_score == 1.0

    # GT q1 + evaluate query_id q1 + predictions q2 => FAIL
    with pytest.raises(ValueError, match="Validation failed"):
        evaluate_ranked_query("q1", "kis", preds_q2, gt, score_kis_prediction)

    # GT q2 + evaluate query_id q1 => FAIL
    gt_q2 = KISGroundTruth("q2", "vid", 100, 200)
    with pytest.raises(ValueError, match="Validation failed"):
        evaluate_ranked_query("q1", "kis", preds_q1, gt_q2, score_kis_prediction)


def test_zero_prediction_gt_query_dataset_evaluation():
    gt1 = KISGroundTruth("q1", "vid", 100, 200)
    gt2 = KISGroundTruth("q2", "vid", 100, 200)

    preds_q1 = [KISPrediction("q1", 1, "vid", 150)]

    report1 = evaluate_ranked_query("q1", "kis", preds_q1, gt1, score_kis_prediction)
    report2 = evaluate_ranked_query("q2", "kis", [], gt2, score_kis_prediction)

    assert report1.final_score == 1.0
    assert report2.prediction_count == 0
    assert report2.r_at_1 == 0.0
    assert report2.r_at_5 == 0.0
    assert report2.r_at_20 == 0.0
    assert report2.r_at_50 == 0.0
    assert report2.r_at_100 == 0.0
    assert report2.final_score == 0.0

    dataset_report = evaluate_dataset([report1, report2])
    assert dataset_report.query_count == 2
    assert dataset_report.mean_query_final_score == 0.5


def test_safe_qa_normalization_cases():
    matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)

    # "  A   RED Bus!!! " == "a red bus"
    assert matcher.normalize("  A   RED Bus!!! ") == "a red bus"
    assert matcher.match("  A   RED Bus!!! ", ("a red bus",))

    # "C++" != "C"
    assert matcher.normalize("C++") == "c++"
    assert not matcher.match("C++", ("C",))

    # "100%" != "100"
    assert matcher.normalize("100%") == "100%"
    assert not matcher.match("100%", ("100",))

    # "$5" != "5"
    assert matcher.normalize("$5") == "$5"
    assert not matcher.match("$5", ("5",))
