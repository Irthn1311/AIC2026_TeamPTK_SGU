from .matching import AnswerMatcher
from .schemas import (
    KISGroundTruth,
    KISPrediction,
    QAGroundTruth,
    QAPrediction,
    TRAKEGroundTruth,
    TRAKEPrediction,
)


def score_kis_prediction(prediction: KISPrediction, ground_truth: KISGroundTruth) -> float:
    if prediction.video_id != ground_truth.video_id:
        return 0.0
    if ground_truth.start_frame_id <= prediction.frame_id <= ground_truth.end_frame_id:
        return 1.0
    return 0.0


def score_qa_prediction(
    prediction: QAPrediction,
    ground_truth: QAGroundTruth,
    answer_matcher: AnswerMatcher,
) -> float:
    if prediction.video_id != ground_truth.video_id:
        return 0.0
    if not (ground_truth.start_frame_id <= prediction.frame_id <= ground_truth.end_frame_id):
        return 0.0
    if answer_matcher.match(prediction.answer, ground_truth.accepted_answers):
        return 1.0
    return 0.0


def score_trake_prediction(prediction: TRAKEPrediction, ground_truth: TRAKEGroundTruth) -> float:
    N = len(ground_truth.event_intervals)
    if len(prediction.frame_ids) != N:
        raise ValueError(
            f"TRAKE event count mismatch: expected {N} frames, got {len(prediction.frame_ids)}"
        )
    if prediction.video_id != ground_truth.video_id:
        return 0.0

    hits = 0
    for pred_frame, (start, end) in zip(prediction.frame_ids, ground_truth.event_intervals):
        if start <= pred_frame <= end:
            hits += 1
    return float(hits) / N
