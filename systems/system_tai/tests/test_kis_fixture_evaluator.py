"""Comprehensive unit tests for KISFixtureEvaluator and scoring protocol.

Tests cover:
- Official scoring protocol: Mean of Top-k R-Scores across {1, 5, 20, 50, 100}
- Protocol lock: official protocol strictly enforces cutoffs=(1,5,20,50,100) and max 100
- Custom protocols: arbitrary cutoffs and max predictions supported under custom ID
- Binary Textual-KIS rank bands: 1.0, 0.8, 0.6, 0.4, 0.2, 0.0
- Hit condition: strictly original video frame index range (zero OR with timestamp)
- Boundary frame conditions (inclusive bounds, frame 0 support)
- Multi-interval ground truth (intra-video and inter-video)
- N < k handling (empty set, short prediction lists N=3, N=15)
- Query handling: missing query evaluates to 0.0; unmapped query raises ValueError
- Strict parser: type(frame_id) is int and type(rank) is int; cấm bool, float, string
- Strict parser: query_id/video_id must be non-empty str; cấm silent str() coercion
- Strict parser: target_frame=0 preserved and valid
- Strict parser: pts_time runtime field loaded properly
- Strict parser: malformed JSON root/records raises ValueError
- Strict parser: mapping key vs item query_id mismatch raises ValueError
- Ground truth strictness: empty GT list fails closed even with missing predictions
- Ground truth strictness: temporal bounds must appear in pairs or both be None
- Prediction validation: non-monotonic scores with valid continuous rank allowed
- Prediction validation: non-finite scores (NaN, Inf) raise ValueError
- Prediction validation: discontinuous ranks raise ValueError
- Prediction validation: duplicate (video_id, frame_id) within a query raises ValueError
- Prediction validation: duplicate across queries allowed
- Prediction validation: N > 100 raises ValueError
- Timestamp guard: timestamp_tolerance_s must be finite and >= 0
- FrameTimestampResolver: cross-check verification and fail-closed on disagreement
- FrameTimestampResolver: non-finite resolver return value raises ValueError
- Cross-check status: verified, unavailable, and unresolved-partial
- Ground truth frame/timestamp consistency check via resolver
- Temporal segment IoU: None when missing temporal data, 0.0 when disjoint, exact IoU
- File-based evaluation (JSON, JSONL, missing file handling)
- EvaluationReport aggregate metrics and Mapping compatibility
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from system_tai.evaluation.kis_fixture import (
    DEFAULT_EVALUATION_CUTOFFS,
    DEFAULT_SCORING_PROTOCOL_ID,
    GroundTruthInterval,
    KISFixtureEvaluator,
    MappingFrameTimestampResolver,
    PredictionRecord,
)


def _make_prediction(
    query_id: str = "Q1",
    video_id: str = "L21_V001",
    frame_id: int = 100,
    rank: int = 1,
    score: float | None = 0.95,
    timestamp_s: float | None = None,
    pred_segment_start_s: float | None = None,
    pred_segment_end_s: float | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        query_id=query_id,
        video_id=video_id,
        frame_id=frame_id,
        rank=rank,
        score=score,
        timestamp_s=timestamp_s,
        pred_segment_start_s=pred_segment_start_s,
        pred_segment_end_s=pred_segment_end_s,
    )


def _make_predictions_list(
    query_id: str,
    hit_rank: int | None,
    total_preds: int = 100,
    target_video: str = "L21_V001",
    target_frame: int = 150,
    distractor_video: str = "L21_V999",
    distractor_frame: int = 9999,
) -> list[PredictionRecord]:
    """Generate 1-indexed continuous predictions with a single hit at hit_rank."""
    preds = []
    for r in range(1, total_preds + 1):
        if hit_rank is not None and r == hit_rank:
            preds.append(
                _make_prediction(
                    query_id=query_id,
                    video_id=target_video,
                    frame_id=target_frame,
                    rank=r,
                    score=1.0 - (r * 0.005),
                )
            )
        else:
            preds.append(
                _make_prediction(
                    query_id=query_id,
                    video_id=distractor_video,
                    frame_id=distractor_frame + r,
                    rank=r,
                    score=1.0 - (r * 0.005),
                )
            )
    return preds


# ==============================================================================
# 1. Official Scoring Protocol & Rank Band Tests
# ==============================================================================

class TestScoringProtocolRankBands:
    @pytest.fixture
    def evaluator(self) -> KISFixtureEvaluator:
        return KISFixtureEvaluator()

    @pytest.fixture
    def gt_q1(self) -> list[GroundTruthInterval]:
        return [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]

    @pytest.mark.parametrize(
        "hit_rank, expected_score, expected_r_at_k",
        [
            (1, 1.0, {1: 1.0, 5: 1.0, 20: 1.0, 50: 1.0, 100: 1.0}),
            (2, 0.8, {1: 0.0, 5: 1.0, 20: 1.0, 50: 1.0, 100: 1.0}),
            (5, 0.8, {1: 0.0, 5: 1.0, 20: 1.0, 50: 1.0, 100: 1.0}),
            (6, 0.6, {1: 0.0, 5: 0.0, 20: 1.0, 50: 1.0, 100: 1.0}),
            (20, 0.6, {1: 0.0, 5: 0.0, 20: 1.0, 50: 1.0, 100: 1.0}),
            (21, 0.4, {1: 0.0, 5: 0.0, 20: 0.0, 50: 1.0, 100: 1.0}),
            (50, 0.4, {1: 0.0, 5: 0.0, 20: 0.0, 50: 1.0, 100: 1.0}),
            (51, 0.2, {1: 0.0, 5: 0.0, 20: 0.0, 50: 0.0, 100: 1.0}),
            (100, 0.2, {1: 0.0, 5: 0.0, 20: 0.0, 50: 0.0, 100: 1.0}),
            (None, 0.0, {1: 0.0, 5: 0.0, 20: 0.0, 50: 0.0, 100: 0.0}),
        ],
    )
    def test_official_rscore_bands(
        self,
        evaluator: KISFixtureEvaluator,
        gt_q1: list[GroundTruthInterval],
        hit_rank: int | None,
        expected_score: float,
        expected_r_at_k: dict[int, float],
    ) -> None:
        preds = _make_predictions_list("Q1", hit_rank=hit_rank, total_preds=100)
        result = evaluator.evaluate_query("Q1", preds, gt_q1)

        assert math.isclose(result.official_final_score, expected_score, rel_tol=1e-6)
        assert result.r_at_k == expected_r_at_k
        assert result.scoring_protocol_id == DEFAULT_SCORING_PROTOCOL_ID
        assert result.official_first_hit_rank == hit_rank

        if hit_rank == 1:
            assert result.point_in_gt_interval is True
            assert result.localized_mrr == 1.0
        elif hit_rank is not None:
            assert result.point_in_gt_interval is False
            assert math.isclose(result.localized_mrr, 1.0 / hit_rank)
        else:
            assert result.point_in_gt_interval is False
            assert result.localized_mrr == 0.0


# ==============================================================================
# 2. Protocol Lock Tests
# ==============================================================================

class TestProtocolLock:
    def test_official_protocol_rejects_non_standard_cutoffs(self) -> None:
        with pytest.raises(ValueError, match="strictly requires cutoffs"):
            KISFixtureEvaluator(
                scoring_protocol_id=DEFAULT_SCORING_PROTOCOL_ID,
                cutoffs=(100,),
            )

    def test_official_protocol_rejects_non_standard_max_predictions(self) -> None:
        with pytest.raises(ValueError, match="strictly requires max_predictions_per_query"):
            KISFixtureEvaluator(
                scoring_protocol_id=DEFAULT_SCORING_PROTOCOL_ID,
                max_predictions_per_query=50,
            )

    def test_custom_protocol_allows_custom_cutoffs_and_limits(self) -> None:
        evaluator = KISFixtureEvaluator(
            scoring_protocol_id="custom-diagnostic-top100",
            cutoffs=(100,),
            max_predictions_per_query=200,
        )
        assert evaluator.scoring_protocol_id == "custom-diagnostic-top100"
        assert evaluator.cutoffs == (100,)
        assert evaluator.max_predictions_per_query == 200

    def test_invalid_cutoffs_rejected(self) -> None:
        # Duplicate cutoffs
        with pytest.raises(ValueError, match="duplicates"):
            KISFixtureEvaluator(scoring_protocol_id="custom", cutoffs=(1, 5, 5))

        # Non-positive cutoff
        with pytest.raises(TypeError, match="positive integer"):
            KISFixtureEvaluator(scoring_protocol_id="custom", cutoffs=(0, 5))

        # Float cutoff
        with pytest.raises(TypeError, match="positive integer"):
            KISFixtureEvaluator(scoring_protocol_id="custom", cutoffs=(1.0, 5))  # type: ignore[arg-type]


# ==============================================================================
# 3. Strict Hit Condition & Boundary Tests (Zero OR with Timestamp)
# ==============================================================================

class TestHitConditionStrictness:
    def test_frame_boundary_inclusivity(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]

        # Exactly at start_frame (100) -> HIT
        res_start = evaluator.evaluate_query(
            "Q1",
            [_make_prediction(frame_id=100, rank=1)],
            gt,
        )
        assert res_start.official_final_score == 1.0
        assert res_start.official_first_hit_rank == 1

        # Exactly at end_frame (200) -> HIT
        res_end = evaluator.evaluate_query(
            "Q1",
            [_make_prediction(frame_id=200, rank=1)],
            gt,
        )
        assert res_end.official_final_score == 1.0

        # One frame before start (99) -> MISS
        res_before = evaluator.evaluate_query(
            "Q1",
            [_make_prediction(frame_id=99, rank=1)],
            gt,
        )
        assert res_before.official_final_score == 0.0

        # One frame after end (201) -> MISS
        res_after = evaluator.evaluate_query(
            "Q1",
            [_make_prediction(frame_id=201, rank=1)],
            gt,
        )
        assert res_after.official_final_score == 0.0

    def test_frame_zero_boundary_preserved(self) -> None:
        """Frame 0 must be recognized as valid and not lost via '0 or default'."""
        evaluator = KISFixtureEvaluator()
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=0, end_frame=10)]
        pred = _make_prediction(frame_id=0, rank=1)
        res = evaluator.evaluate_query("Q1", [pred], gt)
        assert res.official_final_score == 1.0
        assert res.official_first_hit_rank == 1

    def test_zero_or_ambiguity_timestamp_matching_does_not_mask_frame_miss(self) -> None:
        """Timestamp matching GT interval MUST NOT produce a hit if frame_id is outside range."""
        evaluator = KISFixtureEvaluator()
        gt = [
            GroundTruthInterval(
                video_id="L21_V001",
                start_frame=100,
                end_frame=200,
                start_s=10.0,
                end_s=20.0,
            )
        ]

        # Prediction has timestamp 15.0s (in [10.0, 20.0]), but frame_id 50 (outside [100, 200])
        pred = _make_prediction(frame_id=50, timestamp_s=15.0, rank=1)
        res = evaluator.evaluate_query("Q1", [pred], gt)

        assert res.official_final_score == 0.0
        assert res.official_first_hit_rank is None
        assert res.localized_hit_at_k[1] is False
        assert res.video_first_hit_rank == 1
        assert res.video_hit_at_k[1] is True

    def test_video_mismatch_fails_even_with_matching_frame(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]
        pred = _make_prediction(video_id="L21_V999", frame_id=150, rank=1)
        res = evaluator.evaluate_query("Q1", [pred], gt)

        assert res.official_final_score == 0.0
        assert res.official_first_hit_rank is None
        assert res.video_first_hit_rank is None
        assert res.video_mrr == 0.0


# ==============================================================================
# 4. Multi-interval Ground Truth Tests
# ==============================================================================

class TestMultiIntervalGroundTruth:
    def test_multiple_intervals_same_video(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [
            GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200),
            GroundTruthInterval(video_id="L21_V001", start_frame=500, end_frame=600),
        ]

        # Hit in first interval
        res1 = evaluator.evaluate_query("Q1", [_make_prediction(frame_id=150, rank=1)], gt)
        assert res1.official_final_score == 1.0

        # Hit in second interval
        res2 = evaluator.evaluate_query("Q1", [_make_prediction(frame_id=550, rank=1)], gt)
        assert res2.official_final_score == 1.0

        # In between intervals -> Miss
        res3 = evaluator.evaluate_query("Q1", [_make_prediction(frame_id=300, rank=1)], gt)
        assert res3.official_final_score == 0.0

    def test_multiple_intervals_different_videos(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [
            GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200),
            GroundTruthInterval(video_id="L22_V005", start_frame=300, end_frame=400),
        ]

        res1 = evaluator.evaluate_query(
            "Q1",
            [_make_prediction(video_id="L22_V005", frame_id=350, rank=1)],
            gt,
        )
        assert res1.official_final_score == 1.0
        assert res1.video_first_hit_rank == 1


# ==============================================================================
# 5. Handling N < k and Short Prediction Lists
# ==============================================================================

class TestShortPredictionLists:
    def test_n_less_than_k_evaluates_available_predictions(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]

        # Only 3 predictions: rank 1 misses, rank 2 hits, rank 3 misses
        preds = [
            _make_prediction(frame_id=10, rank=1),
            _make_prediction(frame_id=150, rank=2),
            _make_prediction(frame_id=20, rank=3),
        ]
        res = evaluator.evaluate_query("Q1", preds, gt)

        # For k=1: slice length 1 -> max is 0.0
        # For k in (5, 20, 50, 100): slice length 3 -> max is 1.0 (from rank 2)
        assert res.r_at_k == {1: 0.0, 5: 1.0, 20: 1.0, 50: 1.0, 100: 1.0}
        assert math.isclose(res.official_final_score, 0.8)
        assert res.prediction_count == 3
        assert res.localized_hit_at_k[1] is False
        assert res.localized_hit_at_k[5] is True
        assert res.localized_hit_at_k[20] is True

    def test_n_less_than_k_all_miss(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]

        preds = [
            _make_prediction(frame_id=10, rank=1),
            _make_prediction(frame_id=20, rank=2),
        ]
        res = evaluator.evaluate_query("Q1", preds, gt)
        assert res.official_final_score == 0.0
        assert res.official_first_hit_rank is None
        assert res.r_at_k == {1: 0.0, 5: 0.0, 20: 0.0, 50: 0.0, 100: 0.0}

    def test_empty_predictions_gives_zero(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]

        res = evaluator.evaluate_query("Q1", [], gt)
        assert res.official_final_score == 0.0
        assert res.prediction_count == 0
        assert res.official_first_hit_rank is None
        assert res.point_in_gt_interval is False
        assert res.temporal_iou_at_1 is None


# ==============================================================================
# 6. Missing & Unmapped Queries & Ground Truth Strictness
# ==============================================================================

class TestQueryMappingsAndGTStrictness:
    def test_missing_query_scores_zero_and_increments_missing_count(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt_mapping = {
            "Q1": [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)],
            "Q2": [GroundTruthInterval(video_id="L22_V002", start_frame=300, end_frame=400)],
        }
        # Only predictions for Q1 provided; Q2 is missing
        preds_mapping = {
            "Q1": [_make_prediction(query_id="Q1", frame_id=150, rank=1)],
        }

        report = evaluator.evaluate_predictions(preds_mapping, gt_mapping)
        assert report.query_count == 2
        assert report.missing_query_count == 1
        assert report.query_results["Q1"].official_final_score == 1.0
        assert report.query_results["Q2"].official_final_score == 0.0
        assert report.query_results["Q2"].prediction_count == 0
        assert math.isclose(report.mean_official_final_score, 0.5)

    def test_unmapped_query_in_predictions_fails_closed(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt_mapping = {
            "Q1": [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)],
        }
        preds_mapping = {
            "Q1": [_make_prediction(query_id="Q1", frame_id=150, rank=1)],
            "Q_UNKNOWN": [_make_prediction(query_id="Q_UNKNOWN", frame_id=10, rank=1)],
        }

        with pytest.raises(ValueError, match="Unmapped query 'Q_UNKNOWN'"):
            evaluator.evaluate_predictions(preds_mapping, gt_mapping)

    def test_empty_gt_list_fails_closed_even_with_missing_predictions(self) -> None:
        """Blocker fix: GT mapping with empty list must fail closed even if predictions missing."""
        evaluator = KISFixtureEvaluator()
        gt_mapping: dict[str, list[GroundTruthInterval]] = {
            "Q_EMPTY_GT": [],
        }
        preds_mapping: dict[str, list[PredictionRecord]] = {}

        with pytest.raises(
            ValueError,
            match="Ground truth intervals for query 'Q_EMPTY_GT' cannot be empty",
        ):
            evaluator.evaluate_predictions(preds_mapping, gt_mapping)

    def test_gt_query_id_mismatch_with_mapping_key_rejected(self) -> None:
        """Blocker fix: outer mapping key != inner interval query_id must fail closed."""
        evaluator = KISFixtureEvaluator()
        gt_mapping = {
            "outer-q": [
                GroundTruthInterval(
                    video_id="L21_V001",
                    start_frame=100,
                    end_frame=200,
                    query_id="inner-q",
                )
            ]
        }
        preds_mapping = {
            "outer-q": [_make_prediction(query_id="outer-q", frame_id=150, rank=1)]
        }

        with pytest.raises(ValueError, match="Ground truth query ID mismatch"):
            evaluator.evaluate_predictions(preds_mapping, gt_mapping)

    def test_gt_temporal_bounds_must_appear_in_pairs(self) -> None:
        """Major fix: start_s without end_s (or vice versa) must be rejected."""
        with pytest.raises(ValueError, match="must both be provided or both be None"):
            GroundTruthInterval(
                video_id="L21_V001",
                start_frame=10,
                end_frame=20,
                start_s=1.0,
                end_s=None,
            )

        with pytest.raises(ValueError, match="must both be provided or both be None"):
            GroundTruthInterval(
                video_id="L21_V001",
                start_frame=10,
                end_frame=20,
                start_s=None,
                end_s=2.0,
            )

    def test_pred_temporal_segment_must_appear_in_pairs(self) -> None:
        with pytest.raises(ValueError, match="must both be provided or both be None"):
            _make_prediction(pred_segment_start_s=1.0, pred_segment_end_s=None)

        with pytest.raises(ValueError, match="must both be provided or both be None"):
            _make_prediction(pred_segment_start_s=None, pred_segment_end_s=2.0)


# ==============================================================================
# 7. Strict Parser & Type Guard Tests
# ==============================================================================

class TestStrictParserAndTypes:
    def test_bool_as_int_rejected(self) -> None:
        with pytest.raises(TypeError, match="rank must be an integer"):
            _make_prediction(rank=True)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="frame_id must be an integer"):
            _make_prediction(frame_id=False)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="start_frame must be an integer"):
            GroundTruthInterval(video_id="L21_V001", start_frame=True, end_frame=100)  # type: ignore[arg-type]

    def test_float_as_frame_or_rank_rejected(self) -> None:
        with pytest.raises(TypeError, match="frame_id must be an integer"):
            _make_prediction(frame_id=9.9)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="rank must be an integer"):
            _make_prediction(rank=1.0)  # type: ignore[arg-type]

    def test_numeric_string_frame_or_rank_rejected(self) -> None:
        with pytest.raises(TypeError, match="frame_id must be an integer"):
            _make_prediction(frame_id="150")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="rank must be an integer"):
            _make_prediction(rank="1")  # type: ignore[arg-type]

    def test_numeric_or_whitespace_query_and_video_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="query_id must be a non-empty string"):
            _make_prediction(query_id=123)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="query_id must be a non-empty string"):
            _make_prediction(query_id="   ")

        with pytest.raises(ValueError, match="video_id must be a non-empty string"):
            _make_prediction(video_id=456)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="video_id must be a non-empty string"):
            _make_prediction(video_id="   ")

    def test_non_monotonic_scores_with_valid_ranks_are_allowed(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]

        preds = [
            _make_prediction(frame_id=150, rank=1, score=0.4),
            _make_prediction(frame_id=20, rank=2, score=0.9),
            _make_prediction(frame_id=30, rank=3, score=None),
        ]
        res = evaluator.evaluate_query("Q1", preds, gt)
        assert res.official_final_score == 1.0
        assert res.official_first_hit_rank == 1

    def test_non_finite_score_rejected(self) -> None:
        with pytest.raises(ValueError, match="score must be finite"):
            _make_prediction(rank=1, score=float("nan"))

        with pytest.raises(ValueError, match="score must be finite"):
            _make_prediction(rank=1, score=float("inf"))

    def test_discontinuous_ranks_rejected(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]

        with pytest.raises(ValueError, match="must be continuous 1-indexed integers"):
            evaluator.evaluate_query("Q1", [_make_prediction(rank=2)], gt)

        with pytest.raises(ValueError, match="must be continuous 1-indexed integers"):
            evaluator.evaluate_query(
                "Q1",
                [_make_prediction(rank=1), _make_prediction(rank=3)],
                gt,
            )

    def test_duplicate_video_and_frame_within_query_rejected(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]

        preds = [
            _make_prediction(video_id="L21_V001", frame_id=150, rank=1),
            _make_prediction(video_id="L21_V001", frame_id=150, rank=2),
        ]
        with pytest.raises(ValueError, match="Duplicate \\(video_id, frame_id\\)"):
            evaluator.evaluate_query("Q1", preds, gt)

    def test_duplicate_across_different_queries_allowed(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt_mapping = {
            "Q1": [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)],
            "Q2": [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)],
        }
        preds_mapping = {
            "Q1": [_make_prediction(query_id="Q1", video_id="L21_V001", frame_id=150, rank=1)],
            "Q2": [_make_prediction(query_id="Q2", video_id="L21_V001", frame_id=150, rank=1)],
        }
        report = evaluator.evaluate_predictions(preds_mapping, gt_mapping)
        assert report.mean_official_final_score == 1.0

    def test_exceeding_max_predictions_rejected(self) -> None:
        evaluator = KISFixtureEvaluator(
            scoring_protocol_id="custom",
            cutoffs=DEFAULT_EVALUATION_CUTOFFS,
            max_predictions_per_query=100,
        )
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]

        preds = [_make_prediction(frame_id=i, rank=i) for i in range(1, 102)]
        with pytest.raises(ValueError, match="exceeding maximum of 100"):
            evaluator.evaluate_query("Q1", preds, gt)

    def test_pts_time_parsed_properly(self) -> None:
        """Runtime schema field 'pts_time' must be parsed as timestamp_s."""
        evaluator = KISFixtureEvaluator()
        item = {
            "query_id": "Q1",
            "video_id": "L21_V001",
            "frame_id": 150,
            "rank": 1,
            "pts_time": 6.25,
        }
        rec = evaluator._dict_to_prediction_record(item)
        assert rec.timestamp_s == 6.25


# ==============================================================================
# 8. FrameTimestampResolver & Tolerance Strictness Tests
# ==============================================================================

class TestFrameTimestampResolver:
    def test_tolerance_nan_or_negative_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="timestamp_tolerance_s must be a finite non-negative number",
        ):
            KISFixtureEvaluator(timestamp_tolerance_s=float("nan"))

        with pytest.raises(
            ValueError,
            match="timestamp_tolerance_s must be a finite non-negative number",
        ):
            KISFixtureEvaluator(timestamp_tolerance_s=-0.1)

    def test_resolver_agreement_verified(self) -> None:
        mapping = {("L21_V001", 100): 4.0}
        resolver = MappingFrameTimestampResolver(mapping)
        evaluator = KISFixtureEvaluator(
            frame_timestamp_resolver=resolver,
            timestamp_tolerance_s=0.5,
        )

        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=90, end_frame=110)]
        pred = _make_prediction(
            video_id="L21_V001",
            frame_id=100,
            rank=1,
            timestamp_s=4.2,  # Diff 0.2s <= 0.5s tolerance
        )
        res = evaluator.evaluate_query("Q1", [pred], gt)
        assert res.temporal_cross_check_status == "verified"
        assert res.official_final_score == 1.0

    def test_resolver_disagreement_fails_closed(self) -> None:
        mapping = {("L21_V001", 100): 4.0}
        resolver = MappingFrameTimestampResolver(mapping)
        evaluator = KISFixtureEvaluator(
            frame_timestamp_resolver=resolver,
            timestamp_tolerance_s=0.5,
        )

        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=90, end_frame=110)]
        pred = _make_prediction(
            video_id="L21_V001",
            frame_id=100,
            rank=1,
            timestamp_s=15.0,  # Diff 11.0s > 0.5s tolerance -> FAIL CLOSED
        )
        with pytest.raises(ValueError, match="Frame/timestamp disagreement"):
            evaluator.evaluate_query("Q1", [pred], gt)

    def test_gt_frame_timestamp_disagreement_with_resolver_fails_closed(self) -> None:
        mapping = {("L21_V001", 100): 4.0, ("L21_V001", 200): 8.0}
        resolver = MappingFrameTimestampResolver(mapping)
        evaluator = KISFixtureEvaluator(
            frame_timestamp_resolver=resolver,
            timestamp_tolerance_s=0.5,
        )

        # GT says start_frame=100 (4.0s) but start_s=20.0s (contradiction!)
        gt = [
            GroundTruthInterval(
                video_id="L21_V001",
                start_frame=100,
                end_frame=200,
                start_s=20.0,
                end_s=25.0,
            )
        ]
        pred = _make_prediction(video_id="L21_V001", frame_id=100, rank=1)
        with pytest.raises(ValueError, match="Ground truth start frame/timestamp disagreement"):
            evaluator.evaluate_query("Q1", [pred], gt)

    def test_resolver_returning_nan_or_inf_fails_closed(self) -> None:
        def bad_resolver(vid: str, fid: int) -> float:
            return float("nan")

        evaluator = KISFixtureEvaluator(frame_timestamp_resolver=bad_resolver)
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=90, end_frame=110)]
        pred = _make_prediction(video_id="L21_V001", frame_id=100, rank=1, timestamp_s=4.0)
        with pytest.raises(ValueError, match="Resolver returned non-finite timestamp"):
            evaluator.evaluate_query("Q1", [pred], gt)

    def test_partial_unresolved_timestamp_status(self) -> None:
        # Frame 100 resolved, Frame 200 unknown
        mapping = {("L21_V001", 100): 4.0}
        resolver = MappingFrameTimestampResolver(mapping)
        evaluator = KISFixtureEvaluator(frame_timestamp_resolver=resolver)

        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=90, end_frame=210)]
        preds = [
            _make_prediction(frame_id=100, rank=1, timestamp_s=4.0),
            _make_prediction(frame_id=200, rank=2, timestamp_s=8.0),
        ]
        res = evaluator.evaluate_query("Q1", preds, gt)
        assert res.temporal_cross_check_status == "unresolved-partial"

    def test_no_resolver_leaves_status_unavailable(self) -> None:
        evaluator = KISFixtureEvaluator(frame_timestamp_resolver=None)
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=90, end_frame=110)]
        pred = _make_prediction(video_id="L21_V001", frame_id=100, rank=1, timestamp_s=4.0)
        res = evaluator.evaluate_query("Q1", [pred], gt)
        assert res.temporal_cross_check_status == "unavailable"


# ==============================================================================
# 9. Temporal Segment IoU Handling
# ==============================================================================

class TestTemporalSegmentIoU:
    def test_missing_temporal_data_returns_none(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]
        pred = _make_prediction(
            frame_id=150,
            rank=1,
            pred_segment_start_s=5.0,
            pred_segment_end_s=10.0,
        )
        res = evaluator.evaluate_query("Q1", [pred], gt)
        assert res.temporal_iou_at_1 is None
        assert res.best_temporal_iou_at_k[1] is None

        gt_with_s = [
            GroundTruthInterval(
                video_id="L21_V001",
                start_frame=100,
                end_frame=200,
                start_s=5.0,
                end_s=10.0,
            )
        ]
        pred_no_seg = _make_prediction(frame_id=150, rank=1)
        res2 = evaluator.evaluate_query("Q1", [pred_no_seg], gt_with_s)
        assert res2.temporal_iou_at_1 is None

    def test_disjoint_temporal_segments_returns_zero(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [
            GroundTruthInterval(
                video_id="L21_V001",
                start_frame=100,
                end_frame=200,
                start_s=0.0,
                end_s=10.0,
            )
        ]
        pred = _make_prediction(
            frame_id=150,
            rank=1,
            pred_segment_start_s=20.0,
            pred_segment_end_s=30.0,
        )
        res = evaluator.evaluate_query("Q1", [pred], gt)
        assert res.temporal_iou_at_1 == 0.0
        assert res.best_temporal_iou_at_k[1] == 0.0

    def test_overlapping_temporal_segments_exact_iou(self) -> None:
        evaluator = KISFixtureEvaluator()
        gt = [
            GroundTruthInterval(
                video_id="L21_V001",
                start_frame=100,
                end_frame=200,
                start_s=10.0,
                end_s=20.0,
            )
        ]
        pred = _make_prediction(
            frame_id=150,
            rank=1,
            pred_segment_start_s=15.0,
            pred_segment_end_s=25.0,
        )
        res = evaluator.evaluate_query("Q1", [pred], gt)
        assert res.temporal_iou_at_1 is not None
        assert math.isclose(res.temporal_iou_at_1, 1.0 / 3.0, rel_tol=1e-5)


# ==============================================================================
# 10. File-Based Evaluator Integration & Malformed JSON
# ==============================================================================

@pytest.fixture
def temp_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


class TestFileBasedEvaluation:
    def test_evaluate_json_fixtures(self, temp_dir: Path) -> None:
        evaluator = KISFixtureEvaluator()

        pred_file = temp_dir / "candidates.json"
        gt_file = temp_dir / "ground_truth.json"

        preds_data = {
            "records": [
                {
                    "query_id": "Q1",
                    "video_id": "L21_V001",
                    "frame_id": 150,
                    "rank": 1,
                    "score": 0.98,
                },
                {
                    "query_id": "Q1",
                    "video_id": "L21_V002",
                    "frame_id": 300,
                    "rank": 2,
                    "score": 0.85,
                },
            ]
        }
        gt_data = {
            "Q1": [
                {
                    "video_id": "L21_V001",
                    "start_frame": 100,
                    "end_frame": 200,
                }
            ]
        }

        pred_file.write_text(json.dumps(preds_data), encoding="utf-8")
        gt_file.write_text(json.dumps(gt_data), encoding="utf-8")

        report = evaluator.evaluate(pred_file, gt_file)
        assert report.query_count == 1
        assert report.mean_official_final_score == 1.0
        assert report.mean_video_mrr == 1.0
        assert report["mean_official_final_score"] == 1.0

    def test_evaluate_jsonl_fixtures(self, temp_dir: Path) -> None:
        evaluator = KISFixtureEvaluator()

        pred_file = temp_dir / "preds.jsonl"
        gt_file = temp_dir / "gt.jsonl"

        preds_lines = [
            json.dumps({"query_id": "Q1", "video_id": "L21_V001", "frame_id": 150, "rank": 1}),
            json.dumps({"query_id": "Q2", "video_id": "L22_V002", "frame_id": 50, "rank": 1}),
        ]
        gt_lines = [
            json.dumps({
                "query_id": "Q1",
                "video_id": "L21_V001",
                "start_frame": 100,
                "end_frame": 200,
            }),
            json.dumps({
                "query_id": "Q2",
                "video_id": "L22_V002",
                "start_frame": 100,
                "end_frame": 200,
            }),
        ]

        pred_file.write_text("\n".join(preds_lines), encoding="utf-8")
        gt_file.write_text("\n".join(gt_lines), encoding="utf-8")

        report = evaluator.evaluate(pred_file, gt_file)
        assert report.query_count == 2
        assert math.isclose(report.mean_official_final_score, 0.5)

    def test_evaluate_missing_file_raises_file_not_found(self, temp_dir: Path) -> None:
        evaluator = KISFixtureEvaluator()
        with pytest.raises(FileNotFoundError):
            evaluator.evaluate(temp_dir / "non_existent.json", temp_dir / "gt.json")

    def test_malformed_json_records_raises_value_error(self, temp_dir: Path) -> None:
        evaluator = KISFixtureEvaluator()
        bad_pred = temp_dir / "bad_pred.json"
        bad_pred.write_text(json.dumps({"records": "not-a-list"}), encoding="utf-8")
        gt = temp_dir / "gt.json"
        gt_payload = {"Q1": [{"video_id": "L21_V001", "start_frame": 0}]}
        gt.write_text(json.dumps(gt_payload), encoding="utf-8")

        with pytest.raises(ValueError, match="must be a list"):
            evaluator.evaluate(bad_pred, gt)


# ==============================================================================
# 11. Audited Fail-Closed Edge Cases & Hardening
# ==============================================================================

class TestAuditedEdgeCasesAndHardening:
    def test_evaluate_query_gt_query_id_mismatch_without_resolver_fails_closed(self) -> None:
        evaluator = KISFixtureEvaluator()  # resolver is None
        pred = _make_prediction(query_id="outer", video_id="L21_V001", frame_id=100, rank=1)
        gt = [
            GroundTruthInterval(
                video_id="L21_V001",
                start_frame=50,
                end_frame=150,
                query_id="inner",
            )
        ]
        with pytest.raises(ValueError, match="Ground truth query ID mismatch"):
            evaluator.evaluate_query("outer", [pred], gt)

    def test_conflicting_coordinate_aliases_in_predictions_rejected(
        self, temp_dir: Path
    ) -> None:
        evaluator = KISFixtureEvaluator()
        gt_file = temp_dir / "gt.json"
        gt_file.write_text(
            json.dumps({"Q1": [{"video_id": "L21_V001", "start_frame": 100, "end_frame": 200}]}),
            encoding="utf-8",
        )

        # Conflict: frame_id=1 vs actual_frame_id=999
        p1 = temp_dir / "p1.json"
        p1.write_text(
            json.dumps({
                "records": [
                    {
                        "query_id": "Q1",
                        "video_id": "L21_V001",
                        "frame_id": 1,
                        "actual_frame_id": 999,
                        "rank": 1,
                    }
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Conflicting coordinate aliases for frame_id"):
            evaluator.evaluate(p1, gt_file)

        # Conflict: score=0.9 vs confidence_score=0.1
        p2 = temp_dir / "p2.json"
        p2.write_text(
            json.dumps({
                "records": [
                    {
                        "query_id": "Q1",
                        "video_id": "L21_V001",
                        "frame_id": 150,
                        "rank": 1,
                        "score": 0.9,
                        "confidence_score": 0.1,
                    }
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Conflicting values for score"):
            evaluator.evaluate(p2, gt_file)

        # Conflict: timestamp_s=4.0 vs pts_time=10.0
        p3 = temp_dir / "p3.json"
        p3.write_text(
            json.dumps({
                "records": [
                    {
                        "query_id": "Q1",
                        "video_id": "L21_V001",
                        "frame_id": 150,
                        "rank": 1,
                        "timestamp_s": 4.0,
                        "pts_time": 10.0,
                    }
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Conflicting values for timestamp"):
            evaluator.evaluate(p3, gt_file)

        # Conflict with large numbers: timestamp_s=1_000_000.0 vs pts_time=1_000_000.5
        # Must fail-closed without being masked by relative tolerance (rel_tol=0.0, abs_tol=1e-9)
        p3_drift = temp_dir / "p3_drift.json"
        p3_drift.write_text(
            json.dumps({
                "records": [
                    {
                        "query_id": "Q1",
                        "video_id": "L21_V001",
                        "frame_id": 150,
                        "rank": 1,
                        "timestamp_s": 1000000.0,
                        "pts_time": 1000000.5,
                    }
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Conflicting values for timestamp"):
            evaluator.evaluate(p3_drift, gt_file)

        # Conflict: keyframe_id=0 vs keyframe_order_diagnostic=7
        p4 = temp_dir / "p4.json"
        p4.write_text(
            json.dumps({
                "records": [
                    {
                        "query_id": "Q1",
                        "video_id": "L21_V001",
                        "frame_id": 150,
                        "rank": 1,
                        "keyframe_id": 0,
                        "keyframe_order_diagnostic": 7,
                    }
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Conflicting coordinate aliases for keyframe_id"):
            evaluator.evaluate(p4, gt_file)

    def test_conflicting_coordinate_aliases_in_ground_truth_rejected(
        self, temp_dir: Path
    ) -> None:
        evaluator = KISFixtureEvaluator()
        pred_file = temp_dir / "pred.json"
        pred_file.write_text(
            json.dumps({
                "records": [
                    {
                        "query_id": "Q1",
                        "video_id": "L21_V001",
                        "frame_id": 100,
                        "rank": 1,
                    }
                ]
            }),
            encoding="utf-8",
        )

        # Conflict in GT: start_frame=1 vs target_frame=999
        gt1 = temp_dir / "gt1.json"
        gt1.write_text(
            json.dumps({
                "Q1": [
                    {
                        "video_id": "L21_V001",
                        "start_frame": 1,
                        "target_frame": 999,
                    }
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Conflicting coordinate aliases for start_frame"):
            evaluator.evaluate(pred_file, gt1)

        # Conflict in GT: keyframe_id=0 vs keyframe_order_diagnostic=7
        gt2 = temp_dir / "gt2.json"
        gt2.write_text(
            json.dumps({
                "Q1": [
                    {
                        "video_id": "L21_V001",
                        "start_frame": 100,
                        "keyframe_id": 0,
                        "keyframe_order_diagnostic": 7,
                    }
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Conflicting coordinate aliases for keyframe_id"):
            evaluator.evaluate(pred_file, gt2)

    def test_matching_coordinate_aliases_accepted_and_zero_preserved(
        self, temp_dir: Path
    ) -> None:
        evaluator = KISFixtureEvaluator()
        pred_file = temp_dir / "pred_match.json"
        pred_file.write_text(
            json.dumps({
                "records": [
                    {
                        "query_id": "Q1",
                        "video_id": "L21_V001",
                        "frame_id": 0,
                        "actual_frame_id": 0,
                        "rank": 1,
                        "keyframe_id": 0,
                        "keyframe_order_diagnostic": 0,
                        "score": 0.85,
                        "confidence_score": 0.85,
                        "timestamp_s": 0.0,
                        "pts_time": 0.0,
                    }
                ]
            }),
            encoding="utf-8",
        )

        gt_file = temp_dir / "gt_match.json"
        gt_file.write_text(
            json.dumps({
                "Q1": [
                    {
                        "video_id": "L21_V001",
                        "start_frame": 0,
                        "target_frame": 0,
                        "end_frame": 0,
                        "keyframe_id": 0,
                        "keyframe_order_diagnostic": 0,
                    }
                ]
            }),
            encoding="utf-8",
        )

        report = evaluator.evaluate(pred_file, gt_file)
        assert report.query_count == 1
        assert report.mean_official_final_score == 1.0
        res = report.query_results["Q1"]
        assert res.point_in_gt_interval is True

    def test_envelope_query_id_mismatch_fails_closed(
        self, temp_dir: Path
    ) -> None:
        evaluator = KISFixtureEvaluator()
        gt_file = temp_dir / "gt.json"
        gt_file.write_text(
            json.dumps({"outer": [{"video_id": "L21_V001", "start_frame": 100}]}),
            encoding="utf-8",
        )

        # Root envelope query_id="outer" but record specifies "inner"
        p_bad1 = temp_dir / "p_bad1.json"
        p_bad1.write_text(
            json.dumps({
                "query_id": "outer",
                "records": [
                    {"query_id": "inner", "video_id": "L21_V001", "frame_id": 100, "rank": 1}
                ],
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="expected 'outer'"):
            evaluator.evaluate(p_bad1, gt_file)

        # Root envelope query_id="outer" but queries contains "inner"
        p_bad2 = temp_dir / "p_bad2.json"
        p_bad2.write_text(
            json.dumps({
                "query_id": "outer",
                "queries": {"inner": [{"video_id": "L21_V001", "frame_id": 100, "rank": 1}]},
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Envelope query_id mismatch"):
            evaluator.evaluate(p_bad2, gt_file)

        # GT root envelope query_id="outer" but mapping has "inner"
        gt_bad = temp_dir / "gt_bad.json"
        gt_bad.write_text(
            json.dumps({
                "query_id": "outer",
                "inner": [{"video_id": "L21_V001", "start_frame": 100}],
            }),
            encoding="utf-8",
        )
        p_good = temp_dir / "p_good.json"
        p_good.write_text(
            json.dumps({
                "records": [
                    {"query_id": "outer", "video_id": "L21_V001", "frame_id": 100, "rank": 1}
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Envelope query_id mismatch"):
            evaluator.evaluate(p_good, gt_bad)

    def test_invalid_resolver_rejected_at_initialization(self) -> None:
        with pytest.raises(
            TypeError, match="frame_timestamp_resolver must implement FrameTimestampResolver"
        ):
            KISFixtureEvaluator(frame_timestamp_resolver=object())

        with pytest.raises(
            TypeError, match="frame_timestamp_resolver must implement FrameTimestampResolver"
        ):
            KISFixtureEvaluator(frame_timestamp_resolver="invalid_string")

        with pytest.raises(
            TypeError, match="frame_timestamp_resolver must implement FrameTimestampResolver"
        ):
            KISFixtureEvaluator(frame_timestamp_resolver=123)

        # Valid callable succeeds
        ev1 = KISFixtureEvaluator(frame_timestamp_resolver=lambda v, f: 0.0)
        assert ev1.resolver is not None

        # Valid MappingFrameTimestampResolver succeeds
        ev2 = KISFixtureEvaluator(frame_timestamp_resolver=MappingFrameTimestampResolver({}))
        assert ev2.resolver is not None

    def test_temporal_cross_check_status_includes_both_prediction_and_gt(self) -> None:
        mapping = {("L21_V001", 100): 4.0}
        resolver = MappingFrameTimestampResolver(mapping)
        evaluator = KISFixtureEvaluator(
            frame_timestamp_resolver=resolver,
            timestamp_tolerance_s=0.05,
        )

        # 1. Prediction resolves (100 -> 4.0s), but GT endpoints (50, 150) are NOT in resolver
        # Must be "unresolved-partial", NOT "verified"!
        gt_unresolved = [
            GroundTruthInterval(
                video_id="L21_V001",
                start_frame=50,
                end_frame=150,
                start_s=2.0,
                end_s=6.0,
            )
        ]
        pred = _make_prediction(
            query_id="Q1", video_id="L21_V001", frame_id=100, rank=1, timestamp_s=4.0
        )
        res1 = evaluator.evaluate_query("Q1", [pred], gt_unresolved)
        assert res1.temporal_cross_check_status == "unresolved-partial"

        # 2. Both prediction and all GT endpoints resolve and match -> "verified"
        full_mapping = {
            ("L21_V001", 100): 4.0,
            ("L21_V001", 50): 2.0,
            ("L21_V001", 150): 6.0,
        }
        evaluator_full = KISFixtureEvaluator(
            frame_timestamp_resolver=MappingFrameTimestampResolver(full_mapping),
            timestamp_tolerance_s=0.05,
        )
        res2 = evaluator_full.evaluate_query("Q1", [pred], gt_unresolved)
        assert res2.temporal_cross_check_status == "verified"

        # 3. None of the timestamps resolve -> "unresolved"
        evaluator_empty = KISFixtureEvaluator(
            frame_timestamp_resolver=MappingFrameTimestampResolver({}),
            timestamp_tolerance_s=0.05,
        )
        res3 = evaluator_empty.evaluate_query("Q1", [pred], gt_unresolved)
        assert res3.temporal_cross_check_status == "unresolved"

        # 4. Neither prediction nor GT has timestamps -> "unavailable"
        gt_no_ts = [GroundTruthInterval(video_id="L21_V001", start_frame=50, end_frame=150)]
        pred_no_ts = _make_prediction(
            query_id="Q1", video_id="L21_V001", frame_id=100, rank=1
        )
        res4 = evaluator_full.evaluate_query("Q1", [pred_no_ts], gt_no_ts)
        assert res4.temporal_cross_check_status == "unavailable"

    def test_calibrated_tolerance_rejects_sub_second_drift(self) -> None:
        # Default tolerance is 0.05s (50ms). A drift of 0.1s (~2.5 frames) must be rejected.
        mapping = {("L21_V001", 100): 4.0}
        evaluator = KISFixtureEvaluator(
            frame_timestamp_resolver=MappingFrameTimestampResolver(mapping)
        )
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=50, end_frame=150)]
        pred = _make_prediction(
            query_id="Q1",
            video_id="L21_V001",
            frame_id=100,
            rank=1,
            timestamp_s=4.1,  # 100ms drift > 50ms tolerance
        )
        with pytest.raises(ValueError, match="Frame/timestamp disagreement"):
            evaluator.evaluate_query("Q1", [pred], gt)

    def test_fractional_rscore_support_and_scorer_callback(self) -> None:
        def fractional_scorer(
            pred: PredictionRecord,
            ground_truth: Sequence[GroundTruthInterval],
        ) -> float:
            return 0.75 if pred.rank == 1 else 0.25

        custom_protocol = "custom-fractional-eval"
        evaluator = KISFixtureEvaluator(
            scoring_protocol_id=custom_protocol,
            scorer=fractional_scorer,
        )
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=50, end_frame=150)]
        p1 = _make_prediction(query_id="Q1", video_id="L21_V001", frame_id=100, rank=1)
        p2 = _make_prediction(query_id="Q1", video_id="L21_V001", frame_id=200, rank=2)
        res = evaluator.evaluate_query("Q1", [p1, p2], gt)
        assert res.r_at_k[1] == 0.75
        assert res.r_at_k[5] == 0.75
        assert res.official_final_score == 0.75

        # Scorer returning bool -> ValueError
        evaluator_bad = KISFixtureEvaluator(
            scoring_protocol_id=custom_protocol,
            scorer=lambda p, gt: True,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="Scorer returned invalid RScore"):
            evaluator_bad.evaluate_query("Q1", [p1], gt)

        # Scorer returning > 1.0 -> ValueError
        evaluator_out_of_bounds = KISFixtureEvaluator(
            scoring_protocol_id=custom_protocol,
            scorer=lambda p, gt: 1.5,
        )
        with pytest.raises(ValueError, match="Scorer returned invalid RScore"):
            evaluator_out_of_bounds.evaluate_query("Q1", [p1], gt)

        # Invalid scorer object in constructor -> TypeError
        with pytest.raises(TypeError, match="scorer must implement RScoreScorer"):
            KISFixtureEvaluator(
                scoring_protocol_id=custom_protocol,
                scorer=object(),  # type: ignore[arg-type]
            )

    def test_custom_scorer_with_official_protocol_id_fails_closed(self) -> None:
        def custom_scorer(
            pred: PredictionRecord,
            gt: Sequence[GroundTruthInterval],
        ) -> float:
            return 0.5

        # Default scoring_protocol_id (official) with custom scorer must raise ValueError
        with pytest.raises(
            ValueError,
            match="Custom scorer requires a custom scoring_protocol_id",
        ):
            KISFixtureEvaluator(scorer=custom_scorer)

        # Explicitly passing DEFAULT_SCORING_PROTOCOL_ID with custom scorer
        # must also raise ValueError
        with pytest.raises(
            ValueError,
            match="Custom scorer requires a custom scoring_protocol_id",
        ):
            KISFixtureEvaluator(
                scoring_protocol_id=DEFAULT_SCORING_PROTOCOL_ID,
                scorer=custom_scorer,
            )

    def test_scorer_returning_one_on_wrong_video_does_not_corrupt_localized_metrics(self) -> None:
        custom_protocol = "custom-adversarial-scorer-test"
        evaluator = KISFixtureEvaluator(
            scoring_protocol_id=custom_protocol,
            scorer=lambda pred, gt: 1.0,
        )
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]
        p_wrong = _make_prediction(
            query_id="Q1", video_id="L99_V999", frame_id=9999, rank=1
        )

        res = evaluator.evaluate_query("Q1", [p_wrong], gt)

        # Scorer drives R@k and official_final_score:
        assert res.r_at_k[1] == 1.0
        assert res.official_final_score == 1.0

        # BUT localized and video metrics MUST remain strictly uncorrupted:
        assert res.official_first_hit_rank is None
        assert res.video_first_hit_rank is None
        assert res.video_mrr == 0.0
        assert res.localized_mrr == 0.0
        assert res.localized_hit_at_k[1] is False
        assert res.video_hit_at_k[1] is False
        assert res.point_in_gt_interval is False

    def test_scorer_returning_zero_on_correct_hit_preserves_localized_metrics(self) -> None:
        custom_protocol = "custom-zero-scorer-test"
        evaluator = KISFixtureEvaluator(
            scoring_protocol_id=custom_protocol,
            scorer=lambda pred, gt: 0.0,
        )
        gt = [GroundTruthInterval(video_id="L21_V001", start_frame=100, end_frame=200)]
        p_correct = _make_prediction(
            query_id="Q1", video_id="L21_V001", frame_id=150, rank=1
        )

        res = evaluator.evaluate_query("Q1", [p_correct], gt)

        # Scorer drives R@k and official_final_score to 0.0:
        assert res.r_at_k[1] == 0.0
        assert res.official_final_score == 0.0

        # BUT localized and video metrics MUST correctly record the physical hit:
        assert res.official_first_hit_rank == 1
        assert res.video_first_hit_rank == 1
        assert res.video_mrr == 1.0
        assert res.localized_mrr == 1.0
        assert res.localized_hit_at_k[1] is True
        assert res.video_hit_at_k[1] is True
        assert res.point_in_gt_interval is True

    def test_conflicting_float_aliases_with_large_values_rejected(
        self, temp_dir: Path
    ) -> None:
        evaluator = KISFixtureEvaluator()
        gt_file = temp_dir / "gt_large.json"
        gt_file.write_text(
            json.dumps({"Q1": [{"video_id": "L21_V001", "start_frame": 100, "end_frame": 200}]}),
            encoding="utf-8",
        )
        pred_file = temp_dir / "pred_large.json"
        pred_file.write_text(
            json.dumps({
                "records": [
                    {
                        "query_id": "Q1",
                        "video_id": "L21_V001",
                        "frame_id": 150,
                        "rank": 1,
                        "timestamp_s": 1000000.0,
                        "pts_time": 1000000.5,
                    }
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Conflicting values for timestamp"):
            evaluator.evaluate(pred_file, gt_file)
