"""Ground-truth fixture evaluator for KIS (Known Item Search).

Implements the official AI Challenge evaluation protocol:
- Protocol ID: "aichallenge-hcmc-2025-mean-topk-rscore"
- Evaluation Cutoffs: K in {1, 5, 20, 50, 100}
- Formula:
    R@k = max_{1 <= i <= min(k, N)} RScore(r_i)
    FinalScore = (1 / 5) * sum_{k in {1, 5, 20, 50, 100}} R@k

For binary Textual-KIS RScore:
- Rank 1 hit: 1.0 (5/5)
- Rank 2-5 hit: 0.8 (4/5)
- Rank 6-20 hit: 0.6 (3/5)
- Rank 21-50 hit: 0.4 (2/5)
- Rank 51-100 hit: 0.2 (1/5)
- > 100 / Miss: 0.0 (0/5)

Official hit condition:
Hit(p, GT) <=> exists gt in GT: (p.video_id == gt.video_id and
                                 gt.start_frame <= p.frame_id <= gt.end_frame)
Strictly frame-interval based (zero OR with timestamp).
Disagreement between frame and timestamp when resolver is provided raises
ValueError (fail-closed).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

DEFAULT_SCORING_PROTOCOL_ID: str = "aichallenge-hcmc-2025-mean-topk-rscore"
DEFAULT_EVALUATION_CUTOFFS: tuple[int, ...] = (1, 5, 20, 50, 100)
MAX_PREDICTIONS_PER_QUERY: int = 100
# Provisional conservative policy: 0.05s (50ms ~ 1.25 frames at 25 FPS or 1.5 frames at 30 FPS).
# Dataset-wide calibration across all videos is pending.
DEFAULT_TIMESTAMP_TOLERANCE_S: float = 0.05


@dataclass(frozen=True)
class GroundTruthInterval:
    """Ground truth interval specification for a KIS query."""

    video_id: str
    start_frame: int
    end_frame: int
    start_s: float | None = None
    end_s: float | None = None
    keyframe_id: int | None = None
    query_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.video_id) is not str or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        if self.query_id is not None and (
            type(self.query_id) is not str or not self.query_id.strip()
        ):
            raise ValueError("query_id must be a non-empty string if provided")
        if type(self.start_frame) is not int:
            raise TypeError(
                f"start_frame must be an integer, got {type(self.start_frame).__name__}"
            )
        if type(self.end_frame) is not int:
            raise TypeError(
                f"end_frame must be an integer, got {type(self.end_frame).__name__}"
            )
        if self.start_frame < 0 or self.end_frame < 0:
            raise ValueError(
                f"Frame indices must be non-negative, got [{self.start_frame}, {self.end_frame}]"
            )
        if self.start_frame > self.end_frame:
            raise ValueError(
                f"start_frame ({self.start_frame}) cannot exceed end_frame ({self.end_frame})"
            )
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError(
                f"start_s ({self.start_s}) and end_s ({self.end_s}) must both be provided "
                "or both be None"
            )
        if self.start_s is not None:
            if (
                not isinstance(self.start_s, (int, float))
                or isinstance(self.start_s, bool)
                or not math.isfinite(self.start_s)
            ):
                raise ValueError(f"start_s must be a finite number, got {self.start_s}")
            if (
                not isinstance(self.end_s, (int, float))
                or isinstance(self.end_s, bool)
                or not math.isfinite(self.end_s)  # type: ignore[arg-type]
            ):
                raise ValueError(f"end_s must be a finite number, got {self.end_s}")
            if self.start_s > self.end_s:  # type: ignore[operator]
                raise ValueError(
                    f"start_s ({self.start_s}) cannot exceed end_s ({self.end_s})"
                )


@dataclass(frozen=True)
class PredictionRecord:
    """Single candidate prediction record for a KIS query."""

    query_id: str
    video_id: str
    frame_id: int  # Original video frame index for submission
    rank: int  # 1-indexed: 1, 2, ..., N (N <= 100)
    score: float | None = None  # Confidence / fusion score (not required to be monotonic)
    timestamp_s: float | None = None
    keyframe_id: int | None = None
    pred_segment_start_s: float | None = None
    pred_segment_end_s: float | None = None

    def __post_init__(self) -> None:
        if type(self.query_id) is not str or not self.query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if type(self.video_id) is not str or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        if type(self.frame_id) is not int:
            raise TypeError(
                f"frame_id must be an integer, got {type(self.frame_id).__name__}"
            )
        if type(self.rank) is not int:
            raise TypeError(
                f"rank must be an integer, got {type(self.rank).__name__}"
            )
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be non-negative, got {self.frame_id}")
        if self.rank < 1:
            raise ValueError(f"rank must be a 1-indexed positive integer, got {self.rank}")
        if self.score is not None:
            if (
                not isinstance(self.score, (int, float))
                or isinstance(self.score, bool)
                or not math.isfinite(self.score)
            ):
                raise ValueError(f"score must be finite if provided, got {self.score}")
        if self.timestamp_s is not None:
            if (
                not isinstance(self.timestamp_s, (int, float))
                or isinstance(self.timestamp_s, bool)
                or not math.isfinite(self.timestamp_s)
            ):
                raise ValueError(
                    f"timestamp_s must be finite if provided, got {self.timestamp_s}"
                )
        if (self.pred_segment_start_s is None) != (self.pred_segment_end_s is None):
            raise ValueError(
                "pred_segment_start_s and pred_segment_end_s must both be provided or both be None"
            )
        if self.pred_segment_start_s is not None:
            if (
                not isinstance(self.pred_segment_start_s, (int, float))
                or isinstance(self.pred_segment_start_s, bool)
                or not math.isfinite(self.pred_segment_start_s)
            ):
                raise ValueError(
                    f"pred_segment_start_s must be finite, got {self.pred_segment_start_s}"
                )
            if (
                not isinstance(self.pred_segment_end_s, (int, float))
                or isinstance(self.pred_segment_end_s, bool)
                or not math.isfinite(self.pred_segment_end_s)  # type: ignore[arg-type]
            ):
                raise ValueError(
                    f"pred_segment_end_s must be finite, got {self.pred_segment_end_s}"
                )
            if self.pred_segment_start_s > self.pred_segment_end_s:  # type: ignore[operator]
                raise ValueError(
                    f"pred_segment_start_s ({self.pred_segment_start_s}) cannot exceed "
                    f"pred_segment_end_s ({self.pred_segment_end_s})"
                )


@runtime_checkable
class FrameTimestampResolver(Protocol):
    """Protocol for resolving original frame index to timestamp in seconds."""

    def resolve(self, video_id: str, frame_id: int) -> float | None:
        """Resolve original frame_id to timestamp in seconds, or None if unknown."""
        ...


class MappingFrameTimestampResolver:
    """Dictionary-backed implementation of FrameTimestampResolver."""

    def __init__(self, mapping: Mapping[tuple[str, int], float]) -> None:
        self._mapping = mapping

    def resolve(self, video_id: str, frame_id: int) -> float | None:
        return self._mapping.get((video_id, frame_id))


@runtime_checkable
class RScoreScorer(Protocol):
    """Protocol for scoring a single prediction against ground truth intervals.

    Returns an RScore float in [0.0, 1.0].
    """

    def score(
        self,
        prediction: PredictionRecord,
        ground_truth: Sequence[GroundTruthInterval],
    ) -> float:
        """Compute RScore in [0.0, 1.0] for a prediction against ground truth intervals."""
        ...


def _extract_aliased_int(
    item: Mapping[str, Any],
    keys: Sequence[str],
    field_label: str,
    *,
    required: bool = False,
    non_negative: bool = True,
    min_value: int | None = None,
) -> int | None:
    """Extract an integer field from item while verifying alias consistency.

    If multiple aliases appear with different values, raises ValueError (fail-closed).
    Rejects bool, float, and non-integer types with TypeError.
    """
    found: list[tuple[str, int]] = []
    for k in keys:
        if k in item and item[k] is not None:
            val = item[k]
            if type(val) is not int:
                raise TypeError(
                    f"{field_label} ({k}) must be an integer, got {type(val).__name__}: {val!r}"
                )
            if non_negative and val < 0:
                raise ValueError(
                    f"{field_label} ({k}) must be non-negative, got {val}"
                )
            if min_value is not None and val < min_value:
                raise ValueError(
                    f"{field_label} ({k}) must be >= {min_value}, got {val}"
                )
            found.append((k, val))

    if not found:
        if required:
            raise ValueError(
                f"Missing {field_label} (expected one of {list(keys)}) in item: {item}"
            )
        return None

    first_key, first_val = found[0]
    for k, val in found[1:]:
        if val != first_val:
            raise ValueError(
                f"Conflicting coordinate aliases for {field_label}: "
                f"{first_key}={first_val} vs {k}={val}"
            )

    return first_val


def _extract_aliased_float(
    item: Mapping[str, Any],
    keys: Sequence[str],
    field_label: str,
    *,
    required: bool = False,
    rel_tol: float = 0.0,
    abs_tol: float = 1e-9,
) -> float | None:
    """Extract a float field from item while verifying alias consistency.

    If multiple aliases appear with conflicting values, raises ValueError (fail-closed).
    Rejects bool and non-finite numbers.
    Enforces strict identity with rel_tol=0.0 and abs_tol=1e-9 to prevent drift on large values.
    """
    found: list[tuple[str, float]] = []
    for k in keys:
        if k in item and item[k] is not None:
            val = item[k]
            if (
                not isinstance(val, (int, float))
                or isinstance(val, bool)
                or not math.isfinite(val)
            ):
                raise ValueError(
                    f"{field_label} ({k}) must be a finite number, got {val!r}"
                )
            found.append((k, float(val)))

    if not found:
        if required:
            raise ValueError(
                f"Missing {field_label} (expected one of {list(keys)}) in item: {item}"
            )
        return None

    first_key, first_val = found[0]
    for k, val in found[1:]:
        if not math.isclose(val, first_val, rel_tol=rel_tol, abs_tol=abs_tol):
            raise ValueError(
                f"Conflicting values for {field_label}: {first_key}={first_val} vs {k}={val}"
            )

    return first_val


@dataclass(frozen=True)
class QueryEvaluationResult:
    """Detailed evaluation result for a single KIS query."""

    query_id: str
    scoring_protocol_id: str
    official_final_score: float  # Mean of top-k R-Scores across cutoffs
    official_first_hit_rank: int | None
    video_first_hit_rank: int | None
    video_mrr: float
    localized_mrr: float
    r_at_k: dict[int, float]
    localized_hit_at_k: dict[int, bool]
    video_hit_at_k: dict[int, bool]
    point_in_gt_interval: bool
    temporal_iou_at_1: float | None
    best_temporal_iou_at_k: dict[int, float | None]
    prediction_count: int
    temporal_cross_check_status: str = "unavailable"


@dataclass(frozen=True)
class EvaluationReport(Mapping[str, Any]):
    """Aggregated evaluation report across all queries."""

    scoring_protocol_id: str
    query_results: dict[str, QueryEvaluationResult]
    mean_official_final_score: float
    query_count: int
    missing_query_count: int = 0
    mean_video_mrr: float = 0.0
    mean_localized_mrr: float = 0.0
    mean_video_hit_at_k: dict[int, float] = field(default_factory=dict)
    mean_localized_hit_at_k: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


class KISFixtureEvaluator:
    """Official evaluator for KIS ground-truth fixtures."""

    def __init__(
        self,
        scoring_protocol_id: str = DEFAULT_SCORING_PROTOCOL_ID,
        cutoffs: Sequence[int] = DEFAULT_EVALUATION_CUTOFFS,
        max_predictions_per_query: int = MAX_PREDICTIONS_PER_QUERY,
        timestamp_tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
        frame_timestamp_resolver: (
            FrameTimestampResolver | Callable[[str, int], float | None] | None
        ) = None,
        scorer: (
            RScoreScorer
            | Callable[[PredictionRecord, Sequence[GroundTruthInterval]], float]
            | None
        ) = None,
    ) -> None:
        if type(scoring_protocol_id) is not str or not scoring_protocol_id.strip():
            raise ValueError("scoring_protocol_id must be a non-empty string")

        if not cutoffs:
            raise ValueError("cutoffs cannot be empty")
        for k in cutoffs:
            if type(k) is not int or k <= 0:
                raise TypeError(f"Each cutoff must be a positive integer, got {k!r}")
        if len(set(cutoffs)) != len(cutoffs):
            raise ValueError(f"cutoffs cannot contain duplicates: {cutoffs}")

        if type(max_predictions_per_query) is not int or max_predictions_per_query <= 0:
            raise TypeError(
                f"max_predictions_per_query must be a positive integer, got "
                f"{max_predictions_per_query!r}"
            )

        if (
            not isinstance(timestamp_tolerance_s, (int, float))
            or isinstance(timestamp_tolerance_s, bool)
            or not math.isfinite(timestamp_tolerance_s)
            or timestamp_tolerance_s < 0.0
        ):
            raise ValueError(
                f"timestamp_tolerance_s must be a finite non-negative number, got "
                f"{timestamp_tolerance_s!r}"
            )

        self.scoring_protocol_id = scoring_protocol_id
        self.cutoffs = tuple(sorted(cutoffs))
        self.max_predictions_per_query = max_predictions_per_query
        self.timestamp_tolerance_s = float(timestamp_tolerance_s)

        # Enforce strict protocol lock for the official AI Challenge protocol
        if self.scoring_protocol_id == DEFAULT_SCORING_PROTOCOL_ID:
            if scorer is not None:
                raise ValueError(
                    "Custom scorer requires a custom scoring_protocol_id"
                )
            if self.cutoffs != DEFAULT_EVALUATION_CUTOFFS:
                raise ValueError(
                    f"Official scoring protocol '{DEFAULT_SCORING_PROTOCOL_ID}' strictly requires "
                    f"cutoffs={DEFAULT_EVALUATION_CUTOFFS}, got {self.cutoffs}. "
                    "For custom cutoffs, specify a custom scoring_protocol_id."
                )
            if self.max_predictions_per_query != MAX_PREDICTIONS_PER_QUERY:
                raise ValueError(
                    f"Official scoring protocol '{DEFAULT_SCORING_PROTOCOL_ID}' strictly requires "
                    f"max_predictions_per_query={MAX_PREDICTIONS_PER_QUERY}, got "
                    f"{self.max_predictions_per_query}. "
                    "For custom limits, specify a custom scoring_protocol_id."
                )

        if frame_timestamp_resolver is not None:
            has_resolve = hasattr(frame_timestamp_resolver, "resolve") and callable(
                getattr(frame_timestamp_resolver, "resolve")
            )
            is_callable = callable(frame_timestamp_resolver)
            if not (has_resolve or is_callable):
                raise TypeError(
                    "frame_timestamp_resolver must implement FrameTimestampResolver (with callable "
                    f".resolve) or be a callable, got {type(frame_timestamp_resolver).__name__}: "
                    f"{frame_timestamp_resolver!r}"
                )
            if not has_resolve and is_callable:
                class _CallableResolverWrapper:
                    def __init__(self, fn: Callable[[str, int], float | None]) -> None:
                        self._fn = fn

                    def resolve(self, video_id: str, frame_id: int) -> float | None:
                        return self._fn(video_id, frame_id)

                self.resolver: FrameTimestampResolver | None = _CallableResolverWrapper(
                    frame_timestamp_resolver
                )
            else:
                self.resolver = frame_timestamp_resolver
        else:
            self.resolver = None

        if scorer is not None:
            has_score_method = hasattr(scorer, "score") and callable(getattr(scorer, "score"))
            is_callable_scorer = callable(scorer)
            if not (has_score_method or is_callable_scorer):
                raise TypeError(
                    "scorer must implement RScoreScorer or be a callable, "
                    f"got {type(scorer).__name__}"
                )
            self.scorer: (
                RScoreScorer
                | Callable[[PredictionRecord, Sequence[GroundTruthInterval]], float]
                | None
            ) = scorer
        else:
            self.scorer = None

    def _compute_segment_iou(
        self,
        pred_start: float | None,
        pred_end: float | None,
        gt_start: float | None,
        gt_end: float | None,
    ) -> float | None:
        """Compute IoU between two temporal segments.

        Returns None if either segment lacks bounds.
        Returns 0.0 if bounds exist but intervals do not overlap.
        """
        if pred_start is None or pred_end is None or gt_start is None or gt_end is None:
            return None
        inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
        union = max(pred_end, gt_end) - min(pred_start, gt_start)
        return inter / union if union > 0.0 else 0.0

    def evaluate_query(
        self,
        query_id: str,
        predictions: Sequence[PredictionRecord],
        ground_truth: Sequence[GroundTruthInterval],
    ) -> QueryEvaluationResult:
        """Evaluate predictions for a single query against ground truth intervals."""
        if type(query_id) is not str or not query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if not ground_truth:
            raise ValueError(f"Ground truth intervals for query '{query_id}' cannot be empty")

        # BLOCKER 1 FIX: Unconditionally validate ground truth interval query_id
        for gt in ground_truth:
            if gt.query_id is not None and gt.query_id != query_id:
                raise ValueError(
                    f"Ground truth query ID mismatch: evaluating query '{query_id}' but "
                    f"interval has query_id '{gt.query_id}'"
                )

        n_preds = len(predictions)
        if n_preds > self.max_predictions_per_query:
            raise ValueError(
                f"Query '{query_id}' has {n_preds} predictions, exceeding maximum of "
                f"{self.max_predictions_per_query}"
            )

        # Cross-verify ground truth intervals and predictions with resolver
        verified_count = 0
        unresolved_count = 0
        total_timestamps_evaluated = 0

        if self.resolver is not None:
            for gt in ground_truth:
                if gt.start_s is not None:
                    total_timestamps_evaluated += 1
                    resolved_start = self.resolver.resolve(gt.video_id, gt.start_frame)
                    if resolved_start is not None:
                        if (
                            not isinstance(resolved_start, (int, float))
                            or isinstance(resolved_start, bool)
                            or not math.isfinite(resolved_start)
                        ):
                            raise ValueError(
                                f"Resolver returned non-finite timestamp for {gt.video_id}:"
                                f"{gt.start_frame}: {resolved_start!r}"
                            )
                        diff = abs(gt.start_s - resolved_start)
                        if diff > self.timestamp_tolerance_s:
                            raise ValueError(
                                f"Ground truth start frame/timestamp disagreement for query "
                                f"'{query_id}', video '{gt.video_id}', frame {gt.start_frame}: "
                                f"GT {gt.start_s}s vs resolved {resolved_start}s "
                                f"(tolerance {self.timestamp_tolerance_s}s)"
                            )
                        verified_count += 1
                    else:
                        unresolved_count += 1

                if gt.end_s is not None:
                    total_timestamps_evaluated += 1
                    resolved_end = self.resolver.resolve(gt.video_id, gt.end_frame)
                    if resolved_end is not None:
                        if (
                            not isinstance(resolved_end, (int, float))
                            or isinstance(resolved_end, bool)
                            or not math.isfinite(resolved_end)
                        ):
                            raise ValueError(
                                f"Resolver returned non-finite timestamp for {gt.video_id}:"
                                f"{gt.end_frame}: {resolved_end!r}"
                            )
                        diff = abs(gt.end_s - resolved_end)
                        if diff > self.timestamp_tolerance_s:
                            raise ValueError(
                                f"Ground truth end frame/timestamp disagreement for query "
                                f"'{query_id}', video '{gt.video_id}', frame {gt.end_frame}: "
                                f"GT {gt.end_s}s vs resolved {resolved_end}s "
                                f"(tolerance {self.timestamp_tolerance_s}s)"
                            )
                        verified_count += 1
                    else:
                        unresolved_count += 1
        else:
            for gt in ground_truth:
                if gt.start_s is not None or gt.end_s is not None:
                    total_timestamps_evaluated += 1

        # Validate rank continuity and check for duplicates within this query
        seen_pairs: set[tuple[str, int]] = set()

        for expected_rank, p in enumerate(predictions, start=1):
            if p.query_id != query_id:
                raise ValueError(
                    f"Prediction query_id '{p.query_id}' does not match target query '{query_id}'"
                )
            if p.rank != expected_rank:
                raise ValueError(
                    f"Prediction ranks for query '{query_id}' must be continuous 1-indexed "
                    f"integers: expected rank {expected_rank}, got {p.rank}"
                )

            pair = (p.video_id, p.frame_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"Duplicate (video_id, frame_id) found in query '{query_id}': {pair}"
                )
            seen_pairs.add(pair)

            # Resolver cross-check for predictions
            if p.timestamp_s is not None:
                total_timestamps_evaluated += 1
                if self.resolver is not None:
                    resolved_s = self.resolver.resolve(p.video_id, p.frame_id)
                    if resolved_s is not None:
                        if (
                            not isinstance(resolved_s, (int, float))
                            or isinstance(resolved_s, bool)
                            or not math.isfinite(resolved_s)
                        ):
                            raise ValueError(
                                f"Resolver returned non-finite timestamp for {p.video_id}:"
                                f"{p.frame_id}: {resolved_s!r}"
                            )
                        diff = abs(p.timestamp_s - resolved_s)
                        if diff > self.timestamp_tolerance_s:
                            raise ValueError(
                                f"Frame/timestamp disagreement for query '{query_id}', "
                                f"video '{p.video_id}', frame {p.frame_id}: prediction timestamp "
                                f"{p.timestamp_s}s vs resolved {resolved_s}s "
                                f"(diff {diff:.4f}s > tolerance {self.timestamp_tolerance_s}s)"
                            )
                        verified_count += 1
                    else:
                        unresolved_count += 1

        if self.resolver is None or total_timestamps_evaluated == 0:
            cross_check_status = "unavailable"
        elif verified_count > 0 and unresolved_count == 0:
            cross_check_status = "verified"
        elif verified_count > 0 and unresolved_count > 0:
            cross_check_status = "unresolved-partial"
        else:
            # verified_count == 0 and unresolved_count > 0
            cross_check_status = "unresolved"

        # Check hits per prediction
        first_localized_rank: int | None = None
        first_video_rank: int | None = None
        per_pred_rscore: list[float] = []
        per_pred_best_iou: list[float | None] = []

        gt_video_ids = {gt.video_id for gt in ground_truth}

        for p in predictions:
            # Video hit check
            is_video_match = p.video_id in gt_video_ids
            if is_video_match and first_video_rank is None:
                first_video_rank = p.rank

            # Localized hit check: original video frame index range
            is_localized_hit = any(
                p.video_id == gt.video_id and gt.start_frame <= p.frame_id <= gt.end_frame
                for gt in ground_truth
            )
            if is_localized_hit and first_localized_rank is None:
                first_localized_rank = p.rank

            if self.scorer is not None:
                if hasattr(self.scorer, "score") and callable(getattr(self.scorer, "score")):
                    raw_rscore = self.scorer.score(p, ground_truth)
                else:
                    raw_rscore = self.scorer(p, ground_truth)
                if (
                    isinstance(raw_rscore, bool)
                    or not isinstance(raw_rscore, (int, float))
                    or not math.isfinite(raw_rscore)
                    or raw_rscore < 0.0
                    or raw_rscore > 1.0
                ):
                    raise ValueError(
                        f"Scorer returned invalid RScore {raw_rscore!r} for query '{query_id}', "
                        f"rank {p.rank}; must be a finite float in [0.0, 1.0]"
                    )
                rscore = float(raw_rscore)
            else:
                rscore = 1.0 if is_localized_hit else 0.0

            per_pred_rscore.append(rscore)

            # Temporal segment IoU (diagnostic)
            if not is_video_match:
                if p.pred_segment_start_s is not None and p.pred_segment_end_s is not None:
                    per_pred_best_iou.append(0.0)
                else:
                    per_pred_best_iou.append(None)
            else:
                matching_gts = [gt for gt in ground_truth if gt.video_id == p.video_id]
                candidate_ious: list[float] = []
                for gt in matching_gts:
                    iou = self._compute_segment_iou(
                        p.pred_segment_start_s,
                        p.pred_segment_end_s,
                        gt.start_s,
                        gt.end_s,
                    )
                    if iou is not None:
                        candidate_ious.append(iou)
                if candidate_ious:
                    per_pred_best_iou.append(max(candidate_ious))
                else:
                    per_pred_best_iou.append(None)

        # Compute R@k and Hit@k across cutoffs
        r_at_k: dict[int, float] = {}
        localized_hit_at_k: dict[int, bool] = {}
        video_hit_at_k: dict[int, bool] = {}
        best_temporal_iou_at_k: dict[int, float | None] = {}

        for k in self.cutoffs:
            slice_len = min(k, n_preds)
            if slice_len == 0:
                r_at_k[k] = 0.0
                localized_hit_at_k[k] = False
                video_hit_at_k[k] = False
                best_temporal_iou_at_k[k] = None
            else:
                r_at_k[k] = max(per_pred_rscore[:slice_len])
                localized_hit_at_k[k] = (
                    first_localized_rank is not None and first_localized_rank <= k
                )
                video_hit_at_k[k] = (
                    first_video_rank is not None and first_video_rank <= k
                )

                ious_up_to_k = [v for v in per_pred_best_iou[:slice_len] if v is not None]
                best_temporal_iou_at_k[k] = max(ious_up_to_k) if ious_up_to_k else None

        # Official final score = Mean of Top-k R-Scores across evaluation cutoffs
        official_final_score = (
            sum(r_at_k[k] for k in self.cutoffs) / len(self.cutoffs)
            if self.cutoffs
            else 0.0
        )

        video_mrr = 1.0 / first_video_rank if first_video_rank is not None else 0.0
        localized_mrr = 1.0 / first_localized_rank if first_localized_rank is not None else 0.0
        # point_in_gt_interval strictly reflects if top-1 prediction is localized hit
        point_in_gt_interval = localized_hit_at_k.get(1, False)
        temporal_iou_at_1 = best_temporal_iou_at_k.get(1, None)

        return QueryEvaluationResult(
            query_id=query_id,
            scoring_protocol_id=self.scoring_protocol_id,
            official_final_score=official_final_score,
            official_first_hit_rank=first_localized_rank,
            video_first_hit_rank=first_video_rank,
            video_mrr=video_mrr,
            localized_mrr=localized_mrr,
            r_at_k=r_at_k,
            localized_hit_at_k=localized_hit_at_k,
            video_hit_at_k=video_hit_at_k,
            point_in_gt_interval=point_in_gt_interval,
            temporal_iou_at_1=temporal_iou_at_1,
            best_temporal_iou_at_k=best_temporal_iou_at_k,
            prediction_count=n_preds,
            temporal_cross_check_status=cross_check_status,
        )

    def evaluate_predictions(
        self,
        predictions: Mapping[str, Sequence[PredictionRecord]],
        ground_truth: Mapping[str, Sequence[GroundTruthInterval]],
    ) -> EvaluationReport:
        """Evaluate a full set of query predictions against ground truth mappings."""
        if not ground_truth:
            raise ValueError("Ground truth mapping cannot be empty")

        # Validate ground truth intervals strictly first
        for qid, gt_intervals in ground_truth.items():
            if type(qid) is not str or not qid.strip():
                raise ValueError(f"Ground truth query ID must be a non-empty string, got {qid!r}")
            if not gt_intervals:
                raise ValueError(f"Ground truth intervals for query '{qid}' cannot be empty")
            for gt in gt_intervals:
                if gt.query_id is not None and gt.query_id != qid:
                    raise ValueError(
                        f"Ground truth query ID mismatch: mapping key '{qid}' vs "
                        f"interval query_id '{gt.query_id}'"
                    )

        # Check for unmapped queries in predictions (fail-closed)
        gt_query_keys = set(ground_truth.keys())
        for pred_qid in predictions.keys():
            if pred_qid not in gt_query_keys:
                raise ValueError(
                    f"Unmapped query '{pred_qid}' found in predictions but not in ground truth"
                )

        query_results: dict[str, QueryEvaluationResult] = {}
        missing_count = 0

        for qid, gt_intervals in ground_truth.items():
            preds_for_q = predictions.get(qid, [])
            if not preds_for_q:
                missing_count += 1
                r_at_k = {k: 0.0 for k in self.cutoffs}
                loc_hit = {k: False for k in self.cutoffs}
                vid_hit = {k: False for k in self.cutoffs}
                best_iou = {k: None for k in self.cutoffs}
                query_results[qid] = QueryEvaluationResult(
                    query_id=qid,
                    scoring_protocol_id=self.scoring_protocol_id,
                    official_final_score=0.0,
                    official_first_hit_rank=None,
                    video_first_hit_rank=None,
                    video_mrr=0.0,
                    localized_mrr=0.0,
                    r_at_k=r_at_k,
                    localized_hit_at_k=loc_hit,
                    video_hit_at_k=vid_hit,
                    point_in_gt_interval=False,
                    temporal_iou_at_1=None,
                    best_temporal_iou_at_k=best_iou,
                    prediction_count=0,
                    temporal_cross_check_status="unavailable",
                )
            else:
                query_results[qid] = self.evaluate_query(qid, preds_for_q, gt_intervals)

        total_queries = len(ground_truth)
        mean_final_score = (
            sum(r.official_final_score for r in query_results.values()) / total_queries
            if total_queries > 0
            else 0.0
        )
        mean_vid_mrr = (
            sum(r.video_mrr for r in query_results.values()) / total_queries
            if total_queries > 0
            else 0.0
        )
        mean_loc_mrr = (
            sum(r.localized_mrr for r in query_results.values()) / total_queries
            if total_queries > 0
            else 0.0
        )

        mean_video_hit_at_k: dict[int, float] = {}
        mean_localized_hit_at_k: dict[int, float] = {}
        for k in self.cutoffs:
            mean_video_hit_at_k[k] = (
                sum(
                    1.0 if r.video_hit_at_k.get(k, False) else 0.0
                    for r in query_results.values()
                )
                / total_queries
                if total_queries > 0
                else 0.0
            )
            mean_localized_hit_at_k[k] = (
                sum(
                    1.0 if r.localized_hit_at_k.get(k, False) else 0.0
                    for r in query_results.values()
                )
                / total_queries
                if total_queries > 0
                else 0.0
            )

        return EvaluationReport(
            scoring_protocol_id=self.scoring_protocol_id,
            query_results=query_results,
            mean_official_final_score=mean_final_score,
            query_count=total_queries,
            missing_query_count=missing_count,
            mean_video_mrr=mean_vid_mrr,
            mean_localized_mrr=mean_loc_mrr,
            mean_video_hit_at_k=mean_video_hit_at_k,
            mean_localized_hit_at_k=mean_localized_hit_at_k,
        )

    def evaluate(
        self,
        checkpoint_path: Path | str,
        ground_truth_path: Path | str,
    ) -> EvaluationReport:
        """Evaluate file-based predictions and ground truth fixtures."""
        pred_p = Path(checkpoint_path)
        gt_p = Path(ground_truth_path)

        if not pred_p.is_file():
            raise FileNotFoundError(f"Predictions checkpoint file not found: {pred_p}")
        if not gt_p.is_file():
            raise FileNotFoundError(f"Ground truth fixture file not found: {gt_p}")

        predictions = self._load_predictions_from_file(pred_p)
        ground_truth = self._load_ground_truth_from_file(gt_p)

        return self.evaluate_predictions(predictions, ground_truth)

    def _load_predictions_from_file(
        self,
        file_path: Path,
    ) -> dict[str, list[PredictionRecord]]:
        """Load predictions from JSON or JSONL file."""
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            return {}

        results: dict[str, list[PredictionRecord]] = {}

        if file_path.suffix == ".jsonl":
            for line_no, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in line {line_no} of {file_path}: {exc}"
                    ) from exc
                rec = self._dict_to_prediction_record(item)
                results.setdefault(rec.query_id, []).append(rec)
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {file_path}: {exc}") from exc

            if isinstance(data, list):
                for item in data:
                    rec = self._dict_to_prediction_record(item)
                    results.setdefault(rec.query_id, []).append(rec)
            elif isinstance(data, dict):
                envelope_qid = data.get("query_id")
                if envelope_qid is not None:
                    if type(envelope_qid) is not str or not envelope_qid.strip():
                        raise ValueError(
                            f"Envelope query_id must be a non-empty string, got {envelope_qid!r}"
                        )
                    envelope_qid = envelope_qid.strip()

                if "records" in data:
                    if not isinstance(data["records"], list):
                        records_type = type(data["records"]).__name__
                        raise ValueError(f"JSON root 'records' must be a list, got {records_type}")
                    for item in data["records"]:
                        rec = self._dict_to_prediction_record(
                            item,
                            default_query_id=envelope_qid,
                            envelope_query_id=envelope_qid,
                        )
                        results.setdefault(rec.query_id, []).append(rec)
                elif "queries" in data:
                    if not isinstance(data["queries"], dict):
                        queries_type = type(data["queries"]).__name__
                        raise ValueError(f"JSON root 'queries' must be a dict, got {queries_type}")
                    for qid, items in data["queries"].items():
                        if type(qid) is not str or not qid.strip():
                            raise ValueError(
                                f"Query ID in 'queries' must be a non-empty string, got {qid!r}"
                            )
                        if envelope_qid is not None and qid != envelope_qid:
                            raise ValueError(
                                f"Envelope query_id mismatch: root envelope has query_id "
                                f"'{envelope_qid}' but 'queries' contains query '{qid}'"
                            )
                        if not isinstance(items, list):
                            raise ValueError(
                                f"Items for query '{qid}' must be a list, got "
                                f"{type(items).__name__}"
                            )
                        for item in items:
                            rec = self._dict_to_prediction_record(
                                item,
                                default_query_id=qid,
                                envelope_query_id=envelope_qid or qid,
                            )
                            results.setdefault(rec.query_id, []).append(rec)
                else:
                    for qid, items in data.items():
                        if qid == "query_id":
                            continue
                        if type(qid) is not str or not qid.strip():
                            raise ValueError(
                                f"Query ID must be a non-empty string, got {qid!r}"
                            )
                        if envelope_qid is not None and qid != envelope_qid:
                            raise ValueError(
                                f"Envelope query_id mismatch: root envelope has query_id "
                                f"'{envelope_qid}' but mapping key is '{qid}'"
                            )
                        if not isinstance(items, list):
                            raise ValueError(
                                f"Predictions for query '{qid}' must be a list, got "
                                f"{type(items).__name__}"
                            )
                        for item in items:
                            rec = self._dict_to_prediction_record(
                                item,
                                default_query_id=qid,
                                envelope_query_id=envelope_qid or qid,
                            )
                            results.setdefault(rec.query_id, []).append(rec)
            else:
                raise ValueError(
                    f"Root JSON element must be a list or dict, got {type(data).__name__}"
                )

        # Ensure predictions are sorted by rank within each query
        for qid in results:
            results[qid].sort(key=lambda r: r.rank)

        return results

    def _dict_to_prediction_record(
        self,
        item: Any,
        default_query_id: str | None = None,
        envelope_query_id: str | None = None,
    ) -> PredictionRecord:
        """Convert a raw dictionary to a PredictionRecord."""
        if not isinstance(item, dict):
            raise ValueError(
                f"Prediction item must be a dictionary, got {type(item).__name__}: {item!r}"
            )

        raw_qid = item.get("query_id")
        target_qid = envelope_query_id or default_query_id

        if raw_qid is not None:
            if type(raw_qid) is not str or not raw_qid.strip():
                raise ValueError(f"query_id must be a non-empty string, got {raw_qid!r}")
            raw_qid = raw_qid.strip()
            if target_qid is not None and raw_qid != target_qid:
                raise ValueError(
                    f"Query ID mismatch in prediction: expected '{target_qid}' "
                    f"(from envelope/mapping) but item specifies query_id '{raw_qid}'"
                )
            query_id = raw_qid
        else:
            if target_qid is None:
                raise ValueError(f"Missing query_id in prediction item: {item}")
            query_id = target_qid

        raw_vid = item.get("video_id")
        if type(raw_vid) is not str or not raw_vid.strip():
            raise ValueError(
                f"video_id must be a non-empty string, got {raw_vid!r}"
            )
        video_id = raw_vid.strip()

        raw_fid = _extract_aliased_int(
            item,
            ["frame_id", "actual_frame_id"],
            "frame_id",
            required=True,
            non_negative=True,
        )
        assert raw_fid is not None

        raw_rank = _extract_aliased_int(
            item,
            ["rank"],
            "rank",
            required=True,
            non_negative=False,
            min_value=1,
        )
        assert raw_rank is not None

        score = _extract_aliased_float(
            item,
            ["score", "fusion_score", "confidence_score"],
            "score",
            required=False,
        )

        timestamp_s = _extract_aliased_float(
            item,
            ["timestamp_s", "timestamp_seconds", "pts_time"],
            "timestamp",
            required=False,
        )

        keyframe_id = _extract_aliased_int(
            item,
            ["keyframe_id", "keyframe_order_diagnostic"],
            "keyframe_id",
            required=False,
            non_negative=True,
        )

        seg_start = item.get("pred_segment_start_s")
        seg_end = item.get("pred_segment_end_s")
        if (seg_start is None) != (seg_end is None):
            raise ValueError(
                "pred_segment_start_s and pred_segment_end_s must both be provided or both be None"
            )
        if seg_start is not None:
            if (
                not isinstance(seg_start, (int, float))
                or isinstance(seg_start, bool)
                or not math.isfinite(seg_start)
            ):
                raise ValueError(
                    f"pred_segment_start_s must be a finite number, got {seg_start!r}"
                )
            if (
                not isinstance(seg_end, (int, float))
                or isinstance(seg_end, bool)
                or not math.isfinite(seg_end)
            ):
                raise ValueError(
                    f"pred_segment_end_s must be a finite number, got {seg_end!r}"
                )
            pred_segment_start_s: float | None = float(seg_start)
            pred_segment_end_s: float | None = float(seg_end)
            if pred_segment_start_s > pred_segment_end_s:
                raise ValueError(
                    f"pred_segment_start_s ({pred_segment_start_s}) cannot exceed "
                    f"pred_segment_end_s ({pred_segment_end_s})"
                )
        else:
            pred_segment_start_s = None
            pred_segment_end_s = None

        return PredictionRecord(
            query_id=query_id,
            video_id=video_id,
            frame_id=raw_fid,
            rank=raw_rank,
            score=score,
            timestamp_s=timestamp_s,
            keyframe_id=keyframe_id,
            pred_segment_start_s=pred_segment_start_s,
            pred_segment_end_s=pred_segment_end_s,
        )

    def _load_ground_truth_from_file(
        self,
        file_path: Path,
    ) -> dict[str, list[GroundTruthInterval]]:
        """Load ground truth intervals from JSON or JSONL file."""
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            return {}

        results: dict[str, list[GroundTruthInterval]] = {}

        if file_path.suffix == ".jsonl":
            for line_no, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in line {line_no} of {file_path}: {exc}"
                    ) from exc
                gt = self._dict_to_ground_truth_interval(item)
                qid = gt.query_id or item.get("query_id")
                if not qid:
                    raise ValueError(f"Missing query_id in ground truth item: {item}")
                results.setdefault(qid, []).append(gt)
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {file_path}: {exc}") from exc

            if isinstance(data, list):
                for item in data:
                    gt = self._dict_to_ground_truth_interval(item)
                    qid = gt.query_id or item.get("query_id")
                    if not qid:
                        raise ValueError(f"Missing query_id in ground truth item: {item}")
                    results.setdefault(qid, []).append(gt)
            elif isinstance(data, dict):
                envelope_qid = data.get("query_id")
                if envelope_qid is not None:
                    if type(envelope_qid) is not str or not envelope_qid.strip():
                        raise ValueError(
                            f"Envelope query_id must be a non-empty string, got {envelope_qid!r}"
                        )
                    envelope_qid = envelope_qid.strip()

                for qid, val in data.items():
                    if qid == "query_id":
                        continue
                    if type(qid) is not str or not qid.strip():
                        raise ValueError(
                            f"Ground truth query ID must be a non-empty string, got {qid!r}"
                        )
                    if envelope_qid is not None and qid != envelope_qid:
                        raise ValueError(
                            f"Envelope query_id mismatch: root envelope has query_id "
                            f"'{envelope_qid}' but mapping key is '{qid}'"
                        )
                    if isinstance(val, list):
                        if not val:
                            raise ValueError(
                                f"Ground truth interval list for query '{qid}' cannot be empty"
                            )
                        for item in val:
                            gt = self._dict_to_ground_truth_interval(
                                item,
                                default_query_id=qid,
                                envelope_query_id=envelope_qid or qid,
                            )
                            results.setdefault(qid, []).append(gt)
                    elif isinstance(val, dict):
                        gt = self._dict_to_ground_truth_interval(
                            val,
                            default_query_id=qid,
                            envelope_query_id=envelope_qid or qid,
                        )
                        results.setdefault(qid, []).append(gt)
                    else:
                        raise ValueError(
                            f"Ground truth value for query '{qid}' must be a list or dict, "
                            f"got {type(val).__name__}"
                        )
            else:
                raise ValueError(
                    f"Root Ground Truth JSON must be list or dict, got {type(data).__name__}"
                )

        return results

    def _dict_to_ground_truth_interval(
        self,
        item: Any,
        default_query_id: str | None = None,
        envelope_query_id: str | None = None,
    ) -> GroundTruthInterval:
        """Convert a raw dictionary to a GroundTruthInterval."""
        if not isinstance(item, dict):
            raise ValueError(
                f"Ground truth item must be a dictionary, got {type(item).__name__}: {item!r}"
            )

        raw_vid = item.get("video_id")
        if type(raw_vid) is not str or not raw_vid.strip():
            raise ValueError(
                f"video_id in ground truth must be a non-empty string, got {raw_vid!r}"
            )
        video_id = raw_vid.strip()

        start_frame = _extract_aliased_int(
            item,
            ["start_frame", "target_frame", "frame_id"],
            "start_frame",
            required=True,
            non_negative=True,
        )
        assert start_frame is not None

        if "end_frame" in item and item["end_frame"] is not None:
            end_frame = _extract_aliased_int(
                item,
                ["end_frame"],
                "end_frame",
                required=True,
                non_negative=True,
            )
            assert end_frame is not None
            if end_frame < start_frame:
                raise ValueError(
                    f"end_frame ({end_frame}) cannot be less than start_frame ({start_frame})"
                )
        else:
            end_frame = start_frame

        target_qid = envelope_query_id or default_query_id
        raw_qid = item.get("query_id")
        if raw_qid is not None:
            if type(raw_qid) is not str or not raw_qid.strip():
                raise ValueError(
                    f"query_id in ground truth must be a non-empty string, got {raw_qid!r}"
                )
            raw_qid = raw_qid.strip()
            if target_qid is not None and raw_qid != target_qid:
                raise ValueError(
                    f"Ground truth query ID mismatch: expected '{target_qid}' "
                    f"(from envelope/mapping) vs interval query_id '{raw_qid}'"
                )
            gt_query_id = raw_qid
        else:
            gt_query_id = target_qid

        raw_start_s = item.get("start_s")
        raw_end_s = item.get("end_s")
        if (raw_start_s is None) != (raw_end_s is None):
            raise ValueError(
                "Ground truth start_s and end_s must both be provided or both be None"
            )
        if raw_start_s is not None:
            if (
                not isinstance(raw_start_s, (int, float))
                or isinstance(raw_start_s, bool)
                or not math.isfinite(raw_start_s)
            ):
                raise ValueError(
                    f"start_s must be a finite number, got {raw_start_s!r}"
                )
            if (
                not isinstance(raw_end_s, (int, float))
                or isinstance(raw_end_s, bool)
                or not math.isfinite(raw_end_s)
            ):
                raise ValueError(
                    f"end_s must be a finite number, got {raw_end_s!r}"
                )
            start_s: float | None = float(raw_start_s)
            end_s: float | None = float(raw_end_s)
            if start_s > end_s:
                raise ValueError(
                    f"start_s ({start_s}) cannot exceed end_s ({end_s})"
                )
        else:
            start_s = None
            end_s = None

        keyframe_id = _extract_aliased_int(
            item,
            ["keyframe_id", "keyframe_order_diagnostic"],
            "keyframe_id",
            required=False,
            non_negative=True,
        )

        return GroundTruthInterval(
            video_id=video_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_s=start_s,
            end_s=end_s,
            keyframe_id=keyframe_id,
            query_id=gt_query_id,
        )
