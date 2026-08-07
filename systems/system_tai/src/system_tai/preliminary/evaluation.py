from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .validation import validate_ranked_top100


@dataclass(frozen=True, slots=True)
class QueryEvaluationReport:
    query_id: str
    task_type: str
    prediction_count: int
    per_rank_r_scores: tuple[float, ...]
    r_at_1: float
    r_at_5: float
    r_at_20: float
    r_at_50: float
    r_at_100: float
    final_score: float


OFFICIAL_K = (1, 5, 20, 50, 100)


def evaluate_ranked_query(
    query_id: str,
    task_type: str,
    predictions: list[Any],
    ground_truth: Any,
    scorer: Callable[[Any, Any], float],
) -> QueryEvaluationReport:
    errors = validate_ranked_top100(predictions, task_type, ground_truth)
    if errors:
        msg = "; ".join(e.message for e in errors)
        raise ValueError(f"Validation failed for query {query_id}: {msg}")

    # Sort by rank
    preds = sorted(predictions, key=lambda p: p.rank)

    scores = []
    for p in preds:
        score = scorer(p, ground_truth)
        scores.append(score)

    r_scores = {}
    for k in OFFICIAL_K:
        # Get scores for predictions with rank <= k
        valid_scores = [scores[i] for i, p in enumerate(preds) if p.rank <= k]
        r_scores[k] = max(valid_scores) if valid_scores else 0.0

    final_score = sum(r_scores.values()) / len(OFFICIAL_K)

    return QueryEvaluationReport(
        query_id=query_id,
        task_type=task_type,
        prediction_count=len(preds),
        per_rank_r_scores=tuple(scores),
        r_at_1=r_scores[1],
        r_at_5=r_scores[5],
        r_at_20=r_scores[20],
        r_at_50=r_scores[50],
        r_at_100=r_scores[100],
        final_score=final_score,
    )


@dataclass(frozen=True, slots=True)
class DatasetEvaluationReport:
    query_count: int
    mean_query_final_score: float
    query_reports: tuple[QueryEvaluationReport, ...]


def evaluate_dataset(query_reports: list[QueryEvaluationReport]) -> DatasetEvaluationReport:
    if not query_reports:
        return DatasetEvaluationReport(0, 0.0, ())
    mean_score = sum(r.final_score for r in query_reports) / len(query_reports)
    return DatasetEvaluationReport(
        query_count=len(query_reports),
        mean_query_final_score=mean_score,
        query_reports=tuple(query_reports),
    )
