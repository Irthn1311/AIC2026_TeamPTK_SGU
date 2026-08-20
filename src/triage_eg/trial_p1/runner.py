"""GT-free execution and packaging for the immediate Trial P1 B0_SAFE run."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from aic2026_eval.io import write_json, write_jsonl
from aic2026_eval.validation import validate_predictions
from triage_eg.submission import create_submission_zip, validate_submission_zip


def _fuse_kis(variant_rows: list[list[dict[str, Any]]], *, rrf_k: int = 60) -> list[dict[str, Any]]:
    scores: defaultdict[tuple[str, int], float] = defaultdict(float)
    source: dict[tuple[str, int], dict[str, Any]] = {}
    first_seen: dict[tuple[str, int], tuple[int, int]] = {}
    for variant_index, rows in enumerate(variant_rows):
        for row in rows:
            key = str(row["video_id"]), int(row["frame_id"])
            rank = int(row["rank"])
            scores[key] += 1.0 / (rrf_k + rank)
            source.setdefault(key, row)
            first_seen.setdefault(key, (variant_index, rank))
    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key], key))[:100]
    return [
        {
            **source[key],
            "rank": rank,
            "trial_variant_rrf_score": scores[key],
            "trial_fusion": "EQUAL_RRF60_QUERY_VARIANTS",
        }
        for rank, key in enumerate(ordered, 1)
    ]


def run_b0_safe(
    pipeline: Any,
    compiled: list[dict[str, Any]],
    output_root: str | Path,
    submission_zip: str | Path,
) -> dict[str, Any]:
    """Run current B0 only; no GT, repaired modality, or query-specific behavior."""

    root = Path(output_root).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    queries = [plan["team_query"] for plan in compiled]
    predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for plan in compiled:
        query = plan["team_query"]
        if query["task"] == "KIS" and len(plan["retrieval_variants"]) > 1:
            variant_rows: list[list[dict[str, Any]]] = []
            for variant_index, variant in enumerate(plan["retrieval_variants"]):
                variant_query = {**query, "query": variant}
                result = pipeline.predict_query(variant_query, "P0_COARSE")
                variant_rows.append(list(result.predictions))
                diagnostics.append(
                    {
                        "query_id": query["query_id"],
                        "variant_index": variant_index,
                        "variant": variant,
                        "candidate_count": len(result.predictions),
                    }
                )
            predictions.extend(_fuse_kis(variant_rows))
        else:
            result = pipeline.predict_query(query, "P0_COARSE")
            predictions.extend(result.predictions)
            diagnostics.extend(result.diagnostics)
    validation, issues = validate_predictions(queries, predictions)
    if validation["status"] != "PASS":
        raise RuntimeError(f"TRIAL_B0_PREDICTION_VALIDATION_FAILED: {issues}")
    write_jsonl(root / "trial_p1_query_plans.jsonl", compiled)
    write_jsonl(root / "trial_p1_B0_SAFE_predictions.jsonl", predictions)
    write_jsonl(root / "trial_p1_B0_SAFE_diagnostics.jsonl", diagnostics)
    output = create_submission_zip(queries, predictions, Path(submission_zip))
    zip_validation = validate_submission_zip(output, queries)
    write_json(
        root / "trial_p1_B0_SAFE_validation.json",
        {"prediction_validation": validation, "submission_validation": zip_validation},
    )
    return {
        "mode": "B0_SAFE",
        "query_count": len(queries),
        "prediction_validation": validation,
        "submission_validation": zip_validation,
        "submission_zip": str(output),
        "gt_opened": False,
    }


__all__ = ["run_b0_safe"]
