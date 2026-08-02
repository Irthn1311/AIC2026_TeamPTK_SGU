"""Evaluate ranked KIS, Q&A, or TRAKE predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from triage_eg.evaluation.official_metrics import (
    best_rscore_at_k,
    final_score,
    kis_rscore,
    qa_rscore,
    trake_rscore,
)

_SCORERS = {"kis": kis_rscore, "qa": qa_rscore, "trake": trake_rscore}
_KS = (1, 5, 20, 50, 100)


def _load_json_list(path: str) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {input_path}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Expected a list of objects in {input_path}")
    return payload


def evaluate_ranked(
    task: str,
    ground_truths: list[dict[str, Any]],
    prediction_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate each query group and average its final score."""

    scorer = _SCORERS[task]
    predictions_by_id = {item["query_id"]: item["predictions"] for item in prediction_groups}
    query_reports: list[dict[str, Any]] = []
    for ground_truth in ground_truths:
        query_id = ground_truth["query_id"]
        predictions = predictions_by_id.get(query_id, [])
        scores = [scorer(prediction, ground_truth) for prediction in predictions]
        at_k = {k: best_rscore_at_k(scores, k) for k in _KS}
        query_reports.append(
            {"query_id": query_id, "rscore_at_k": at_k, "final": final_score(at_k)}
        )
    overall = (
        sum(item["final"] for item in query_reports) / len(query_reports) if query_reports else 0.0
    )
    return {"task": task, "queries": query_reports, "final_score": overall}


def main() -> int:
    """Parse files, evaluate, and print a JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(_SCORERS))
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    try:
        report = evaluate_ranked(
            args.task,
            _load_json_list(args.ground_truth),
            _load_json_list(args.predictions),
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
