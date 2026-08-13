"""CLI for authoritative deterministic TEAM-EVAL scoring and scoreboard reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_jsonl, sha256_file, write_jsonl
from .report import write_evaluation_reports
from .scoring import evaluate
from .validation import validate_predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--git-commit")
    parser.add_argument("--wall-clock-json")
    args = parser.parse_args(argv)
    queries = read_jsonl(args.queries)
    predictions = read_jsonl(args.predictions)
    inventory = read_jsonl(args.inventory) if args.inventory else None
    validation_summary, validation_issues = validate_predictions(
        queries,
        predictions,
        inventory=inventory,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if validation_issues:
        write_jsonl(args.output / "issues.jsonl", validation_issues)
        print(json.dumps(validation_summary, ensure_ascii=False, indent=2))
        return 1
    wall_clock = json.loads(args.wall_clock_json) if args.wall_clock_json else None
    summary, by_query, by_slice, issues = evaluate(
        queries,
        predictions,
        read_jsonl(args.ground_truth),
        inventory=inventory,
        metadata={
            "system_id": args.system_id,
            "git_commit": args.git_commit,
            "config_id": args.config_id,
            "dataset_version": args.dataset_version,
            "prediction_sha256": sha256_file(args.predictions),
            "wall_clock_metadata": wall_clock,
        },
    )
    write_evaluation_reports(args.output, summary, by_query, by_slice, issues)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
