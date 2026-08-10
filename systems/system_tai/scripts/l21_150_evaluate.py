"""Evaluate L21-150 runtime predictions with explicit proposed/validated GT policy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = SYSTEM_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from system_tai.quality.l21_150_evaluator import evaluate_l21_150  # noqa: E402
from system_tai.quality.l21_150_schema import (  # noqa: E402
    L21150FormatError,
    load_l21_150_benchmark,
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSONL line {line_number}: {exc}") from exc
        if type(payload) is not dict:
            raise ValueError(f"JSONL line {line_number} must be an object")
        records.append(payload)
    return records


def _write_csv(report: dict[str, Any], path: Path) -> None:
    fields = (
        "query_id",
        "task",
        "split",
        "branch",
        "difficulty",
        "prediction_count",
        "result_valid",
        "first_relevant_rank",
        "reciprocal_rank",
        "final_score",
        "video_hit",
        "frame_hit",
        "full_hit",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for query_report in report["query_reports"]:
            writer.writerow({field: query_report.get(field) for field in fields})


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    overall = report["overall"]
    lines = [
        "# L21-150 Diagnostic Evaluation",
        "",
        f"- GT evidence mode: `{report['gt_evidence_mode']}`",
        "- Official BTC ground truth: `false`",
        "- Semantic/competition accuracy claim: `false`",
        f"- Scored queries: {report['selected_query_count']}",
        f"- Excluded unvalidated queries: {report['excluded_unvalidated_query_count']}",
        f"- Final Score: {overall['final_score']:.6f}",
        f"- Valid results: {overall['valid_result_count']}",
        f"- Invalid results: {overall['invalid_result_count']}",
        "",
        "## Recall summary",
        "",
    ]
    lines.extend(
        f"- R@{cutoff}: {overall['r_at_k'][str(cutoff)]:.6f}"
        for cutoff in (1, 5, 20, 50, 100)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt-policy", choices=("proposed", "validated-only"), required=True)
    parser.add_argument("--mapping-validation", type=Path)
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--task", choices=("all", "kis", "qa", "trake"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        benchmark = load_l21_150_benchmark(args.benchmark)
        predictions = load_jsonl(args.predictions)
        mapping_report = None
        if args.mapping_validation is not None:
            mapping_report = json.loads(args.mapping_validation.read_text(encoding="utf-8"))
        report = evaluate_l21_150(
            benchmark,
            predictions,
            gt_policy=args.gt_policy,
            mapping_validation_report=mapping_report,
            split=args.split,
            task=args.task,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if args.csv_output is not None:
            _write_csv(report, args.csv_output)
        if args.markdown_output is not None:
            _write_markdown(report, args.markdown_output)
    except (FileNotFoundError, L21150FormatError, OSError, ValueError) as exc:
        print(f"L21-150 evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(
        "L21-150 evaluation complete: "
        f"queries={report['selected_query_count']} "
        f"final_score={report['overall']['final_score']:.6f} "
        f"gt_mode={report['gt_evidence_mode']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
