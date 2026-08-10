"""Generate deterministic JSON/CSV/Markdown L21-150 error analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = SYSTEM_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from system_tai.quality.l21_150_error_analysis import analyze_l21_150_errors  # noqa: E402


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_failures(path: Path | None) -> list[dict]:
    if path is None:
        return []
    failures: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if type(payload) is not dict:
            raise ValueError(f"failure line {line_number} must be an object")
        failures.append(payload)
    return failures


def _write_csv(report: dict, path: Path) -> None:
    fields = (
        "query_id",
        "task",
        "branch",
        "difficulty",
        "video_id",
        "split",
        "categories",
        "failure_reason",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in report["query_errors"]:
            output = dict(row)
            output["categories"] = "|".join(row["categories"])
            writer.writerow(output)


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# L21-150 Mechanical Error Analysis",
        "",
        "This report classifies observed outcomes; it does not infer root cause "
        "from branch labels.",
        "",
        f"- Queries analyzed: {report['query_count']}",
        "",
        "## Category counts",
        "",
    ]
    if report["category_counts"]:
        lines.extend(
            f"- {category}: {count}"
            for category, count in report["category_counts"].items()
        )
    else:
        lines.append("- No mechanical error category was triggered.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--failures-jsonl", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evaluation = _load_json(args.evaluation_report)
        report = analyze_l21_150_errors(
            evaluation,
            failures=_load_failures(args.failures_jsonl),
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_csv(report, args.output_csv)
        _write_markdown(report, args.output_markdown)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"L21-150 error analysis failed: {exc}", file=sys.stderr)
        return 2
    print(
        "L21-150 error analysis complete: "
        f"queries={report['query_count']} categories={report['category_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
