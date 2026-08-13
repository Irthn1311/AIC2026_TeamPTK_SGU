"""CLI entry point for strict shared final-prediction validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_jsonl, write_jsonl
from .validation import validate_predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--issues", type=Path)
    args = parser.parse_args(argv)
    inventory = read_jsonl(args.inventory) if args.inventory else None
    summary, issues = validate_predictions(
        read_jsonl(args.queries),
        read_jsonl(args.predictions),
        inventory=inventory,
    )
    if args.issues:
        write_jsonl(args.issues, issues)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for issue in issues:
        print(json.dumps(issue, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
