"""Run a bounded, read-only survey of an attached dataset layout."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from triage_eg.data.dataset_survey import (
    DEFAULT_OUTPUT_ROOT,
    SurveyLimits,
    resolve_dataset_root,
    summary_json,
    survey_dataset,
    write_survey_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", help="CLI value overrides AIC_DATA_ROOT and Kaggle default"
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-examples-per-group", type=int, default=5)
    parser.add_argument("--max-listed-per-directory", type=int, default=20)
    parser.add_argument("--max-stat-operations", type=int, default=5_000)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--strict-root", action="store_true")
    parser.add_argument("--json-output", default="dataset_survey.json")
    parser.add_argument("--text-output", default="dataset_survey.md")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        limits = SurveyLimits(
            max_depth=args.max_depth,
            max_listed_per_directory=args.max_listed_per_directory,
            max_examples_per_group=args.max_examples_per_group,
            max_stat_operations=args.max_stat_operations,
        )
        result = survey_dataset(
            resolve_dataset_root(args.dataset_root),
            limits=limits,
            strict_root=args.strict_root,
            seed=args.seed,
        )
        if args.no_write:
            print(summary_json(result))
            return 0
        paths = write_survey_outputs(
            result,
            Path(args.output_root),
            json_output=args.json_output,
            text_output=args.text_output,
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print("This is a bounded layout survey, not a complete Data Audit.")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
