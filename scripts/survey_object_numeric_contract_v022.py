#!/usr/bin/env python3
"""Run the final bounded Object numeric-string contract patch v0.2.2."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from triage_eg.data.object_numeric_contract import (
    DEFAULT_OUTPUT_ROOT,
    NumericLimits,
    resolve_dataset_root,
    result_json,
    run_survey,
    write_outputs,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--v021-summary", type=Path)
    result.add_argument("--max-object-json-total", type=int, default=15)
    result.add_argument("--max-object-json-bytes", type=int, default=1_048_576)
    result.add_argument("--strict-root", action="store_true")
    result.add_argument("--no-write", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_survey(
        resolve_dataset_root(args.dataset_root),
        limits=NumericLimits(args.max_object_json_total, args.max_object_json_bytes),
        strict_root=args.strict_root,
        v021_summary=args.v021_summary,
    )
    if not args.no_write:
        paths = write_outputs(result, args.output_root)
        logging.info("ZIP ready: %s", paths["zip"])
    print(result_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
