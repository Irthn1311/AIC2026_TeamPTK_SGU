#!/usr/bin/env python3
"""Run the bounded TRIAGE-EG cross-asset patch v0.2.1."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from triage_eg.data.cross_asset_patch_v021 import (
    DEFAULT_OUTPUT_ROOT,
    PatchLimits,
    resolve_dataset_root,
    result_json,
    run_patch,
    write_outputs,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--v02-summary", type=Path)
    result.add_argument("--max-object-json-total", type=int, default=15)
    result.add_argument("--max-object-json-bytes", type=int, default=1_048_576)
    result.add_argument("--max-boxes-per-file", type=int, default=20)
    result.add_argument("--max-boxes-total", type=int, default=100)
    result.add_argument("--strict-root", action="store_true")
    result.add_argument("--no-write", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    limits = PatchLimits(
        max_object_json_total=args.max_object_json_total,
        max_object_json_bytes=args.max_object_json_bytes,
        max_boxes_per_file=args.max_boxes_per_file,
        max_boxes_total=args.max_boxes_total,
    )
    result = run_patch(
        resolve_dataset_root(args.dataset_root),
        limits=limits,
        v02_summary=args.v02_summary,
        strict_root=args.strict_root,
    )
    if not args.no_write:
        paths = write_outputs(result, args.output_root)
        logging.info("ZIP ready: %s", paths["zip"])
    print(result_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
