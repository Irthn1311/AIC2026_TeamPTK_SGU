#!/usr/bin/env python3
"""Patch frozen Stage 1D v0.1.0 artifacts with blinded review visuals only."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from triage_eg.retrieval.stage1d import (
    create_stage1d_bundle,
    patch_blinded_review_visuals,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1d-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        result = patch_blinded_review_visuals(
            args.stage1d_root,
            args.dataset_root,
            output_root=args.output_root,
        )
        if args.zip_path:
            create_stage1d_bundle(result["output_root"], args.zip_path)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        logging.error("%s", error)
        return 2
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    print("BLINDED_REVIEW_SHEETS =", result["blinded_sheet_count"])
    print("REVIEW_ROWS =", result["review_rows"])
    print("FORMAL_HUMAN_REVIEW_EXECUTABILITY = READY")
    print("LANGUAGE_BRIDGE_QUALITY_STATUS = NOT_REVIEWED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
