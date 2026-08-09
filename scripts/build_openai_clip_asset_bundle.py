#!/usr/bin/env python3
"""Package existing official OpenAI CLIP source/checkpoint assets without downloads."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from triage_eg.retrieval.stage1b.asset_bundle import (
    AssetBundleConfig,
    build_openai_clip_asset_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--zip", action="store_true", dest="create_zip")
    parser.add_argument(
        "--dependency-wheel",
        action="append",
        default=[],
        type=Path,
        help="Existing pure-Python wheel to bundle for offline runtime; repeat as needed.",
    )
    args = parser.parse_args()
    try:
        result = build_openai_clip_asset_bundle(
            AssetBundleConfig(
                source_root=args.source_root,
                checkpoint=args.checkpoint,
                output_root=args.output_root,
                source_commit=args.source_commit,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                create_zip=args.create_zip,
                dependency_wheels=tuple(args.dependency_wheel),
            )
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        logging.error("%s", error)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
