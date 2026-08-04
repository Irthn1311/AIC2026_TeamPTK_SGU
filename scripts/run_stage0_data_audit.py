#!/usr/bin/env python3
"""Run TRIAGE-EG Stage 0 BTC Data Audit."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from triage_eg.data.stage0_audit import AuditConfig, run_audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--mode", choices=("sample", "full"), default="sample")
    result.add_argument("--sample-size", type=int, default=10)
    result.add_argument("--video-ids", nargs="*", default=())
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--clip-validation", choices=("shape", "full"), default="full")
    result.add_argument("--object-validation", choices=("filenames", "full"), default="full")
    result.add_argument("--max-object-json-bytes", type=int, default=1_048_576)
    result.add_argument("--ffprobe-timeout", type=int, default=30)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--strict-root", action="store_true")
    result.add_argument("--fail-on-error", action="store_true")
    result.add_argument("--log-level", default="INFO")
    return result


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        result = run_audit(
            AuditConfig(
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                mode=args.mode,
                sample_size=args.sample_size,
                video_ids=tuple(args.video_ids),
                seed=args.seed,
                clip_validation=args.clip_validation,
                object_validation=args.object_validation,
                max_object_json_bytes=args.max_object_json_bytes,
                ffprobe_timeout_seconds=args.ffprobe_timeout,
                resume=args.resume,
                overwrite=args.overwrite,
                strict_root=args.strict_root,
            ),
            project_root=Path(__file__).resolve().parents[1],
        )
    except (FileNotFoundError, FileExistsError, OSError, TypeError, ValueError) as error:
        logging.error("%s", error)
        return 2
    print(result.output_root)
    print(result.summary["gates"])
    if args.fail_on_error and result.summary["issues"]["by_severity"].get("ERROR", 0):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
