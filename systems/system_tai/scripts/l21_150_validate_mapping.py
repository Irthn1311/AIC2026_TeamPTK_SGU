"""Validate proposed L21-150 frame intervals against BTC mapping evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = SYSTEM_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from system_tai.quality.l21_150_mapping import validate_l21_150_mapping  # noqa: E402
from system_tai.quality.l21_150_schema import (  # noqa: E402
    L21150FormatError,
    load_l21_150_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        benchmark = load_l21_150_benchmark(args.benchmark)
        report = validate_l21_150_mapping(
            benchmark,
            args.mapping_root,
            video_root=args.video_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (FileNotFoundError, L21150FormatError, OSError, ValueError) as exc:
        print(f"L21-150 mapping validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        "L21-150 mapping validation complete: "
        f"records={report['record_count']} statuses={report['status_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
