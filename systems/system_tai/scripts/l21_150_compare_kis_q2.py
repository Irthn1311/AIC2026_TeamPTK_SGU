"""Compare L21-150 KIS VI-only and translation-augmented evaluation reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = SYSTEM_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from system_tai.quality.l21_150_kis_comparison import (  # noqa: E402
    L21150KISComparisonError,
    compare_l21_150_kis_arms,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a-evaluation", type=Path, required=True)
    parser.add_argument("--arm-b-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compare_l21_150_kis_arms(
            _load(args.arm_a_evaluation),
            _load(args.arm_b_evaluation),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (FileNotFoundError, OSError, ValueError, L21150KISComparisonError) as exc:
        print(f"L21-150 KIS comparison failed: {exc}", file=sys.stderr)
        return 2
    print(
        "L21-150 KIS comparison complete: "
        f"paired_queries={report['paired_query_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
