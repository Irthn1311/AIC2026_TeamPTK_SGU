"""Compare paired L21-150 KIS DEV results for the VI, VI+EN, and EN arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = SYSTEM_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from system_tai.quality.l21_150_kis_abc_comparison import (  # noqa: E402
    L21150KISABCComparisonError,
    compare_l21_150_kis_abc_arms,
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path} must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} must be valid UTF-8") from exc
    value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if type(value) is not dict:
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a-evaluation", type=Path, required=True)
    parser.add_argument("--arm-b-evaluation", type=Path, required=True)
    parser.add_argument("--arm-c-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compare_l21_150_kis_abc_arms(
            _load(args.arm_a_evaluation),
            _load(args.arm_b_evaluation),
            _load(args.arm_c_evaluation),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (FileNotFoundError, OSError, ValueError, L21150KISABCComparisonError) as exc:
        print(f"L21-150 KIS A/B/C comparison failed: {exc}", file=sys.stderr)
        return 2
    print(
        "L21-150 KIS A/B/C comparison complete: "
        f"paired_queries={report['paired_query_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
