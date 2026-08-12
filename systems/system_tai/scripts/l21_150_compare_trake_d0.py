"""Offline GT join and three-arm comparison for TR-A2-D0 nomination artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = SYSTEM_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from system_tai.quality.l21_150_schema import (  # noqa: E402
    L21150FormatError,
    load_l21_150_benchmark,
)
from system_tai.quality.l21_150_trake_nomination import (  # noqa: E402
    TRAKENominationError,
    compare_nomination_reports,
    evaluate_nomination_artifact,
    load_nomination_artifact,
    write_json_document,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--vi-only", type=Path, required=True)
    parser.add_argument("--vi-plus-en-weighted-rrf", type=Path, required=True)
    parser.add_argument("--en-only", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        benchmark = load_l21_150_benchmark(args.benchmark)
        artifact_paths = {
            "vi_only": args.vi_only,
            "vi_plus_en_weighted_rrf": args.vi_plus_en_weighted_rrf,
            "en_only": args.en_only,
        }
        reports = {
            policy: evaluate_nomination_artifact(
                load_nomination_artifact(path), benchmark
            )
            for policy, path in artifact_paths.items()
        }
        comparison = compare_nomination_reports(reports)
        comparison["offline_reports"] = reports
        write_json_document(args.output, comparison)
    except (L21150FormatError, OSError, TRAKENominationError, ValueError) as exc:
        print(f"TR-A2-D0 comparison failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "TR_A2_D0_COMPARISON_COMPLETE",
                "output": str(args.output),
                "language_decision": comparison["language_decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
