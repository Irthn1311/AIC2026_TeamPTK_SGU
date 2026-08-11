"""Validate the L21-150 KIS DEV English experiment-input sidecar."""

from __future__ import annotations

import argparse
import hashlib
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
from system_tai.quality.l21_150_translation import (  # noqa: E402
    KISTranslationSidecarError,
    load_kis_dev_translation_sidecar,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        benchmark = load_l21_150_benchmark(args.benchmark)
        sidecar = load_kis_dev_translation_sidecar(
            args.sidecar,
            benchmark,
            args.benchmark,
        )
    except (FileNotFoundError, L21150FormatError, KISTranslationSidecarError, OSError) as exc:
        print(f"KIS translation sidecar validation failed: {exc}", file=sys.stderr)
        return 2
    digest = hashlib.sha256(args.sidecar.read_bytes()).hexdigest()
    print(
        "KIS translation sidecar valid: "
        f"queries={sidecar.query_count} status={sidecar.translation_status} sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
