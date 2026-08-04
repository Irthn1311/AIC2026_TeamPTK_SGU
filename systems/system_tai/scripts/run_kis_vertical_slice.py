"""CLI placeholder for the real KIS vertical slice."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a system_tai KIS config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise NotImplementedError(
        "The system_tai KIS vertical slice is a skeleton; no pipeline is implemented"
    )


if __name__ == "__main__":
    raise SystemExit(main())
