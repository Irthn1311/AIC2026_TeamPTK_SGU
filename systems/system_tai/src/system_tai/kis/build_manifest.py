"""Build or validate a reusable full-corpus BTC feature manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from system_tai.data.corpus_discovery import (
    CorpusDiscoveryError,
    discover_corpus,
    load_corpus_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path)
    parser.add_argument("--expected-dimension", type=int, default=512)
    parser.add_argument("--max-root-depth", type=int, default=4)
    return parser


def run(args: argparse.Namespace) -> int:
    reused = args.reuse_manifest is not None
    if reused:
        manifest = load_corpus_manifest(args.reuse_manifest)
    else:
        manifest = discover_corpus(
            args.input_root,
            expected_dimension=args.expected_dimension,
            max_root_depth=args.max_root_depth,
        )
    destination = manifest.write(args.output)
    print(
        json.dumps(
            {
                "status": "REUSED" if reused else "BUILT",
                "manifest": str(destination),
                "fingerprint": manifest.fingerprint,
                "video_count": len(manifest.videos),
                "feature_row_count": manifest.total_rows,
                "copied_source_artifacts": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (CorpusDiscoveryError, FileNotFoundError, ValueError) as exc:
        print(f"feature manifest failed: {exc}", file=sys.stderr)
        if isinstance(exc, CorpusDiscoveryError):
            for issue in exc.issues:
                print(f"- {issue}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
