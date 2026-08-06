"""Build or validate a reusable full-corpus BTC feature manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from system_tai.data.corpus_discovery import (
    CorpusDiscoveryError,
    DiscoveryValidation,
    discover_corpus,
    load_corpus_manifest,
    load_or_build_manifest_cache,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--reuse-manifest", type=Path)
    source.add_argument("--manifest-cache", type=Path)
    parser.add_argument("--rebuild-invalid-manifest-cache", action="store_true")
    parser.add_argument("--portable", action="store_true")
    parser.add_argument(
        "--discovery-validation",
        choices=tuple(mode.value for mode in DiscoveryValidation),
        default=DiscoveryValidation.STRICT.value,
    )
    parser.add_argument("--expected-dimension", type=int, default=512)
    parser.add_argument("--max-root-depth", type=int, default=4)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.rebuild_invalid_manifest_cache and args.manifest_cache is None:
        raise ValueError("--rebuild-invalid-manifest-cache requires --manifest-cache")
    if args.reuse_manifest is not None:
        manifest = load_corpus_manifest(
            args.reuse_manifest,
            input_root=args.input_root,
            max_root_depth=args.max_root_depth,
        )
        status = "REUSED"
    elif args.manifest_cache is not None:
        cache = load_or_build_manifest_cache(
            args.manifest_cache,
            input_root=args.input_root,
            expected_dimension=args.expected_dimension,
            max_root_depth=args.max_root_depth,
            rebuild_invalid=args.rebuild_invalid_manifest_cache,
        )
        manifest = cache.manifest
        status = cache.status
    else:
        manifest = discover_corpus(
            args.input_root,
            expected_dimension=args.expected_dimension,
            max_root_depth=args.max_root_depth,
            validation_mode=DiscoveryValidation(args.discovery_validation),
            portable=args.portable,
        )
        status = "BUILT"
    write_start = time.perf_counter()
    destination = manifest.write(args.output, portable=args.portable)
    write_seconds = time.perf_counter() - write_start
    timings = manifest.discovery_metrics.to_payload()
    timings["manifest_write_seconds"] = write_seconds
    print(
        json.dumps(
            {
                "status": status,
                "manifest": str(destination),
                "fingerprint": manifest.to_payload(portable=args.portable)["manifest_fingerprint"],
                "schema_version": (2 if args.portable else 1),
                "portable": args.portable,
                "video_count": len(manifest.videos),
                "feature_row_count": manifest.total_rows,
                "copied_source_artifacts": False,
                "discovery_timings": timings,
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
