"""Contest-ready end-to-end Textual KIS CLI MVP for system_tai."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from system_tai.data.corpus_discovery import (
    CorpusDiscoveryError,
    discover_corpus,
    load_corpus_manifest,
)
from system_tai.features.query_encoder import TextEncoderUnavailable
from system_tai.kis.benchmark import resolve_device
from system_tai.kis.contest_runner import ContestRunConfig, ContestRunner
from system_tai.kis.contest_schema import ContestQuery, load_contest_queries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--reuse-manifest", type=Path)
    query_source = parser.add_mutually_exclusive_group(required=True)
    query_source.add_argument("--queries", type=Path)
    query_source.add_argument("--query-id")
    parser.add_argument("--query-vi")
    parser.add_argument("--query-en")
    parser.add_argument("--query-en-expansion")
    parser.add_argument("--weight-vi", type=float, default=1.0)
    parser.add_argument("--weight-en", type=float, default=1.0)
    parser.add_argument("--weight-en-expansion", type=float, default=1.0)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k-per-variant", type=int, default=100)
    parser.add_argument("--output-top-k", type=int, default=100)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--inspection-top-n", type=int, default=50)
    failure = parser.add_mutually_exclusive_group()
    failure.add_argument("--fail-fast", action="store_true")
    failure.add_argument("--continue-on-query-error", action="store_true")
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--clip-cache-dir", type=Path)
    parser.add_argument("--expected-dimension", type=int, default=512)
    parser.add_argument("--max-root-depth", type=int, default=4)
    return parser


def _queries_from_args(args: argparse.Namespace) -> tuple[ContestQuery, ...]:
    if args.queries is not None:
        return load_contest_queries(args.queries).queries
    if not args.query_vi:
        raise ValueError("--query-vi is required for a single query")
    return (
        ContestQuery(
            query_id=args.query_id,
            query_vi=args.query_vi,
            query_en=args.query_en,
            query_en_expansion=args.query_en_expansion,
            weight_vi=args.weight_vi,
            weight_en=args.weight_en,
            weight_en_expansion=args.weight_en_expansion,
            output_top_k=args.output_top_k,
            metadata={"input_mode": "single_cli"},
        ),
    )


def run(args: argparse.Namespace, *, runner: ContestRunner | None = None) -> int:
    start = time.perf_counter()
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    feature_manifest_path = output / "feature_manifest.json"
    if args.reuse_manifest is not None:
        discovery_seconds = 0.0
        manifest_start = time.perf_counter()
        manifest = load_corpus_manifest(args.reuse_manifest)
        manifest.write(feature_manifest_path)
        manifest_seconds = time.perf_counter() - manifest_start
        print(f"manifest reused: {args.reuse_manifest}")
    else:
        discovery_start = time.perf_counter()
        manifest = discover_corpus(
            args.input_root,
            expected_dimension=args.expected_dimension,
            max_root_depth=args.max_root_depth,
        )
        discovery_seconds = time.perf_counter() - discovery_start
        manifest_start = time.perf_counter()
        manifest.write(feature_manifest_path)
        manifest_seconds = time.perf_counter() - manifest_start
        print(f"dataset root resolved: {manifest.dataset_root}")
        print(
            f"manifest built: videos={len(manifest.videos)} rows={manifest.total_rows}"
        )
    queries = _queries_from_args(args)
    device = resolve_device(args.device)
    print(f"device selected: {device}")
    print(f"queries loaded: {len(queries)}")
    config = ContestRunConfig(
        device=device,
        top_k_per_variant=args.top_k_per_variant,
        output_top_k_override=args.output_top_k,
        rrf_constant=args.rrf_constant,
        chunk_size=args.chunk_size,
        inspection_top_n=args.inspection_top_n,
        continue_on_query_error=args.continue_on_query_error,
        create_contact_sheet=args.contact_sheet,
        allow_model_download=args.allow_model_download,
        clip_cache_dir=args.clip_cache_dir,
    )
    outcome = (runner or ContestRunner()).run(
        manifest_path=feature_manifest_path,
        manifest=manifest,
        queries=queries,
        output_directory=output,
        config=config,
        bootstrap_timings={
            "discovery_seconds": discovery_seconds,
            "manifest_load_or_build_seconds": manifest_seconds,
            "pre_runner_total_seconds": time.perf_counter() - start,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS" if outcome.exit_code == 0 else "FAILED",
                "successful_queries": list(outcome.successful_query_ids),
                "failed_queries": [query_id for query_id, _reason in outcome.failed_queries],
                "validator_valid": outcome.validation.valid,
                "output_directory": str(output),
                "generated_files": [
                    str(path.relative_to(output)).replace("\\", "/")
                    for path in outcome.output_files
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return outcome.exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (
        CorpusDiscoveryError,
        FileNotFoundError,
        TextEncoderUnavailable,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"contest KIS failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if isinstance(exc, CorpusDiscoveryError):
            for issue in exc.issues:
                print(f"- {issue}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
