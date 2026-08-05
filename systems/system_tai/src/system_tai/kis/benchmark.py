"""Validate or evaluate the Phase 2.5 ground-truth KIS benchmark."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from system_tai.evaluation.benchmark_validator import BenchmarkValidator
from system_tai.evaluation.kis_benchmark import (
    KISBenchmarkEvaluator,
    NoVerifiedQueriesResult,
)
from system_tai.evaluation.reports import write_benchmark_reports
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.features.query_encoder import OpenAIClipTextEncoder, TextEncoderUnavailable
from system_tai.retrieval.vector_search import ExactNumpyRetriever

DEFAULT_OUTPUT_DIRECTORY = Path("/kaggle/working/system_tai_outputs/kis_benchmark")
DEFAULT_TOP_KS = (1, 5, 20, 50, 100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_KS))
    parser.add_argument("--include-draft-validation", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--clip-cache-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser


def resolve_device(requested: str, *, torch_module: Any | None = None) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported device: {requested}")
    if requested == "cpu":
        return "cpu"
    try:
        torch = torch_module or importlib.import_module("torch")
    except ImportError as exc:
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but Torch is unavailable") from exc
        return "cpu"
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable")
    return "cuda" if available else "cpu"


def run(args: argparse.Namespace) -> int:
    registry = FeatureStoreRegistry.from_manifest(args.manifest)
    validation = BenchmarkValidator().validate_file(
        args.benchmark,
        registry,
        include_drafts=args.include_draft_validation,
    )
    print(
        "benchmark validation: "
        f"valid={validation.valid} errors={len(validation.errors)} "
        f"invalid_queries={validation.invalid_query_count} "
        f"verified={len(validation.verified_queries)} drafts={len(validation.draft_queries)}"
    )
    for issue in validation.errors:
        print(
            f"validation error: query={issue.query_id} field={issue.field} "
            f"code={issue.code} message={issue.message}",
            file=sys.stderr,
        )
    if not validation.valid:
        print("validation-only: invalid benchmark; evaluation not run")
        return 2 if args.fail_on_invalid else 0
    if args.validation_only:
        print("validation-only: completed; retrieval evaluation not requested")
        return 0
    if not validation.verified_queries:
        outcome = KISBenchmarkEvaluator().evaluate(
            validation,
            None,
            top_ks=tuple(args.top_k),
        )
        assert isinstance(outcome, NoVerifiedQueriesResult)
        print(
            f"evaluation state={outcome.evaluation_state}; "
            f"evaluated={outcome.evaluated_query_count} "
            f"excluded_drafts={outcome.excluded_draft_query_count} "
            f"invalid_queries={outcome.invalid_query_count}"
        )
        return 0

    device = resolve_device(args.device)
    encoder = OpenAIClipTextEncoder(
        device=device,
        allow_model_download=args.allow_model_download,
        cache_dir=args.clip_cache_dir,
    )
    retriever = ExactNumpyRetriever(registry, encoder, chunk_size=args.chunk_size)
    report = KISBenchmarkEvaluator().evaluate(
        validation,
        retriever,
        top_ks=tuple(args.top_k),
    )
    if isinstance(report, NoVerifiedQueriesResult):
        raise RuntimeError("verified query set changed during evaluation")
    paths = write_benchmark_reports(report, args.output_directory)
    print(
        "evaluation completed: "
        f"queries={report.evaluated_query_count} device={device} "
        f"canonical_unsuppressed={report.canonical_unsuppressed}"
    )
    print(
        f"reports: json={paths.json_path} csv={paths.csv_path} "
        f"markdown={paths.markdown_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        TextEncoderUnavailable,
    ) as exc:
        print(f"KIS benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
