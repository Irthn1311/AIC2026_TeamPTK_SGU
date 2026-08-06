"""Run the opt-in Weighted RRF KIS pilot over comparable verified groups."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from system_tai.evaluation.benchmark_schema import VariantType
from system_tai.evaluation.benchmark_validator import BenchmarkValidator
from system_tai.evaluation.fusion_benchmark import (
    FusionBenchmarkEvaluator,
    NoComparableFusionGroupsError,
    select_comparable_fusion_groups,
)
from system_tai.evaluation.fusion_reports import write_fusion_reports
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.features.query_encoder import OpenAIClipTextEncoder, TextEncoderUnavailable
from system_tai.kis.benchmark import resolve_device
from system_tai.retrieval.multi_query import WeightedRRFRetriever
from system_tai.retrieval.vector_search import ExactNumpyRetriever

DEFAULT_OUTPUT_DIRECTORY = Path("/kaggle/working/system_tai_outputs/kis_fusion_pilot")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 20, 50, 100])
    parser.add_argument("--top-k-per-variant", type=int, default=100)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument(
        "--variant-weight",
        action="append",
        default=[],
        metavar="VARIANT_TYPE=WEIGHT",
    )
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="accepted for parity; invalid fusion benchmarks always fail clearly",
    )
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--clip-cache-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser


def _parse_variant_weights(values: list[str]) -> dict[VariantType, float]:
    weights: dict[VariantType, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("variant weights must use VARIANT_TYPE=WEIGHT")
        raw_type, raw_weight = value.split("=", maxsplit=1)
        variant_type = VariantType(raw_type)
        if variant_type in weights:
            raise ValueError(f"duplicate weight for {variant_type.value}")
        try:
            weights[variant_type] = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"invalid weight for {variant_type.value}") from exc
    return weights


def run(args: argparse.Namespace) -> int:
    registry = FeatureStoreRegistry.from_manifest(args.manifest)
    validation = BenchmarkValidator().validate_file(args.benchmark, registry)
    print(
        "benchmark validation: "
        f"valid={validation.valid} errors={len(validation.errors)} "
        f"verified={len(validation.verified_queries)} drafts={len(validation.draft_queries)}"
    )
    for issue in validation.errors:
        print(
            f"validation error: query={issue.query_id} field={issue.field} "
            f"code={issue.code} message={issue.message}",
            file=sys.stderr,
        )
    if not validation.valid or validation.benchmark is None:
        return 2
    selection = select_comparable_fusion_groups(validation.benchmark)
    print(
        f"fusion groups: comparable={len(selection.groups)} "
        f"issues={len(selection.issues)} excluded_drafts={selection.draft_query_count}"
    )
    for issue in selection.issues:
        print(
            f"fusion group issue: group={issue.semantic_group_id} "
            f"code={issue.code} message={issue.message}",
            file=sys.stderr,
        )
    if not selection.groups:
        raise NoComparableFusionGroupsError("no comparable verified groups")
    if args.validation_only:
        print("validation-only: completed; fusion retrieval not requested")
        return 0

    variant_weights = _parse_variant_weights(args.variant_weight)
    device = resolve_device(args.device)
    encoder = OpenAIClipTextEncoder(
        device=device,
        allow_model_download=args.allow_model_download,
        cache_dir=args.clip_cache_dir,
    )
    exact = ExactNumpyRetriever(registry, encoder, chunk_size=args.chunk_size)
    report = FusionBenchmarkEvaluator().evaluate(
        validation,
        WeightedRRFRetriever(exact),
        top_ks=tuple(args.top_k),
        top_k_per_variant=args.top_k_per_variant,
        rrf_constant=args.rrf_constant,
        variant_weights=variant_weights,
    )
    paths = write_fusion_reports(report, args.output_directory)
    print(
        f"fusion evaluation completed: groups={report.evaluated_group_count} "
        f"device={device} canonical_per_variant_unsuppressed=true"
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
        print(f"KIS fusion benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
