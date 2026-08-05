"""Run the exact Phase 2 KIS retrieval baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.common.schemas import KISQuery
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.features.query_encoder import OpenAIClipTextEncoder, TextEncoderUnavailable
from system_tai.ranking.kis_ranker import KISRanker, TemporalSuppressionConfig
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from system_tai.validation.checkpoint_validator import CheckpointValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--expected-dimension", type=int, default=512)
    parser.add_argument("--clip-cache-dir", type=Path)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--internal-checkpoint", action="store_true")
    parser.add_argument("--temporal-suppression", action="store_true")
    parser.add_argument("--minimum-frame-gap", type=int, default=0)
    parser.add_argument("--maximum-candidates-per-video", type=int)
    return parser


def run(args: argparse.Namespace) -> int:
    registry = FeatureStoreRegistry.from_manifest(
        args.manifest,
        expected_dimension=args.expected_dimension,
        memory_map=True,
    )
    encoder = OpenAIClipTextEncoder(
        device=args.device,
        allow_model_download=args.allow_model_download,
        cache_dir=args.clip_cache_dir,
        expected_dimension=args.expected_dimension,
    )
    result = ExactNumpyRetriever(
        registry,
        encoder,
        chunk_size=args.chunk_size,
    ).retrieve(KISQuery(query_id=args.query_id, text=args.query, top_k=args.top_k))
    result, suppression = KISRanker().apply(
        result,
        TemporalSuppressionConfig(
            enabled=args.temporal_suppression,
            minimum_frame_gap=args.minimum_frame_gap,
            maximum_candidates_per_video=args.maximum_candidates_per_video,
        ),
    )
    summary = CheckpointExporter().export(
        result,
        args.output,
        include_internal=args.internal_checkpoint,
    )
    validation = CheckpointValidator().validate(args.output, registry=registry)
    if not validation.valid:
        for issue in validation.errors:
            print(
                f"validation error: line={issue.line_number} code={issue.code} "
                f"message={issue.message}",
                file=sys.stderr,
            )
        return 2
    print(
        "KIS retrieval complete: "
        f"videos={len(registry.stores)} features={registry.total_rows} "
        f"query_id={args.query_id} results={summary.record_count} "
        f"suppressed={suppression.removed_count} output={summary.destination}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError, TextEncoderUnavailable, RuntimeError) as exc:
        print(f"KIS retrieval failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
