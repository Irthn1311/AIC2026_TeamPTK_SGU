"""Run the QA-D0 DEV-only localization-language nomination diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from system_tai.data.corpus_discovery import load_corpus_manifest
from system_tai.features.btc_clip_store import (
    FeatureStoreRegistry,
    VideoFeatureStoreLoader,
)
from system_tai.features.query_encoder import SharedOpenAIClipEncoder
from system_tai.kis.benchmark import resolve_device
from system_tai.quality.l21_150_qa_nomination import (
    QALanguagePolicy,
    QANominationError,
    assert_runtime_input_gt_isolated,
    build_dev_target_video_ids,
    build_nomination_inputs,
    ensure_dev_only_scope,
    evaluate_nomination_results,
    run_nomination_runtime,
    write_json_document,
)
from system_tai.quality.l21_150_qa_translation import (
    QATranslationSidecarError,
    load_qa_dev_translation_sidecar,
)
from system_tai.quality.l21_150_schema import load_l21_150_benchmark
from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=tuple(policy.value for policy in QALanguagePolicy),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--clip-cache-dir", type=Path)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if len(value) == 40 else None


def _load_registry(manifest) -> FeatureStoreRegistry:
    loader = VideoFeatureStoreLoader(expected_dimension=512, memory_map=True)
    return FeatureStoreRegistry(
        [
            loader.load(
                video_id=video.video_id,
                mapping_csv_path=video.mapping_csv_path,
                clip_npy_path=video.clip_npy_path,
            )
            for video in manifest.videos
        ]
    )


def run(args: argparse.Namespace) -> int:
    ensure_dev_only_scope(args.split)
    if type(args.chunk_size) is not int or args.chunk_size <= 0:
        raise QANominationError("chunk-size must be a positive integer")
    expected_manifest_sha = str(args.manifest_sha256).casefold()
    if len(expected_manifest_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_manifest_sha
    ):
        raise QANominationError("manifest-sha256 must be a lowercase SHA256 digest")
    actual_manifest_sha = _sha256(args.reuse_manifest)
    if actual_manifest_sha != expected_manifest_sha:
        raise QANominationError(
            "feature manifest SHA256 mismatch: "
            f"expected={expected_manifest_sha}, actual={actual_manifest_sha}"
        )

    benchmark = load_l21_150_benchmark(args.benchmark)
    sidecar = load_qa_dev_translation_sidecar(
        args.translations,
        benchmark,
        args.benchmark,
    )
    sidecar_sha = _sha256(args.translations)
    policy = QALanguagePolicy(args.policy)
    inputs = build_nomination_inputs(
        benchmark,
        language_policy=policy,
        sidecar=sidecar,
    )
    assert_runtime_input_gt_isolated(inputs)

    manifest = load_corpus_manifest(
        args.reuse_manifest,
        input_root=args.input_root,
    )
    registry = _load_registry(manifest)
    device = resolve_device(args.device)
    encoder = SharedOpenAIClipEncoder(
        device=device,
        allow_model_download=args.allow_model_download,
        cache_dir=args.clip_cache_dir,
    )
    searcher = VideoRestrictedFeatureSearcher(registry, chunk_size=args.chunk_size)
    print(
        "QA-D0 runtime started: "
        f"policy={policy.value} queries={len(inputs)} videos={len(registry.stores)} "
        f"device={device}"
    )
    runtime_results = run_nomination_runtime(
        inputs,
        language_policy=policy,
        encoder=encoder,
        searcher=searcher,
    )

    # This is the only target-video join and occurs after every retrieval has finished.
    target_video_ids = build_dev_target_video_ids(benchmark)
    report = evaluate_nomination_results(
        runtime_results,
        target_video_ids=target_video_ids,
        benchmark_id=benchmark.benchmark_id,
        policy=policy,
        translation_sidecar_sha256=sidecar_sha,
        manifest_sha256=actual_manifest_sha,
        corpus_fingerprint=manifest.fingerprint,
        git_sha=_git_sha(),
        model_identity={
            "library": encoder.identifiers.get("library"),
            "model": encoder.identifiers.get("model"),
            "device": encoder.identifiers.get("device", device),
        },
    )
    write_json_document(args.output, report)
    print(
        "QA-D0 complete: "
        f"policy={policy.value} R@32={report['target_video_recall_at_32']:.6f} "
        f"MRR={report['mean_reciprocal_rank']:.6f} "
        f"misses={report['target_video_miss_count_at_full_depth']} "
        f"output={args.output}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (
        FileNotFoundError,
        OSError,
        QANominationError,
        QATranslationSidecarError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"QA-D0 failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
